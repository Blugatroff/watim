use crate::{ast::Program, interpreter, scanner::Location, step_interpreter::StepInterpreter};
use crossterm::{
    cursor::{MoveTo, MoveToColumn, MoveToNextLine},
    event::{Event, KeyCode, KeyModifiers},
    style::{Attribute, Color, Colors, Print, SetAttribute, SetColors},
    terminal::{Clear, ClearType},
    ExecutableCommand,
};
use itertools::Itertools;
use std::{
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(thiserror::Error, Debug)]
pub enum DebuggerError {
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error(transparent)]
    InterpreterError(#[from] interpreter::Error),
}

pub fn debug(program: Program) -> Result<(), DebuggerError> {
    crossterm::terminal::enable_raw_mode().unwrap();
    let mut debugger = StepInterpreter::new(program)?;
    let mut file_lookup = FileLookup::default();
    'outer: loop {
        let mut stdout = std::io::stdout();
        stdout
            .execute(Clear(ClearType::All))?
            .execute(MoveTo(0, 0))?;
        if let Some(word) = debugger.current_word() {
            let location = word.location();
            let line = file_lookup.get_line(location);
            let location = word.location();
            let (lines_before, lines_after) = file_lookup.get_surrounding(location, 5);
            let lines_before = lines_before;
            let lines_after = lines_after;
            let (before, after) = line.split_at(location.column - 1);
            let (word, after) = after.split_at(location.len);

            for line in lines_before {
                stdout.execute(Print(line))?.execute(MoveToNextLine(1))?;
            }
            stdout
                .execute(Print(before))?
                .execute(SetColors(Colors::new(Color::Yellow, Color::Reset)))?
                .execute(Print(word))?
                .execute(SetColors(Colors::new(Color::White, Color::Reset)))?
                .execute(Print(after))?
                .execute(MoveToNextLine(1))?;
            for line in lines_after {
                stdout.execute(Print(line))?.execute(MoveToNextLine(1))?;
            }
        }
        let line: String = (0..80).map(|_| '#').collect();
        stdout
            .execute(Print(&line))?
            .execute(MoveToNextLine(1))?
            .execute(SetAttribute(Attribute::Bold))?
            .execute(Print("Stackptr:"))?
            .execute(SetAttribute(Attribute::NoBold))?
            .execute(MoveToNextLine(1))?;
        stdout
            .execute(MoveToColumn(4))?
            .execute(Print(&debugger.stack_ptr()))?
            .execute(MoveToNextLine(1))?;

        stdout
            .execute(Print(&line))?
            .execute(MoveToNextLine(1))?
            .execute(SetAttribute(Attribute::Bold))?
            .execute(Print("Callstack:"))?
            .execute(SetAttribute(Attribute::NoBold))?
            .execute(MoveToNextLine(1))?;
        for f in debugger.call_stack() {
            stdout
                .execute(MoveToColumn(4))?
                .execute(Print(&f.signature.ident))?
                .execute(MoveToNextLine(1))?;
        }
        stdout
            .execute(Print(&line))?
            .execute(MoveToNextLine(1))?
            .execute(SetAttribute(Attribute::Bold))?
            .execute(Print("Locals:"))?
            .execute(SetAttribute(Attribute::NoBold))?
            .execute(MoveToNextLine(1))?;
        for (ident, local) in debugger.locals() {
            stdout
                .execute(MoveToColumn(4))?
                .execute(Print(ident))?
                .execute(Print(format!(" {:?} {:?}", local.clone().ty(), local)))?
                .execute(MoveToNextLine(1))?;
        }
        stdout
            .execute(SetAttribute(Attribute::Bold))?
            .execute(Print("Stack:"))?
            .execute(SetAttribute(Attribute::NoBold))?
            .execute(MoveToNextLine(1))?;
        for value in debugger.stack() {
            stdout
                .execute(MoveToColumn(4))?
                .execute(Print(format_args!("{} {:?}", value.ty(), value)))?
                .execute(MoveToNextLine(1))?;
        }
        stdout.execute(Print(&line))?.execute(MoveToNextLine(1))?;
        let out = String::from_utf8_lossy(debugger.stdout()).to_string();
        for line in out.lines() {
            stdout.execute(Print(&line))?.execute(MoveToNextLine(1))?;
        }
        stdout.execute(MoveTo(80, 0))?;
        let height = crossterm::terminal::size()?.1;
        for h in 0..height - 1 {
            stdout.execute(MoveTo(80, h))?.execute(Print("|"))?;
        }
        stdout
            .execute(MoveTo(81, 0))?
            .execute(SetAttribute(Attribute::Bold))?
            .execute(Print("Memory:"))?
            .execute(SetAttribute(Attribute::NoBold))?;

        fn nibble_to_hex(nibble: u8) -> char {
            (if nibble < 10 {
                48 + nibble
            } else {
                97 + (nibble - 10)
            }) as char
        }
        for y in 0..height - 2 {
            let y = y as usize;
            let start = y * 16;
            let end = ((y + 1) * 16).min(debugger.memory().len());
            let data = &debugger.memory()[start..end];
            let text: String = data
                .iter()
                .map(|b| {
                    if b.is_ascii() && !b.is_ascii_control() {
                        *b as char
                    } else {
                        '.'
                    }
                })
                .collect();
            let data: String = Itertools::intersperse(
                data.iter()
                    .flat_map(|b| [*b >> 4, *b & 0b00001111].map(nibble_to_hex))
                    .collect::<Vec<char>>()
                    .chunks(4),
                &[' '],
            )
            .flatten()
            .collect();
            stdout
                .execute(MoveTo(81, y as u16 + 1))?
                .execute(Print(format_args!("{start:08x}: {data}  {text}")))?;
            if end == debugger.memory().len() {
                break;
            }
        }
        stdout
            .execute(MoveTo(0, height - 1))?
            .execute(Print("step: n, quit: q, up: u"))?;
        if debugger.current_loop_location().is_some() {
            stdout.execute(Print(", loop once: l"))?;
        }
        loop {
            if let Event::Key(event) = crossterm::event::read()? {
                match event.code {
                    KeyCode::Char('n') => {
                        debugger.step()?;
                        break;
                    }
                    KeyCode::Char('q') => {
                        crossterm::terminal::disable_raw_mode().unwrap();
                        break 'outer;
                    }
                    KeyCode::Char('u') => {
                        let level = debugger.depth();
                        while !debugger.done() && debugger.depth() >= level {
                            debugger.step()?;
                        }
                        continue 'outer;
                    }
                    KeyCode::Char('l') => {
                        if debugger.done() {
                            continue;
                        }
                        if let Some(location) = debugger.current_loop_location().cloned() {
                            let depth = debugger.depth();
                            let word = debugger.current_word().map(|w| w.location()).cloned();
                            loop {
                                debugger.step()?;
                                if let Some(loc) = debugger.current_loop_location() {
                                    if &location == loc {
                                        if debugger.current_word().map(|w| w.location())
                                            == word.as_ref()
                                        {
                                            break;
                                        }
                                    } else if depth < debugger.depth() {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }
                            continue 'outer;
                        }
                    }
                    KeyCode::Char('d') => {
                        std::fs::File::create("./mem.raw")
                            .unwrap()
                            .write_all(debugger.memory())?;
                    }
                    KeyCode::Char('c') => {
                        if event.modifiers.contains(KeyModifiers::CONTROL) {
                            crossterm::terminal::disable_raw_mode().unwrap();
                            return Ok(());
                        }
                    }
                    _ => {}
                }
            }
        }
        if debugger.done() {
            break;
        }
    }
    crossterm::terminal::disable_raw_mode().unwrap();
    Ok(())
}

#[derive(Default)]
struct FileLookup {
    files: HashMap<PathBuf, Arc<str>>,
}

impl FileLookup {
    fn file(&mut self, path: &Path) -> Arc<str> {
        if let Some(file) = self.files.get(path) {
            Arc::clone(file)
        } else {
            let text: Arc<str> = std::fs::read_to_string(path)
                .unwrap()
                .into_boxed_str()
                .into();
            self.files.insert(path.to_path_buf(), text);
            self.file(path)
        }
    }
    fn get_surrounding(
        &mut self,
        location: &Location,
        radius: usize,
    ) -> (Vec<String>, Vec<String>) {
        let start = location.line.saturating_sub(radius);
        let before = self
            .file(&location.path)
            .lines()
            .skip(start)
            .take(radius - 1)
            .map(String::from)
            .collect();
        let after = self
            .file(&location.path)
            .lines()
            .skip(location.line)
            .take(radius + 1)
            .map(String::from)
            .collect();
        (before, after)
    }
    fn get_line(&mut self, location: &Location) -> String {
        match self.file(&location.path).lines().nth(location.line - 1) {
            Some(line) => line.to_string(),
            None => String::from(""),
        }
    }
}
