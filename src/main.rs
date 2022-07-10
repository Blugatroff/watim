#![feature(iter_intersperse)]
#![feature(if_let_guard)]

mod ast;
mod checker;
mod generator;
mod interpreter;
mod parser;
mod scanner;

use crate::interpreter::{Interpreter, StepInterpreter};
use ast::{Module, Program};
use checker::ModuleChecker;
use crossterm::{
    cursor::{MoveTo, MoveToColumn, MoveToNextLine},
    event::{Event, KeyCode, KeyModifiers},
    style::{Attribute, Color, Colors, Print, SetAttribute, SetColors},
    terminal::{Clear, ClearType},
    ExecutableCommand,
};
use scanner::{Location, Scanner};
use std::{
    collections::HashMap,
    fmt::Debug,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

fn load_module(path: impl AsRef<Path>) -> Module {
    let path = path.as_ref().to_path_buf();
    let input = std::fs::read_to_string(&path).unwrap();
    let tokens = match Scanner::new(input, path.clone()).scan_tokens() {
        Ok(tokens) => tokens.to_vec(),
        Err(e) => {
            for e in e {
                eprintln!("{}", e)
            }
            std::process::exit(1);
        }
    };
    match parser::Parser::new(tokens).parse(path) {
        Ok(program) => program,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

fn main() {
    let file = PathBuf::from(std::env::args().nth(2).unwrap())
        .canonicalize()
        .unwrap();
    let mode = std::env::args().nth(1).unwrap();
    let mode = match mode.as_str() {
        mode @ "com" => mode,
        mode @ "sim" => mode,
        mode @ "debug" => mode,
        mode => {
            eprintln!("unknown mode: `{mode}`");
            std::process::exit(1)
        }
    };
    let module = load_module(&file);
    let mut modules = HashMap::from([(module.path.clone(), module.clone())]);
    for import in &module.imports {
        let import = match ModuleChecker::check_import(import.clone(), &module.path) {
            Ok(import) => import,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        };
        match modules.get(&import.path) {
            Some(_) => {}
            None => {
                modules.insert(import.path.clone(), load_module(&import.path));
            }
        }
    }
    let mut data = Vec::new();
    let modules: HashMap<PathBuf, (Module, String)> = modules
        .into_iter()
        .enumerate()
        .map(|(i, (k, v))| (k, (v, format!("{i}"))))
        .collect();
    let mut checked_modules = HashMap::new();
    for (path, (module, _)) in &modules {
        let module = match ModuleChecker::check(module.clone(), &modules, &mut data) {
            Ok(module) => module,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        };
        checked_modules.insert(path.clone(), module);
    }
    let program = Program {
        data,
        modules: checked_modules,
    };
    match mode {
        "com" => {
            println!("{program}");
        }
        "sim" => {
            Interpreter::new(program).unwrap();
        }
        "debug" => {
            crossterm::terminal::enable_raw_mode().unwrap();
            debug(program).unwrap();
            crossterm::terminal::disable_raw_mode().unwrap();
        }
        _ => todo!(),
    }
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

#[derive(Debug)]
enum DebuggerError {
    IoError(std::io::Error),
    InterpreterError(interpreter::Error),
}

impl From<std::io::Error> for DebuggerError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl From<interpreter::Error> for DebuggerError {
    fn from(e: interpreter::Error) -> Self {
        Self::InterpreterError(e)
    }
}

fn debug(program: Program) -> Result<(), DebuggerError> {
    let mut debugger = StepInterpreter::new(program)?;
    let mut file_lookup = FileLookup::default();
    loop {
        if debugger.done() {
            break;
        }
        let mut stdout = std::io::stdout();
        stdout
            .execute(Clear(ClearType::All))?
            .execute(MoveTo(0, 0))?;
        /*         println!();
        println!("CALLSTACK:");
        for function in debugger.call_stack() {
            println!("\tFUNCTION: {}", function.signature.ident);
        }
        println!("LOCALS:");
        for (ident, value) in debugger.locals() {
            println!("\t{ident}: {:?}", value);
        } */
        match debugger.current_word() {
            Some(word) => {
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
            None => {}
        }
        let width = crossterm::terminal::size()?.0;
        let line: String = (0..width).map(|_| '#').collect();
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
                .execute(Print(format!(" {:?} {:?}", local.ty(), local)))?
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
                .execute(Print(value))?
                .execute(MoveToNextLine(1))?;
        }
        stdout.execute(Print(&line))?.execute(MoveToNextLine(1))?;
        let out = String::from_utf8_lossy(debugger.stdout()).to_string();
        for line in out.lines() {
            stdout.execute(Print(&line))?.execute(MoveToNextLine(1))?;
        }
        loop {
            if let Event::Key(event) = crossterm::event::read()? {
                match event.code {
                    KeyCode::Char('n') => break,
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char('d') => {
                        std::fs::File::create("./mem.raw")
                            .unwrap()
                            .write_all(debugger.memory())?;
                    }
                    KeyCode::Char('c') => {
                        if event.modifiers.contains(KeyModifiers::CONTROL) {
                            return Ok(());
                        }
                    }
                    _ => {}
                }
            }
        }
        debugger.step()?;
    }
    Ok(())
}
