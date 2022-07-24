#![feature(iter_intersperse)]
#![feature(if_let_guard)]

mod ast;
mod checker;
mod debugger;
mod generator;
mod interpreter;
mod intrinsics;
mod parser;
mod prepass;
mod scanner;
mod step_interpreter;

use crate::interpreter::Interpreter;
use checker::TypeError;
use parser::ParseError;
use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
pub enum WatimError {
    #[error(transparent)]
    Scanning(#[from] scanner::ScanError),
    #[error(transparent)]
    Parsing(#[from] ParseError),
    #[error(transparent)]
    Type(#[from] TypeError),
    #[error(transparent)]
    Interpet(#[from] interpreter::Error),
    #[error(transparent)]
    Debug(#[from] debugger::DebuggerError),
    #[error(transparent)]
    IO(#[from] std::io::Error),
}

pub const SEP: &str = "================================\n";

fn run(program: &str, input: impl std::io::Read) -> Vec<u8> {
    match prepass::UncheckedProgram::load_with_custom_file_loader(
        &mut |_| Ok(program.to_string()),
        "test.watim",
    )
    .and_then(|p| p.resolve().map_err(WatimError::from))
    .and_then(|p| p.check().map_err(WatimError::from))
    {
        Ok(program) => {
            match {
                let mut output = Vec::new();
                let res = Interpreter::interpret_program(program, input, &mut output);
                res.map(|_| output)
            } {
                Ok(output) => output,
                Err(e) => format!("{e}").into_bytes(),
            }
        }
        Err(e) => format!("{e}").into_bytes(),
    }
}

#[test]
fn test() {
    let dir = std::fs::read_dir("./tests").unwrap();
    for entry in dir {
        let entry = entry.unwrap();
        let test = std::fs::read_to_string(entry.path()).unwrap();
        let (input, program) = test.split_once(SEP).unwrap();
        let (program, output) = program.split_once(SEP).unwrap();
        let out = run(program, input.as_bytes());
        let out = String::from_utf8_lossy(&out);
        let output = std::borrow::Cow::Borrowed(output);
        assert_eq!(output, out);
    }
}

fn regen_tests() {
    let dir = std::fs::read_dir("./tests").unwrap();
    for entry in dir {
        let entry = entry.unwrap();
        let input = std::fs::read_to_string(entry.path()).unwrap();
        let (input, program) = input.split_once(SEP).unwrap();
        let (program, _) = program.split_once(SEP).unwrap();
        let mut output = run(program, input.as_bytes());
        let mut out = format!("{input}{SEP}{program}{SEP}").into_bytes();
        out.append(&mut output);
        std::fs::write(entry.path(), out).unwrap();
    }
}

fn main() {
    fn inner() -> Result<(), WatimError> {
        let mode = std::env::args().nth(1).unwrap();
        let mode = match mode.as_str() {
            mode @ "com" => mode,
            mode @ "sim" => mode,
            mode @ "debug" => mode,
            "test-regen" => {
                regen_tests();
                return Ok(());
            }
            mode => {
                eprintln!("unknown mode: `{mode}`");
                std::process::exit(1)
            }
        };
        let file = PathBuf::from(std::env::args().nth(2).unwrap())
            .canonicalize()
            .unwrap();
        let program = prepass::UncheckedProgram::load(&file)?.resolve()?.check()?;
        match mode {
            "com" => {
                println!("{program}");
            }
            "sim" => {
                Interpreter::interpret_program(program, std::io::stdin(), std::io::stdout())?;
            }
            "debug" => {
                crate::debugger::debug(program)?;
            }
            _ => {
                println!("unknown mode `{mode}`");
                std::process::exit(1);
            }
        }
        Ok(())
    }
    match inner() {
        Ok(()) => {}
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1)
        }
    }
}
