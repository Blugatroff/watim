#![feature(iter_intersperse)]
#![feature(if_let_guard)]

mod ast;
mod checker;
mod debugger;
mod generator;
mod interpreter;
mod intrinsics;
mod parser;
mod scanner;
mod step_interpreter;

use crate::interpreter::Interpreter;
use ast::{Module, Program};
use checker::{ModuleChecker, TypeError};
use parser::ParseError;
use scanner::Scanner;
use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
};

fn inner_load(
    modules: &mut BTreeMap<PathBuf, Module>,
    path: impl AsRef<Path>,
    input: String,
) -> Result<(), WatimError> {
    let path = path.as_ref().to_path_buf();
    if modules.get(&path).is_some() {
        return Ok(());
    }
    let tokens = Scanner::scan_tokens(input, path.clone())?;
    let module = parser::Parser::new(tokens).parse(&path)?;
    let imports = module.imports.clone();
    modules.insert(path.clone(), module);
    for import in imports {
        let path = path.parent().unwrap().join(import.path);
        let input = std::fs::read_to_string(&path).unwrap();
        inner_load(modules, path, input)?;
    }
    Ok(())
}

fn load(path: impl AsRef<Path>) -> Result<BTreeMap<PathBuf, Module>, WatimError> {
    let mut modules = BTreeMap::new();
    let path = path.as_ref().canonicalize().unwrap();
    let input = std::fs::read_to_string(&path).unwrap();
    inner_load(&mut modules, path, input)?;
    Ok(modules)
}

fn load_raw(
    path: impl AsRef<Path>,
    input: String,
) -> Result<BTreeMap<PathBuf, Module>, WatimError> {
    let mut modules = BTreeMap::new();
    inner_load(&mut modules, path, input)?;
    Ok(modules)
}

fn check(modules: BTreeMap<PathBuf, Module>) -> Result<Program, TypeError> {
    let modules: HashMap<_, _> = modules
        .into_iter()
        .enumerate()
        .map(|(i, (k, v))| (k, (v, format!("{i}"))))
        .collect();
    let mut data = Vec::new();
    let mut checked_modules = HashMap::new();
    for (path, (module, _)) in &modules {
        let module = ModuleChecker::check(module.clone(), &modules, &mut data)?;
        checked_modules.insert(path.clone(), module);
    }
    Ok(Program {
        data,
        modules: checked_modules,
    })
}

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
}

pub const SEP: &'static str = "================================\n";

fn run(program: &str, input: impl std::io::Read) -> Vec<u8> {
    match load_raw("test.watim", program.to_string())
        .and_then(|m| check(m).map_err(WatimError::from))
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
        let modules = load(&file)?;
        let program = check(modules)?;
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
