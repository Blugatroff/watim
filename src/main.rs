#![feature(iter_intersperse)]
#![feature(if_let_guard)]

use crate::interpreter::Interpreter;
use ast::{Module, Program};
use checker::ModuleChecker;
use scanner::Scanner;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

mod ast;
mod checker;
mod generator;
mod interpreter;
mod parser;
mod scanner;

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
    let sim = match mode.as_str() {
        "com" => false,
        "sim" => true,
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
    if sim {
        Interpreter::new(program).unwrap();
    } else {
        println!("{program}");
    }
}
