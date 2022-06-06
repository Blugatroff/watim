#![feature(iter_intersperse)]

use scanner::Scanner;

mod ast;
mod generator;
mod parser;
mod scanner;

fn main() {
    let file = std::env::args().nth(1).unwrap();
    let input = std::fs::read_to_string(&file).unwrap();
    let tokens = match Scanner::new(input, file).scan_tokens() {
        Ok(tokens) => tokens.to_vec(),
        Err(e) => {
            for e in e {
                eprintln!("{}", e)
            }
            std::process::exit(1);
        }
    };
    let program = match parser::Parser::new(tokens).parse() {
        Ok(program) => program,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    println!("{program}")
}
