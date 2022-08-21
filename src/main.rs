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

pub fn align_to(value: i32, alignment: i32) -> i32 {
    (value % alignment > 0) as i32 * (alignment - (value % alignment)) + value
}

pub const SEP: &str = "================================\n";

fn run(program: &str, mut input: impl std::io::Read) -> Vec<u8> {
    let file_path = "./test.watim";
    let mut file = std::fs::File::create(file_path).unwrap();

    let mut buf = Vec::new();
    input.read_to_end(&mut buf).unwrap();
    file.write_all(program.as_bytes()).unwrap();
    let mut command = std::process::Command::new("./run.sh");
    let command = command
        .arg(&file_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = command.spawn().unwrap();
    use std::io::Write;
    child.stdin.as_mut().unwrap().write_all(&buf).unwrap();
    let mut out = child.wait_with_output().unwrap();
    out.stdout.append(&mut out.stderr);
    out.stdout
}

#[test]
fn test() {
    env_logger::init();
    let dir = std::fs::read_dir("./tests").unwrap();
    for entry in dir {
        let entry = entry.unwrap();
        log::info!("Testing {}", entry.path().display());
        let test = std::fs::read_to_string(entry.path()).unwrap();
        let (input, program) = test.split_once(SEP).unwrap();
        let (program, output) = program.split_once(SEP).unwrap();
        let output = std::borrow::Cow::Borrowed(output);
        let out = run(program, input.as_bytes());
        let out = String::from_utf8_lossy(&out);
        let out = std::borrow::Cow::Borrowed(&out);
        if output != out {
            log::error!("Test {} FAILED", entry.path().display());
        }
        assert_eq!(output, out);
    }
}

fn regen_tests() {
    let dir = std::fs::read_dir("./tests").unwrap();
    for entry in dir {
        let entry = entry.unwrap();
        log::info!("Generating Test {}", entry.path().display());
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
    env_logger::init();
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
