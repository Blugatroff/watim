use crate::scanner::Location;
use std::{collections::HashMap, path::PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    I32,
    Bool,
    Ptr(Box<Type>),
    AnyPtr,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::I32 => f.write_str("i32"),
            Type::Bool => f.write_str("bool"),
            Type::Ptr(_) => f.write_str(".i32"),
            Type::AnyPtr => todo!(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Intrinsic {
    Add,
    Store32,
    Store8,
    Load32,
    Load8,
    Drop,
    Sub,
    Eq,
    NotEq,
    Mod,
    Div,
    And,
    Or,
    L,
    G,
    LE,
    GE,
    Mul,
    Cast(Type),
}

#[derive(Debug, Clone)]
pub struct Iff {
    pub location: Location,
    pub body: Vec<Word>,
    pub el: Option<Vec<Word>>,
}

#[derive(Debug, Clone)]
pub struct CheckedIff {
    pub location: Location,
    pub body: Vec<CheckedWord>,
    pub el: Option<Vec<CheckedWord>>,
    pub ret: Vec<Type>,
}

#[derive(Debug, Clone)]
pub struct Loop {
    pub location: Location,
    pub body: Vec<Word>,
}

#[derive(Debug, Clone)]
pub struct CheckedLoop {
    pub location: Location,
    pub body: Vec<CheckedWord>,
    pub ret: Vec<Type>,
}

#[derive(Debug, Clone)]
pub enum Ident {
    Direct(String),
    Qualified(String, String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedIdent {
    pub module_prefix: String,
    pub ident: String,
}

#[derive(Debug, Clone)]
pub enum Word {
    Call {
        location: Location,
        ident: Ident,
    },
    Var {
        location: Location,
        ident: String,
    },
    Set {
        location: Location,
        ident: String,
    },
    Number {
        location: Location,
        number: i32,
    },
    Intrinsic {
        location: Location,
        intrinsic: Intrinsic,
    },
    If(Iff),
    Loop(Loop),
    Break {
        location: Location,
    },
    String {
        location: Location,
        value: String,
    },
}

#[derive(Debug, Clone)]
pub enum CheckedWord {
    Call {
        location: Location,
        ident: CheckedIdent,
    },
    Var {
        location: Location,
        ident: String,
    },
    Set {
        location: Location,
        ident: String,
    },
    Number {
        location: Location,
        number: i32,
    },
    Intrinsic {
        location: Location,
        intrinsic: Intrinsic,
    },
    If(CheckedIff),
    Loop(CheckedLoop),
    Break {
        location: Location,
    },
    String {
        location: Location,
        addr: i32,
        size: i32,
    },
}

impl CheckedWord {
    pub fn location(&self) -> &Location {
        match self {
            CheckedWord::Call { location, .. } => location,
            CheckedWord::Var { location, .. } => location,
            CheckedWord::Set { location, .. } => location,
            CheckedWord::Number { location, .. } => location,
            CheckedWord::Intrinsic { location, .. } => location,
            CheckedWord::If(iff) => &iff.location,
            CheckedWord::Loop(lop) => &lop.location,
            CheckedWord::Break { location } => location,
            CheckedWord::String { location, .. } => location,
        }
    }
}

impl Word {
    pub fn location(&self) -> &Location {
        match self {
            Word::Call { location, .. } => location,
            Word::Var { location, .. } => location,
            Word::Set { location, .. } => location,
            Word::Number { location, .. } => location,
            Word::Intrinsic { location, .. } => location,
            Word::If(iff) => &iff.location,
            Word::Loop(lop) => &lop.location,
            Word::Break { location } => location,
            Word::String { location, .. } => location,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Local {
    pub ident: String,
    pub location: Location,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct Memory {
    pub ident: String,
    pub location: Location,
    pub size: i32,
    pub alignment: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub location: Location,
    pub ident: String,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub location: Location,
    pub params: Vec<Param>,
    pub ret: Vec<Type>,
    pub ident: String,
    pub export: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CheckedFunctionSignature {
    pub location: Location,
    pub params: Vec<Param>,
    pub ret: Vec<Type>,
    pub ident: String,
    pub export: Option<String>,
    pub prefix: String,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub signature: FunctionSignature,
    pub locals: Vec<Local>,
    pub body: Vec<Word>,
    pub memory: Vec<Memory>,
}

#[derive(Debug, Clone)]
pub struct CheckedFunction {
    pub signature: CheckedFunctionSignature,
    pub locals: Vec<Local>,
    pub body: Vec<CheckedWord>,
    pub memory: Vec<Memory>,
}

#[derive(Debug, Clone)]
pub struct Extern {
    pub location: Location,
    pub signature: FunctionSignature,
    pub path: (String, String),
}

#[derive(Debug, Clone)]
pub struct CheckedExtern {
    pub location: Location,
    pub signature: CheckedFunctionSignature,
    pub path: (String, String),
}

#[derive(Debug, Clone)]
pub struct Import {
    pub path: String,
    pub ident: String,
}

#[derive(Debug, Clone)]
pub struct CheckedImport {
    pub path: PathBuf,
    pub ident: String,
}

#[derive(Debug, Clone)]
pub struct Module {
    pub externs: Vec<Extern>,
    pub imports: Vec<Import>,
    pub functions: Vec<Function>,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct CheckedModule {
    pub externs: Vec<CheckedExtern>,
    pub imports: HashMap<String, CheckedImport>,
    pub functions: Vec<CheckedFunction>,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub data: Vec<u8>,
    pub modules: HashMap<PathBuf, CheckedModule>,
}

#[derive(Debug, Clone)]
pub struct Data {
    pub location: Location,
    pub addr: i32,
    pub data: String,
}
