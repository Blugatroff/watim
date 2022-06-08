use crate::scanner::Location;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    I32,
    Bool,
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
    Mod,
    Div,
    Stack,
    And,
    Or,
    L,
    G,
    LE,
    GE,
    Mul,
}

#[derive(Debug, Clone)]
pub struct Iff {
    pub location: Location,
    pub body: Vec<Word>,
    pub el: Option<Vec<Word>>,
    pub ret: Vec<Type>,
}

#[derive(Debug, Clone)]
pub struct Loop {
    pub location: Location,
    pub body: Vec<Word>,
    pub ret: Vec<Type>,
}

#[derive(Debug, Clone)]
pub enum Word {
    Call {
        location: Location,
        ident: String,
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
        addr: i32,
        size: i32,
    },
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
pub struct Function {
    pub signature: FunctionSignature,
    pub locals: Vec<Local>,
    pub body: Vec<Word>,
    pub memory: Vec<Memory>,
}

#[derive(Debug, Clone)]
pub struct Extern {
    pub location: Location,
    pub signature: FunctionSignature,
    pub path: (String, String),
}

#[derive(Debug, Clone)]
pub struct Program {
    pub externs: Vec<Extern>,
    pub functions: Vec<Function>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Data {
    pub location: Location,
    pub addr: i32,
    pub data: String,
}
