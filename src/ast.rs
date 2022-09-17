use crate::{
    checker::Returns,
    scanner::{Location, TokenWithLocation},
};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnResolvedType {
    I32,
    I64,
    Bool,
    Ptr(Box<UnResolvedType>),
    Custom(Ident),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedType {
    I32,
    I64,
    Bool,
    Ptr(Box<ResolvedType>),
    AnyPtr,
    Custom(Arc<Struct<ResolvedType>>),
}

impl std::fmt::Display for ResolvedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolvedType::I32 => f.write_str("i32"),
            ResolvedType::Bool => f.write_str("bool"),
            ResolvedType::Ptr(ty) => f.write_fmt(format_args!(".{ty}")),
            ResolvedType::Custom(struc) => f.write_fmt(format_args!("{}", &struc.ident.lexeme)),
            ResolvedType::AnyPtr => f.write_str("AnyPtr"),
            ResolvedType::I64 => f.write_str("i64"),
        }
    }
}

impl std::fmt::Display for UnResolvedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnResolvedType::I32 => f.write_str("i32"),
            UnResolvedType::I64 => f.write_str("i64"),
            UnResolvedType::Bool => f.write_str("bool"),
            UnResolvedType::Ptr(ty) => f.write_fmt(format_args!(".{ty}")),
            UnResolvedType::Custom(struc) => f.write_fmt(format_args!(
                "{}",
                match struc {
                    Ident::Direct(n) => n,
                    Ident::Qualified(_, n) => n,
                }
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Intrinsic<Type> {
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
    Not,
    Or,
    L,
    G,
    LE,
    GE,
    Mul,
    Rotr,
    Rotl,
    MemGrow,
    MemCopy,
    Flip,
    Cast(Type),
}

#[derive(Debug, Clone)]
pub enum CheckedIntrinsic {
    Add,
    Store32,
    Store8,
    Load32,
    Load8,
    Drop,
    Sub,
    Eq(ResolvedType),
    NotEq(ResolvedType),
    Mod(ResolvedType),
    Div(ResolvedType),
    And(ResolvedType),
    Not,
    Or(ResolvedType),
    L,
    G,
    LE,
    GE,
    Mul,
    MemGrow,
    MemCopy,
    Flip,
    Rotr(ResolvedType),
    Rotl(ResolvedType),
    Cast(ResolvedType, ResolvedType),
}

#[derive(Debug, Clone)]
pub struct Iff<Type> {
    pub location: Location,
    pub body: Vec<Word<Type>>,
    pub el: Option<Vec<Word<Type>>>,
}

#[derive(Debug, Clone)]
pub struct CheckedIff {
    pub location: Location,
    pub body: Vec<CheckedWord>,
    pub el: Option<Vec<CheckedWord>>,
    pub ret: Vec<ResolvedType>,
    pub param: Vec<ResolvedType>,
}

#[derive(Debug, Clone)]
pub struct Loop<Type> {
    pub location: Location,
    pub body: Vec<Word<Type>>,
}

#[derive(Debug, Clone)]
pub struct CheckedLoop {
    pub location: Location,
    pub body: Vec<CheckedWord>,
    pub ret: Vec<ResolvedType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
pub enum Word<Type> {
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
        intrinsic: Intrinsic<Type>,
    },
    If(Iff<Type>),
    Loop(Loop<Type>),
    Break {
        location: Location,
    },
    String {
        location: Location,
        value: String,
    },
    FieldDeref {
        location: Location,
        field: String,
    },
}

#[derive(Debug, Clone)]
pub enum CheckedWord {
    Call {
        location: Location,
        ident: CheckedIdent,
    },
    Local {
        location: Location,
        ident: String,
    },
    Global {
        location: Location,
        ident: CheckedIdent,
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
        intrinsic: CheckedIntrinsic,
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
    FieldDeref {
        location: Location,
        offset: u32,
        ty: ResolvedType,
    },
}

impl CheckedWord {
    pub fn location(&self) -> &Location {
        match self {
            CheckedWord::Call { location, .. } => location,
            CheckedWord::Local { location, .. } => location,
            CheckedWord::Set { location, .. } => location,
            CheckedWord::Number { location, .. } => location,
            CheckedWord::Intrinsic { location, .. } => location,
            CheckedWord::If(iff) => &iff.location,
            CheckedWord::Loop(lop) => &lop.location,
            CheckedWord::Break { location } => location,
            CheckedWord::String { location, .. } => location,
            CheckedWord::FieldDeref { location, .. } => location,
            CheckedWord::Global { location, .. } => location,
        }
    }
}

impl<Type> Word<Type> {
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
            Word::FieldDeref { location, .. } => location,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Local<Type> {
    pub ident: String,
    pub location: Location,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct Memory<Type, Ident = String> {
    pub ident: Ident,
    pub location: Location,
    pub size: i32,
    pub alignment: Option<i32>,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct Param<Type> {
    pub location: Location,
    pub ident: String,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct FunctionSignature<Type> {
    pub location: Location,
    pub params: Vec<Param<Type>>,
    pub ret: Vec<Type>,
    pub ident: String,
    pub export: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CheckedFunctionSignature {
    pub location: Location,
    pub params: Vec<Param<ResolvedType>>,
    pub ret: Vec<ResolvedType>,
    pub ident: String,
    pub export: Option<String>,
    pub prefix: String,
}

#[derive(Debug, Clone)]
pub struct Function<Type> {
    pub signature: FunctionSignature<Type>,
    pub locals: Vec<Local<Type>>,
    pub body: Vec<Word<Type>>,
    pub memory: Vec<Memory<Type>>,
}

#[derive(Debug, Clone)]
pub struct CheckedFunction {
    pub signature: CheckedFunctionSignature,
    pub locals: Vec<Local<ResolvedType>>,
    pub body: Vec<CheckedWord>,
    pub memory: Vec<Memory<ResolvedType>>,
    pub returns: Returns,
}

#[derive(Debug, Clone)]
pub struct Extern<Type> {
    pub location: Location,
    pub signature: FunctionSignature<Type>,
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
    pub location: Location,
}

#[derive(Debug, Clone)]
pub struct CheckedImport {
    pub path: PathBuf,
    pub ident: String,
}

#[derive(Debug, Clone)]
pub struct Module<Type> {
    pub externs: Vec<Extern<Type>>,
    pub imports: Vec<Import>,
    pub functions: Vec<Function<Type>>,
    pub path: PathBuf,
    pub structs: Vec<Arc<Struct<Type>>>,
    pub memory: Vec<Memory<Type>>,
}

#[derive(Debug, Clone)]
pub struct CheckedModule {
    pub externs: Vec<CheckedExtern>,
    pub imports: BTreeMap<String, CheckedImport>,
    pub functions: Vec<CheckedFunction>,
    pub path: PathBuf,
    pub globals: Vec<Memory<ResolvedType, CheckedIdent>>,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub data: Vec<u8>,
    pub modules: BTreeMap<PathBuf, CheckedModule>,
    pub max_pages: u32,
}

#[derive(Debug, Clone)]
pub struct Data {
    pub location: Location,
    pub addr: i32,
    pub data: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Struct<Type> {
    pub ident: TokenWithLocation,
    pub fields: Vec<(TokenWithLocation, Type)>,
}
