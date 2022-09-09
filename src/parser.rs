use crate::{
    ast::{
        Extern, Function, FunctionSignature, Ident, Iff, Import, Intrinsic, Local, Loop, Memory,
        Module, Param, Struct, UnResolvedType, Word,
    },
    scanner::{Location, Token, TokenType, TokenWithLocation},
};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;

pub struct Parser {
    tokens: Vec<TokenWithLocation>,
    current: usize,
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug)]
pub enum ParseErrorType {
    ExpectedIdent,
    ExpectedType,
    ExpectedColon,
    ExpectedToken(TokenType),
    ExpectedWord,
}

impl std::fmt::Display for ParseErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseErrorType::ExpectedIdent => f.write_str("Expected Identifier"),
            ParseErrorType::ExpectedType => f.write_str("Expected Type"),
            ParseErrorType::ExpectedColon => f.write_str("Expected ':'"),
            ParseErrorType::ExpectedToken(ty) => f.write_fmt(format_args!("expected {:?}", ty)),
            ParseErrorType::ExpectedWord => f.write_str("expected word"),
        }
    }
}

#[derive(Error, Debug)]
pub struct ParseError {
    ty: ParseErrorType,
    token: TokenWithLocation,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{} ParsingError: {} -> {}",
            self.token.location, self.ty, self.token.lexeme
        ))
    }
}

impl ParseError {
    fn new(token: TokenWithLocation, ty: ParseErrorType) -> Self {
        Self { ty, token }
    }
}

impl Parser {
    pub fn new(tokens: Vec<TokenWithLocation>) -> Self {
        Parser { tokens, current: 0 }
    }
    fn matsch(&mut self, ty: TokenType) -> Option<TokenWithLocation> {
        if self.check(ty) {
            self.advance()
        } else {
            None
        }
    }
    fn matsch_any(
        &mut self,
        types: impl IntoIterator<Item = TokenType>,
    ) -> Option<TokenWithLocation> {
        for ty in types {
            if let Some(token) = self.matsch(ty) {
                return Some(token);
            }
        }
        None
    }
    fn check(&self, ty: TokenType) -> bool {
        match self.peek() {
            Some(t) => t.ty() == ty,
            None => false,
        }
    }
    fn peek(&self) -> Option<TokenWithLocation> {
        self.tokens.get(self.current).cloned()
    }
    fn advance(&mut self) -> Option<TokenWithLocation> {
        self.current += 1;
        self.previous()
    }
    fn previous(&self) -> Option<TokenWithLocation> {
        self.tokens.get(self.current - 1).cloned()
    }
    fn expect(&mut self, token_type: TokenType) -> Result<TokenWithLocation, ParseError> {
        match self.matsch(token_type) {
            None => Err(ParseError::new(
                self.peek().unwrap(),
                ParseErrorType::ExpectedToken(token_type),
            )),
            Some(t) => Ok(t),
        }
    }
    fn ext(&mut self) -> Result<Extern<UnResolvedType>, ParseError> {
        let ext = self.expect(TokenType::Extern)?;
        let path = self.expect(TokenType::String)?;
        let path_0 = {
            if let Token::String(path) = path.token {
                path
            } else {
                unreachable!()
            }
        };
        let path = self.expect(TokenType::String)?;
        let path_1 = {
            if let Token::String(path) = path.token {
                path
            } else {
                unreachable!()
            }
        };
        let path = (path_0, path_1);
        let signature = self.function_signature()?;
        Ok(Extern {
            location: ext.location,
            signature,
            path,
        })
    }
    fn import(&mut self) -> Result<Import, ParseError> {
        let location = self.expect(TokenType::Import)?.location;
        match self.advance() {
            Some(TokenWithLocation {
                token: Token::String(path),
                ..
            }) => {
                self.expect(TokenType::As)?;
                let ident = self.ident()?;
                Ok(Import {
                    path,
                    ident: ident.lexeme,
                    location,
                })
            }
            _ => Err(ParseError::new(
                self.peek().unwrap(),
                ParseErrorType::ExpectedToken(TokenType::String),
            )),
        }
    }
    fn struc(&mut self) -> Result<Struct<UnResolvedType>, ParseError> {
        self.expect(TokenType::Struct)?;
        let ident = self.ident()?;
        self.expect(TokenType::LeftBrace)?;
        let mut fields = Vec::new();
        while self
            .peek()
            .map(|t| t.ty() != TokenType::RightBrace)
            .unwrap_or(false)
        {
            let ident = self.ident()?;
            self.expect(TokenType::Colon)?;
            let ty = self.ty()?;
            fields.push((ident, ty));
        }
        self.expect(TokenType::RightBrace)?;
        Ok(Struct { ident, fields })
    }
    fn function(&mut self) -> Result<Function<UnResolvedType>, ParseError> {
        let signature = self.function_signature()?;
        self.expect(TokenType::LeftBrace)?;
        let mut locals = Vec::new();
        let mut memory = Vec::new();
        while self
            .peek()
            .map(|t| t.ty() == TokenType::Memory)
            .unwrap_or(false)
        {
            memory.push(self.memory()?);
        }
        while self
            .peek()
            .map(|t| t.ty() == TokenType::Local)
            .unwrap_or(false)
        {
            locals.push(self.local()?);
        }
        let body = self.body()?;
        self.expect(TokenType::RightBrace)?;
        Ok(Function {
            signature,
            locals,
            body,
            memory,
        })
    }
    fn body(&mut self) -> Result<Vec<Word<UnResolvedType>>, ParseError> {
        let mut words = Vec::new();
        let is_start_of_word = |t: TokenWithLocation| {
            let ty = t.ty();
            ty == TokenType::Identifier
                || ty == TokenType::Dollar
                || ty == TokenType::Number
                || ty == TokenType::If
                || ty == TokenType::Loop
                || ty == TokenType::Break
                || ty == TokenType::Hash
                || ty == TokenType::String
                || ty == TokenType::Bang
                || ty == TokenType::Dot
        };
        while self.peek().map(is_start_of_word).unwrap_or(false) {
            let word = self.word()?;
            words.push(word);
        }
        Ok(words)
    }
    fn iff(&mut self) -> Result<Iff<UnResolvedType>, ParseError> {
        let location = self.expect(TokenType::If)?.location;
        self.expect(TokenType::LeftBrace)?;
        let body = self.body()?;
        self.expect(TokenType::RightBrace)?;
        let el = if self.matsch(TokenType::Else).is_some() {
            self.expect(TokenType::LeftBrace)?;
            let body = self.body()?;
            self.expect(TokenType::RightBrace)?;
            Some(body)
        } else {
            None
        };
        Ok(Iff { location, body, el })
    }
    fn lop(&mut self) -> Result<Loop<UnResolvedType>, ParseError> {
        let location = self.expect(TokenType::Loop)?.location;
        self.expect(TokenType::LeftBrace)?;
        let body = self.body()?;
        self.expect(TokenType::RightBrace)?;
        Ok(Loop { location, body })
    }
    fn word(&mut self) -> Result<Word<UnResolvedType>, ParseError> {
        if self
            .peek()
            .map(|t| t.ty() == TokenType::If)
            .unwrap_or(false)
        {
            return self.iff().map(Word::If);
        }
        if self
            .peek()
            .map(|t| t.ty() == TokenType::Loop)
            .unwrap_or(false)
        {
            return self.lop().map(Word::Loop);
        }
        if let Some(t) = self.matsch(TokenType::Break) {
            return Ok(Word::Break {
                location: t.location,
            });
        }
        if let Some(t) = self.matsch(TokenType::String) {
            return Ok(Word::String {
                location: t.location,
                value: match t.token {
                    Token::String(value) => value,
                    _ => unreachable!(),
                },
            });
        }
        if let Some(t) = self.matsch(TokenType::Bang) {
            let ty = self.ty()?;
            return Ok(Word::Intrinsic {
                location: t.location,
                intrinsic: Intrinsic::Cast(ty),
            });
        }
        if let Some(t) = self.matsch(TokenType::Dot) {
            let field = self.ident()?;
            return Ok(Word::FieldDeref {
                location: t.location,
                field: field.lexeme,
            });
        }
        match self.advance() {
            Some(ident) if ident.ty() == TokenType::Identifier => {
                if &ident.lexeme == "+" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Add,
                    })
                } else if &ident.lexeme == "store32" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Store32,
                    })
                } else if &ident.lexeme == "load32" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Load32,
                    })
                } else if &ident.lexeme == "store8" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Store8,
                    })
                } else if &ident.lexeme == "load8" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Load8,
                    })
                } else if &ident.lexeme == "drop" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Drop,
                    })
                } else if &ident.lexeme == "-" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Sub,
                    })
                } else if &ident.lexeme == "=" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Eq,
                    })
                } else if &ident.lexeme == "/=" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::NotEq,
                    })
                } else if &ident.lexeme == "%" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Mod,
                    })
                } else if &ident.lexeme == "/" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Div,
                    })
                } else if &ident.lexeme == "<" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::L,
                    })
                } else if &ident.lexeme == ">" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::G,
                    })
                } else if &ident.lexeme == "<=" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::LE,
                    })
                } else if &ident.lexeme == ">=" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::GE,
                    })
                } else if &ident.lexeme == "and" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::And,
                    })
                } else if &ident.lexeme == "or" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Or,
                    })
                } else if &ident.lexeme == "not" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Not,
                    })
                } else if &ident.lexeme == "*" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Mul,
                    })
                } else if &ident.lexeme == "rotr" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Rotr,
                    })
                } else if &ident.lexeme == "rotl" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::Rotl,
                    })
                } else if &ident.lexeme == "mem-grow" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::MemGrow,
                    })
                } else if &ident.lexeme == "mem-copy" {
                    Ok(Word::Intrinsic {
                        location: ident.location,
                        intrinsic: Intrinsic::MemCopy,
                    })
                } else if self.matsch(TokenType::Colon).is_some() {
                    let ident_2 = self.ident()?;
                    Ok(Word::Call {
                        location: ident.location,
                        ident: Ident::Qualified(ident.lexeme, ident_2.lexeme),
                    })
                } else {
                    Ok(Word::Call {
                        location: ident.location,
                        ident: Ident::Direct(ident.lexeme),
                    })
                }
            }
            Some(token) if token.ty() == TokenType::Dollar => {
                let ident = self.ident()?;
                Ok(Word::Var {
                    location: Location {
                        path: ident.location.path,
                        line: ident.location.line,
                        column: token.location.column,
                        len: ident.location.len + (ident.location.column - token.location.column),
                    },
                    ident: ident.lexeme,
                })
            }
            Some(token) if token.ty() == TokenType::Hash => {
                let ident = self.ident()?;
                Ok(Word::Set {
                    location: ident.location,
                    ident: ident.lexeme,
                })
            }
            Some(TokenWithLocation {
                location,
                token: Token::Number(number),
                ..
            }) => Ok(Word::Number { location, number }),
            Some(token) => Err(ParseError::new(token, ParseErrorType::ExpectedWord)),
            None => unreachable!(),
        }
    }
    fn local(&mut self) -> Result<Local<UnResolvedType>, ParseError> {
        self.expect(TokenType::Local)?;
        let ident = self.ident()?;
        self.expect(TokenType::Colon)?;
        let ty = self.ty()?;
        Ok(Local {
            ident: ident.lexeme,
            location: ident.location,
            ty,
        })
    }
    fn memory(&mut self) -> Result<Memory<UnResolvedType>, ParseError> {
        let location = self.expect(TokenType::Memory)?.location;
        let ident = self.ident()?;
        self.expect(TokenType::Colon)?;
        let ty = self.ty()?;
        match self.advance() {
            Some(TokenWithLocation {
                token: Token::Number(size),
                ..
            }) => {
                let alignment = if self
                    .peek()
                    .map(|t| t.ty() == TokenType::Number)
                    .unwrap_or(false)
                {
                    match self.expect(TokenType::Number)?.token {
                        Token::Number(alignment) => Some(alignment),
                        _ => unreachable!(),
                    }
                } else {
                    None
                };
                self.expect(TokenType::Semicolon)?;
                Ok(Memory {
                    ident: ident.lexeme,
                    location,
                    size,
                    alignment,
                    ty,
                })
            }
            _ => Err(ParseError::new(
                self.peek().unwrap(),
                ParseErrorType::ExpectedToken(TokenType::String),
            )),
        }
    }
    fn function_signature(&mut self) -> Result<FunctionSignature<UnResolvedType>, ParseError> {
        self.expect(TokenType::Fn)?;
        let ident = self.ident()?;
        let export = if let Some(Token::String(export)) = self.peek().map(|t| t.token) {
            self.expect(TokenType::String)?;
            Some(export)
        } else {
            None
        };
        self.expect(TokenType::LeftParen)?;
        let params = self.params()?;
        self.expect(TokenType::RightParen)?;
        let ret = if self.matsch(TokenType::Arrow).is_some() {
            let mut rets = vec![self.ty()?];
            while self
                .peek()
                .map(|t| t.ty() == TokenType::Comma)
                .unwrap_or(false)
            {
                self.expect(TokenType::Comma)?;
                rets.push(self.ty()?);
            }
            rets
        } else {
            Vec::new()
        };
        Ok(FunctionSignature {
            location: ident.location,
            ident: ident.lexeme,
            params,
            ret,
            export,
        })
    }
    fn ident(&mut self) -> Result<TokenWithLocation, ParseError> {
        match self.matsch(TokenType::Identifier) {
            Some(token) => Ok(token),
            None => Err(ParseError::new(
                self.peek().unwrap(),
                ParseErrorType::ExpectedIdent,
            )),
        }
    }
    fn params(&mut self) -> Result<Vec<Param<UnResolvedType>>, ParseError> {
        let mut params = Vec::new();
        while self
            .peek()
            .map(|t| t.ty() != TokenType::RightParen)
            .unwrap_or(false)
        {
            params.push(self.param()?);
            if self.matsch(TokenType::Comma).is_none() {
                break;
            }
        }
        Ok(params)
    }
    fn param(&mut self) -> Result<Param<UnResolvedType>, ParseError> {
        let ident = self.ident()?;
        if self.matsch(TokenType::Colon).is_none() {
            return Err(ParseError::new(
                self.peek().unwrap(),
                ParseErrorType::ExpectedColon,
            ));
        }
        let ty = self.ty()?;
        Ok(Param {
            ident: ident.lexeme,
            location: ident.location,
            ty,
        })
    }
    fn ty(&mut self) -> Result<UnResolvedType, ParseError> {
        match self
            .matsch_any([
                TokenType::I32,
                TokenType::I64,
                TokenType::Bool,
                TokenType::Dot,
                TokenType::Identifier,
            ])
            .as_deref()
        {
            Some(Token::I32) => return Ok(UnResolvedType::I32),
            Some(Token::I64) => return Ok(UnResolvedType::I64),
            Some(Token::Bool) => return Ok(UnResolvedType::Bool),
            Some(Token::Dot) => {
                let ty = self.ty()?;
                return Ok(UnResolvedType::Ptr(Box::new(ty)));
            }
            Some(Token::Identifier(ident_1)) => match self.matsch(TokenType::Colon) {
                Some(_) => {
                    let ident_2 = self.ident()?;
                    return Ok(UnResolvedType::Custom(Ident::Qualified(
                        ident_1.clone(),
                        ident_2.lexeme,
                    )));
                }
                None => return Ok(UnResolvedType::Custom(Ident::Direct(ident_1.clone()))),
            },
            _ => {}
        }
        Err(ParseError::new(
            self.peek().unwrap(),
            ParseErrorType::ExpectedType,
        ))
    }
    pub fn parse(&mut self, path: impl AsRef<Path>) -> Result<Module<UnResolvedType>, ParseError> {
        let mut externs = Vec::new();
        let mut functions = Vec::new();
        let mut imports = Vec::new();
        let mut structs = Vec::new();
        let mut memory = Vec::new();
        while let Some(next) = self.peek().map(|t| t.ty()) {
            match next {
                TokenType::Extern => {
                    externs.push(self.ext()?);
                }
                TokenType::Import => {
                    imports.push(self.import()?);
                }
                TokenType::Struct => {
                    structs.push(Arc::new(self.struc()?));
                }
                TokenType::Memory => {
                    memory.push(self.memory()?);
                }
                TokenType::Eof => {
                    break;
                }
                _ => {
                    functions.push(self.function()?);
                }
            }
        }
        Ok(Module {
            externs,
            functions,
            path: PathBuf::from(path.as_ref()).canonicalize().unwrap(),
            imports,
            structs,
            memory,
        })
    }
}
