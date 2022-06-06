use crate::{
    ast::{
        Data, Extern, Function, FunctionSignature, Iff, Intrinsic, Local, Loop, Param, Program,
        Type, Word,
    },
    scanner::{Token, TokenType, TokenWithLocation},
};

pub struct Parser {
    tokens: Vec<TokenWithLocation>,
    current: usize,
}

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

#[derive(Debug)]
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
    fn ext(&mut self) -> Result<Extern, ParseError> {
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
    fn function(&mut self) -> Result<Function, ParseError> {
        let signature = self.function_signature()?;
        self.expect(TokenType::LeftBrace)?;
        let mut locals = Vec::new();
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
        })
    }
    fn data(&mut self) -> Result<Data, ParseError> {
        let location = self.expect(TokenType::Data)?.location;
        let addr = match self.advance() {
            Some(TokenWithLocation {
                token: Token::Number(addr),
                ..
            }) => addr,
            _ => {
                return Err(ParseError::new(
                    self.peek().unwrap(),
                    ParseErrorType::ExpectedToken(TokenType::Number),
                ))
            }
        };
        let data = match self.advance() {
            Some(TokenWithLocation {
                token: Token::String(data),
                ..
            }) => data,
            _ => {
                return Err(ParseError::new(
                    self.peek().unwrap(),
                    ParseErrorType::ExpectedToken(TokenType::String),
                ))
            }
        };
        Ok(Data {
            location,
            addr,
            data,
        })
    }
    fn body(&mut self) -> Result<Vec<Word>, ParseError> {
        let mut words = Vec::new();
        while self
            .peek()
            .map(|t| {
                let ty = t.ty();
                ty == TokenType::Identifier
                    || ty == TokenType::Dollar
                    || ty == TokenType::Number
                    || ty == TokenType::If
                    || ty == TokenType::Loop
                    || ty == TokenType::Break
                    || ty == TokenType::Hash
            })
            .unwrap_or(false)
        {
            let word = self.word()?;
            words.push(word);
        }
        Ok(words)
    }
    fn iff(&mut self) -> Result<Iff, ParseError> {
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
    fn lop(&mut self) -> Result<Loop, ParseError> {
        let location = self.expect(TokenType::Loop)?.location;
        self.expect(TokenType::LeftBrace)?;
        let body = self.body()?;
        self.expect(TokenType::RightBrace)?;
        Ok(Loop { location, body })
    }
    fn word(&mut self) -> Result<Word, ParseError> {
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
                } else {
                    Ok(Word::Call {
                        location: ident.location,
                        ident: ident.lexeme,
                    })
                }
            }
            Some(token) if token.ty() == TokenType::Dollar => {
                let ident = self.ident()?;
                Ok(Word::Var {
                    location: ident.location,
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
    fn local(&mut self) -> Result<Local, ParseError> {
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
    fn function_signature(&mut self) -> Result<FunctionSignature, ParseError> {
        self.expect(TokenType::Fn)?;
        let ident = self.ident()?;
        let export = if let Some(Token::String(export)) = self.peek().map(|t| t.token) {
            self.expect(TokenType::String)?;
            Some(export.clone())
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
            None => {
                return Err(ParseError::new(
                    self.peek().unwrap(),
                    ParseErrorType::ExpectedIdent,
                ))
            }
        }
    }
    fn params(&mut self) -> Result<Vec<Param>, ParseError> {
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
    fn param(&mut self) -> Result<Param, ParseError> {
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
    fn ty(&mut self) -> Result<Type, ParseError> {
        match self.matsch_any([TokenType::I32]).as_deref() {
            Some(Token::I32) => return Ok(Type::I32),
            _ => {}
        }
        return Err(ParseError::new(
            self.peek().unwrap(),
            ParseErrorType::ExpectedType,
        ));
    }
    pub fn parse(&mut self) -> Result<Program, ParseError> {
        let mut externs = Vec::new();
        let mut functions = Vec::new();
        let mut data = Vec::new();
        while self
            .peek()
            .map(|t| t.ty() != TokenType::Eof)
            .unwrap_or(false)
        {
            if self
                .peek()
                .map(|t| t.ty() == TokenType::Extern)
                .unwrap_or(false)
            {
                externs.push(self.ext()?);
            } else if self
                .peek()
                .map(|t| t.ty() == TokenType::Data)
                .unwrap_or(false)
            {
                data.push(self.data()?);
            } else {
                functions.push(self.function()?);
            }
        }
        Ok(Program {
            externs,
            functions,
            data,
        })
    }
}
