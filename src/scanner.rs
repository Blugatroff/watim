use std::{
    fmt::Display,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug, Clone, Default)]
pub struct Location {
    pub path: Arc<PathBuf>,
    pub line: usize,
    pub column: usize,
    pub len: usize,
}

impl std::cmp::PartialEq for Location {
    fn eq(&self, other: &Self) -> bool {
        self.line == other.line && self.column == other.column && self.path == other.path
    }
}
impl std::cmp::Eq for Location {}

impl Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}:{}:{}",
            self.path.display(),
            self.line,
            self.column
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Eof,
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    String(String),
    Number(i32),
    Identifier(String),
    Fn,
    I32,
    Arrow,
    Extern,
    Local,
    Colon,
    Dollar,
    If,
    Else,
    Loop,
    Break,
    Hash,
    Memory,
    Semicolon,
    Bool,
    Dot,
    Import,
    As,
    Bang,
}

impl Token {
    pub fn ty(&self) -> TokenType {
        TokenType::from(self)
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum TokenType {
    Eof,
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    String,
    Number,
    Identifier,
    Fn,
    I32,
    Arrow,
    Extern,
    Local,
    Colon,
    Dollar,
    If,
    Else,
    Loop,
    Break,
    Hash,
    Memory,
    Semicolon,
    Bool,
    Dot,
    Import,
    As,
    Bang,
}

impl From<&Token> for TokenType {
    fn from(token: &Token) -> Self {
        match token {
            Token::Eof => Self::Eof,
            Token::LeftParen => Self::LeftParen,
            Token::RightParen => Self::RightParen,
            Token::LeftBrace => Self::LeftBrace,
            Token::RightBrace => Self::RightBrace,
            Token::Comma => Self::Comma,
            Token::String(_) => Self::String,
            Token::Number(_) => Self::Number,
            Token::Identifier(_) => Self::Identifier,
            Token::Fn => Self::Fn,
            Token::I32 => Self::I32,
            Token::Arrow => Self::Arrow,
            Token::Extern => Self::Extern,
            Token::Local => Self::Local,
            Token::Colon => Self::Colon,
            Token::Dollar => Self::Dollar,
            Token::If => Self::If,
            Token::Else => Self::Else,
            Token::Loop => Self::Loop,
            Token::Break => Self::Break,
            Token::Hash => Self::Hash,
            Token::Memory => Self::Memory,
            Token::Semicolon => Self::Semicolon,
            Token::Bool => Self::Bool,
            Token::Dot => Self::Dot,
            Token::Import => Self::Import,
            Token::As => Self::As,
            Token::Bang => Self::Bang,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenWithLocation {
    pub location: Location,
    pub lexeme: String,
    pub token: Token,
}

impl std::ops::Deref for TokenWithLocation {
    type Target = Token;

    fn deref(&self) -> &Self::Target {
        &self.token
    }
}

impl TokenWithLocation {
    fn new(lexeme: impl Into<String>, location: Location, token: Token) -> Self {
        Self {
            location,
            lexeme: lexeme.into(),
            token,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ScanError {
    #[error("{0}: unexpected character!")]
    UnexpectedCharacter(Location),
    #[error("{0}: unterminated string!")]
    UnterminatedString(Location),
}

pub struct Scanner {
    source: Vec<char>,
    tokens: Vec<TokenWithLocation>,
    start: usize,
    line: usize,
    current: usize,
    path: Arc<PathBuf>,
}

impl Scanner {
    pub fn new(source: String, file: impl AsRef<Path>) -> Self {
        let path = Arc::new(PathBuf::from(file.as_ref()));
        Self {
            source: source.chars().collect(),
            tokens: Vec::new(),
            start: 0,
            current: 0,
            line: 1,
            path,
        }
    }
    pub fn scan_tokens(
        source: String,
        file: impl AsRef<Path>,
    ) -> Result<Vec<TokenWithLocation>, ScanError> {
        let mut this = Self::new(source, file);
        loop {
            this.start = this.current;
            match this.scan_token()? {
                Some(token) => this.tokens.push(token),
                None => break,
            }
        }
        let column = this.column();
        this.tokens.push(TokenWithLocation {
            location: Location {
                path: this.path.clone(),
                line: this.line,
                column,
                len: 1,
            },
            lexeme: "".into(),
            token: Token::Eof,
        });
        Ok(this.tokens)
    }
    fn column(&self) -> usize {
        let mut column = 0;
        loop {
            if column > self.start {
                break column;
            }
            match self.source.get(self.start - column) {
                None => break column,
                Some('\n') => break column,
                _ => {
                    column += 1;
                }
            }
        }
    }
    fn scan_token(&mut self) -> Result<Option<TokenWithLocation>, ScanError> {
        let c = match self.advance() {
            Some(c) => c,
            None => return Ok(None),
        };
        Ok(Some(match c {
            '(' => TokenWithLocation::new('(', self.location(), Token::LeftParen),
            ')' => TokenWithLocation::new(')', self.location(), Token::RightParen),
            '{' => TokenWithLocation::new('{', self.location(), Token::LeftBrace),
            '}' => TokenWithLocation::new('}', self.location(), Token::RightBrace),
            ',' => TokenWithLocation::new(',', self.location(), Token::Comma),
            ':' => TokenWithLocation::new(':', self.location(), Token::Colon),
            '$' => TokenWithLocation::new("$", self.location(), Token::Dollar),
            '#' => TokenWithLocation::new("#", self.location(), Token::Hash),
            ';' => TokenWithLocation::new(";", self.location(), Token::Semicolon),
            '.' => TokenWithLocation::new(".", self.location(), Token::Dot),
            '!' => TokenWithLocation::new("!", self.location(), Token::Bang),
            '-' if self.matsch('>') => TokenWithLocation::new("->", self.location(), Token::Arrow),
            '/' if self.matsch('/') => {
                loop {
                    match self.advance() {
                        None => break,
                        Some('\n') => {
                            self.line += 1;
                            break;
                        }
                        _ => {}
                    }
                }
                self.start = self.current;
                return self.scan_token();
            }
            ' ' | '\r' | '\t' => {
                return {
                    self.start = self.current;
                    self.scan_token()
                }
            }
            '\n' => {
                self.line += 1;
                self.start = self.current;
                return self.scan_token();
            }
            '"' => {
                return self.string().map(Some);
            }
            c if c.is_ascii_digit() => return self.number().map(Some),
            c if allowed_in_ident(c) => return self.identifier().map(Some),
            _ => return Err(ScanError::UnexpectedCharacter(self.location())),
        }))
    }
    fn peek(&self) -> Option<char> {
        self.source.get(self.current).copied()
    }
    fn matsch(&mut self, expected: char) -> bool {
        if self.source.get(self.current) == Some(&expected) {
            self.current += 1;
            return true;
        }
        false
    }
    fn location(&self) -> Location {
        Location {
            line: self.line,
            column: self.column(),
            path: self.path.clone(),
            len: self.current - self.start,
        }
    }
    fn advance(&mut self) -> Option<char> {
        let c = self.source.get(self.current)?;
        self.current += 1;
        Some(*c)
    }
    fn lexeme(&self) -> String {
        String::from_iter(&self.source[self.start..self.current])
    }
    fn string(&mut self) -> Result<TokenWithLocation, ScanError> {
        let mut string: String = String::new();
        loop {
            match self.peek() {
                None => return Err(ScanError::UnterminatedString(self.location())),
                Some('"') => break,
                Some('\n') => {
                    self.line += 1;
                }
                _ => {}
            }
            string.push(self.advance().unwrap());
        }
        self.advance(); // the closing "
        Ok(TokenWithLocation::new(
            self.lexeme(),
            self.location(),
            Token::String(string),
        ))
    }
    fn number(&mut self) -> Result<TokenWithLocation, ScanError> {
        loop {
            match self.peek() {
                Some(c) if c.is_ascii_digit() => {
                    self.advance();
                }
                _ => break,
            }
        }
        let number = self.lexeme();
        Ok(TokenWithLocation {
            lexeme: number.clone(),
            location: self.location(),
            token: Token::Number(number.parse().unwrap()),
        })
    }
    fn identifier(&mut self) -> Result<TokenWithLocation, ScanError> {
        while self.peek().map(allowed_in_ident).unwrap_or(false) {
            self.advance();
        }
        let ident = self.lexeme().trim().to_string();
        Ok(TokenWithLocation::new(
            ident.clone(),
            self.location(),
            match ident.as_str() {
                "fn" => Token::Fn,
                "local" => Token::Local,
                "i32" => Token::I32,
                "extern" => Token::Extern,
                "if" => Token::If,
                "else" => Token::Else,
                "loop" => Token::Loop,
                "break" => Token::Break,
                "memory" => Token::Memory,
                "bool" => Token::Bool,
                "import" => Token::Import,
                "as" => Token::As,
                _ => Token::Identifier(ident),
            },
        ))
    }
}

fn allowed_in_ident(char: char) -> bool {
    let disallowed = ['{', '}', '(', ')', ' ', ';', '\t', '\n', ':', ',', '.'];
    for c in disallowed {
        if char == c {
            return false;
        }
    }
    true
}
