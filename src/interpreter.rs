use crate::{
    ast::{
        CheckedFunction, CheckedFunctionSignature, CheckedIdent, CheckedIff, CheckedLoop,
        CheckedWord, Intrinsic, Local, Memory, Param, Program, Type,
    },
    scanner::Location,
};
use std::{
    collections::HashMap,
    io::{Read, Write},
};

#[derive(Debug)]
pub struct Interpreter {
    memory: [u8; 2usize.pow(16)],
    functions: HashMap<String, HashMap<String, InterpreterFunction>>,
    mem_stack: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Value {
    True,
    False,
    I32(i32),
}

impl Value {
    fn default_of_type(ty: &Type) -> Self {
        match ty {
            Type::I32 => Self::I32(0),
            Type::Bool => Self::False,
        }
    }
    fn ty(self) -> Type {
        match self {
            Self::True | Self::False => Type::Bool,
            Self::I32(_) => Type::I32,
        }
    }
}

#[derive(Debug, Clone)]
enum InterpreterFunction {
    Normal(CheckedFunction),
    Extern((String, String), CheckedFunctionSignature),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    FunctionNotFound(Location, CheckedIdent),
    NoEntryPoint,
    LocalNotFound(Location, String),
    LocalSetTypeMismatch(Location, Type, Type),
    ArgsMismatch(Location, Vec<Type>, Vec<Option<Type>>),
    ExpectedBool(Location, Option<Type>),
}

impl Interpreter {
    pub fn new(program: Program) -> Result<Self, Error> {
        let mut memory = [0u8; 2usize.pow(16)];
        for (i, b) in program.data.iter().enumerate() {
            memory[i] = *b;
        }
        let mut functions: HashMap<String, HashMap<String, InterpreterFunction>> = HashMap::new();
        for (_, module) in program.modules {
            for function in module.functions {
                functions
                    .entry(function.signature.prefix.clone())
                    .or_default()
                    .insert(
                        function.signature.ident.clone(),
                        InterpreterFunction::Normal(function),
                    );
            }
            for ex in module.externs {
                functions
                    .entry(ex.signature.prefix.clone())
                    .or_default()
                    .insert(
                        ex.signature.ident.clone(),
                        InterpreterFunction::Extern(ex.path, ex.signature),
                    );
            }
        }
        let mem_stack = program.data.len() as i32;
        let mut this = Self {
            memory,
            functions,
            mem_stack,
        };
        let entry = match this.find_entry_point() {
            Some(entry) => entry.clone(),
            None => return Err(Error::NoEntryPoint),
        }
        .clone();
        this.execute_fn(&entry, Vec::new())?;
        Ok(this)
    }
    fn find_entry_point(&self) -> Option<&InterpreterFunction> {
        self.functions
            .values()
            .flat_map(|m| m.values())
            .find(|f| match f {
                InterpreterFunction::Normal(f) => f.signature.export.as_deref() == Some("_start"),
                InterpreterFunction::Extern(_, _) => false,
            })
    }
    fn execute_fn(
        &mut self,
        function: &InterpreterFunction,
        mut args: Vec<Value>,
    ) -> Result<Vec<Value>, Error> {
        match function {
            InterpreterFunction::Normal(function) => {
                let mut locals: HashMap<String, Value> = function
                    .locals
                    .iter()
                    .map(|local: &Local| (local.ident.clone(), Value::default_of_type(&local.ty)))
                    .chain(function.signature.params.iter().map(|p: &Param| {
                        let v = args.pop().unwrap();
                        (p.ident.clone(), v)
                    }))
                    .chain(function.memory.iter().map(|mem: &Memory| {
                        let value = Value::I32(self.mem_stack);
                        self.mem_stack += mem.size;
                        (mem.ident.clone(), value)
                    }))
                    .collect();
                let mut stack: Vec<Value> = Vec::new();
                for word in &function.body {
                    self.execute_word(word, &mut locals, &mut stack)?;
                }
                Ok(stack)
            }
            InterpreterFunction::Extern((path_0, path_1), ..) => {
                match (path_0.as_str(), path_1.as_str()) {
                    ("wasi_unstable", "fd_write") => {
                        let file = match args.pop().unwrap() {
                            Value::I32(file) => file,
                            _ => todo!(),
                        };
                        let iovec_ptr = match args.pop().unwrap() {
                            Value::I32(ptr) => ptr,
                            _ => todo!(),
                        };
                        let len = match args.pop().unwrap() {
                            Value::I32(len) => len,
                            _ => {
                                todo!()
                            }
                        };
                        let nwritten = match args.pop().unwrap() {
                            Value::I32(nwritten) => nwritten,
                            _ => todo!(),
                        };
                        let iovec_ptr = iovec_ptr as usize;
                        let nwritten = nwritten as usize;
                        match file {
                            // stdout
                            1 => {
                                let mut written = 0;
                                for i in 0..len as usize {
                                    let i = i * 8;
                                    let ptr: [u8; 4] = self.memory
                                        [iovec_ptr + i..iovec_ptr + 4 + i]
                                        .try_into()
                                        .unwrap();
                                    let ptr: i32 = i32::from_le_bytes(ptr);
                                    let ptr = ptr as usize;

                                    let len: [u8; 4] = self.memory[iovec_ptr + 4..iovec_ptr + 8]
                                        .try_into()
                                        .unwrap();
                                    let len: i32 = i32::from_le_bytes(len);
                                    written += len;
                                    let len = len as usize;
                                    let data = &self.memory[ptr..ptr + len];
                                    //dbg!(String::from_utf8(data.to_vec()).unwrap());
                                    std::io::stdout().write_all(data).unwrap();
                                    std::io::stdout().flush().unwrap();
                                }
                                let res = Value::I32(written);
                                let written: [u8; 4] = written.to_le_bytes();
                                for (i, b) in written.into_iter().enumerate() {
                                    self.memory[nwritten + i] = b;
                                }
                                Ok(vec![res])
                            }
                            file => {
                                todo!("unhandled fd_write for file: {file}")
                            }
                        }
                    }
                    ("wasi_unstable", "fd_read") => {
                        let file = match args.pop().unwrap() {
                            Value::I32(file) => file,
                            _ => todo!(),
                        };
                        let iovec_ptr = match args.pop().unwrap() {
                            Value::I32(ptr) => ptr,
                            _ => todo!(),
                        };
                        let len = match args.pop().unwrap() {
                            Value::I32(len) => len,
                            _ => {
                                todo!()
                            }
                        };
                        let nwritten = match args.pop().unwrap() {
                            Value::I32(nwritten) => nwritten,
                            _ => todo!(),
                        };
                        let iovec_ptr = iovec_ptr as usize;
                        let nwritten = nwritten as usize;
                        match file {
                            // stdin
                            0 => {
                                let mut read = 0;
                                for i in 0..len as usize {
                                    let i = i * 8;
                                    let ptr: [u8; 4] = self.memory
                                        [iovec_ptr + i..iovec_ptr + 4 + i]
                                        .try_into()
                                        .unwrap();
                                    let ptr: i32 = i32::from_le_bytes(ptr);
                                    let ptr = ptr as usize;

                                    let len: [u8; 4] = self.memory
                                        [iovec_ptr + 4 + i..iovec_ptr + 8 + i]
                                        .try_into()
                                        .unwrap();
                                    let len = i32::from_le_bytes(len) as usize;
                                    read += std::io::stdin()
                                        .read(&mut self.memory[ptr..ptr + len])
                                        .unwrap()
                                        as i32;
                                }
                                let written: [u8; 4] = read.to_le_bytes();
                                for (i, b) in written.into_iter().enumerate() {
                                    self.memory[nwritten + i] = b;
                                }
                                Ok(vec![Value::I32(read)])
                            }
                            file => {
                                todo!("unhandled fd_write for file: {file}")
                            }
                        }
                    }
                    ("wasi_unstable", "proc_exit") => {
                        let code = args.pop().unwrap();
                        let code = match code {
                            Value::I32(code) => code,
                            _ => todo!(),
                        };
                        std::process::exit(code)
                    }
                    path => {
                        todo!("{path:?}")
                    }
                }
            }
        }
    }
    fn execute_word(
        &mut self,
        word: &CheckedWord,
        locals: &mut HashMap<String, Value>,
        stack: &mut Vec<Value>,
    ) -> Result<bool, Error> {
        match word {
            CheckedWord::Call { location, ident } => {
                match self
                    .functions
                    .get(&ident.module_prefix)
                    .and_then(|m| m.get(&ident.ident))
                {
                    Some(function) => {
                        let function = function.clone();
                        let signature = match function.clone() {
                            InterpreterFunction::Normal(function) => function.signature,
                            InterpreterFunction::Extern(_, sig) => sig,
                        };
                        let mut args = Vec::new();
                        for _ in signature.params {
                            match stack.pop() {
                                Some(v) => args.push(v),
                                None => {
                                    todo!()
                                }
                            }
                        }
                        stack.extend(self.execute_fn(&function, args)?);
                    }
                    None => return Err(Error::FunctionNotFound(location.clone(), ident.clone())),
                }
            }
            CheckedWord::Var { location, ident } => match locals.get(ident) {
                Some(value) => {
                    stack.push(*value);
                }
                None => return Err(Error::LocalNotFound(location.clone(), ident.clone())),
            },
            CheckedWord::Set { location, ident } => match locals.get_mut(ident) {
                Some(local) => match stack.pop() {
                    Some(val) if local.ty() == val.ty() => {
                        *local = val;
                    }
                    Some(val) => {
                        return Err(Error::LocalSetTypeMismatch(
                            location.clone(),
                            val.ty().clone(),
                            local.ty().clone(),
                        ))
                    }
                    None => {
                        panic!()
                    }
                },
                None => return Err(Error::LocalNotFound(location.clone(), ident.clone())),
            },
            CheckedWord::Number { number, .. } => stack.push(Value::I32(*number)),
            CheckedWord::Intrinsic {
                intrinsic,
                location,
            } => {
                self.execute_intrinsic(intrinsic, location, stack)?;
            }
            CheckedWord::If(iff) => return self.execute_if(iff, locals, stack),
            CheckedWord::Loop(lop) => {
                self.execute_loop(lop, locals, stack)?;
            }
            CheckedWord::Break { .. } => return Ok(true),
            CheckedWord::String { addr, size } => {
                stack.push(Value::I32(*addr));
                stack.push(Value::I32(*size));
            }
        }
        Ok(false)
    }
    fn execute_intrinsic(
        &mut self,
        intrinsic: &Intrinsic,
        location: &Location,
        stack: &mut Vec<Value>,
    ) -> Result<(), Error> {
        match intrinsic {
            Intrinsic::Add => match (stack.pop(), stack.pop()) {
                (Some(Value::I32(a)), Some(Value::I32(b))) => stack.push(Value::I32(a + b)),
                (a, b) => {
                    return Err(Error::ArgsMismatch(
                        location.clone(),
                        vec![Type::I32, Type::I32],
                        vec![a.map(Value::ty), b.map(Value::ty)],
                    ))
                }
            },
            Intrinsic::Store32 => match (stack.pop(), stack.pop()) {
                (Some(Value::I32(value)), Some(Value::I32(addr))) => {
                    let addr = addr as usize;
                    let bytes = value.to_le_bytes();
                    for (i, v) in bytes.into_iter().enumerate() {
                        self.memory[addr + i] = v;
                    }
                }
                (a, b) => {
                    return Err(Error::ArgsMismatch(
                        location.clone(),
                        vec![Type::I32, Type::I32],
                        vec![a.map(Value::ty), b.map(Value::ty)],
                    ))
                }
            },
            Intrinsic::Store8 => match (stack.pop(), stack.pop()) {
                (Some(Value::I32(value)), Some(Value::I32(addr))) => {
                    let addr = addr as usize;
                    let byte = value.to_le_bytes()[0];
                    self.memory[addr] = byte;
                }
                (a, b) => {
                    return Err(Error::ArgsMismatch(
                        location.clone(),
                        vec![Type::I32, Type::I32],
                        vec![a.map(Value::ty), b.map(Value::ty)],
                    ))
                }
            },
            Intrinsic::Load32 => match stack.pop() {
                Some(Value::I32(addr)) => {
                    let addr = addr as usize;
                    let bytes: [u8; 4] = self.memory[addr..addr + 4].try_into().unwrap();
                    stack.push(Value::I32(i32::from_le_bytes(bytes)))
                }
                v => {
                    return Err(Error::ArgsMismatch(
                        location.clone(),
                        vec![Type::I32],
                        vec![v.map(Value::ty)],
                    ))
                }
            },
            Intrinsic::Load8 => match stack.pop() {
                Some(Value::I32(addr)) => {
                    let addr = addr as usize;
                    let byte = self.memory[addr];
                    stack.push(Value::I32(byte as i32))
                }
                v => {
                    return Err(Error::ArgsMismatch(
                        location.clone(),
                        vec![Type::I32],
                        vec![v.map(Value::ty)],
                    ))
                }
            },
            Intrinsic::Drop => {
                stack.pop();
            }
            Intrinsic::Sub => match (stack.pop(), stack.pop()) {
                (Some(Value::I32(a)), Some(Value::I32(b))) => stack.push(Value::I32(b - a)),
                (a, b) => {
                    return Err(Error::ArgsMismatch(
                        location.clone(),
                        vec![Type::I32, Type::I32],
                        vec![a.map(Value::ty), b.map(Value::ty)],
                    ))
                }
            },
            Intrinsic::Eq => match (stack.pop(), stack.pop()) {
                (Some(Value::I32(a)), Some(Value::I32(b))) => {
                    stack.push(if a == b { Value::True } else { Value::False })
                }
                (a, b) => {
                    return Err(Error::ArgsMismatch(
                        location.clone(),
                        vec![Type::I32, Type::I32],
                        vec![a.map(Value::ty), b.map(Value::ty)],
                    ))
                }
            },
            Intrinsic::Mod => match (stack.pop(), stack.pop()) {
                (Some(Value::I32(a)), Some(Value::I32(b))) => stack.push(Value::I32(b % a)),
                (a, b) => {
                    return Err(Error::ArgsMismatch(
                        location.clone(),
                        vec![Type::I32, Type::I32],
                        vec![a.map(Value::ty), b.map(Value::ty)],
                    ))
                }
            },
            Intrinsic::Div => match (stack.pop(), stack.pop()) {
                (Some(Value::I32(a)), Some(Value::I32(b))) => stack.push(Value::I32(b / a)),
                (a, b) => {
                    return Err(Error::ArgsMismatch(
                        location.clone(),
                        vec![Type::I32, Type::I32],
                        vec![a.map(Value::ty), b.map(Value::ty)],
                    ))
                }
            },
            Intrinsic::Stack => todo!(),
            Intrinsic::And => match (stack.pop(), stack.pop()) {
                (
                    Some(a @ Value::True | a @ Value::False),
                    Some(b @ Value::True | b @ Value::False),
                ) => {
                    let a = match a {
                        Value::True => true,
                        Value::False => false,
                        Value::I32(_) => unreachable!(),
                    };
                    let b = match b {
                        Value::True => true,
                        Value::False => false,
                        Value::I32(_) => unreachable!(),
                    };
                    stack.push(if a && b { Value::True } else { Value::False })
                }
                (a, b) => {
                    return Err(Error::ArgsMismatch(
                        location.clone(),
                        vec![Type::Bool, Type::Bool],
                        vec![a.map(Value::ty), b.map(Value::ty)],
                    ))
                }
            },
            Intrinsic::Or => todo!(),
            Intrinsic::L => todo!(),
            Intrinsic::G => todo!(),
            Intrinsic::LE => match (stack.pop(), stack.pop()) {
                (Some(Value::I32(a)), Some(Value::I32(b))) => {
                    stack.push(if b <= a { Value::True } else { Value::False })
                }
                (a, b) => {
                    return Err(Error::ArgsMismatch(
                        location.clone(),
                        vec![Type::I32, Type::I32],
                        vec![a.map(Value::ty), b.map(Value::ty)],
                    ))
                }
            },
            Intrinsic::GE => match (stack.pop(), stack.pop()) {
                (Some(Value::I32(a)), Some(Value::I32(b))) => {
                    stack.push(if b >= a { Value::True } else { Value::False })
                }
                (a, b) => {
                    return Err(Error::ArgsMismatch(
                        location.clone(),
                        vec![Type::I32, Type::I32],
                        vec![a.map(Value::ty), b.map(Value::ty)],
                    ))
                }
            },
            Intrinsic::Mul => match (stack.pop(), stack.pop()) {
                (Some(Value::I32(a)), Some(Value::I32(b))) => stack.push(Value::I32(a * b)),
                (a, b) => {
                    return Err(Error::ArgsMismatch(
                        location.clone(),
                        vec![Type::I32, Type::I32],
                        vec![a.map(Value::ty), b.map(Value::ty)],
                    ))
                }
            },
        }
        Ok(())
    }
    fn execute_loop(
        &mut self,
        lop: &CheckedLoop,
        locals: &mut HashMap<String, Value>,
        stack: &mut Vec<Value>,
    ) -> Result<(), Error> {
        'lop: loop {
            for word in &lop.body {
                if self.execute_word(word, locals, stack)? {
                    break 'lop;
                }
            }
        }
        Ok(())
    }
    fn execute_if(
        &mut self,
        iff: &CheckedIff,
        locals: &mut HashMap<String, Value>,
        stack: &mut Vec<Value>,
    ) -> Result<bool, Error> {
        match stack.pop() {
            Some(Value::True) => {
                for word in &iff.body {
                    if self.execute_word(word, locals, stack)? {
                        return Ok(true)
                    }
                }
            }
            Some(Value::False) if let Some(el) = &iff.el => {
                for word in el {
                    if self.execute_word(word, locals, stack)? {
                        return Ok(true)
                    }
                }
            }
            Some(Value::False) => {
}
            v => return Err(Error::ExpectedBool(iff.location.clone(), v.map(Value::ty))),
        }
        Ok(false)
    }
}

#[test]
fn test_1() {
    assert_eq!(
        Interpreter::new(Program {
            data: Vec::new(),
            modules: HashMap::new(),
        })
        .unwrap_err(),
        Error::NoEntryPoint
    );
}

#[test]
fn test_2() {
    use crate::ast::CheckedModule;
    use std::path::PathBuf;
    let mut modules = HashMap::new();
    modules.insert(
        PathBuf::default(),
        CheckedModule {
            externs: Default::default(),
            imports: Default::default(),
            functions: vec![CheckedFunction {
                signature: CheckedFunctionSignature {
                    location: Location::default(),
                    params: vec![],
                    ret: Vec::new(),
                    ident: String::from("main"),
                    export: Some(String::from("_start")),
                    prefix: String::from("mp"),
                },
                locals: Vec::new(),
                body: Vec::new(),
                memory: Vec::new(),
            }],
            path: Default::default(),
        },
    );
}
