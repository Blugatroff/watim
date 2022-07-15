use crate::{
    ast::{
        CheckedFunction, CheckedFunctionSignature, CheckedIdent, CheckedIff, CheckedLoop,
        CheckedWord, Intrinsic, Local, Memory, Param, Program, Type,
    },
    intrinsics::execute_intrinsic,
    scanner::Location,
};
use std::{collections::HashMap, io::Read};

#[derive(Debug)]
pub struct Interpreter {
    memory: [u8; 2usize.pow(16)],
    functions: HashMap<String, HashMap<String, InterpreterFunction>>,
    mem_stack: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Bool(bool),
    I32(i32),
    Ptr(i32, Type),
}

impl Value {
    pub fn default_of_type(ty: &Type) -> Self {
        match ty {
            Type::I32 => Self::I32(0),
            Type::Bool => Self::Bool(false),
            Type::Ptr(ty) => Self::Ptr(0, (**ty).clone()),
            Type::AnyPtr => Self::Ptr(0, Type::AnyPtr),
        }
    }
    pub fn ty(&self) -> Type {
        match self {
            Self::Bool(_) => Type::Bool,
            Self::I32(_) => Type::I32,
            Self::Ptr(_, ty) => Type::Ptr(Box::new(ty.clone())),
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Bool(true) => f.write_str("True"),
            Value::Bool(false) => f.write_str("False"),
            Value::I32(n) => n.fmt(f),
            Value::Ptr(ptr, ty) => f.write_fmt(format_args!(".{ty}={ptr}")),
        }
    }
}

#[derive(Debug, Clone)]
pub enum InterpreterFunction {
    Normal(CheckedFunction),
    Extern((String, String), CheckedFunctionSignature),
}

#[derive(Debug, Clone)]
pub enum Error {
    FunctionNotFound(Location, CheckedIdent),
    NoEntryPoint,
    LocalNotFound(Location, String),
    LocalSetTypeMismatch(Location, Type, Type),
    ArgsMismatch(Location, Vec<Vec<Type>>, Vec<Option<Type>>),
    ExpectedBool(Location, Option<Type>),
}

impl Interpreter {
    pub fn interpret_program(program: Program) -> Result<(), Error> {
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
        };
        this.execute_fn(&entry, std::iter::empty())?;
        Ok(())
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
        mut args: impl Iterator<Item = Value>,
    ) -> Result<Vec<Value>, Error> {
        match function {
            InterpreterFunction::Normal(function) => {
                let ostack = self.mem_stack;
                let mut locals: HashMap<String, Value> = function
                    .locals
                    .iter()
                    .map(|local: &Local| (local.ident.clone(), Value::default_of_type(&local.ty)))
                    .chain(function.signature.params.iter().rev().map(|p: &Param| {
                        let v = args.next().unwrap();
                        (p.ident.clone(), v)
                    }))
                    .chain(function.memory.iter().map(|mem: &Memory| {
                        let ptr = if let Some(alignment) = mem.alignment {
                            alignment - (self.mem_stack % alignment) + self.mem_stack
                        } else {
                            self.mem_stack
                        };
                        let value = Value::Ptr(ptr, Type::I32);
                        self.mem_stack = ptr + mem.size;
                        (mem.ident.clone(), value)
                    }))
                    .collect();
                let mut stack: Vec<Value> = Vec::new();
                for word in &function.body {
                    self.execute_word(word, &mut locals, &mut stack)?;
                }
                self.mem_stack = ostack;
                Ok(stack)
            }
            InterpreterFunction::Extern((path_0, path_1), ..) => {
                execute_extern(path_0, path_1, args, &mut self.memory, std::io::stdout())
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
                        let res = self.execute_fn(&function, std::iter::from_fn(|| stack.pop()))?;
                        stack.extend(res);
                    }
                    None => return Err(Error::FunctionNotFound(location.clone(), ident.clone())),
                }
            }
            CheckedWord::Var { location, ident } => match locals.get(ident) {
                Some(value) => {
                    stack.push(value.clone());
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
                            val.ty(),
                            local.ty(),
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
            CheckedWord::String { addr, size, .. } => {
                stack.push(Value::Ptr(*addr, Type::I32));
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
        execute_intrinsic(intrinsic, location, stack, &mut self.memory)
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
            Some(Value::Bool(true)) => {
                for word in &iff.body {
                    if self.execute_word(word, locals, stack)? {
                        return Ok(true)
                    }
                }
            }
            Some(Value::Bool(false)) if let Some(el) = &iff.el => {
                for word in el {
                    if self.execute_word(word, locals, stack)? {
                        return Ok(true)
                    }
                }
            }
            Some(Value::Bool(false)) => {}
            v => return Err(Error::ExpectedBool(iff.location.clone(), v.as_ref().map(Value::ty))),
        }
        Ok(false)
    }
}

pub fn execute_extern(
    path_0: &str,
    path_1: &str,
    mut args: impl Iterator<Item = Value>,
    memory: &mut [u8],
    mut stdout: impl std::io::Write,
) -> Result<Vec<Value>, Error> {
    match (path_0, path_1) {
        ("wasi_unstable", "fd_write") => {
            let nwritten = match args.next().unwrap() {
                Value::Ptr(nwritten, _) => nwritten,
                _ => todo!(),
            };
            let len = match args.next().unwrap() {
                Value::I32(len) => len,
                _ => {
                    todo!()
                }
            };
            let iovec_ptr = match args.next().unwrap() {
                Value::Ptr(ptr, _) => ptr,
                _ => todo!(),
            };
            let file = match args.next().unwrap() {
                Value::I32(file) => file,
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
                        let ptr: [u8; 4] =
                            memory[iovec_ptr + i..iovec_ptr + 4 + i].try_into().unwrap();
                        let ptr: i32 = i32::from_le_bytes(ptr);
                        let ptr = ptr as usize;

                        let len: [u8; 4] = memory[iovec_ptr + 4..iovec_ptr + 8].try_into().unwrap();
                        let len: i32 = i32::from_le_bytes(len);
                        written += len;
                        let len = len as usize;
                        let data = &memory[ptr..ptr + len];
                        stdout.write_all(data).unwrap();
                        stdout.flush().unwrap();
                    }
                    let res = Value::I32(written);
                    let written: [u8; 4] = written.to_le_bytes();
                    for (i, b) in written.into_iter().enumerate() {
                        memory[nwritten + i] = b;
                    }
                    Ok(vec![res])
                }
                file => {
                    todo!("unhandled fd_write for file: {file}")
                }
            }
        }
        ("wasi_unstable", "fd_read") => {
            let nwritten = match args.next().unwrap() {
                Value::Ptr(nwritten, _) => nwritten,
                _ => todo!(),
            };
            let len = match args.next().unwrap() {
                Value::I32(len) => len,
                _ => {
                    todo!()
                }
            };
            let iovec_ptr = match args.next().unwrap() {
                Value::Ptr(ptr, _) => ptr,
                _ => todo!(),
            };
            let file = match args.next().unwrap() {
                Value::I32(file) => file,
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
                        let ptr: [u8; 4] =
                            memory[iovec_ptr + i..iovec_ptr + 4 + i].try_into().unwrap();
                        let ptr: i32 = i32::from_le_bytes(ptr);
                        let ptr = ptr as usize;

                        let len: [u8; 4] = memory[iovec_ptr + 4 + i..iovec_ptr + 8 + i]
                            .try_into()
                            .unwrap();
                        let len = i32::from_le_bytes(len) as usize;
                        read += std::io::stdin().read(&mut memory[ptr..ptr + len]).unwrap() as i32;
                    }
                    let written: [u8; 4] = read.to_le_bytes();
                    for (i, b) in written.into_iter().enumerate() {
                        memory[nwritten + i] = b;
                    }
                    Ok(vec![Value::I32(read)])
                }
                file => {
                    todo!("unhandled fd_write for file: {file}")
                }
            }
        }
        ("wasi_unstable", "proc_exit") => {
            let code = args.next().unwrap();
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
