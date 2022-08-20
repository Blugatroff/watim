use crate::{
    align_to,
    ast::{
        CheckedFunction, CheckedFunctionSignature, CheckedIdent, CheckedIff, CheckedIntrinsic,
        CheckedLoop, CheckedWord, Local, Memory, Param, Program, ResolvedType,
    },
    intrinsics::execute_intrinsic,
    scanner::Location,
};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Bool(bool),
    I32(i32),
    I64(i64),
    Ptr(i32, ResolvedType),
}

impl Value {
    pub fn default_of_type(ty: &ResolvedType) -> Self {
        match ty {
            ResolvedType::I32 => Self::I32(0),
            ResolvedType::I64 => Self::I64(0),
            ResolvedType::Bool => Self::Bool(false),
            ResolvedType::Ptr(ty) => Self::Ptr(0, (**ty).clone()),
            ResolvedType::AnyPtr => Self::Ptr(0, ResolvedType::AnyPtr),
            ResolvedType::Custom(_) => todo!(),
        }
    }
    pub fn ty(&self) -> ResolvedType {
        match self {
            Self::Bool(_) => ResolvedType::Bool,
            Self::I32(_) => ResolvedType::I32,
            Self::I64(_) => ResolvedType::I64,
            Self::Ptr(_, ty) => ResolvedType::Ptr(Box::new(ty.clone())),
        }
    }
}

impl TryFrom<Value> for i32 {
    type Error = &'static str;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::I32(v) => Ok(v),
            _ => Err("tried to unwrap non-I32 Value into i32"),
        }
    }
}

impl TryFrom<Value> for usize {
    type Error = &'static str;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Ptr(v, _) => Ok(v as usize),
            _ => Err("tried to unwrap non-Ptr Value into usize"),
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Bool(true) => f.write_str("True"),
            Value::Bool(false) => f.write_str("False"),
            Value::I32(n) => n.fmt(f),
            Value::I64(n) => n.fmt(f),
            Value::Ptr(ptr, ty) => f.write_fmt(format_args!(".{ty}={ptr}")),
        }
    }
}

#[derive(Debug, Clone)]
pub enum InterpreterFunction {
    Normal(CheckedFunction),
    Extern((String, String), CheckedFunctionSignature),
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum Error {
    #[error("{0}: function `{1}` not found!")]
    FunctionNotFound(Location, CheckedIdent),
    #[error("No entry point found! Export a function as '_start' to make it the entry point.")]
    NoEntryPoint,
    #[error("{0}: local `{1}` not found!")]
    LocalNotFound(Location, String),
    #[error("{0}: expected {1:?} but found {2:?}")]
    ArgsMismatch(Location, Vec<Vec<ResolvedType>>, Vec<Option<ResolvedType>>),
}

pub struct Interpreter<'stdin, 'stdout> {
    functions: HashMap<String, HashMap<String, InterpreterFunction>>,
    globals: HashMap<String, HashMap<String, (i32, ResolvedType)>>,
    mem_stack: i32,
    stdout: Box<dyn std::io::Write + 'stdin>,
    stdin: Box<dyn std::io::Read + 'stdout>,
    memory: [u8; 2usize.pow(16)],
}

impl<'stdin, 'stdout> Interpreter<'stdin, 'stdout> {
    pub fn interpret_program(
        program: Program,
        stdin: impl std::io::Read + 'stdout,
        stdout: impl std::io::Write + 'stdin,
    ) -> Result<(), Error> {
        let mut memory = [0u8; 2usize.pow(16)];
        for (i, b) in program.data.iter().enumerate() {
            memory[i] = *b;
        }

        let mut globals: HashMap<String, HashMap<String, (i32, ResolvedType)>> = HashMap::new();
        let mut global_mem_addr = program.data.len() as i32;
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
            for mem in module.globals {
                if let Some(alignment) = mem.alignment {
                    global_mem_addr = align_to(global_mem_addr, alignment);
                }
                globals
                    .entry(mem.ident.module_prefix.clone())
                    .or_default()
                    .insert(mem.ident.ident.clone(), (global_mem_addr, mem.ty.clone()));
                global_mem_addr += mem.size;
            }
        }
        let mem_stack = global_mem_addr;
        let mut this = Self {
            functions,
            mem_stack,
            stdout: Box::new(stdout),
            stdin: Box::new(stdin),
            memory,
            globals,
        };
        let entry = match this.find_entry_point() {
            Some(entry) => entry.clone(),
            None => return Ok(()),
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
                let mut locals: HashMap<String, Value> =
                    function
                        .locals
                        .iter()
                        .map(|local: &Local<ResolvedType>| {
                            (local.ident.clone(), Value::default_of_type(&local.ty))
                        })
                        .chain(function.signature.params.iter().rev().map(
                            |p: &Param<ResolvedType>| {
                                let v = args.next().unwrap();
                                (p.ident.clone(), v)
                            },
                        ))
                        .chain(function.memory.iter().map(|mem: &Memory<ResolvedType>| {
                            let ptr = if let Some(alignment) = mem.alignment {
                                alignment - (self.mem_stack % alignment) + self.mem_stack
                            } else {
                                self.mem_stack
                            };
                            let value = Value::Ptr(ptr, mem.ty.clone());
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
            InterpreterFunction::Extern((path_0, path_1), ..) => execute_extern(
                path_0,
                path_1,
                args,
                &mut self.memory,
                &mut self.stdin,
                &mut self.stdout,
            ),
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
            CheckedWord::Local { location, ident } => match locals.get(ident) {
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
                    val => {
                        return Err(Error::ArgsMismatch(
                            location.clone(),
                            vec![vec![local.ty()]],
                            vec![val.as_ref().map(Value::ty)],
                        ))
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
                stack.push(Value::Ptr(*addr, ResolvedType::I32));
                stack.push(Value::I32(*size));
            }
            CheckedWord::FieldDeref { offset, ty, .. } => match stack.pop().unwrap() {
                Value::Ptr(pt, _) => stack.push(Value::Ptr(pt + *offset as i32, ty.clone())),
                _ => {
                    todo!()
                }
            },
            CheckedWord::Global { ident, .. } => {
                match self
                    .globals
                    .get(&ident.module_prefix)
                    .and_then(|m| m.get(&ident.ident))
                {
                    Some((value, ty)) => {
                        stack.push(Value::Ptr(*value, ty.clone()))
                    }
                    None => todo!(),
                }
            }
        }
        Ok(false)
    }
    fn execute_intrinsic(
        &mut self,
        intrinsic: &CheckedIntrinsic,
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
                        return Ok(true);
                    }
                }
            }
            Some(Value::Bool(false)) => {
                if let Some(el) = &iff.el {
                    for word in el {
                        if self.execute_word(word, locals, stack)? {
                            return Ok(true);
                        }
                    }
                }
            }
            v => {
                return Err(Error::ArgsMismatch(
                    iff.location.clone(),
                    vec![vec![ResolvedType::Bool]],
                    vec![v.as_ref().map(Value::ty)],
                ))
            }
        }
        Ok(false)
    }
}

pub fn execute_extern(
    path_0: &str,
    path_1: &str,
    mut args: impl Iterator<Item = Value>,
    memory: &mut [u8],
    mut stdin: impl std::io::Read,
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
                    let written: [u8; 4] = written.to_le_bytes();
                    for (i, b) in written.into_iter().enumerate() {
                        memory[nwritten + i] = b;
                    }
                    let res = Value::I32(0);
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
                        read += stdin.read(&mut memory[ptr..ptr + len]).unwrap() as i32;
                    }
                    let written: [u8; 4] = read.to_le_bytes();
                    for (i, b) in written.into_iter().enumerate() {
                        memory[nwritten + i] = b;
                    }
                    Ok(vec![Value::I32(0)])
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
