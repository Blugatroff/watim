use crate::{
    ast::{CheckedFunction, CheckedIff, CheckedLoop, CheckedWord, Program, Type},
    interpreter::{execute_extern, Error, InterpreterFunction, Value},
    intrinsics::execute_intrinsic,
    scanner::Location,
};
use std::{collections::HashMap, sync::Arc};

#[derive(Clone)]
enum ScopeKind {
    Function(Arc<CheckedFunction>, i32),
    Loop(Location),
    If,
}

#[derive(Clone)]
struct Scope {
    parent: Option<Box<Scope>>,
    kind: ScopeKind,
    words: Arc<Vec<CheckedWord>>,
    locals: HashMap<String, Value>,
    stack: Vec<Value>,
    word: usize,
}

pub struct StepInterpreter {
    memory: [u8; 2usize.pow(16)],
    functions: HashMap<String, HashMap<String, InterpreterFunction>>,
    scope: Scope,
    done: bool,
    stdout: Vec<u8>,
    mem_stack: i32,
}

impl StepInterpreter {
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
        let mut mem_stack = program.data.len() as i32;

        let function = match Self::find_entry_point(functions.values().flat_map(|m| m.values())) {
            Some(InterpreterFunction::Normal(function)) => Arc::new(function.clone()),
            Some(InterpreterFunction::Extern(_, _)) => {
                todo!()
            }
            None => todo!(),
        };
        let scope = {
            let words = Arc::new(function.body.clone());
            let mut locals = HashMap::new();
            for local in &function.locals {
                locals.insert(local.ident.clone(), Value::default_of_type(&local.ty));
            }
            let mem_stack_start = mem_stack;
            for mem in &function.memory {
                let ptr = if let Some(alignment) = mem.alignment {
                    alignment - (mem_stack % alignment) + mem_stack
                } else {
                    mem_stack
                };
                mem_stack = ptr + mem.size;
                locals.insert(mem.ident.clone(), Value::Ptr(ptr, Type::I32));
            }
            Scope {
                words,
                parent: None,
                locals,
                stack: Vec::new(),
                word: 0,
                kind: ScopeKind::Function(function, mem_stack_start),
            }
        };
        Ok(Self {
            memory,
            functions,
            scope,
            done: false,
            stdout: Vec::new(),
            mem_stack,
        })
    }
    pub fn depth(&self) -> usize {
        let mut depth = 1;
        let mut current = &self.scope;
        while let Some(parent) = &current.parent {
            current = parent;
            depth += 1;
        }
        depth
    }
    pub fn call_stack(&self) -> impl IntoIterator<Item = Arc<CheckedFunction>> {
        fn inner(scope: &Scope, functions: &mut Vec<Arc<CheckedFunction>>) {
            match scope.kind.clone() {
                ScopeKind::Function(function, _) => {
                    functions.push(function);
                }
                ScopeKind::Loop(_) | ScopeKind::If => {}
            }
            match &scope.parent {
                Some(parent) => inner(parent, functions),
                None => {}
            }
        }
        let mut functions = Vec::new();
        inner(&self.scope, &mut functions);
        functions.into_iter().rev()
    }
    pub fn current_loop_location(&self) -> Option<&Location> {
        fn inner(scope: &Scope) -> Option<&Location> {
            match &scope.kind {
                ScopeKind::Loop(location) => Some(location),
                _ => scope.parent.as_deref().and_then(inner),
            }
        }
        inner(&self.scope)
    }
    pub fn locals(&self) -> &HashMap<String, Value> {
        &self.scope.locals
    }
    pub fn stack(&self) -> &[Value] {
        &self.scope.stack
    }
    pub fn stack_ptr(&self) -> i32 {
        self.mem_stack
    }
    pub fn current_word(&self) -> Option<&CheckedWord> {
        self.scope.words.get(self.scope.word)
    }
    pub fn done(&self) -> bool {
        self.done
    }
    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }
    pub fn memory(&self) -> &[u8] {
        &self.memory
    }
    fn find_entry_point<'a>(
        functions: impl IntoIterator<Item = &'a InterpreterFunction>,
    ) -> Option<&'a InterpreterFunction> {
        functions.into_iter().find(|f| match f {
            InterpreterFunction::Normal(f) => f.signature.export.as_deref() == Some("_start"),
            InterpreterFunction::Extern(_, _) => false,
        })
    }
    pub fn step(&mut self) -> Result<(), Error> {
        if self.done {
            return Ok(());
        }
        match self.scope.words.get(self.scope.word) {
            Some(word) => {
                self.scope.word += 1;
                let word = word.clone();
                self.execute_word(word)?;
                if self.scope.words.get(self.scope.word).is_none() {
                    self.step()?;
                }
                Ok(())
            }
            None => {
                if let ScopeKind::Loop(_) = &self.scope.kind {
                    self.scope.word = 0;
                    return self.step();
                } else if let Some(mut parent) = self.scope.parent.clone() {
                    if let ScopeKind::Function(_, stack) = self.scope.kind {
                        self.mem_stack = stack;
                    } else {
                        parent.locals = self.scope.locals.clone();
                    }
                    parent.stack.append(&mut self.scope.stack);
                    self.scope = *parent;
                } else {
                    self.done = true;
                }
                Ok(())
            }
        }
    }
    fn execute_word(&mut self, word: CheckedWord) -> Result<(), Error> {
        match word {
            CheckedWord::Call { ident, .. } => {
                match self
                    .functions
                    .get(&ident.module_prefix)
                    .and_then(|m| m.get(&ident.ident))
                {
                    Some(InterpreterFunction::Normal(function)) => {
                        let function = Arc::new(function.clone());
                        let words = Arc::new(function.body.clone());
                        let mut locals = HashMap::new();
                        for param in function.signature.params.iter().rev() {
                            locals.insert(param.ident.clone(), self.scope.stack.pop().unwrap());
                        }
                        for local in &function.locals {
                            locals.insert(local.ident.clone(), Value::default_of_type(&local.ty));
                        }
                        let mem_stack_start = self.mem_stack;
                        for mem in &function.memory {
                            let ptr = if let Some(alignment) = mem.alignment {
                                alignment - (self.mem_stack % alignment) + self.mem_stack
                            } else {
                                self.mem_stack
                            };
                            self.mem_stack = ptr + mem.size;
                            locals.insert(mem.ident.clone(), Value::Ptr(ptr, Type::I32));
                        }
                        self.scope = Scope {
                            words,
                            parent: Some(Box::new(self.scope.clone())),
                            locals,
                            stack: Vec::new(),
                            word: 0,
                            kind: ScopeKind::Function(function, mem_stack_start),
                        };
                        Ok(())
                    }
                    Some(InterpreterFunction::Extern((prefix, ident), _)) => {
                        let res = execute_extern(
                            prefix,
                            ident,
                            std::iter::from_fn(|| self.scope.stack.pop()),
                            &mut self.memory,
                            &mut self.stdout,
                        )?;
                        self.scope.stack.extend(res);
                        Ok(())
                    }
                    None => todo!(),
                }
            }
            CheckedWord::Var { ident, .. } => match self.scope.locals.get(&ident) {
                Some(local) => {
                    self.scope.stack.push(local.clone());
                    Ok(())
                }
                None => {
                    panic!("Local {ident} not found");
                }
            },
            CheckedWord::Set { ident, .. } => match self.scope.stack.pop() {
                Some(value) => match self.scope.locals.get_mut(&ident) {
                    Some(local) => {
                        *local = value;
                        Ok(())
                    }
                    None => todo!(),
                },
                None => todo!(),
            },
            CheckedWord::Number { number, .. } => {
                self.scope.stack.push(Value::I32(number));
                Ok(())
            }
            CheckedWord::Intrinsic {
                intrinsic,
                location,
                ..
            } => execute_intrinsic(
                &intrinsic,
                &location,
                &mut self.scope.stack,
                &mut self.memory,
            ),
            CheckedWord::If(CheckedIff { body, el, .. }) => {
                let condition = match self.scope.stack.pop() {
                    Some(Value::Bool(v)) => v,
                    _ => {
                        todo!()
                    }
                };
                if condition {
                    self.scope = Scope {
                        kind: ScopeKind::If,
                        parent: Some(Box::new(self.scope.clone())),
                        words: Arc::new(body),
                        word: 0,
                        locals: self.scope.locals.clone(),
                        stack: Vec::new(),
                    };
                } else if let Some(el) = el {
                    self.scope = Scope {
                        kind: ScopeKind::If,
                        parent: Some(Box::new(self.scope.clone())),
                        words: Arc::new(el),
                        word: 0,
                        locals: self.scope.locals.clone(),
                        stack: Vec::new(),
                    }
                }
                Ok(())
            }
            CheckedWord::Loop(CheckedLoop { body, location, .. }) => {
                self.scope = Scope {
                    parent: Some(Box::new(self.scope.clone())),
                    kind: ScopeKind::Loop(location),
                    words: Arc::new(body),
                    locals: self.scope.locals.clone(),
                    stack: Vec::new(),
                    word: 0,
                };
                Ok(())
            }
            CheckedWord::Break { .. } => {
                loop {
                    match &self.scope.kind {
                        ScopeKind::Loop(_) => {
                            let mut parent = self.scope.parent.clone().unwrap();
                            parent.locals = self.scope.locals.clone();
                            parent.stack.extend(self.scope.stack.clone());
                            self.scope = *parent;
                            break;
                        }
                        _ => {
                            let mut parent = self.scope.parent.clone().unwrap();
                            parent.locals = self.scope.locals.clone();
                            parent.stack.extend(self.scope.stack.clone());
                            self.scope = *parent;
                        }
                    }
                }
                Ok(())
            }
            CheckedWord::String { addr, size, .. } => {
                self.scope.stack.push(Value::Ptr(addr, Type::I32));
                self.scope.stack.push(Value::I32(size));
                Ok(())
            }
        }
    }
}
