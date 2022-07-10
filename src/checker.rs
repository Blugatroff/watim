use crate::{
    ast::{
        CheckedExtern, CheckedFunction, CheckedFunctionSignature, CheckedIdent, CheckedIff,
        CheckedImport, CheckedLoop, CheckedModule, CheckedWord, Function, FunctionSignature, Ident,
        Iff, Import, Intrinsic, Local, Loop, Module, Type, Word,
    },
    scanner::Location,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

pub struct ModuleChecker<'a> {
    functions: HashMap<String, FunctionSignature>,
    modules: &'a HashMap<PathBuf, (Module, String)>,
    imports: HashMap<String, CheckedImport>,
    prefix: String,
}

#[derive(Debug, Clone)]
pub enum TypeError {
    RedefinedFunction(String, Location),
    RedefinedLocal(String, Location),
    LocalNotFound(String, Location),
    FunctionNotFound(String, Location),
    TypeMismatch(Option<Type>, Word),
    IfBlocksMismatch(Location, Vec<Type>, Vec<Type>),
    BreakTypeMismatch(Location, Vec<BreakStack>),
    ReturnTypeMismatch(Location, Vec<Type>, Vec<Type>),
    ValuesLeftInLoop(Location, Vec<Type>),
    Io(Arc<std::io::Error>),
    FunctionInModuleNotFound(String, Location, PathBuf),
    ModuleNotFound(String, Location),
}

#[derive(Debug, Clone)]
pub struct BreakStack {
    #[allow(dead_code)]
    location: Location,
    stack: Vec<Type>,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Returns {
    Yes,
    No,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeError::RedefinedFunction(name, location) => f.write_fmt(format_args!(
                "{location} Redefinition of function: `{name}`"
            )),
            TypeError::RedefinedLocal(name, location) => {
                f.write_fmt(format_args!("{location} Redefinition of local: `{name}`"))
            }
            TypeError::LocalNotFound(name, location) => {
                f.write_fmt(format_args!("{location} local `{name}` not found"))
            }
            TypeError::FunctionNotFound(name, location) => {
                f.write_fmt(format_args!("{location} function `{name}` not found"))
            }
            TypeError::TypeMismatch(ty, word) => {
                f.write_fmt(format_args!("{} type mismatch {ty:?}", word.location()))
            }
            TypeError::IfBlocksMismatch(location, if_stack, else_stack) => f.write_fmt(
                format_args!("{location} if else stack mismatch {if_stack:?} <-> {else_stack:?}"),
            ),
            TypeError::BreakTypeMismatch(location, stacks) => f.write_fmt(format_args!(
                "{location} break type mismatches: {:?}",
                stacks.iter().map(|s| &s.stack).collect::<Vec<_>>()
            )),
            TypeError::ReturnTypeMismatch(location, stack, expected) => f.write_fmt(format_args!(
                "{location} return type mismatch: {:?} <-> {:?}",
                stack, expected,
            )),
            TypeError::ValuesLeftInLoop(location, stack) => f.write_fmt(format_args!(
                "{location} values left on top of stack at end of loop {stack:?}"
            )),
            TypeError::Io(e) => f.write_fmt(format_args!("{:?}", e)),
            TypeError::FunctionInModuleNotFound(ident, location, path) => {
                f.write_fmt(format_args!(
                    "{location} function `{ident}` in file {} not found",
                    path.display()
                ))
            }
            TypeError::ModuleNotFound(ident, location) => {
                f.write_fmt(format_args!("{location} module `{ident}` not found"))
            }
        }
    }
}

impl<'a> ModuleChecker<'a> {
    pub fn check(
        module: Module,
        modules: &'a HashMap<PathBuf, (Module, String)>,
        data: &mut Vec<u8>,
    ) -> Result<CheckedModule, TypeError> {
        let mut functions: HashMap<String, FunctionSignature> = HashMap::new();
        for ext in &module.externs {
            if functions
                .insert(ext.signature.ident.clone(), ext.signature.clone())
                .is_some()
            {
                return Err(TypeError::RedefinedFunction(
                    ext.signature.ident.clone(),
                    ext.signature.location.clone(),
                ));
            }
        }
        for function in &module.functions {
            if functions
                .insert(function.signature.ident.clone(), function.signature.clone())
                .is_some()
            {
                return Err(TypeError::RedefinedFunction(
                    function.signature.ident.clone(),
                    function.signature.location.clone(),
                ));
            }
        }
        let mut imports = HashMap::new();
        for import in module.imports {
            let import = Self::check_import(import, &module.path)?;
            imports.insert(import.ident.clone(), import);
        }
        let prefix = { modules.get(&module.path).unwrap().1.clone() };
        let this = Self {
            functions,
            modules,
            imports: imports.clone(),
            prefix: prefix.clone(),
        };
        let mut checked_functions = Vec::new();
        for function in module.functions {
            checked_functions.push(this.check_function(function, data)?);
        }
        let mut checked_externs = Vec::new();
        for ext in module.externs {
            checked_externs.push({
                CheckedExtern {
                    location: ext.location,
                    signature: CheckedFunctionSignature {
                        location: ext.signature.location,
                        params: ext.signature.params,
                        ret: ext.signature.ret,
                        ident: ext.signature.ident,
                        export: ext.signature.export,
                        prefix: prefix.clone(),
                    },
                    path: ext.path,
                }
            });
        }
        Ok(CheckedModule {
            externs: checked_externs,
            functions: checked_functions,
            path: module.path,
            imports,
        })
    }
    fn check_function(
        &self,
        function: Function,
        data: &mut Vec<u8>,
    ) -> Result<CheckedFunction, TypeError> {
        let mut locals = HashMap::new();
        for local in &function.locals {
            if locals.insert(local.ident.clone(), local.clone()).is_some() {
                return Err(TypeError::RedefinedLocal(
                    local.ident.clone(),
                    local.location.clone(),
                ));
            }
        }
        for mem in &function.memory {
            if locals
                .insert(
                    mem.ident.clone(),
                    Local {
                        ident: mem.ident.clone(),
                        location: mem.location.clone(),
                        ty: Type::I32,
                    },
                )
                .is_some()
            {
                return Err(TypeError::RedefinedLocal(
                    mem.ident.clone(),
                    mem.location.clone(),
                ));
            }
        }
        for param in &function.signature.params {
            if locals
                .insert(
                    param.ident.clone(),
                    Local {
                        ident: param.ident.clone(),
                        location: param.location.clone(),
                        ty: param.ty.clone(),
                    },
                )
                .is_some()
            {
                return Err(TypeError::RedefinedLocal(
                    param.ident.clone(),
                    param.location.clone(),
                ));
            }
        }
        let mut stack: Vec<Type> = Vec::new();
        let mut checked_words = Vec::new();
        for word in function.body {
            let (_, _, checked_word) = self.check_word(word, &mut stack, &locals, data)?;
            checked_words.push(checked_word);
        }
        if stack != function.signature.ret {
            return Err(TypeError::ReturnTypeMismatch(
                function.signature.location.clone(),
                stack,
                function.signature.ret,
            ));
        }
        Ok(CheckedFunction {
            signature: CheckedFunctionSignature {
                location: function.signature.location,
                params: function.signature.params,
                ret: function.signature.ret,
                ident: function.signature.ident,
                export: function.signature.export,
                prefix: self.prefix.clone(),
            },
            locals: function.locals,
            body: checked_words,
            memory: function.memory,
        })
    }
    pub fn check_import(import: Import, path: &Path) -> Result<CheckedImport, TypeError> {
        let path = path.parent().unwrap().join(import.path);
        let path = match path.canonicalize() {
            Ok(path) => path,
            Err(e) => return Err(TypeError::Io(Arc::new(e))),
        };
        Ok(CheckedImport {
            path,
            ident: import.ident,
        })
    }
    fn check_word(
        &self,
        word: Word,
        stack: &mut Vec<Type>,
        locals: &HashMap<String, Local>,
        data: &mut Vec<u8>,
    ) -> Result<(Returns, Vec<BreakStack>, CheckedWord), TypeError> {
        match word {
            Word::Call { location, ident } => {
                let (function, checked_ident) = match ident.clone() {
                    Ident::Direct(ident) => match self.functions.get(&ident) {
                        Some(function) => (
                            function,
                            CheckedIdent {
                                ident,
                                module_prefix: self.prefix.clone(),
                            },
                        ),
                        None => return Err(TypeError::FunctionNotFound(ident.clone(), location)),
                    },
                    Ident::Qualified(qualifier, ident) => match self.imports.get(&qualifier) {
                        Some(import) => match self.modules.get(&import.path) {
                            Some((module, module_prefix)) => {
                                match module.functions.iter().find(|f| f.signature.ident == ident) {
                                    Some(function) => (
                                        &function.signature,
                                        CheckedIdent {
                                            module_prefix: module_prefix.clone(),
                                            ident,
                                        },
                                    ),
                                    None => {
                                        return Err(TypeError::FunctionInModuleNotFound(
                                            ident.clone(),
                                            location,
                                            import.path.clone(),
                                        ))
                                    }
                                }
                            }
                            None => {
                                panic!("all imported modules should be present")
                            }
                        },
                        None => return Err(TypeError::ModuleNotFound(qualifier.clone(), location)),
                    },
                };
                for param in function.params.iter().rev() {
                    match stack.pop() {
                        Some(ty) if ty == param.ty => {}
                        Some(ty) => {
                            return Err(TypeError::TypeMismatch(
                                Some(ty),
                                Word::Call { location, ident },
                            ))
                        }
                        None => {
                            return Err(TypeError::TypeMismatch(
                                None,
                                Word::Call { location, ident },
                            ))
                        }
                    }
                }
                stack.extend(function.ret.clone());
                Ok((
                    Returns::Yes,
                    Vec::new(),
                    CheckedWord::Call {
                        location,
                        ident: checked_ident,
                    },
                ))
            }
            Word::Var { location, ident } => {
                if let Some(local) = locals.get(&ident) {
                    stack.push(local.ty.clone());
                } else {
                    return Err(TypeError::LocalNotFound(ident.clone(), location));
                }
                Ok((
                    Returns::Yes,
                    Vec::new(),
                    CheckedWord::Var { location, ident },
                ))
            }
            Word::Set { ident, location } => {
                if let Some(local) = locals.get(&ident) {
                    if let Some(ty) = stack.last() {
                        if ty != &local.ty {
                            return Err(TypeError::TypeMismatch(
                                Some(ty.clone()),
                                Word::Set { location, ident },
                            ));
                        }
                        stack.pop();
                    } else {
                        return Err(TypeError::TypeMismatch(None, Word::Set { location, ident }));
                    }
                } else {
                    return Err(TypeError::LocalNotFound(ident.clone(), location));
                }
                Ok((
                    Returns::Yes,
                    Vec::new(),
                    CheckedWord::Set { ident, location },
                ))
            }
            Word::Number { number, location } => {
                stack.push(Type::I32);
                Ok((
                    Returns::Yes,
                    Vec::new(),
                    CheckedWord::Number { number, location },
                ))
            }
            Word::Intrinsic {
                ref intrinsic,
                ref location,
            } => match intrinsic {
                Intrinsic::Add => {
                    self.expect_stack(stack, &word, [Type::I32, Type::I32])?;
                    stack.push(Type::I32);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            intrinsic: intrinsic.clone(),
                            location: location.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::Store32 => {
                    self.expect_stack(stack, &word, [Type::I32, Type::I32])?;
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::Store8 => {
                    self.expect_stack(stack, &word, [Type::I32, Type::I32])?;
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::Load32 => {
                    self.expect_stack(stack, &word, [Type::I32])?;
                    stack.push(Type::I32);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::Load8 => {
                    self.expect_stack(stack, &word, [Type::I32])?;
                    stack.push(Type::I32);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::Drop => {
                    self.expect_stack(stack, &word, [Type::I32])?;
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::Sub => {
                    self.expect_stack(stack, &word, [Type::I32, Type::I32])?;
                    stack.push(Type::I32);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::Eq | intrinsic @ Intrinsic::NotEq => {
                    self.expect_stack(stack, &word, [Type::I32, Type::I32])?;
                    stack.push(Type::Bool);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::Mod => {
                    self.expect_stack(stack, &word, [Type::I32, Type::I32])?;
                    stack.push(Type::I32);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::Div => {
                    self.expect_stack(stack, &word, [Type::I32, Type::I32])?;
                    stack.push(Type::I32);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::Stack => {
                    stack.push(Type::I32);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::And => {
                    self.expect_stack(stack, &word, [Type::Bool, Type::Bool])?;
                    stack.push(Type::Bool);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::Or => {
                    self.expect_stack(stack, &word, [Type::Bool, Type::Bool])?;
                    stack.push(Type::Bool);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::L => {
                    self.expect_stack(stack, &word, [Type::I32, Type::I32])?;
                    stack.push(Type::Bool);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::G => {
                    self.expect_stack(stack, &word, [Type::I32, Type::I32])?;
                    stack.push(Type::Bool);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::LE => {
                    self.expect_stack(stack, &word, [Type::I32, Type::I32])?;
                    stack.push(Type::Bool);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::GE => {
                    self.expect_stack(stack, &word, [Type::I32, Type::I32])?;
                    stack.push(Type::Bool);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
                intrinsic @ Intrinsic::Mul => {
                    self.expect_stack(stack, &word, [Type::I32, Type::I32])?;
                    stack.push(Type::I32);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: intrinsic.clone(),
                        },
                    ))
                }
            },
            ref word @ Word::If(Iff {
                ref location,
                ref body,
                ref el,
            }) => {
                match stack.pop() {
                    Some(Type::Bool) => {}
                    ty => return Err(TypeError::TypeMismatch(ty, word.clone())),
                }
                let mut if_block_termination = Returns::Yes;
                let mut break_stacks = Vec::new();
                let mut if_block_stack = stack.clone();
                let mut if_block_checked_words = Vec::new();
                for word in body {
                    let (t, break_stack, checked_word) =
                        self.check_word(word.clone(), &mut if_block_stack, locals, data)?;
                    if t == Returns::No {
                        if_block_termination = Returns::No
                    }
                    if_block_checked_words.push(checked_word);
                    break_stacks.extend(break_stack);
                }
                let mut else_block_termination = Returns::Yes;
                let mut else_block_stack = stack.clone();
                let else_block_checked_words = match el {
                    Some(block) => {
                        let mut else_block_checked_words = Vec::new();
                        for word in block {
                            let (t, break_stack, checked_word) =
                                self.check_word(word.clone(), &mut else_block_stack, locals, data)?;
                            else_block_checked_words.push(checked_word);
                            if t == Returns::No {
                                else_block_termination = Returns::No;
                            }
                            break_stacks.extend(break_stack);
                        }
                        Some(else_block_checked_words)
                    }
                    None => None,
                };
                match (if_block_termination, else_block_termination) {
                    (Returns::Yes, Returns::Yes) => {
                        if if_block_stack != else_block_stack {
                            return Err(TypeError::IfBlocksMismatch(
                                location.clone(),
                                if_block_stack,
                                else_block_stack,
                            ));
                        }
                        let ret = if if_block_stack.len() > stack.len() {
                            if_block_stack[stack.len()..if_block_stack.len()].to_vec()
                        } else {
                            Vec::new()
                        };
                        *stack = if_block_stack;
                        Ok((
                            Returns::Yes,
                            break_stacks,
                            CheckedWord::If(CheckedIff {
                                location: location.clone(),
                                body: if_block_checked_words,
                                el: else_block_checked_words,
                                ret,
                            }),
                        ))
                    }
                    (Returns::Yes, Returns::No) => {
                        let ret = if if_block_stack.len() > stack.len() {
                            if_block_stack[stack.len()..if_block_stack.len()].to_vec()
                        } else {
                            Vec::new()
                        };
                        *stack = if_block_stack;
                        Ok((
                            Returns::Yes,
                            break_stacks,
                            CheckedWord::If(CheckedIff {
                                location: location.clone(),
                                body: if_block_checked_words,
                                el: else_block_checked_words,
                                ret,
                            }),
                        ))
                    }
                    (Returns::No, Returns::Yes) => {
                        let ret = if else_block_stack.len() > stack.len() {
                            else_block_stack[stack.len()..else_block_stack.len()].to_vec()
                        } else {
                            Vec::new()
                        };
                        *stack = else_block_stack;
                        Ok((
                            Returns::Yes,
                            break_stacks,
                            CheckedWord::If(CheckedIff {
                                location: location.clone(),
                                body: if_block_checked_words,
                                el: else_block_checked_words,
                                ret,
                            }),
                        ))
                    }
                    (Returns::No, Returns::No) => Ok((
                        Returns::No,
                        break_stacks,
                        CheckedWord::If(CheckedIff {
                            location: location.clone(),
                            body: if_block_checked_words,
                            el: else_block_checked_words,
                            ret: Vec::new(),
                        }),
                    )),
                }
            }
            Word::Loop(Loop { location, body }) => {
                let mut loop_stack = Vec::new();
                let mut break_stacks = Vec::new();
                let mut checked_words = Vec::new();
                for word in body {
                    let (_, break_stack, checked_word) =
                        self.check_word(word, &mut loop_stack, locals, data)?;
                    checked_words.push(checked_word);
                    break_stacks.extend(break_stack);
                }
                if !loop_stack.is_empty() {
                    return Err(TypeError::ValuesLeftInLoop(location, loop_stack));
                }
                if let Some(first) = break_stacks.last() {
                    for stack in &break_stacks {
                        if stack.stack != first.stack {
                            return Err(TypeError::BreakTypeMismatch(location, break_stacks));
                        }
                    }
                    let ret = break_stacks
                        .first()
                        .map(|s| s.stack.clone())
                        .unwrap_or_default();
                    stack.extend(ret.clone());
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Loop(CheckedLoop {
                            location,
                            body: checked_words,
                            ret,
                        }),
                    ))
                } else {
                    Ok((
                        Returns::No,
                        Vec::new(),
                        CheckedWord::Loop(CheckedLoop {
                            location,
                            body: checked_words,
                            ret: Vec::new(),
                        }),
                    ))
                }
            }
            Word::Break { location } => Ok((
                Returns::No,
                vec![BreakStack {
                    location: location.clone(),
                    stack: stack.clone(),
                }],
                CheckedWord::Break { location },
            )),
            Word::String { value, location } => {
                let addr = data.len() as i32;
                let size = value.as_bytes().len() as i32;
                data.extend(value.as_bytes());
                stack.push(Type::I32);
                stack.push(Type::I32);
                Ok((
                    Returns::Yes,
                    Vec::new(),
                    CheckedWord::String {
                        addr,
                        size,
                        location,
                    },
                ))
            }
        }
    }
    fn expect_stack<const L: usize>(
        &self,
        stack: &mut Vec<Type>,
        word: &Word,
        expected: [Type; L],
    ) -> Result<(), TypeError> {
        for expected_ty in expected.into_iter().rev() {
            match stack.pop() {
                Some(ty) if ty == expected_ty => {}
                Some(ty) => return Err(TypeError::TypeMismatch(Some(ty), word.clone())),
                None => return Err(TypeError::TypeMismatch(None, word.clone())),
            }
        }
        Ok(())
    }
}
