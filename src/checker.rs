use crate::{
    ast::{
        CheckedExtern, CheckedFunction, CheckedFunctionSignature, CheckedIdent, CheckedIff,
        CheckedImport, CheckedIntrinsic, CheckedLoop, CheckedModule, CheckedWord, Function,
        FunctionSignature, Ident, Iff, Import, Intrinsic, Local, Loop, Memory, Module,
        ResolvedType, Word,
    },
    scanner::Location,
};
use itertools::Itertools;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;

fn display_maybe_list<T: std::fmt::Display>(items: &[T], sep: &'static str) -> String {
    match items.first() {
        Some(v) if items.len() == 1 => format!("{v}"),
        _ => {
            let a = Itertools::intersperse(items.iter().map(|t| format!("{t}")), String::from(sep))
                .fold(String::new(), |a, b| a + &b);
            format!("[{a}]")
        }
    }
}

#[derive(Error, Debug, Clone)]
pub enum TypeError {
    RedefinedFunction(String, Location),
    RedefinedLocal(String, Location),
    LocalNotFound(String, Location),
    FunctionNotFound(String, Location),
    ArgsMismatch(
        Word<ResolvedType>,
        Vec<Vec<ResolvedType>>,
        Vec<Option<ResolvedType>>,
    ),
    IfBlocksMismatch(Location, Vec<ResolvedType>, Vec<ResolvedType>),
    BreakTypeMismatch(Location, Vec<BreakStack>),
    ReturnTypeMismatch(Location, Vec<ResolvedType>, Vec<ResolvedType>),
    ValuesLeftInLoop(Location, Vec<ResolvedType>),
    Io(#[from] Arc<std::io::Error>),
    FunctionInModuleNotFound(String, Location, PathBuf),
    ModuleNotFound(String, Location),
    StructNotFound(String, Location),
    StoreArgs(Location, Option<ResolvedType>, Option<ResolvedType>),
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeError::RedefinedFunction(fun, loc) => {
                write!(f, "{loc}: Redefinition of function `{fun}`!")
            }
            TypeError::RedefinedLocal(local, loc) => {
                write!(f, "{loc}: Redefinition of local `{local}`!")
            }
            TypeError::LocalNotFound(local, loc) => {
                write!(f, "{loc}: local `{local}` not found!")
            }
            TypeError::FunctionNotFound(function, loc) => {
                write!(f, "{loc}: function `{function}` not found!")
            }
            TypeError::StructNotFound(struc, loc) => {
                write!(f, "{loc}: struct `{struc}` not found!")
            }
            TypeError::ArgsMismatch(word, expected, got) => {
                let expected: Vec<_> = expected
                    .iter()
                    .map(|t| display_maybe_list(t, ", "))
                    .collect();
                let got = display_maybe_list(
                    &got.iter()
                        .map(|t| match t {
                            Some(t) => format!("{t}"),
                            None => String::from("_"),
                        })
                        .collect::<Vec<_>>(),
                    ", ",
                );
                let expected = display_maybe_list(&expected, ", ");
                write!(f, "{}: expected: {expected}, got: {got}!", word.location())
            }
            TypeError::IfBlocksMismatch(loc, iff, el) => {
                let iff = display_maybe_list(iff, ", ");
                let el = display_maybe_list(el, ", ");
                write!(f, "{loc}: {iff} {el}!")
            }
            TypeError::BreakTypeMismatch(loc, stack) => {
                let stack = stack
                    .iter()
                    .map(|s| display_maybe_list(&s.stack, ", "))
                    .collect::<Vec<_>>();
                let stack = display_maybe_list(&stack, ", ");
                write!(f, "{loc}: break types mismatch {stack}!")
            }
            TypeError::ReturnTypeMismatch(loc, expected, got) => {
                let expected = display_maybe_list(expected, ", ");
                let got = display_maybe_list(got, ", ");
                write!(
                    f,
                    "{loc}: return types mismatch expected: {expected}, got: {got}!"
                )
            }
            TypeError::ValuesLeftInLoop(loc, vals) => {
                let vals = display_maybe_list(vals, ", ");
                write!(f, "{loc}: values left in loop: {vals}!")
            }
            TypeError::Io(e) => e.fmt(f),
            TypeError::FunctionInModuleNotFound(fun, loc, path) => {
                write!(f, "{loc}: function `{fun}` not found in module `{path:?}`!")
            }
            TypeError::ModuleNotFound(mo, loc) => {
                write!(f, "{loc}: module `{mo}` not found!")
            }
            TypeError::StoreArgs(loc, a, b) => {
                let items: Vec<&ResolvedType> = [a, b].into_iter().flatten().collect();
                write!(
                    f,
                    "{loc}: expected [.T, T], got: {}",
                    display_maybe_list(&items, ", ")
                )
            }
        }
    }
}

pub struct ModuleChecker<'a, Type> {
    functions: HashMap<String, FunctionSignature<Type>>,
    modules: &'a HashMap<PathBuf, (Module<Type>, String)>,
    imports: HashMap<String, CheckedImport>,
    prefix: String,
    globals: HashMap<String, Local<Type>>,
}

#[derive(Debug, Clone)]
pub struct BreakStack {
    #[allow(dead_code)]
    pub location: Location,
    pub stack: Vec<ResolvedType>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Returns {
    Yes,
    No,
}

type Signature<'a, T, const L: usize> = ([ResolvedType; L], &'a dyn Fn([ResolvedType; L]) -> T);

impl<'a> ModuleChecker<'a, ResolvedType> {
    pub fn check(
        module: Module<ResolvedType>,
        modules: &'a HashMap<PathBuf, (Module<ResolvedType>, String)>,
        data: &mut Vec<u8>,
    ) -> Result<CheckedModule, TypeError> {
        let mut functions: HashMap<String, FunctionSignature<ResolvedType>> = HashMap::new();
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
        let mut globals = HashMap::new();
        for global in &module.memory {
            globals.insert(
                global.ident.clone(),
                Local {
                    ident: global.ident.clone(),
                    location: global.location.clone(),
                    ty: global.ty.clone(),
                },
            );
        }
        let prefix = { modules.get(&module.path).unwrap().1.clone() };
        let this = Self {
            functions,
            modules,
            imports: imports.clone(),
            prefix: prefix.clone(),
            globals,
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
        let mut globals = Vec::new();
        for global in module.memory {
            globals.push(Memory {
                ident: CheckedIdent {
                    ident: global.ident.clone(),
                    module_prefix: prefix.clone(),
                },
                location: global.location.clone(),
                size: global.size,
                alignment: global.alignment,
                ty: global.ty,
            });
        }
        Ok(CheckedModule {
            externs: checked_externs,
            functions: checked_functions,
            path: module.path,
            imports,
            globals,
        })
    }
    fn check_function(
        &self,
        function: Function<ResolvedType>,
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
                        ty: ResolvedType::Ptr(Box::new(mem.ty.clone())),
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
        let mut stack: Vec<ResolvedType> = Vec::new();
        let mut checked_words = Vec::new();
        let mut returns: Returns = Returns::Yes;
        for word in function.body {
            let (returns_, _, checked_word) =
                self.check_word(word, &mut stack, &locals, &self.globals, data)?;
            returns = if returns_ == Returns::No {
                Returns::No
            } else {
                returns
            };
            checked_words.push(checked_word);
        }

        if returns == Returns::Yes && stack != function.signature.ret {
            return Err(TypeError::ReturnTypeMismatch(
                function.signature.location.clone(),
                function.signature.ret,
                stack,
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
            returns,
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
        word: Word<ResolvedType>,
        stack: &mut Vec<ResolvedType>,
        locals: &HashMap<String, Local<ResolvedType>>,
        globals: &HashMap<String, Local<ResolvedType>>,
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
                let mut popped = Vec::new();
                for param in function.params.iter().rev() {
                    match stack.pop() {
                        Some(ty) if ty == param.ty => {
                            popped.push(Some(ty));
                        }
                        ty => {
                            popped.push(ty);
                            return Err(TypeError::ArgsMismatch(
                                Word::Call { location, ident },
                                vec![function.params.iter().map(|p| p.ty.clone()).collect()],
                                popped,
                            ));
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
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Local { location, ident },
                    ))
                } else if let Some(global) = globals.get(&ident) {
                    stack.push(ResolvedType::Ptr(Box::new(global.ty.clone())));
                    let ident = CheckedIdent {
                        module_prefix: self.prefix.clone(),
                        ident,
                    };
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Global { location, ident },
                    ))
                } else {
                    Err(TypeError::LocalNotFound(ident.clone(), location))
                }
            }
            Word::Set { ident, location } => {
                if let Some(local) = locals.get(&ident) {
                    match stack.last() {
                        Some(ty) if ty == &local.ty => {
                            stack.pop();
                        }
                        ty => {
                            return Err(TypeError::ArgsMismatch(
                                Word::Set { location, ident },
                                vec![vec![local.ty.clone()]],
                                vec![ty.cloned()],
                            ));
                        }
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
                stack.push(ResolvedType::I32);
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
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [
                            ([ResolvedType::I32, ResolvedType::I32], &|_| {
                                ResolvedType::I32
                            }),
                            ([ResolvedType::I64, ResolvedType::I64], &|_| {
                                ResolvedType::I64
                            }),
                            ([ResolvedType::AnyPtr, ResolvedType::I32], &|[a, _]| a),
                        ],
                    )?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            intrinsic: CheckedIntrinsic::Add,
                            location: location.clone(),
                        },
                    ))
                }
                Intrinsic::Not => {
                    let ret = self.expect_stack(stack, &word, [([ResolvedType::Bool], &|[t]| t)])?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            intrinsic: CheckedIntrinsic::Not,
                            location: location.clone(),
                        },
                    ))
                }
                Intrinsic::Store32 => {
                    match (stack.pop(), stack.pop()) {
                        (Some(v_ty), Some(ResolvedType::Ptr(ty))) if *ty == v_ty => {}
                        (b, a) => return Err(TypeError::StoreArgs(location.clone(), a, b)),
                    }
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::Store32,
                        },
                    ))
                }
                Intrinsic::Store8 => {
                    self.expect_stack(
                        stack,
                        &word,
                        [(
                            [
                                ResolvedType::Ptr(Box::new(ResolvedType::I32)),
                                ResolvedType::I32,
                            ],
                            &|_| (),
                        )],
                    )?;
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::Store8,
                        },
                    ))
                }
                Intrinsic::Load32 => {
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [(
                            [ResolvedType::Ptr(Box::new(ResolvedType::AnyPtr))],
                            &|[ty]| match ty {
                                ResolvedType::Ptr(ty) => *ty,
                                _ => unreachable!(),
                            },
                        )],
                    )?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::Load32,
                        },
                    ))
                }
                Intrinsic::Load8 => {
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [([ResolvedType::Ptr(Box::new(ResolvedType::I32))], &|_| {
                            ResolvedType::I32
                        })],
                    )?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::Load8,
                        },
                    ))
                }
                Intrinsic::Drop => {
                    self.expect_stack(
                        stack,
                        &word,
                        [
                            ([ResolvedType::I32], &|_| ()),
                            ([ResolvedType::AnyPtr], &|_| ()),
                            ([ResolvedType::Bool], &|_| ()),
                        ],
                    )?;
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::Drop,
                        },
                    ))
                }
                Intrinsic::Sub => {
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [
                            ([ResolvedType::I32, ResolvedType::I32], &|_| {
                                ResolvedType::I32
                            }),
                            ([ResolvedType::AnyPtr, ResolvedType::I32], &|[pt_ty, _]| {
                                pt_ty
                            }),
                        ],
                    )?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::Sub,
                        },
                    ))
                }
                Intrinsic::Eq => {
                    let (ty, ret) = self.expect_stack(
                        stack,
                        &word,
                        [
                            ([ResolvedType::I32, ResolvedType::I32], &|_| {
                                (ResolvedType::I32, ResolvedType::Bool)
                            }),
                            ([ResolvedType::I64, ResolvedType::I64], &|_| {
                                (ResolvedType::I64, ResolvedType::Bool)
                            }),
                            ([ResolvedType::AnyPtr, ResolvedType::AnyPtr], &|[ty, _]| {
                                (ty, ResolvedType::Bool)
                            }),
                        ],
                    )?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::Eq(ty),
                        },
                    ))
                }
                Intrinsic::NotEq => {
                    let (ty, ret) = self.expect_stack(
                        stack,
                        &word,
                        [
                            ([ResolvedType::I32, ResolvedType::I32], &|_| {
                                (ResolvedType::I32, ResolvedType::Bool)
                            }),
                            ([ResolvedType::I64, ResolvedType::I64], &|_| {
                                (ResolvedType::I32, ResolvedType::Bool)
                            }),
                        ],
                    )?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::NotEq(ty),
                        },
                    ))
                }
                Intrinsic::Mod => {
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [
                            ([ResolvedType::I32, ResolvedType::I32], &|_| {
                                ResolvedType::I32
                            }),
                            ([ResolvedType::I64, ResolvedType::I64], &|_| {
                                ResolvedType::I64
                            }),
                        ],
                    )?;
                    stack.push(ret.clone());
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::Mod(ret),
                        },
                    ))
                }
                Intrinsic::Div => {
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [
                            ([ResolvedType::I32, ResolvedType::I32], &|_| {
                                ResolvedType::I32
                            }),
                            ([ResolvedType::I64, ResolvedType::I64], &|_| {
                                ResolvedType::I64
                            }),
                        ],
                    )?;
                    stack.push(ret.clone());
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::Div(ret),
                        },
                    ))
                }
                Intrinsic::And => {
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [([ResolvedType::Bool, ResolvedType::Bool], &|_| {
                            ResolvedType::Bool
                        })],
                    )?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::And,
                        },
                    ))
                }
                Intrinsic::Or => {
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [([ResolvedType::Bool, ResolvedType::Bool], &|_| {
                            ResolvedType::Bool
                        })],
                    )?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::Or,
                        },
                    ))
                }
                Intrinsic::L => {
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [([ResolvedType::I32, ResolvedType::I32], &|_| {
                            ResolvedType::Bool
                        })],
                    )?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::L,
                        },
                    ))
                }
                Intrinsic::G => {
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [([ResolvedType::I32, ResolvedType::I32], &|_| {
                            ResolvedType::Bool
                        })],
                    )?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::G,
                        },
                    ))
                }
                Intrinsic::LE => {
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [([ResolvedType::I32, ResolvedType::I32], &|_| {
                            ResolvedType::Bool
                        })],
                    )?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::LE,
                        },
                    ))
                }
                Intrinsic::GE => {
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [([ResolvedType::I32, ResolvedType::I32], &|_| {
                            ResolvedType::Bool
                        })],
                    )?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::GE,
                        },
                    ))
                }
                Intrinsic::Mul => {
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [([ResolvedType::I32, ResolvedType::I32], &|_| {
                            ResolvedType::I32
                        })],
                    )?;
                    stack.push(ret);
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::Mul,
                        },
                    ))
                }
                Intrinsic::Rotr => {
                    let ret = self.expect_stack(
                        stack,
                        &word,
                        [
                            ([ResolvedType::I32, ResolvedType::I32], &|_| {
                                ResolvedType::I32
                            }),
                            ([ResolvedType::I64, ResolvedType::I32], &|_| {
                                ResolvedType::I64
                            }),
                        ],
                    )?;
                    stack.push(ret.clone());
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::Rotr(ret),
                        },
                    ))
                }
                Intrinsic::Cast(ty) => {
                    let from = stack.last().unwrap().clone();
                    match ty {
                        ResolvedType::I32 => match stack.pop() {
                            Some(ResolvedType::Ptr(_)) => stack.push(ResolvedType::I32),
                            Some(ResolvedType::I64) => stack.push(ResolvedType::I32),
                            Some(ResolvedType::Bool) => stack.push(ResolvedType::I32),
                            _ => {
                                todo!()
                            }
                        },
                        ResolvedType::Ptr(ty) => {
                            let ret = self.expect_stack(
                                stack,
                                &word,
                                [
                                    ([ResolvedType::I32], &|_| ResolvedType::Ptr(ty.clone())),
                                    ([ResolvedType::AnyPtr], &|_| ResolvedType::Ptr(ty.clone())),
                                ],
                            )?;
                            stack.push(ret);
                        }
                        ResolvedType::Custom(_) => {
                            todo!()
                        }
                        ResolvedType::Bool => todo!(),
                        ResolvedType::AnyPtr => todo!(),
                        ResolvedType::I64 => {
                            let ret = self.expect_stack(
                                stack,
                                &word,
                                [
                                    ([ResolvedType::I32], &|_| ResolvedType::I64),
                                    ([ResolvedType::AnyPtr], &|_| ResolvedType::I64),
                                ],
                            )?;
                            stack.push(ret);
                        }
                    };
                    Ok((
                        Returns::Yes,
                        Vec::new(),
                        CheckedWord::Intrinsic {
                            location: location.clone(),
                            intrinsic: CheckedIntrinsic::Cast(from, ty.clone()),
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
                    Some(ResolvedType::Bool) => {}
                    ty => {
                        return Err(TypeError::ArgsMismatch(
                            word.clone(),
                            vec![vec![ResolvedType::Bool]],
                            vec![ty],
                        ))
                    }
                }
                let mut if_block_termination = Returns::Yes;
                let mut break_stacks = Vec::new();
                let mut if_block_stack = stack.clone();
                let mut if_block_checked_words = Vec::new();
                let mut min = stack.len();
                for word in body {
                    let (t, break_stack, checked_word) =
                        self.check_word(word.clone(), &mut if_block_stack, locals, globals, data)?;
                    if t == Returns::No {
                        if_block_termination = Returns::No
                    }
                    if_block_checked_words.push(checked_word);
                    break_stacks.extend(break_stack);
                    min = min.min(if_block_stack.len());
                }
                let param = stack[min..stack.len()].to_vec();
                let mut else_block_termination = Returns::Yes;
                let mut else_block_stack = stack.clone();
                let else_block_checked_words = match el {
                    Some(block) => {
                        let mut min = stack.len();
                        let mut else_block_checked_words = Vec::new();
                        for word in block {
                            let (t, break_stack, checked_word) = self.check_word(
                                word.clone(),
                                &mut else_block_stack,
                                locals,
                                globals,
                                data,
                            )?;
                            else_block_checked_words.push(checked_word);
                            if t == Returns::No {
                                else_block_termination = Returns::No;
                            }
                            break_stacks.extend(break_stack);
                            min = min.min(if_block_stack.len());
                        }
                        let else_param = stack[min..stack.len()].to_vec();
                        assert_eq!(&param, &else_param);
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
                                param,
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
                                param,
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
                                param,
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
                            param,
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
                        self.check_word(word, &mut loop_stack, locals, globals, data)?;
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
                stack.push(ResolvedType::Ptr(Box::new(ResolvedType::I32)));
                stack.push(ResolvedType::I32);
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
            Word::FieldDeref { location, field } => {
                let (ty, offset) = match stack.pop() {
                    Some(ResolvedType::Ptr(ty)) => match &*ty {
                        ResolvedType::Custom(struc) => {
                            let mut offset: u32 = 0;
                            let mut fields = struc.fields.iter();
                            loop {
                                match fields.next() {
                                    Some((n, t)) => {
                                        if n.lexeme == field {
                                            break (t.clone(), offset);
                                        }
                                    }
                                    None => {
                                        dbg!(&struc.fields, &field);
                                        todo!()
                                    }
                                }
                                offset += 4;
                            }
                        }
                        _ => todo!(),
                    },
                    _ => todo!(),
                };
                stack.push(ResolvedType::Ptr(Box::new(ty.clone())));
                Ok((
                    Returns::Yes,
                    Vec::new(),
                    CheckedWord::FieldDeref {
                        location,
                        offset,
                        ty,
                    },
                ))
            }
        }
    }
    fn expect_stack<T, const L: usize, const O: usize>(
        &self,
        stack: &mut Vec<ResolvedType>,
        word: &Word<ResolvedType>,
        expected: [Signature<T, L>; O],
    ) -> Result<T, TypeError> {
        assert!(!expected.is_empty());
        let expected_types: Vec<Vec<ResolvedType>> =
            expected.iter().map(|(tys, _)| tys.to_vec()).collect();
        let res = expected.into_iter().map(|(overload, f)| {
            let mut stack: Vec<ResolvedType> = stack.clone();
            let mut overloads = overload.into_iter().rev();
            let mut popped = Vec::new();
            loop {
                match overloads.next() {
                    Some(expected_ty) => match stack.pop() {
                        Some(ty) if ty == expected_ty => popped.push(ty),
                        Some(ty @ ResolvedType::Ptr(_))
                            if expected_ty == ResolvedType::AnyPtr
                                || matches!(expected_ty, ResolvedType::Ptr(_)) =>
                        {
                            popped.push(ty)
                        }
                        ty => {
                            popped.extend(ty.clone());
                            break Err(TypeError::ArgsMismatch(
                                word.clone(),
                                expected_types.clone(),
                                popped.into_iter().map(Some).chain([ty]).collect(),
                            ));
                        }
                    },
                    None => {
                        let mut popped: [ResolvedType; L] = popped.try_into().unwrap();
                        popped.reverse();
                        let ret = f(popped);
                        break Ok((stack, ret));
                    }
                }
            }
        });
        let mut error = None;
        for res in res {
            match res {
                Ok((s, rets)) => {
                    *stack = s;
                    return Ok(rets);
                }
                Err(e) => {
                    error = Some(e);
                }
            }
        }
        Err(error.unwrap())
    }
}
