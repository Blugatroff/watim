use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{
    ast::{
        Extern, Function, FunctionSignature, Ident, Iff, Intrinsic, Local, Loop, Memory, Module,
        Param, Program, ResolvedType, Struct, UnResolvedType, Word,
    },
    checker::{ModuleChecker, TypeError},
    parser,
    scanner::{Location, Scanner},
    WatimError,
};

pub struct UncheckedProgram<Type> {
    modules: BTreeMap<PathBuf, Module<Type>>,
    root: PathBuf,
}

impl UncheckedProgram<UnResolvedType> {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, WatimError> {
        Self::load_with_custom_file_loader(&mut |path| std::fs::read_to_string(path), path)
    }
    pub fn load_with_custom_file_loader(
        loader: &mut dyn FnMut(&Path) -> Result<String, std::io::Error>,
        path: impl AsRef<Path>,
    ) -> Result<Self, WatimError> {
        fn inner_load(
            loader: &mut dyn FnMut(&Path) -> Result<String, std::io::Error>,
            modules: &mut BTreeMap<PathBuf, Module<UnResolvedType>>,
            path: impl AsRef<Path>,
            input: String,
        ) -> Result<(), WatimError> {
            let path = path.as_ref().to_path_buf();
            if modules.get(&path).is_some() {
                return Ok(());
            }
            let tokens = Scanner::scan_tokens(input, path.clone())?;
            let module = parser::Parser::new(tokens).parse(&path)?;
            let imports = module.imports.clone();
            modules.insert(path.canonicalize()?, module);
            for import in imports {
                let path = path.parent().unwrap().join(import.path);
                let input = loader(path.as_path())?;
                inner_load(loader, modules, path, input)?;
            }
            Ok(())
        }
        let mut modules = BTreeMap::new();
        let root = path.as_ref().to_path_buf();
        let input = loader(path.as_ref())?;
        inner_load(loader, &mut modules, path, input)?;
        Ok(Self { modules, root })
    }
    pub fn resolve(self) -> Result<UncheckedProgram<ResolvedType>, WatimError> {
        let root = self.modules.get(&self.root).unwrap();
        let mut structs = HashMap::new();
        fn inner(
            module: &Module<UnResolvedType>,
            modules: &BTreeMap<PathBuf, Module<UnResolvedType>>,
            structs: &mut HashMap<PathBuf, HashMap<String, Arc<Struct<ResolvedType>>>>,
            resolved_modules: &mut BTreeMap<PathBuf, Module<ResolvedType>>,
        ) -> Result<(), WatimError> {
            let mut modules_idents = HashMap::new();
            for import in &module.imports {
                let path = module
                    .path
                    .parent()
                    .unwrap()
                    .join(&import.path)
                    .canonicalize()
                    .unwrap();
                let m = match modules.get(&path) {
                    Some(m) => m,
                    None => {
                        return Err(WatimError::Type(TypeError::ModuleNotFound(
                            import.path.clone(),
                            import.location.clone(),
                        )))
                    }
                };
                inner(m, modules, structs, resolved_modules)?;
                modules_idents.insert(import.ident.clone(), path);
            }
            let current_module = &module.path;
            let mut resolved_structs = Vec::new();
            for struc in &module.structs {
                if structs
                    .get(current_module)
                    .and_then(|m| m.get(&struc.ident.lexeme))
                    .is_some()
                {
                    continue;
                }
                let mut fields = Vec::new();
                for (t, ty) in &struc.fields {
                    let location = &t.location;
                    let ty = resolve_ty(ty, structs, &modules_idents, current_module, location)?;
                    fields.push((t.clone(), ty));
                }
                let struc = Struct {
                    fields,
                    ident: struc.ident.clone(),
                };
                let struc = Arc::new(struc);
                structs
                    .entry(current_module.clone())
                    .or_default()
                    .insert(struc.ident.lexeme.clone(), Arc::clone(&struc));
                resolved_structs.push(struc);
            }
            fn resolve_signature(
                structs: &mut HashMap<PathBuf, HashMap<String, Arc<Struct<ResolvedType>>>>,
                signature: &FunctionSignature<UnResolvedType>,
                current_module: &PathBuf,
                modules_idents: &HashMap<String, PathBuf>,
            ) -> Result<FunctionSignature<ResolvedType>, WatimError> {
                let mut ret = Vec::new();
                for t in &signature.ret {
                    let location = &signature.location;
                    let t = resolve_ty(t, structs, modules_idents, current_module, location)?;
                    ret.push(t);
                }
                let mut params = Vec::new();
                for p in &signature.params {
                    let location = &p.location;
                    let t = &p.ty;
                    let t = resolve_ty(t, structs, modules_idents, current_module, location)?;
                    params.push(Param {
                        ident: p.ident.clone(),
                        ty: t,
                        location: location.clone(),
                    });
                }
                Ok(FunctionSignature {
                    location: signature.location.clone(),
                    params,
                    ret,
                    ident: signature.ident.clone(),
                    export: signature.export.clone(),
                })
            }
            let mut functions = Vec::new();
            for f in &module.functions {
                let signature =
                    resolve_signature(structs, &f.signature, current_module, &modules_idents)?;
                let mut locals = Vec::new();
                for l in &f.locals {
                    let location = &l.location;
                    let t = &l.ty;
                    let t = resolve_ty(t, structs, &modules_idents, current_module, location)?;
                    locals.push(Local {
                        ident: l.ident.clone(),
                        location: location.clone(),
                        ty: t,
                    });
                }
                fn check_body(
                    words: impl IntoIterator<Item = Word<UnResolvedType>>,
                    structs: &mut HashMap<PathBuf, HashMap<String, Arc<Struct<ResolvedType>>>>,
                    current_module: &PathBuf,
                    modules_idents: &HashMap<String, PathBuf>,
                ) -> Result<Vec<Word<ResolvedType>>, WatimError> {
                    let mut body = Vec::new();
                    for w in words {
                        let w = match w.clone() {
                            Word::Call { location, ident } => Word::Call { location, ident },
                            Word::Var { location, ident } => Word::Var { location, ident },
                            Word::Set { location, ident } => Word::Set { location, ident },
                            Word::Number { location, number } => Word::Number { location, number },
                            Word::Intrinsic {
                                location,
                                intrinsic,
                            } => Word::Intrinsic {
                                intrinsic: match intrinsic {
                                    Intrinsic::Add => Intrinsic::Add,
                                    Intrinsic::Store32 => Intrinsic::Store32,
                                    Intrinsic::Store8 => Intrinsic::Store8,
                                    Intrinsic::Load32 => Intrinsic::Load32,
                                    Intrinsic::Load8 => Intrinsic::Load8,
                                    Intrinsic::Drop => Intrinsic::Drop,
                                    Intrinsic::Sub => Intrinsic::Sub,
                                    Intrinsic::Eq => Intrinsic::Eq,
                                    Intrinsic::NotEq => Intrinsic::NotEq,
                                    Intrinsic::Mod => Intrinsic::Mod,
                                    Intrinsic::Div => Intrinsic::Div,
                                    Intrinsic::And => Intrinsic::And,
                                    Intrinsic::Not => Intrinsic::Not,
                                    Intrinsic::Or => Intrinsic::Or,
                                    Intrinsic::L => Intrinsic::L,
                                    Intrinsic::G => Intrinsic::G,
                                    Intrinsic::LE => Intrinsic::LE,
                                    Intrinsic::GE => Intrinsic::GE,
                                    Intrinsic::Mul => Intrinsic::Mul,
                                    Intrinsic::Rotr => Intrinsic::Rotr,
                                    Intrinsic::Cast(ty) => Intrinsic::Cast(resolve_ty(
                                        &ty,
                                        structs,
                                        modules_idents,
                                        current_module,
                                        &location,
                                    )?),
                                },
                                location,
                            },
                            Word::If(Iff { location, body, el }) => Word::If(Iff {
                                location,
                                body: check_body(body, structs, current_module, modules_idents)?,
                                el: match el {
                                    Some(el) => Some(check_body(
                                        el,
                                        structs,
                                        current_module,
                                        modules_idents,
                                    )?),
                                    None => None,
                                },
                            }),
                            Word::Loop(Loop { location, body }) => Word::Loop(Loop {
                                location,
                                body: check_body(body, structs, current_module, modules_idents)?,
                            }),
                            Word::Break { location } => Word::Break { location },
                            Word::String { location, value } => Word::String { location, value },
                            Word::FieldDeref { location, field } => {
                                Word::FieldDeref { location, field }
                            }
                        };
                        body.push(w);
                    }
                    Ok(body)
                }
                let body = check_body(
                    f.body.iter().cloned(),
                    structs,
                    current_module,
                    &modules_idents,
                )?;
                let mut memory = Vec::new();
                for m in &f.memory {
                    let location = &m.location;
                    let ty = resolve_ty(&m.ty, structs, &modules_idents, current_module, location)?;
                    memory.push(Memory {
                        ident: m.ident.clone(),
                        location: m.location.clone(),
                        size: m.size,
                        alignment: m.alignment,
                        ty,
                    })
                }
                let f = Function {
                    memory,
                    signature,
                    body,
                    locals,
                };
                functions.push(f);
            }
            let mut externs = Vec::new();
            for ext in &module.externs {
                let signature =
                    resolve_signature(structs, &ext.signature, current_module, &modules_idents)?;
                externs.push(Extern {
                    signature,
                    location: ext.location.clone(),
                    path: ext.path.clone(),
                })
            }
            let mut memory = Vec::new();
            for mem in &module.memory {
                let ty = resolve_ty(
                    &mem.ty,
                    structs,
                    &modules_idents,
                    current_module,
                    &mem.location,
                )?;
                memory.push(Memory {
                    alignment: mem.alignment,
                    ident: mem.ident.clone(),
                    location: mem.location.clone(),
                    size: mem.size,
                    ty,
                });
            }
            resolved_modules.insert(
                module.path.canonicalize()?,
                Module {
                    externs,
                    functions,
                    imports: module.imports.clone(),
                    structs: resolved_structs,
                    path: module.path.clone(),
                    memory,
                },
            );
            Ok(())
        }
        let mut resolved_modules = BTreeMap::new();
        inner(root, &self.modules, &mut structs, &mut resolved_modules)?;
        Ok(UncheckedProgram {
            modules: resolved_modules,
            root: self.root,
        })
    }
}

impl UncheckedProgram<ResolvedType> {
    pub fn check(self) -> Result<Program, TypeError> {
        fn check(modules: BTreeMap<PathBuf, Module<ResolvedType>>) -> Result<Program, TypeError> {
            let modules: HashMap<_, _> = modules
                .into_iter()
                .enumerate()
                .map(|(i, (k, v))| (k, (v, format!("{i}"))))
                .collect();
            let mut data = Vec::new();
            let mut checked_modules = HashMap::new();
            for (path, (module, _)) in &modules {
                let module = ModuleChecker::check(module.clone(), &modules, &mut data)?;
                checked_modules.insert(path.clone(), module);
            }
            Ok(Program {
                data,
                modules: checked_modules,
            })
        }
        check(self.modules)
    }
}

fn resolve_ty(
    ty: &UnResolvedType,
    structs: &HashMap<PathBuf, HashMap<String, Arc<Struct<ResolvedType>>>>,
    modules: &HashMap<String, PathBuf>,
    current_module: &PathBuf,
    location: &Location,
) -> Result<ResolvedType, WatimError> {
    match ty {
        UnResolvedType::I32 => Ok(ResolvedType::I32),
        UnResolvedType::I64 => Ok(ResolvedType::I64),
        UnResolvedType::Bool => Ok(ResolvedType::Bool),
        UnResolvedType::Ptr(t) => Ok(ResolvedType::Ptr(Box::new(resolve_ty(
            t,
            structs,
            modules,
            current_module,
            location,
        )?))),
        UnResolvedType::Custom(ident) => {
            let (ident, module) = match ident {
                Ident::Direct(ident) => (ident, current_module),
                Ident::Qualified(module, ident) => (
                    ident,
                    match modules.get(module) {
                        Some(module) => module,
                        None => {
                            return Err(WatimError::Type(TypeError::ModuleNotFound(
                                module.clone(),
                                location.clone(),
                            )))
                        }
                    },
                ),
            };
            let struc = match structs.get(module).and_then(|m| m.get(ident)) {
                Some(struc) => struc,
                None => {
                    return Err(WatimError::Type(TypeError::StructNotFound(
                        ident.clone(),
                        location.clone(),
                    )))
                }
            };
            Ok(ResolvedType::Custom(struc.clone()))
        }
    }
}
