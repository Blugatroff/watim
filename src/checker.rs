use std::collections::HashMap;

use crate::{
    ast::{Function, FunctionSignature, Iff, Intrinsic, Local, Program, Type, Word},
    scanner::Location,
};

pub struct TypeChecker {
    functions: HashMap<String, FunctionSignature>,
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
        }
    }
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }
    pub fn check(&mut self, program: &mut Program) -> Result<(), TypeError> {
        for ext in &program.externs {
            if self
                .functions
                .insert(ext.signature.ident.clone(), ext.signature.clone())
                .is_some()
            {}
        }
        for function in &program.functions {
            if self
                .functions
                .insert(function.signature.ident.clone(), function.signature.clone())
                .is_some()
            {
                return Err(TypeError::RedefinedFunction(
                    function.signature.ident.clone(),
                    function.signature.location.clone(),
                ));
            }
        }
        for function in &mut program.functions {
            self.check_function(function, &mut program.data)?;
        }
        Ok(())
    }
    fn check_function(
        &mut self,
        function: &mut Function,
        data: &mut Vec<u8>,
    ) -> Result<(), TypeError> {
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
        for word in &mut function.body {
            self.check_word(word, &mut stack, &locals, data)?;
        }
        if stack != function.signature.ret {
            return Err(TypeError::ReturnTypeMismatch(
                function.signature.location.clone(),
                stack,
                function.signature.ret.clone(),
            ));
        }
        Ok(())
    }
    fn check_word(
        &self,
        word: &mut Word,
        stack: &mut Vec<Type>,
        locals: &HashMap<String, Local>,
        data: &mut Vec<u8>,
    ) -> Result<(Returns, Vec<BreakStack>), TypeError> {
        match word {
            Word::Call { location, ident } => match self.functions.get(ident) {
                Some(function) => {
                    for param in function.params.iter().rev() {
                        match stack.pop() {
                            Some(ty) if &ty == &param.ty => {}
                            Some(ty) => {
                                return Err(TypeError::TypeMismatch(Some(ty), word.clone()))
                            }
                            None => return Err(TypeError::TypeMismatch(None, word.clone())),
                        }
                    }
                    stack.extend(function.ret.clone());
                    Ok((Returns::Yes, Vec::new()))
                }
                None => return Err(TypeError::FunctionNotFound(ident.clone(), location.clone())),
            },
            Word::Var { location, ident } => {
                if let Some(local) = locals.get(ident) {
                    stack.push(local.ty.clone());
                } else {
                    return Err(TypeError::LocalNotFound(ident.clone(), location.clone()));
                }
                Ok((Returns::Yes, Vec::new()))
            }
            Word::Set { ident, location } => {
                if let Some(local) = locals.get(ident) {
                    if let Some(ty) = stack.last() {
                        if ty != &local.ty {
                            return Err(TypeError::TypeMismatch(Some(ty.clone()), word.clone()));
                        }
                        stack.pop();
                    } else {
                        return Err(TypeError::TypeMismatch(None, word.clone()));
                    }
                } else {
                    return Err(TypeError::LocalNotFound(ident.clone(), location.clone()));
                }
                Ok((Returns::Yes, Vec::new()))
            }
            Word::Number { .. } => {
                stack.push(Type::I32);
                Ok((Returns::Yes, Vec::new()))
            }
            Word::Intrinsic { intrinsic, .. } => match intrinsic {
                Intrinsic::Add => {
                    self.expect_stack(stack, word, [Type::I32, Type::I32])?;
                    stack.push(Type::I32);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::Store32 => {
                    self.expect_stack(stack, word, [Type::I32, Type::I32])?;
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::Store8 => {
                    self.expect_stack(stack, word, [Type::I32, Type::I32])?;
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::Load32 => {
                    self.expect_stack(stack, word, [Type::I32])?;
                    stack.push(Type::I32);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::Load8 => {
                    self.expect_stack(stack, word, [Type::I32])?;
                    stack.push(Type::I32);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::Drop => {
                    self.expect_stack(stack, word, [Type::I32])?;
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::Sub => {
                    self.expect_stack(stack, word, [Type::I32, Type::I32])?;
                    stack.push(Type::I32);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::Eq => {
                    self.expect_stack(stack, word, [Type::I32, Type::I32])?;
                    stack.push(Type::Bool);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::Mod => {
                    self.expect_stack(stack, word, [Type::I32, Type::I32])?;
                    stack.push(Type::I32);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::Div => {
                    self.expect_stack(stack, word, [Type::I32, Type::I32])?;
                    stack.push(Type::I32);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::Stack => {
                    stack.push(Type::I32);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::And => {
                    self.expect_stack(stack, word, [Type::Bool, Type::Bool])?;
                    stack.push(Type::Bool);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::Or => {
                    self.expect_stack(stack, word, [Type::Bool, Type::Bool])?;
                    stack.push(Type::Bool);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::L => {
                    self.expect_stack(stack, word, [Type::I32, Type::I32])?;
                    stack.push(Type::Bool);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::G => {
                    self.expect_stack(stack, word, [Type::I32, Type::I32])?;
                    stack.push(Type::Bool);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::LE => {
                    self.expect_stack(stack, word, [Type::I32, Type::I32])?;
                    stack.push(Type::Bool);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::GE => {
                    self.expect_stack(stack, word, [Type::I32, Type::I32])?;
                    stack.push(Type::Bool);
                    Ok((Returns::Yes, Vec::new()))
                }
                Intrinsic::Mul => {
                    self.expect_stack(stack, word, [Type::I32, Type::I32])?;
                    stack.push(Type::I32);
                    Ok((Returns::Yes, Vec::new()))
                }
            },
            Word::If(Iff {
                location,
                body,
                el,
                ret,
            }) => {
                match stack.pop() {
                    Some(Type::Bool) => {}
                    ty => return Err(TypeError::TypeMismatch(ty, word.clone())),
                }
                let mut if_block_termination = Returns::Yes;
                let mut break_stacks = Vec::new();
                let mut if_block_stack = stack.clone();
                for word in body {
                    let (t, break_stack) =
                        self.check_word(word, &mut if_block_stack, locals, data)?;
                    if t == Returns::No {
                        if_block_termination = Returns::No
                    }
                    break_stacks.extend(break_stack);
                }
                let mut else_block_termination = Returns::Yes;
                let mut else_block_stack = stack.clone();
                match el {
                    Some(block) => {
                        for word in block {
                            let (t, break_stack) =
                                self.check_word(word, &mut else_block_stack, locals, data)?;
                            if t == Returns::No {
                                else_block_termination = Returns::No;
                            }
                            break_stacks.extend(break_stack);
                        }
                    }
                    None => {}
                }
                match (if_block_termination, else_block_termination) {
                    (Returns::Yes, Returns::Yes) => {
                        if if_block_stack != else_block_stack {
                            return Err(TypeError::IfBlocksMismatch(
                                location.clone(),
                                if_block_stack,
                                else_block_stack,
                            ));
                        }
                        *ret = if if_block_stack.len() > stack.len() {
                            if_block_stack[stack.len()..if_block_stack.len()].to_vec()
                        } else {
                            Vec::new()
                        };
                        *stack = if_block_stack;
                        Ok((Returns::Yes, break_stacks))
                    }
                    (Returns::Yes, Returns::No) => {
                        *ret = if if_block_stack.len() > stack.len() {
                            if_block_stack[stack.len()..if_block_stack.len()].to_vec()
                        } else {
                            Vec::new()
                        };
                        *stack = if_block_stack;
                        Ok((Returns::Yes, break_stacks))
                    }
                    (Returns::No, Returns::Yes) => {
                        *ret = if else_block_stack.len() > stack.len() {
                            else_block_stack[stack.len()..else_block_stack.len()].to_vec()
                        } else {
                            Vec::new()
                        };
                        *stack = else_block_stack;
                        Ok((Returns::Yes, break_stacks))
                    }
                    (Returns::No, Returns::No) => Ok((Returns::No, break_stacks)),
                }
            }
            Word::Loop(lop) => {
                let mut loop_stack = Vec::new();
                let mut break_stacks = Vec::new();
                for word in &mut lop.body {
                    let (_, break_stack) = self.check_word(word, &mut loop_stack, locals, data)?;
                    break_stacks.extend(break_stack);
                }
                if !loop_stack.is_empty() {
                    return Err(TypeError::ValuesLeftInLoop(
                        lop.location.clone(),
                        loop_stack,
                    ));
                }
                if let Some(first) = break_stacks.last() {
                    for stack in &break_stacks {
                        if stack.stack != first.stack {
                            return Err(TypeError::BreakTypeMismatch(
                                lop.location.clone(),
                                break_stacks,
                            ));
                        }
                    }
                    lop.ret = break_stacks
                        .first()
                        .map(|s| s.stack.clone())
                        .unwrap_or_default();
                    stack.extend(lop.ret.clone());
                    Ok((Returns::Yes, Vec::new()))
                } else {
                    Ok((Returns::No, Vec::new()))
                }
            }
            Word::Break { location } => Ok((
                Returns::No,
                vec![BreakStack {
                    location: location.clone(),
                    stack: stack.clone(),
                }],
            )),
            Word::String {
                value, addr, size, ..
            } => {
                *addr = data.len() as i32;
                *size = value.as_bytes().len() as i32;
                data.extend(value.as_bytes());
                stack.push(Type::I32);
                stack.push(Type::I32);
                Ok((Returns::Yes, Vec::new()))
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
                Some(ty) if &ty == &expected_ty => {}
                Some(ty) => return Err(TypeError::TypeMismatch(Some(ty), word.clone())),
                None => return Err(TypeError::TypeMismatch(None, word.clone())),
            }
        }
        Ok(())
    }
}
