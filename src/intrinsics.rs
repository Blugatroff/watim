use crate::{
    ast::{Intrinsic, ResolvedType},
    interpreter::{Error, Value},
    scanner::Location,
};

type Signature<'a, R, const L: usize> = (
    [ResolvedType; L],
    &'a mut dyn FnMut(&[Value; L]) -> Option<R>,
);
fn expect_args<R: IntoIterator<Item = Value>, const O: usize, const L: usize>(
    location: &Location,
    stack: &mut Vec<Value>,
    overloads: [Signature<'_, R, L>; O],
) -> Result<(), Error> {
    let expected = overloads
        .iter()
        .map(|(expected, _)| expected.to_vec())
        .collect();
    let overloads = overloads.map(|(_, f)| f);
    let mut args = Vec::new();
    for _ in 0..L {
        match stack.pop() {
            Some(arg) => {
                args.push(arg);
            }
            None => {
                let args = args
                    .iter()
                    .map(Value::ty)
                    .map(Some)
                    .chain(std::iter::once(None))
                    .collect();
                return Err(Error::ArgsMismatch(location.clone(), expected, args));
            }
        }
    }
    let mut args: [Value; L] = args.try_into().unwrap();
    args.reverse();
    for f in overloads {
        if let Some(ret) = f(&args) {
            stack.extend(ret);
            return Ok(());
        }
    }
    Err(Error::ArgsMismatch(
        location.clone(),
        expected,
        args.iter().map(Value::ty).map(Some).collect(),
    ))
}

pub fn execute_intrinsic(
    intrinsic: &Intrinsic<ResolvedType>,
    location: &Location,
    stack: &mut Vec<Value>,
    memory: &mut [u8],
) -> Result<(), Error> {
    match intrinsic {
        Intrinsic::Add => expect_args(
            location,
            stack,
            [
                (
                    [ResolvedType::I32, ResolvedType::I32],
                    &mut |[a, b]| match (a, b) {
                        (Value::I32(a), Value::I32(b)) => Some([Value::I32(a + b)]),
                        _ => None,
                    },
                ),
                (
                    [ResolvedType::AnyPtr, ResolvedType::I32],
                    &mut |[a, b]| match (a, b) {
                        (Value::Ptr(a, ty), Value::I32(b)) => Some([Value::Ptr(a + b, ty.clone())]),
                        _ => None,
                    },
                ),
            ],
        ),
        Intrinsic::Sub => expect_args(
            location,
            stack,
            [
                (
                    [ResolvedType::I32, ResolvedType::I32],
                    &mut |[a, b]| match (a, b) {
                        (Value::I32(a), Value::I32(b)) => Some([Value::I32(a - b)]),
                        _ => None,
                    },
                ),
                (
                    [ResolvedType::AnyPtr, ResolvedType::I32],
                    &mut |[a, b]| match (a, b) {
                        (Value::Ptr(a, ty), Value::I32(b)) => Some([Value::Ptr(a - b, ty.clone())]),
                        _ => None,
                    },
                ),
            ],
        ),
        Intrinsic::Eq => expect_args(
            location,
            stack,
            [
                (
                    [ResolvedType::I32, ResolvedType::I32],
                    &mut |[a, b]| match (a, b) {
                        (Value::I32(a), Value::I32(b)) => Some([Value::Bool(a == b)]),
                        _ => None,
                    },
                ),
                (
                    [ResolvedType::AnyPtr, ResolvedType::AnyPtr],
                    &mut |[a, b]| match (a, b) {
                        (Value::Ptr(a, _), Value::Ptr(b, _)) => Some([Value::Bool(a == b)]),
                        _ => None,
                    },
                ),
                (
                    [ResolvedType::Bool, ResolvedType::Bool],
                    &mut |[a, b]| match (a, b) {
                        (Value::Bool(a), Value::Bool(b)) => Some([Value::Bool(a == b)]),
                        _ => None,
                    },
                ),
            ],
        ),
        Intrinsic::NotEq => expect_args(
            location,
            stack,
            [
                (
                    [ResolvedType::I32, ResolvedType::I32],
                    &mut |[a, b]| match (a, b) {
                        (Value::I32(a), Value::I32(b)) => Some([Value::Bool(a != b)]),
                        _ => None,
                    },
                ),
                (
                    [ResolvedType::AnyPtr, ResolvedType::AnyPtr],
                    &mut |[a, b]| match (a, b) {
                        (Value::Ptr(a, _), Value::Ptr(b, _)) => Some([Value::Bool(a != b)]),
                        _ => None,
                    },
                ),
                (
                    [ResolvedType::Bool, ResolvedType::Bool],
                    &mut |[a, b]| match (a, b) {
                        (Value::Bool(a), Value::Bool(b)) => Some([Value::Bool(a != b)]),
                        _ => None,
                    },
                ),
            ],
        ),
        Intrinsic::Mod => expect_args(
            location,
            stack,
            [(
                [ResolvedType::I32, ResolvedType::I32],
                &mut |[a, b]| match (a, b) {
                    (Value::I32(a), Value::I32(b)) => Some([Value::I32(a % b)]),
                    _ => None,
                },
            )],
        ),
        Intrinsic::Div => expect_args(
            location,
            stack,
            [(
                [ResolvedType::I32, ResolvedType::I32],
                &mut |[a, b]| match (a, b) {
                    (Value::I32(a), Value::I32(b)) => Some([Value::I32(a / b)]),
                    _ => None,
                },
            )],
        ),
        Intrinsic::Mul => expect_args(
            location,
            stack,
            [(
                [ResolvedType::I32, ResolvedType::I32],
                &mut |[a, b]| match (a, b) {
                    (Value::I32(a), Value::I32(b)) => Some([Value::I32(a * b)]),
                    _ => None,
                },
            )],
        ),
        Intrinsic::Store32 => {
            let b = stack.pop().unwrap();
            let a = stack.pop().unwrap();
            match (a, b) {
                (Value::Ptr(addr, ty), v) if ty == v.ty() => {
                    let value = match v {
                        Value::Bool(b) => b as i32,
                        Value::I32(v) => v,
                        Value::Ptr(v, _) => v,
                    };
                    let addr = addr as usize;
                    let bytes = value.to_le_bytes();
                    for (i, v) in bytes.into_iter().enumerate() {
                        memory[addr + i] = v;
                    }
                    Ok(())
                }
                a => {
                    dbg!(a);
                    todo!()
                }
            }
        }
        Intrinsic::Store8 => expect_args(
            location,
            stack,
            [(
                [
                    ResolvedType::Ptr(Box::new(ResolvedType::I32)),
                    ResolvedType::I32,
                ],
                &mut |[a, b]| match (a, b) {
                    (&Value::Ptr(addr, ResolvedType::I32), Value::I32(value)) => {
                        let addr = addr as usize;
                        let byte = value.to_le_bytes()[0];
                        memory[addr] = byte;
                        Some([])
                    }
                    _ => None,
                },
            )],
        ),
        Intrinsic::Load32 => expect_args(
            location,
            stack,
            [(
                [ResolvedType::Ptr(Box::new(ResolvedType::I32))],
                &mut |[a]| match a {
                    &Value::Ptr(addr, ResolvedType::I32) => {
                        let addr = addr as usize;
                        let bytes: [u8; 4] = memory[addr..addr + 4].try_into().unwrap();
                        Some([Value::I32(i32::from_le_bytes(bytes))])
                    }
                    _ => None,
                },
            )],
        ),
        Intrinsic::Load8 => expect_args(
            location,
            stack,
            [(
                [ResolvedType::Ptr(Box::new(ResolvedType::I32))],
                &mut |[a]| match a {
                    &Value::Ptr(addr, ResolvedType::I32) => {
                        let addr = addr as usize;
                        let byte = memory[addr];
                        Some([Value::I32(byte as i32)])
                    }
                    _ => None,
                },
            )],
        ),
        Intrinsic::Drop => expect_args(
            location,
            stack,
            [
                ([ResolvedType::AnyPtr], &mut |[a]| match a {
                    &Value::Ptr(_, _) => Some([]),
                    _ => None,
                }),
                ([ResolvedType::I32], &mut |[a]| match a {
                    &Value::I32(_) => Some([]),
                    _ => None,
                }),
                ([ResolvedType::Bool], &mut |[a]| match a {
                    &Value::Bool(_) => Some([]),
                    _ => None,
                }),
            ],
        ),
        Intrinsic::And => expect_args(
            location,
            stack,
            [(
                [ResolvedType::Bool, ResolvedType::Bool],
                &mut |[a, b]| match (a, b) {
                    (&Value::Bool(a), &Value::Bool(b)) => Some([Value::Bool(a && b)]),
                    _ => None,
                },
            )],
        ),
        Intrinsic::LE => expect_args(
            location,
            stack,
            [(
                [ResolvedType::I32, ResolvedType::I32],
                &mut |[a, b]| match (a, b) {
                    (&Value::I32(a), &Value::I32(b)) => Some([Value::Bool(a <= b)]),
                    _ => None,
                },
            )],
        ),
        Intrinsic::GE => expect_args(
            location,
            stack,
            [(
                [ResolvedType::I32, ResolvedType::I32],
                &mut |[a, b]| match (a, b) {
                    (&Value::I32(a), &Value::I32(b)) => Some([Value::Bool(a >= b)]),
                    _ => None,
                },
            )],
        ),
        Intrinsic::Or => todo!(),
        Intrinsic::L => todo!(),
        Intrinsic::G => todo!(),
        Intrinsic::Cast(ResolvedType::I32) => expect_args(
            location,
            stack,
            [([ResolvedType::AnyPtr], &mut |[v]| match v {
                &Value::Ptr(v, _) => Some([Value::I32(v)]),
                _ => None,
            })],
        ),
        Intrinsic::Cast(ResolvedType::Ptr(ty)) => expect_args(
            location,
            stack,
            [
                ([ResolvedType::I32], &mut |[v]| match v {
                    &Value::I32(v) => Some([Value::Ptr(v, (**ty).clone())]),
                    _ => None,
                }),
                ([ResolvedType::AnyPtr], &mut |[v]| match v {
                    &Value::Ptr(v, _) => Some([Value::Ptr(v, (**ty).clone())]),
                    _ => None,
                }),
            ],
        ),
        Intrinsic::Cast(ResolvedType::Bool) => todo!(),
        Intrinsic::Cast(ResolvedType::AnyPtr) => todo!(),
        Intrinsic::Cast(ResolvedType::Custom(_)) => todo!(),
    }
}
