use crate::{
    ast::{Intrinsic, Type},
    interpreter::{Error, Value},
    scanner::Location,
};

macro_rules! expect_args {
    ($location:expr, $stack:expr, [$(
        $var:ident: $ty:expr => $pat:pat
    ),*] => $body:block) => {
        match ($({
            let $var = ();
            drop($var);
            $stack.pop()
        },)*) {
            ($(
                Some($pat),
            )*) => {
                $body
            },
            ($($var,)*) => {
                return Err(Error::ArgsMismatch(
                    $location.clone(),
                    vec![$($ty,)*],
                    vec![$($var.map(Value::ty),)*]
                ))
            }
        }
    };
}

pub fn execute_intrinsic(
    intrinsic: &Intrinsic,
    location: &Location,
    stack: &mut Vec<Value>,
    memory: &mut [u8],
) -> Result<(), Error> {
    match intrinsic {
        Intrinsic::Add => expect_args!(location, stack, [
            a: Type::I32 => Value::I32(a),
            b: Type::I32 => Value::I32(b)
        ] => {
            stack.push(Value::I32(a + b))
        }),
        Intrinsic::Sub => expect_args!(location, stack, [
            a: Type::I32 => Value::I32(a),
            b: Type::I32 => Value::I32(b)
        ] => {
            stack.push(Value::I32(a - b))
        }),
        Intrinsic::Eq => expect_args!(location, stack, [
            a: Type::I32 => Value::I32(a),
            b: Type::I32 => Value::I32(b)
        ] => {
            stack.push(Value::Bool(a == b))
        }),
        Intrinsic::NotEq => expect_args!(location, stack, [
            a: Type::I32 => Value::I32(a),
            b: Type::I32 => Value::I32(b)
        ] => {
            stack.push(Value::Bool(a != b))
        }),
        Intrinsic::Mod => expect_args!(location, stack, [
            a: Type::I32 => Value::I32(a),
            b: Type::I32 => Value::I32(b)
        ] => {
            stack.push(Value::I32(b % a))
        }),
        Intrinsic::Div => expect_args!(location, stack, [
            a: Type::I32 => Value::I32(a),
            b: Type::I32 => Value::I32(b)
        ] => {
            stack.push(Value::I32(b / a))
        }),
        Intrinsic::Mul => expect_args!(location, stack, [
            a: Type::I32 => Value::I32(a),
            b: Type::I32 => Value::I32(b)
        ] => {
            stack.push(Value::I32(a * b))
        }),
        Intrinsic::Store32 => expect_args!(location, stack, [
            value: Type::I32 => Value::I32(value),
            addr: Type::I32 => Value::I32(addr)
        ] => {
            let addr = addr as usize;
            let bytes = value.to_le_bytes();
            for (i, v) in bytes.into_iter().enumerate() {
                memory[addr + i] = v;
            }
        }),
        Intrinsic::Store8 => expect_args!(location, stack, [
            value: Type::I32 => Value::I32(value),
            addr: Type::I32 => Value::I32(addr)
        ] => {
            let addr = addr as usize;
            let byte = value.to_le_bytes()[0];
            memory[addr] = byte;
        }),
        Intrinsic::Load32 => expect_args!(location, stack, [
            addr: Type::I32 => Value::I32(addr)
        ] => {
            let addr = addr as usize;
            let bytes: [u8; 4] = memory[addr..addr + 4].try_into().unwrap();
            stack.push(Value::I32(i32::from_le_bytes(bytes)))
        }),
        Intrinsic::Load8 => expect_args!(location, stack, [
            addr: Type::I32 => Value::I32(addr)
        ] => {
            let addr = addr as usize;
            let byte = memory[addr];
            stack.push(Value::I32(byte as i32))
        }),
        Intrinsic::Drop => {
            stack.pop();
        }
        Intrinsic::And => expect_args!(location, stack, [
            a: Type::Bool => Value::Bool(a),
            b: Type::Bool => Value::Bool(b)
        ] => {
            stack.push(Value::Bool(a && b))
        }),
        Intrinsic::LE => expect_args!(location, stack, [
            a: Type::I32 => Value::I32(a),
            b: Type::I32 => Value::I32(b)
        ] => {
            stack.push(Value::Bool(b <= a))
        }),
        Intrinsic::GE => expect_args!(location, stack, [
            a: Type::I32 => Value::I32(a),
            b: Type::I32 => Value::I32(b)
        ] => {
            stack.push(Value::Bool(b >= a))
        }),
        Intrinsic::Or => todo!(),
        Intrinsic::L => todo!(),
        Intrinsic::G => todo!(),
    }
    Ok(())
}
