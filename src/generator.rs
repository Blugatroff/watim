use crate::ast::{
    Data, Extern, Function, FunctionSignature, Iff, Intrinsic, Local, Loop, Param, Program, Type,
    Word,
};

fn indent(input: &str) -> String {
    let mut res = input
        .lines()
        .map(|l| format!("\t{l}\n"))
        .fold(String::new(), |a, b| a + &b);
    if !input.is_empty() && Some(b'\n') != input.as_bytes().get(input.len() - 1).copied() {
        res.pop();
    }
    res
}

impl std::fmt::Display for Extern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "(import \"{}\" \"{}\" ({}))",
            &self.path.0, &self.path.1, self.signature
        ))
    }
}

impl std::fmt::Display for FunctionSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ident = &self.ident;
        let params = self
            .params
            .iter()
            .map(|p| format!("{p}"))
            .intersperse(String::from(" "))
            .reduce(|a, b| a + &b)
            .unwrap_or_default();
        let ret = self
            .ret
            .iter()
            .map(|t| format!("{t}"))
            .intersperse(String::from(" "))
            .reduce(|a, b| a + &b)
            .map(|r| format!(" (result {r})"))
            .unwrap_or_default();
        let export = self
            .export
            .as_ref()
            .map(|e| format!("(export \"{e}\") "))
            .unwrap_or_default();
        f.write_fmt(format_args!("func ${ident} {export}{params}{ret}",))
    }
}

impl std::fmt::Display for Param {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ident = &self.ident;
        let ty = &self.ty;
        f.write_fmt(format_args!("(param ${ident} {ty})"))
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::I32 => f.write_str("i32"),
            Type::Bool => f.write_str("i32"),
        }
    }
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let externs = self
            .externs
            .iter()
            .map(|e| format!("{e}"))
            .intersperse(String::from("\n"))
            .reduce(|a, b| a + &b)
            .unwrap_or_default();
        let functions = self
            .functions
            .iter()
            .map(|f| format!("{f}"))
            .intersperse(String::from("\n"))
            .reduce(|a, b| a + &b)
            .unwrap_or_default();
        let data = String::from_utf8(self.data.clone()).unwrap();
        let data = format!("(data (i32.const 0) \"{data}\")");
        let module = format!("\n{functions}");
        let externs = indent(&externs);
        let stack_start = self.data.len();
        f.write_fmt(format_args!(
            "(module\n{externs}\n\n\t(memory 1)\n\t(export \"memory\" (memory 0))\n\t(global $stac:k (mut i32) (i32.const {stack_start}))\n\t{data}\n{}\n)",
            indent(&module)
        ))
    }
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let locals = self
            .locals
            .iter()
            .cloned()
            .chain(self.memory.iter().cloned().map(|m| Local {
                ident: m.ident,
                location: m.location,
                ty: Type::I32,
            }))
            .map(|local| format!("{local}"))
            .intersperse(String::from("\n"))
            .reduce(|a, b| a + &b)
            .unwrap_or_default();
        let body = self
            .body
            .iter()
            .map(|w| format!("{w}"))
            .intersperse(String::from("\n"))
            .reduce(|a, b| a + &b)
            .unwrap_or_default();
        let locals = if locals.is_empty() {
            locals
        } else {
            format!("\n{locals}")
        };
        let body = if body.is_empty() {
            body
        } else {
            format!("\n{body}\n")
        };
        let stack: i32 = self.memory.iter().map(|m| m.size).sum();
        let mut mem_str = if stack > 0 {
            String::from("\n(local $stac:k i32)\nglobal.get $stac:k\nlocal.set $stac:k")
        } else {
            String::new()
        };
        for mem in &self.memory {
            let size = mem.size;
            let ident = &mem.ident;
            let align = match mem.alignment {
                Some(alignment) => format!("\nglobal.get $stac:k\ni32.const {alignment}\nglobal.get $stac:k\ni32.const {alignment}\ni32.rem_u\ni32.sub\ni32.add\nglobal.set $stac:k"),
                None => String::new(),
            };

            mem_str.push_str(&format!("{align}\nglobal.get $stac:k\nglobal.get $stac:k\ni32.const {size}\ni32.add\nglobal.set $stac:k\nlocal.set ${ident}"));
        }
        let drop = if stack == 0 {
            String::new()
        } else {
            format!("local.get $stac:k\nglobal.set $stac:k\n")
        };
        f.write_fmt(format_args!(
            "({}{}{}{}{})",
            &self.signature,
            indent(&locals),
            indent(&mem_str),
            indent(&body),
            indent(&drop),
        ))
    }
}

impl std::fmt::Display for Local {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("(local ${} {})", self.ident, self.ty))
    }
}

impl std::fmt::Display for Word {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Word::Call { ident, .. } => f.write_fmt(format_args!("call ${ident}")),
            Word::Var { ident, .. } => f.write_fmt(format_args!("local.get ${ident}")),
            Word::Number { number, .. } => f.write_fmt(format_args!("i32.const {number}")),
            Word::Intrinsic { intrinsic, .. } => intrinsic.fmt(f),
            Word::If(iff) => iff.fmt(f),
            Word::Loop(lop) => lop.fmt(f),
            Word::Break { .. } => f.write_str("br $block"),
            Word::Set { ident, .. } => f.write_fmt(format_args!("local.set ${ident}")),
            Word::String { addr, size, .. } => {
                f.write_fmt(format_args!("i32.const {addr}\ni32.const {size}"))
            }
        }
    }
}

impl std::fmt::Display for Iff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let body = self
            .body
            .iter()
            .map(|w| format!("{w}"))
            .intersperse(String::from("\n"))
            .reduce(|a, b| a + &b)
            .unwrap_or_default();
        let el = match &self.el {
            Some(el) => el
                .iter()
                .map(|w| format!("{w}"))
                .intersperse(String::from("\n"))
                .reduce(|a, b| a + &b)
                .map(|el| format!("\n(else\n{}\n)", indent(&el)))
                .unwrap_or_default(),
            None => String::new(),
        };
        let ret = self
            .ret
            .iter()
            .map(|t| format!("(result {t})"))
            .intersperse(String::from(" "))
            .fold(String::from(" "), |a, b| a + &b);
        f.write_fmt(format_args!(
            "(if{ret}\n\t(then\n{}\n\t){}\n)",
            indent(&indent(&body)),
            indent(&el)
        ))
    }
}

impl std::fmt::Display for Loop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let body = self
            .body
            .iter()
            .map(|w| format!("{w}\n"))
            .reduce(|a, b| a + &b)
            .unwrap_or_default();
        let ret = self
            .ret
            .iter()
            .map(|t| format!("(result {t})"))
            .intersperse(String::from(" "))
            .fold(String::from(" "), |a, b| a + &b);
        f.write_fmt(format_args!(
            "(block $block{ret}\n\t(loop $loop{ret}\n{}\t\tbr $loop\n\t)\n)",
            indent(&indent(&body)),
        ))
    }
}

impl std::fmt::Display for Intrinsic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Intrinsic::Add => "i32.add",
            Intrinsic::Store32 => "i32.store",
            Intrinsic::Store8 => "i32.store8",
            Intrinsic::Load32 => "i32.load",
            Intrinsic::Load8 => "i32.load8_u",
            Intrinsic::Drop => "drop",
            Intrinsic::Sub => "i32.sub",
            Intrinsic::Eq => "i32.eq",
            Intrinsic::Div => "i32.div_u",
            Intrinsic::Mod => "i32.rem_u",
            Intrinsic::Stack => "global.get $stac:k",
            Intrinsic::And => "i32.and",
            Intrinsic::Or => "i32.or",
            Intrinsic::L => "i32.lt_u",
            Intrinsic::G => "i32.gt_u",
            Intrinsic::LE => "i32.le_u",
            Intrinsic::GE => "i32.ge_u",
            Intrinsic::Mul => "i32.mul",
        })
    }
}

impl std::fmt::Display for Data {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "(data (i32.const {}) \"{}\")",
            self.addr, &self.data
        ))
    }
}
