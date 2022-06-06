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
        let data = self
            .data
            .iter()
            .map(|d| format!("{d}"))
            .intersperse(String::from("\n"))
            .reduce(|a, b| a + &b)
            .unwrap_or_default();
        let module = format!("\n{functions}");
        let externs = indent(&externs);
        f.write_fmt(format_args!(
            "(module\n{externs}\n\n\t(memory 1)\n\t(export \"memory\" (memory 0))\n\t{data}\n{}\n)",
            indent(&module)
        ))
    }
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let locals = self
            .locals
            .iter()
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
        f.write_fmt(format_args!(
            "({}{}{})",
            &self.signature,
            indent(&locals),
            indent(&body)
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
        f.write_fmt(format_args!(
            "(if\n\t(then\n{}\n\t){}\n)",
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
        f.write_fmt(format_args!(
            "(block $block\n\t(loop $loop\n{}\t\tbr $loop\n\t)\n)",
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
            Intrinsic::Load8 => "i32.load8",
            Intrinsic::Drop => "drop",
            Intrinsic::Sub => "i32.sub",
            Intrinsic::Eq => "i32.eq",
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
