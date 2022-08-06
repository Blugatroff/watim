use crate::{
    ast::{
        CheckedExtern, CheckedFunction, CheckedFunctionSignature, CheckedIdent, CheckedIff,
        CheckedIntrinsic, CheckedLoop, CheckedWord, Data, Local, Param, Program, ResolvedType,
    },
    checker::Returns,
};
use itertools::Itertools;

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

impl std::fmt::Display for CheckedExtern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "(import \"{}\" \"{}\" ({}))",
            &self.path.0, &self.path.1, self.signature
        ))
    }
}

impl std::fmt::Display for CheckedFunctionSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ident = format!("{}:{}", &self.prefix, &self.ident);
        let params = Itertools::intersperse(
            self.params.iter().map(|p| format!("{p}")),
            String::from(" "),
        )
        .reduce(|a, b| a + &b)
        .unwrap_or_default();
        let ret = Itertools::intersperse(
            self.ret.iter().map(|t| gen_type(t).to_string()),
            String::from(" "),
        )
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

fn gen_type(ty: &ResolvedType) -> &'static str {
    match ty {
        ResolvedType::I32 => "i32",
        ResolvedType::I64 => "i64",
        ResolvedType::Bool => "i32",
        ResolvedType::Ptr(_) => "i32",
        ResolvedType::AnyPtr => "i32",
        ResolvedType::Custom(_) => todo!(),
    }
}

impl std::fmt::Display for Param<ResolvedType> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ident = &self.ident;
        let ty = gen_type(&self.ty);
        f.write_fmt(format_args!("(param ${ident} {ty})"))
    }
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let externs = Itertools::intersperse(
            self.modules
                .iter()
                .flat_map(|a| &a.1.externs)
                .map(|e| format!("{e}")),
            String::from("\n"),
        )
        .reduce(|a, b| a + &b)
        .unwrap_or_default();
        let functions = Itertools::intersperse(
            self.modules
                .iter()
                .flat_map(|a| &a.1.functions)
                .map(|f| format!("{f}")),
            String::from("\n"),
        )
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

impl std::fmt::Display for CheckedFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let locals = Itertools::intersperse(
            self.locals
                .iter()
                .cloned()
                .chain(self.memory.iter().cloned().map(|m| Local {
                    ident: m.ident,
                    location: m.location,
                    ty: ResolvedType::I32,
                }))
                .map(|local| format!("{local}")),
            String::from("\n"),
        )
        .reduce(|a, b| a + &b)
        .unwrap_or_default();
        let body =
            Itertools::intersperse(self.body.iter().map(|w| format!("{w}")), String::from("\n"))
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
                Some(alignment) => {
                    // align = (n, alignment) => n + (alignment - (n % alignment)) * (n % alignment > 0)
                    // n alignment n alignment % - n alignment % 0 > * +
                    format!("\nglobal.get $stac:k\ni32.const {alignment}\nglobal.get $stac:k\ni32.const {alignment}\ni32.rem_u\ni32.sub\nglobal.get $stac:k\ni32.const {alignment}\ni32.rem_u\ni32.const 0\ni32.gt_u\ni32.mul\ni32.add\nglobal.set $stac:k")
                }
                None => String::new(),
            };
            use std::fmt::Write;
            write!(&mut mem_str, "{align}\nglobal.get $stac:k\nglobal.get $stac:k\ni32.const {size}\ni32.add\nglobal.set $stac:k\nlocal.set ${ident}").unwrap();
        }
        let drop = if stack == 0 {
            String::new()
        } else {
            String::from("local.get $stac:k\nglobal.set $stac:k\n")
        };
        let unreachable_filler = match self.returns {
            Returns::Yes => String::new(),
            Returns::No => Itertools::intersperse(self
                .signature
                .ret
                .iter()
                .map(|t| match t {
                    ResolvedType::I32 => "i32.const 0",
                    ResolvedType::I64 => "i64.const 0",
                    ResolvedType::Bool => "i32.const 0",
                    ResolvedType::Ptr(_) => "i32.const 0",
                    ResolvedType::AnyPtr => "i32.const 0",
                    ResolvedType::Custom(_) => "i32.const 0",
                }), " ")
                .flat_map(|s| s.chars())
                .collect(),
        };
        f.write_fmt(format_args!(
            "({}{}{}{}{}{})",
            &self.signature,
            indent(&locals),
            indent(&mem_str),
            indent(&body),
            indent(&drop),
            indent(&unreachable_filler)
        ))
    }
}

impl std::fmt::Display for Local<ResolvedType> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "(local ${} {})",
            self.ident,
            gen_type(&self.ty)
        ))
    }
}

impl std::fmt::Display for CheckedWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckedWord::Call { ident, .. } => f.write_fmt(format_args!("call ${ident}")),
            CheckedWord::Var { ident, .. } => f.write_fmt(format_args!("local.get ${ident}")),
            CheckedWord::Number { number, .. } => f.write_fmt(format_args!("i32.const {number}")),
            CheckedWord::Intrinsic { intrinsic, .. } => intrinsic.fmt(f),
            CheckedWord::If(iff) => iff.fmt(f),
            CheckedWord::Loop(lop) => lop.fmt(f),
            CheckedWord::Break { .. } => f.write_str("br $block"),
            CheckedWord::Set { ident, .. } => f.write_fmt(format_args!("local.set ${ident}")),
            CheckedWord::String { addr, size, .. } => {
                f.write_fmt(format_args!("i32.const {addr}\ni32.const {size}"))
            }
            CheckedWord::FieldDeref { offset, .. } => {
                f.write_fmt(format_args!("i32.const {offset} i32.add"))
            }
        }
    }
}

impl std::fmt::Display for CheckedIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let CheckedIdent {
            module_prefix,
            ident,
        } = self;
        f.write_fmt(format_args!("{module_prefix}:{ident}",))
    }
}

impl std::fmt::Display for CheckedIff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let body =
            Itertools::intersperse(self.body.iter().map(|w| format!("{w}")), String::from("\n"))
                .reduce(|a, b| a + &b)
                .unwrap_or_default();
        let el = match &self.el {
            Some(el) => {
                Itertools::intersperse(el.iter().map(|w| format!("{w}")), String::from("\n"))
                    .reduce(|a, b| a + &b)
                    .map(|el| format!("\n(else\n{}\n)", indent(&el)))
                    .unwrap_or_default()
            }
            None => String::new(),
        };
        let param = Itertools::intersperse(
            self.param.iter().map(|t| format!("(param {t})")),
            String::from(" "),
        )
        .fold(String::from(" "), |a, b| a + &b);
        let ret = Itertools::intersperse(
            self.ret.iter().map(|t| format!("(result {t})")),
            String::from(" "),
        )
        .fold(String::from(" "), |a, b| a + &b);
        f.write_fmt(format_args!(
            "(if {param} {ret}\n\t(then\n{}\n\t){}\n)",
            indent(&indent(&body)),
            indent(&el)
        ))
    }
}

impl std::fmt::Display for CheckedLoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let body = self
            .body
            .iter()
            .map(|w| format!("{w}\n"))
            .reduce(|a, b| a + &b)
            .unwrap_or_default();
        let ret = Itertools::intersperse(
            self.ret.iter().map(|t| format!("(result {})", gen_type(t))),
            String::from(" "),
        )
        .fold(String::from(" "), |a, b| a + &b);
        f.write_fmt(format_args!(
            "(block $block{ret}\n\t(loop $loop{ret}\n{}\t\tbr $loop\n\t)\n)",
            indent(&indent(&body)),
        ))
    }
}

impl std::fmt::Display for CheckedIntrinsic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            CheckedIntrinsic::Add => "i32.add",
            CheckedIntrinsic::Store32 => "i32.store",
            CheckedIntrinsic::Store8 => "i32.store8",
            CheckedIntrinsic::Load32 => "i32.load",
            CheckedIntrinsic::Load8 => "i32.load8_u",
            CheckedIntrinsic::Drop => "drop",
            CheckedIntrinsic::Sub => "i32.sub",
            CheckedIntrinsic::Eq(ty) => return write!(f, "{ty}.eq"),
            CheckedIntrinsic::Div(ty) => return write!(f, "{ty}.div_u"),
            CheckedIntrinsic::Mod(ty) => return write!(f, "{ty}.rem_u"),
            CheckedIntrinsic::And => "i32.and",
            CheckedIntrinsic::Or => "i32.or",
            CheckedIntrinsic::L => "i32.lt_u",
            CheckedIntrinsic::G => "i32.gt_u",
            CheckedIntrinsic::LE => "i32.le_u",
            CheckedIntrinsic::GE => "i32.ge_u",
            CheckedIntrinsic::Mul => "i32.mul",
            CheckedIntrinsic::NotEq(ty) => return write!(f, "{ty}.ne"),
            CheckedIntrinsic::Rotr(ResolvedType::I32) => "i32.rotr",
            CheckedIntrinsic::Rotr(ResolvedType::I64) => "i64.extend_i32_s i64.rotr",
            CheckedIntrinsic::Rotr(_) => todo!(),
            CheckedIntrinsic::Cast(from, to) => match (from, to) {
                (from, to) if from == to => "",
                (ResolvedType::Ptr(_), ResolvedType::Ptr(_)) => "",
                (ResolvedType::Ptr(_), ResolvedType::I32) => "",
                (ResolvedType::Bool, ResolvedType::I32) => "",
                (ResolvedType::I32, ResolvedType::I64) => "i64.extend_i32_s",
                (ResolvedType::I64, ResolvedType::I32) => "i32.wrap_i64",
                (ResolvedType::I32, ResolvedType::Ptr(_)) => "",
                _ => {
                    todo!()
                }
            },
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
