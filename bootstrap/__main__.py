#!/usr/bin/env python
from dataclasses import dataclass
from typing import List, Dict, Literal
import sys
import os
import unittest

import parser
from util import sys_stdin
from indexed_dict import IndexedDict
from format import format_str, format
from parser import Parser, ParserException
from lexer import TokenLocation, Lexer
import resolver
from resolver import determine_compilation_order, ResolveCtx, TypeLookup, ResolverException
from monomizer import Monomizer, merge_locals_module, DetermineLoadsToValueTests
from wat_generator import WatGenerator

def load_recursive(
        modules: Dict[str, parser.Module],
        path: str,
        path_location: TokenLocation | None,
        stdin: str | None = None,
        import_stack: List[str]=[]):
    if path == "-":
        file = stdin if stdin is not None else sys_stdin.get()
    else:
        try:
            with open(path, 'r') as reader:
                file = reader.read()
        except FileNotFoundError:
            raise ParserException(path_location, f"File not found: ./{path}")

    tokens = Lexer(file).lex()
    module = Parser(path, file, tokens).parse()
    modules[path] = module
    for imp in module.imports:
        if os.path.dirname(path) != "":
            p = os.path.normpath(os.path.dirname(path) + "/" + imp.file_path.lexeme[1:-1])
        else:
            p = os.path.normpath(imp.file_path.lexeme[1:-1])
        if p in import_stack:
            error_message = "Module import cycle detected: "
            for a in import_stack:
                error_message += f"{a} -> "
            raise ParserException(TokenLocation(path, imp.file_path.line, imp.file_path.column), error_message)
        if p in modules:
            continue
        import_stack.append(p)
        load_recursive(
            modules,
            p,
            TokenLocation(path, imp.file_path.line, imp.file_path.column),
            stdin,
            import_stack,
        )
        import_stack.pop()

def resolve_modules(modules_unordered: Dict[str, List[parser.TopItem]]) -> IndexedDict[str, resolver.Module]:
    modules = determine_compilation_order({
        ("./" + path if path != "-" else path): module
        for path, module in modules_unordered.items()
    })
    resolved_modules: IndexedDict[str, resolver.Module] = IndexedDict()
    other_module_types: List[List[resolver.CustomType]] = []
    for id,(module_path,top_items) in enumerate(modules.items()):
        ctx = ResolveCtx(modules, resolved_modules, top_items, id, bytearray())
        imports = ctx.resolve_imports()
        custom_types = ctx.resolve_custom_types(imports)
        globals = ctx.resolve_globals(imports)
        signatures = ctx.resolve_signatures(imports)

        type_lookup = TypeLookup(module=id, types=list(custom_types.values()), other_modules=other_module_types)
        functions = ctx.resolve_functions(imports, type_lookup, signatures, globals)

        ctx.forbid_directly_recursive_types(type_lookup)

        other_module_types.append(list(custom_types.values()))
        resolved_modules[module_path] = resolver.Module(
            module_path,
            id,
            imports,
            custom_types,
            globals,
            functions,
            bytes(ctx.static_data),
        )
    return resolved_modules

Mode = Literal["lex"] | Literal["parse"] | Literal["check"] | Literal["monomize"] | Literal["compile"] | Literal["inference-tree"]

def run(path: str, mode: Mode, guard_stack: bool, stdin: str | None = None) -> str:
    if path == "-":
        file = stdin if stdin is not None else sys_stdin.get()
    else:
        with open(path, 'r') as reader:
            file = reader.read()
    tokens = Lexer(file).lex()
    if mode == "lex":
        return "\n".join([format(token.format_instrs()) for token in tokens])
    if mode == "parse":
        module = Parser(path, file, tokens).parse()
        return format(module.format_instrs())
    modules: Dict[str, parser.Module] = {}
    load_recursive(modules, os.path.normpath(path), None, stdin)
    resolved_modules = resolve_modules({ k: m.top_items for k,m in modules.items()})
    if mode == "check":
        return format(resolved_modules.format_instrs(format_str))
    function_table, mono_modules = Monomizer(resolved_modules).monomize()
    if mode == "monomize":
        return "TODO"
    if mode == "inference-tree":
        return ""
    for mono_module in mono_modules.values():
        merge_locals_module(mono_module)
    return WatGenerator(mono_modules, function_table, guard_stack).write_wat_module()

help = """The native Watim compiler

Usage: watim <command> <watim-source-file> [options]
Commands:
  lex       [path]   Lex code and print the Tokens.
  parse     [path]   Parse code and print the AST
  check     [path]   Typecheck and print the AST
  monomize  [path]   Monomize the entire program
  optimize  [path]   Optimize the entire program
  compile   [path]   Compile the entire program
Options:
  -q, --quiet  Don't print any logs to stderr
"""

@dataclass
class CliArgException(Exception):
    message: str

def main(argv: List[str], stdin: str | None = None) -> str:
    argv = [arg for arg in argv if arg != "-q"]
    if len(argv) == 1:
        raise CliArgException(help)
    if argv[1] == "units":
        suite = unittest.TestSuite()
        classes = [DetermineLoadsToValueTests]
        for klass in classes:
            for method in dir(klass):
                if method.startswith("test_"):
                    suite.addTest(klass(method))
        runner = unittest.TextTestRunner()
        runner.run(suite)
        return ""
    mode: Mode = "compile"
    if len(argv) >= 2 and argv[1] == "lex":
        mode = "lex"
        path = argv[2] if len(argv) > 2 else "-"
    elif len(argv) >= 2 and argv[1] == "parse":
        mode = "parse"
        path = argv[2] if len(argv) > 2 else "-"
    elif len(argv) > 2 and argv[1] == "check":
        mode = "check"
        path = argv[2]
    elif len(argv) > 2 and argv[1] == "monomize":
        mode = "monomize"
        path = argv[2]
    elif len(argv) > 2 and argv[1] == "compile":
        mode = "compile"
        path = argv[2]
    elif len(argv) > 2 and argv[1] == "inference-tree":
        mode = "inference-tree"
        path = argv[2]
    else:
        path = argv[1]
    return run(path, mode, "--guard-stack" in argv, stdin)

if __name__ == "__main__":
    try:
        print(main(sys.argv))
    except CliArgException as e:
        print(e.message, file=sys.stderr)
        exit(1)
    except ParserException as e:
        print(e.display(), file=sys.stderr)
        exit(1)
    except ResolverException as e:
        print(e.display(), file=sys.stderr)
        exit(1)
