from typing import List, Callable, Sequence, Tuple, Any, Protocol, runtime_checkable, assert_never
from dataclasses import dataclass

from util import intersperse

class Indent:
    pass

@dataclass
class For[T]:
    items: Sequence[T]
    body: Callable[[T], List['FormatInstr']]

class WriteIndent:
    pass

type FormatInstr = None | bool | str | int | List[FormatInstr] | Indent | WriteIndent | For | Formattable

@runtime_checkable
class Formattable(Protocol):
    def format_instrs(self) -> List[FormatInstr]:
        return [type(self).__name__]
    def __str__(self) -> str:
        return format(self.format_instrs())

@dataclass
class Instrs(Formattable):
    instrs: List[FormatInstr]
    def format_instrs(self) -> List[FormatInstr]:
        return self.instrs

def format_instrs(item: Any) -> List[FormatInstr]:
    if isinstance(item, Formattable):
        return item.format_instrs()
    return [str(item)]

def unnamed_record(name: str, fields: List[FormatInstr]) -> List[FormatInstr]:
    return ["(", name, For(fields, lambda field: [" ", field]), ")"]

def named_record(name: str, fields: List[Tuple[str, FormatInstr]]) -> List[FormatInstr]:
    if len(fields) == 0:
        return ["(", name, ")"]
    return ["(", name, "\n",
            Indent(),
            list(intersperse(
                [",\n"],
                ([WriteIndent(), name, "=", value] for name,value in fields))),
            ")"]

def format_seq(elems: Sequence['Formattable'], multi_line=False) -> List[FormatInstr]:
    if len(elems) == 0:
        return ["[]"]
    return [
        "[",
        Indent() if multi_line else [],
        ["\n", WriteIndent()] if multi_line else [],
        elems[0],
        For(elems[1:], lambda elem: [",\n", WriteIndent(), elem.format_instrs()] if multi_line else [", ", elem]),
        "]"
    ]
def format_list[T: Formattable](elems: List[T], multi_line=False) -> List[FormatInstr]:
    return format_seq(elems, multi_line)

def format_dict[K, V](
        dictionary: dict,
        format_key: Callable[[K], List[FormatInstr]] = format_instrs,
        format_value: Callable[[V], List[FormatInstr]] = format_instrs) -> List[FormatInstr]:
    if len(dictionary) == 0:
        return ["(Map)"]

    return [
        "(Map\n",
        [
            Indent(),
            list(intersperse([",\n"], ([WriteIndent(), format_key(k), "=", format_value(v)] for k,v in dictionary.items()))),
        ],
        ")"
    ]

def format_str(s: str) -> List[FormatInstr]:
    return ["\"", s, "\""]

def format_optional[T](item: T | None, format_item: Callable[[T], List[FormatInstr]] = format_instrs) -> List[FormatInstr]:
    if item is None:
        return ["None"]
    return ["(Some ", format_item(item), ")"]

def format(instrs: List[FormatInstr], indent="  ") -> str:
    s = []
    level = 0
    def format(instrs: List[FormatInstr]) -> None:
        nonlocal level
        for instr in instrs:
            if isinstance(instr, str):
                s.append(instr)
                continue
            if isinstance(instr, int):
                s.append(str(instr))
                continue
            if isinstance(instr, bool):
                s.append(str(instr))
                continue
            if isinstance(instr, list):
                prev_level = level
                format(instr)
                level = prev_level
                continue
            if isinstance(instr, Formattable):
                format([instr.format_instrs()])
                continue
            if instr is None:
                s.append("None")
                continue
            if isinstance(instr, Indent):
                level += 1
                continue
            if isinstance(instr, WriteIndent):
                for i in range(level):
                    s.append(indent)
                continue
            if isinstance(instr, For):
                for item in instr.items:
                    format(instr.body(item))
                continue
            assert_never(instr)
    format(instrs)
    return ''.join(s)
