# Watim 

Watim is a simple stack-based language which compiles to [Webassembly Text Format (WAT)](https://developer.mozilla.org/en-US/docs/WebAssembly/Understanding_the_text_format).
Which can then be compiled to wasm and run in your favorite browser or by runtimes like [wasmtime](https://github.com/bytecodealliance/wasmtime) and [wasm3](https://github.com/wasm3/wasm3).

The goal is to eventually rewrite the Watim compiler in Watim to make it self-hosted.

This project is inspired by [Porth](https://gitlab.com/tsoding/porth).

Watim = WAT Improved

## Features added to WAT include
- inline string literals
- function local memory using the **memory** keyword
- more intuitive loop construct with **break** keyword
- bool type
- static type checking

## How to run
First [install wasm3](https://github.com/wasm3/wasm3/blob/main/docs/Installation.md).

Then run:
```bash
./run.sh <source-file>
```

## Editor Support
VIM syntax highlighting in [./editor/watim.vim](https://github.com/Blugatroff/watim/tree/main/editor/watim.vim)

## Example Program
This program exits with the exit code read from stdin.
```
extern "wasi_unstable" "proc_exit" fn exit(code: i32)
extern "wasi_unstable" "fd_read" fn raw_read(file: i32, iovs: i32, iovs_count: i32, result: i32) -> i32
extern "wasi_unstable" "fd_write" fn raw_write(file: i32, ptr: i32, len: i32, nwritten: i32) -> i32

fn read(file: i32, buf_addr: i32, buf_size: i32) -> i32 {
    memory space 8 4;
    $space $buf_addr store32 
    $space 4 + $buf_size store32 
    $file $space 1 $space raw_read drop
    $space load32
}

fn write(file: i32, ptr: i32, len: i32) -> i32 {
    memory space 8 4;
    $space $ptr store32
    $space 4 + $len store32
    $file $space 1 $space raw_write
}

fn print(n: i32) {
    memory buf 16;
    memory buf_reversed 16;
    local l: i32
    local i: i32
    1 100 7 write drop
    0 #l
    $n 0 = if {
        1 #l // length = 1
        $buf 48 store8 // put '0' in buf
    } else {
        loop {
            $n 0 = if { break }
            $buf $l +
            $n 10 % // rightmost digit
            48 + // + ascii 'a'
            store8
            $n 10 / #n // shift right in decimal
            $l 1 + #l
        }
    }
    $l #i
    loop {
        $buf_reversed $l 1 - $i - + $buf $i + load32 store32
        $i 0 = if { break }
        $i 1 - #i
    }
    $buf_reversed $l + 10 store8
    1 $buf_reversed $l 1 + write drop
}

fn parse(ptr: i32, len: i32) -> i32 {
    local n: i32
    local d: i32
    local original-ptr: i32
    local original-len: i32
    $ptr #original-ptr
    $len #original-len
    loop {
        $ptr load8 #d
        $d 48 >= $d 58 <= and if { // 48 is ascii '0'
            $n $d 48 - + #n            
        } else {
            1 "Failed to parse: '" write drop
            1 $original-ptr $original-len write drop
            1 "'\n" write drop
            1 exit
        }
        $ptr 1 + #ptr // advance pointer
        $len 1 - #len // reduce length
        $len 0 = if { $n break }
        $n 10 * #n
    }
}

fn dup(a: i32) -> i32, i32 {
    $a $a
}

fn main "_start" () {
    memory buf 32;
    local nread: i32
    0 $buf 32 read #nread
    $buf $nread 1 - parse 
    dup print exit
}
```
