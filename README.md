# Watim 

Watim is a simple stack-based language which compiles to [Webassembly Text Format (WAT)](https://developer.mozilla.org/en-US/docs/WebAssembly/Understanding_the_text_format).
Which can then be compiled to wasm and run in your favorite browser or by runtimes like [wasmtime](https://github.com/bytecodealliance/wasmtime) and [wasm3](https://github.com/wasm3/wasm3).

The goal is to eventually rewrite the Watim compiler in Watim to make it self-hosted.

This project is inspired by [Porth](https://gitlab.com/tsoding/porth).

Watim = WAT Improved

## Features
- local variables
- inline string literals
- function local memory using the **memory** keyword
- loop expression with **break** keyword
- if expression
- static type checking

## How to run
First [install wasm3](https://github.com/wasm3/wasm3/blob/main/docs/Installation.md).

Then run:
```bash
./run.sh <source-file>
```

## Editor Support
VIM syntax highlighting in [./editor/watim.vim](https://github.com/Blugatroff/watim/tree/main/editor/watim.vim)

## Debugger
```bash
cargo run -- debug <source-file>
```

## Example Program
This program exits with the exit code read from stdin.
```
extern "wasi_unstable" "fd_read" fn raw_read(file: i32, iovs: .i32, iovs_count: i32, result: .i32) -> i32
extern "wasi_unstable" "fd_write" fn raw_write(file: i32, iovs: .i32, iovs_count: i32, nwritten: .i32) -> i32
extern "wasi_unstable" "proc_exit" fn exit(code: i32)

fn write(file: i32, pt: .i32, len: i32) -> i32 {
    memory space 8 4;
    local written: i32
    $space $pt !i32 store32
    $space 4 + $len store32
    $file $space 1 $space raw_write drop
    $space load32 #written
    $written $len = if {
        $len
    } else {
        $file $pt !i32 $written + !.i32 $len $written - write $written +
    }
}

fn read(file: i32, buf_addr: .i32, buf_size: i32) -> i32 {
    memory space 8 4;
    $space $buf_addr !i32 store32 
    $space !i32 4 + !.i32 $buf_size store32 
    $file $space 1 $space raw_read drop
    $space load32
}

fn print(n: i32) {
    memory buf 16;
    memory buf_reversed 16;
    local l: i32
    local i: i32
    0 #l
    $n 0 = if {
        1 #l // length = 1
        $buf 48 store8 // put '0' in buf
    } else {
        loop {
            $n 0 = if { break }
            $buf !i32 $l + !.i32
            $n 10 % // rightmost digit
            48 + // + ascii 'a'
            store8
            $n 10 / #n // shift right in decimal
            $l 1 + #l
        }
    }
    0 #i
    loop {
        $buf_reversed !i32 $i + !.i32
        $buf !i32 $l 1 - $i - + !.i32 load8
        store8
        $i 1 + #i
        $i $l = if { break }
    }
    1 $buf_reversed $l write drop
}

fn write_byte(file: i32, b: i32) {
    memory buf 1;
    $buf $b store8
    $file $buf 1 write drop
}

fn parse(pt: .i32, len: i32) -> i32 {
    local n: i32
    local d: i32
    local original-ptr: .i32
    local original-len: i32
    $pt #original-ptr
    $len #original-len
    loop {
        $pt load8 #d
        $d 48 >= $d 58 <= and if { // 48 is ascii '0'
            $n $d 48 - + #n
        } else {
            1 "Failed to parse: '" write drop
            1 $original-ptr $original-len write drop
            1 "'" write drop
            //1 "\n" write drop
            1 10 write_byte
            1 exit
        }
        $pt 1 + #pt // advance pointer
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
