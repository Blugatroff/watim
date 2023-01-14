# Watim 

Watim is a simple, low level, stack-based language which compiles to [Webassembly Text Format (WAT)](https://developer.mozilla.org/en-US/docs/WebAssembly/Understanding_the_text_format).
Which can then be compiled to wasm and run in your favorite browser or by runtimes like [wasmtime](https://github.com/bytecodealliance/wasmtime) and [wasm3](https://github.com/wasm3/wasm3).

The Watim compiler is written in Watim.

This project was inspired by [Porth](https://gitlab.com/tsoding/porth).

Watim = WAT Improved

## Features
- structs
- inline string literals
- loop expression with `break` keyword
- if expression
- static type checking
- terse syntax
- modules

## TODO
- `continue` instruction for loops.
- Variable declaration and initiliziation anywhere in function.
- Generics using monomorphization.
- Some sort of Typeclass/Trait system perhaps?
- function pointers

## How to run
First [install Wasmtime](https://wasmtime.dev/).

Then compile:
```bash
wasmtime --dir=. ./watim.wasm <watim-source-file> > out.wat
```

Then run:
```bash
wasmtime --dir=. ./out.wat [args]...
```

Or just use the provided script:
```bash
./run.sh <watim-source-file> [args]...
```

For example:
```bash
./run.sh ./samples/cat.watim ./README.md
```

## Editor Support
VIM syntax highlighting in [./editor/watim.vim](https://github.com/Blugatroff/watim/tree/main/editor/watim.vim)

## Example Program
This program exits with the exit code read from stdin.
```
// import standard WASI functions, see the docs: https://github.com/WebAssembly/WASI/blob/main/phases/snapshot/docs.md
extern "wasi_unstable" "fd_read" fn raw_read(file: i32, iovs: .Iov, iovs_count: i32, written: .i32) -> i32
extern "wasi_unstable" "fd_write" fn raw_write(file: i32, iovs: .Iov, iovs_count: i32, written: .i32) -> i32
extern "wasi_unstable" "proc_exit" fn exit(code: i32)

fn main "_start" () {
    // allocate 32 bytes on the stack
    memory buf: i32 32
    local nread: i32
    // read up to 32 bytes from file descriptor 0 (STDIN) into `buf`
    0 $buf 32 read #nread
    // parse the input to an i32
    $buf $nread 1 - parse
    // duplicate the parsed number on the stack
    dup 
    // print the parsed number to the console
    print
    // and use it as the exit code
    exit
}

fn print(n: i32) {
    // allocate 16 bytes for the string, 
    // a 32 bit number will always have less than 16 digits in decimal.
    memory buf: i32 16
    memory buf-reversed: i32 16
    local l: i32
    local i: i32
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
    0 #i
    loop {
        $buf-reversed $i +
        $buf $l 1 - $i - + load8
        store8
        $i 1 + #i
        $i $l = if { break }
    }
    1 $buf-reversed $l write drop
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
            1 "\n" write drop
            1 10 write_byte
            1 exit
        }
        $pt 1 + #pt // advance pointer
        $len 1 - #len // reduce length
        $len 0 = if { $n break }
        $n 10 * #n
    }
}

struct I32 { inner: i32 }

struct Iov {
    ptr: .i32
    len: i32
}

fn read(file: i32, buf-addr: .i32, buf-size: i32) -> i32 {
    local iov: Iov
    local nread: I32
    $buf-addr #iov.ptr
    $buf-size #iov.len
    $file &iov 1 &nread.inner raw_read drop
    $nread.inner
}

fn write(file: i32, ptr: .i32, len: i32) -> i32 {
    local iov: Iov
    local written-ptr: I32
    local written: i32
    $ptr #iov.ptr
    $len #iov.len
    $file &iov 1 &written-ptr.inner raw_write drop
    $written-ptr.inner #written
    $written $len = if {
        $len
    } else {
        $file $ptr $written + $len $written - write $written +
    }
}

fn write_byte(file: i32, b: i32) {
    memory buf: i32 1
    $buf $b store8
    $file $buf 1 write drop
}

fn dup(a: i32) -> i32, i32 {
    $a $a
}
```
