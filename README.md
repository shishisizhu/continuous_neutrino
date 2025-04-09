# Development Branch for Neutrino Tracing Language

> NOTE: Still under development, not yet ready to use, even as alpha

A high-level tracing interface like [bpftrace](https://bpftrace.org) or [dtrace](https://dtrace.org/) but trageting cross-platform compatibility.

## Motivation

Currently Neutrino supports directly programming assembly.
However, writing assembly is non-trivial for developers, and assembly for profiling is significantly different from computing.
Neutrino Trace Language envisions a high-level interface through Python that allows you quickly customize probes in your familiar language.

Moreover, a tracing language can enhance the hardware independence of the probe itself. 
Notably, as current probes are written in assembly, a low-level and hardware-dependent language, they're hardware-dependent.
Thus, a high-level tracing language could help bridge this gap and enhance our claim of hardware independence.

However, it's exepcted that Neutrino Tracing Language would support less features than directly programming the assembly, but it will be a good start for writing probes in assembly.

## Example
We envision an example like follows:

```python
import neutrino
import neutrino.language as nl # API borrowed from Triton :)

gstart : nl.u64 = 0
gend   : nl.u64 = 0
elapsed: nl.u64 = 0

@nl.probe(pos="kernel", level="warp") # broadcast to warp leader
def block_start():
    gstart = nl.time()

@nl.probe(pos="kernel", after=True, level="warp", size=16) # save 16 bytes per warp
def block_sched():
    gend = nl.time()
    elapsed = gend - gstart 
    nl.save(gstart, dtype=nl.u64)
    nl.save((elapsed, nl.smid()), dtype=nl.u32) # auto casted
```

This example shall be JIT transformed into platform-specific assembly like PTX:
```toml
[block_sched]
position = "kernel"   # kernel-level probe
datamodel = "warp:16" # every warp save 16bytes
before = """.reg .b64 %gstart;// global start (ns)
.reg .b64 %gend;    // global end time (ns)
.reg .b64 %elapsed; // thread elapsed time in u64
.reg .b32 %elapse;  // thread elapsed time in u32
mov.u64 %gstart, %globaltimer;"""
after = """mov.u64 %gend, %globaltimer;
sub.u64 %elapsed, %gend, %gstart;
cvt.u32.u64 %elapse, %elapsed;
SAVE.u64 {%gstart}; // SAVE is Neutrino extension
SAVE.u32 {%elapse, %smid}; // save time & core"""
```

and GCNAsm:
```
TODO
```

## Design
We propose a minimalistic workflow:
1. Python program will be **parsed** into syntax tree via Python `ast` module.
2. Syntax tree will be flattened into list of `Assign`, `BinOp`, `Call` via `ast.NodeVisitor`.
3. Flattened Syntax tree will be translated into corresponding assmebly like PTX/GCNAsm.

> NOTE: This program will not be executed by Python, only parsed and transformed.

## Limitations
Due to the unique characteristic of GPU, we can not support full Python syntax, following are addressed backups:
1. Function calls are limited to `neutrino.language`'s function call, currently only `time()`, `clock()`, `save()`, `smid()`.
2. Operations that might be transformed to `jmp` are not allowed, such as `for/while` statement.
3. Variables are strictly typed and must be declared globally via `var: tl.u64 = 0` statement given the dtype (currently only `u32` and `u64`) and the initial value (mostly 0).

## Roadmap
* Consumption (`neutrino/language/main.py`)
    - [ ] Build a CLI entry like `neutrino compile` or `neutrino-compile`
* Language Primitive (`neutrino/langauge/__init__.py`)
    - [x] Helper operands: `src/dst/out/in1/in2/in3/in4`
    - [x] Helper functions: `time()`, `clock()`, `save()`, `smid()`.
* Parser and Transformation (`neutrino/langauge/frontend.py`)
    - [x] Support register declaration and initial value
    - [x] Support binary and unary operator parsing
    - [x] Support function call parsing
    - [ ] Support `if-else` statement for flexible profiling (`if` is implememted via execution mask other than `jmp`)
* Backend
    * NVIDIA PTX Backend (`neutrino/langauge/cuda.py`)
        - [ ] Declare register and assign initial value
        - [ ] Basic Arithemetic
        - [ ] Primitive function calling
    * AMD GCNAsm Backend (`neutrino/langauge/rocm.py`)
        - [ ] Declare register and assign initial value
        - [ ] Basic Arithemetic
        - [ ] Primitive function calling
