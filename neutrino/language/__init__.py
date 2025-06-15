"""Neutrino Trace Language Primitive"""
from functools import wraps
from typing import TypeAlias

def _disable_execution_(func):
    """Block function from execution, mainly because we want these primitives
    to be compiled to target assembly instead of running by Python"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise RuntimeError(f"{func.__name__} shall be jit other than run")
    return wrapper

# NOTE Neutrino Language's type system is incomplete and tiny because they're 
# placeholders for compilers instead of functional code for Python.
u32: TypeAlias = int
u64: TypeAlias = int
reg: TypeAlias = int

TYPES = ["u32", "u64"]
FUNCS = ["smid", "time", "clock", "save"]

@_disable_execution_
def smid() -> u32: ...

@_disable_execution_
def time() -> u64: ...

@_disable_execution_
def clock() -> u64: ...

@_disable_execution_
def save(regs: list[reg], dtype) -> None: ...

# @_disable_execution_
# def tid() -> None: ...

# @_disable_execution_
# def pid() -> None: ...

# Following are helpers for parsing register operands
src: reg = ...
dst: reg = ...
out: reg = ...
in1: reg = ...
in2: reg = ...
in3: reg = ...
in4: reg = ...