from typing import NamedTuple

class TraceHeader(NamedTuple):
    gridDimX: int
    gridDimY: int
    gridDimZ: int
    blockDimX: int
    blockDimY: int
    blockDimZ: int
    sharedMemBytes: int
    numProbes: int

class TraceSection(NamedTuple):
    size:   int
    offset: int

def probe(pos: str, after: bool = False, level: str = "thread", size: int = 0):
    """Neutrino Probe Entry"""
    from functools import wraps
    # Just preventing the execution as we take it as part of AST only
    def inner(func: callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
          raise RuntimeError(f"{func.__name__} shall be jit other than run")
        return wrapper
    return inner
