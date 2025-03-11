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