"""Generate the CUDA PTX Assembly, a C-style asm"""

from typing import Optional
from dataclasses import dataclass

@dataclass
class Register:
    name: str
    dtype: str
    init: int

@dataclass
class Probe:
    name:   str                   # name is the key in TOML
    level:  str                   # level of the probe
    pos:    list[str]             # := tracepoint in the paper
    size:   Optional[int] = 0     # number of bytes per thread
    before: Optional[list] = None # snippet inserted before, one of before and after shall be given
    after:  Optional[list] = None # snippet inserted after,  one of before and after shall be given

def cvt_inst(inst: list[str]) -> str:
    match inst[0]:
        # ALU Instructions
        case "add":
            return f"add.u64 %{inst[1]}, %{inst[2]}, %{inst[3]};"
        case "sub":
            return f"sub.u64 %{inst[1]}, %{inst[2]}, %{inst[3]};"
        case "mul":
            return f"mul.u64 %{inst[1]}, %{inst[2]}, %{inst[3]};"
        case "div":
            return f"div.u64 %{inst[1]}, %{inst[2]}, %{inst[3]};"
        case "mod":
            return f"rem.u64 %{inst[1]}, %{inst[2]}, %{inst[3]};"
        case "lsh":
            return f"shl.u64 %{inst[1]}, %{inst[2]}, %{inst[3]};"
        case "rsh":
            return f"shr.u64 %{inst[1]}, %{inst[2]}, %{inst[3]};"
        # case "and":
        #     return f"and.u64 {inst[1]}, {inst[2]}, {inst[3]};"
        # case "and":
        #     return
        # case "or":
        #     return
        # case "xor":
        #     return
        # Memory Instructions
        case "stw":
            return f"SAVE.u32 {inst[1]}"
        case "stdw":
            return f"SAVE.u64 {inst[1]}"
        # Other Instructions
        case "mov":
            return f"mov.u64 %{inst[1]}, %{inst[2]};"
        case "clock":
            return f"mov.u64 %{inst[1]}, %clock64;"
        case "time":
            return f"mov.u64 %{inst[1]}, %globaltimer;"
        case "smid":
            return f"mov.u64 %{inst[1]}, %smid;"
        case _:
            raise NotImplementedError(f"{inst} not yet supported")

def gencode(probes: list[Probe]) -> list[Probe]:
    # First handle the initialization of regs

    # Then handle the syntax of probes
    for probe in probes:
        # only change the instructions, i.e., before and after part
        if probe.before is not None:
            insts: list[str] = []
            for inst in probe.before:
                insts.append(cvt_inst(inst))
            probe.before = "\n".join(insts)
        elif probe.after is not None:
            insts: list[str] = []
            for inst in probe.after:
                insts.append(cvt_inst(inst))
            probe.after = "\n".join(insts)
    
    return probes

if __name__ == "__main__":
    probes = [Probe(name='block_sched_start', level='warp', pos='kernel', size=0, before=None, after=[['clock', 'R0']]), Probe(name='block_sched_end', level='warp', pos='kernel', size=16, before=[['clock', 'R1'], ['sub', 'R2', 'R1', 'R0'], ['stdw', 'R0'], ['smid', 'R3'], ['stw', 'R2'], ['stw', 'R3']], after=None)]
    print(gencode(probes))
