import struct
from typing import NamedTuple
from neutrino import TraceHeader, TraceSection
import sys

class block_sched(NamedTuple):
    start: int
    elapsed: int
    cuid: int

def parse(path: str) -> tuple[TraceHeader, list[TraceSection], dict[str, list[list[NamedTuple]]]]:
    with open(path, "rb") as f:
        header: TraceHeader = TraceHeader(*struct.unpack("iiiiiiii", f.read(32)))
        sections: list[TraceSection] = []
        for _ in range(header.numProbes):
            sections.append(TraceSection(*struct.unpack("IIQ", f.read(16))))
        gridSize = header.gridDimX * header.gridDimY * header.gridDimZ
        blockSize = header.blockDimX * header.blockDimY * header.blockDimZ
        records: dict[str, list[list[NamedTuple]]] = dict()

        # Read block_sched
        for i in range(gridSize):
            records["block_sched"].append([])
            for j in range(blockSize // sections[0].warpDiv):
                records["block_sched"][-1].append([])
                for k in range(sections[0].size // 16):
                    records["block_sched"][i][j].append(block_sched(*struct.unpack("qII", f.read(16))))

    return header, sections, records
# END of Neutrino Auto-Generated Code for Trace Reading

filename = sys.argv[1]
header,sections, records = parse(filename)
print("TraceHeader(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, numProbes): {}".format(header))
gridSize = header.gridDimX * header.gridDimY * header.gridDimZ
blockSize = header.blockDimX * header.blockDimY * header.blockDimZ
for i in range(gridSize):
    for j in range(blockSize // sections[0].warpDiv):
        for k in range(sections[0].size // 16):
            elem =records["block_sched"][i][j][k]
            print(f"gridID: {i} groupIDinBlock: {j} warpIDingroup: {k} : start - {elem.start}, elapsed - {elem.elapsed}, cuid - {elem.cuid}")