# Neutrino Generated Code for Reading Trace
import struct
from typing import NamedTuple, List, Tuple
from neutrino import TraceHeader, TraceSection

class block_sched(NamedTuple):
    lstart: int
    elapse: int
    smid: int

def parse(path: str):
  with open(path, "rb") as f:
    header: TraceHeader = TraceHeader(*struct.unpack("iiiiiiii", f.read(32)))
    sections: List[TraceSection] = []
    for _ in range(header.numProbes):
        size, offset = struct.unpack("QQ", f.read(16))
        sections.append(TraceSection(size, offset))
    gridSize = header.gridDimX * header.gridDimY  * header.gridDimZ
    blockSize = header.blockDimX * header.blockDimY * header.blockDimZ
    records: List[List[block_sched]] = []
    for i in range(gridSize):
        records.append([])
        for j in range(blockSize // 32):
            lstart, elapse, smid = struct.unpack("QII", f.read(16))
            records[i].append(block_sched(lstart, elapse, smid))
  return header, sections, records
# END OF GENERATED CODE
import sys
import numpy as np
header, sections, records = parse(sys.argv[1]) # filled by path to trace

unique_sms = set()    
for block in records:
    unique_sms.add(block[0].smid)

sm_timelines = []
for _ in range(len(unique_sms)):
    sm_timelines.append([])
sched_times = [0.0] * len(unique_sms)
work_times = [0.0] * len(unique_sms)

for cur in records:
    # print(sm_timelines[cur[0].smid])
    sched_out = False
    smid = cur[0].smid
    if len(sm_timelines[smid]) > 0:
        for block in sm_timelines[smid]:
            if block.lstart + block.elapse <= cur[0].lstart:
                # if cur[0].lstart - (block.lstart + block.elapse) < 100000:
                #     print(cur[0], block)
                sched_times[smid] += cur[0].lstart - (block.lstart + block.elapse)
                sm_timelines[smid].remove(block)
                sm_timelines[smid].append(cur[0])
                work_times[smid] += cur[0].elapse
                sched_out = True
                break
            if not sched_out:
                sm_timelines[smid].append(cur[0]) 
                work_times[smid] += cur[0].elapse
                break
    else:
        sm_timelines[smid].append(cur[0]) 
        work_times[smid] += cur[0].elapse

print(f"No.block:{header.gridDimX * header.gridDimY  * header.gridDimZ} Running:{int(np.array(work_times).mean())} Scheduling:{int(np.array(sched_times).mean())}(cycle)")
