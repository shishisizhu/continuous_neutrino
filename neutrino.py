"""System Entry for NEUTRINO: Fine-grained GPU Kernel Profiling via Programmable Probing

The system interface is similar to `strace` like: neutrino -t/-p <command>"""

import subprocess, os, sys, toml
from typing import Dict

# directory of this python file and other toolkits
tool_dir = os.path.dirname(os.path.realpath(__file__))

config = toml.load(os.path.join(tool_dir, "config.toml")) # read the config.toml

# default configurations, can be overwritten by CLI parameters
NEUTRINO_HOOK_CUDA_LIB = "libcuda.so.1" # don't change this, usually refer to libcuda.so.1 not libcuda.so
NEUTRINO_REAL_CUDA_LIB = "libcuda.so.550.54.15"
# Directory of the real cuda lib
NEUTRINO_CUDA_LIB_DIR: str = "/usr/lib/x86_64-linux-gnu"
# same as this executable
NEUTRINO_PYTHON: str = sys.executable # default to be this executable
# directory to put the trace
NEUTRINO_TRACEDIR: str = "./trace"
# filter of kernel
NEUTRINO_FILTER: str= ""
# available built-in tools
NEUTRINO_TOOLS: Dict[str, str] = config["tools"]
# Benchmark mode, will include an additional launch after the trace kernel
# Used to measure the kernel-level slowdown of Neutrino, disabled by default
NEUTRINO_BENCHMARK: str = "0"
# info string to print if failed
NEUTRINO_INFO_STRING = f"""usage: gpumemtrace.py [-t/-p] [--benchmark] command

note:
  one of --tool (-t) or --probe (-p) must be given

options:
  --tool -t   tool of neutrino                   required, available: {",".join(NEUTRINO_TOOLS.keys())}
  --probe -p  probe for neutrino in toml         required, file name to toml
  --trace-dir specify where to put trace,        default: {NEUTRINO_TRACEDIR} (inside workdir)
  --cuda-dir  specify directory of libcuda.so.1, default: {NEUTRINO_CUDA_LIB_DIR}
  --cuda-name name of cuda driver shared lib,    default: {NEUTRINO_HOOK_CUDA_LIB}
  --python    path to python executable          default: {NEUTRINO_PYTHON}
  --filter    filter out buggy kernels by name   default: {NEUTRINO_FILTER}
  --benchmark run original kernel again to measure Neutrino overhead, default: disabled
  --help      print help message"""

if sys.argv[1] == "--help":
  print(NEUTRINO_INFO_STRING)
  exit(0)

# command to be executed
command: str = ""
# one of following is required
neutrino_tool = ""
neutrino_probe = ""

# a manual argparser as we want all following argument integrated
i: int = 1
while i < len(sys.argv): # 1st argument is name of script, ignored
  if sys.argv[i] == "-p" or sys.argv[i] == "--probe":
    neutrino_probe = sys.argv[i + 1]
    i += 2
  if sys.argv[i] == "-t" or sys.argv[i] == "--tool":
    neutrino_tool = sys.argv[i + 1]
    i += 2
  if sys.argv[i] == "--trace-dir":
    NEUTRINO_TRACEDIR = sys.argv[i + 1]
    i += 2
  elif sys.argv[i] == "--cuda-dir":
    NEUTRINO_CUDA_LIB_DIR = sys.argv[i + 1]
    i += 2
  elif sys.argv[i] == "--cuda-name":
    NEUTRINO_REAL_CUDA_LIB = sys.argv[i + 1]
    i += 2
  elif sys.argv[i] == "--python":
    NEUTRINO_PYTHON = sys.argv[i + 1]
    i += 2
  elif sys.argv[i] == "--filter":
    NEUTRINO_FILTER = sys.argv[i + 1]
    i += 2
  elif sys.argv[i] == "--benchmark": 
    NEUTRINO_BENCHMARK = "1"
    i += 1
  else:
    command = sys.argv[i:]
    break

# try to load the probe from file or available toolkit
if len(neutrino_probe) == 0:
  # check if neutrino_tool is given
  if len(neutrino_tool) == 0 or neutrino_tool not in NEUTRINO_TOOLS:
    print(NEUTRINO_INFO_STRING)
    exit(0)
  else:
    neutrino_probe = toml.load(os.path.join(tool_dir, NEUTRINO_TOOLS[neutrino_tool]))
else:
  neutrino_probe = toml.load(neutrino_probe)


# a copied environment variables
env = os.environ.copy()
# configure gpumemtrace related environment variables
env["NEUTRINO_REAL_CUDA"]  = os.path.join(NEUTRINO_CUDA_LIB_DIR, NEUTRINO_REAL_CUDA_LIB)
env["NEUTRINO_HOOK_CUDA"]  = os.path.join(NEUTRINO_CUDA_LIB_DIR, NEUTRINO_HOOK_CUDA_LIB)
env["NEUTRINO_PYTHON"]     = NEUTRINO_PYTHON
env["NEUTRINO_PROCESS_PY"] = os.path.join(tool_dir, "build", "process.py")
env["NEUTRINO_FILTER"]     = NEUTRINO_FILTER
env["NEUTRINO_TRACEDIR"]   = NEUTRINO_TRACEDIR
env["NEUTRINO_PROBES"]     = toml.dumps(neutrino_probe) # dump it to string
# GNU LD_PRELOAD to overwrite dlopen, https://man7.org/linux/man-pages/man8/ld.so.8.html
env["LD_PRELOAD"]          = os.path.join(tool_dir, "build","preload.so")
# An Environmental Variable to enable the trace
env["NEUTRINO_ENABLE"] = "1"
# An Environmental Variable to enable the benchmark mode
env["NEUTRINO_BENCHMARK"] = NEUTRINO_BENCHMARK
# An Environmental Variables to enable the debug mode -> more messages
# env["NEUTRINO_VERBOSE"] = "1"

# start the program with new environment
if len(command) > 0:
  proc = subprocess.Popen(command, env=env)
  proc.wait()