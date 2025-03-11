import os
import sys
import subprocess
from pprint import pprint

# install toml
import toml

CURDIR = os.path.dirname(os.path.realpath(__file__))

# Internal Configurations
SRC_DIR = os.path.join(CURDIR, "src")
BUILD_DIR = os.path.join(CURDIR, "build")
CC = "cc"
PY = sys.executable
NEUTRINO_CUDA_HEADER_NAME = "cuda.h"
NEUTRINO_HOOK_CUDA_LIB_NAME = "libcuda.so.1"

# NOTE detect CUDA Headers
NEUTRINO_CUDA_HEADER_SEARCH_PATH = [
  "/usr/local/cuda/targets/x86_64-linux/include/", # for x86
  "/usr/local/cuda/targets/aarch64-linux/include/", # for ARM
  # add if missed
]

for dir_ in NEUTRINO_CUDA_HEADER_SEARCH_PATH:
  try:
    if NEUTRINO_CUDA_HEADER_NAME in os.listdir(dir_):
      break
  except:
    pass
# NOTE this will be written in config.toml
NEUTRINO_CUDA_HEADER_DIR = dir_

# NOTE Locate CUDA Shared Library
# inspired by: https://github.com/triton-lang/triton/commit/58c54455ffa691be64f90f4e856501162373572c#diff-3d1f29795218f61553ab953426c15fa1e4162b224405b85529022293054da57aR25
# but we need to further locate the real cuda library
libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
if env_ld_library_path and not locs:
  locs = [os.path.join(dir_, "libcuda.so.1") for dir_ in env_ld_library_path.split(":")
          if os.path.exists(os.path.join(dir_, "libcuda.so.1"))]

# try to locate the pointed path
NEUTRINO_REAL_CUDA_LIB_NAME = ""
NEUTRINO_REAL_CUDA_LIB_DIR = ""
real_libs = []
for loc in locs:
  real_lib = os.readlink(loc)
  if "lib32" not in loc and "lib32" not in real_lib:
    if not real_lib.startswith("/"):
      NEUTRINO_REAL_CUDA_LIB_DIR = os.path.dirname(loc)
      NEUTRINO_REAL_CUDA_LIB_NAME = real_lib
    else:
      NEUTRINO_REAL_CUDA_LIB_DIR = os.path.dirname(loc)
      NEUTRINO_REAL_CUDA_LIB_NAME = os.path.basename(loc)

print(NEUTRINO_REAL_CUDA_LIB_DIR, NEUTRINO_REAL_CUDA_LIB_NAME, file=sys.stderr)

# NOTE call parse.py
cmd = [PY, os.path.join(SRC_DIR, "parse.py"), 
    os.path.join(NEUTRINO_CUDA_HEADER_DIR, NEUTRINO_CUDA_HEADER_NAME), 
    os.path.join(NEUTRINO_REAL_CUDA_LIB_DIR, NEUTRINO_REAL_CUDA_LIB_NAME)]
print(" ".join(cmd), file=sys.stderr)
subprocess.check_output(cmd)

# NOTE call compile.py
cmd = [CC, os.path.join(SRC_DIR, "modified.c"), "-fPIC", "-shared", "-ldl", "-O3", 
        "-I", NEUTRINO_CUDA_HEADER_DIR, "-o", os.path.join(BUILD_DIR, NEUTRINO_HOOK_CUDA_LIB_NAME)]
print(" ".join(cmd), file=sys.stderr)
subprocess.check_output(cmd)

# NOTE call preload.c
cmd = [CC, os.path.join(SRC_DIR, "preload.c"), "-fPIC", "-shared", "-O3", 
       "-o", os.path.join(BUILD_DIR, "preload.so")]
print(" ".join(cmd), file=sys.stderr)
subprocess.check_output(cmd)

# NOTE create a symbolic link of libcuda.so
cmd = ["ln", "-sf", "libcuda.so.1", os.path.join(BUILD_DIR, "libcuda.so")]
print(" ".join(cmd), file=sys.stderr)
subprocess.check_output(cmd)

# NOTE write things into config.toml in 'system' section
config = {}
config["system"] = {
  "NEUTRINO_CUDA_HEADER_NAME"   : NEUTRINO_CUDA_HEADER_NAME,
  "NEUTRINO_CUDA_HEADER_DIR"    : NEUTRINO_CUDA_HEADER_DIR,
  "NEUTRINO_HOOK_CUDA_LIB_NAME" : NEUTRINO_HOOK_CUDA_LIB_NAME,
  "NEUTRINO_REAL_CUDA_LIB_NAME" : NEUTRINO_REAL_CUDA_LIB_NAME,
  "NEUTRINO_REAL_CUDA_LIB_DIR"  : NEUTRINO_REAL_CUDA_LIB_DIR,
}
toml.dump(config, open(os.path.join(BUILD_DIR, "config.toml"), "w"))

print("Build Success, Configuration")
print("============================")
pprint(config)