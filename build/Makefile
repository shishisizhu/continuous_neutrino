# NOTE: we are transiting to Python-based build system to auto-detect following terms
# But for now, please update this manually for your system
CC = gcc
PY = python
CUDA_HEADER_DIR = /usr/local/cuda/targets/x86_64-linux/include
CUDA_HEADER_NAME = cuda.h
CUDA_LIB_DIR = /usr/lib/x86_64-linux-gnu
CUDA_LIB_NAME = libcuda.so.1

# Tool-specific definition
SRC_DIR = ../src
BUILD_DIR = .

all: parse compile preload

parse:
	$(PY) $(SRC_DIR)/parse.py $(CUDA_HEADER_DIR)/$(CUDA_HEADER_NAME) $(CUDA_LIB_DIR)/$(CUDA_LIB_NAME)

compile:
	$(CC) $(SRC_DIR)/modified.c -fPIC -shared -ldl -O3 -o $(BUILD_DIR)/$(CUDA_LIB_NAME) -I$(CUDA_HEADER_DIR)

preload:
	$(CC) $(SRC_DIR)/preload.c -shared -fPIC -o $(BUILD_DIR)/preload.so -ldl
