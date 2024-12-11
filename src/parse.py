""" Code Generator for Unmodified CUDA Driver Functions and Symbols 
based on parsing the cuda.h and libcuda.so

NOTE How it works?
All CUDA Driver symbols are exposed via <cuda.h> for link and documentation, 
with signatures like:

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev);

By parsing this symbol allow us to get and generate a mask driver like:
```c
CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    if (shared_lib == NULL)
        init();
    // code insert here := eBPF uprobe
    CUresult (*real)(char*, int, CUdevice) = dlsym(shared_lib, "cuDeviceGetName");
    CHECK_DL(); // checking if any dl error presented
    CUresult ret = real(name, len, dev);
    // code insert here := eBPF uretprobe
    return ret;
}
```

But CUDA Symbols might be versioned slightly, like cuMemAlloc now has two
symbol cuMemAlloc (actually a macro) and cuMemAlloc_v2(real but not in cuda.h)
so we need to check libcuda.so to compromise the missed symbol (assume the
signature identical to unversioned.

NOTE You may see warnings functions such as lacking symbols for 
cuGL, cudbg, cuEGL, cuMem, cuProfiler, cuVDP, but they can be ignored mostly
"""
from typing import NamedTuple, List, Tuple
import sys
import os
import subprocess

# NOTE list of modified symbols -> handled by modified.c unless code generation
MODIFIED_FUCTIONS: List[str] = ["cuMemAlloc_v2", "cuMemFree_v2", "cuMemcpy_v2", 
    "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2", "cuModuleLoadData", "cuModuleGetFunction",
    "cuKernelGetFunction", "cuLibraryGetKernel", "cuLibraryGetModule", 
    "cuLibraryLoadData", "cuLaunchKernel", "cuMemsetD32_v2", "cuGetProcAddress_v2", 
    "cuGetProcAddress", "cuDeviceGetAttribute", "cuGetExportTable", "cuModuleLoadDataEx",
    "cuMemAllocHost_v2", "cuGetErrorName", "cuModuleLoad", "cuModuleLoadFatBinary", 
    "cuLaunchKernelEx", "cuStreamSynchronize"]

# TODO following is default configuration for backup, 
# real configuration is given via CLI from Makefile
CUDA_LIB_PATH = "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
CUDA_HEADER_PATH = "/usr/local/cuda/targets/x86_64-linux/include/cuda.h"

UNMODIFIED_C_NAME = "unmodified.c"
SIGNATURE_C_NAME  = "signature.c"

# Template for codegen
CODEGEN_TEMPLATE = """
CUresult {func_name}({param_list}) {{
    if (real_{func_name} == NULL)
        init();
    
    CUresult retval = real_{func_name}({param_val_list}); // call the symbol

    if (VERBOSE) // print function name and return value
        fprintf(log, "[info] {func_name} %d\\n", retval);
    
    return retval;
}}
"""

SIGNATURE_TEMPLATE = 'CUresult (*real_{func_name})({param_list}) = NULL;'
INIT_TEMPLATE = '    real_{func_name} = dlsym(shared_lib, "{func_name}");'


class Parameter(NamedTuple):
    type_name: str
    var_name: str

class Signature(NamedTuple):
    func_name: str
    params: List[Parameter]

class VersionedSymbol(NamedTuple):
    name: str
    version: str

def parse_parameter(param: str) -> Parameter:
    # Split the parameter into type and name
    param_parts = param.rsplit(' ', 1)
    if len(param_parts) == 2:
        type_name = param_parts[0].strip()
        var_name: str = param_parts[1].strip()
        if var_name.startswith("*"):
            num_star = var_name.rfind("*") + 1
            type_name = type_name + "*" * num_star
            var_name = var_name[num_star:]
    else:
        type_name = param_parts[0].strip()
        var_name = ''
    
    return Parameter(type_name, var_name)

def parse_function_signature(signature: str) -> Signature:
    # Remove the trailing semicolon
    signature = signature.strip().rstrip(';\n')
    
    # Find the opening parenthesis for parameters
    paren_index = signature.find('(')
    space_index = signature.rfind(' ', 0, paren_index)
    func_name = signature[space_index:paren_index].strip()
    if "\n" in func_name:
        space_index = func_name.rfind('\n')
        func_name = func_name[space_index + 1:]

    # Extract parameters
    params_str = signature[paren_index + 1:].strip()
    params_str = params_str[:-1].strip()  # Remove closing parenthesis

    # Parse parameters
    param_list: List[Parameter] = []
    if params_str:
        # Split by commas, considering pointer types
        param_parts = []
        current_param = ''
        depth = 0
        
        for char in params_str:
            if char == ',' and depth == 0:
                param_parts.append(current_param.strip())
                current_param = ''
            else:
                current_param += char
                if char == '<':
                    depth += 1
                elif char == '>':
                    depth -= 1
            
        # Add the last parameter
        if current_param:
            param_parts.append(current_param.strip())

        for param in param_parts:
            param = param.strip()
            if param:
                param_list.append(parse_parameter(param))

    return Signature(func_name, param_list)

def parse_symbol(nm_line: str) -> str:
    if len(nm_line.strip()) != 0:
        return nm_line.rsplit(" ", 1)[1]
    else:
        return ""
    
def parse_version_symbol(symbol: str) -> Tuple[str, str]:
    if "_" in symbol:
        name, version = symbol.split("_", 1)
        return name, "_"+version
    else:
        return symbol, ""

def gencode(signature: Signature) -> str:
    param_list = []
    param_type_list = []
    param_val_list = []
    for param in signature.params:
        param_type_list.append(param.type_name)
        param_val_list.append(param.var_name)
        param_list.append(param.type_name + " " + param.var_name)
    return CODEGEN_TEMPLATE.format(
        func_name = signature.func_name,
        param_list = ", ".join(param_list),
        param_type_list = ", ".join(param_type_list),
        param_val_list = ", ".join(param_val_list)
    )

def gensignature(signature: Signature) -> str:
    param_list = []
    param_type_list = []
    param_val_list = []
    for param in signature.params:
        param_type_list.append(param.type_name)
        param_val_list.append(param.var_name)
        param_list.append(param.type_name + " " + param.var_name)
    return SIGNATURE_TEMPLATE.format(
        func_name = signature.func_name,
        param_list = ", ".join(param_list)
    )

def geninit(signature: Signature) -> str:
    return INIT_TEMPLATE.format(func_name = signature.func_name)

if __name__ == "__main__":
    # parse cli param if given, usage is python parse.py CUDA_HEADER_PATH, CUDA_LIB_PATH
    if len(sys.argv) >= 3:
        CUDA_HEADER_PATH = sys.argv[1]
        CUDA_LIB_PATH = sys.argv[2]
    elif len(sys.argv) == 2:
        CUDA_HEADER_PATH = sys.argv[1]

    unmodified_c = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), UNMODIFIED_C_NAME), "w")
    signature_c  = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), SIGNATURE_C_NAME),  "w")

    print(f"[INFO] use {CUDA_HEADER_PATH} and {CUDA_LIB_PATH}", file=sys.stderr)

    signatures: List[Signature] = []

    # parse cuda.h to extract cuda headers
    with open(CUDA_HEADER_PATH, "r") as cuda_header_file:
        headers = cuda_header_file.readlines()
        idx = 0
        start_idx, ending_idx = 0, 0
        for idx in range(len(headers)):
            if "#define CUDAAPI" in headers[idx]:
                start_idx = idx
            elif "CUDA API versioning support" in headers[idx]:
                ending_idx = idx
                break
        idx = start_idx
        # print(f"start: {start_idx}, end: {ending_idx}", file=sys.stderr)
        while idx < ending_idx:
            if "CUresult CUDAAPI" in headers[idx]:
                end_idx = idx + 1
                if ";" in headers[idx]: # a full signature
                    parsed_signature = parse_function_signature(headers[idx])
                    signatures.append(parsed_signature)
                else:
                    while ";" not in headers[end_idx]:
                        end_idx += 1
                    parsed_signature = parse_function_signature("".join(headers[idx:end_idx+1]))
                    signatures.append(parsed_signature)
                idx = end_idx
            else:
                idx += 1


    # extract missing symbols from libcuda.so
    cuda_symbol_list: List[str] = []
    result = subprocess.run(["nm", "-D", CUDA_LIB_PATH], stdout=subprocess.PIPE, text=True)
    cuda_log = result.stdout.split("\n")
    for line in cuda_log:
        symbol = parse_symbol(line)
        if symbol.startswith("cu"):
            cuda_symbol_list.append(symbol)
    
    # get the symbols missed in our cuda lib
    our_symbol_dict = {signature.func_name: signature for signature in signatures}
    missed_symbols = [symbol for symbol in cuda_symbol_list if symbol not in our_symbol_dict]
    print(f"[INFO] Extract {len(signatures)} Symbols from {CUDA_HEADER_PATH}", file=sys.stderr)

    for symbol in missed_symbols:
        # try to extract symbol and version
        raw_symbol_name, version = parse_version_symbol(symbol)
        # check if raw_symbol in
        if raw_symbol_name in our_symbol_dict:
            # versioned symbol share the same parameter list
            raw_symbol = our_symbol_dict[raw_symbol_name]
            signatures.append(Signature(func_name=symbol, params=raw_symbol.params))
        else:
            print(f"[WARNING] can't resolve {symbol}", file=sys.stderr)
    
    print(f"[INFO] Resolved {len(signatures)} Symbols from {len(cuda_symbol_list)} Symbols in {CUDA_LIB_PATH}", file=sys.stderr)

    print("// auto-generated by parse.py, used with modified.c", file=unmodified_c)
    print("// auto-generated by parse.py, used with modified.c", file=signature_c)
    inits = []
    # remove
    for signature in signatures:
        if signature.func_name not in MODIFIED_FUCTIONS:
            code = gencode(signature)
            print(code, file=unmodified_c)
            signautre_ = gensignature(signature)
            print(signautre_, file=signature_c)
            init_ = geninit(signature)
            inits.append(init_)
    
    print("\nstatic void init_unmodified(void) {", file=signature_c)
    print("\n".join(inits), file=signature_c)
    print("}", file=signature_c)