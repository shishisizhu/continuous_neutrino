#include <time.h>    // for timing
#include <elf.h>     // for ELF Header
#include <stdint.h>  // for uint64_t defn
#include <stdio.h>   // for I/O
#include <stdbool.h> // for true false
#include <cuda.h>    // for cuda related
#include "uthash.h"  // for hashmap
#include "sha1.h"    // for SHA1 hash algorithm

// @note change to enum for better
#define PROBE_TYPE_THREAD 0
#define PROBE_TYPE_WARP 1
#define WARP_SIZE 32
#define CDIV(a,b) (a + b - 1) / (b)

static FILE* log; // file pointer to log:  NEUTRINO_TRACEDIR/MM_DD_HH_MM_SS/event.log

// Neutrino Trace Headers being dumped
// @todo add section table similar to ELF for faster parsing
typedef struct {
    // basic launch configuration
    uint32_t gridDimX;
    uint32_t gridDimY;
    uint32_t gridDimZ;
    uint32_t blockDimX;
    uint32_t blockDimY;
    uint32_t blockDimZ;
    uint32_t sharedMemBytes; 
    // all above from CUDA launch configuration
    uint32_t numProbes; // number of traces exposed
    // followed by an array of trace_section_t
} trace_header_t;

// Neutrino Trace Sections Headers being dumped
// @note check data model for more details here
typedef struct {
    uint64_t size;   // number of record, depends on datamodel
    uint64_t offset; // size of each record
} trace_section_t;

// binary header definitions

// CUBIN is standard ELF, FatBinary is special with headers below:

// fat binary header defined for fatbin
// @see https://github.com/rvbelapure/gpu-virtmem/blob/master/cudaFatBinary.h
typedef struct {
    unsigned int           magic;   // magic numbers, checked it before
    unsigned int           version; // fatbin version
    unsigned long long int size;    // fatbin size excluding 
} fatBinaryHeader;

// the fat binary header wrapper 
// @see fatbinary_section.h in cuda toolkit
typedef struct {
    int magic;
    int version;
    unsigned long long* data;  // pointer to real fatbin
    void *filename_or_fatbin;  /* version 1: offline filename,
                               * version 2: array of prelinked fatoutbuf */
} fatBinaryWrapper;

/**
 * Binary Size Calculation mainly because in C, code is given by void*
 * and we need to parse the header to get the size
 */

// check if content of void *ptr is ELF format or FatBinary Format
// return 1 if ELF, 2 if fatbin, and 0 otherwise
#define CUBIN 1
#define FATBIN 2
#define WRAPPED_FATBIN 3
#define PTX 4
#define ERROR_TYPE 0

static const char *code_types[] = { "error", "cubin", "fatbin", "warpped_fatbin", "ptx" };

static int check_magic(const int magic) {
    if (magic == 0x464c457f || magic == 0x7f454c46) {
        return CUBIN;
    } else if (magic == 0xba55ed50 || magic == 0x50ed55ba) {
        return FATBIN;
    } else if (magic == 0x466243B1 || magic == 0xB1436246) {
        return WRAPPED_FATBIN;
    } else {
        return ERROR_TYPE;
    }
}

static unsigned long long get_elf_size(const Elf64_Ehdr *header) {    
    // for standard executable, use section header
    size_t size = header->e_shoff + header->e_shentsize * header->e_shnum;

    // for cubin, only program header can give correct size
    if (header->e_phoff + header->e_phentsize * header->e_phnum > size)
        size = header->e_phoff + header->e_phentsize * header->e_phnum;

    return size;
}

static unsigned long long get_fatbin_size(const fatBinaryHeader *header) {
    // size of fatbin is given by header->size and don't forget sizeof header
    return header->size + sizeof(fatBinaryHeader); 
}

/**
 * Hash map (uthash) as Code Cache to avoid re-probing the same GPU function, include:
 * 1. Binary Map for GPU code before probe
 * 2. CUfunction Map for probed code
 */
typedef struct {
    void* key;  // could be CUlibrary, CUmodule, CUkernel, finally CUfunction
    void* code; // the binary code
    char* name; // name of function
    unsigned long long size; // size of bin
    UT_hash_handle hh; 
} binmap_item;

static binmap_item*  binmap  = NULL;

// add item to bin hashmap, won't raise
int binmap_set(void* key, void* code, unsigned long long size, char* name) {
    binmap_item* item = (binmap_item*) malloc(sizeof(binmap_item));
    item->key = key;
    item->code = code;
    item->size = size;
    item->name = name;
    HASH_ADD_PTR(binmap, key, item);
    return 0;
}

int binmap_update_key(void* old_key, void* new_key) {
    binmap_item* item;
    HASH_FIND_PTR(binmap, &old_key, item);
    if (item != NULL) {
        HASH_DEL(binmap, item);
        item->key = new_key;
        HASH_ADD_PTR(binmap, key, item);
        return 0;
    } else {
        return -1;
    }
}

/**
 * Update both the name and the key, favored by cuModuleGetFunction
 * and cuLibraryGetKernel, which will create new entry to hold the
 * new key and value, but underlying binary and size will be shared
 */
int binmap_update_name_key(void* old_key, void* new_key, char* name) {
    binmap_item* old_item;
    HASH_FIND_PTR(binmap, &old_key, old_item);
    if (old_item != NULL) { 
        binmap_item* new_item = (binmap_item*) malloc(sizeof(binmap_item));
        new_item->name = name;
        new_item->key  = new_key;
        new_item->size = old_item->size;
        new_item->code  = old_item->code;
        HASH_ADD_PTR(binmap, key, new_item);
        return 0;
    } else {
        return -1;
    }
}

int binmap_get(void* key, size_t* size, char** name, void** code) {
    binmap_item* item;
    HASH_FIND_PTR(binmap, &key, item);
    if (item != NULL) { 
        *size = item->size;
        *name = item->name;
        *code = item->code;
        fprintf(log, "[binmap] get %p key %p code %p size %zu name %s\n", item, key, *code, *size, *name);
        return 0;
    } else {
        fprintf(log, "[binmap] get %p key %p not-found\n", item, key);
        return -1;
    }
}

// function map items, used as JIT code cache to avoid re-compilation
typedef struct {
    CUfunction original; // original CUfunction
    char* name;          // name of function, if made possible, can be NULL
    int n_param;         // number of parameters, obtained from parsing
    int n_probe;         // number of probes that would dump memory
    int* probe_sizes;    // sizes of probe memory, order matches
    int* probe_types;    // types of probe, 
    bool succeed;        // specify JIT status -> if failed, always goto backup
    CUfunction probed;   // probed CUfunction
    CUfunction pruned;   // pruned CUfunction, for benchmark usage only
    UT_hash_handle hh;   // reserved by uthash
} funcmap_item_t;

static funcmap_item_t* funcmap = NULL;

// add an item to the hashmap-based code cache
int funcmap_set(CUfunction original, char* name, int n_param, int n_probe, int* probe_sizes, int* probe_types, bool succeed, CUfunction probed, CUfunction pruned) {
    funcmap_item_t* item = (funcmap_item_t*) malloc(sizeof(funcmap_item_t));
    item->original = original;
    item->probed = probed;
    item->pruned = pruned;
    item->name = name;
    item->n_param = n_param;
    item->n_probe = n_probe;
    item->probe_sizes = probe_sizes;
    item->probe_types = probe_types;
    item->succeed = succeed; // add func status -> if failed then no need to try probing again and again
    HASH_ADD_PTR(funcmap, original, item);
    return 0;
}

// get an item from hashmap-based code cache CUfunction original, char* name, int n_param, int n_probe, int* probe_sizes, bool succeed, CUfunction probed
int funcmap_get(CUfunction original, char** name, int* n_param, int* n_probe, int** probe_sizes, int** probe_types, bool* succeed, CUfunction* probed, CUfunction* pruned) {
    funcmap_item_t* item;
    HASH_FIND_PTR(funcmap, &original, item);
    if (item != NULL) {
        *name        = item->name;
        *n_param     = item->n_param;
        *n_probe     = item->n_probe;
        *probe_sizes = item->probe_sizes;
        *probe_types = item->probe_types;
        *succeed     = item->succeed;
        *probed      = item->probed;
        *pruned      = item->pruned;
        return 0;
    } else { 
        return -1;
    }        
}

// time utilities
#define TIME_FORMAT_LEN 16
static const char *months[] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
// get the formatted current time (need char [15])
void get_formatted_time(char* holder) {
    time_t rawtime;
    struct tm *timeinfo;
    time(&rawtime); // get time 
    timeinfo = localtime(&rawtime); // format time
    sprintf(holder, "%s_%02d_%02d_%02d_%02d",
                    months[timeinfo->tm_mon],   // Month
                    timeinfo->tm_mday,  // Day of the month
                    timeinfo->tm_hour,  // Hour
                    timeinfo->tm_min,   // Minutes
                    timeinfo->tm_sec);  // Seconds
}

/**
 * hash text based on sha1 algorithm
 * @example char* result = sha1("_ZN2at6native29vectorized_elementwise_kernelILi4ENS0");
 * @note remember to free pointer returned
 */
char* sha1(const char* text) {
	SHA1_CTX ctx;
	sha1_init(&ctx);
	sha1_update(&ctx, text, strlen(text));
	BYTE hash[SHA1_BLOCK_SIZE];
	sha1_final(&ctx, hash);
	char* hexed = malloc(41 * sizeof(char)); // 1 for '\0'
	sprintf(hexed, "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
		hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7], hash[8], hash[9],
		hash[10],hash[11],hash[12],hash[13],hash[14],hash[15],hash[16],hash[17],hash[18],hash[19]);
	return hexed;
}

/**
 * File Utilities, Read File without knowing size, remember to free the ptr
 */
inline void* readf(char* path, const char* mode) {
    FILE* file = fopen(path, mode);
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    void* ptr = malloc(file_size);
    size_t read_size = fread(ptr, 1, file_size, file);
    if (read_size != file_size)
        fprintf(stderr, "read size mismatched\n");
    fclose(file);
    return ptr;
}

// undefine some macros of the original cuda headers
// following symbol are updated to _v2 to match the CUdeviceptr_v2
// original _v1 ptr is uint32_t and now _v2 ptr is uint64_t (unsigned long long)
#undef cuMemAlloc
#undef cuStreamGetCaptureInfo
#undef cuArray3DCreate
#undef cuArray3DGetDescriptor
#undef cuArrayCreate
#undef cuArrayGetDescriptor
#undef cuCtxCreate
#undef cuCtxDestroy
#undef cuCtxPopCurrent
#undef cuCtxPushCurrent
#undef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxSetFlags
#undef cuDeviceTotalMem
#undef cuEventDestroy
#undef cuGetProcAddress
#undef cuGraphAddKernelNode
#undef cuGraphExecKernelNodeSetParams
#undef cuGraphExecUpdate
#undef cuGraphicsResourceGetMappedPointer
#undef cuGraphicsResourceSetMapFlags
#undef cuGraphKernelNodeGetParams
#undef cuGraphKernelNodeSetParams
#undef cuIpcOpenMemHandle
#undef cuLinkAddData
#undef cuLinkAddFile
#undef cuLinkCreate
#undef cuMemAllocHost
#undef cuMemAllocPitch
#undef cuMemcpy2DAsync
#undef cuMemcpy2DUnaligned
#undef cuMemcpy2D
#undef cuMemcpy3DAsync
#undef cuMemcpy3D
#undef cuMemcpyAtoA
#undef cuMemcpyAtoD
#undef cuMemcpyAtoHAsync
#undef cuMemcpyAtoH
#undef cuMemcpyDtoA
#undef cuMemcpyDtoDAsync
#undef cuMemcpyDtoD
#undef cuMemcpyDtoHAsync
#undef cuMemcpyDtoH
#undef cuMemcpyHtoAAsync
#undef cuMemcpyHtoA
#undef cuMemcpyHtoDAsync
#undef cuMemcpyHtoD
#undef cuMemFree
#undef cuMemGetAddressRange
#undef cuMemGetInfo
#undef cuMemHostGetDevicePointer
#undef cuMemHostRegister
#undef cuMemsetD16
#undef cuMemsetD2D16
#undef cuMemsetD2D32
#undef cuMemsetD2D8
#undef cuMemsetD32
#undef cuMemsetD8
#undef cuModuleGetGlobal
#undef cuStreamBatchMemOp
#undef cuStreamBeginCapture
#undef cuStreamDestroy
#undef cuStreamWaitValue32
#undef cuStreamWaitValue64
#undef cuStreamWriteValue32
#undef cuStreamWriteValue64
#undef cuTexRefGetAddress
#undef cuTexRefSetAddress2D
#undef cuTexRefSetAddress