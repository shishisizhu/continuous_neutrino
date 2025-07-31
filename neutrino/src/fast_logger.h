#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>
#include <unistd.h>
#include <stdarg.h>
#include <stdint.h>

// config params
#define MAX_LOG_MSG_LEN     256
#define RING_BUFFER_SIZE    (1 << 18)     
#define LOG_FILE            "/home/admin/logs/xpu_kernel.log"

#define CACHE_LINE_SIZE     64
#define ALIGNED __attribute__((aligned(CACHE_LINE_SIZE)))

// log record entry
typedef struct {
    uint64_t timestamp;
    uint32_t tid;
    char message[MAX_LOG_MSG_LEN];
} LogEntry;

typedef struct {
    LogEntry buffer[RING_BUFFER_SIZE];
    _Atomic uint32_t head;      // producer write
    _Atomic uint32_t tail;      // consumer read
} ALIGNED LogRingBuffer;

static LogRingBuffer g_log_buffer ALIGNED = { .head = 0, .tail = 0 };

// global logger state
static volatile int g_logger_running = 0;
static pthread_t g_consumer_thread;

static _Atomic uint64_t g_log_drop_count = 0;
static _Atomic uint64_t g_log_write_count = 0;

static inline uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static inline uint32_t get_tid(void) {
    return (uint32_t)(uintptr_t)pthread_self();
}

/*
 * @note keep short delay : use atomic non-lock methods to keep efficiency
 */
static inline int log_write(const char *msg) {
    uint32_t head = atomic_load_explicit(&g_log_buffer.head, memory_order_relaxed);
    uint32_t next_head = (head + 1) & (RING_BUFFER_SIZE - 1);

    uint32_t tail = g_log_buffer.tail;  // relaxed load
    if (next_head == tail) {
        atomic_fetch_add(&g_log_drop_count, 1);
        return -1;
    }

    // CAS flush head
    if (!atomic_compare_exchange_weak_explicit(
            &g_log_buffer.head, &head, next_head,
            memory_order_acq_rel, memory_order_relaxed)) {
        // retry while contend
        return log_write(msg);
    }

    // （head is old value）
    LogEntry *entry = &g_log_buffer.buffer[head];
    entry->timestamp = get_timestamp_ns();
    entry->tid = get_tid();
    memcpy(entry->message, msg, MAX_LOG_MSG_LEN - 1);
    entry->message[MAX_LOG_MSG_LEN - 1] = '\0';

    atomic_thread_fence(memory_order_release);
    atomic_fetch_add(&g_log_write_count, 1);
    return 0;
}

/*
 *@note buffer in stack ,avoid malloc to keep speed
 */
void log_msg(const char *fmt, ...) {
    char buf[MAX_LOG_MSG_LEN] __attribute__((aligned(8)));
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (len > 0) {
        log_write(buf);
    }
}

/*
 *@note log consumer thread, thread continuously inquiry to keep high 
 */
static void* logger_consumer(void *arg) {
    char line[512];
    FILE *fp = fopen(LOG_FILE, "a");
    if (!fp) {
        fprintf(stderr, "Failed to open log file: %s\n", LOG_FILE);
        return NULL;
    }

    // 大缓冲提升 fwrite 效率
    setvbuf(fp, NULL, _IOFBF, 1024 * 1024);  // 1MB buffer

    uint32_t tail, head;
    int batch_count;

    while (g_logger_running) {
        tail = atomic_load_explicit(&g_log_buffer.tail, memory_order_acquire);
        head = atomic_load_explicit(&g_log_buffer.head, memory_order_acquire);

        batch_count = 0;
        while (tail != head && batch_count < 1000) {  // 每批最多 1000 条
            LogEntry *entry = &g_log_buffer.buffer[tail];
            uint64_t sec = entry->timestamp / 1000000000;
            uint64_t ns = entry->timestamp % 1000000000;

            int n = snprintf(line, sizeof(line), "[%lu.%09lu] TID-%u: %s\n",
                             sec, ns, entry->tid, entry->message);
            if (n > 0 && n < sizeof(line)) {
                fwrite(line, 1, n, fp);
            }

            tail = (tail + 1) & (RING_BUFFER_SIZE - 1);
            batch_count++;
        }

        if (batch_count > 0) {
            atomic_store_explicit(&g_log_buffer.tail, tail, memory_order_release);
            // no fflush，setvbuf flush
        } 
    }

    fflush(fp);
    fclose(fp);
    return NULL;
}

/**
 * @note init thread
 */
 
void init_logger(void) {
    if (g_logger_running) return;

    g_logger_running = 1;

    if (pthread_create(&g_consumer_thread, NULL, logger_consumer, NULL) != 0) {
        perror("pthread_create");
        exit(1);
    }
}

void logger_shutdown(void) {
    if (!g_logger_running) return;
    g_logger_running = 0;

    void *result;
    if (pthread_join(g_consumer_thread, &result) == 0) {
        fprintf(stderr, "[LOGGER] Consumer stopped.\n");
    }

    uint64_t drops = atomic_load(&g_log_drop_count);
    uint64_t writes = atomic_load(&g_log_write_count);
    fprintf(stderr, "[LOGGER] Stats: %lu written, %lu dropped.\n", writes, drops);
}

