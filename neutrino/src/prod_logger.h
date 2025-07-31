#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>
#include <unistd.h>
#include <sys/eventfd.h>
#include <sys/epoll.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <fcntl.h>

//config params
#define MAX_LOG_MSG_LEN     256
#define RING_BUFFER_SIZE    (1 << 18)          
#define LOG_FILE            "/home/admin/logs/xpu_kernel.log"
#define MAX_FILENAME_LEN    512

// log entry to present one log record
typedef struct {
    uint64_t timestamp;
    uint32_t tid;
    char message[MAX_LOG_MSG_LEN];
} LogEntry;

//present buffer
typedef struct {
    LogEntry buffer[RING_BUFFER_SIZE];
    _Atomic uint32_t head;      
    _Atomic uint32_t tail;    
    _Atomic uint8_t  need_wakeup; 
} LogRingBuffer;

static LogRingBuffer g_log_buffer = { .head = 0, .tail = 0, .need_wakeup = 1 };

// global state
static int g_logger_running = 0;
static pthread_t g_consumer_thread = 0;
static int g_eventfd = -1;
static int g_epollfd = -1;

static _Atomic uint64_t g_log_drop_count = 0;   //discard log count
static _Atomic uint64_t g_log_write_count = 0;  //log success count


static inline uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static inline uint32_t get_tid(void) {
    return (uint32_t)(uintptr_t)pthread_self();
}

//log write 
int log_write(const char *msg) {
    uint32_t head = atomic_load_explicit(&g_log_buffer.head, memory_order_relaxed);
    uint32_t next_head = (head + 1) & (RING_BUFFER_SIZE - 1);

    uint32_t tail = atomic_load_explicit(&g_log_buffer.tail, memory_order_acquire);
    if (next_head == tail) {
        atomic_fetch_add(&g_log_drop_count, 1);
        return -1;  // buffer full
    }

    if (!atomic_compare_exchange_weak_explicit(
            &g_log_buffer.head, &head, next_head,
            memory_order_acquire, memory_order_relaxed)) {
        return log_write(msg);  // retry while CAS fail
    }

    LogEntry *entry = &g_log_buffer.buffer[head];
    entry->timestamp = get_timestamp_ns();
    entry->tid = get_tid();
    strncpy(entry->message, msg, MAX_LOG_MSG_LEN - 1);
    entry->message[MAX_LOG_MSG_LEN - 1] = '\0';

    atomic_thread_fence(memory_order_release);

    // Wakeup only need_wakeup == 1（lazy_wakeup）
    if (atomic_exchange(&g_log_buffer.need_wakeup, 0) == 1) {
        uint64_t one = 1;
        write(g_eventfd, &one, sizeof(one));  
    }

    atomic_fetch_add(&g_log_write_count, 1);
    return 0;
}


void log_msg(const char *fmt, ...) {
    char buf[MAX_LOG_MSG_LEN];
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (len > 0) {
        log_write(buf);
    }
}

// consumer thread read from buffer and write to log file(epoll + )
static void* logger_consumer(void *arg) {
    char line[512];
    struct epoll_event ev;
    uint64_t counter;
    int nfds;

    FILE *fp = fopen(LOG_FILE, "a");
    if (!fp) {
        fprintf(stderr, "Failed to open log file: %s\n", LOG_FILE);
        return NULL;
    }

    // set file buffer to improve 
    setvbuf(fp, NULL, _IOFBF, 64 * 1024);  // 64KB buffer

    while (g_logger_running) {
        nfds = epoll_wait(g_epollfd, &ev, 1, -1);  // wait continuously
        if (nfds == -1) {
            if (errno == EINTR) continue;
            perror("epoll_wait");
            break;
        }

        // clear eventfd count
        if (read(g_eventfd, &counter, sizeof(counter)) != sizeof(counter)) {
            perror("read epoll g_eventfd error");
        }

        // batch consume all the logs
        uint32_t tail, head;
        int processed = 0;

        do {
            tail = atomic_load_explicit(&g_log_buffer.tail, memory_order_acquire);
            head = atomic_load_explicit(&g_log_buffer.head, memory_order_acquire);
            if (tail == head) break;

            LogEntry *entry = &g_log_buffer.buffer[tail];
            uint64_t sec = entry->timestamp / 1000000000;
            uint64_t ns = entry->timestamp % 1000000000;

            int n = snprintf(line, sizeof(line), "[%lu.%09lu] TID-%u: %s\n",
                             sec, ns, entry->tid, entry->message);
            if (n > 0 && n < sizeof(line)) {
                fwrite(line, 1, n, fp);
            }

            atomic_store_explicit(&g_log_buffer.tail, (tail + 1) & (RING_BUFFER_SIZE - 1),
                                  memory_order_release);
            processed++;
        } while (1);

        // only buffer is empty, wakeup this
        if (processed > 0) {
            fflush(fp);  
            atomic_store_explicit(&g_log_buffer.need_wakeup, 1, memory_order_release);
        }
    }

    fclose(fp);
    return NULL;
}

void init_logger(void) {
    if (g_logger_running) return;

    // eventfd
    g_eventfd = eventfd(0, EFD_CLOEXEC | EFD_SEMAPHORE);
    if (g_eventfd == -1) {
        perror("eventfd");
        exit(1);
    }

    // epoll
    g_epollfd = epoll_create1(EPOLL_CLOEXEC);
    if (g_epollfd == -1) {
        perror("epoll_create1");
        close(g_eventfd);
        exit(1);
    }

    struct epoll_event ev = {0};
    ev.events = EPOLLIN;
    ev.data.fd = g_eventfd;

    if (epoll_ctl(g_epollfd, EPOLL_CTL_ADD, g_eventfd, &ev) == -1) {
        perror("epoll_ctl");
        close(g_epollfd);
        close(g_eventfd);
        exit(1);
    }

    g_logger_running = 1;

    if (pthread_create(&g_consumer_thread, NULL, logger_consumer, NULL) != 0) {
        perror("pthread_create");
        g_logger_running = 0;
        exit(1);
    }

    pthread_detach(g_consumer_thread);
}


void logger_shutdown(void) {
    if (!g_logger_running) return;
    g_logger_running = 0;

    usleep(2000);

    uint64_t drops = atomic_load(&g_log_drop_count);
    uint64_t writes = atomic_load(&g_log_write_count);

    fprintf(stderr, "[LOGGER] Shutdown: %lu logs written, %lu dropped.\n", writes, drops);
}

