#include "logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>
#include <unistd.h>
#include <sys/eventfd.h>
#include <sys/epoll.h>
#include <fcntl.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>

#define MAX_LOG_MSG_LEN   256
#define RING_BUFFER_SIZE  (1 << 16)  // 65536 entries (power of 2)

typedef struct {
    uint64_t timestamp;
    uint32_t tid;
    char message[MAX_LOG_MSG_LEN];
} LogEntry;

typedef struct {
    LogEntry buffer[RING_BUFFER_SIZE];
    _Atomic uint32_t head;
    _Atomic uint32_t tail;
} LogRingBuffer;

static LogRingBuffer g_log_buffer = {.head = 0, .tail = 0};

static int g_eventfd = -1;
static int g_epollfd = -1;

// get ns timesatmp
static inline uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.nsec;
}

// get thread id 
static inline uint32_t get_tid(void) {
    return (uint32_t)(uintptr_t)pthread_self();
}

// write log（non-block， discard if fail）
int log_write(const char *msg) {
    uint32_t head = atomic_load_explicit(&g_log_buffer.head, memory_order_relaxed);
    uint32_t next_head = (head + 1) & (RING_BUFFER_SIZE - 1);

    uint32_t tail = atomic_load_explicit(&g_log_buffer.tail, memory_order_acquire);
    if (next_head == tail) {
        return -1;  // buffer full
    }

    if (!atomic_compare_exchange_weak_explicit(
            &g_log_buffer.head, &head, next_head,
            memory_order_acquire, memory_order_relaxed)) {
        return log_write(msg);  // retry (rare)
    }

    LogEntry *entry = &g_log_buffer.buffer[head];
    entry->timestamp = get_timestamp_ns();
    entry->tid = get_tid();
    strncpy(entry->message, msg, MAX_LOG_MSG_LEN - 1);
    entry->message[MAX_LOG_MSG_LEN - 1] = '\0';

    atomic_thread_fence(memory_order_release);

    //notify consumer
    uint64_t one = 1;
    if (write(g_eventfd, &one, sizeof(one)) != sizeof(one)) {
        // ignore error
    }
    return 0;
}

// 格式化日志（线程安全，栈上缓冲）
void logf(const char *fmt, ...) {
    char buf[MAX_LOG_MSG_LEN];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    log_write(buf);
}

// consumer thread
static void* log_consumer_thread(void *arg) {
    FILE *fp = fopen("/home/admin/logs/xpu_kernel.log", "a");
    if (!fp) {
        perror("fopen /home/admin/logs/xpu_kernel.log");
        return NULL;
    }

    struct epoll_event ev;
    int nfds;

    while (1) {
        nfds = epoll_wait(g_epollfd, &ev, 1, -1);
        if (nfds == -1) {
            if (errno == EINTR) continue;
            perror("epoll_wait");
            break;
        }

        // clear eventfd count
        uint64_t counter;
        if (read(g_eventfd, &counter, sizeof(counter)) != sizeof(counter)) {
            // ignore mistake
        }

        // batch consume all the binaries
        uint32_t tail, head;
        do {
            tail = atomic_load_explicit(&g_log_buffer.tail, memory_order_acquire);
            head = atomic_load_explicit(&g_log_buffer.head, memory_order_acquire);
            if (tail == head) break;

            LogEntry *entry = &g_log_buffer.buffer[tail];
            uint64_t sec = entry->timestamp / 1000000000;
            uint64_t ns = entry->timestamp % 1000000000;

            fprintf(fp, "[%lu.%09lu] TID-%u: %s\n", sec, ns, entry->tid, entry->message);

            atomic_store_explicit(&g_log_buffer.tail, (tail + 1) & (RING_BUFFER_SIZE - 1),
                                  memory_order_release);
        } while (1);
        fflush(fp); // 
    }

    fclose(fp);
    return NULL;
}

// init logger system
void init_logger(void) {
    g_eventfd = eventfd(0, EFD_CLOEXEC);
    if (g_eventfd == -1) {
        perror("eventfd");
        exit(1);
    }

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
        close(g_eventfd);
        close(g_epollfd);
        exit(1);
    }

    pthread_t thread;
    if (pthread_create(&thread, NULL, log_consumer_thread, NULL) != 0) {
        perror("pthread_create");
        exit(1);
    }

    pthread_detach(thread); 
}

// logger fd cleanup (no need, complete this part work when process exit)
void cleanup_logger(void) {
    if (g_eventfd != -1) {
        close(g_eventfd);
        g_eventfd = -1;
    }
    if (g_epollfd != -1) {
        close(g_epollfd);
        g_epollfd = -1;
    }
}
