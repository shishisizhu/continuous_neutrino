#ifndef LOGGER_H
#define LOGGER_H

#include <stdint.h>

//init logger system
void init_logger(void);

// write log content
void logf(const char *fmt, ...);

// no need , logger clean automaticly when process exit
void cleanup_logger(void);

#endif
