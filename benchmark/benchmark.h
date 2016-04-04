#ifndef BENCHMARK_H
#define BENCHMARK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct benchmark_ctx benchmark_ctx;

benchmark_ctx* benchmark_initialize();

void benchmark_execute(benchmark_ctx* ctx);

void benchmark_destroy(benchmark_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif // BENCHMARK_H
