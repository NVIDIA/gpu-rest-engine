#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct classifier_ctx classifier_ctx;

classifier_ctx* classifier_initialize(char* model_file, char* trained_file,
                                      char* mean_file, char* label_file);

const char* classifier_classify(classifier_ctx* ctx,
                                char* buffer, size_t length);

void classifier_destroy(classifier_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif
