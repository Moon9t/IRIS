#ifndef IRIS_ONNX_SHIM_H
#define IRIS_ONNX_SHIM_H

#include <stddef.h>
#include "iris_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// Simple ONNX Runtime shim API. If built with ONNX runtime available,
// these functions will forward to ORT; otherwise they return failure codes
// or NULL to indicate the feature is unavailable.

void* iris_onnx_session_create(const char* model_path);
int   iris_onnx_session_run(void* session, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs);
void  iris_onnx_session_free(void* session);

#ifdef __cplusplus
}
#endif

#endif // IRIS_ONNX_SHIM_H
