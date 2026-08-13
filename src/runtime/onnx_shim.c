#include "onnx_shim.h"
#include "iris_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

// ONNX Runtime is resolved lazily at runtime (dlopen/LoadLibrary) rather than
// linked. iris_ml_dynload.h pulls in the vendored ORT C API header for the ABI
// layout and provides iris_ort_api(), which returns NULL when onnxruntime is
// unavailable. Nothing here creates a link-time dependency on the SDK.
#include "iris_ml_dynload.h"

// Compatibility: some ORT distributions may not define ORT_INVALID_DEVICE_ID
#ifndef ORT_INVALID_DEVICE_ID
#define ORT_INVALID_DEVICE_ID -1
#endif

typedef struct {
    OrtEnv* env;
    OrtSession* session;
    OrtAllocator* allocator;
    OrtSessionOptions* sess_opts;
} ONNXModel;

void* iris_onnx_session_create(const char* model_path) {
    const OrtApi* g_ort = iris_ort_api();
    if (!g_ort) return NULL;
    OrtEnv* env = NULL;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "iris", &env);
    if (status) { fprintf(stderr, "iris: ORT CreateEnv failed\n"); g_ort->ReleaseStatus(status); return NULL; }
    OrtSessionOptions* sess_opts = NULL;
    status = g_ort->CreateSessionOptions(&sess_opts);
    if (status) { fprintf(stderr, "iris: ORT CreateSessionOptions failed\n"); g_ort->ReleaseStatus(status); g_ort->ReleaseEnv(env); return NULL; }
    g_ort->SetIntraOpNumThreads(sess_opts, 1);
    OrtSession* session = NULL;
#ifdef _WIN32
    wchar_t w_model_path[4096];
    int w_len = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, w_model_path, 4096);
    if (w_len <= 0) {
        fprintf(stderr, "iris: failed to convert model path to wide string\n");
        g_ort->ReleaseSessionOptions(sess_opts);
        g_ort->ReleaseEnv(env);
        return NULL;
    }
    status = g_ort->CreateSession(env, w_model_path, sess_opts, &session);
#else
    status = g_ort->CreateSession(env, model_path, sess_opts, &session);
#endif
    if (status) {
        fprintf(stderr, "iris: ORT CreateSession failed\n");
        // Print ORT error message if available
        const char* msg = g_ort->GetErrorMessage(status);
        if (msg) {
            fprintf(stderr, "iris: ORT error: %s\n", msg);
        }
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(sess_opts);
        g_ort->ReleaseEnv(env);
        return NULL;
    }
    OrtAllocator* allocator = NULL;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);
    ONNXModel* m = malloc(sizeof(ONNXModel));
    m->env = env; m->session = session; m->allocator = allocator; m->sess_opts = sess_opts;
    return (void*)m;
}

int iris_onnx_session_run(void* session, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs) {
    if (!session || !inputs || !outputs || !n_outputs) {
        fprintf(stderr, "iris: ORT Run error: invalid arguments\n");
        return -1;
    }
    const OrtApi* g_ort = iris_ort_api();
    if (!g_ort) return -1;
    ONNXModel* m = (ONNXModel*)session;

    // Get input and output names from session metadata
    size_t num_input_nodes = 0;
    OrtStatus* status = g_ort->SessionGetInputCount(m->session, &num_input_nodes);
    if (status) {
        fprintf(stderr, "iris: ORT Run error: SessionGetInputCount failed: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -1;
    }
    
    size_t num_output_nodes = 0;
    status = g_ort->SessionGetOutputCount(m->session, &num_output_nodes);
    if (status) {
        fprintf(stderr, "iris: ORT Run error: SessionGetOutputCount failed: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -1;
    }

    // Allocate input and output name arrays
    const char** input_names = malloc(sizeof(char*) * num_input_nodes);
    const char** output_names = malloc(sizeof(char*) * num_output_nodes);
    if (!input_names || !output_names) {
        fprintf(stderr, "iris: ORT Run error: out of memory for names\n");
        free(input_names); free(output_names);
        return -1;
    }

    // Get input names
    for (size_t i = 0; i < num_input_nodes; ++i) {
        char* name = NULL;
        status = g_ort->SessionGetInputName(m->session, i, m->allocator, &name);
        if (status) {
            fprintf(stderr, "iris: ORT Run error: SessionGetInputName failed: %s\n", g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status); free(input_names); free(output_names);
            return -1;
        }
        input_names[i] = name;
    }

    // Get output names
    for (size_t i = 0; i < num_output_nodes; ++i) {
        char* name = NULL;
        status = g_ort->SessionGetOutputName(m->session, i, m->allocator, &name);
        if (status) {
            fprintf(stderr, "iris: ORT Run error: SessionGetOutputName failed: %s\n", g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status); free(input_names); free(output_names);
            return -1;
        }
        output_names[i] = name;
    }

    // Prepare input OrtValue tensors
    OrtValue** input_tensors = malloc(sizeof(OrtValue*) * n_inputs);
    if (!input_tensors) {
        fprintf(stderr, "iris: ORT Run error: out of memory for input tensors\n");
        free(input_names); free(output_names);
        return -1;
    }

    for (size_t i = 0; i < n_inputs && i < num_input_nodes; ++i) {
        IrisTensor* it = inputs[i];
        OrtMemoryInfo* mem_info = NULL;
        status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, ORT_INVALID_DEVICE_ID, &mem_info);
        if (status) {
            fprintf(stderr, "iris: ORT Run error: CreateCpuMemoryInfo failed: %s\n", g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status); free(input_tensors); free(input_names); free(output_names);
            return -1;
        }

        OrtValue* val = NULL;
        status = g_ort->CreateTensorWithDataAsOrtValue(mem_info, it->data, sizeof(float) * it->numel, it->shape, it->ndim, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &val);
        g_ort->ReleaseMemoryInfo(mem_info);
        if (status) { g_ort->ReleaseStatus(status); free(input_tensors); free(input_names); free(output_names); return -1; }
        input_tensors[i] = val;
    }

    // Run the session
    OrtRunOptions* run_opts = NULL;
    OrtValue** output_values = calloc(num_output_nodes, sizeof(OrtValue*));
    if (!output_values) {
        for (size_t i = 0; i < n_inputs; ++i) {
            if (input_tensors[i]) g_ort->ReleaseValue(input_tensors[i]);
        }
        free(input_tensors);
        free(input_names);
        free(output_names);
        return -1;
    }
    status = g_ort->Run(m->session, run_opts, input_names, (const OrtValue* const*)input_tensors, n_inputs, output_names, num_output_nodes, output_values);
    
    // Cleanup input tensors and names
    for (size_t i = 0; i < n_inputs; ++i) {
        if (input_tensors[i]) g_ort->ReleaseValue(input_tensors[i]);
    }
    free(input_tensors);

    if (status) {
        const char* msg = g_ort->GetErrorMessage(status);
        if (msg) {
            fprintf(stderr, "iris: ORT Run failed: %s\n", msg);
        }
        g_ort->ReleaseStatus(status);
        free(output_values);
        free(input_names);
        free(output_names);
        return -1;
    }

    // Convert output OrtValues to IrisTensor
    *outputs = malloc(sizeof(IrisTensor*) * num_output_nodes);
    if (!*outputs) {
        for (size_t i = 0; i < num_output_nodes; ++i) {
            if (output_values[i]) g_ort->ReleaseValue(output_values[i]);
        }
        free(output_values);
        free(input_names);
        free(output_names);
        return -1;
    }
    *n_outputs = num_output_nodes;

    for (size_t i = 0; i < num_output_nodes; ++i) {
        OrtValue* ort_val = output_values[i];
        
        // Get tensor information
        OrtTensorTypeAndShapeInfo* info = NULL;
        status = g_ort->GetTensorTypeAndShape(ort_val, &info);
        if (status) { g_ort->ReleaseStatus(status); free(*outputs); free(input_names); free(output_names); return -1; }

        size_t ndim_ortsize = 0;
        status = g_ort->GetDimensionsCount(info, &ndim_ortsize);
        if (status) { g_ort->ReleaseStatus(status); g_ort->ReleaseTensorTypeAndShapeInfo(info); free(*outputs); free(input_names); free(output_names); return -1; }
        int32_t ndim = (int32_t)ndim_ortsize;

        // Get shape
        int64_t* ort_shape = malloc(sizeof(int64_t) * ndim);
        if (!ort_shape) { g_ort->ReleaseTensorTypeAndShapeInfo(info); free(*outputs); free(input_names); free(output_names); return -1; }
        status = g_ort->GetDimensions(info, ort_shape, ndim);
        if (status) { g_ort->ReleaseStatus(status); g_ort->ReleaseTensorTypeAndShapeInfo(info); free(ort_shape); free(*outputs); free(input_names); free(output_names); return -1; }

        // Compute numel
        int64_t numel = 1;
        for (int32_t d = 0; d < ndim; ++d) numel *= ort_shape[d];

        // Get tensor data
        float* ort_data = NULL;
        status = g_ort->GetTensorMutableData(ort_val, (void**)&ort_data);
        if (status) { g_ort->ReleaseStatus(status); g_ort->ReleaseTensorTypeAndShapeInfo(info); free(ort_shape); free(*outputs); free(input_names); free(output_names); return -1; }

        // Create IrisTensor (copies data)
        IrisTensor* iris_out = iris_tensor_alloc(ndim, ort_shape);
        if (!iris_out) { g_ort->ReleaseTensorTypeAndShapeInfo(info); free(ort_shape); free(*outputs); free(input_names); free(output_names); return -1; }
        memcpy(iris_out->data, ort_data, sizeof(float) * numel);
        
        (*outputs)[i] = iris_out;
        g_ort->ReleaseTensorTypeAndShapeInfo(info);
        free(ort_shape);
    }

    // Cleanup
    for (size_t i = 0; i < num_output_nodes; ++i) {
        if (output_values[i]) g_ort->ReleaseValue(output_values[i]);
    }
    if (output_values) free(output_values);
    free(input_names);
    free(output_names);

    return 0;
}

void iris_onnx_session_free(void* session) {
    if (!session) return;
    const OrtApi* g_ort = iris_ort_api();
    if (!g_ort) return;
    ONNXModel* m = (ONNXModel*)session;
    if (m->session) g_ort->ReleaseSession(m->session);
    if (m->sess_opts) g_ort->ReleaseSessionOptions(m->sess_opts);
    if (m->env) g_ort->ReleaseEnv(m->env);
    free(m);
}

int64_t iris_onnx_get_input_count(int64_t session) {
    if (!session) return 0;
    const OrtApi* g_ort = iris_ort_api();
    if (!g_ort) return 0;
    ONNXModel* m = (ONNXModel*)(intptr_t)session;
    size_t count = 0;
    OrtStatus* status = g_ort->SessionGetInputCount(m->session, &count);
    if (status) { g_ort->ReleaseStatus(status); return 0; }
    return (int64_t)count;
}

int64_t iris_onnx_get_output_count(int64_t session) {
    if (!session) return 0;
    const OrtApi* g_ort = iris_ort_api();
    if (!g_ort) return 0;
    ONNXModel* m = (ONNXModel*)(intptr_t)session;
    size_t count = 0;
    OrtStatus* status = g_ort->SessionGetOutputCount(m->session, &count);
    if (status) { g_ort->ReleaseStatus(status); return 0; }
    return (int64_t)count;
}

const char* iris_onnx_get_input_name(int64_t session, int64_t idx) {
    if (!session) return "";
    const OrtApi* g_ort = iris_ort_api();
    if (!g_ort) return "";
    ONNXModel* m = (ONNXModel*)(intptr_t)session;
    char* name = NULL;
    OrtStatus* status = g_ort->SessionGetInputName(m->session, (size_t)idx, m->allocator, &name);
    if (status) { g_ort->ReleaseStatus(status); return ""; }
    return name;
}

const char* iris_onnx_get_output_name(int64_t session, int64_t idx) {
    if (!session) return "";
    const OrtApi* g_ort = iris_ort_api();
    if (!g_ort) return "";
    ONNXModel* m = (ONNXModel*)(intptr_t)session;
    char* name = NULL;
    OrtStatus* status = g_ort->SessionGetOutputName(m->session, (size_t)idx, m->allocator, &name);
    if (status) { g_ort->ReleaseStatus(status); return ""; }
    return name;
}

/* The former "#else" branch — stub definitions used when ONNX_RUNTIME_ENABLED
 * was not set at build time — is gone. There is no longer a build-time switch:
 * these functions are always compiled, and iris_ort_api() reports the missing
 * library at the point of use instead. */

