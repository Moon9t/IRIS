#include "onnx_shim.h"
#include "iris_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Build-time switch: if ONNX_RUNTIME_ENABLED is defined, include ORT C API
#ifdef ONNX_RUNTIME_ENABLED
#include <onnxruntime_c_api.h>

typedef struct {
    OrtEnv* env;
    OrtSession* session;
    OrtAllocator* allocator;
    OrtSessionOptions* sess_opts;
} ONNXModel;

void* iris_onnx_session_create(const char* model_path) {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env = NULL;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "iris", &env);
    if (status) { fprintf(stderr, "iris: ORT CreateEnv failed\n"); g_ort->ReleaseStatus(status); return NULL; }
    OrtSessionOptions* sess_opts = NULL;
    status = g_ort->CreateSessionOptions(&sess_opts);
    if (status) { fprintf(stderr, "iris: ORT CreateSessionOptions failed\n"); g_ort->ReleaseStatus(status); g_ort->ReleaseEnv(env); return NULL; }
    g_ort->SetIntraOpNumThreads(sess_opts, 1);
    OrtSession* session = NULL;
    status = g_ort->CreateSession(env, model_path, sess_opts, &session);
    if (status) { fprintf(stderr, "iris: ORT CreateSession failed\n"); g_ort->ReleaseStatus(status); g_ort->ReleaseSessionOptions(sess_opts); g_ort->ReleaseEnv(env); return NULL; }
    OrtAllocator* allocator = NULL;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);
    ONNXModel* m = malloc(sizeof(ONNXModel));
    m->env = env; m->session = session; m->allocator = allocator; m->sess_opts = sess_opts;
    return (void*)m;
}

int iris_onnx_session_run(void* session, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs) {
    if (!session || !inputs || !outputs || !n_outputs) return -1;
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ONNXModel* m = (ONNXModel*)session;

    // Get input and output names from session metadata
    size_t num_input_nodes = 0;
    OrtStatus* status = g_ort->SessionGetInputCount(m->session, &num_input_nodes);
    if (status) { g_ort->ReleaseStatus(status); return -1; }
    
    size_t num_output_nodes = 0;
    status = g_ort->SessionGetOutputCount(m->session, &num_output_nodes);
    if (status) { g_ort->ReleaseStatus(status); return -1; }

    // Allocate input and output name arrays
    const char** input_names = malloc(sizeof(char*) * num_input_nodes);
    const char** output_names = malloc(sizeof(char*) * num_output_nodes);
    if (!input_names || !output_names) { free(input_names); free(output_names); return -1; }

    // Get input names
    for (size_t i = 0; i < num_input_nodes; ++i) {
        char* name = NULL;
        status = g_ort->SessionGetInputName(m->session, i, m->allocator, &name);
        if (status) { g_ort->ReleaseStatus(status); free(input_names); free(output_names); return -1; }
        input_names[i] = name;
    }

    // Get output names
    for (size_t i = 0; i < num_output_nodes; ++i) {
        char* name = NULL;
        status = g_ort->SessionGetOutputName(m->session, i, m->allocator, &name);
        if (status) { g_ort->ReleaseStatus(status); free(input_names); free(output_names); return -1; }
        output_names[i] = name;
    }

    // Prepare input OrtValue tensors
    OrtValue** input_tensors = malloc(sizeof(OrtValue*) * n_inputs);
    if (!input_tensors) { free(input_names); free(output_names); return -1; }

    for (size_t i = 0; i < n_inputs && i < num_input_nodes; ++i) {
        IrisTensor* it = inputs[i];
        OrtMemoryInfo* mem_info = NULL;
        status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, ORT_INVALID_DEVICE_ID, &mem_info);
        if (status) { g_ort->ReleaseStatus(status); free(input_tensors); free(input_names); free(output_names); return -1; }

        OrtValue* val = NULL;
        status = g_ort->CreateTensorWithDataAsOrtValue(mem_info, it->data, sizeof(float) * it->numel, it->shape, it->ndim, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &val);
        g_ort->ReleaseMemoryInfo(mem_info);
        if (status) { g_ort->ReleaseStatus(status); free(input_tensors); free(input_names); free(output_names); return -1; }
        input_tensors[i] = val;
    }

    // Run the session
    OrtRunOptions* run_opts = NULL;
    OrtValue** output_values = NULL;
    status = g_ort->Run(m->session, run_opts, input_names, (const OrtValue* const*)input_tensors, n_inputs, output_names, num_output_nodes, &output_values);
    
    // Cleanup input tensors and names
    for (size_t i = 0; i < n_inputs; ++i) {
        if (input_tensors[i]) g_ort->ReleaseValue(input_tensors[i]);
    }
    free(input_tensors);

    if (status) { g_ort->ReleaseStatus(status); free(input_names); free(output_names); return -1; }

    // Convert output OrtValues to IrisTensor
    *outputs = malloc(sizeof(IrisTensor*) * num_output_nodes);
    if (!*outputs) { free(input_names); free(output_names); return -1; }
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
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ONNXModel* m = (ONNXModel*)session;
    if (m->session) g_ort->ReleaseSession(m->session);
    if (m->sess_opts) g_ort->ReleaseSessionOptions(m->sess_opts);
    if (m->env) g_ort->ReleaseEnv(m->env);
    free(m);
}

#else

// Fallback when ORT not available
void* iris_onnx_session_create(const char* model_path) { (void)model_path; fprintf(stderr, "iris: ONNX Runtime not enabled at build time\n"); return NULL; }
int   iris_onnx_session_run(void* session, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs) { (void)session;(void)inputs;(void)n_inputs;(void)outputs;(void)n_outputs; fprintf(stderr, "iris: ONNX Runtime not enabled at build time\n"); return -1; }
void  iris_onnx_session_free(void* session) { (void)session; }

#endif /* ONNX_RUNTIME_ENABLED */
