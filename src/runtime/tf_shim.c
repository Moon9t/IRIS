#include "iris_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(TENSORFLOW_ENABLED)
#include <tensorflow/c/c_api.h>

struct TFModel { TF_Graph* graph; TF_Session* session; TF_Status* status; };

static void deallocator_noop(void* data, size_t len, void* arg) {
    (void)data; (void)len; (void)arg;
}

void* iris_tf_load_saved_model(const char* path) {
    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* opts = TF_NewSessionOptions();
    const char* tags = "serve";
    int ntags = 1;
    TF_Session* sess = TF_LoadSessionFromSavedModel(opts, NULL, path, &tags, ntags, graph, NULL, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "iris: TF load error: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteGraph(graph);
        TF_DeleteSessionOptions(opts);
        return NULL;
    }
    TFModel* m = malloc(sizeof(TFModel));
    m->graph = graph; m->session = sess; m->status = status;
    TF_DeleteSessionOptions(opts);
    return (void*)m;
}

int iris_tf_run(void* model, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs) {
    if (!model || n_inputs == 0 || !inputs) return -1;
    TFModel* m = (TFModel*)model;

    // Scan operations to discover input and output nodes
    TF_Operation* input_op = NULL;
    TF_Operation* output_op = NULL;

    size_t pos = 0;
    TF_Operation* oper;
    while ((oper = TF_GraphNextOperation(m->graph, &pos)) != NULL) {
        const char* op_type = TF_OperationOpType(oper);
        if (strcmp(op_type, "Placeholder") == 0) {
            input_op = oper;
        } else if (strcmp(op_type, "Identity") == 0 || strcmp(op_type, "StatefulPartitionedCall") == 0) {
            output_op = oper;
        }
    }

    // Fallbacks based on name search
    if (!input_op || !output_op) {
        pos = 0;
        while ((oper = TF_GraphNextOperation(m->graph, &pos)) != NULL) {
            const char* name = TF_OperationName(oper);
            if (!input_op && (strstr(name, "input") || strstr(name, "x") || strstr(name, "Placeholder"))) {
                input_op = oper;
            }
            if (!output_op && (strstr(name, "output") || strstr(name, "Identity") || strstr(name, "predictions") || strstr(name, "StatefulPartitionedCall"))) {
                output_op = oper;
            }
        }
    }

    // Ultimate fallbacks: first for input, last for output
    if (!input_op) {
        pos = 0;
        input_op = TF_GraphNextOperation(m->graph, &pos);
    }
    if (!output_op) {
        pos = 0;
        while ((oper = TF_GraphNextOperation(m->graph, &pos)) != NULL) {
            output_op = oper;
        }
    }

    if (!input_op || !output_op) {
        fprintf(stderr, "iris: TF run failed: could not locate input/output operations in graph\n");
        return -1;
    }

    // Allocate input TF_Tensor
    IrisTensor* it = inputs[0];
    int64_t* dims = malloc(sizeof(int64_t) * it->ndim);
    for (int i = 0; i < it->ndim; ++i) {
        dims[i] = it->shape[i];
    }
    
    TF_Tensor* tf_in = TF_NewTensor(
        TF_FLOAT, 
        dims, it->ndim, 
        it->data, sizeof(float) * it->numel, 
        deallocator_noop, NULL
    );
    free(dims);
    
    if (!tf_in) {
        fprintf(stderr, "iris: TF run failed: could not create input tensor\n");
        return -1;
    }

    // Setup input/output ports for SessionRun
    TF_Output input_port = { input_op, 0 };
    TF_Output output_port = { output_op, 0 };
    
    TF_Output input_ports[] = { input_port };
    TF_Tensor* input_values[] = { tf_in };
    TF_Output output_ports[] = { output_port };
    TF_Tensor* output_values[] = { NULL };

    // Reset status for safety
    TF_SetStatus(m->status, TF_OK, "");

    TF_SessionRun(
        m->session, 
        NULL, 
        input_ports, input_values, 1, 
        output_ports, output_values, 1, 
        NULL, 0, NULL, 
        m->status
    );

    TF_DeleteTensor(tf_in);

    if (TF_GetCode(m->status) != TF_OK) {
        fprintf(stderr, "iris: TF run error: %s\n", TF_Message(m->status));
        return -1;
    }

    TF_Tensor* tf_out = output_values[0];
    if (!tf_out) {
        fprintf(stderr, "iris: TF run failed: returned output tensor was NULL\n");
        return -1;
    }

    // Convert output TF_Tensor to IrisTensor
    int ndim = TF_NumDims(tf_out);
    int64_t* out_shape = malloc(sizeof(int64_t) * (ndim > 0 ? ndim : 1));
    if (ndim > 0) {
        for (int d = 0; d < ndim; ++d) {
            out_shape[d] = TF_Dim(tf_out, d);
        }
    } else {
        out_shape[0] = 1;
        ndim = 1;
    }

    IrisTensor* iris_out = iris_tensor_alloc(ndim, out_shape);
    free(out_shape);

    if (!iris_out) {
        TF_DeleteTensor(tf_out);
        fprintf(stderr, "iris: TF run failed: could not allocate IrisTensor for output\n");
        return -1;
    }

    float* tf_data = (float*)TF_TensorData(tf_out);
    memcpy(iris_out->data, tf_data, sizeof(float) * iris_out->numel);
    TF_DeleteTensor(tf_out);

    // Set results
    *outputs = malloc(sizeof(IrisTensor*) * 1);
    (*outputs)[0] = iris_out;
    *n_outputs = 1;

    return 0;
}

void iris_tf_free(void* model) {
    if (!model) return;
    TFModel* m = (TFModel*)model;
    if (m->session) TF_CloseSession(m->session, m->status);
    if (m->session) TF_DeleteSession(m->session, m->status);
    if (m->graph) TF_DeleteGraph(m->graph);
    if (m->status) TF_DeleteStatus(m->status);
    free(m);
}

#else

void* iris_tf_load_saved_model(const char* path) { (void)path; fprintf(stderr, "iris: TensorFlow support not enabled at build time\n"); return NULL; }
int   iris_tf_run(void* model, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs) { (void)model;(void)inputs;(void)n_inputs;(void)outputs;(void)n_outputs; fprintf(stderr, "iris: TensorFlow support not enabled at build time\n"); return -1; }
void  iris_tf_free(void* model) { (void)model; }

#endif
