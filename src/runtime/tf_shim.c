#include "iris_runtime.h"
#include <stdio.h>
#include <stdlib.h>

#if defined(TENSORFLOW_ENABLED)
#include <tensorflow/c/c_api.h>

struct TFModel { TF_Graph* graph; TF_Session* session; TF_Status* status; };

void* iris_tf_load_saved_model(const char* path) {
    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* opts = TF_NewSessionOptions();
    TF_Buffer* run_options = NULL;
    TF_Session* sess = TF_LoadSessionFromSavedModel(opts, NULL, path, NULL, 0, graph, NULL, status);
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
    if (!model) return -1;
    TFModel* m = (TFModel*)model;
    // NOTE: implementing a full generic TF runner requires symbol lookup of input/output names.
    // For now return -1 to indicate unsupported until user wires specific names via stdlib.
    (void)inputs; (void)n_inputs; (void)outputs; (void)n_outputs;
    fprintf(stderr, "iris: TF shim run requires explicit input/output names from stdlib wrapper\n");
    return -1;
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
