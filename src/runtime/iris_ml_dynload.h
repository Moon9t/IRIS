/* iris_ml_dynload.h — lazy runtime loading of ML backend shared libraries.
 *
 * WHY THIS EXISTS
 * ---------------
 * The ML backends used to be selected at *compile* time via -DONNX_RUNTIME_ENABLED
 * / -DTENSORFLOW_ENABLED / -DLIBTORCH_ENABLED and bound at *link* time via
 * -lonnxruntime / -ltensorflow. That had two costs:
 *
 *   1. `iris_runtime.o` differed depending on which SDKs happened to be installed
 *      on the build machine, so there was no single object we could prebuild and
 *      ship — which in turn forced every user to have a C compiler.
 *   2. A binary built with ONNX enabled hard-required onnxruntime.dll at process
 *      start, even for programs that never touch ML.
 *
 * Now the SDK headers describe the ABI at compile time, and the actual library is
 * resolved on first use with dlopen/LoadLibrary. Nothing links against an ML SDK.
 * If the library is absent, the entry points fail gracefully with a clear message
 * instead of the process refusing to start.
 *
 * This follows the same pattern already used for SQLite in iris_runtime.c
 * (fn_##name typedef + p_##name pointer + LOAD macro).
 *
 * Header-only on purpose: each translation unit that needs a backend gets its own
 * handle and pointer table. onnx_shim.c uses only the ORT half, tf_shim.c only the
 * TF half, so there is no duplicated state in practice and no extra object file.
 */

#ifndef IRIS_ML_DYNLOAD_H
#define IRIS_ML_DYNLOAD_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#define IRIS_DL_OPEN(path)      ((void*)LoadLibraryA(path))
#define IRIS_DL_SYM(h, name)    ((void*)GetProcAddress((HMODULE)(h), (name)))
#else
#include <dlfcn.h>
#define IRIS_DL_OPEN(path)      dlopen((path), RTLD_LAZY)
#define IRIS_DL_SYM(h, name)    dlsym((h), (name))
#endif

/* Try each candidate name in order; return the first that loads, or NULL. */
static void* iris_dl_open_any(const char* const* candidates, int n) {
    for (int i = 0; i < n; ++i) {
        void* h = IRIS_DL_OPEN(candidates[i]);
        if (h) return h;
    }
    return NULL;
}

/* ------------------------------------------------------------------------- */
/* ONNX Runtime                                                              */
/* ------------------------------------------------------------------------- */
/* The whole ORT C API hangs off a single exported symbol, OrtGetApiBase().
 * Everything else is a function pointer inside the returned OrtApi struct, so
 * exactly one dlsym is required. */

#include "onnxruntime_c_api.h"

typedef const OrtApiBase* (ORT_API_CALL* fn_OrtGetApiBase)(void);

/* Returns the ORT API table, or NULL if onnxruntime could not be loaded.
 * Caches both success and failure so we only warn once. */
static const OrtApi* iris_ort_api(void) {
    static const OrtApi* cached = NULL;
    static int attempted = 0;

    if (attempted) return cached;
    attempted = 1;

    static const char* const candidates[] = {
#ifdef _WIN32
        "onnxruntime.dll",
        "C:\\onnxruntime\\lib\\onnxruntime.dll",
#elif defined(__APPLE__)
        "libonnxruntime.dylib",
        "/usr/local/lib/libonnxruntime.dylib",
        "/opt/homebrew/lib/libonnxruntime.dylib",
#else
        "libonnxruntime.so",
        "libonnxruntime.so.1",
        "/usr/local/lib/libonnxruntime.so",
#endif
    };

    void* lib = iris_dl_open_any(candidates,
                                (int)(sizeof(candidates) / sizeof(candidates[0])));
    if (!lib) {
        fprintf(stderr,
                "iris: ONNX Runtime not found. Install onnxruntime and ensure the "
                "shared library is on the library search path.\n");
        return NULL;
    }

    fn_OrtGetApiBase get_base = (fn_OrtGetApiBase)IRIS_DL_SYM(lib, "OrtGetApiBase");
    if (!get_base) {
        fprintf(stderr, "iris: onnxruntime loaded but OrtGetApiBase is missing — "
                        "the library may be corrupt or not ONNX Runtime.\n");
        return NULL;
    }

    const OrtApiBase* base = get_base();
    if (!base || !base->GetApi) {
        fprintf(stderr, "iris: OrtGetApiBase() returned no API table.\n");
        return NULL;
    }

    /* Negotiate downward: GetApi(v) returns NULL when the loaded library is older
     * than v. Walking down lets one IRIS build work against a range of ORT
     * versions instead of pinning the exact one present at compile time — which
     * the old statically-linked path could not do. */
    for (uint32_t v = ORT_API_VERSION; v >= 1; --v) {
        const OrtApi* api = base->GetApi(v);
        if (api) {
            cached = api;
            return cached;
        }
    }

    fprintf(stderr,
            "iris: onnxruntime is too old — no compatible API version "
            "(IRIS was built against %d).\n", ORT_API_VERSION);
    return NULL;
}

/* ------------------------------------------------------------------------- */
/* TensorFlow                                                                */
/* ------------------------------------------------------------------------- */
/* The TF C API is flat (plain exported TF_* functions), so each one needs its
 * own dlsym. We declare the ABI by hand rather than vendoring c_api.h: the
 * surface IRIS uses is small and the TF C ABI is stable across releases.
 *
 * The p_##name / #define pair means tf_shim.c's body needs no changes — a call
 * to TF_NewGraph() resolves to (*p_TF_NewGraph)(). */

typedef struct TF_Graph TF_Graph;
typedef struct TF_Session TF_Session;
typedef struct TF_SessionOptions TF_SessionOptions;
typedef struct TF_Status TF_Status;
typedef struct TF_Tensor TF_Tensor;
typedef struct TF_Operation TF_Operation;
typedef struct TF_Buffer TF_Buffer;

typedef struct TF_Output {
    TF_Operation* oper;
    int index;
} TF_Output;

/* Only the values IRIS references. */
#define TF_FLOAT 1
#define TF_OK    0
typedef int TF_DataType;
typedef int TF_Code;

typedef void (*TF_DeallocatorFn)(void* data, size_t len, void* arg);

#define IRIS_TF_FUNCS(X)                                                                   \
    X(TF_Status*,   TF_NewStatus,      (void))                                             \
    X(void,         TF_DeleteStatus,   (TF_Status*))                                       \
    X(void,         TF_SetStatus,      (TF_Status*, TF_Code, const char*))                 \
    X(TF_Code,      TF_GetCode,        (const TF_Status*))                                 \
    X(const char*,  TF_Message,        (const TF_Status*))                                 \
    X(TF_Graph*,    TF_NewGraph,       (void))                                             \
    X(void,         TF_DeleteGraph,    (TF_Graph*))                                        \
    X(TF_SessionOptions*, TF_NewSessionOptions, (void))                                    \
    X(void,         TF_DeleteSessionOptions, (TF_SessionOptions*))                         \
    X(TF_Session*,  TF_LoadSessionFromSavedModel,                                          \
        (const TF_SessionOptions*, const TF_Buffer*, const char*,                          \
         const char* const*, int, TF_Graph*, TF_Buffer*, TF_Status*))                      \
    X(void,         TF_CloseSession,   (TF_Session*, TF_Status*))                          \
    X(void,         TF_DeleteSession,  (TF_Session*, TF_Status*))                          \
    X(void,         TF_SessionRun,                                                         \
        (TF_Session*, const TF_Buffer*, const TF_Output*, TF_Tensor* const*, int,          \
         const TF_Output*, TF_Tensor**, int, const TF_Operation* const*, int,              \
         TF_Buffer*, TF_Status*))                                                          \
    X(TF_Tensor*,   TF_NewTensor,                                                          \
        (TF_DataType, const int64_t*, int, void*, size_t, TF_DeallocatorFn, void*))        \
    X(void,         TF_DeleteTensor,   (TF_Tensor*))                                       \
    X(void*,        TF_TensorData,     (const TF_Tensor*))                                 \
    X(int,          TF_NumDims,        (const TF_Tensor*))                                 \
    X(int64_t,      TF_Dim,            (const TF_Tensor*, int))                            \
    X(TF_Operation*, TF_GraphNextOperation, (TF_Graph*, size_t*))                          \
    X(const char*,  TF_OperationName,  (TF_Operation*))                                    \
    X(const char*,  TF_OperationOpType, (TF_Operation*))

/* Declare fn_NAME typedefs and p_NAME pointers. */
#define IRIS_TF_DECL(ret, name, args) typedef ret (*fn_##name) args; static fn_##name p_##name;
IRIS_TF_FUNCS(IRIS_TF_DECL)
#undef IRIS_TF_DECL

/* Returns 1 if TensorFlow is usable, 0 otherwise. Caches the outcome. */
static int iris_tf_available(void) {
    static int state = -1; /* -1 unknown, 0 unavailable, 1 ready */
    if (state >= 0) return state;

    static const char* const candidates[] = {
#ifdef _WIN32
        "tensorflow.dll",
        "C:\\tensorflow\\lib\\tensorflow.dll",
#elif defined(__APPLE__)
        "libtensorflow.dylib",
        "/usr/local/lib/libtensorflow.dylib",
#else
        "libtensorflow.so",
        "libtensorflow.so.2",
        "/usr/local/lib/libtensorflow.so",
#endif
    };

    void* lib = iris_dl_open_any(candidates,
                                (int)(sizeof(candidates) / sizeof(candidates[0])));
    if (!lib) {
        fprintf(stderr,
                "iris: TensorFlow not found. Install the TensorFlow C library and "
                "ensure it is on the library search path.\n");
        state = 0;
        return 0;
    }

    int missing = 0;
#define IRIS_TF_LOAD(ret, name, args)                                     \
    p_##name = (fn_##name)IRIS_DL_SYM(lib, #name);                        \
    if (!p_##name) { fprintf(stderr, "iris: TensorFlow symbol '%s' not found\n", #name); missing = 1; }
    IRIS_TF_FUNCS(IRIS_TF_LOAD)
#undef IRIS_TF_LOAD

    if (missing) {
        fprintf(stderr, "iris: TensorFlow library is missing required symbols — "
                        "version mismatch?\n");
        state = 0;
        return 0;
    }

    state = 1;
    return 1;
}

/* Route the plain TF_* spellings in tf_shim.c through the loaded pointers. */
#define IRIS_TF_ALIAS(ret, name, args) name
#define TF_NewStatus                  (*p_TF_NewStatus)
#define TF_DeleteStatus               (*p_TF_DeleteStatus)
#define TF_SetStatus                  (*p_TF_SetStatus)
#define TF_GetCode                    (*p_TF_GetCode)
#define TF_Message                    (*p_TF_Message)
#define TF_NewGraph                   (*p_TF_NewGraph)
#define TF_DeleteGraph                (*p_TF_DeleteGraph)
#define TF_NewSessionOptions          (*p_TF_NewSessionOptions)
#define TF_DeleteSessionOptions       (*p_TF_DeleteSessionOptions)
#define TF_LoadSessionFromSavedModel  (*p_TF_LoadSessionFromSavedModel)
#define TF_CloseSession               (*p_TF_CloseSession)
#define TF_DeleteSession              (*p_TF_DeleteSession)
#define TF_SessionRun                 (*p_TF_SessionRun)
#define TF_NewTensor                  (*p_TF_NewTensor)
#define TF_DeleteTensor               (*p_TF_DeleteTensor)
#define TF_TensorData                 (*p_TF_TensorData)
#define TF_NumDims                    (*p_TF_NumDims)
#define TF_Dim                        (*p_TF_Dim)
#define TF_GraphNextOperation         (*p_TF_GraphNextOperation)
#define TF_OperationName              (*p_TF_OperationName)
#define TF_OperationOpType            (*p_TF_OperationOpType)

#endif /* IRIS_ML_DYNLOAD_H */
