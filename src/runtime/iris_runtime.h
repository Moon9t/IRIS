// iris_runtime.h — IRIS Language Runtime Library
// Type definitions, heap structures, and function declarations.
//
// All pointer-typed iris_* functions operate on the types defined here.
// Scalars (i64, f64, i32, f32, bool) are passed by value in LLVM IR;
// everything else is an opaque ptr pointing to one of the structs below.

#ifndef IRIS_RUNTIME_H
#define IRIS_RUNTIME_H

#include <stdint.h>
#include <stddef.h>

#ifdef __wasm__
/* WASM/WASI preview 1: no threading support.
   wasi-libc provides pthread type definitions via sys/types.h (transitively
   through bits/alltypes.h) but libpthread.a is an empty stub archive (8 bytes).
   Include types early for struct definitions below, then provide no-op macros
   so channel, task_group, and spawn code compiles single-threaded.
   pthread_create is special — it MUST execute fn(arg) synchronously. */
#include <sys/types.h>
#ifndef PTHREAD_MUTEX_INITIALIZER
#define PTHREAD_MUTEX_INITIALIZER {0}
#endif
#ifndef PTHREAD_COND_INITIALIZER
#define PTHREAD_COND_INITIALIZER {0}
#endif
#define pthread_mutex_init(mu, a)      ((void)(mu),(void)(a),0)
#define pthread_mutex_destroy(mu)      ((void)(mu),0)
#define pthread_mutex_lock(mu)         ((void)(mu),0)
#define pthread_mutex_unlock(mu)       ((void)(mu),0)
#define pthread_cond_init(c, a)        ((void)(c),(void)(a),0)
#define pthread_cond_destroy(c)        ((void)(c),0)
#define pthread_cond_wait(c, m)        ((void)(c),(void)(m),0)
#define pthread_cond_signal(c)         ((void)(c),0)
#define pthread_detach(t)              ((void)(t),0)
#define pthread_join(t, r)             ((void)(t),(void)(r),0)
/* pthread_create uses void* to avoid depending on pthread_t definition
   (which comes from wasi-libc headers included later). The macro casts
   &t (pthread_t*) to void* — valid since pthread_t is a pointer type. */
int iris_wasm_pthread_create(void* t, const void* a, void*(*fn)(void*), void* arg);
#define pthread_create(t, a, fn, arg)  iris_wasm_pthread_create((void*)(t), a, fn, arg)
#elif defined(_WIN32)
#if defined(__MINGW32__) || defined(__MINGW64__) || defined(_PTHREAD_H)
#include <pthread.h>
#else
typedef struct { long state; } pthread_mutex_t;
typedef struct { long state; } pthread_cond_t;
typedef void* pthread_t;

#ifndef PTHREAD_MUTEX_INITIALIZER
#define PTHREAD_MUTEX_INITIALIZER {0}
#endif
#ifndef PTHREAD_COND_INITIALIZER
#define PTHREAD_COND_INITIALIZER {0}
#endif

static inline int pthread_mutex_init(pthread_mutex_t* mu, const void* attr) {
    (void)mu; (void)attr; return 0;
}
static inline int pthread_mutex_destroy(pthread_mutex_t* mu) {
    (void)mu; return 0;
}
static inline int pthread_mutex_lock(pthread_mutex_t* mu) {
    (void)mu; return 0;
}
static inline int pthread_mutex_unlock(pthread_mutex_t* mu) {
    (void)mu; return 0;
}
static inline int pthread_cond_init(pthread_cond_t* cond, const void* attr) {
    (void)cond; (void)attr; return 0;
}
static inline int pthread_cond_destroy(pthread_cond_t* cond) {
    (void)cond; return 0;
}
static inline int pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mu) {
    (void)cond; (void)mu; return 0;
}
static inline int pthread_cond_signal(pthread_cond_t* cond) {
    (void)cond; return 0;
}
static inline int pthread_cond_timedwait(pthread_cond_t* cond, pthread_mutex_t* mu, const struct timespec* ts) {
    (void)cond; (void)mu; (void)ts; return 0;
}
static inline int pthread_create(pthread_t* t, const void* attr, void* (*fn)(void*), void* arg) {
    (void)attr; if (t) *t = NULL; if (fn) fn(arg); return 0;
}
static inline int pthread_detach(pthread_t t) {
    (void)t; return 0;
}
static inline int pthread_join(pthread_t t, void** ret) {
    (void)t; if (ret) *ret = NULL; return 0;
}
#endif
#else
#include <pthread.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Integer arithmetic with overflow checking
// ---------------------------------------------------------------------------
#include <stdint.h>

int64_t iris_add_checked(int64_t a, int64_t b);
int64_t iris_sub_checked(int64_t a, int64_t b);
int64_t iris_mul_checked(int64_t a, int64_t b);

// ---------------------------------------------------------------------------
// Tagged value type — used for boxed heap values (lists, maps, closures, etc.)
// ---------------------------------------------------------------------------
typedef enum {
    IRIS_TAG_I64     = 0,
    IRIS_TAG_I32     = 1,
    IRIS_TAG_F64     = 2,
    IRIS_TAG_F32     = 3,
    IRIS_TAG_BOOL    = 4,
    IRIS_TAG_STR     = 5,
    IRIS_TAG_LIST    = 6,
    IRIS_TAG_MAP     = 7,
    IRIS_TAG_OPTION  = 8,
    IRIS_TAG_RESULT  = 9,
    IRIS_TAG_CLOSURE = 10,
    IRIS_TAG_TUPLE   = 11,
    IRIS_TAG_STRUCT  = 12,
    IRIS_TAG_CHAN    = 13,
    IRIS_TAG_ATOMIC  = 14,
    IRIS_TAG_GRAD    = 15,
    IRIS_TAG_SPARSE  = 16,
    IRIS_TAG_UNIT    = 17,
    IRIS_TAG_ENUM    = 18,
    IRIS_TAG_MUTEX   = 19,
    IRIS_TAG_TASK_GROUP = 20,
    IRIS_TAG_WEAK_REF = 21,
} IrisTag;

typedef struct IrisVal {
    IrisTag tag;
    union {
        int64_t  i64;
        int32_t  i32;
        double   f64;
        float    f32;
        uint8_t  boolean;
        char*    str;   /* null-terminated, heap-allocated */
        void*    ptr;   /* typed pointer for complex types */
    };
} IrisVal;

typedef struct IrisEnum {
    int64_t  tag;
    IrisVal** fields;
    size_t   len;
} IrisEnum;

// ---------------------------------------------------------------------------
// Complex heap types
// ---------------------------------------------------------------------------

typedef struct {
    IrisVal** data;
    size_t    len;
    size_t    cap;
} IrisList;

typedef struct IrisMapEntry {
    char*                key;
    IrisVal*             val;
    struct IrisMapEntry* next;
} IrisMapEntry;

typedef struct {
    IrisMapEntry** buckets;
    size_t         n_buckets;
    size_t         len;
} IrisMap;

typedef struct {
    uint8_t  has_value;
    IrisVal* value;
} IrisOption;

typedef struct {
    uint8_t  is_ok;
    IrisVal* value;
} IrisResult;

typedef struct {
    double value;
    double tangent;
} IrisGrad;

typedef struct {
    size_t*   indices;
    IrisVal** values;
    size_t    len;
    size_t    cap;
} IrisSparse;

// Channel: blocking bounded FIFO backed by pthreads
typedef struct {
    IrisVal**       buf;
    size_t          cap;
    int64_t         max_cap;
    size_t          head;
    size_t          tail;
    size_t          count;
    pthread_mutex_t mu;
    pthread_cond_t  not_empty;
    pthread_cond_t  not_full;
} IrisChannel;

typedef struct {
    pthread_mutex_t mu;
    IrisVal*        val;
} IrisAtomic;

typedef struct {
    pthread_mutex_t mu;
    IrisVal*        val;
} IrisMutex;

// TaskGroup: structured concurrency group of spawned threads.
typedef struct {
    pthread_t*      handles;
    size_t          count;
    size_t          cap;
    volatile int    cancelled;
    pthread_mutex_t mu;
} IrisTaskGroup;

typedef struct IrisWeakRef {
    void*               target;
    struct IrisWeakRef* next_weak;
} IrisWeakRef;

typedef enum {
    IRIS_RC_BOXED  = 0,
    IRIS_RC_STR    = 1,
    IRIS_RC_LIST   = 2,
    IRIS_RC_MAP    = 3,
    IRIS_RC_OPTION = 4,
    IRIS_RC_RESULT = 5,
    IRIS_RC_CHAN   = 6,
    IRIS_RC_ATOMIC = 7,
    IRIS_RC_MUTEX  = 8,
    IRIS_RC_GRAD   = 9,
    IRIS_RC_SPARSE = 10,
    IRIS_RC_TASK_GROUP = 11,
    IRIS_RC_WEAK_REF   = 12,
} IrisRcKind;

// ---------------------------------------------------------------------------
// Boxing / unboxing
// ---------------------------------------------------------------------------
IrisVal* iris_box_i64(int64_t v);
IrisVal* iris_box_i32(int32_t v);
IrisVal* iris_box_f64(double v);
IrisVal* iris_box_f32(float v);
IrisVal* iris_box_bool(int v);
IrisVal* iris_box_str(const char* s);
IrisVal* iris_box_list(IrisList* list);
IrisVal* iris_box_map(IrisMap* map);
IrisVal* iris_box_option(IrisOption* opt);
IrisVal* iris_box_result(IrisResult* res);
IrisVal* iris_box_chan(IrisChannel* chan);
IrisVal* iris_box_task_group(IrisTaskGroup* tg);
IrisVal* iris_box_weak_ref(IrisWeakRef* w);
IrisVal* iris_box_atomic(IrisAtomic* atomic);
IrisVal* iris_box_mutex(IrisMutex* mutex);
IrisVal* iris_box_grad(IrisGrad* grad);
IrisVal* iris_box_sparse(IrisSparse* sparse);
int64_t  iris_unbox_i64(IrisVal* v);
double   iris_unbox_f64(IrisVal* v);
int      iris_unbox_bool(IrisVal* v);
char*    iris_unbox_str(IrisVal* v);
IrisList* iris_unbox_list(IrisVal* v);
IrisMap*  iris_unbox_map(IrisVal* v);
IrisOption* iris_unbox_option(IrisVal* v);
IrisResult* iris_unbox_result(IrisVal* v);
IrisChannel*  iris_unbox_chan(IrisVal* v);
IrisTaskGroup* iris_unbox_task_group(IrisVal* v);
IrisWeakRef*   iris_unbox_weak_ref(IrisVal* v);
IrisAtomic*   iris_unbox_atomic(IrisVal* v);
IrisMutex*    iris_unbox_mutex(IrisVal* v);
IrisGrad*     iris_unbox_grad(IrisVal* v);
IrisSparse*   iris_unbox_sparse(IrisVal* v);

// ---------------------------------------------------------------------------
// Print
// ---------------------------------------------------------------------------
void iris_print(void* v);
void iris_print_i64(int64_t v);
void iris_print_i32(int32_t v);
void iris_print_f64(double v);
void iris_print_f32(float v);
void iris_print_bool(int v);
void iris_print_str(const char* s);
void iris_panic(const char* msg);
void iris_panic_at(const char* msg, const char* location);
void iris_bounds_check_abort(int64_t index, int64_t size);

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------
char*   iris_read_line(void);
int64_t iris_read_i64(void);
double  iris_read_f64(void);

// ---------------------------------------------------------------------------
// String operations
// ---------------------------------------------------------------------------
int64_t  iris_str_len(const char* s);
char*    iris_str_concat(const char* a, const char* b);
int      iris_str_eq(const char* a, const char* b);
int64_t  iris_str_cmp(const char* a, const char* b);
int      iris_str_contains(const char* s, const char* sub);
int      iris_str_starts_with(const char* s, const char* prefix);
int      iris_str_ends_with(const char* s, const char* suffix);
char*    iris_str_to_upper(const char* s);
char*    iris_str_to_lower(const char* s);
char*    iris_str_trim(const char* s);
char*    iris_str_repeat(const char* s, int64_t n);
int64_t  iris_str_index(const char* s, int64_t i);
char*    iris_str_slice(const char* s, int64_t start, int64_t end);
IrisOption* iris_str_find(const char* s, const char* sub);
char*    iris_str_replace(const char* s, const char* old_s, const char* new_s);
char*    iris_const_str(void);
/* Phase 95: split/join */
IrisList* iris_str_split(const char* s, const char* delim);
char*     iris_str_join(IrisList* list, const char* delim);

// Typed value-to-string conversions
char*    iris_i64_to_str(int64_t v);
char*    iris_i32_to_str(int32_t v);
char*    iris_f64_to_str(double v);
char*    iris_f32_to_str(float v);
char*    iris_bool_to_str(int v);
char*    iris_str_to_str(const char* s);
char*    iris_value_to_str(IrisVal* v);     /* boxed values */

// Parse helpers
IrisOption* iris_parse_i64(const char* s);
IrisOption* iris_parse_f64(const char* s);

// ---------------------------------------------------------------------------
// Math helpers (integer / special cases not covered by LLVM intrinsics)
// ---------------------------------------------------------------------------
int64_t iris_pow_i64(int64_t base, int64_t exp);
int64_t iris_min_i64(int64_t a, int64_t b);
int64_t iris_max_i64(int64_t a, int64_t b);
int64_t iris_abs_i64(int64_t v);
double  iris_sign_f64(double v);
double  iris_clamp_f64(double x, double lo, double hi);
double  iris_pow_f64(double base, double exp);
double  iris_min_f64(double a, double b);
double  iris_max_f64(double a, double b);

// ---------------------------------------------------------------------------
// Option
// ---------------------------------------------------------------------------
IrisOption* iris_make_some(IrisVal* val);
IrisOption* iris_make_none(void);
int         iris_is_some(IrisOption* opt);
IrisVal*    iris_option_unwrap(IrisOption* opt);

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------
IrisResult* iris_make_ok(IrisVal* val);
IrisResult* iris_make_err(IrisVal* val);
int         iris_is_ok(IrisResult* res);
IrisVal*    iris_result_unwrap(IrisResult* res);
IrisVal*    iris_result_unwrap_err(IrisResult* res);

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------
IrisList* iris_list_new(void);
void      iris_list_push(IrisList* list, IrisVal* val);
int64_t   iris_list_len(IrisList* list);
IrisVal*  iris_list_get(IrisList* list, int64_t idx);
void      iris_list_set(IrisList* list, int64_t idx, IrisVal* val);
IrisVal*  iris_list_pop(IrisList* list);

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------
IrisMap* iris_map_new(void);
void     iris_map_set(IrisMap* map, IrisVal* key, IrisVal* val);
IrisOption* iris_map_get(IrisMap* map, IrisVal* key);
int      iris_map_contains(IrisMap* map, IrisVal* key);
void     iris_map_remove(IrisMap* map, IrisVal* key);
int64_t  iris_map_len(IrisMap* map);

// ---------------------------------------------------------------------------
// Extended list operations
// ---------------------------------------------------------------------------
int      iris_list_contains(IrisList* list, IrisVal* val);
void     iris_list_sort(IrisList* list);
IrisList* iris_list_concat(IrisList* a, IrisList* b);
IrisList* iris_list_slice(IrisList* list, int64_t start, int64_t end);

// ---------------------------------------------------------------------------
// Extended map operations
// ---------------------------------------------------------------------------
IrisList* iris_map_keys(IrisMap* map);
IrisList* iris_map_values(IrisMap* map);

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------
IrisResult* iris_file_read_all(const char* path);
IrisResult* iris_file_write_all(const char* path, const char* contents);
int      iris_file_exists(const char* path);
IrisList* iris_file_lines(const char* path);
/* Streaming File I/O */
int64_t  iris_file_open(const char* path, const char* mode);
int      iris_file_close(int64_t handle);
char*    iris_file_read(int64_t handle, int64_t bytes);
int      iris_file_write(int64_t handle, const char* data);

// ---------------------------------------------------------------------------
// Database operations (SQLite via embedded sqlite3)
// ---------------------------------------------------------------------------
int64_t  iris_db_open(const char* path);
int64_t  iris_db_exec(int64_t db, const char* sql);
IrisList* iris_db_query(int64_t db, const char* sql);
int64_t  iris_db_close(int64_t db);
/* Parameterized query/exec helpers (prepared-statement style) */
IrisList* iris_db_query_params(int64_t db, const char* sql, IrisList* params);
int64_t  iris_db_exec_params(int64_t db, const char* sql, IrisList* params);

// ---------------------------------------------------------------------------
// Process and environment
// ---------------------------------------------------------------------------
void     iris_set_argv(int argc, char** argv);  /* call from generated main before user main */
IrisList* iris_process_args(void);
IrisOption* iris_env_var(const char* key);

// ---------------------------------------------------------------------------
// Channels and concurrency
// ---------------------------------------------------------------------------
IrisChannel* iris_chan_new(int64_t capacity);
void         iris_chan_send(IrisChannel* c, IrisVal* val);
IrisVal*     iris_chan_recv(IrisChannel* chan);
int64_t      iris_chan_len(IrisChannel* c);
IrisOption*  iris_chan_try_recv(IrisChannel* c);
int64_t      iris_select(int64_t n, ...);
int          iris_timeout(int64_t ms);
void         iris_spawn_fn(void* fn, void* arg);
void         iris_par_for(void (*fn)(int64_t, void*), int64_t start, int64_t end, void* arg);
IrisList*    iris_par_map(IrisList* list, void* (*fn)(IrisVal*));
void         iris_barrier(void);
IrisTaskGroup* iris_task_group_new(void);
void         iris_task_group_spawn(IrisTaskGroup* tg, void* fn, void* arg);
void         iris_task_group_join(IrisTaskGroup* tg);
void         iris_task_group_cancel(IrisTaskGroup* tg);
int32_t      iris_task_group_join_timeout(IrisTaskGroup* tg, int64_t timeout_ms);
IrisOption*  iris_chan_recv_timeout(IrisChannel* c, int64_t timeout_ms);

IrisWeakRef* iris_weak_ref_new(void* target);
IrisOption*  iris_weak_ref_upgrade(IrisWeakRef* w);
int32_t      iris_weak_ref_alive(IrisWeakRef* w);
void         iris_gc_stats(int64_t* out_alloc, int64_t* out_freed, int64_t* out_cycles, int64_t* out_weak_inval);

// ---- New builtins (called via generic BuiltinCall codegen path) ----
int64_t      iris_list_remove(IrisList* list, int64_t idx);
int64_t      iris_list_insert(IrisList* list, int64_t idx, IrisVal* val);
IrisVal*     iris_map_entries(IrisVal* map);
IrisVal*     iris_recv_timeout(IrisVal* chan, int64_t timeout_ms);
void         iris_chan_send_b(IrisVal* chan, IrisVal* val);
IrisWeakRef* iris_weak_ref(IrisVal* val);
int32_t      iris_weak_alive(IrisWeakRef* w);
IrisVal*     iris_weak_upgrade(IrisWeakRef* w);
IrisVal*     iris_gc_stats_map(void);
int32_t      iris_gc_collect_call(void);

// ---------------------------------------------------------------------------
// Effect handlers — thread-local handler stack with real dispatch
// ---------------------------------------------------------------------------

/// Continuation struct used for effect handler resume.
/// 16 bytes: i32 filled + padding + i64 value. Must match LLVM's %Continuation.
typedef struct {
    int32_t filled;     ///< 0 = not resumed; 1 = resumed
    int64_t value;      ///< The resumed value (boxed to 64 bits)
} Continuation;

/// Push a handler arm onto the thread-local handler stack.
/// fn_name is the handler function name (for interpreter dispatch).
void         iris_push_handler_arm(const char* effect_name, const char* fn_name, int64_t num_args, int32_t has_resume);
/// Store a native function pointer for the most recently pushed handler arm.
void         iris_push_handler_fn(void* fn);
void         iris_push_handler_frame(void);
void         iris_pop_handler(void);

/// Check if any handler on the stack can handle the named effect.
int32_t      iris_can_handle(const char* name);

/// Check if the matching handler for `name` has resume capability.
int32_t      iris_handler_has_resume(const char* name);

/// Resume a continuation: fills the Continuation struct with the value.
void         iris_resume_cont(Continuation* cont, int64_t value);

/// Return the current handler stack depth (0 = no handlers active).
int32_t      iris_handler_depth(void);

/// Find the native function pointer for a handler arm matching `name`.
/// Returns NULL if no handler matches.
void*        iris_find_handler_fn(const char* name);

/// Dispatch an effect call: if a handler is registered, call it; otherwise
/// call the real extern function. All args are packed as int64_t (works for
/// i64 and ptr on x86-64). cont is the Continuation* (NULL for non-resume).
/// Returns the effective result as int64_t (caller casts to expected type).
int64_t      iris_effect_dispatch_or_call(
                 const char* effect_name,
                 void* real_fn,
                 void* cont,
                 int nargs,
                 const int64_t* args);

// ---------------------------------------------------------------------------
// Atomics and mutexes
// ---------------------------------------------------------------------------
IrisAtomic* iris_atomic_new(IrisVal* initial);
IrisVal*    iris_atomic_load(IrisAtomic* a);
void        iris_atomic_store(IrisAtomic* a, IrisVal* val);
IrisVal*    iris_atomic_add(IrisAtomic* a, IrisVal* val);
IrisMutex*  iris_mutex_new(IrisVal* initial);
IrisVal*    iris_mutex_lock(IrisMutex* mu);
void        iris_mutex_unlock(IrisMutex* mu);

// ---------------------------------------------------------------------------
// Grad (forward-mode autodiff — dual numbers)
// ---------------------------------------------------------------------------
IrisGrad* iris_make_grad(double value, double tangent);
double    iris_grad_value(IrisGrad* g);
double    iris_grad_tangent(IrisGrad* g);

// Forward declaration so sparse-tensor prototypes can reference it.
typedef struct IrisTensor IrisTensor;

// ---------------------------------------------------------------------------
// Sparse tensors
// ---------------------------------------------------------------------------
IrisSparse* iris_sparsify(IrisList* dense);
IrisSparse* iris_sparsify_i64_array(int64_t* data, int64_t len);
IrisSparse* iris_sparsify_f64_array(double* data, int64_t len);
IrisList*   iris_densify(IrisSparse* sparse);
int64_t     iris_sparse_nnz(IrisSparse* sparse);

/// Sparsify a dense tensor: extract non-zero (index, value) pairs.
IrisSparse* iris_tensor_sparsify(IrisTensor* t);
/// Densify a sparse representation back to a dense 1D tensor.
IrisTensor* iris_sparse_to_tensor(IrisSparse* sp, int64_t size);
/// Sparse-dense vector dot product.
double      iris_sparse_dot(IrisSparse* sp, IrisTensor* dense);
/// Number of non-zero elements in sparse.
int64_t     iris_sparse_nnz(IrisSparse* sp);

// ---------------------------------------------------------------------------
// Reverse-mode AD runtime
// ---------------------------------------------------------------------------
void*  iris_tape_record(double value, const char* op, int64_t parent_count,
                        void* const* parents, const double* parent_primals);
void   iris_backward(void* loss);
double iris_tape_grad(void* tape_node);

// ---------------------------------------------------------------------------
// Non-scalar array fallback (for complex element types)
// ---------------------------------------------------------------------------
IrisList*  iris_alloc_array(void);
IrisVal*   iris_array_load(IrisList* arr, int64_t idx);
void       iris_array_store(IrisList* arr, int64_t idx, IrisVal* val);

// Tensor ops — real compute
// IrisTensor holds a contiguous f32 buffer with shape metadata.
struct IrisTensor {
    float*  data;     // row-major contiguous data
    int64_t* shape;   // shape array (heap-allocated)
    int32_t  ndim;    // number of dimensions
    int64_t  numel;   // total number of elements (product of shape)
};

IrisTensor* iris_tensor_alloc(int32_t ndim, const int64_t* shape);
void        iris_tensor_free(IrisTensor* t);
int64_t     iris_tensor_pool_init(int64_t limit_bytes);
int64_t     iris_tensor_pool_destroy(void);
IrisTensor* iris_tensor_zeros(int32_t ndim, const int64_t* shape);
IrisTensor* iris_tensor_fill(int32_t ndim, const int64_t* shape, float val);
float       iris_tensor_get(IrisTensor* t, int64_t flat_idx);
void        iris_tensor_set(IrisTensor* t, int64_t flat_idx, float val);
IrisTensor* iris_tensor_matmul(IrisTensor* a, IrisTensor* b);
IrisTensor* iris_tensor_add(IrisTensor* a, IrisTensor* b);
IrisTensor* iris_tensor_sub(IrisTensor* a, IrisTensor* b);
IrisTensor* iris_tensor_mul(IrisTensor* a, IrisTensor* b);
IrisTensor* iris_tensor_div(IrisTensor* a, IrisTensor* b);
IrisTensor* iris_tensor_neg(IrisTensor* t);
IrisTensor* iris_tensor_relu(IrisTensor* t);
IrisTensor* iris_tensor_sigmoid(IrisTensor* t);
IrisTensor* iris_tensor_tanh_act(IrisTensor* t);
IrisTensor* iris_tensor_exp(IrisTensor* t);
IrisTensor* iris_tensor_log(IrisTensor* t);
IrisTensor* iris_tensor_sqrt(IrisTensor* t);
IrisTensor* iris_tensor_abs(IrisTensor* t);
IrisTensor* iris_tensor_reshape(IrisTensor* t, int32_t new_ndim, const int64_t* new_shape);
IrisTensor* iris_tensor_transpose(IrisTensor* t, const int32_t* axes);
IrisTensor* iris_tensor_reduce_sum(IrisTensor* t, int32_t axis, int keepdims);
IrisTensor* iris_tensor_reduce_max(IrisTensor* t, int32_t axis, int keepdims);
IrisTensor* iris_tensor_reduce_mean(IrisTensor* t, int32_t axis, int keepdims);

// Legacy stubs (kept for backward compat, deprecated)
void* iris_tensor_op(void);
void* iris_tensor_load(void* t, ...);
void  iris_tensor_store(void* t, ...);


// ---------------------------------------------------------------------------
// Backend integration (ONNX / LibTorch / TensorFlow)
// ---------------------------------------------------------------------------

// ONNX Runtime shim: load/run/free
void* iris_onnx_session_create(const char* model_path);
int   iris_onnx_session_run(void* session, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs);
void  iris_onnx_session_free(void* session);

// LibTorch (PyTorch) shim: load/run/free (C++ shim exposes C ABI)
void* iris_pytorch_load(const char* model_path);
int   iris_pytorch_run(void* model, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs);
void  iris_pytorch_free(void* model);

// TensorFlow saved model shim (C API based)
void* iris_tf_load_saved_model(const char* path);
int   iris_tf_run(void* model, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs);
void  iris_tf_free(void* model);

// IRIS language-facing ML runtime wrappers.
// Tensor values are represented as (list<f64>, list<i64>) tuple values.
#ifndef IRIS_EXPORT
#ifdef _WIN32
#define IRIS_EXPORT __declspec(dllexport)
#else
#define IRIS_EXPORT __attribute__((visibility("default")))
#endif
#endif

IRIS_EXPORT int64_t  iris_mlrt_onnx_load(const char* model_path);
IRIS_EXPORT int64_t  iris_mlrt_onnx_free(int64_t session);
IRIS_EXPORT IrisVal*  iris_mlrt_onnx_run(int64_t session, IrisVal* input);
IRIS_EXPORT IrisList* iris_mlrt_onnx_run_multi(int64_t session, IrisList* inputs_list);
IRIS_EXPORT int64_t  iris_mlrt_pytorch_load(const char* model_path);
IRIS_EXPORT int64_t  iris_mlrt_pytorch_free(int64_t model);
IRIS_EXPORT IrisVal*  iris_mlrt_pytorch_run(int64_t model, IrisVal* input);
IRIS_EXPORT IrisList* iris_mlrt_pytorch_run_multi(int64_t model, IrisList* inputs_list);
IRIS_EXPORT double   iris_mlrt_pytorch_train_step(int64_t model, IrisList* inputs_list, IrisList* targets_list, double lr);
IRIS_EXPORT int64_t  iris_mlrt_tf_load(const char* model_path);
IRIS_EXPORT int64_t  iris_mlrt_tf_free(int64_t model);
IRIS_EXPORT IrisVal*  iris_mlrt_tf_run(int64_t model, IrisVal* input);
IRIS_EXPORT IrisList* iris_mlrt_tf_run_multi(int64_t model, IrisList* inputs_list);

// Metadata / Reflection APIs
IRIS_EXPORT int64_t     iris_onnx_get_input_count(int64_t session);
IRIS_EXPORT int64_t     iris_onnx_get_output_count(int64_t session);
IRIS_EXPORT const char* iris_onnx_get_input_name(int64_t session, int64_t idx);
IRIS_EXPORT const char* iris_onnx_get_output_name(int64_t session, int64_t idx);
IRIS_EXPORT double      iris_pytorch_train_step(void* model, IrisTensor** inputs, size_t n_inputs, IrisTensor** targets, size_t n_targets, double lr);
// ---------------------------------------------------------------------------
// Time / OS
// ---------------------------------------------------------------------------
int64_t iris_now_ms(void);
void    iris_sleep_ms(int64_t ms);

// ---------------------------------------------------------------------------
// Struct / Tuple / Closure fallback helpers (opaque path)
// ---------------------------------------------------------------------------
IrisVal* iris_make_struct(int nfields, ...);
IrisVal* iris_get_field(IrisVal* s, int32_t idx);
IrisVal* iris_make_tuple(int nelems, ...);
IrisVal* iris_get_element(IrisVal* t, int32_t idx);
IrisVal* iris_make_closure(void* fn, int ncaptures, ...);
IrisVal* iris_call_closure(IrisVal* closure, ...);
void     iris_call_closure_void(IrisVal* closure, ...);

// ---------------------------------------------------------------------------
// Trait Object Helpers (Phase 91)
// ---------------------------------------------------------------------------
// A `dyn Trait` value is a fat pointer: { data_ptr, vtable_id }.
// The vtable is keyed at runtime by an i64 id; the runtime provides a
// forwarder by method-name lookup using a string-encoded `method_name`.
IrisVal* iris_make_trait_object(void* data, void* vtable_id);
// `iris_dyn_call(obj, method_name_ptr, args_count, ...)` performs the
// indirect dispatch by reading `obj`'s vtable_id and matching
// method_name against registered (vtable_id, method_name) pairs.
IrisVal* iris_dyn_call(IrisVal* obj, const char* method_name, int32_t nargs, ...);

// ---------------------------------------------------------------------------
// Enum Variant Helpers
// ---------------------------------------------------------------------------
IrisVal* iris_make_variant(int64_t tag, int32_t nfields, ...);
int64_t  iris_get_variant_tag(IrisVal* v);
IrisVal* iris_extract_variant_field(IrisVal* v, int64_t field_idx);

// ---------------------------------------------------------------------------
// Terminal / Interactive Input
// ---------------------------------------------------------------------------
int64_t iris_read_key(void);           /* read one keypress (no echo, no Enter) */
char*   iris_read_password(const char* prompt); /* read line with echo off */
void    iris_term_clear(void);         /* clear screen */
void    iris_term_cursor(int64_t row, int64_t col); /* move cursor */
void    iris_term_show_cursor(int show); /* show/hide cursor (1=show, 0=hide) */
void    iris_term_set_color(int64_t fg, int64_t bg); /* set ANSI color (0-255) */
void    iris_term_reset(void);         /* reset terminal to normal */
int64_t iris_term_rows(void);          /* terminal height */
int64_t iris_term_cols(void);          /* terminal width */

// ---------------------------------------------------------------------------
// UDP Networking
// ---------------------------------------------------------------------------
int64_t iris_udp_open(int64_t port);      /* open UDP socket bound to port (0 = ephemeral) */
void    iris_udp_send(int64_t fd, const char* addr_port, int64_t data_len); /* send datagram */
char*   iris_udp_recv(int64_t fd);         /* receive datagram, returns "addr:port:data" */
void    iris_udp_close(int64_t fd);

// ---------------------------------------------------------------------------
// HTTP (extended)
// ---------------------------------------------------------------------------
char*   iris_http_request(const char* method, const char* url,
                          const char* body, const char* content_type);

// ---------------------------------------------------------------------------
// TCP Networking
// ---------------------------------------------------------------------------
int64_t iris_tcp_connect(const char* host, int64_t port);
int64_t iris_tcp_listen(int64_t port);
int64_t iris_tcp_accept(int64_t listener);
char*   iris_tcp_read(int64_t conn);
void    iris_tcp_write(int64_t conn, const char* data);
void    iris_tcp_close(int64_t conn);

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------
char*   iris_http_get(const char* url);
char*   iris_http_post(const char* url, const char* body, const char* content_type);
char*   iris_http_post_json(const char* url, const char* json_body);

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------
IrisVal* iris_json_parse(const char* str);
char*    iris_json_stringify(IrisVal* val);

// ---------------------------------------------------------------------------
// Set collection (backed by a sorted list)
// ---------------------------------------------------------------------------
IrisList* iris_set_new(void);
void      iris_set_add(IrisList* set, IrisVal* val);
int       iris_set_contains(IrisList* set, IrisVal* val);
void      iris_set_remove(IrisList* set, IrisVal* val);
int64_t   iris_set_len(IrisList* set);
IrisList* iris_set_to_list(IrisList* set);

// ---------------------------------------------------------------------------
// Regex (POSIX-compatible)
// ---------------------------------------------------------------------------
int       iris_regex_match(const char* pattern, const char* str);
IrisList* iris_regex_find_all(const char* pattern, const char* str);
char*     iris_regex_replace(const char* pattern, const char* str, const char* replacement);
char*     iris_regex_replace_all(const char* pattern, const char* str, const char* replacement);

// ---------------------------------------------------------------------------
// DateTime
// ---------------------------------------------------------------------------
char*     iris_datetime_now(void);
int64_t   iris_datetime_timestamp(void);
char*     iris_datetime_format(int64_t timestamp, const char* fmt);

// ---------------------------------------------------------------------------
// OS / Path
// ---------------------------------------------------------------------------
char*     iris_cwd(void);
IrisList* iris_listdir(const char* path);
char*     iris_path_join(const char* a, const char* b);
int       iris_path_exists(const char* path);
int       iris_mkdir(const char* path);
int       iris_remove_file(const char* path);

// ---------------------------------------------------------------------------
// Type introspection
// ---------------------------------------------------------------------------
char*     iris_type_of(IrisVal* val);

// ---------------------------------------------------------------------------
// Random
// ---------------------------------------------------------------------------
int64_t   iris_seed(int64_t seed);
int64_t   iris_random_seed(void);
double    iris_random(void);
int64_t   iris_random_range(int64_t lo, int64_t hi);

// ---------------------------------------------------------------------------
// Hashing / Encoding
// ---------------------------------------------------------------------------
int64_t   iris_hash(const char* str);
char*     iris_base64_encode(const char* str);
char*     iris_base64_decode(const char* str);

// ---------------------------------------------------------------------------
// String extras
// ---------------------------------------------------------------------------
char*     iris_char_at(const char* str, int64_t idx);
char*     iris_str_reverse(const char* str);

// ---------------------------------------------------------------------------
// Phase 105: Extended builtins
// ---------------------------------------------------------------------------

// -- String extras --
char*     iris_str_pad_left(const char* str, int64_t width, const char* pad);
char*     iris_str_pad_right(const char* str, int64_t width, const char* pad);
IrisList* iris_str_chars(const char* str);
IrisList* iris_str_bytes(const char* str);
int64_t   iris_str_count(const char* str, const char* sub);

// -- Math constants / predicates --
double    iris_math_pi(void);
double    iris_math_e(void);
double    iris_math_inf(void);
int       iris_is_nan(double x);
int       iris_is_inf(double x);

// -- OS / System --
char*     iris_env_get(const char* key);
void      iris_env_set(const char* key, const char* val);
void      iris_exit_code(int64_t code);
char*     iris_exec_cmd(const char* cmd);
int64_t   iris_pid(void);

// -- Crypto / UUID --
char*     iris_uuid(void);
char*     iris_sha256(const char* input);
char*     iris_hex_encode(const char* input);
char*     iris_hex_decode(const char* input);

// -- Deque --
IrisList* iris_deque_new(void);
IrisList* iris_deque_push_front(IrisList* dq, int64_t val);
IrisList* iris_deque_push_back(IrisList* dq, int64_t val);
int64_t   iris_deque_pop_front(IrisList* dq);
int64_t   iris_deque_pop_back(IrisList* dq);
int64_t   iris_deque_len(IrisList* dq);
int64_t   iris_deque_front(IrisList* dq);
int64_t   iris_deque_back(IrisList* dq);

// -- BitSet --
IrisList* iris_bitset_new(int64_t nbits);
IrisList* iris_bitset_set(IrisList* bs, int64_t pos);
int       iris_bitset_get(IrisList* bs, int64_t pos);
int64_t   iris_bitset_count(IrisList* bs);
IrisList* iris_bitset_clear(IrisList* bs, int64_t pos);

// -- FFI --
void*     iris_ffi_open(const char* path);
int64_t   iris_ffi_call(void* handle, const char* func_name);
int       iris_ffi_close(void* handle);
// Expanded C FFI with typed arguments (up to 6 i64 args)
int64_t   iris_ffi_call_i64(void* handle, const char* func_name, int64_t* args, int nargs);
double    iris_ffi_call_f64(void* handle, const char* func_name, int64_t* args, int nargs);
const char* iris_ffi_call_str(void* handle, const char* func_name, int64_t* args, int nargs);
void      iris_ffi_call_void(void* handle, const char* func_name, int64_t* args, int nargs);

/* FFI out-parameter cells — let IRIS own memory and pass its address, so C
 * functions that return through a pointer become callable. See the block
 * comment in iris_runtime.c. */
int64_t      iris_ffi_out_new(int64_t nbytes);
void         iris_ffi_out_free(int64_t cell);
int64_t      iris_ffi_out_sizeof_f64(void);
int64_t      iris_ffi_out_sizeof_i64(void);
double       iris_ffi_out_get_f64(int64_t cell, int64_t index);
int64_t      iris_ffi_out_get_i64(int64_t cell, int64_t index);
int32_t      iris_ffi_out_get_i32(int64_t cell, int64_t index);
char*        iris_ffi_out_get_str(int64_t cell);
void         iris_ffi_out_set_f64(int64_t cell, int64_t index, double v);
void         iris_ffi_out_set_i64(int64_t cell, int64_t index, int64_t v);
// Python FFI
const char* iris_python_eval(const char* code);
int64_t   iris_python_exec(const char* code_or_path);
const char* iris_python_call(const char* module, const char* func, const char* args_json);
const char* iris_python_version(void);
// Rust FFI (aliases for C FFI — Rust cdylibs export extern "C" symbols)
void*     iris_rust_lib_open(const char* path);
int64_t   iris_rust_call_i64(void* handle, const char* func_name, int64_t* args, int nargs);
double    iris_rust_call_f64(void* handle, const char* func_name, int64_t* args, int nargs);
void      iris_rust_call_void(void* handle, const char* func_name, int64_t* args, int nargs);

// -- Functional list ops --
double    iris_list_sum(IrisList* list);
int64_t   iris_list_min(IrisList* list);
int64_t   iris_list_max(IrisList* list);
int64_t   iris_list_index_of(IrisList* list, int64_t val);
int64_t   iris_list_count(IrisList* list, int64_t val);
IrisList* iris_list_reverse(IrisList* list);
IrisList* iris_list_take(IrisList* list, int64_t n);
IrisList* iris_list_drop(IrisList* list, int64_t n);

// -- Concurrency extras --
int64_t   iris_thread_count(void);

// -- Reference Counting GC --
// Each heap-allocated value (IrisVal*) has a reference count stored in
// a separate side table. iris_retain increments, iris_release decrements
// and frees when the count reaches zero.
void      iris_retain(void* ptr);
void      iris_release(void* ptr);
void      iris_retain_kind(void* ptr, int32_t kind);
void      iris_release_kind(void* ptr, int32_t kind);
int64_t   iris_refcount(void* ptr);
void      iris_gc_collect(void);   // Force collection of zero-refcount objects.
int64_t   iris_gc_stats_allocated(void);  // Total live allocations.
int64_t   iris_gc_stats_freed(void);      // Total freed since start.

// -- Security / Sandboxing --
// Check whether an operation is allowed by the current sandbox policy.
// Returns 0 if allowed, non-zero if denied.
int       iris_sandbox_check_fs_read(const char* path);
int       iris_sandbox_check_fs_write(const char* path);
int       iris_sandbox_check_network(const char* host);
int       iris_sandbox_check_ffi(const char* lib_path);
void      iris_sandbox_set_policy(int allow_fs, int allow_net, int allow_ffi);

// ── Adaptive AI runtime (std.adaptive) ────────────────────────────────────
typedef struct IrisAdaptiveState IrisAdaptiveState;

typedef struct {
    double  mean_error;
    double  max_error;
    int64_t observations;
    int64_t errors;
    double  last_risk;
    double  confidence;
} IrisRiskMetrics;

typedef struct {
    double  mean;
    double  variance;
    double  lower_95;
    double  upper_95;
    double  confidence;
} IrisUncertainty;

// Internal implementations (IrisAdaptiveState*)
IrisAdaptiveState* iris_adaptive_new_impl(const char* name, int64_t n_params,
                                          double learning_rate, double risk_threshold);
void               iris_adaptive_free_impl(IrisAdaptiveState* state);
const char*        iris_adaptive_name_impl(IrisAdaptiveState* state);
double             iris_adaptive_get_param_impl(IrisAdaptiveState* state, int64_t idx);
void               iris_adaptive_set_param_impl(IrisAdaptiveState* state, int64_t idx, double value);
int64_t            iris_adaptive_n_params_impl(IrisAdaptiveState* state);
double             iris_adaptive_learning_rate_impl(IrisAdaptiveState* state);
void               iris_adaptive_set_learning_rate_impl(IrisAdaptiveState* state, double lr);
void               iris_adaptive_observe_impl(IrisAdaptiveState* state,
                                              const double* inputs, int64_t n_inputs, double target);
double             iris_adaptive_predict_impl(IrisAdaptiveState* state,
                                              const double* inputs, int64_t n_inputs);
double             iris_adaptive_train_batch_impl(IrisAdaptiveState* state,
                                                  const double* inputs, int64_t n_samples,
                                                  int64_t n_features, const double* targets);
void               iris_adaptive_record_error_impl(IrisAdaptiveState* state, double error);
IrisRiskMetrics    iris_adaptive_get_risk_impl(IrisAdaptiveState* state);
int                iris_adaptive_is_unsafe_impl(IrisAdaptiveState* state);
void               iris_adaptive_set_risk_threshold_impl(IrisAdaptiveState* state, double threshold);
IrisUncertainty    iris_adaptive_predict_with_uncertainty_impl(IrisAdaptiveState* state,
                                                               const double* inputs, int64_t n_inputs);
double             iris_adaptive_uncertainty_bayes_update_impl(IrisAdaptiveState* state,
                                                                double prior_mean, double prior_var,
                                                                double observation, double obs_var);
int                iris_adaptive_should_retrain_impl(IrisAdaptiveState* state);
double             iris_adaptive_auto_retrain_impl(IrisAdaptiveState* state,
                                                   const double* inputs, int64_t n_samples,
                                                   int64_t n_features, const double* targets);
void               iris_adaptive_set_retrain_threshold_impl(IrisAdaptiveState* state, double threshold);
void               iris_adaptive_set_min_observations_for_retrain_impl(IrisAdaptiveState* state, int64_t n);
void               iris_adaptive_adapt_threshold_impl(IrisAdaptiveState* state, double observed_error);
double             iris_adaptive_current_threshold_impl(IrisAdaptiveState* state);
int64_t            iris_adaptive_observation_count_impl(IrisAdaptiveState* state);
double             iris_adaptive_mean_error_impl(IrisAdaptiveState* state);
void               iris_adaptive_reset_stats_impl(IrisAdaptiveState* state);

// Extern-compatible wrappers (int64_t handle, matching adaptive.iris extern def)
int64_t      iris_adaptive_new(const char* name, int64_t n_params,
                                double learning_rate, double risk_threshold);
int64_t      iris_adaptive_free(int64_t handle);
const char*  iris_adaptive_name(int64_t handle);
double       iris_adaptive_get_param(int64_t handle, int64_t idx);
int64_t      iris_adaptive_set_param(int64_t handle, int64_t idx, double value);
int64_t      iris_adaptive_n_params(int64_t handle);
double       iris_adaptive_learning_rate(int64_t handle);
int64_t      iris_adaptive_set_learning_rate(int64_t handle, double lr);
int64_t      iris_adaptive_record_error(int64_t handle, double error);
int          iris_adaptive_is_unsafe(int64_t handle);
int64_t      iris_adaptive_set_risk_threshold(int64_t handle, double threshold);
int          iris_adaptive_should_retrain(int64_t handle);
int64_t      iris_adaptive_set_retrain_threshold(int64_t handle, double threshold);
int64_t      iris_adaptive_set_min_observations_for_retrain(int64_t handle, int64_t n);
int64_t      iris_adaptive_adapt_threshold(int64_t handle, double observed_error);
double       iris_adaptive_current_threshold(int64_t handle);
int64_t      iris_adaptive_observation_count(int64_t handle);
double       iris_adaptive_mean_error(int64_t handle);
int64_t      iris_adaptive_reset_stats(int64_t handle);
double       iris_adaptive_uncertainty_bayes_update(int64_t handle,
                                                     double prior_mean, double prior_var,
                                                     double observation, double obs_var);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_RUNTIME_H */
