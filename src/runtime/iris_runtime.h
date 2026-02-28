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
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

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
} IrisMutex;

// ---------------------------------------------------------------------------
// Boxing / unboxing
// ---------------------------------------------------------------------------
IrisVal* iris_box_i64(int64_t v);
IrisVal* iris_box_i32(int32_t v);
IrisVal* iris_box_f64(double v);
IrisVal* iris_box_f32(float v);
IrisVal* iris_box_bool(int v);
IrisVal* iris_box_str(const char* s);
int64_t  iris_unbox_i64(IrisVal* v);
double   iris_unbox_f64(IrisVal* v);
int      iris_unbox_bool(IrisVal* v);
char*    iris_unbox_str(IrisVal* v);

// ---------------------------------------------------------------------------
// Print
// ---------------------------------------------------------------------------
void iris_print(void* v);
void iris_print_i64(int64_t v);
void iris_print_i32(int32_t v);
void iris_print_f64(double v);
void iris_print_f32(float v);
void iris_print_bool(int v);
void iris_panic(const char* msg);

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
void     iris_map_set(IrisMap* map, const char* key, IrisVal* val);
IrisVal* iris_map_get(IrisMap* map, const char* key);
int      iris_map_contains(IrisMap* map, const char* key);
void     iris_map_remove(IrisMap* map, const char* key);
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
char*    iris_file_read_all(const char* path);
char*    iris_file_write_all(const char* path, const char* contents);
int      iris_file_exists(const char* path);
IrisList* iris_file_lines(const char* path);

// ---------------------------------------------------------------------------
// Process and environment
// ---------------------------------------------------------------------------
void     iris_set_argv(int argc, char** argv);  /* call from generated main before user main */
IrisList* iris_process_args(void);
char*    iris_env_var(const char* key);

// ---------------------------------------------------------------------------
// Channels and concurrency
// ---------------------------------------------------------------------------
IrisChannel* iris_chan_new(void);
void         iris_chan_send(IrisChannel* chan, IrisVal* val);
IrisVal*     iris_chan_recv(IrisChannel* chan);
void         iris_spawn_fn(void* fn);
void         iris_par_for(void (*fn)(int64_t), int64_t start, int64_t end);
void         iris_barrier(void);

// ---------------------------------------------------------------------------
// Atomics and mutexes
// ---------------------------------------------------------------------------
IrisAtomic* iris_atomic_new(IrisVal* initial);
IrisVal*    iris_atomic_load(IrisAtomic* a);
void        iris_atomic_store(IrisAtomic* a, IrisVal* val);
IrisVal*    iris_atomic_add(IrisAtomic* a, IrisVal* val);
IrisMutex*  iris_mutex_new(void);
IrisVal*    iris_mutex_lock(IrisMutex* mu);
void        iris_mutex_unlock(IrisMutex* mu);

// ---------------------------------------------------------------------------
// Grad (forward-mode autodiff — dual numbers)
// ---------------------------------------------------------------------------
IrisGrad* iris_make_grad(double value, double tangent);
double    iris_grad_value(IrisGrad* g);
double    iris_grad_tangent(IrisGrad* g);

// ---------------------------------------------------------------------------
// Sparse tensors
// ---------------------------------------------------------------------------
IrisSparse* iris_sparsify(IrisList* dense);
IrisList*   iris_densify(IrisSparse* sparse);

// ---------------------------------------------------------------------------
// Non-scalar array fallback (for complex element types)
// ---------------------------------------------------------------------------
IrisList*  iris_alloc_array(void);
IrisVal*   iris_array_load(IrisList* arr, int64_t idx);
void       iris_array_store(IrisList* arr, int64_t idx, IrisVal* val);

// Tensor ops (stub — shape tracking only)
void* iris_tensor_op(void);
void* iris_tensor_load(void* t, ...);
void  iris_tensor_store(void* t, ...);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_RUNTIME_H */
