// iris_runtime.c — IRIS Language Runtime Library
//
// Implements all iris_* functions declared in iris_runtime.h.
// Memory model: malloc-based arena — allocations are never explicitly freed
// (suitable for scripting and ML workloads that run-and-exit).
// Concurrency: real pthreads for spawn, par_for, channels, atomics.

#include "iris_runtime.h"

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>
#include <stdarg.h>
#include <errno.h>
#include <time.h>
#include <inttypes.h>

#include <sys/types.h>

/* WASM/WASI: wasi-libc provides standard C + POSIX headers below.
   Socket/terminal headers exist in wasi-libc but the function
   declarations are absent in WASI preview 1 (no networking/termios).
   WASI preview 2 (__wasip2__) has wasi-sockets for networking. */
/* Stub flag: 1 on WASM P1 (no networking), 0 on native/P2 */
#if defined(__wasm__)
  #if !defined(__wasip2__)
    #define __IRIS_WASM_STUB 1
  #endif
  #include <unistd.h>
  #include <dirent.h>
  #include <sys/stat.h>
  #include <dlfcn.h>
  #include <poll.h>
  #ifdef __wasip2__
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
  #endif
#elif defined(_WIN32)
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <windows.h>
  #include <winhttp.h>
  #include <direct.h>
  #include <io.h>
  #include <conio.h>        /* _getch() for read_key */
  #pragma comment(lib, "ws2_32.lib")
  #pragma comment(lib, "winhttp.lib")
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <unistd.h>
  #include <dirent.h>
  #include <sys/stat.h>
  #include <sys/ioctl.h>    /* TIOCGWINSZ for term_rows/cols */
  #include <termios.h>      /* tcgetattr/tcsetattr for read_key */
  #include <dlfcn.h>
#endif

// ---------------------------------------------------------------------------
// Internal memory helpers
// ---------------------------------------------------------------------------

// Dynamic allocations registry to differentiate from static string constants
static pthread_mutex_t ds_mu = PTHREAD_MUTEX_INITIALIZER;
static void** ds_table = NULL;
static size_t ds_len = 0;
static size_t ds_cap = 0;

static void ds_add(void* ptr) {
    if (!ptr) return;
    pthread_mutex_lock(&ds_mu);
    if (ds_len >= ds_cap) {
        ds_cap = ds_cap == 0 ? 1024 : ds_cap * 2;
        ds_table = realloc(ds_table, ds_cap * sizeof(void*));
    }
    ds_table[ds_len++] = ptr;
    pthread_mutex_unlock(&ds_mu);
}

static int ds_contains_and_remove(void* ptr) {
    if (!ptr) return 0;
    pthread_mutex_lock(&ds_mu);
    for (size_t i = 0; i < ds_len; i++) {
        if (ds_table[i] == ptr) {
            ds_table[i] = ds_table[ds_len - 1];
            ds_len--;
            pthread_mutex_unlock(&ds_mu);
            return 1;
        }
    }
    pthread_mutex_unlock(&ds_mu);
    return 0;
}

static void* xmalloc(size_t n) {
    void* p = malloc(n);
    if (!p) { fprintf(stderr, "iris: out of memory\n"); abort(); }
    ds_add(p);
    return p;
}

static void* xcalloc(size_t n, size_t sz) {
    void* p = calloc(n, sz);
    if (!p) { fprintf(stderr, "iris: out of memory\n"); abort(); }
    ds_add(p);
    return p;
}

static void* xrealloc(void* p, size_t n) {
    pthread_mutex_lock(&ds_mu);
    for (size_t i = 0; i < ds_len; i++) {
        if (ds_table[i] == p) {
            ds_table[i] = ds_table[ds_len - 1];
            ds_len--;
            break;
        }
    }
    pthread_mutex_unlock(&ds_mu);

    void* q = realloc(p, n);
    if (!q) { fprintf(stderr, "iris: out of memory\n"); abort(); }
    ds_add(q);
    return q;
}

static char* xstrdup(const char* s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char* d = xmalloc(n);
    memcpy(d, s, n);
    return d;
}

// ---------------------------------------------------------------------------
// Boxing / Unboxing
// ---------------------------------------------------------------------------

static IrisVal* box_heap_ref(IrisTag tag, void* ptr, int32_t rc_kind) {
    IrisVal* r = xmalloc(sizeof(IrisVal));
    r->tag = tag;
    r->ptr = ptr;
    if (ptr) iris_retain_kind(ptr, rc_kind);
    return r;
}

static void* unbox_heap_ref(IrisVal* v, IrisTag expected_tag, const char* name) {
    if (!v) return NULL;
    if (v->tag == expected_tag) return v->ptr;
    fprintf(stderr, "iris: %s type mismatch (tag=%d)\n", name, v->tag);
    abort();
}

IrisVal* iris_box_i64(int64_t v) {
    IrisVal* r = xmalloc(sizeof(IrisVal));
    r->tag = IRIS_TAG_I64;  r->i64 = v;
    return r;
}
IrisVal* iris_box_i32(int32_t v) {
    IrisVal* r = xmalloc(sizeof(IrisVal));
    r->tag = IRIS_TAG_I32;  r->i32 = v;
    return r;
}
IrisVal* iris_box_f64(double v) {
    IrisVal* r = xmalloc(sizeof(IrisVal));
    r->tag = IRIS_TAG_F64;  r->f64 = v;
    return r;
}
IrisVal* iris_box_f32(float v) {
    IrisVal* r = xmalloc(sizeof(IrisVal));
    r->tag = IRIS_TAG_F32;  r->f32 = v;
    return r;
}
IrisVal* iris_box_bool(int v) {
    IrisVal* r = xmalloc(sizeof(IrisVal));
    r->tag = IRIS_TAG_BOOL; r->boolean = (uint8_t)(v != 0);
    return r;
}
IrisVal* iris_box_str(const char* s) {
    IrisVal* r = xmalloc(sizeof(IrisVal));
    r->tag = IRIS_TAG_STR;  r->str = xstrdup(s);
    if (r->str) iris_retain_kind(r->str, IRIS_RC_STR);
    return r;
}
IrisVal* iris_box_list(IrisList* list) {
    return box_heap_ref(IRIS_TAG_LIST, list, IRIS_RC_LIST);
}
IrisVal* iris_box_map(IrisMap* map) {
    return box_heap_ref(IRIS_TAG_MAP, map, IRIS_RC_MAP);
}
IrisVal* iris_box_option(IrisOption* opt) {
    return box_heap_ref(IRIS_TAG_OPTION, opt, IRIS_RC_OPTION);
}
IrisVal* iris_box_result(IrisResult* res) {
    return box_heap_ref(IRIS_TAG_RESULT, res, IRIS_RC_RESULT);
}
IrisVal* iris_box_chan(IrisChannel* chan) {
    return box_heap_ref(IRIS_TAG_CHAN, chan, IRIS_RC_CHAN);
}
IrisVal* iris_box_task_group(IrisTaskGroup* tg) {
    return box_heap_ref(IRIS_TAG_TASK_GROUP, tg, IRIS_RC_TASK_GROUP);
}
IrisVal* iris_box_weak_ref(IrisWeakRef* w) {
    return box_heap_ref(IRIS_TAG_WEAK_REF, w, IRIS_RC_WEAK_REF);
}
IrisVal* iris_box_atomic(IrisAtomic* atomic) {
    return box_heap_ref(IRIS_TAG_ATOMIC, atomic, IRIS_RC_ATOMIC);
}
IrisVal* iris_box_mutex(IrisMutex* mutex) {
    return box_heap_ref(IRIS_TAG_MUTEX, mutex, IRIS_RC_MUTEX);
}
IrisVal* iris_box_grad(IrisGrad* grad) {
    return box_heap_ref(IRIS_TAG_GRAD, grad, IRIS_RC_GRAD);
}
IrisVal* iris_box_sparse(IrisSparse* sparse) {
    return box_heap_ref(IRIS_TAG_SPARSE, sparse, IRIS_RC_SPARSE);
}

int64_t iris_unbox_i64(IrisVal* v) {
    if (!v) return 0;
    switch (v->tag) {
        case IRIS_TAG_I64:  return v->i64;
        case IRIS_TAG_I32:  return (int64_t)v->i32;
        case IRIS_TAG_F64:  return (int64_t)v->f64;
        case IRIS_TAG_F32:  return (int64_t)v->f32;
        default: fprintf(stderr, "iris: unbox_i64 type mismatch (tag=%d)\n", v->tag); abort();
    }
}
double iris_unbox_f64(IrisVal* v) {
    if (!v) return 0.0;
    switch (v->tag) {
        case IRIS_TAG_F64:  return v->f64;
        case IRIS_TAG_F32:  return (double)v->f32;
        case IRIS_TAG_I64:  return (double)v->i64;
        case IRIS_TAG_I32:  return (double)v->i32;
        default: fprintf(stderr, "iris: unbox_f64 type mismatch (tag=%d)\n", v->tag); abort();
    }
}
int iris_unbox_bool(IrisVal* v) {
    if (!v) return 0;
    if (v->tag == IRIS_TAG_BOOL) return (int)v->boolean;
    fprintf(stderr, "iris: unbox_bool type mismatch (tag=%d)\n", v->tag); abort();
}
char* iris_unbox_str(IrisVal* v) {
    if (!v) return (char*)"";
    if (v->tag == IRIS_TAG_STR) return v->str;
    fprintf(stderr, "iris: unbox_str type mismatch (tag=%d)\n", v->tag); abort();
}
IrisList* iris_unbox_list(IrisVal* v) {
    return (IrisList*)unbox_heap_ref(v, IRIS_TAG_LIST, "unbox_list");
}
IrisMap* iris_unbox_map(IrisVal* v) {
    return (IrisMap*)unbox_heap_ref(v, IRIS_TAG_MAP, "unbox_map");
}
IrisOption* iris_unbox_option(IrisVal* v) {
    return (IrisOption*)unbox_heap_ref(v, IRIS_TAG_OPTION, "unbox_option");
}
IrisResult* iris_unbox_result(IrisVal* v) {
    return (IrisResult*)unbox_heap_ref(v, IRIS_TAG_RESULT, "unbox_result");
}
IrisChannel* iris_unbox_chan(IrisVal* v) {
    return (IrisChannel*)unbox_heap_ref(v, IRIS_TAG_CHAN, "unbox_chan");
}
IrisTaskGroup* iris_unbox_task_group(IrisVal* v) {
    return (IrisTaskGroup*)unbox_heap_ref(v, IRIS_TAG_TASK_GROUP, "unbox_task_group");
}
IrisWeakRef* iris_unbox_weak_ref(IrisVal* v) {
    return (IrisWeakRef*)unbox_heap_ref(v, IRIS_TAG_WEAK_REF, "unbox_weak_ref");
}
IrisAtomic* iris_unbox_atomic(IrisVal* v) {
    return (IrisAtomic*)unbox_heap_ref(v, IRIS_TAG_ATOMIC, "unbox_atomic");
}
IrisMutex* iris_unbox_mutex(IrisVal* v) {
    return (IrisMutex*)unbox_heap_ref(v, IRIS_TAG_MUTEX, "unbox_mutex");
}
IrisGrad* iris_unbox_grad(IrisVal* v) {
    return (IrisGrad*)unbox_heap_ref(v, IRIS_TAG_GRAD, "unbox_grad");
}
IrisSparse* iris_unbox_sparse(IrisVal* v) {
    return (IrisSparse*)unbox_heap_ref(v, IRIS_TAG_SPARSE, "unbox_sparse");
}

// ---------------------------------------------------------------------------
// Internal print helper
// ---------------------------------------------------------------------------

static void print_val_inline(IrisVal* v) {
    if (!v) { printf("unit"); return; }
    switch (v->tag) {
        case IRIS_TAG_I64:  printf("%ld",  (long)v->i64);              break;
        case IRIS_TAG_I32:  printf("%d",   v->i32);                    break;
        case IRIS_TAG_F64:  printf("%g",   v->f64);                    break;
        case IRIS_TAG_F32:  printf("%g",   (double)v->f32);            break;
        case IRIS_TAG_BOOL: printf("%s",   v->boolean ? "true" : "false"); break;
        case IRIS_TAG_STR:  printf("%s",   v->str);                    break;
        case IRIS_TAG_UNIT: printf("unit");                            break;
        case IRIS_TAG_ENUM: {
            IrisEnum* e = (IrisEnum*)v->ptr;
            printf("variant(%" PRId64 ")", e ? e->tag : 0);
            break;
        }
        case IRIS_TAG_OPTION: {
            IrisOption* o = (IrisOption*)v->ptr;
            if (o && o->has_value) { printf("some("); print_val_inline(o->value); printf(")"); }
            else printf("none");
            break;
        }
        case IRIS_TAG_RESULT: {
            IrisResult* r = (IrisResult*)v->ptr;
            if (r->is_ok) { printf("ok(");  print_val_inline(r->value); printf(")"); }
            else           { printf("err("); print_val_inline(r->value); printf(")"); }
            break;
        }
        case IRIS_TAG_LIST: {
            IrisList* l = (IrisList*)v->ptr;
            printf("[");
            for (size_t i = 0; i < l->len; i++) {
                if (i > 0) printf(", ");
                print_val_inline(l->data[i]);
            }
            printf("]");
            break;
        }
        default: printf("<val:%d>", v->tag); break;
    }
}

// ---------------------------------------------------------------------------
// Print (public)
// ---------------------------------------------------------------------------

void iris_print(void* v) {
    if (!v) { printf("unit\n"); return; }
    print_val_inline((IrisVal*)v);
    printf("\n");
}
void iris_print_i64(int64_t v)  { printf("%" PRId64 "\n", v); }
void iris_print_i32(int32_t v)  { printf("%d\n", v); }
void iris_print_f64(double v) {
    /* Print integer-valued doubles without decimal to match interpreter output */
    if (v == (double)(long long)v && v > -1e15 && v < 1e15)
        printf("%lld\n", (long long)v);
    else
        printf("%g\n", v);
}
void iris_print_f32(float v)    { iris_print_f64((double)v); }
void iris_print_bool(int v)     { printf("%s\n", v ? "true" : "false"); }
void iris_print_str(const char* s)  { printf("%s\n", s ? s : ""); }

void iris_panic(const char* msg) {
    fprintf(stderr, "\x1b[1;31mpanic\x1b[0m: %s\n", msg ? msg : "(null)");
    fflush(stderr);
    abort();
}

void iris_bounds_check_abort(int64_t index, int64_t size) {
    fprintf(stderr, "\x1b[1;31mbounds error\x1b[0m: index %ld out of bounds (size=%ld)\n",
            (long)index, (long)size);
    fflush(stderr);
    abort();
}

/* iris_panic_at — like iris_panic but includes a compile-time source location
 * string (e.g. "in function 'foo'") embedded by the IRIS LLVM codegen. */
void iris_panic_at(const char* msg, const char* location) {
    if (location && *location) {
        fprintf(stderr, "\x1b[1;31mpanic\x1b[0m: %s\n    at %s\n",
                msg ? msg : "(null)", location);
    } else {
        fprintf(stderr, "\x1b[1;31mpanic\x1b[0m: %s\n", msg ? msg : "(null)");
    }
    fflush(stderr);
    abort();
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

char* iris_read_line(void) {
    char buf[4096];
    if (!fgets(buf, sizeof(buf), stdin)) return xstrdup("");
    size_t n = strlen(buf);
    if (n > 0 && buf[n-1] == '\n') buf[--n] = '\0';
    return xstrdup(buf);
}
int64_t iris_read_i64(void) {
    int64_t v = 0;
    if (scanf("%ld", (long*)&v) != 1) v = 0;
    return v;
}
double iris_read_f64(void) {
    double v = 0.0;
    if (scanf("%lf", &v) != 1) v = 0.0;
    return v;
}

// ---------------------------------------------------------------------------
// String operations
// ---------------------------------------------------------------------------

int64_t iris_str_len(const char* s) { 
    return (int64_t)strlen(s); 
}

char* iris_str_concat(const char* a, const char* b) {
    size_t la = strlen(a), lb = strlen(b);
    char* r = xmalloc(la + lb + 1);
    memcpy(r, a, la);
    memcpy(r + la, b, lb + 1);
    return r;
}

int iris_str_eq(const char* a, const char* b)            { return strcmp(a, b) == 0; }
int64_t iris_str_cmp(const char* a, const char* b)      { return (int64_t)strcmp(a, b); }
int iris_str_contains(const char* s, const char* sub)    { return strstr(s, sub) != NULL; }
int iris_str_starts_with(const char* s, const char* pfx) { return strncmp(s, pfx, strlen(pfx)) == 0; }
int iris_str_ends_with(const char* s, const char* sfx) {
    size_t ls = strlen(s), lp = strlen(sfx);
    return lp <= ls && strcmp(s + ls - lp, sfx) == 0;
}

char* iris_str_to_upper(const char* s) {
    size_t n = strlen(s);
    char* r = xmalloc(n + 1);
    for (size_t i = 0; i <= n; i++) r[i] = (char)toupper((unsigned char)s[i]);
    return r;
}
char* iris_str_to_lower(const char* s) {
    size_t n = strlen(s);
    char* r = xmalloc(n + 1);
    for (size_t i = 0; i <= n; i++) r[i] = (char)tolower((unsigned char)s[i]);
    return r;
}
char* iris_str_trim(const char* s) {
    while (isspace((unsigned char)*s)) s++;
    const char* end = s + strlen(s);
    while (end > s && isspace((unsigned char)*(end-1))) end--;
    size_t n = (size_t)(end - s);
    char* r = xmalloc(n + 1);
    memcpy(r, s, n);  r[n] = '\0';
    return r;
}
char* iris_str_repeat(const char* s, int64_t n) {
    if (n <= 0) { char* r = xmalloc(1); r[0] = '\0'; return r; }
    size_t ls = strlen(s), total = ls * (size_t)n;
    char* r = xmalloc(total + 1);
    for (int64_t i = 0; i < n; i++) memcpy(r + (size_t)i * ls, s, ls);
    r[total] = '\0';
    return r;
}
int64_t iris_str_index(const char* s, int64_t i) {
    size_t n = strlen(s);
    if (i < 0 || (size_t)i >= n) {
        fprintf(stderr, "iris: string index %ld out of bounds (len=%zu)\n", (long)i, n);
        abort();
    }
    return (int64_t)(unsigned char)s[i];
}
char* iris_str_slice(const char* s, int64_t start, int64_t end_idx) {
    size_t n = strlen(s);
    if (start < 0) start = 0;
    if ((size_t)end_idx > n) end_idx = (int64_t)n;
    if (start >= end_idx) { char* r = xmalloc(1); r[0] = '\0'; return r; }
    size_t len = (size_t)(end_idx - start);
    char* r = xmalloc(len + 1);
    memcpy(r, s + start, len);  r[len] = '\0';
    return r;
}
IrisOption* iris_str_find(const char* s, const char* sub) {
    IrisOption* opt = xmalloc(sizeof(IrisOption));
    const char* p = strstr(s, sub);
    if (!p) { opt->has_value = 0; opt->value = NULL; }
    else     { opt->has_value = 1; opt->value = iris_box_i64((int64_t)(p - s)); }
    return opt;
}
char* iris_str_replace(const char* s, const char* old_s, const char* new_s) {
    size_t ls = strlen(s), lo = strlen(old_s), ln = strlen(new_s);
    if (lo == 0) return xstrdup(s);
    size_t count = 0;
    const char* p = s;
    while ((p = strstr(p, old_s)) != NULL) { count++; p += lo; }
    size_t rlen = ls + count * (ln - lo);
    char* r = xmalloc(rlen + 1);
    char* w = r;
    p = s;
    const char* next;
    while ((next = strstr(p, old_s)) != NULL) {
        size_t seg = (size_t)(next - p);
        memcpy(w, p, seg);  w += seg;
        memcpy(w, new_s, ln); w += ln;
        p = next + lo;
    }
    strcpy(w, p);
    return r;
}
char* iris_const_str(void) { return (char*)""; }  /* fallback; should never be reached */

/* Phase 95: split/join */
IrisList* iris_str_split(const char* s, const char* delim) {
    IrisList* result = iris_list_new();
    size_t dlen = strlen(delim);
    if (dlen == 0) {
        /* split into individual characters */
        while (*s) {
            char buf[5] = {0};
            /* simple single-byte split (ASCII) */
            buf[0] = *s++;
            IrisVal* v = (IrisVal*)xmalloc(sizeof(IrisVal));
            v->tag = IRIS_TAG_STR;
            v->str = xstrdup(buf);
            iris_list_push(result, v);
        }
        return result;
    }
    const char* p = s;
    const char* found;
    while ((found = strstr(p, delim)) != NULL) {
        size_t seg = (size_t)(found - p);
        char* part = (char*)xmalloc(seg + 1);
        memcpy(part, p, seg);
        part[seg] = '\0';
        IrisVal* v = (IrisVal*)xmalloc(sizeof(IrisVal));
        v->tag = IRIS_TAG_STR;
        v->str = part;
        iris_list_push(result, v);
        p = found + dlen;
    }
    /* last segment */
    IrisVal* v = (IrisVal*)xmalloc(sizeof(IrisVal));
    v->tag = IRIS_TAG_STR;
    v->str = xstrdup(p);
    iris_list_push(result, v);
    return result;
}

char* iris_str_join(IrisList* list, const char* delim) {
    if (!list || list->len == 0) return xstrdup("");
    size_t dlen = strlen(delim);
    size_t total = 0;
    for (int64_t i = 0; i < list->len; i++) {
        IrisVal* v = list->data[i];
        if (v && v->tag == IRIS_TAG_STR && v->str) total += strlen(v->str);
        if (i + 1 < list->len) total += dlen;
    }
    char* r = (char*)xmalloc(total + 1);
    char* w = r;
    for (int64_t i = 0; i < list->len; i++) {
        IrisVal* v = list->data[i];
        if (v && v->tag == IRIS_TAG_STR && v->str) {
            size_t sl = strlen(v->str);
            memcpy(w, v->str, sl);
            w += sl;
        }
        if (i + 1 < list->len) {
            memcpy(w, delim, dlen);
            w += dlen;
        }
    }
    *w = '\0';
    return r;
}

// ---------------------------------------------------------------------------
// Typed value-to-string conversions
// ---------------------------------------------------------------------------

char* iris_i64_to_str(int64_t v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%ld", (long)v);
    return xstrdup(buf);
}
char* iris_i32_to_str(int32_t v) {
    char buf[24];
    snprintf(buf, sizeof(buf), "%d", v);
    return xstrdup(buf);
}
char* iris_f64_to_str(double v) {
    char buf[64];
    if (v == (double)(long long)v && v > -1e15 && v < 1e15)
        snprintf(buf, sizeof(buf), "%lld", (long long)v);
    else
        snprintf(buf, sizeof(buf), "%g", v);
    return xstrdup(buf);
}
char* iris_f32_to_str(float v)  { return iris_f64_to_str((double)v); }
char* iris_bool_to_str(int v)   { return xstrdup(v ? "true" : "false"); }
char* iris_str_to_str(const char* s) { return xstrdup(s); }

char* iris_value_to_str(IrisVal* v) {
    if (!v) return xstrdup("unit");
    switch (v->tag) {
        case IRIS_TAG_I64:  return iris_i64_to_str(v->i64);
        case IRIS_TAG_I32:  return iris_i32_to_str(v->i32);
        case IRIS_TAG_F64:  return iris_f64_to_str(v->f64);
        case IRIS_TAG_F32:  return iris_f32_to_str(v->f32);
        case IRIS_TAG_BOOL: return iris_bool_to_str(v->boolean);
        case IRIS_TAG_STR:  return xstrdup(v->str);
        case IRIS_TAG_UNIT: return xstrdup("unit");
        default: {
            char buf[32];
            snprintf(buf, sizeof(buf), "<val:%d>", v->tag);
            return xstrdup(buf);
        }
    }
}

// Parse helpers
IrisOption* iris_parse_i64(const char* s) {
    IrisOption* opt = xmalloc(sizeof(IrisOption));
    char* end;
    errno = 0;
    long long v = strtoll(s, &end, 10);
    if (end == s || *end != '\0' || errno != 0) {
        opt->has_value = 0; opt->value = NULL;
    } else {
        opt->has_value = 1; opt->value = iris_box_i64((int64_t)v);
    }
    return opt;
}
IrisOption* iris_parse_f64(const char* s) {
    IrisOption* opt = xmalloc(sizeof(IrisOption));
    char* end;
    errno = 0;
    double v = strtod(s, &end);
    if (end == s || *end != '\0' || errno != 0) {
        opt->has_value = 0; opt->value = NULL;
    } else {
        opt->has_value = 1; opt->value = iris_box_f64(v);
    }
    return opt;
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

int64_t iris_pow_i64(int64_t base, int64_t exp) {
    if (exp < 0) return 0;
    int64_t result = 1;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}
int64_t iris_min_i64(int64_t a, int64_t b) { return a < b ? a : b; }
int64_t iris_max_i64(int64_t a, int64_t b) { return a > b ? a : b; }
int64_t iris_abs_i64(int64_t v)            { return v < 0 ? -v : v; }
double  iris_sign_f64(double v)            { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0); }
double  iris_clamp_f64(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
double  iris_pow_f64(double base, double exp) { return pow(base, exp); }
double  iris_min_f64(double a, double b)     { return a < b ? a : b; }
double  iris_max_f64(double a, double b)     { return a > b ? a : b; }

// ---------------------------------------------------------------------------
// Integer overflow-checked arithmetic
// ---------------------------------------------------------------------------

#if defined(__has_builtin) && __has_builtin(__builtin_add_overflow)

int64_t iris_add_checked(int64_t a, int64_t b) {
    int64_t result;
    if (__builtin_add_overflow(a, b, &result)) {
        fprintf(stderr, "iris: integer overflow in addition (%" PRId64 " + %" PRId64 ")\n", a, b);
        abort();
    }
    return result;
}
int64_t iris_sub_checked(int64_t a, int64_t b) {
    int64_t result;
    if (__builtin_sub_overflow(a, b, &result)) {
        fprintf(stderr, "iris: integer overflow in subtraction (%" PRId64 " - %" PRId64 ")\n", a, b);
        abort();
    }
    return result;
}
int64_t iris_mul_checked(int64_t a, int64_t b) {
    int64_t result;
    if (__builtin_mul_overflow(a, b, &result)) {
        fprintf(stderr, "iris: integer overflow in multiplication (%" PRId64 " * %" PRId64 ")\n", a, b);
        abort();
    }
    return result;
}

#else

#include <limits.h>

int64_t iris_add_checked(int64_t a, int64_t b) {
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) {
        fprintf(stderr, "iris: integer overflow in addition\n");
        abort();
    }
    return a + b;
}
int64_t iris_sub_checked(int64_t a, int64_t b) {
    if ((b > 0 && a < INT64_MIN + b) || (b < 0 && a > INT64_MAX + b)) {
        fprintf(stderr, "iris: integer overflow in subtraction\n");
        abort();
    }
    return a - b;
}
int64_t iris_mul_checked(int64_t a, int64_t b) {
    if (a == 0 || b == 0) return 0;
    int64_t result = a * b;
    if (a != result / b) {
        fprintf(stderr, "iris: integer overflow in multiplication\n");
        abort();
    }
    return result;
}

#endif

// ---------------------------------------------------------------------------
// Option
// ---------------------------------------------------------------------------

IrisOption* iris_make_some(IrisVal* val) {
    IrisOption* o = xmalloc(sizeof(IrisOption));
    o->has_value = 1;
    o->value = val;
    if (val) iris_retain(val);
    return o;
}
IrisOption* iris_make_none(void) {
    IrisOption* o = xmalloc(sizeof(IrisOption));
    o->has_value = 0;  o->value = NULL;
    return o;
}
int      iris_is_some(IrisOption* opt) { return opt ? opt->has_value : 0; }
IrisVal* iris_option_unwrap(IrisOption* opt) {
    if (!opt || !opt->has_value) { fprintf(stderr, "iris: unwrap called on none\n"); abort(); }
    return opt->value;
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

IrisResult* iris_make_ok(IrisVal* val) {
    IrisResult* r = xmalloc(sizeof(IrisResult));
    r->is_ok = 1;
    r->value = val;
    if (val) iris_retain(val);
    return r;
}
IrisResult* iris_make_err(IrisVal* val) {
    IrisResult* r = xmalloc(sizeof(IrisResult));
    r->is_ok = 0;
    r->value = val;
    if (val) iris_retain(val);
    return r;
}
int      iris_is_ok(IrisResult* res) {
    return res ? res->is_ok : 0;
}
IrisVal* iris_result_unwrap(IrisResult* res) {
    if (!res || !res->is_ok) { fprintf(stderr, "iris: unwrap called on err\n"); abort(); }
    return res->value;
}
IrisVal* iris_result_unwrap_err(IrisResult* res) {
    if (!res || res->is_ok) { fprintf(stderr, "iris: unwrap_err called on ok\n"); abort(); }
    return res->value;
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

/* ---- Collection lock ---------------------------------------------------
 *
 * `iris_list_push` grew the buffer with `xrealloc` and advanced `len++` with no
 * synchronisation at all, so two threads inside a `par for` could lose an
 * update, or one could move the buffer while the other wrote through the stale
 * pointer. That is a data race reachable from ordinary safe IRIS:
 *
 *     par for i in 0..2000 { push(shared, i); }
 *
 * It produced the right answer on every run of an eleven-run probe, which is
 * the most dangerous possible result -- two accidents were hiding it. Every
 * push passes through `iris_retain`, which takes a *global* refcount mutex and
 * serialises most of the window; and `iris_par_for` created one thread per
 * iteration, so on a 2-core box thread startup dominated and overlap was
 * minimal. Both disappear on a larger machine.
 *
 * A single global lock rather than per-list: the refcount mutex above is
 * already taken on every element operation, so the serialisation exists
 * regardless and a second global adds little. Per-list mutexes are the upgrade
 * path if profiling ever shows this to matter.
 *
 * Recursive, so that a public entry point which internally calls another
 * (`iris_str_split` builds its result with `iris_list_push`) cannot deadlock.
 * Auditing every internal caller instead would be a standing trap for anyone
 * adding a collection helper later.
 */
static pthread_mutex_t coll_mu = PTHREAD_MUTEX_INITIALIZER;

static void coll_lock(void)   { pthread_mutex_lock(&coll_mu); }
static void coll_unlock(void) { pthread_mutex_unlock(&coll_mu); }

IrisList* iris_list_new(void) {
    IrisList* l = xcalloc(1, sizeof(IrisList));
    l->cap  = 8;
    l->data = xmalloc(sizeof(IrisVal*) * l->cap);
    return l;
}
void iris_list_push(IrisList* l, IrisVal* val) {
    coll_lock();
    if (l->len == l->cap) {
        l->cap *= 2;
        l->data = xrealloc(l->data, sizeof(IrisVal*) * l->cap);
    }
    if (val) iris_retain(val);
    l->data[l->len++] = val;
    coll_unlock();
}
int64_t  iris_list_len(IrisList* l) { return (int64_t)l->len; }
IrisVal* iris_list_get(IrisList* l, int64_t idx) {
    if (idx < 0 || (size_t)idx >= l->len) {
        fprintf(stderr, "iris: list index %ld out of bounds (len=%zu)\n", (long)idx, l->len);
        abort();
    }
    return l->data[idx];
}
void iris_list_set(IrisList* l, int64_t idx, IrisVal* val) {
    coll_lock();
    if (idx < 0 || (size_t)idx >= l->len) {
        coll_unlock();
        fprintf(stderr, "iris: list set index %ld out of bounds\n", (long)idx);
        abort();
    }
    if (val) iris_retain(val);
    if (l->data[idx]) iris_release(l->data[idx]);
    l->data[idx] = val;
    coll_unlock();
}
IrisVal* iris_list_pop(IrisList* l) {
    coll_lock();
    if (l->len == 0) {
        coll_unlock();
        fprintf(stderr, "iris: pop on empty list\n");
        abort();
    }
    IrisVal* v = l->data[--l->len];
    coll_unlock();
    return v;
}

// ---------------------------------------------------------------------------
// Map (separate-chaining hash map, string keys)
// ---------------------------------------------------------------------------

#define MAP_INIT_BUCKETS 16u

static size_t hash_str(const char* s) {
    size_t h = 5381;
    while (*s) h = h * 33u + (unsigned char)*s++;
    return h;
}

IrisMap* iris_map_new(void) {
    IrisMap* m = xcalloc(1, sizeof(IrisMap));
    m->n_buckets = MAP_INIT_BUCKETS;
    m->buckets   = xcalloc(m->n_buckets, sizeof(IrisMapEntry*));
    return m;
}
void iris_map_set(IrisMap* m, IrisVal* key, IrisVal* val) {
    /* Stringify the key BEFORE taking the lock: `iris_value_to_str` walks a
     * list value and would re-enter `iris_list_get`, deadlocking on a
     * non-recursive mutex. */
    char* key_str = iris_value_to_str(key);
    coll_lock();
    size_t h = hash_str(key_str) % m->n_buckets;
    for (IrisMapEntry* e = m->buckets[h]; e; e = e->next) {
        if (strcmp(e->key, key_str) == 0) {
            if (val) iris_retain(val);
            if (e->val) iris_release(e->val);
            e->val = val;
            free(key_str);
            coll_unlock();
            return;
        }
    }
    IrisMapEntry* e = xmalloc(sizeof(IrisMapEntry));
    e->key = key_str;
    e->val = val;
    if (val) iris_retain(val);
    e->next = m->buckets[h];
    m->buckets[h] = e;  m->len++;
    coll_unlock();
}
IrisOption* iris_map_get(IrisMap* m, IrisVal* key) {
    char* key_str = iris_value_to_str(key);
    size_t h = hash_str(key_str) % m->n_buckets;
    for (IrisMapEntry* e = m->buckets[h]; e; e = e->next)
        if (strcmp(e->key, key_str) == 0) {
            free(key_str);
            return iris_make_some(e->val);
        }
    free(key_str);
    return iris_make_none();
}
int iris_map_contains(IrisMap* m, IrisVal* key) {
    char* key_str = iris_value_to_str(key);
    size_t h = hash_str(key_str) % m->n_buckets;
    for (IrisMapEntry* e = m->buckets[h]; e; e = e->next)
        if (strcmp(e->key, key_str) == 0) {
            free(key_str);
            return 1;
        }
    free(key_str);
    return 0;
}
void iris_map_remove(IrisMap* m, IrisVal* key) {
    char* key_str = iris_value_to_str(key);   /* see iris_map_set */
    coll_lock();
    size_t h = hash_str(key_str) % m->n_buckets;
    IrisMapEntry** pp = &m->buckets[h];
    while (*pp) {
        if (strcmp((*pp)->key, key_str) == 0) {
            IrisMapEntry* doomed = *pp;
            *pp = doomed->next;
            if (doomed->val) iris_release(doomed->val);
            free(doomed->key);
            free(doomed);
            m->len--;
            free(key_str);
            coll_unlock();
            return;
        }
        pp = &(*pp)->next;
    }
    free(key_str);
    coll_unlock();
}
int64_t iris_map_len(IrisMap* m) { return (int64_t)m->len; }

// ---------------------------------------------------------------------------
// Extended list operations
// ---------------------------------------------------------------------------

static int iris_val_equal(IrisVal* a, IrisVal* b) {
    if (a == b) return 1;
    if (!a || !b) return 0;
    if (a->tag != b->tag) return 0;
    switch (a->tag) {
        case IRIS_TAG_I64:  return a->i64 == b->i64;
        case IRIS_TAG_I32:  return a->i32 == b->i32;
        case IRIS_TAG_F64:  return a->f64 == b->f64;
        case IRIS_TAG_F32:  return a->f32 == b->f32;
        case IRIS_TAG_BOOL: return a->boolean == b->boolean;
        case IRIS_TAG_STR:  return (a->str && b->str && strcmp(a->str, b->str) == 0);
        default: return 0;
    }
}

int iris_list_contains(IrisList* l, IrisVal* val) {
    if (!l || !val) return 0;
    for (size_t i = 0; i < l->len; i++) {
        if (iris_val_equal(l->data[i], val)) return 1;
    }
    return 0;
}

static int iris_val_compare(IrisVal* a, IrisVal* b) {
    if (!a && !b) return 0;
    if (!a) return -1;
    if (!b) return 1;
    if (a->tag != b->tag) return (int)a->tag - (int)b->tag;
    switch (a->tag) {
        case IRIS_TAG_I64:  return (a->i64 > b->i64) ? 1 : (a->i64 < b->i64 ? -1 : 0);
        case IRIS_TAG_I32:  return (a->i32 > b->i32) ? 1 : (a->i32 < b->i32 ? -1 : 0);
        case IRIS_TAG_F64:  return (a->f64 > b->f64) ? 1 : (a->f64 < b->f64 ? -1 : 0);
        case IRIS_TAG_F32:  return (a->f32 > b->f32) ? 1 : (a->f32 < b->f32 ? -1 : 0);
        case IRIS_TAG_BOOL: return (int)a->boolean - (int)b->boolean;
        case IRIS_TAG_STR:
            if (!a->str && !b->str) return 0;
            if (!a->str) return -1;
            if (!b->str) return 1;
            return strcmp(a->str, b->str);
        default: return 0;
    }
}

/* ---- stable merge sort (O(n log n), preserves equal-element order) ---- */
static void iris_merge(IrisVal** arr, IrisVal** tmp, size_t lo, size_t mid, size_t hi) {
    size_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        if (iris_val_compare(arr[i], arr[j]) <= 0)
            tmp[k++] = arr[i++];
        else
            tmp[k++] = arr[j++];
    }
    while (i < mid) tmp[k++] = arr[i++];
    while (j < hi)  tmp[k++] = arr[j++];
    for (size_t x = lo; x < hi; x++) arr[x] = tmp[x];
}

static void iris_merge_sort_rec(IrisVal** arr, IrisVal** tmp, size_t lo, size_t hi) {
    if (hi - lo <= 1) return;
    size_t mid = lo + (hi - lo) / 2;
    iris_merge_sort_rec(arr, tmp, lo, mid);
    iris_merge_sort_rec(arr, tmp, mid, hi);
    iris_merge(arr, tmp, lo, mid, hi);
}

void iris_list_sort(IrisList* l) {
    if (!l || l->len <= 1) return;
    IrisVal** tmp = (IrisVal**)malloc(l->len * sizeof(IrisVal*));
    if (!tmp) return;  /* OOM — leave list unsorted rather than crash */
    iris_merge_sort_rec(l->data, tmp, 0, l->len);
    free(tmp);
}

IrisList* iris_list_concat(IrisList* a, IrisList* b) {
    IrisList* r = iris_list_new();
    if (a) for (size_t i = 0; i < a->len; i++) iris_list_push(r, a->data[i]);
    if (b) for (size_t i = 0; i < b->len; i++) iris_list_push(r, b->data[i]);
    return r;
}

IrisList* iris_list_slice(IrisList* l, int64_t start, int64_t end_idx) {
    IrisList* r = iris_list_new();
    if (!l) return r;
    size_t len = l->len;
    if (start < 0) start = 0;
    if ((size_t)end_idx > len) end_idx = (int64_t)len;
    if (start >= end_idx) return r;
    for (int64_t i = start; i < end_idx; i++) iris_list_push(r, l->data[(size_t)i]);
    return r;
}

// ---------------------------------------------------------------------------
// Extended map operations
// ---------------------------------------------------------------------------

IrisList* iris_map_keys(IrisMap* m) {
    IrisList* r = iris_list_new();
    if (!m) return r;
    for (size_t b = 0; b < m->n_buckets; b++) {
        for (IrisMapEntry* e = m->buckets[b]; e; e = e->next) {
            iris_list_push(r, iris_box_str(e->key));
        }
    }
    return r;
}

IrisList* iris_map_values(IrisMap* m) {
    IrisList* r = iris_list_new();
    if (!m) return r;
    for (size_t b = 0; b < m->n_buckets; b++) {
        for (IrisMapEntry* e = m->buckets[b]; e; e = e->next) {
            iris_list_push(r, e->val);
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

IrisResult* iris_file_read_all(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        return iris_make_err(iris_box_str("Failed to open file for reading"));
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return iris_make_err(iris_box_str("Fseek end error"));
    }
    long sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return iris_make_err(iris_box_str("Ftell error"));
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return iris_make_err(iris_box_str("Fseek set error"));
    }
    size_t size = (size_t)sz;
    char* buf = xmalloc(size + 1);
    size_t n = fread(buf, 1, size, f);
    buf[n] = '\0';
    fclose(f);
    IrisVal* contents = iris_box_str(buf);
    free(buf);
    IrisResult* r = iris_make_ok(contents);
    return r;
}

IrisResult* iris_file_write_all(const char* path, const char* contents) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        return iris_make_err(iris_box_str("Failed to open file for writing"));
    }
    size_t len = strlen(contents);
    size_t written = fwrite(contents, 1, len, f);
    fclose(f);
    if (written != len) {
        return iris_make_err(iris_box_str("Incomplete write"));
    }
    IrisVal* b = iris_box_i64((int64_t)written);
    IrisResult* r = iris_make_ok(b);
    return r;
}

int iris_file_exists(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    fclose(f);
    return 1;
}

IrisList* iris_file_lines(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return iris_list_new();
    IrisList* r = iris_list_new();
    char buf[8192];
    while (fgets(buf, sizeof(buf), f)) {
        size_t n = strlen(buf);
        if (n > 0 && buf[n-1] == '\n') buf[--n] = '\0';
        iris_list_push(r, iris_box_str(buf));
    }
    fclose(f);
    return r;
}

/* Streaming File I/O */
int64_t iris_file_open(const char* path, const char* mode) {
    if (!path || !mode) return 0;
    FILE* f = fopen(path, mode);
    return (int64_t)f;
}

int iris_file_close(int64_t handle) {
    FILE* f = (FILE*)handle;
    if (!f) return 0;
    return fclose(f) == 0 ? 1 : 0;
}

char* iris_file_read(int64_t handle, int64_t bytes) {
    FILE* f = (FILE*)handle;
    if (!f || bytes <= 0) {
        char* empty = xmalloc(1);
        empty[0] = '\0';
        return empty;
    }
    char* buf = xmalloc(bytes + 1);
    size_t n = fread(buf, 1, (size_t)bytes, f);
    buf[n] = '\0';
    return buf;
}

int iris_file_write(int64_t handle, const char* data) {
    FILE* f = (FILE*)handle;
    if (!f || !data) return 0;
    size_t len = strlen(data);
    size_t written = fwrite(data, 1, len, f);
    return written == len ? 1 : 0;
}


// ---------------------------------------------------------------------------
// Database operations (SQLite via dynamic loading)
// ---------------------------------------------------------------------------

#ifdef _WIN32
#include <windows.h>
static HMODULE sqlite3_lib = NULL;
#else
#include <dlfcn.h>
static void* sqlite3_lib = NULL;
#endif

// SQLite3 type definitions (avoid requiring sqlite3.h)
typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;
#define SQLITE_OK    0
#define SQLITE_ROW   100
#define SQLITE_DONE  101

// Function pointer types
typedef int (*fn_sqlite3_open)(const char*, sqlite3**);
typedef int (*fn_sqlite3_close)(sqlite3*);
typedef int (*fn_sqlite3_exec)(sqlite3*, const char*, void*, void*, char**);
typedef int (*fn_sqlite3_prepare_v2)(sqlite3*, const char*, int, sqlite3_stmt**, const char**);
typedef int (*fn_sqlite3_step)(sqlite3_stmt*);
typedef int (*fn_sqlite3_finalize)(sqlite3_stmt*);
typedef int (*fn_sqlite3_column_count)(sqlite3_stmt*);
typedef const unsigned char* (*fn_sqlite3_column_text)(sqlite3_stmt*, int);
typedef void (*fn_sqlite3_free)(void*);
typedef int (*fn_sqlite3_bind_text)(sqlite3_stmt*, int, const char*, int, void(*)(void*));
typedef int (*fn_sqlite3_bind_double)(sqlite3_stmt*, int, double);
typedef int (*fn_sqlite3_bind_int64)(sqlite3_stmt*, int, long long);

#ifndef SQLITE_TRANSIENT
#define SQLITE_TRANSIENT ((void(*)(void*))-1)
#endif

// Loaded function pointers
static fn_sqlite3_open         p_sqlite3_open = NULL;
static fn_sqlite3_close        p_sqlite3_close = NULL;
static fn_sqlite3_exec         p_sqlite3_exec = NULL;
static fn_sqlite3_prepare_v2   p_sqlite3_prepare_v2 = NULL;
static fn_sqlite3_step         p_sqlite3_step = NULL;
static fn_sqlite3_finalize     p_sqlite3_finalize = NULL;
static fn_sqlite3_column_count p_sqlite3_column_count = NULL;
static fn_sqlite3_column_text  p_sqlite3_column_text = NULL;
static fn_sqlite3_free         p_sqlite3_free = NULL;
static fn_sqlite3_bind_text    p_sqlite3_bind_text = NULL;
static fn_sqlite3_bind_double  p_sqlite3_bind_double = NULL;
static fn_sqlite3_bind_int64   p_sqlite3_bind_int64 = NULL;

static int iris_load_sqlite3(void) {
    if (p_sqlite3_open) return 1; // already loaded
#ifdef _WIN32
    sqlite3_lib = LoadLibraryA("sqlite3.dll");
    if (!sqlite3_lib) return 0;
    #define LOAD(name) p_##name = (fn_##name)GetProcAddress(sqlite3_lib, #name)
#else
    sqlite3_lib = dlopen("libsqlite3.so", 1 /* RTLD_LAZY */);
    if (!sqlite3_lib) sqlite3_lib = dlopen("libsqlite3.dylib", 1);
    if (!sqlite3_lib) return 0;
    #define LOAD(name) p_##name = (fn_##name)dlsym(sqlite3_lib, #name)
#endif
    LOAD(sqlite3_open);
    LOAD(sqlite3_close);
    LOAD(sqlite3_exec);
    LOAD(sqlite3_prepare_v2);
    LOAD(sqlite3_step);
    LOAD(sqlite3_finalize);
    LOAD(sqlite3_column_count);
    LOAD(sqlite3_column_text);
    LOAD(sqlite3_bind_text);
    LOAD(sqlite3_bind_double);
    LOAD(sqlite3_bind_int64);
    LOAD(sqlite3_free);
    #undef LOAD
    return p_sqlite3_open ? 1 : 0;
}

int64_t iris_db_open(const char* path) {
    if (!iris_load_sqlite3()) return 0;
    sqlite3* db = NULL;
    if (p_sqlite3_open(path, &db) != SQLITE_OK) return 0;
    return (int64_t)(intptr_t)db;
}

int64_t iris_db_exec(int64_t db, const char* sql) {
    if (!db || !p_sqlite3_exec) return -1;
    sqlite3* conn = (sqlite3*)(intptr_t)db;
    char* err = NULL;
    int rc = p_sqlite3_exec(conn, sql, NULL, NULL, &err);
    if (err) p_sqlite3_free(err);
    return rc == SQLITE_OK ? 0 : -1;
}

IrisList* iris_db_query(int64_t db, const char* sql) {
    IrisList* rows = iris_list_new();
    if (!db || !p_sqlite3_prepare_v2) return rows;
    sqlite3* conn = (sqlite3*)(intptr_t)db;
    sqlite3_stmt* stmt = NULL;
    if (p_sqlite3_prepare_v2(conn, sql, -1, &stmt, NULL) != SQLITE_OK) return rows;
    int ncols = p_sqlite3_column_count(stmt);
    while (p_sqlite3_step(stmt) == SQLITE_ROW) {
        IrisList* row = iris_list_new();
        for (int i = 0; i < ncols; i++) {
            const unsigned char* txt = p_sqlite3_column_text(stmt, i);
            iris_list_push(row, iris_box_str(txt ? (const char*)txt : ""));
        }
        iris_list_push(rows, iris_box_list(row));
    }
    p_sqlite3_finalize(stmt);
    return rows;
}

IrisList* iris_db_query_params(int64_t db, const char* sql, IrisList* params) {
    IrisList* rows = iris_list_new();
    if (!db || !p_sqlite3_prepare_v2) return rows;
    sqlite3* conn = (sqlite3*)(intptr_t)db;
    sqlite3_stmt* stmt = NULL;
    if (p_sqlite3_prepare_v2(conn, sql, -1, &stmt, NULL) != SQLITE_OK) return rows;
    int nParams = params ? (int)iris_list_len(params) : 0;
    for (int i = 0; i < nParams; i++) {
        IrisVal* v = iris_list_get(params, i);
        if (!v) continue;
        if (v->tag == IRIS_TAG_STR && p_sqlite3_bind_text) {
            const char* txt = iris_unbox_str(v);
            p_sqlite3_bind_text(stmt, i + 1, txt ? txt : "", -1, SQLITE_TRANSIENT);
        } else if (v->tag == IRIS_TAG_F64 && p_sqlite3_bind_double) {
            p_sqlite3_bind_double(stmt, i + 1, v->f64);
        } else if (v->tag == IRIS_TAG_I64 && p_sqlite3_bind_int64) {
            p_sqlite3_bind_int64(stmt, i + 1, (long long)v->i64);
        } else if (p_sqlite3_bind_text) {
            char* s = iris_value_to_str(v);
            p_sqlite3_bind_text(stmt, i + 1, s ? s : "", -1, SQLITE_TRANSIENT);
            if (s) p_sqlite3_free(s);
        }
    }
    int ncols = p_sqlite3_column_count(stmt);
    while (p_sqlite3_step(stmt) == SQLITE_ROW) {
        IrisList* row = iris_list_new();
        for (int i = 0; i < ncols; i++) {
            const unsigned char* txt = p_sqlite3_column_text(stmt, i);
            iris_list_push(row, iris_box_str(txt ? (const char*)txt : ""));
        }
        iris_list_push(rows, iris_box_list(row));
    }
    p_sqlite3_finalize(stmt);
    return rows;
}

int64_t iris_db_exec_params(int64_t db, const char* sql, IrisList* params) {
    if (!db || !p_sqlite3_prepare_v2) return -1;
    sqlite3* conn = (sqlite3*)(intptr_t)db;
    sqlite3_stmt* stmt = NULL;
    if (p_sqlite3_prepare_v2(conn, sql, -1, &stmt, NULL) != SQLITE_OK) return -1;
    int nParams = params ? (int)iris_list_len(params) : 0;
    for (int i = 0; i < nParams; i++) {
        IrisVal* v = iris_list_get(params, i);
        if (!v) continue;
        if (v->tag == IRIS_TAG_STR && p_sqlite3_bind_text) {
            const char* txt = iris_unbox_str(v);
            p_sqlite3_bind_text(stmt, i + 1, txt ? txt : "", -1, SQLITE_TRANSIENT);
        } else if (v->tag == IRIS_TAG_F64 && p_sqlite3_bind_double) {
            p_sqlite3_bind_double(stmt, i + 1, v->f64);
        } else if (v->tag == IRIS_TAG_I64 && p_sqlite3_bind_int64) {
            p_sqlite3_bind_int64(stmt, i + 1, (long long)v->i64);
        } else if (p_sqlite3_bind_text) {
            char* s = iris_value_to_str(v);
            p_sqlite3_bind_text(stmt, i + 1, s ? s : "", -1, SQLITE_TRANSIENT);
            if (s) p_sqlite3_free(s);
        }
    }
    int rc = SQLITE_OK;
    int step_rc = p_sqlite3_step(stmt);
    if (step_rc != SQLITE_DONE && step_rc != SQLITE_ROW) rc = -1;
    p_sqlite3_finalize(stmt);
    return rc == SQLITE_OK ? 0 : -1;
}

int64_t iris_db_close(int64_t db) {
    if (!db || !p_sqlite3_close) return -1;
    sqlite3* conn = (sqlite3*)(intptr_t)db;
    return p_sqlite3_close(conn) == SQLITE_OK ? 0 : -1;
}

// ---------------------------------------------------------------------------
// Process and environment
// ---------------------------------------------------------------------------

static int saved_argc = 0;
static char** saved_argv = NULL;

void iris_set_argv(int argc, char** argv) {
    saved_argc = argc;
    saved_argv = argv;
}

IrisList* iris_process_args(void) {
    IrisList* r = iris_list_new();
    if (!saved_argv) return r;
    for (int i = 0; i < saved_argc; i++)
        iris_list_push(r, iris_box_str(saved_argv[i]));
    return r;
}

IrisOption* iris_env_var(const char* key) {
#ifdef _WIN32
    const char* v = getenv(key);
    if (!v) {
        if (_stricmp(key, "PATH") == 0) {
            v = getenv("Path");
            if (!v) v = getenv("path");
        }
    }
#else
    const char* v = getenv(key);
#endif
    if (v) {
        return iris_make_some(iris_box_str(v));
    } else {
        return iris_make_none();
    }
}

// ---------------------------------------------------------------------------
// Channels and concurrency
// ---------------------------------------------------------------------------

#define CHAN_INIT_CAP 64u

static void chan_grow(IrisChannel* c) {
    size_t new_cap = c->cap * 2;
    IrisVal** new_buf = xmalloc(sizeof(IrisVal*) * new_cap);
    for (size_t i = 0; i < c->count; i++) {
        new_buf[i] = c->buf[(c->head + i) % c->cap];
    }
    free(c->buf);
    c->buf = new_buf;
    c->cap = new_cap;
    c->head = 0;
    c->tail = c->count;
}

/* WASM pthread_create override: runs fn(arg) synchronously since
   WASI preview 1 has no threading. The declaration is in iris_runtime.h
   as a macro redirect; the prototype uses void* to avoid requiring
   pthread_t at header inclusion time. */
#if defined(__wasm__)
int iris_wasm_pthread_create(void* t, const void* a, void*(*fn)(void*), void* arg) {
    (void)a; if(t) *(pthread_t*)t = NULL; if(fn) fn(arg); return 0; }
#endif

IrisChannel* iris_chan_new(int64_t capacity) {
    IrisChannel* c = xmalloc(sizeof(IrisChannel));
    c->cap   = capacity > 0 ? capacity : CHAN_INIT_CAP;
    c->buf   = xmalloc(sizeof(IrisVal*) * c->cap);
    c->head  = c->tail = c->count = 0;
    c->max_cap = capacity; // Store intended capacity, -1 if unbounded
    pthread_mutex_init(&c->mu,        NULL);
    pthread_cond_init (&c->not_empty, NULL);
    pthread_cond_init (&c->not_full,  NULL);
    return c;
}
void iris_chan_send(IrisChannel* c, IrisVal* val) {
    pthread_mutex_lock(&c->mu);
    if (c->max_cap >= 0) {
        while (c->count >= c->max_cap) {
            pthread_cond_wait(&c->not_full, &c->mu);
        }
    } else {
        if (c->count == c->cap) chan_grow(c);
    }
    if (val) iris_retain(val);
    c->buf[c->tail] = val;
    c->tail = (c->tail + 1) % c->cap;
    c->count++;
    pthread_cond_signal(&c->not_empty);
    pthread_mutex_unlock(&c->mu);
}
IrisVal* iris_chan_recv(IrisChannel* c) {
    pthread_mutex_lock(&c->mu);
    while (c->count == 0) pthread_cond_wait(&c->not_empty, &c->mu);
    IrisVal* val = c->buf[c->head];
    c->head = (c->head + 1) % c->cap;
    c->count--;
    if (c->max_cap >= 0) pthread_cond_signal(&c->not_full);
    pthread_mutex_unlock(&c->mu);
    return val;
}

int64_t iris_chan_len(IrisChannel* c) {
    if (!c) return 0;
    pthread_mutex_lock(&c->mu);
    int64_t len = (int64_t)c->count;
    pthread_mutex_unlock(&c->mu);
    return len;
}

IrisOption* iris_chan_try_recv(IrisChannel* c) {
    if (!c) return iris_make_none();
    pthread_mutex_lock(&c->mu);
    if (c->count == 0) {
        pthread_mutex_unlock(&c->mu);
        return iris_make_none();
    }
    IrisVal* val = c->buf[c->head];
    c->head = (c->head + 1) % c->cap;
    c->count--;
    if (c->max_cap >= 0) {
        pthread_cond_signal(&c->not_full);
    }
    pthread_mutex_unlock(&c->mu);
    
    IrisOption* opt = iris_make_some(val);
    if (val) {
        iris_release(val); // Transfer reference from channel buffer to Option
    }
    return opt;
}

int64_t iris_select(int64_t n, ...) {
    va_list args;
    va_start(args, n);
    for (int64_t i = 0; i < n; i++) {
        IrisChannel* c = va_arg(args, IrisChannel*);
        if (c) {
            pthread_mutex_lock(&c->mu);
            if (c->count > 0) {
                pthread_mutex_unlock(&c->mu);
                va_end(args);
                return i;
            }
            pthread_mutex_unlock(&c->mu);
        }
    }
    va_end(args);
    return -1;
}

int iris_timeout(int64_t ms) {
    iris_sleep_ms(ms);
    return 1;
}
void iris_spawn_fn(void* fn, void* arg) {
    pthread_t t;
    /* The spawned trampoline takes a single void* arg (packed captures)
       and returns void*. The detached thread discards the return value. */
    pthread_create(&t, NULL, (void*(*)(void*))fn, arg);
    pthread_detach(t);
}

typedef struct { void (*fn)(int64_t, void*); int64_t i; void* arg; } ParArg;
static void* par_for_worker(void* arg) {
    ParArg* a = (ParArg*)arg;
    a->fn(a->i, a->arg);
    free(a);
    return NULL;
}
/* Worker arguments for a strided share of the iteration space. */
typedef struct {
    void (*fn)(int64_t, void*);
    int64_t start;
    int64_t end;
    int64_t stride;
    void*   arg;
} ParRangeArg;

static void* par_for_range_worker(void* p) {
    ParRangeArg* a = (ParRangeArg*)p;
    for (int64_t i = a->start; i < a->end; i += a->stride) {
        a->fn(i, a->arg);
    }
    return NULL;
}

static int64_t iris_hw_threads(void) {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    int64_t n = (int64_t)si.dwNumberOfProcessors;
#elif defined(_SC_NPROCESSORS_ONLN)
    int64_t n = (int64_t)sysconf(_SC_NPROCESSORS_ONLN);
#else
    int64_t n = 4;
#endif
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    return n;
}

/* A fixed pool striding the iteration space, not one thread per iteration.
 *
 * The previous implementation called pthread_create once per index, so
 * `par for i in 0..2000` created two thousand OS threads. At roughly 1 MB of
 * reserved stack each that is an address-space problem before it is a
 * performance one, and the creation cost so dominated that iterations barely
 * overlapped -- which was one of two accidents hiding the unsynchronised
 * collection mutation now fixed above.
 *
 * Striding rather than contiguous blocks keeps the split even when the body
 * cost varies with the index, which it usually does. */
void iris_par_for(void (*fn)(int64_t, void*), int64_t start, int64_t end, void* arg) {
    int64_t n = end - start;
    if (n <= 0) return;

    int64_t workers = iris_hw_threads();
    if (workers > n) workers = n;
    if (workers <= 1) {
        for (int64_t i = start; i < end; i++) fn(i, arg);
        return;
    }

    pthread_t* threads = xmalloc(sizeof(pthread_t) * (size_t)workers);
    ParRangeArg* args = xmalloc(sizeof(ParRangeArg) * (size_t)workers);
    /* Per-slot, not a high-water mark: a failed create in the middle would
     * otherwise leave an uninitialised handle inside the join range. */
    unsigned char* created = xmalloc((size_t)workers);
    for (int64_t w = 0; w < workers; w++) {
        args[w].fn = fn;
        args[w].start = start + w;
        args[w].end = end;
        args[w].stride = workers;
        args[w].arg = arg;
        if (pthread_create(&threads[w], NULL, par_for_range_worker, &args[w]) == 0) {
            created[w] = 1;
        } else {
            /* Out of threads: run this share inline rather than dropping it. */
            created[w] = 0;
            par_for_range_worker(&args[w]);
        }
    }
    for (int64_t w = 0; w < workers; w++) {
        if (created[w]) pthread_join(threads[w], NULL);
    }
    free(created);
    free(threads);
    free(args);
}

typedef struct {
    void* (*fn)(IrisVal*);
    IrisVal* arg;
    IrisVal* result;
    pthread_mutex_t* mu;
} ParMapArg;

static void* par_map_worker(void* arg) {
    ParMapArg* a = (ParMapArg*)arg;
    a->result = a->fn(a->arg);
    return NULL;
}

IrisList* iris_par_map(IrisList* list, void* (*fn)(IrisVal*)) {
    int64_t n = iris_list_len(list);
    IrisList* results = iris_list_new();
    if (n <= 0) return results;
    pthread_t* threads = xmalloc(sizeof(pthread_t) * (size_t)n);
    ParMapArg* args = xmalloc(sizeof(ParMapArg) * (size_t)n);
    /* Pre-size results list */
    for (int64_t i = 0; i < n; i++) {
        iris_list_push(results, iris_box_i64(0)); /* placeholder */
    }
    for (int64_t i = 0; i < n; i++) {
        args[i].fn = fn;
        args[i].arg = iris_list_get(list, i);
        args[i].result = NULL;
        pthread_create(&threads[i], NULL, par_map_worker, &args[i]);
    }
    for (int64_t i = 0; i < n; i++) {
        pthread_join(threads[i], NULL);
        /* Replace placeholder at index i with actual result */
        ((IrisVal**)results->data)[i] = args[i].result;
    }
    free(threads);
    free(args);
    return results;
}
void iris_barrier(void) { /* no-op outside par_for; par_for already joins all */ }

// ── TaskGroup ──────────────────────────────────────────────────────────
IrisTaskGroup* iris_task_group_new(void) {
    IrisTaskGroup* tg = xmalloc(sizeof(IrisTaskGroup));
    tg->cap = 8;
    tg->count = 0;
    tg->cancelled = 0;
    tg->handles = xmalloc(sizeof(pthread_t) * tg->cap);
    pthread_mutex_init(&tg->mu, NULL);
    return tg;
}
void iris_task_group_spawn(IrisTaskGroup* tg, void* fn, void* arg) {
    pthread_t t;
    pthread_create(&t, NULL, (void*(*)(void*))fn, arg);
    pthread_mutex_lock(&tg->mu);
    if (tg->count == tg->cap) {
        tg->cap *= 2;
        tg->handles = xrealloc(tg->handles, sizeof(pthread_t) * tg->cap);
    }
    tg->handles[tg->count++] = t;
    pthread_mutex_unlock(&tg->mu);
}
void iris_task_group_join(IrisTaskGroup* tg) {
    pthread_mutex_lock(&tg->mu);
    for (size_t i = 0; i < tg->count; i++) {
        pthread_join(tg->handles[i], NULL);
    }
    tg->count = 0;
    pthread_mutex_unlock(&tg->mu);
}
void iris_task_group_cancel(IrisTaskGroup* tg) {
    tg->cancelled = 1;
}

// ---------------------------------------------------------------------------
// Effect handlers — thread-local handler stack with direct LLVM dispatch
// ---------------------------------------------------------------------------

#include <string.h>

#ifdef _MSC_VER
#define strdup _strdup
#endif

#define MAX_HANDLER_ARMS 64
#define MAX_HANDLER_FRAMES 64

typedef struct {
    char* effect_name;
    char* fn_name;
    void* handler_fn;
    int64_t num_args;
    int32_t has_resume;
} HandlerArm;

typedef struct {
    HandlerArm arms[MAX_HANDLER_ARMS];
    int narms;
} HandlerFrame;

#ifdef _MSC_VER
static __declspec(thread) HandlerFrame handler_frames[MAX_HANDLER_FRAMES];
static __declspec(thread) int handler_frame_count = 0;
static __declspec(thread) int handler_cur_narms = 0;
#else
static __thread HandlerFrame handler_frames[MAX_HANDLER_FRAMES];
static __thread int handler_frame_count = 0;
static __thread int handler_cur_narms = 0;
#endif

void iris_push_handler_arm(const char* effect_name, const char* fn_name, int64_t num_args, int32_t has_resume) {
    if (handler_cur_narms >= MAX_HANDLER_ARMS) return;
    HandlerFrame* frame = &handler_frames[handler_frame_count];
    frame->arms[handler_cur_narms].effect_name = strdup(effect_name);
    frame->arms[handler_cur_narms].fn_name = strdup(fn_name);
    frame->arms[handler_cur_narms].handler_fn = NULL;
    frame->arms[handler_cur_narms].num_args = num_args;
    frame->arms[handler_cur_narms].has_resume = has_resume;
    handler_cur_narms++;
}

void iris_push_handler_fn(void* fn) {
    if (handler_cur_narms == 0) return;
    HandlerFrame* frame = &handler_frames[handler_frame_count];
    frame->arms[handler_cur_narms - 1].handler_fn = fn;
}

void iris_push_handler_frame(void) {
    if (handler_frame_count >= MAX_HANDLER_FRAMES) return;
    handler_frames[handler_frame_count].narms = handler_cur_narms;
    handler_cur_narms = 0;
    handler_frame_count++;
}

void iris_pop_handler(void) {
    if (handler_frame_count <= 0) return;
    handler_frame_count--;
    HandlerFrame* frame = &handler_frames[handler_frame_count];
    for (int i = 0; i < frame->narms; i++) {
        free(frame->arms[i].effect_name);
        free(frame->arms[i].fn_name);
    }
    handler_cur_narms = 0;
}

static HandlerArm* find_handler_arm(const char* name) {
    for (int f = handler_frame_count - 1; f >= 0; f--) {
        HandlerFrame* frame = &handler_frames[f];
        for (int a = 0; a < frame->narms; a++) {
            if (strcmp(frame->arms[a].effect_name, name) == 0) {
                return &frame->arms[a];
            }
        }
    }
    return NULL;
}

int32_t iris_can_handle(const char* name) {
    return find_handler_arm(name) != NULL ? 1 : 0;
}

int32_t iris_handler_has_resume(const char* name) {
    HandlerArm* arm = find_handler_arm(name);
    return arm ? arm->has_resume : 0;
}

void iris_resume_cont(Continuation* cont, int64_t value) {
    if (cont == NULL) return;
    cont->filled = 1;
    cont->value = value;
}

int32_t iris_handler_depth(void) {
    return handler_frame_count;
}

void* iris_find_handler_fn(const char* name) {
    HandlerArm* arm = find_handler_arm(name);
    return arm ? arm->handler_fn : NULL;
}

int64_t iris_effect_dispatch_or_call(
    const char* effect_name,
    void* real_fn,
    void* cont,
    int nargs,
    const int64_t* args)
{
    /* Fast path: no handlers active → call real extern directly. */
    if (handler_frame_count == 0) goto call_real;

    {
        void* hfn = iris_find_handler_fn(effect_name);
        if (hfn) {
            /* Handler signature: i64 handler(void* cont, i64, i64, ...) */
            typedef int64_t (*hfn_t)(void*, int64_t, int64_t, int64_t,
                                     int64_t, int64_t, int64_t, int64_t);
            hfn_t fn = (hfn_t)hfn;
            int64_t r = 0;
            switch (nargs) {
                case 0: r = fn(cont, 0, 0, 0, 0, 0, 0, 0); break;
                case 1: r = fn(cont, args[0], 0, 0, 0, 0, 0, 0); break;
                case 2: r = fn(cont, args[0], args[1], 0, 0, 0, 0, 0); break;
                case 3: r = fn(cont, args[0], args[1], args[2], 0, 0, 0, 0); break;
                case 4: r = fn(cont, args[0], args[1], args[2], args[3], 0, 0, 0); break;
                case 5: r = fn(cont, args[0], args[1], args[2], args[3], args[4], 0, 0); break;
                case 6: r = fn(cont, args[0], args[1], args[2], args[3], args[4], args[5], 0); break;
                default: r = fn(cont, args[0], args[1], args[2], args[3], args[4], args[5], args[6]); break;
            }
            /* If resume was triggered, return the continuation value. */
            if (cont) {
                Continuation* c = (Continuation*)cont;
                if (c->filled) return c->value;
            }
            return r;
        }
    }

call_real:
    if (!real_fn) {
        /* No handler and no real function — panic. */
        fprintf(stderr, "error: no handler for effect '%s' and no real implementation\n", effect_name);
        abort();
    }
    {
        /* Real extern signature: i64 real_fn(i64, i64, i64, i64, i64, i64, i64) */
        typedef int64_t (*efn_t)(int64_t, int64_t, int64_t, int64_t,
                                 int64_t, int64_t, int64_t);
        efn_t fn = (efn_t)real_fn;
        switch (nargs) {
            case 0: return fn(0, 0, 0, 0, 0, 0, 0);
            case 1: return fn(args[0], 0, 0, 0, 0, 0, 0);
            case 2: return fn(args[0], args[1], 0, 0, 0, 0, 0);
            case 3: return fn(args[0], args[1], args[2], 0, 0, 0, 0);
            case 4: return fn(args[0], args[1], args[2], args[3], 0, 0, 0);
            case 5: return fn(args[0], args[1], args[2], args[3], args[4], 0, 0);
            case 6: return fn(args[0], args[1], args[2], args[3], args[4], args[5], 0);
            default: return fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
        }
    }
}

// ---------------------------------------------------------------------------
// Atomics and mutexes
// ---------------------------------------------------------------------------

IrisAtomic* iris_atomic_new(IrisVal* initial) {
    IrisAtomic* a = xmalloc(sizeof(IrisAtomic));
    pthread_mutex_init(&a->mu, NULL);
    a->val = initial;
    if (initial) iris_retain(initial);
    return a;
}
IrisVal* iris_atomic_load(IrisAtomic* a) {
    pthread_mutex_lock(&a->mu);
    IrisVal* v = a->val;
    pthread_mutex_unlock(&a->mu);
    return v;
}
void iris_atomic_store(IrisAtomic* a, IrisVal* val) {
    pthread_mutex_lock(&a->mu);
    if (val) iris_retain(val);
    if (a->val) iris_release(a->val);
    a->val = val;
    pthread_mutex_unlock(&a->mu);
}
IrisVal* iris_atomic_add(IrisAtomic* a, IrisVal* delta) {
    pthread_mutex_lock(&a->mu);
    IrisVal* result = xmalloc(sizeof(IrisVal));
    if (a->val && a->val->tag == IRIS_TAG_I64 && delta && delta->tag == IRIS_TAG_I64) {
        a->val->i64 += delta->i64;
        result->tag = IRIS_TAG_I64;  result->i64 = a->val->i64;
    } else if (a->val && (a->val->tag == IRIS_TAG_F64 || a->val->tag == IRIS_TAG_F32)) {
        double d = iris_unbox_f64(a->val) + iris_unbox_f64(delta);
        a->val->tag = IRIS_TAG_F64;  a->val->f64 = d;
        result->tag = IRIS_TAG_F64;  result->f64 = d;
    } else {
        result->tag = IRIS_TAG_I64;  result->i64 = 0;
    }
    pthread_mutex_unlock(&a->mu);
    return result;
}
IrisMutex* iris_mutex_new(IrisVal* initial) {
    IrisMutex* m = xmalloc(sizeof(IrisMutex));
    pthread_mutex_init(&m->mu, NULL);
    m->val = initial;
    return m;
}
IrisVal* iris_mutex_lock(IrisMutex* m) {
    pthread_mutex_lock(&m->mu);
    return m->val;
}
void iris_mutex_unlock(IrisMutex* m) {
    pthread_mutex_unlock(&m->mu);
}

// ---------------------------------------------------------------------------
// Grad (forward-mode autodiff — dual numbers)
// ---------------------------------------------------------------------------

IrisGrad* iris_make_grad(double value, double tangent) {
    IrisGrad* g = xmalloc(sizeof(IrisGrad));
    g->value = value;  g->tangent = tangent;
    return g;
}
double iris_grad_value(IrisGrad* g)   { return g ? g->value   : 0.0; }
double iris_grad_tangent(IrisGrad* g) { return g ? g->tangent : 0.0; }

// ---------------------------------------------------------------------------
// Sparse tensors (COO format over IrisList of IrisVal)
// ---------------------------------------------------------------------------

IrisSparse* iris_sparsify(IrisList* dense) {
    IrisSparse* sp = xcalloc(1, sizeof(IrisSparse));
    sp->cap     = 8;
    sp->indices = xmalloc(sizeof(size_t)    * sp->cap);
    sp->values  = xmalloc(sizeof(IrisVal*)  * sp->cap);
    for (size_t i = 0; i < dense->len; i++) {
        IrisVal* v = dense->data[i];
        int is_zero = 0;
        if (v) {
            if      (v->tag == IRIS_TAG_I64 && v->i64 == 0) is_zero = 1;
            else if (v->tag == IRIS_TAG_F64 && v->f64 == 0.0) is_zero = 1;
        } else is_zero = 1;
        if (!is_zero) {
            if (sp->len == sp->cap) {
                sp->cap *= 2;
                sp->indices = xrealloc(sp->indices, sizeof(size_t)   * sp->cap);
                sp->values  = xrealloc(sp->values,  sizeof(IrisVal*) * sp->cap);
            }
            sp->indices[sp->len] = i;
            sp->values [sp->len] = v;
            sp->len++;
        }
    }
    return sp;
}

IrisSparse* iris_sparsify_i64_array(int64_t* data, int64_t len) {
    IrisSparse* sp = xcalloc(1, sizeof(IrisSparse));
    sp->cap     = 8;
    sp->indices = xmalloc(sizeof(size_t)    * sp->cap);
    sp->values  = xmalloc(sizeof(IrisVal*)  * sp->cap);
    for (int64_t i = 0; i < len; i++) {
        int64_t v = data[i];
        if (v != 0) {
            if (sp->len == sp->cap) {
                sp->cap *= 2;
                sp->indices = xrealloc(sp->indices, sizeof(size_t)   * sp->cap);
                sp->values  = xrealloc(sp->values,  sizeof(IrisVal*) * sp->cap);
            }
            sp->indices[sp->len] = i;
            sp->values [sp->len] = iris_box_i64(v);
            sp->len++;
        }
    }
    return sp;
}

IrisSparse* iris_sparsify_f64_array(double* data, int64_t len) {
    IrisSparse* sp = xcalloc(1, sizeof(IrisSparse));
    sp->cap     = 8;
    sp->indices = xmalloc(sizeof(size_t)    * sp->cap);
    sp->values  = xmalloc(sizeof(IrisVal*)  * sp->cap);
    for (int64_t i = 0; i < len; i++) {
        double v = data[i];
        if (v != 0.0) {
            if (sp->len == sp->cap) {
                sp->cap *= 2;
                sp->indices = xrealloc(sp->indices, sizeof(size_t)   * sp->cap);
                sp->values  = xrealloc(sp->values,  sizeof(IrisVal*) * sp->cap);
            }
            sp->indices[sp->len] = i;
            sp->values [sp->len] = iris_box_f64(v);
            sp->len++;
        }
    }
    return sp;
}
IrisList* iris_densify(IrisSparse* sparse) {
    /* Determine dense size from max index in sparse data. */
    int64_t size = 0;
    for (size_t i = 0; i < sparse->len; i++) {
        if ((int64_t)sparse->indices[i] >= size) size = (int64_t)sparse->indices[i] + 1;
    }
    IrisList* l = iris_list_new();
    /* Fill with zeros */
    for (int64_t i = 0; i < size; i++) iris_list_push(l, iris_box_i64(0));
    for (size_t i = 0; i < sparse->len; i++) {
        size_t idx = sparse->indices[i];
        if ((int64_t)idx < size) l->data[idx] = sparse->values[i];
    }
    return l;
}

// ---------------------------------------------------------------------------
// Sparse tensor operations
// ---------------------------------------------------------------------------

IrisSparse* iris_tensor_sparsify(IrisTensor* t) {
    IrisSparse* sp = xcalloc(1, sizeof(IrisSparse));
    sp->cap     = 8;
    sp->indices = xmalloc(sizeof(size_t)    * sp->cap);
    sp->values  = xmalloc(sizeof(IrisVal*)  * sp->cap);
    for (int64_t i = 0; i < t->numel; i++) {
        if (t->data[i] != 0.0f) {
            if (sp->len == sp->cap) {
                sp->cap *= 2;
                sp->indices = xrealloc(sp->indices, sizeof(size_t)   * sp->cap);
                sp->values  = xrealloc(sp->values,  sizeof(IrisVal*) * sp->cap);
            }
            sp->indices[sp->len] = (size_t)i;
            sp->values [sp->len] = iris_box_f64((double)t->data[i]);
            sp->len++;
        }
    }
    return sp;
}

IrisTensor* iris_sparse_to_tensor(IrisSparse* sp, int64_t size) {
    IrisTensor* t = xcalloc(1, sizeof(IrisTensor));
    t->ndim  = 1;
    t->numel = size;
    t->shape = xmalloc(sizeof(int64_t));
    t->shape[0] = size;
    t->data  = xcalloc((size_t)size, sizeof(float));
    for (size_t i = 0; i < sp->len; i++) {
        size_t idx = sp->indices[i];
        if ((int64_t)idx < size && sp->values[i]) {
            if (sp->values[i]->tag == IRIS_TAG_F64)
                t->data[idx] = (float)sp->values[i]->f64;
            else if (sp->values[i]->tag == IRIS_TAG_I64)
                t->data[idx] = (float)sp->values[i]->i64;
        }
    }
    return t;
}

double iris_sparse_dot(IrisSparse* sp, IrisTensor* dense) {
    double sum = 0.0;
    for (size_t i = 0; i < sp->len; i++) {
        size_t idx = sp->indices[i];
        if ((int64_t)idx < dense->numel && sp->values[i]) {
            double sv = 0.0;
            if (sp->values[i]->tag == IRIS_TAG_F64) sv = sp->values[i]->f64;
            else if (sp->values[i]->tag == IRIS_TAG_I64) sv = (double)sp->values[i]->i64;
            sum += sv * (double)dense->data[idx];
        }
    }
    return sum;
}

int64_t iris_sparse_nnz(IrisSparse* sp) {
    return (int64_t)sp->len;
}

// ---------------------------------------------------------------------------
// Reverse-mode AD runtime
// ---------------------------------------------------------------------------
#if defined(_MSC_VER)
  #define IRIS_THREAD_LOCAL __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  #define IRIS_THREAD_LOCAL _Thread_local
#else
  #define IRIS_THREAD_LOCAL __thread
#endif

#define IRIS_TAPE_MAGIC ((uint64_t)0x4952495354415045ULL)

// The tape was a *fixed thread-local array* of IRIS_TAPE_ARENA_SIZE nodes, plus
// a topological buffer of the same length. At 88 bytes a node that is 11.5 MB +
// 1 MB reserved per thread, whether or not the thread ever called `grad`:
//
//   $ size -A iris_runtime.o
//   .tls$    12747312        <- 12.7 MB, against 86896 bytes of .text
//
// Measured at 77.4 MB for 4 threads and 808.1 MB for 64 -- ~12.2 MB per thread
// of committed working set, for threads that only slept and sent an integer.
// It also put a 12.7 MB TLS template in every binary IRIS produced. See
// known-issues #47.
//
// Now allocated lazily, in chunks. Chunked rather than a single growable block
// because nodes reference their parents by *pointer*: reallocating one block
// would invalidate every parent pointer already recorded, silently corrupting
// the graph rather than failing. Chunk addresses are stable for the life of the
// tape, so growth is safe.
//
// The ceiling is unchanged, so wrap-around behaviour is exactly as before.
#define IRIS_TAPE_CHUNK_NODES 4096
#define IRIS_TAPE_MAX_CHUNKS  32
#define IRIS_TAPE_ARENA_SIZE  (IRIS_TAPE_CHUNK_NODES * IRIS_TAPE_MAX_CHUNKS)

typedef struct IrisTapeNode {
    uint64_t               magic;
    double                 primal;
    double                 grad;
    const char*            op;
    struct IrisTapeNode*   parents[2];
    double                 parent_primals[2];
    int64_t                parent_count;
    uint64_t               grad_epoch;
    uint64_t               visit_epoch;
} IrisTapeNode;

typedef struct {
    IrisTapeNode** data;
    size_t         len;
    size_t         cap;
} IrisTapeVec;

// Thread-local state is now a chunk table rather than the chunks themselves:
// 32 pointers and a few counters, ~300 bytes instead of 12.6 MB.
// Two regions, because they have different lifetimes.
//
// `iris_backward` resets the arena index when it finishes, so the next
// `tape(...)` used to hand back an address the previous leaf was still holding
// -- `grad(x)` then read a later node's gradient and returned a plausible wrong
// number with no error (known-issues #48):
//
//   dx=6 dy=6        <- dx should be 4
//
// Leaves are the only nodes a program keeps a handle to across a backward pass,
// so leaves are now *pinned*: they live in their own region and are never
// recycled. Intermediates, which nothing outlives the pass to reference, still
// reset -- that is what keeps a training loop from growing without bound.
static IRIS_THREAD_LOCAL IrisTapeNode* iris_tape_chunks[IRIS_TAPE_MAX_CHUNKS];
static IRIS_THREAD_LOCAL size_t iris_tape_chunk_count = 0;
static IRIS_THREAD_LOCAL size_t iris_tape_arena_index = 0;

static IRIS_THREAD_LOCAL IrisTapeNode* iris_leaf_chunks[IRIS_TAPE_MAX_CHUNKS];
static IRIS_THREAD_LOCAL size_t iris_leaf_chunk_count = 0;
static IRIS_THREAD_LOCAL size_t iris_leaf_index = 0;
static IRIS_THREAD_LOCAL IrisTapeNode** iris_topo_buffer = NULL;
static IRIS_THREAD_LOCAL size_t iris_topo_buffer_cap = 0;

static IRIS_THREAD_LOCAL uint64_t iris_tape_grad_epoch  = 0;
static IRIS_THREAD_LOCAL uint64_t iris_tape_visit_epoch = 0;

// Releases this thread's tape. Called from a TLS destructor on thread exit and
// from iris_runtime_cleanup for the main thread.
static void iris_tape_thread_free(void) {
    for (size_t i = 0; i < iris_tape_chunk_count; i++) {
        free(iris_tape_chunks[i]);
        iris_tape_chunks[i] = NULL;
    }
    iris_tape_chunk_count = 0;
    iris_tape_arena_index = 0;
    for (size_t i = 0; i < iris_leaf_chunk_count; i++) {
        free(iris_leaf_chunks[i]);
        iris_leaf_chunks[i] = NULL;
    }
    iris_leaf_chunk_count = 0;
    iris_leaf_index = 0;
    free(iris_topo_buffer);
    iris_topo_buffer = NULL;
    iris_topo_buffer_cap = 0;
}

// Without this, a thread that used autodiff would leak its chunks on exit --
// which the old fixed array never had to worry about, because it was never
// allocated. `spawn` calls the user function directly (no wrapper to free
// from), and par_for workers are separate threads again, so a TLS destructor is
// the one hook that covers every thread the runtime creates.
#if !defined(_MSC_VER) && !defined(__wasm__)
static pthread_key_t  iris_tape_tls_key;
static pthread_once_t iris_tape_tls_once = PTHREAD_ONCE_INIT;

static void iris_tape_tls_dtor(void* unused) {
    (void)unused;
    // Runs on the exiting thread, so the thread-local pointers are still ours.
    iris_tape_thread_free();
}
static void iris_tape_tls_init(void) {
    pthread_key_create(&iris_tape_tls_key, iris_tape_tls_dtor);
}
static void iris_tape_tls_arm(void) {
    pthread_once(&iris_tape_tls_once, iris_tape_tls_init);
    // The value is a sentinel; the destructor reads the thread-locals directly.
    // It must be non-NULL or the destructor is not run.
    if (!pthread_getspecific(iris_tape_tls_key)) {
        pthread_setspecific(iris_tape_tls_key, (void*)1);
    }
}
#else
static void iris_tape_tls_arm(void) {}
#endif

// Hands out the next node from one of the two regions, allocating a chunk only
// when one is actually needed. Returns NULL if allocation fails; callers must
// tolerate that, and do, because iris_is_tape_node(NULL) is false and every
// consumer checks it.
static IrisTapeNode* iris_tape_alloc_from(IrisTapeNode** chunks,
                                          size_t* chunk_count,
                                          size_t* index,
                                          int recycle) {
    if (*index >= IRIS_TAPE_ARENA_SIZE) {
        // Intermediates wrap, as before. Leaves must not: recycling a leaf is
        // exactly the aliasing that #48 was, so a program that exceeds the leaf
        // budget gets NULL -- a gradient of zero -- rather than another leaf's
        // gradient reported as its own.
        if (!recycle) return NULL;
        *index = 0;
    }
    size_t idx   = *index;
    size_t chunk = idx / IRIS_TAPE_CHUNK_NODES;
    size_t off   = idx % IRIS_TAPE_CHUNK_NODES;

    if (chunk >= *chunk_count) {
        // Indices advance by one and wrap, so this is always the next chunk.
        IrisTapeNode* c = (IrisTapeNode*)calloc(IRIS_TAPE_CHUNK_NODES, sizeof(IrisTapeNode));
        if (!c) return NULL;
        iris_tape_tls_arm();
        chunks[chunk] = c;
        *chunk_count  = chunk + 1;
    }
    (*index)++;
    return &chunks[chunk][off];
}

// A leaf is a value the program holds a handle to and reads a gradient from
// after `backward` returns, so its address must stay stable. Everything else is
// internal to one backward pass and is recycled.
static IrisTapeNode* iris_tape_alloc_node(int is_leaf) {
    if (is_leaf) {
        return iris_tape_alloc_from(iris_leaf_chunks, &iris_leaf_chunk_count,
                                    &iris_leaf_index, 0);
    }
    return iris_tape_alloc_from(iris_tape_chunks, &iris_tape_chunk_count,
                                &iris_tape_arena_index, 1);
}

static int iris_is_tape_node(const void* ptr) {
    if (!ptr) return 0;
    return ((const IrisTapeNode*)ptr)->magic == IRIS_TAPE_MAGIC;
}

static uint64_t iris_tape_next_epoch(uint64_t* epoch) {
    (*epoch)++;
    if (*epoch == 0) (*epoch)++;
    return *epoch;
}

static void iris_tape_vec_push(IrisTapeVec* vec, IrisTapeNode* node) {
    if (vec->len < vec->cap) {
        vec->data[vec->len++] = node;
    }
}

static void iris_tape_collect_topo(IrisTapeNode* node, uint64_t visit_epoch, IrisTapeVec* topo) {
    if (!iris_is_tape_node(node) || node->visit_epoch == visit_epoch) return;
    node->visit_epoch = visit_epoch;
    for (int64_t i = 0; i < node->parent_count; i++) {
        iris_tape_collect_topo(node->parents[i], visit_epoch, topo);
    }
    iris_tape_vec_push(topo, node);
}

static double iris_tape_parent_primal(IrisTapeNode* node, int64_t idx) {
    IrisTapeNode* parent = node->parents[idx];
    return parent ? parent->primal : node->parent_primals[idx];
}

static void iris_tape_accumulate(IrisTapeNode* parent, double delta, uint64_t grad_epoch) {
    if (!iris_is_tape_node(parent)) return;
    if (parent->grad_epoch != grad_epoch) {
        parent->grad_epoch = grad_epoch;
        parent->grad = 0.0;
    }
    parent->grad += delta;
}

void* iris_tape_record(double value, const char* op, int64_t parent_count,
                       void* const* parents, const double* parent_primals) {
    // A leaf is recorded with op "leaf" by the lowerer (ensure_taped_leaf), and
    // is the only kind of node whose handle outlives the backward pass.
    int is_leaf = (op && strcmp(op, "leaf") == 0);
    IrisTapeNode* node = iris_tape_alloc_node(is_leaf);
    if (!node) return NULL;
    node->magic = IRIS_TAPE_MAGIC;
    node->primal = value;
    node->op = op ? op : "";
    node->parent_count = parent_count > 0 ? (parent_count > 2 ? 2 : parent_count) : 0;
    node->grad = 0.0;
    node->grad_epoch = 0;
    node->visit_epoch = 0;

    for (int64_t i = 0; i < node->parent_count; i++) {
        const void* parent = parents ? parents[i] : NULL;
        node->parents[i] = iris_is_tape_node(parent) ? (IrisTapeNode*)parent : NULL;
        if (parent_primals) {
            node->parent_primals[i] = parent_primals[i];
        } else if (node->parents[i]) {
            node->parent_primals[i] = node->parents[i]->primal;
        } else {
            node->parent_primals[i] = 0.0;
        }
    }

    return node;
}

void iris_backward(void* loss) {
    IrisTapeNode* loss_node = iris_is_tape_node(loss) ? (IrisTapeNode*)loss : NULL;
    if (!loss_node) return;

    // Size the topological buffer to the tape actually in use, not to the
    // ceiling. A thread that recorded 300 nodes needs one chunk's worth, not
    // 1 MB.
    size_t needed = (iris_tape_chunk_count + iris_leaf_chunk_count) * IRIS_TAPE_CHUNK_NODES;
    if (needed == 0) return;
    if (iris_topo_buffer_cap < needed) {
        IrisTapeNode** grown =
            (IrisTapeNode**)realloc(iris_topo_buffer, needed * sizeof(IrisTapeNode*));
        if (!grown) return;
        iris_tape_tls_arm();
        iris_topo_buffer     = grown;
        iris_topo_buffer_cap = needed;
    }

    IrisTapeVec topo;
    topo.data = iris_topo_buffer;
    topo.len = 0;
    topo.cap = iris_topo_buffer_cap;

    uint64_t visit_epoch = iris_tape_next_epoch(&iris_tape_visit_epoch);
    uint64_t grad_epoch = iris_tape_next_epoch(&iris_tape_grad_epoch);

    iris_tape_collect_topo(loss_node, visit_epoch, &topo);
    for (size_t i = 0; i < topo.len; i++) {
        topo.data[i]->grad = 0.0;
        topo.data[i]->grad_epoch = grad_epoch;
    }
    loss_node->grad = 1.0;

    for (size_t i = topo.len; i > 0; i--) {
        IrisTapeNode* node = topo.data[i - 1];
        double grad = node->grad_epoch == grad_epoch ? node->grad : 0.0;
        if (!node->op) continue;

        if (strcmp(node->op, "add") == 0) {
            for (int64_t p = 0; p < node->parent_count; p++) {
                iris_tape_accumulate(node->parents[p], grad, grad_epoch);
            }
        } else if (strcmp(node->op, "sub") == 0) {
            if (node->parent_count >= 1) iris_tape_accumulate(node->parents[0], grad, grad_epoch);
            if (node->parent_count >= 2) iris_tape_accumulate(node->parents[1], -grad, grad_epoch);
        } else if (strcmp(node->op, "mul") == 0) {
            if (node->parent_count >= 2) {
                double a = iris_tape_parent_primal(node, 0);
                double b = iris_tape_parent_primal(node, 1);
                iris_tape_accumulate(node->parents[0], grad * b, grad_epoch);
                iris_tape_accumulate(node->parents[1], grad * a, grad_epoch);
            }
        } else if (strcmp(node->op, "div") == 0) {
            if (node->parent_count >= 2) {
                double a = iris_tape_parent_primal(node, 0);
                double b = iris_tape_parent_primal(node, 1);
                iris_tape_accumulate(node->parents[0], grad / b, grad_epoch);
                iris_tape_accumulate(node->parents[1], -grad * a / (b * b), grad_epoch);
            }
        } else if (strcmp(node->op, "neg") == 0) {
            if (node->parent_count >= 1) iris_tape_accumulate(node->parents[0], -grad, grad_epoch);
        } else if (strcmp(node->op, "sin") == 0) {
            if (node->parent_count >= 1) {
                double x = iris_tape_parent_primal(node, 0);
                iris_tape_accumulate(node->parents[0], grad * cos(x), grad_epoch);
            }
        } else if (strcmp(node->op, "cos") == 0) {
            if (node->parent_count >= 1) {
                double x = iris_tape_parent_primal(node, 0);
                iris_tape_accumulate(node->parents[0], -grad * sin(x), grad_epoch);
            }
        } else if (strcmp(node->op, "exp") == 0) {
            if (node->parent_count >= 1) {
                double x = iris_tape_parent_primal(node, 0);
                iris_tape_accumulate(node->parents[0], grad * exp(x), grad_epoch);
            }
        } else if (strcmp(node->op, "log") == 0) {
            if (node->parent_count >= 1) {
                double x = iris_tape_parent_primal(node, 0);
                iris_tape_accumulate(node->parents[0], grad / x, grad_epoch);
            }
        } else if (strcmp(node->op, "sqrt") == 0) {
            if (node->parent_count >= 1) {
                double x = iris_tape_parent_primal(node, 0);
                iris_tape_accumulate(node->parents[0], grad / (2.0 * sqrt(x)), grad_epoch);
            }
        } else if (strcmp(node->op, "relu") == 0) {
            if (node->parent_count >= 1) {
                double x = iris_tape_parent_primal(node, 0);
                iris_tape_accumulate(node->parents[0], x > 0.0 ? grad : 0.0, grad_epoch);
            }
        } else if (strcmp(node->op, "sigmoid") == 0) {
            if (node->parent_count >= 1) {
                double x = iris_tape_parent_primal(node, 0);
                double s = 1.0 / (1.0 + exp(-x));
                iris_tape_accumulate(node->parents[0], grad * s * (1.0 - s), grad_epoch);
            }
        } else if (strcmp(node->op, "tanh") == 0) {
            if (node->parent_count >= 1) {
                double x = iris_tape_parent_primal(node, 0);
                double t = tanh(x);
                iris_tape_accumulate(node->parents[0], grad * (1.0 - t * t), grad_epoch);
            }
        } else if (strcmp(node->op, "pow") == 0) {
            if (node->parent_count >= 2) {
                double base = iris_tape_parent_primal(node, 0);
                double exponent = iris_tape_parent_primal(node, 1);
                iris_tape_accumulate(
                    node->parents[0],
                    grad * exponent * pow(base, exponent - 1.0),
                    grad_epoch
                );
                iris_tape_accumulate(
                    node->parents[1],
                    grad * pow(base, exponent) * log(base),
                    grad_epoch
                );
            }
        } else if (strcmp(node->op, "abs") == 0) {
            if (node->parent_count >= 1) {
                double x = iris_tape_parent_primal(node, 0);
                iris_tape_accumulate(node->parents[0], grad * (x >= 0.0 ? 1.0 : -1.0), grad_epoch);
            }
        }
    }

    // Recycle the intermediates for the next training step. The leaf region is
    // deliberately untouched: the program still holds handles to those nodes
    // and reads gradients from them after this returns. Recycling them was #48.
    iris_tape_arena_index = 0;
}

double iris_tape_grad(void* node) {
    IrisTapeNode* tape_node = iris_is_tape_node(node) ? (IrisTapeNode*)node : NULL;
    if (!tape_node || tape_node->grad_epoch != iris_tape_grad_epoch) return 0.0;
    return tape_node->grad;
}

// ---------------------------------------------------------------------------
// Non-scalar array fallback (for complex / mixed-type arrays)
// ---------------------------------------------------------------------------

IrisList*  iris_alloc_array(void)                      { return iris_list_new(); }
IrisVal*   iris_array_load(IrisList* arr, int64_t idx) { return iris_list_get(arr, idx); }
void       iris_array_store(IrisList* arr, int64_t idx, IrisVal* val) { iris_list_set(arr, idx, val); }

// ---------------------------------------------------------------------------
// Tensor ops — real compute (replacing shape-tracking stubs)
// ---------------------------------------------------------------------------

// Legacy stubs (kept for backward compat)
void* iris_tensor_op(void)                { return NULL; }
void* iris_tensor_load(void* t, ...)      { (void)t; return NULL; }
void  iris_tensor_store(void* t, ...)     { (void)t; }

// --- Allocation / lifecycle ------------------------------------------------

typedef struct IrisMemBlock {
    void* ptr;
    size_t size;
    int in_use;
    struct IrisMemBlock* next;
} IrisMemBlock;

static IrisMemBlock* tensor_pool_head = NULL;
static int tensor_pool_enabled = 0;
static size_t tensor_pool_max_bytes = 0;
static pthread_mutex_t tensor_pool_mu = PTHREAD_MUTEX_INITIALIZER;

int64_t iris_tensor_pool_init(int64_t limit_bytes) {
    pthread_mutex_lock(&tensor_pool_mu);
    if (!tensor_pool_enabled) {
        tensor_pool_enabled = 1;
        tensor_pool_max_bytes = (size_t)limit_bytes;
        tensor_pool_head = NULL;
    }
    pthread_mutex_unlock(&tensor_pool_mu);
    return 0;
}

int64_t iris_tensor_pool_destroy(void) {
    pthread_mutex_lock(&tensor_pool_mu);
    if (tensor_pool_enabled) {
        IrisMemBlock* curr = tensor_pool_head;
        while (curr) {
            IrisMemBlock* next = curr->next;
            free(curr->ptr);
            free(curr);
            curr = next;
        }
        tensor_pool_head = NULL;
        tensor_pool_enabled = 0;
        tensor_pool_max_bytes = 0;
    }
    pthread_mutex_unlock(&tensor_pool_mu);
    return 0;
}

IrisTensor* iris_tensor_alloc(int32_t ndim, const int64_t* shape) {
    IrisTensor* t = xmalloc(sizeof(IrisTensor));
    t->ndim = ndim;
    t->shape = xmalloc(ndim * sizeof(int64_t));
    t->numel = 1;
    for (int32_t i = 0; i < ndim; i++) {
        if (shape[i] < 0) {
            fprintf(stderr, "IRIS ML: Negative shape dimension\n");
            abort();
        }
        if (shape[i] > 0 && t->numel > (INT64_MAX / shape[i])) {
            fprintf(stderr, "IRIS ML: Tensor shape overflow\n");
            abort();
        }
        t->shape[i] = shape[i];
        t->numel *= shape[i];
    }
    
    if ((uint64_t)t->numel > (uint64_t)(SIZE_MAX / sizeof(float))) {
        fprintf(stderr, "IRIS ML: Tensor too large\n");
        abort();
    }
    size_t req_bytes = (size_t)t->numel * sizeof(float);
    
    pthread_mutex_lock(&tensor_pool_mu);
    if (tensor_pool_enabled) {
        IrisMemBlock* best_block = NULL;
        IrisMemBlock* curr = tensor_pool_head;
        while (curr) {
            if (!curr->in_use && curr->size >= req_bytes) {
                if (!best_block || curr->size < best_block->size) {
                    best_block = curr;
                }
            }
            curr = curr->next;
        }
        
        if (best_block) {
            best_block->in_use = 1;
            t->data = best_block->ptr;
            pthread_mutex_unlock(&tensor_pool_mu);
            return t;
        }
        
        if (tensor_pool_max_bytes > 0) {
            size_t total_cached = 0;
            IrisMemBlock* p = tensor_pool_head;
            while (p) {
                total_cached += p->size;
                p = p->next;
            }
            
            while (total_cached + req_bytes > tensor_pool_max_bytes) {
                IrisMemBlock* prev = NULL;
                IrisMemBlock* curr_evict = tensor_pool_head;
                IrisMemBlock* to_evict = NULL;
                IrisMemBlock* to_evict_prev = NULL;
                
                while (curr_evict) {
                    if (!curr_evict->in_use) {
                        to_evict = curr_evict;
                        to_evict_prev = prev;
                        break;
                    }
                    prev = curr_evict;
                    curr_evict = curr_evict->next;
                }
                
                if (to_evict) {
                    if (to_evict_prev) {
                        to_evict_prev->next = to_evict->next;
                    } else {
                        tensor_pool_head = to_evict->next;
                    }
                    total_cached -= to_evict->size;
                    free(to_evict->ptr);
                    free(to_evict);
                } else {
                    break;
                }
            }
        }
        
        void* ptr = malloc(req_bytes);
        if (!ptr) {
            fprintf(stderr, "iris: tensor pool out of memory\n");
            abort();
        }
        
        IrisMemBlock* block = xmalloc(sizeof(IrisMemBlock));
        block->ptr = ptr;
        block->size = req_bytes;
        block->in_use = 1;
        block->next = tensor_pool_head;
        tensor_pool_head = block;
        
        t->data = ptr;
        pthread_mutex_unlock(&tensor_pool_mu);
        return t;
    } else {
        pthread_mutex_unlock(&tensor_pool_mu);
        t->data = xmalloc(req_bytes);
        return t;
    }
}

void iris_tensor_free(IrisTensor* t) {
    if (!t) return;
    
    pthread_mutex_lock(&tensor_pool_mu);
    if (tensor_pool_enabled) {
        IrisMemBlock* curr = tensor_pool_head;
        while (curr) {
            if (curr->ptr == t->data) {
                curr->in_use = 0;
                break;
            }
            curr = curr->next;
        }
        
        if (curr) {
            pthread_mutex_unlock(&tensor_pool_mu);
        } else {
            pthread_mutex_unlock(&tensor_pool_mu);
            free(t->data);
        }
    } else {
        pthread_mutex_unlock(&tensor_pool_mu);
        free(t->data);
    }
    
    free(t->shape);
    free(t);
}

IrisTensor* iris_tensor_zeros(int32_t ndim, const int64_t* shape) {
    IrisTensor* t = iris_tensor_alloc(ndim, shape);
    memset(t->data, 0, t->numel * sizeof(float));
    return t;
}

IrisTensor* iris_tensor_fill(int32_t ndim, const int64_t* shape, float val) {
    IrisTensor* t = iris_tensor_alloc(ndim, shape);
    for (int64_t i = 0; i < t->numel; i++) t->data[i] = val;
    return t;
}

// --- Element access --------------------------------------------------------

float iris_tensor_get(IrisTensor* t, int64_t flat_idx) {
    if (!t || flat_idx < 0 || flat_idx >= t->numel) return 0.0f;
    return t->data[flat_idx];
}

void iris_tensor_set(IrisTensor* t, int64_t flat_idx, float val) {
    if (!t || flat_idx < 0 || flat_idx >= t->numel) return;
    t->data[flat_idx] = val;
}

// --- Matrix multiplication -------------------------------------------------
// Supports 2D matmul: (M,K) @ (K,N) -> (M,N)

IrisTensor* iris_tensor_matmul(IrisTensor* a, IrisTensor* b) {
    if (!a || !b || a->ndim < 2 || b->ndim < 2) return NULL;
    int64_t m = a->shape[a->ndim - 2];
    int64_t k = a->shape[a->ndim - 1];
    int64_t n = b->shape[b->ndim - 1];
    if (b->shape[b->ndim - 2] != k) return NULL;

    int64_t out_shape[2] = { m, n };
    IrisTensor* out = iris_tensor_zeros(2, out_shape);

    for (int64_t i = 0; i < m; i++) {
        for (int64_t l = 0; l < k; l++) {
            float a_il = a->data[i * k + l];
            for (int64_t j = 0; j < n; j++) {
                out->data[i * n + j] += a_il * b->data[l * n + j];
            }
        }
    }
    return out;
}

// --- Element-wise binary ops -----------------------------------------------

static IrisTensor* tensor_binop(IrisTensor* a, IrisTensor* b, int op) {
    if (!a || !b || a->numel != b->numel) return NULL;
    IrisTensor* out = iris_tensor_alloc(a->ndim, a->shape);
    for (int64_t i = 0; i < a->numel; i++) {
        float x = a->data[i], y = b->data[i];
        switch (op) {
            case 0: out->data[i] = x + y; break;
            case 1: out->data[i] = x - y; break;
            case 2: out->data[i] = x * y; break;
            case 3: out->data[i] = (y != 0.0f) ? x / y : 0.0f; break;
            default: out->data[i] = 0.0f; break;
        }
    }
    return out;
}

IrisTensor* iris_tensor_add(IrisTensor* a, IrisTensor* b) { return tensor_binop(a, b, 0); }
IrisTensor* iris_tensor_sub(IrisTensor* a, IrisTensor* b) { return tensor_binop(a, b, 1); }
IrisTensor* iris_tensor_mul(IrisTensor* a, IrisTensor* b) { return tensor_binop(a, b, 2); }
IrisTensor* iris_tensor_div(IrisTensor* a, IrisTensor* b) { return tensor_binop(a, b, 3); }

// --- Element-wise unary ops ------------------------------------------------

static IrisTensor* tensor_unary(IrisTensor* t, int op) {
    if (!t) return NULL;
    IrisTensor* out = iris_tensor_alloc(t->ndim, t->shape);
    for (int64_t i = 0; i < t->numel; i++) {
        float x = t->data[i];
        switch (op) {
            case 0: out->data[i] = -x; break;                             // neg
            case 1: out->data[i] = x > 0.0f ? x : 0.0f; break;          // relu
            case 2: out->data[i] = 1.0f / (1.0f + expf(-x)); break;     // sigmoid
            case 3: out->data[i] = tanhf(x); break;                       // tanh
            case 4: out->data[i] = expf(x); break;                        // exp
            case 5: out->data[i] = (x > 0.0f) ? logf(x) : -INFINITY; break; // log
            case 6: out->data[i] = (x >= 0.0f) ? sqrtf(x) : 0.0f; break;// sqrt
            case 7: out->data[i] = fabsf(x); break;                       // abs
            default: out->data[i] = x; break;
        }
    }
    return out;
}

IrisTensor* iris_tensor_neg(IrisTensor* t)      { return tensor_unary(t, 0); }
IrisTensor* iris_tensor_relu(IrisTensor* t)     { return tensor_unary(t, 1); }
IrisTensor* iris_tensor_sigmoid(IrisTensor* t)  { return tensor_unary(t, 2); }
IrisTensor* iris_tensor_tanh_act(IrisTensor* t) { return tensor_unary(t, 3); }
IrisTensor* iris_tensor_exp(IrisTensor* t)      { return tensor_unary(t, 4); }
IrisTensor* iris_tensor_log(IrisTensor* t)      { return tensor_unary(t, 5); }
IrisTensor* iris_tensor_sqrt(IrisTensor* t)     { return tensor_unary(t, 6); }
IrisTensor* iris_tensor_abs(IrisTensor* t)      { return tensor_unary(t, 7); }

// --- Reshape ---------------------------------------------------------------

IrisTensor* iris_tensor_reshape(IrisTensor* t, int32_t new_ndim, const int64_t* new_shape) {
    if (!t) return NULL;
    int64_t new_numel = 1;
    for (int32_t i = 0; i < new_ndim; i++) new_numel *= new_shape[i];
    if (new_numel != t->numel) return NULL;

    IrisTensor* out = xmalloc(sizeof(IrisTensor));
    out->ndim = new_ndim;
    out->numel = new_numel;
    out->shape = xmalloc(new_ndim * sizeof(int64_t));
    memcpy(out->shape, new_shape, new_ndim * sizeof(int64_t));
    // Share a copy of the data (reshape is zero-copy semantically, but we copy for safety)
    out->data = xmalloc(out->numel * sizeof(float));
    memcpy(out->data, t->data, out->numel * sizeof(float));
    return out;
}

// --- Transpose -------------------------------------------------------------
// General N-dim transpose with explicit axis permutation.

IrisTensor* iris_tensor_transpose(IrisTensor* t, const int32_t* axes) {
    if (!t) return NULL;
    int32_t ndim = t->ndim;

    // Compute new shape
    int64_t* new_shape = xmalloc(ndim * sizeof(int64_t));
    for (int32_t i = 0; i < ndim; i++) new_shape[i] = t->shape[axes[i]];

    IrisTensor* out = iris_tensor_alloc(ndim, new_shape);

    // Compute strides for source tensor
    int64_t* src_strides = xmalloc(ndim * sizeof(int64_t));
    src_strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; i--)
        src_strides[i] = src_strides[i + 1] * t->shape[i + 1];

    // Compute strides for destination tensor
    int64_t* dst_strides = xmalloc(ndim * sizeof(int64_t));
    dst_strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; i--)
        dst_strides[i] = dst_strides[i + 1] * new_shape[i + 1];

    // Iterate over all elements using N-digit counter
    int64_t* coords = xcalloc(ndim, sizeof(int64_t));
    for (int64_t flat = 0; flat < t->numel; flat++) {
        // Compute source multi-index
        int64_t rem = flat;
        for (int32_t d = 0; d < ndim; d++) {
            coords[d] = rem / src_strides[d];
            rem %= src_strides[d];
        }
        // Compute destination flat index by permuting coords
        int64_t dst_flat = 0;
        for (int32_t d = 0; d < ndim; d++) {
            dst_flat += coords[axes[d]] * dst_strides[d];
        }
        out->data[dst_flat] = t->data[flat];
    }

    free(coords);
    free(src_strides);
    free(dst_strides);
    free(new_shape);
    return out;
}

// --- Reductions ------------------------------------------------------------
// Reduce along a single axis with sum/max/mean.

static IrisTensor* tensor_reduce(IrisTensor* t, int32_t axis, int keepdims, int op) {
    if (!t || axis < 0 || axis >= t->ndim) return NULL;

    int32_t ndim = t->ndim;
    int64_t axis_len = t->shape[axis];
    if (axis_len == 0) return NULL;

    // Compute output shape
    int32_t out_ndim = keepdims ? ndim : ndim - 1;
    if (out_ndim == 0) out_ndim = 1; // scalar result as 1D [1]
    int64_t* out_shape = xmalloc(out_ndim * sizeof(int64_t));
    int32_t j = 0;
    for (int32_t i = 0; i < ndim; i++) {
        if (i == axis) {
            if (keepdims) out_shape[j++] = 1;
        } else {
            out_shape[j++] = t->shape[i];
        }
    }
    if (j == 0) { out_shape[0] = 1; j = 1; }

    IrisTensor* out = iris_tensor_zeros(out_ndim, out_shape);

    // Compute strides
    int64_t* strides = xmalloc(ndim * sizeof(int64_t));
    strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; i--)
        strides[i] = strides[i + 1] * t->shape[i + 1];

    int64_t outer_size = 1, inner_size = 1;
    for (int32_t i = 0; i < axis; i++) outer_size *= t->shape[i];
    for (int32_t i = axis + 1; i < ndim; i++) inner_size *= t->shape[i];

    // Initialize max to -inf if needed
    if (op == 1) { // max
        for (int64_t i = 0; i < out->numel; i++) out->data[i] = -INFINITY;
    }

    for (int64_t o = 0; o < outer_size; o++) {
        for (int64_t a = 0; a < axis_len; a++) {
            for (int64_t inn = 0; inn < inner_size; inn++) {
                int64_t src_idx = o * (axis_len * inner_size) + a * inner_size + inn;
                int64_t dst_idx = o * inner_size + inn;
                float v = t->data[src_idx];
                switch (op) {
                    case 0: out->data[dst_idx] += v; break;          // sum
                    case 1: if (v > out->data[dst_idx]) out->data[dst_idx] = v; break; // max
                    case 2: out->data[dst_idx] += v; break;          // mean (sum first, divide later)
                    default: break;
                }
            }
        }
    }

    // For mean, divide by axis length
    if (op == 2) {
        for (int64_t i = 0; i < out->numel; i++)
            out->data[i] /= (float)axis_len;
    }

    free(strides);
    free(out_shape);
    return out;
}

IrisTensor* iris_tensor_reduce_sum(IrisTensor* t, int32_t axis, int keepdims)  { return tensor_reduce(t, axis, keepdims, 0); }
IrisTensor* iris_tensor_reduce_max(IrisTensor* t, int32_t axis, int keepdims)  { return tensor_reduce(t, axis, keepdims, 1); }
IrisTensor* iris_tensor_reduce_mean(IrisTensor* t, int32_t axis, int keepdims) { return tensor_reduce(t, axis, keepdims, 2); }

// ---------------------------------------------------------------------------
// Time / OS (Phase 97)
// ---------------------------------------------------------------------------

#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/time.h>
#  include <unistd.h>
#endif

int64_t iris_now_ms(void) {
#ifdef _WIN32
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    /* FILETIME is 100-ns intervals since 1601-01-01.
       Subtract epoch offset (1601→1970) then convert to ms. */
    uint64_t t = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    t -= 116444736000000000ULL;  /* 1601→1970 in 100-ns ticks */
    return (int64_t)(t / 10000ULL);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + (int64_t)tv.tv_usec / 1000;
#endif
}

void iris_sleep_ms(int64_t ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    usleep((useconds_t)(ms * 1000));
#endif
}

// ---------------------------------------------------------------------------
// Struct / Tuple / Closure fallback helpers (opaque path)
// ---------------------------------------------------------------------------

/* iris_make_struct(ptr f0, ptr f1, …)
   — stores N boxed field values in a list-backed struct. */
IrisVal* iris_make_struct(int nfields, ...) {
    IrisList* l = iris_list_new();
    va_list ap;
    va_start(ap, nfields);
    for (int i = 0; i < nfields; i++) {
        IrisVal* v = va_arg(ap, IrisVal*);
        iris_list_push(l, v);
    }
    va_end(ap);
    IrisVal* r = (IrisVal*)xmalloc(sizeof(IrisVal));
    r->tag = IRIS_TAG_STRUCT;
    r->ptr = l;
    return r;
}

IrisVal* iris_get_field(IrisVal* s, int32_t idx) {
    if (!s) return iris_box_i64(0);
    if (s->tag == IRIS_TAG_STRUCT) {
        IrisList* l = (IrisList*)s->ptr;
        return iris_list_get(l, (int64_t)idx);
    }
    return iris_box_i64(0);
}

IrisVal* iris_make_tuple(int nelems, ...) {
    IrisList* l = iris_list_new();
    va_list ap;
    va_start(ap, nelems);
    for (int i = 0; i < nelems; i++) {
        IrisVal* v = va_arg(ap, IrisVal*);
        iris_list_push(l, v);
    }
    va_end(ap);
    IrisVal* r = (IrisVal*)xmalloc(sizeof(IrisVal));
    r->tag = IRIS_TAG_TUPLE;
    r->ptr = l;
    return r;
}

IrisVal* iris_get_element(IrisVal* t, int32_t idx) {
    if (!t) return iris_box_i64(0);
    if (t->tag == IRIS_TAG_TUPLE) {
        IrisList* l = (IrisList*)t->ptr;
        return iris_list_get(l, (int64_t)idx);
    }
    return iris_box_i64(0);
}

// ---------------------------------------------------------------------------
// IRIS language-facing ML runtime wrappers
// ---------------------------------------------------------------------------

typedef int (*IrisMlRunFn)(void*, IrisTensor**, size_t, IrisTensor***, size_t*);

static IrisVal* iris_mlrt_empty_tensor_pair(void) {
    IrisList* data = iris_list_new();
    IrisList* shape = iris_list_new();
    return iris_make_tuple(2, iris_box_list(data), iris_box_list(shape));
}

static IrisTensor* iris_mlrt_tensor_from_pair(IrisVal* pair) {
    if (!pair || pair->tag != IRIS_TAG_TUPLE) return NULL;

    IrisVal* data_val = iris_get_element(pair, 0);
    IrisVal* shape_val = iris_get_element(pair, 1);
    IrisList* data = iris_unbox_list(data_val);
    IrisList* shape_list = iris_unbox_list(shape_val);
    int64_t ndim64 = iris_list_len(shape_list);
    if (ndim64 <= 0 || ndim64 > INT32_MAX) return NULL;

    int32_t ndim = (int32_t)ndim64;
    int64_t* shape = (int64_t*)xmalloc(sizeof(int64_t) * (size_t)ndim);
    for (int32_t i = 0; i < ndim; i++) {
        shape[i] = iris_unbox_i64(iris_list_get(shape_list, i));
    }

    IrisTensor* tensor = iris_tensor_alloc(ndim, shape);
    free(shape);
    if (!tensor) return NULL;

    if (iris_list_len(data) != tensor->numel) {
        iris_tensor_free(tensor);
        return NULL;
    }

    for (int64_t i = 0; i < tensor->numel; i++) {
        tensor->data[i] = (float)iris_unbox_f64(iris_list_get(data, i));
    }
    return tensor;
}

static IrisVal* iris_mlrt_pair_from_tensor(IrisTensor* tensor) {
    if (!tensor) return iris_mlrt_empty_tensor_pair();

    IrisList* data = iris_list_new();
    IrisList* shape = iris_list_new();
    for (int64_t i = 0; i < tensor->numel; i++) {
        iris_list_push(data, iris_box_f64((double)tensor->data[i]));
    }
    for (int32_t i = 0; i < tensor->ndim; i++) {
        iris_list_push(shape, iris_box_i64(tensor->shape[i]));
    }
    return iris_make_tuple(2, iris_box_list(data), iris_box_list(shape));
}

static IrisList* iris_mlrt_empty_tensor_pair_unboxed(void) {
    IrisList* data = iris_list_new();
    IrisList* shape = iris_list_new();
    IrisList* t_list = iris_list_new();
    iris_list_push(t_list, iris_box_list(data));
    iris_list_push(t_list, iris_box_list(shape));
    return t_list;
}

static IrisVal* iris_mlrt_run_single(int64_t handle, IrisVal* input, IrisMlRunFn run_fn) {
    if (handle == 0 || !run_fn) return iris_mlrt_empty_tensor_pair();

    IrisTensor* native_input = iris_mlrt_tensor_from_pair(input);
    if (!native_input) return iris_mlrt_empty_tensor_pair();

    IrisTensor* inputs[1];
    inputs[0] = native_input;
    IrisTensor** outputs = NULL;
    size_t n_outputs = 0;
    int rc = run_fn((void*)(intptr_t)handle, inputs, 1, &outputs, &n_outputs);
    iris_tensor_free(native_input);

    if (rc != 0 || !outputs || n_outputs == 0) {
        if (outputs) free(outputs);
        return iris_mlrt_empty_tensor_pair();
    }

    IrisVal* result = iris_mlrt_pair_from_tensor(outputs[0]);

    for (size_t i = 0; i < n_outputs; i++) {
        if (outputs[i]) iris_tensor_free(outputs[i]);
    }
    free(outputs);
    return result;
}

static IrisList* iris_mlrt_run_multi(int64_t handle, IrisList* inputs_list, IrisMlRunFn run_fn) {
    if (handle == 0 || !run_fn || !inputs_list) {
        return iris_list_new();
    }
    int64_t n_inputs = iris_list_len(inputs_list);
    if (n_inputs <= 0) {
        return iris_list_new();
    }

    IrisTensor** inputs_array = (IrisTensor**)xmalloc(sizeof(IrisTensor*) * (size_t)n_inputs);
    for (int64_t i = 0; i < n_inputs; i++) {
        inputs_array[i] = iris_mlrt_tensor_from_pair(iris_list_get(inputs_list, i));
    }

    IrisTensor** outputs_array = NULL;
    size_t n_outputs = 0;
    int rc = run_fn((void*)(intptr_t)handle, inputs_array, (size_t)n_inputs, &outputs_array, &n_outputs);

    for (int64_t i = 0; i < n_inputs; i++) {
        if (inputs_array[i]) iris_tensor_free(inputs_array[i]);
    }
    free(inputs_array);

    IrisList* out_list = iris_list_new();
    if (rc == 0 && outputs_array && n_outputs > 0) {
        for (size_t i = 0; i < n_outputs; i++) {
            if (outputs_array[i]) {
                iris_list_push(out_list, iris_mlrt_pair_from_tensor(outputs_array[i]));
                iris_tensor_free(outputs_array[i]);
            }
        }
    }
    if (outputs_array) free(outputs_array);

    return out_list;
}

#ifdef _WIN32
#define IRIS_EXPORT __declspec(dllexport)
#else
#define IRIS_EXPORT __attribute__((visibility("default")))
#endif

IRIS_EXPORT int64_t iris_mlrt_onnx_load(const char* model_path) {
    return (int64_t)(intptr_t)iris_onnx_session_create(model_path);
}

IRIS_EXPORT int64_t iris_mlrt_onnx_free(int64_t session) {
    iris_onnx_session_free((void*)(intptr_t)session);
    return 0;
}

IRIS_EXPORT IrisVal* iris_mlrt_onnx_run(int64_t session, IrisVal* input) {
    return iris_mlrt_run_single(session, input, iris_onnx_session_run);
}

IRIS_EXPORT IrisList* iris_mlrt_onnx_run_multi(int64_t session, IrisList* inputs_list) {
    return iris_mlrt_run_multi(session, inputs_list, iris_onnx_session_run);
}

/* ---------------------------------------------------------------------------
 * LibTorch plugin (loaded at runtime).
 *
 * LibTorch exposes a C++ API, so unlike ONNX/TensorFlow it cannot be dlopen'd
 * directly — the mangled symbols and C++ ABI make that impractical. Instead
 * pytorch_shim.cpp (which already declares its entry points `extern "C"`) is
 * built into a *separate* shared library that links LibTorch, and the core
 * runtime resolves those four C symbols from it on first use.
 *
 * The upshot: iris_runtime.o never references LibTorch, so it stays identical
 * whether or not the SDK is installed — which is what lets us ship one
 * prebuilt runtime object per target.
 * ------------------------------------------------------------------------- */

typedef void*  (*fn_iris_pytorch_load)(const char*);
typedef int    (*fn_iris_pytorch_run)(void*, IrisTensor**, size_t, IrisTensor***, size_t*);
typedef void   (*fn_iris_pytorch_free)(void*);
typedef double (*fn_iris_pytorch_train_step)(void*, IrisTensor**, size_t, IrisTensor**, size_t, double);

static fn_iris_pytorch_load       p_iris_pytorch_load       = NULL;
static fn_iris_pytorch_run        p_iris_pytorch_run        = NULL;
static fn_iris_pytorch_free       p_iris_pytorch_free       = NULL;
static fn_iris_pytorch_train_step p_iris_pytorch_train_step = NULL;

/* Returns 1 when the LibTorch plugin is loaded and usable, 0 otherwise.
 * Caches both outcomes so the warning is printed at most once. */
static int iris_torch_available(void) {
    static int state = -1; /* -1 unknown, 0 unavailable, 1 ready */
    if (state >= 0) return state;

    static const char* const candidates[] = {
#ifdef _WIN32
        "iris_torch_plugin.dll",
#elif defined(__APPLE__)
        "libiris_torch_plugin.dylib",
#else
        "libiris_torch_plugin.so",
#endif
    };

    void* lib = NULL;
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
#ifdef _WIN32
        lib = (void*)LoadLibraryA(candidates[i]);
#else
        lib = dlopen(candidates[i], RTLD_LAZY);
#endif
        if (lib) break;
    }

    if (!lib) {
        fprintf(stderr,
                "iris: LibTorch plugin (iris_torch_plugin) not found. It must be "
                "built once against your LibTorch install and placed next to the "
                "binary or on the library search path. "
                "See docs/ml-backends.md.\n");
        state = 0;
        return 0;
    }

#ifdef _WIN32
#define IRIS_TORCH_SYM(n) (void*)GetProcAddress((HMODULE)lib, n)
#else
#define IRIS_TORCH_SYM(n) dlsym(lib, n)
#endif
    p_iris_pytorch_load       = (fn_iris_pytorch_load)      IRIS_TORCH_SYM("iris_pytorch_load");
    p_iris_pytorch_run        = (fn_iris_pytorch_run)       IRIS_TORCH_SYM("iris_pytorch_run");
    p_iris_pytorch_free       = (fn_iris_pytorch_free)      IRIS_TORCH_SYM("iris_pytorch_free");
    p_iris_pytorch_train_step = (fn_iris_pytorch_train_step)IRIS_TORCH_SYM("iris_pytorch_train_step");
#undef IRIS_TORCH_SYM

    if (!p_iris_pytorch_load || !p_iris_pytorch_run ||
        !p_iris_pytorch_free || !p_iris_pytorch_train_step) {
        fprintf(stderr, "iris: LibTorch plugin is missing required symbols — "
                        "rebuild it against this version of IRIS.\n");
        state = 0;
        return 0;
    }

    state = 1;
    return 1;
}

IRIS_EXPORT int64_t iris_mlrt_pytorch_load(const char* model_path) {
    if (!iris_torch_available()) return 0;
    return (int64_t)(intptr_t)p_iris_pytorch_load(model_path);
}

IRIS_EXPORT int64_t iris_mlrt_pytorch_free(int64_t model) {
    if (!iris_torch_available()) return 0;
    p_iris_pytorch_free((void*)(intptr_t)model);
    return 0;
}

IRIS_EXPORT IrisVal* iris_mlrt_pytorch_run(int64_t model, IrisVal* input) {
    if (!iris_torch_available()) return iris_mlrt_empty_tensor_pair();
    return iris_mlrt_run_single(model, input, p_iris_pytorch_run);
}

IRIS_EXPORT IrisList* iris_mlrt_pytorch_run_multi(int64_t model, IrisList* inputs_list) {
    if (!iris_torch_available()) return iris_list_new();
    return iris_mlrt_run_multi(model, inputs_list, p_iris_pytorch_run);
}

IRIS_EXPORT double iris_mlrt_pytorch_train_step(int64_t model, IrisList* inputs_list, IrisList* targets_list, double lr) {
    if (!iris_torch_available()) return 0.0;
    if (model == 0 || !inputs_list || !targets_list) {
        return 0.0;
    }
    int64_t n_inputs = iris_list_len(inputs_list);
    int64_t n_targets = iris_list_len(targets_list);
    if (n_inputs <= 0 || n_targets <= 0) return 0.0;

    IrisTensor** inputs_array = (IrisTensor**)xmalloc(sizeof(IrisTensor*) * (size_t)n_inputs);
    for (int64_t i = 0; i < n_inputs; i++) {
        inputs_array[i] = iris_mlrt_tensor_from_pair(iris_list_get(inputs_list, i));
    }

    IrisTensor** targets_array = (IrisTensor**)xmalloc(sizeof(IrisTensor*) * (size_t)n_targets);
    for (int64_t i = 0; i < n_targets; i++) {
        targets_array[i] = iris_mlrt_tensor_from_pair(iris_list_get(targets_list, i));
    }

    double loss = p_iris_pytorch_train_step((void*)(intptr_t)model, inputs_array, (size_t)n_inputs, targets_array, (size_t)n_targets, lr);

    for (int64_t i = 0; i < n_inputs; i++) {
        if (inputs_array[i]) iris_tensor_free(inputs_array[i]);
    }
    free(inputs_array);
    for (int64_t i = 0; i < n_targets; i++) {
        if (targets_array[i]) iris_tensor_free(targets_array[i]);
    }
    free(targets_array);

    return loss;
}

IRIS_EXPORT int64_t iris_mlrt_tf_load(const char* model_path) {
    return (int64_t)(intptr_t)iris_tf_load_saved_model(model_path);
}

IRIS_EXPORT int64_t iris_mlrt_tf_free(int64_t model) {
    iris_tf_free((void*)(intptr_t)model);
    return 0;
}

IRIS_EXPORT IrisVal* iris_mlrt_tf_run(int64_t model, IrisVal* input) {
    return iris_mlrt_run_single(model, input, iris_tf_run);
}

IRIS_EXPORT IrisList* iris_mlrt_tf_run_multi(int64_t model, IrisList* inputs_list) {
    return iris_mlrt_run_multi(model, inputs_list, iris_tf_run);
}

/* Closure: stores a function pointer and captured environment. */
typedef struct {
    void*     fn;        /* function pointer */
    IrisList* captures;  /* captured values */
} IrisClosure;

IrisVal* iris_make_closure(void* fn, int ncaptures, ...) {
    IrisClosure* c = (IrisClosure*)xmalloc(sizeof(IrisClosure));
    c->fn = fn;
    c->captures = iris_list_new();
    va_list ap;
    va_start(ap, ncaptures);
    for (int i = 0; i < ncaptures; i++) {
        IrisVal* v = va_arg(ap, IrisVal*);
        iris_list_push(c->captures, v);
    }
    va_end(ap);
    IrisVal* r = (IrisVal*)xmalloc(sizeof(IrisVal));
    r->tag = IRIS_TAG_CLOSURE;
    r->ptr = c;
    return r;
}

IrisVal* iris_call_closure(IrisVal* closure, ...) {
    /* Stub: closure invocation now handled inline by LLVM codegen.
       Kept for link compatibility — should not be reached at runtime. */
    (void)closure;
    return iris_box_i64(0);
}

void iris_call_closure_void(IrisVal* closure, ...) {
    (void)closure;
}

/* ---- Trait-object (dyn Trait) helpers (Phase 91) ---- */

#define IRIS_TAG_TRAIT_OBJECT 22

/* Layout: { void* data, void* vtable_id } handled as a small heap object. */
static IrisVal* iris_make_trait_object_impl(void* data, void* vtable_id) {
    struct IrisTraitObject {
        void* data;
        void* vtable_id;
    };
    struct IrisTraitObject* p = (struct IrisTraitObject*)xmalloc(sizeof(struct IrisTraitObject));
    p->data = data;
    p->vtable_id = vtable_id;
    IrisVal* r = (IrisVal*)xmalloc(sizeof(IrisVal));
    r->tag = IRIS_TAG_TRAIT_OBJECT;
    r->ptr = p;
    return r;
}

IrisVal* iris_make_trait_object(void* data, void* vtable_id) {
    return iris_make_trait_object_impl(data, vtable_id);
}

/* Stub: in stub backend, the method-name string paired with the vtable_id
   is not actually resolved against any registered table. Return NULL so
   error paths surface rather than silently miscalling. */
IrisVal* iris_dyn_call(IrisVal* obj, const char* method_name, int32_t nargs, ...) {
    (void)obj;
    (void)method_name;
    (void)nargs;
    return NULL;
}

/* ---- Closure accessor helpers (called from generated LLVM IR) ---- */

void* iris_closure_fn(IrisVal* closure) {
    return ((IrisClosure*)closure->ptr)->fn;
}

int iris_closure_ncaptures(IrisVal* closure) {
    return ((IrisClosure*)closure->ptr)->captures->len;
}

IrisVal* iris_closure_get_capture(IrisVal* closure, int idx) {
    return ((IrisClosure*)closure->ptr)->captures->data[idx];
}

/* ======================================================================== */
/*  Terminal / Interactive Input                                            */
/* ======================================================================== */

int64_t iris_read_key(void) {
#if defined(__wasm__)
    /* WASM: terminal raw mode not available; use getchar */
    int c = getchar();
    return (int64_t)c;
#elif defined(_WIN32)
    /* Windows: use _getch() — no echo, no Enter needed */
    int c = _getch();
    /* Extended keys (arrows, F-keys) produce 0 or 224 prefix */
    if (c == 0 || c == 224) {
        int ext = _getch();
        /* Encode as -(ext) to distinguish from regular keys */
        return -(int64_t)ext;
    }
    return (int64_t)c;
#else
    /* POSIX: switch tty to raw mode, read one byte, restore */
    struct termios old, raw;
    tcgetattr(STDIN_FILENO, &old);
    raw = old;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    int c = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &old);
    return (int64_t)c;
#endif
}

char* iris_read_password(const char* prompt) {
    if (prompt && *prompt) fputs(prompt, stdout);
    fflush(stdout);

    size_t cap = 256, len = 0;
    char* buf = (char*)xmalloc(cap);

#if defined(__wasm__)
    /* WASM: no echo-toggle; just read line from stdin */
    {
        int c;
        while ((c = getchar()) != '\n' && c != EOF) {
            if (c == 127 && len > 0) { len--; continue; }
            if (len + 1 >= cap) { cap *= 2; buf = (char*)realloc(buf, cap); }
            buf[len++] = (char)c;
        }
    }
#elif defined(_WIN32)
    int c;
    while ((c = _getch()) != '\r' && c != '\n' && c != EOF) {
        if (c == '\b' && len > 0) { len--; continue; }
        if (c == '\b') continue;
        if (len + 1 >= cap) { cap *= 2; buf = (char*)realloc(buf, cap); }
        buf[len++] = (char)c;
    }
#else
    struct termios old, noecho;
    tcgetattr(STDIN_FILENO, &old);
    noecho = old;
    noecho.c_lflag &= ~ECHO;
    tcsetattr(STDIN_FILENO, TCSANOW, &noecho);
    int c;
    while ((c = getchar()) != '\n' && c != EOF) {
        if (c == 127 && len > 0) { len--; continue; }
        if (len + 1 >= cap) { cap *= 2; buf = (char*)realloc(buf, cap); }
        buf[len++] = (char)c;
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &old);
#endif
    buf[len] = '\0';
    putchar('\n');
    return buf;
}

void iris_term_clear(void) {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(h, &csbi);
    DWORD cells = csbi.dwSize.X * csbi.dwSize.Y, written;
    COORD origin = {0, 0};
    FillConsoleOutputCharacter(h, ' ', cells, origin, &written);
    FillConsoleOutputAttribute(h, csbi.wAttributes, cells, origin, &written);
    SetConsoleCursorPosition(h, origin);
#else
    fputs("\033[2J\033[H", stdout);
    fflush(stdout);
#endif
}

void iris_term_cursor(int64_t row, int64_t col) {
#ifdef _WIN32
    COORD pos = {(SHORT)(col - 1), (SHORT)(row - 1)};
    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
#else
    printf("\033[%lld;%lldH", (long long)row, (long long)col);
    fflush(stdout);
#endif
}

void iris_term_show_cursor(int show) {
#ifdef _WIN32
    CONSOLE_CURSOR_INFO ci;
    GetConsoleCursorInfo(GetStdHandle(STD_OUTPUT_HANDLE), &ci);
    ci.bVisible = show ? TRUE : FALSE;
    SetConsoleCursorInfo(GetStdHandle(STD_OUTPUT_HANDLE), &ci);
#else
    fputs(show ? "\033[?25h" : "\033[?25l", stdout);
    fflush(stdout);
#endif
}

void iris_term_set_color(int64_t fg, int64_t bg) {
    /* ANSI 256-color: foreground=38;5;N, background=48;5;N */
    if (fg >= 0) printf("\033[38;5;%lldm", (long long)fg);
    if (bg >= 0) printf("\033[48;5;%lldm", (long long)bg);
    fflush(stdout);
}

void iris_term_reset(void) {
    fputs("\033[0m", stdout);
    fflush(stdout);
}

int64_t iris_term_rows(void) {
#if defined(__wasm__)
    return 24; /* fallback */
#elif defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
#else
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return (int64_t)w.ws_row;
#endif
}

int64_t iris_term_cols(void) {
#if defined(__wasm__)
    return 80; /* fallback */
#elif defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return csbi.srWindow.Right - csbi.srWindow.Left + 1;
#else
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return (int64_t)w.ws_col;
#endif
}

/* ======================================================================== */
/*  UDP / TCP Networking                                                     */
/* ======================================================================== */

/* Forward declaration — defined in TCP section below. */
#ifdef _WIN32
static void ensure_wsa(void);
#endif

int64_t iris_udp_open(int64_t port) {
#ifdef __IRIS_WASM_STUB
    (void)port; return -1; /* no networking on WASM P1 */
#elif defined(_WIN32)
    ensure_wsa();
    SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == INVALID_SOCKET) return -1;
    if (port > 0) {
        struct sockaddr_in addr = {0};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port);
        addr.sin_addr.s_addr = INADDR_ANY;
        if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
            closesocket(s); return -1;
        }
    }
    return (int64_t)s;
#else
    int s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s < 0) return -1;
    if (port > 0) {
        struct sockaddr_in addr = {0};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port);
        addr.sin_addr.s_addr = INADDR_ANY;
        if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
            close(s); return -1;
        }
    }
    return (int64_t)s;
#endif
}

#ifndef __IRIS_WASM_STUB
void iris_udp_send(int64_t fd, const char* addr_port, int64_t data_len) {
    /* addr_port format: "host:port:data" — data starts after second colon */
    char host[256] = {0}; uint16_t port = 0;
    char* p = strdup(addr_port);
    char* colon = strrchr(p, ':');
    if (!colon) { free(p); return; }
    char* data = colon + 1;
    *colon = '\0';
    char* port_colon = strrchr(p, ':');
    if (port_colon) { port = (uint16_t)atoi(port_colon + 1); *port_colon = '\0'; strncpy(host, p, sizeof(host)-1); }
    else { strncpy(host, p, sizeof(host)-1); }
    struct sockaddr_in dst = {0};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &dst.sin_addr) != 1) dst.sin_addr.s_addr = INADDR_NONE;
    size_t dlen = data_len > 0 ? (size_t)data_len : strlen(data);
#ifdef _WIN32
    sendto((SOCKET)fd, data, (int)dlen, 0, (struct sockaddr*)&dst, sizeof(dst));
#else
    sendto((int)fd, data, dlen, 0, (struct sockaddr*)&dst, sizeof(dst));
#endif
    free(p);
}

char* iris_udp_recv(int64_t fd) {
    char buf[65536];
    struct sockaddr_in src;
#ifdef _WIN32
    int slen = sizeof(src);
    int n = recvfrom((SOCKET)fd, buf, sizeof(buf)-1, 0, (struct sockaddr*)&src, &slen);
#else
    socklen_t slen = sizeof(src);
    int n = recvfrom((int)fd, buf, sizeof(buf)-1, 0, (struct sockaddr*)&src, &slen);
#endif
    if (n < 0) { char* e = (char*)xmalloc(1); *e = '\0'; return e; }
    buf[n] = '\0';
    char ip_buf[64];
    inet_ntop(AF_INET, &src.sin_addr, ip_buf, sizeof(ip_buf));
    uint16_t port = ntohs(src.sin_port);
    size_t needed = strlen(ip_buf) + 6 + n + 2;
    char* result = (char*)xmalloc(needed);
    snprintf(result, needed, "%s:%d:%s", ip_buf, port, buf);
    return result;
}

void iris_udp_close(int64_t fd) {
#ifdef _WIN32
    closesocket((SOCKET)fd);
#else
    close((int)fd);
#endif
}
#else
/* WASM: networking stubs */
void iris_udp_send(int64_t fd, const char* addr_port, int64_t data_len) { (void)fd;(void)addr_port;(void)data_len; }
char* iris_udp_recv(int64_t fd) { (void)fd;char*e=xmalloc(1);*e='\0';return e; }
void iris_udp_close(int64_t fd) { (void)fd; }
#endif

/* ======================================================================== */
/*  HTTP (extended)                                                          */
/* ======================================================================== */

/* Three parameters, matching the IRIS-level `http_request(method, url, body)`.
 *
 * This took a `content_type` fourth parameter that IRIS never passed, so
 * codegen emitted a three-argument call against a four-parameter declaration
 * and the callee read a garbage pointer -- an access violation on every native
 * `http_request`. Fourth instance of a call disagreeing with its own `declare`,
 * after iris_select, json_stringify and the FFI argument array (#33).
 *
 * KNOWN LIMITATION: any non-GET method is still sent as POST, because this
 * delegates to `iris_http_post`, whose request line hardcodes the verb. The
 * interpreter builds the line itself and honours the method, so the two
 * backends disagree for PUT and PATCH. See known-issues #43. */
char* iris_http_request(const char* method, const char* url,
                        const char* body) {
    const char* content_type = "text/plain";
#ifdef __IRIS_WASM_STUB
    (void)method;(void)url;(void)body;(void)content_type;
    { char* e = (char*)xmalloc(1); *e = '\0'; return e; }
#else
    /* Delegate to GET or POST based on method */
    if (!method || strcmp(method, "GET") == 0) return iris_http_get(url);
    return iris_http_post(url, body ? body : "",
                         content_type ? content_type : "application/json");
#endif
}

/* ======================================================================== */
/*  TCP Networking                                                          */
/* ======================================================================== */

#ifndef __IRIS_WASM_STUB
#ifdef _WIN32
static int wsa_initialized = 0;
static void ensure_wsa(void) {
    if (!wsa_initialized) {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
        wsa_initialized = 1;
    }
}
#endif

int64_t iris_tcp_connect(const char* host, int64_t port) {
#ifdef _WIN32
    ensure_wsa();
    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return -1;
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%lld", (long long)port);
    if (getaddrinfo(host, port_str, &hints, &res) != 0) { closesocket(s); return -1; }
    if (connect(s, res->ai_addr, (int)res->ai_addrlen) != 0) { freeaddrinfo(res); closesocket(s); return -1; }
    freeaddrinfo(res);
    return (int64_t)s;
#else
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return -1;
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%lld", (long long)port);
    if (getaddrinfo(host, port_str, &hints, &res) != 0) { close(s); return -1; }
    if (connect(s, res->ai_addr, res->ai_addrlen) != 0) { freeaddrinfo(res); close(s); return -1; }
    freeaddrinfo(res);
    return (int64_t)s;
#endif
}

int64_t iris_tcp_listen(int64_t port) {
#ifdef _WIN32
    ensure_wsa();
    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return -1;
    int opt = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
    struct sockaddr_in addr = {0};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons((u_short)port);
    if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) != 0) { closesocket(s); return -1; }
    if (listen(s, SOMAXCONN) != 0) { closesocket(s); return -1; }
    return (int64_t)s;
#else
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return -1;
    int opt = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {0};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons((uint16_t)port);
    if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) != 0) { close(s); return -1; }
    if (listen(s, SOMAXCONN) != 0) { close(s); return -1; }
    return (int64_t)s;
#endif
}

int64_t iris_tcp_accept(int64_t listener) {
#ifdef _WIN32
    SOCKET c = accept((SOCKET)listener, NULL, NULL);
    return (c == INVALID_SOCKET) ? -1 : (int64_t)c;
#else
    int c = accept((int)listener, NULL, NULL);
    return (int64_t)c;
#endif
}

char* iris_tcp_read(int64_t conn) {
    char buf[8192];
    int n;
#ifdef _WIN32
    n = recv((SOCKET)conn, buf, sizeof(buf) - 1, 0);
#else
    n = recv((int)conn, buf, sizeof(buf) - 1, 0);
#endif
    if (n <= 0) {
        char* e = (char*)xmalloc(1);
        e[0] = '\0';
        return e;
    }
    buf[n] = '\0';
    char* result = (char*)xmalloc(n + 1);
    memcpy(result, buf, n + 1);
    return result;
}

void iris_tcp_write(int64_t conn, const char* data) {
    if (!data) return;
    size_t len = strlen(data);
#ifdef _WIN32
    send((SOCKET)conn, data, (int)len, 0);
#else
    send((int)conn, data, len, 0);
#endif
}

void iris_tcp_close(int64_t conn) {
#ifdef _WIN32
    closesocket((SOCKET)conn);
#else
    close((int)conn);
#endif
}
#else /* __IRIS_WASM_STUB */
/* WASM P1: no TCP networking */
int64_t iris_tcp_connect(const char* host, int64_t port) { (void)host;(void)port;return -1; }
int64_t iris_tcp_listen(int64_t port) { (void)port;return -1; }
int64_t iris_tcp_accept(int64_t listener) { (void)listener;return -1; }
char* iris_tcp_read(int64_t conn) { (void)conn;char*e=xmalloc(1);*e='\0';return e; }
void iris_tcp_write(int64_t conn, const char* data) { (void)conn;(void)data; }
void iris_tcp_close(int64_t conn) { (void)conn; }
#endif /* __IRIS_WASM_STUB */

/* ======================================================================== */
/*  HTTP (simple implementation using TCP sockets)                          */
/* ======================================================================== */

#ifndef __IRIS_WASM_STUB
/* Parse a URL into host, port, path.  Returns 0 on success. */
static int parse_url(const char* url, char* host, int* port, char* path) {
    *port = 80;
    const char* p = url;
    if (strncmp(p, "http://", 7) == 0)       { p += 7; *port = 80; }
    else if (strncmp(p, "https://", 8) == 0)  { p += 8; *port = 443; }
    const char* slash = strchr(p, '/');
    const char* colon = strchr(p, ':');
    if (colon && (!slash || colon < slash)) {
        size_t hlen = colon - p;
        memcpy(host, p, hlen); host[hlen] = '\0';
        *port = atoi(colon + 1);
        p = slash ? slash : p + strlen(p);
    } else if (slash) {
        size_t hlen = slash - p;
        memcpy(host, p, hlen); host[hlen] = '\0';
        p = slash;
    } else {
        strcpy(host, p);
        p = p + strlen(p);
    }
    if (*p == '/') strcpy(path, p);
    else           strcpy(path, "/");
    return 0;
}

char* iris_http_get(const char* url) {
    char host[256] = {0}, path[2048] = {0};
    int port = 80;
    if (parse_url(url, host, &port, path) != 0) {
        char* e = (char*)xmalloc(1); e[0] = '\0'; return e;
    }
    int64_t fd = iris_tcp_connect(host, port);
    if (fd < 0) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }

    /* Send HTTP/1.0 GET request */
    char req[4096];
    snprintf(req, sizeof(req),
        "GET %s HTTP/1.0\r\nHost: %s\r\nConnection: close\r\n\r\n", path, host);
    iris_tcp_write(fd, req);

    /* Read full response */
    size_t cap = 16384, len = 0;
    char* resp = (char*)xmalloc(cap);
    for (;;) {
        char buf[4096];
        int n;
#ifdef _WIN32
        n = recv((SOCKET)fd, buf, sizeof(buf), 0);
#else
        n = recv((int)fd, buf, sizeof(buf), 0);
#endif
        if (n <= 0) break;
        while (len + n + 1 > cap) { cap *= 2; resp = (char*)realloc(resp, cap); }
        memcpy(resp + len, buf, n);
        len += n;
    }
    resp[len] = '\0';
    iris_tcp_close(fd);

    /* Skip HTTP headers — find \r\n\r\n */
    char* body = strstr(resp, "\r\n\r\n");
    if (body) {
        body += 4;
        size_t blen = len - (body - resp);
        char* result = (char*)xmalloc(blen + 1);
        memcpy(result, body, blen);
        result[blen] = '\0';
        free(resp);
        return result;
    }
    return resp; /* No header separator found — return as-is */
}

char* iris_http_post(const char* url, const char* body, const char* content_type) {
    char host[256] = {0}, path[2048] = {0};
    int port = 80;
    if (parse_url(url, host, &port, path) != 0) {
        char* e = (char*)xmalloc(1); e[0] = '\0'; return e;
    }
    int64_t fd = iris_tcp_connect(host, port);
    if (fd < 0) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }

    size_t body_len = body ? strlen(body) : 0;
    char req[8192];
    snprintf(req, sizeof(req),
        "POST %s HTTP/1.0\r\nHost: %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
        path, host, content_type ? content_type : "text/plain", body_len);
    iris_tcp_write(fd, req);
    if (body) iris_tcp_write(fd, body);

    size_t cap = 16384, len = 0;
    char* resp = (char*)xmalloc(cap);
    for (;;) {
        char buf[4096];
        int n;
#ifdef _WIN32
        n = recv((SOCKET)fd, buf, sizeof(buf), 0);
#else
        n = recv((int)fd, buf, sizeof(buf), 0);
#endif
        if (n <= 0) break;
        while (len + n + 1 > cap) { cap *= 2; resp = (char*)realloc(resp, cap); }
        memcpy(resp + len, buf, n);
        len += n;
    }
    resp[len] = '\0';
    iris_tcp_close(fd);

    char* hdr_end = strstr(resp, "\r\n\r\n");
    if (hdr_end) {
        hdr_end += 4;
        size_t blen = len - (hdr_end - resp);
        char* result = (char*)xmalloc(blen + 1);
        memcpy(result, hdr_end, blen);
        result[blen] = '\0';
        free(resp);
        return result;
    }
    return resp;
}

char* iris_http_post_json(const char* url, const char* json_body) {
    return iris_http_post(url, json_body, "application/json");
}
#else /* __IRIS_WASM_STUB */
char* iris_http_get(const char* url) { (void)url;char*e=xmalloc(1);*e='\0';return e; }
char* iris_http_post(const char* url, const char* body, const char* content_type) { (void)url;(void)body;(void)content_type;char*e=xmalloc(1);*e='\0';return e; }
char* iris_http_post_json(const char* url, const char* json_body) { return iris_http_post(url, json_body, "application/json"); }
#endif /* __IRIS_WASM_STUB */

/* ======================================================================== */
/*  JSON (minimal recursive descent parser + serializer)                    */
/* ======================================================================== */

static const char* json_skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static IrisVal* json_parse_value(const char** p);

static IrisVal* json_parse_string(const char** p) {
    if (**p != '"') return iris_box_str("");
    (*p)++;
    const char* start = *p;
    size_t cap = 256, len = 0;
    char* buf = (char*)xmalloc(cap);
    while (**p && **p != '"') {
        if (**p == '\\') {
            (*p)++;
            char c = **p;
            switch (c) {
                case 'n': buf[len++] = '\n'; break;
                case 't': buf[len++] = '\t'; break;
                case 'r': buf[len++] = '\r'; break;
                case '"': buf[len++] = '"'; break;
                case '\\': buf[len++] = '\\'; break;
                case '/': buf[len++] = '/'; break;
                default: buf[len++] = c; break;
            }
        } else {
            buf[len++] = **p;
        }
        if (len + 2 >= cap) { cap *= 2; buf = (char*)realloc(buf, cap); }
        (*p)++;
    }
    if (**p == '"') (*p)++;
    buf[len] = '\0';
    IrisVal* v = iris_box_str(buf);
    free(buf);
    return v;
}

static IrisVal* json_parse_number(const char** p) {
    const char* start = *p;
    int is_float = 0;
    if (**p == '-') (*p)++;
    while (**p >= '0' && **p <= '9') (*p)++;
    if (**p == '.') { is_float = 1; (*p)++; while (**p >= '0' && **p <= '9') (*p)++; }
    if (**p == 'e' || **p == 'E') { is_float = 1; (*p)++; if (**p == '+' || **p == '-') (*p)++; while (**p >= '0' && **p <= '9') (*p)++; }
    if (is_float) return iris_box_f64(strtod(start, NULL));
    return iris_box_i64(strtoll(start, NULL, 10));
}

static IrisVal* json_parse_array(const char** p) {
    (*p)++; /* skip '[' */
    IrisList* list = iris_list_new();
    *p = json_skip_ws(*p);
    if (**p == ']') {
        (*p)++;
        return iris_box_list(list);
    }
    for (;;) {
        IrisVal* elem = json_parse_value(p);
        iris_list_push(list, elem);
        *p = json_skip_ws(*p);
        if (**p == ',') { (*p)++; *p = json_skip_ws(*p); }
        else break;
    }
    if (**p == ']') (*p)++;
    return iris_box_list(list);
}

static IrisVal* json_parse_object(const char** p) {
    (*p)++; /* skip '{' */
    IrisMap* map = iris_map_new();
    *p = json_skip_ws(*p);
    if (**p == '}') {
        (*p)++;
        return iris_box_map(map);
    }
    for (;;) {
        *p = json_skip_ws(*p);
        /* Parse key (must be a string) */
        if (**p != '"') break;
        IrisVal* kv = json_parse_string(p);
        char* key = kv->str ? kv->str : "";
        *p = json_skip_ws(*p);
        if (**p == ':') (*p)++;
        *p = json_skip_ws(*p);
        IrisVal* val = json_parse_value(p);
        (void)key;
        iris_map_set(map, kv, val);
        *p = json_skip_ws(*p);
        if (**p == ',') { (*p)++; }
        else break;
    }
    if (**p == '}') (*p)++;
    return iris_box_map(map);
}

static IrisVal* json_parse_value(const char** p) {
    *p = json_skip_ws(*p);
    if (**p == '"') return json_parse_string(p);
    if (**p == '{') return json_parse_object(p);
    if (**p == '[') return json_parse_array(p);
    if (**p == 't' && strncmp(*p, "true", 4) == 0)  { *p += 4; return iris_box_bool(1); }
    if (**p == 'f' && strncmp(*p, "false", 5) == 0)  { *p += 5; return iris_box_bool(0); }
    if (**p == 'n' && strncmp(*p, "null", 4) == 0)   { *p += 4; return iris_box_option(iris_make_none()); }
    if (**p == '-' || (**p >= '0' && **p <= '9')) return json_parse_number(p);
    return iris_box_str(""); /* parse error fallback */
}

IrisVal* iris_json_parse(const char* str) {
    if (!str) return iris_box_str("");
    const char* p = str;
    return json_parse_value(&p);
}

/* Stringify helper — recursive */
static void json_stringify_val(IrisVal* v, char** out, size_t* len, size_t* cap) {
    #define JSON_APPEND(s) do { \
        size_t slen = strlen(s); \
        while (*len + slen + 1 > *cap) { *cap *= 2; *out = (char*)realloc(*out, *cap); } \
        memcpy(*out + *len, s, slen); *len += slen; \
    } while(0)
    #define JSON_APPEND_CHAR(c) do { \
        if (*len + 2 > *cap) { *cap *= 2; *out = (char*)realloc(*out, *cap); } \
        (*out)[(*len)++] = (c); \
    } while(0)

    if (!v) { JSON_APPEND("null"); return; }
    switch (v->tag) {
        case IRIS_TAG_I64: {
            char buf[32]; snprintf(buf, sizeof(buf), "%lld", (long long)v->i64);
            JSON_APPEND(buf); break;
        }
        case IRIS_TAG_I32: {
            char buf[32]; snprintf(buf, sizeof(buf), "%d", v->i32);
            JSON_APPEND(buf); break;
        }
        case IRIS_TAG_F64: {
            char buf[64]; snprintf(buf, sizeof(buf), "%.17g", v->f64);
            JSON_APPEND(buf); break;
        }
        case IRIS_TAG_F32: {
            char buf[64]; snprintf(buf, sizeof(buf), "%.9g", (double)v->f32);
            JSON_APPEND(buf); break;
        }
        case IRIS_TAG_BOOL:
            JSON_APPEND(v->boolean ? "true" : "false"); break;
        case IRIS_TAG_STR: {
            JSON_APPEND_CHAR('"');
            if (v->str) {
                for (const char* s = v->str; *s; s++) {
                    switch (*s) {
                        case '"':  JSON_APPEND("\\\""); break;
                        case '\\': JSON_APPEND("\\\\"); break;
                        case '\n': JSON_APPEND("\\n"); break;
                        case '\r': JSON_APPEND("\\r"); break;
                        case '\t': JSON_APPEND("\\t"); break;
                        default:   JSON_APPEND_CHAR(*s); break;
                    }
                }
            }
            JSON_APPEND_CHAR('"');
            break;
        }
        case IRIS_TAG_LIST: {
            IrisList* l = (IrisList*)v->ptr;
            JSON_APPEND_CHAR('[');
            if (l) {
                for (size_t i = 0; i < l->len; i++) {
                    if (i > 0) JSON_APPEND_CHAR(',');
                    json_stringify_val(l->data[i], out, len, cap);
                }
            }
            JSON_APPEND_CHAR(']');
            break;
        }
        case IRIS_TAG_MAP: {
            IrisMap* m = (IrisMap*)v->ptr;
            JSON_APPEND_CHAR('{');
            int first = 1;
            if (m) {
                for (size_t i = 0; i < m->n_buckets; i++) {
                    IrisMapEntry* e = m->buckets[i];
                    while (e) {
                        if (!first) JSON_APPEND_CHAR(',');
                        first = 0;
                        JSON_APPEND_CHAR('"');
                        if (e->key) JSON_APPEND(e->key);
                        JSON_APPEND_CHAR('"');
                        JSON_APPEND_CHAR(':');
                        json_stringify_val(e->val, out, len, cap);
                        e = e->next;
                    }
                }
            }
            JSON_APPEND_CHAR('}');
            break;
        }
        case IRIS_TAG_OPTION: {
            IrisOption* opt = (IrisOption*)v->ptr;
            if (opt && opt->has_value) json_stringify_val(opt->value, out, len, cap);
            else JSON_APPEND("null");
            break;
        }
        case IRIS_TAG_RESULT: {
            IrisResult* res = (IrisResult*)v->ptr;
            if (res && res->is_ok) {
                JSON_APPEND("{\"ok\":");
                json_stringify_val(res->value, out, len, cap);
                JSON_APPEND_CHAR('}');
            } else {
                JSON_APPEND("{\"err\":");
                if (res) json_stringify_val(res->value, out, len, cap);
                else JSON_APPEND("null");
                JSON_APPEND_CHAR('}');
            }
            break;
        }
        case IRIS_TAG_TUPLE: {
            IrisList* t = (IrisList*)v->ptr;
            JSON_APPEND_CHAR('[');
            if (t) {
                for (size_t i = 0; i < t->len; i++) {
                    if (i > 0) JSON_APPEND_CHAR(',');
                    json_stringify_val(t->data[i], out, len, cap);
                }
            }
            JSON_APPEND_CHAR(']');
            break;
        }
        case IRIS_TAG_STRUCT: {
            // Struct is stored as IrisList of field values.
            // Field names are not available at runtime; use numeric indices.
            IrisList* fields = (IrisList*)v->ptr;
            JSON_APPEND_CHAR('{');
            if (fields) {
                for (size_t i = 0; i < fields->len; i++) {
                    if (i > 0) JSON_APPEND_CHAR(',');
                    char idx_buf[16]; snprintf(idx_buf, sizeof(idx_buf), "\"%zu\":", i);
                    JSON_APPEND(idx_buf);
                    json_stringify_val(fields->data[i], out, len, cap);
                }
            }
            JSON_APPEND_CHAR('}');
            break;
        }
        case IRIS_TAG_ENUM: {
            IrisEnum* e = (IrisEnum*)v->ptr;
            if (e) {
                char buf[64]; snprintf(buf, sizeof(buf), "{\"tag\":%lld,\"fields\":[", (long long)e->tag);
                JSON_APPEND(buf);
                for (size_t i = 0; i < e->len; i++) {
                    if (i > 0) JSON_APPEND_CHAR(',');
                    json_stringify_val(e->fields[i], out, len, cap);
                }
                JSON_APPEND("]}");
            } else {
                JSON_APPEND("null");
            }
            break;
        }
        case IRIS_TAG_UNIT:
            JSON_APPEND("null"); break;
        default: JSON_APPEND("null"); break;
    }
    #undef JSON_APPEND
    #undef JSON_APPEND_CHAR
}

char* iris_json_stringify(IrisVal* val) {
    size_t cap = 256, len = 0;
    char* out = (char*)xmalloc(cap);
    json_stringify_val(val, &out, &len, &cap);
    out[len] = '\0';
    return out;
}

/* ======================================================================== */
/*  Set collection (uses list with linear search — simple and correct)      */
/* ======================================================================== */

/* iris_val_equal already defined above */

IrisList* iris_set_new(void) { return iris_list_new(); }

void iris_set_add(IrisList* set, IrisVal* val) {
    if (!set || !val) return;
    for (size_t i = 0; i < set->len; i++) {
        if (iris_val_equal(set->data[i], val)) return; /* already present */
    }
    iris_list_push(set, val);
}

int iris_set_contains(IrisList* set, IrisVal* val) {
    if (!set || !val) return 0;
    for (size_t i = 0; i < set->len; i++) {
        if (iris_val_equal(set->data[i], val)) return 1;
    }
    return 0;
}

void iris_set_remove(IrisList* set, IrisVal* val) {
    if (!set || !val) return;
    for (size_t i = 0; i < set->len; i++) {
        if (iris_val_equal(set->data[i], val)) {
            /* Shift remaining elements */
            for (size_t j = i; j + 1 < set->len; j++)
                set->data[j] = set->data[j+1];
            set->len--;
            return;
        }
    }
}

int64_t iris_set_len(IrisList* set) { return set ? (int64_t)set->len : 0; }

IrisList* iris_set_to_list(IrisList* set) {
    IrisList* out = iris_list_new();
    if (!set) return out;
    for (size_t i = 0; i < set->len; i++) iris_list_push(out, set->data[i]);
    return out;
}

/* ======================================================================== */
/*  Regex (simple pattern matching — no external dependency)                */
/* ======================================================================== */
/* We implement a simple regex subset: exact match, ., *, +, ?, ^, $        */
/* Extended: [...] char classes, (...) grouping, | alternation,              */
/*           \d \w \s shorthand classes, [^...] negated classes             */
/* For full regex, compiled code can use platform regex via FFI.             */

static int simple_match(const char* pat, const char* str);
static int match_here(const char* re, const char* text);

static int match_char_class(const char* cc, char c, const char** end) {
    /* cc points past the opening '['. Returns 1 if c matches, sets *end past ']'. */
    int negated = 0;
    if (*cc == '^') { negated = 1; cc++; }
    int matched = 0;
    while (*cc != '\0' && *cc != ']') {
        char lo = *cc;
        if (cc[1] == '-' && cc[2] != '\0' && cc[2] != ']') {
            /* Range: lo-hi */
            char hi = cc[2];
            if (c >= lo && c <= hi) matched = 1;
            cc += 3;
            if (*cc == '-') { cc++; } /* skip trailing dash if present */
        } else if (lo == '\\' && cc[1] != '\0') {
            /* Escape inside class */
            cc++;
            char esc = *cc++;
            if (esc == 'd' && isdigit((unsigned char)c)) matched = 1;
            else if (esc == 'w' && (isalnum((unsigned char)c) || c == '_')) matched = 1;
            else if (esc == 's' && (c == ' ' || c == '\t' || c == '\n' || c == '\r')) matched = 1;
            else if (c == esc) matched = 1;
        } else {
            if (c == lo) matched = 1;
            cc++;
        }
    }
    if (*cc == ']') cc++;
    *end = cc;
    return negated ? !matched : matched;
}

static int match_here(const char* re, const char* text) {
    if (re[0] == '\0') return 1;
    if (re[0] == '$' && re[1] == '\0') return *text == '\0';

    /* Grouping: (sub-pattern) */
    if (re[0] == '(') {
        /* Find matching ')' accounting for nested parens */
        int depth = 1;
        const char* p = re + 1;
        while (*p != '\0' && depth > 0) {
            if (*p == '(') depth++;
            else if (*p == ')') depth--;
            p++;
        }
        /* p now points past the matching ')' */
        size_t group_len = (size_t)(p - re - 2); /* exclude parens */
        char* group = (char*)xmalloc(group_len + 1);
        memcpy(group, re + 1, group_len);
        group[group_len] = '\0';

        /* Check for alternation inside the group */
        const char* alt = group;
        int found_alt = 0;
        int adepth = 0;
        for (size_t i = 0; i < group_len; i++) {
            if (group[i] == '(') adepth++;
            else if (group[i] == ')') adepth--;
            else if (group[i] == '|' && adepth == 0) {
                found_alt = 1;
                break;
            }
        }

        if (found_alt) {
            /* Split on | and try each alternative */
            char* save = group;
            char* alt_start = group;
            int match_result = 0;
            while (1) {
                char* sep = NULL;
                adepth = 0;
                for (char* q = alt_start; *q != '\0'; q++) {
                    if (*q == '(') adepth++;
                    else if (*q == ')') adepth--;
                    else if (*q == '|' && adepth == 0) { sep = q; break; }
                }
                if (sep) *sep = '\0';
                /* Try matching alt_start then the rest of re */
                /* Build combined pattern: alt + rest_of_re */
                size_t alt_len = strlen(alt_start);
                size_t rest_len = strlen(p);
                char* combined = (char*)xmalloc(alt_len + rest_len + 1);
                memcpy(combined, alt_start, alt_len);
                memcpy(combined + alt_len, p, rest_len + 1);
                if (simple_match(combined, text)) { match_result = 1; free(combined); break; }
                free(combined);
                if (sep) { alt_start = sep + 1; } else break;
            }
            free(save);
            return match_result;
        } else {
            /* No alternation: try matching the group content then the rest */
            size_t rest_len = strlen(p);
            char* combined = (char*)xmalloc(group_len + rest_len + 1);
            memcpy(combined, group, group_len);
            memcpy(combined + group_len, p, rest_len + 1);
            int result = simple_match(combined, text);
            free(combined);
            free(group);
            return result;
        }
    }

    /* Escape sequences: \d, \w, \s */
    if (re[0] == '\\' && re[1] != '\0') {
        char esc = re[1];
        int match = 0;
        if (esc == 'd') match = isdigit((unsigned char)*text);
        else if (esc == 'w') match = (isalnum((unsigned char)*text) || *text == '_');
        else if (esc == 's') match = (*text == ' ' || *text == '\t' || *text == '\n' || *text == '\r');
        else match = (*text == esc);
        const char* after_esc = re + 2;
        if (after_esc[0] == '+' || after_esc[0] == '*' || after_esc[0] == '?') {
            char q = after_esc[0];
            after_esc++;
            if (q == '+') {
                /* One or more */
                if (!match || *text == '\0') return 0;
                text++;
                /* Greedy: try matching rest first, then consume more */
                do {
                    if (match_here(after_esc, text)) return 1;
                } while (*text != '\0' && ((esc == 'd' && isdigit((unsigned char)*text)) ||
                                           (esc == 'w' && (isalnum((unsigned char)*text) || *text == '_')) ||
                                           (esc == 's' && (*text == ' ' || *text == '\t' || *text == '\n' || *text == '\r')) ||
                                           (esc != 'd' && esc != 'w' && esc != 's' && *text == esc)));
                return 0;
            } else if (q == '*') {
                /* Zero or more */
                do {
                    if (match_here(after_esc, text)) return 1;
                } while (*text != '\0' && ((esc == 'd' && isdigit((unsigned char)*text)) ||
                                           (esc == 'w' && (isalnum((unsigned char)*text) || *text == '_')) ||
                                           (esc == 's' && (*text == ' ' || *text == '\t' || *text == '\n' || *text == '\r')) ||
                                           (esc != 'd' && esc != 'w' && esc != 's' && *text == esc)));
                return 0;
            } else {
                /* ? : zero or one */
                if (match && *text != '\0') {
                    if (match_here(after_esc, text + 1)) return 1;
                }
                return match_here(after_esc, text);
            }
        }
        if (match && text[0] != '\0')
            return match_here(after_esc, text + 1);
        return 0;
    }

    /* Character class: [...] */
    if (re[0] == '[') {
        const char* end;
        int cm = match_char_class(re + 1, *text, &end);
        if (*end == '+' || *end == '*' || *end == '?') {
            char q = *end;
            end++;
            if (q == '+') {
                /* One or more */
                if (!cm || *text == '\0') return 0;
                text++;
                if (match_here(end, text)) return 1;
                while (*text != '\0') {
                    const char* dummy;
                    if (match_char_class(re + 1, *text, &dummy)) {
                        text++;
                        if (match_here(end, text)) return 1;
                    } else break;
                }
                return 0;
            } else if (q == '*') {
                /* Zero or more */
                if (match_here(end, text)) return 1;
                while (*text != '\0') {
                    const char* dummy;
                    if (match_char_class(re + 1, *text, &dummy)) {
                        text++;
                        if (match_here(end, text)) return 1;
                    } else break;
                }
                return 0;
            } else {
                /* ? : zero or one */
                if (cm && *text != '\0') {
                    if (match_here(end, text + 1)) return 1;
                }
                return match_here(end, text);
            }
        }
        if (cm && *text != '\0')
            return match_here(end, text + 1);
        return 0;
    }

    if (re[1] == '*') {
        /* Match zero or more of re[0] */
        do {
            if (match_here(re + 2, text)) return 1;
        } while (*text != '\0' && (re[0] == '.' || *text == re[0]) && text++);
        return 0;
    }
    if (re[1] == '+') {
        /* Match one or more of re[0] */
        while (*text != '\0' && (re[0] == '.' || *text == re[0])) {
            text++;
            if (match_here(re + 2, text)) return 1;
        }
        return 0;
    }
    if (re[1] == '?') {
        /* Match zero or one of re[0] */
        if (match_here(re + 2, text)) return 1;
        if (*text != '\0' && (re[0] == '.' || *text == re[0]))
            return match_here(re + 2, text + 1);
        return 0;
    }
    if (*text != '\0' && (re[0] == '.' || *text == re[0]))
        return match_here(re + 1, text + 1);
    return 0;
}

static int simple_match(const char* pat, const char* str) {
    if (pat[0] == '^') return match_here(pat + 1, str);
    /* Unanchored: try at every position */
    do {
        if (match_here(pat, str)) return 1;
    } while (*str++ != '\0');
    return 0;
}

int iris_regex_match(const char* pattern, const char* str) {
    if (!pattern || !str) return 0;
    return simple_match(pattern, str);
}

/*
 * mpl: match prefix length — returns chars consumed, or -1.
 */
static int mpl(const char* re, const char* text) {
    if (re[0] == '\0') return 0;
    if (re[0] == '$' && re[1] == '\0') return (*text == '\0') ? 0 : -1;

    /* Grouping: (sub-pattern) */
    if (re[0] == '(') {
        int depth = 1;
        const char* p = re + 1;
        while (*p != '\0' && depth > 0) {
            if (*p == '(') depth++;
            else if (*p == ')') { depth--; if (depth == 0) { p++; break; } }
            p++;
        }
        size_t glen = (size_t)(p - re - 2);
        char* group = (char*)xmalloc(glen + 1);
        memcpy(group, re + 1, glen);
        group[glen] = '\0';
        const char* rest = p;

        int found_alt = 0; int ad = 0;
        for (size_t i = 0; i < glen; i++) {
            if (group[i] == '(') ad++;
            else if (group[i] == ')') ad--;
            else if (group[i] == '|' && ad == 0) { found_alt = 1; break; }
        }

        int best = -1;
        if (found_alt) {
            char* save = group;
            char* astart = group;
            while (1) {
                char* sep = NULL; ad = 0;
                for (char* q = astart; *q != '\0'; q++) {
                    if (*q == '(') ad++;
                    else if (*q == ')') ad--;
                    else if (*q == '|' && ad == 0) { sep = q; break; }
                }
                if (sep) *sep = '\0';
                size_t al = strlen(astart); size_t rl = strlen(rest);
                char* c = (char*)xmalloc(al + rl + 1);
                memcpy(c, astart, al); memcpy(c + al, rest, rl + 1);
                int r = mpl(c, text);
                free(c);
                if (r >= 0 && r > best) best = r;
                if (sep) { *sep = '|'; astart = sep + 1; } else break;
            }
            free(save);
        } else {
            size_t rl = strlen(rest);
            char* c = (char*)xmalloc(glen + rl + 1);
            memcpy(c, group, glen); memcpy(c + glen, rest, rl + 1);
            best = mpl(c, text);
            free(c);
            free(group);
        }
        return best;
    }

    /* Escape: \d, \w, \s */
    if (re[0] == '\\' && re[1] != '\0') {
        char esc = re[1]; const char* af = re + 2;
        int m = 0;
        if (esc == 'd') m = isdigit((unsigned char)*text);
        else if (esc == 'w') m = (isalnum((unsigned char)*text) || *text == '_');
        else if (esc == 's') m = (*text == ' ' || *text == '\t' || *text == '\n' || *text == '\r');
        else m = (*text == esc);
        /* helper macro for counting matching chars */
        #define ESC_MATCH(ch) ((esc=='d' && isdigit((unsigned char)(ch))) || \
            (esc=='w' && (isalnum((unsigned char)(ch)) || (ch)=='_')) || \
            (esc=='s' && ((ch)==' '||(ch)=='\t'||(ch)=='\n'||(ch)=='\r')) || \
            (esc!='d' && esc!='w' && esc!='s' && (ch)==esc))
        if (af[0] == '+' || af[0] == '*' || af[0] == '?') {
            char q = af[0]; const char* rest = af + 1;
            if (q == '+') {
                if (!m) return -1;
                int cnt = 0; const char* t = text;
                while (*t != '\0' && ESC_MATCH(*t)) { cnt++; t++; }
                for (int i = cnt; i >= 1; i--) { int r = mpl(rest, text + i); if (r >= 0) return i + r; }
                return -1;
            } else if (q == '*') {
                int cnt = 0; const char* t = text;
                while (*t != '\0' && ESC_MATCH(*t)) { cnt++; t++; }
                for (int i = cnt; i >= 0; i--) { int r = mpl(rest, text + i); if (r >= 0) return i + r; }
                return -1;
            } else {
                if (m && *text != '\0') { int r = mpl(rest, text + 1); if (r >= 0) return 1 + r; }
                return mpl(rest, text);
            }
        }
        #undef ESC_MATCH
        if (m && *text != '\0') { int r = mpl(af, text + 1); if (r >= 0) return 1 + r; }
        return -1;
    }

    /* Character class: [...] */
    if (re[0] == '[') {
        const char* end;
        int cm = match_char_class(re + 1, *text, &end);
        if (*end == '+' || *end == '*' || *end == '?') {
            char q = *end; const char* rest = end + 1;
            if (q == '+') {
                if (!cm) return -1;
                int cnt = 0; const char* t = text;
                while (*t != '\0') { const char* d; if (!match_char_class(re+1,*t,&d)) break; cnt++; t++; }
                for (int i = cnt; i >= 1; i--) { int r = mpl(rest, text + i); if (r >= 0) return i + r; }
                return -1;
            } else if (q == '*') {
                int cnt = 0; const char* t = text;
                while (*t != '\0') { const char* d; if (!match_char_class(re+1,*t,&d)) break; cnt++; t++; }
                for (int i = cnt; i >= 0; i--) { int r = mpl(rest, text + i); if (r >= 0) return i + r; }
                return -1;
            } else {
                if (cm && *text != '\0') { int r = mpl(rest, text + 1); if (r >= 0) return 1 + r; }
                return mpl(rest, text);
            }
        }
        if (cm && *text != '\0') { int r = mpl(end, text + 1); if (r >= 0) return 1 + r; }
        return -1;
    }

    /* Single char with quantifiers */
    if (re[1] == '*') {
        int cnt = 0; const char* t = text;
        while (*t != '\0' && (re[0] == '.' || *t == re[0])) { cnt++; t++; }
        for (int i = cnt; i >= 0; i--) { int r = mpl(re + 2, text + i); if (r >= 0) return i + r; }
        return -1;
    }
    if (re[1] == '+') {
        if (*text == '\0' || (re[0] != '.' && *text != re[0])) return -1;
        int cnt = 0; const char* t = text;
        while (*t != '\0' && (re[0] == '.' || *t == re[0])) { cnt++; t++; }
        for (int i = cnt; i >= 1; i--) { int r = mpl(re + 2, text + i); if (r >= 0) return i + r; }
        return -1;
    }
    if (re[1] == '?') {
        if (*text != '\0' && (re[0] == '.' || *text == re[0])) {
            int r = mpl(re + 2, text + 1); if (r >= 0) return 1 + r;
        }
        return mpl(re + 2, text);
    }
    if (*text != '\0' && (re[0] == '.' || *text == re[0])) {
        int r = mpl(re + 1, text + 1); if (r >= 0) return 1 + r;
    }
    return -1;
}

IrisList* iris_regex_find_all(const char* pattern, const char* str) {
    IrisList* results = iris_list_new();
    if (!pattern || !str) return results;
    const char* pat = pattern;
    if (pat[0] == '^') pat++;
    size_t slen = strlen(str);
    size_t pos = 0;
    while (pos < slen) {
        int len = mpl(pat, str + pos);
        if (len > 0) {
            char* match = (char*)xmalloc(len + 1);
            memcpy(match, str + pos, len);
            match[len] = '\0';
            iris_list_push(results, iris_box_str(match));
            free(match);
            pos += len;
        } else {
            pos++;
        }
    }
    return results;
}

char* iris_regex_replace(const char* pattern, const char* str, const char* replacement) {
    if (!pattern || !str || !replacement) {
        char* e = (char*)xmalloc(1); e[0] = '\0'; return e;
    }
    const char* pat = pattern[0] == '^' ? pattern + 1 : pattern;
    size_t slen = strlen(str);
    size_t rlen = strlen(replacement);
    size_t cap = slen + rlen + 64;
    char* out = (char*)xmalloc(cap);
    size_t olen = 0;
    size_t pos = 0;
    int replaced = 0;
    while (pos < slen) {
        if (!replaced) {
            int len = mpl(pat, str + pos);
            if (len > 0) {
                while (olen + rlen + 1 > cap) { cap *= 2; out = (char*)realloc(out, cap); }
                memcpy(out + olen, replacement, rlen);
                olen += rlen;
                pos += len;
                replaced = 1;
                continue;
            }
        }
        if (olen + 2 > cap) { cap *= 2; out = (char*)realloc(out, cap); }
        out[olen++] = str[pos];
        pos++;
    }
    out[olen] = '\0';
    return out;
}

char* iris_regex_replace_all(const char* pattern, const char* str, const char* replacement) {
    if (!pattern || !str || !replacement) {
        char* e = (char*)xmalloc(1); e[0] = '\0'; return e;
    }
    const char* pat = pattern[0] == '^' ? pattern + 1 : pattern;
    size_t slen = strlen(str);
    size_t rlen = strlen(replacement);
    size_t cap = slen * 2 + rlen + 64;
    char* out = (char*)xmalloc(cap);
    size_t olen = 0;
    size_t pos = 0;
    while (pos < slen) {
        int len = mpl(pat, str + pos);
        if (len > 0) {
            while (olen + rlen + 1 > cap) { cap *= 2; out = (char*)realloc(out, cap); }
            memcpy(out + olen, replacement, rlen);
            olen += rlen;
            pos += len;
        } else {
            if (olen + 2 > cap) { cap *= 2; out = (char*)realloc(out, cap); }
            out[olen++] = str[pos];
            pos++;
        }
    }
    out[olen] = '\0';
    return out;
}

/* ======================================================================== */
/*  DateTime                                                                */
/* ======================================================================== */

char* iris_datetime_now(void) {
    time_t t = time(NULL);
    struct tm* tm = localtime(&t);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", tm);
    return iris_box_str(buf)->str;
}

int64_t iris_datetime_timestamp(void) {
    return (int64_t)time(NULL);
}

char* iris_datetime_format(int64_t timestamp, const char* fmt) {
    time_t t = (time_t)timestamp;
    struct tm* tm = localtime(&t);
    char buf[256];
    strftime(buf, sizeof(buf), fmt ? fmt : "%Y-%m-%dT%H:%M:%S", tm);
    size_t len = strlen(buf);
    char* result = (char*)xmalloc(len + 1);
    memcpy(result, buf, len + 1);
    return result;
}

/* ======================================================================== */
/*  OS / Path                                                               */
/* ======================================================================== */

char* iris_cwd(void) {
    char buf[4096];
#ifdef _WIN32
    if (_getcwd(buf, sizeof(buf)))
#else
    if (getcwd(buf, sizeof(buf)))
#endif
    {
        size_t len = strlen(buf);
        char* result = (char*)xmalloc(len + 1);
        memcpy(result, buf, len + 1);
        return result;
    }
    char* e = (char*)xmalloc(1); e[0] = '\0'; return e;
}

IrisList* iris_listdir(const char* path) {
    IrisList* list = iris_list_new();
    if (!path) return list;
#ifdef _WIN32
    char pattern[4096];
    snprintf(pattern, sizeof(pattern), "%s\\*", path);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return list;
    do {
        if (strcmp(fd.cFileName, ".") != 0 && strcmp(fd.cFileName, "..") != 0)
            iris_list_push(list, iris_box_str(fd.cFileName));
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR* d = opendir(path);
    if (!d) return list;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0)
            iris_list_push(list, iris_box_str(ent->d_name));
    }
    closedir(d);
#endif
    return list;
}

char* iris_path_join(const char* a, const char* b) {
    if (!a || !b) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }
    size_t alen = strlen(a), blen = strlen(b);
    char* result = (char*)xmalloc(alen + blen + 2);
#ifdef _WIN32
    char sep = '\\';
#else
    char sep = '/';
#endif
    memcpy(result, a, alen);
    if (alen > 0 && a[alen-1] != '/' && a[alen-1] != '\\') {
        result[alen] = sep;
        memcpy(result + alen + 1, b, blen);
        result[alen + 1 + blen] = '\0';
    } else {
        memcpy(result + alen, b, blen);
        result[alen + blen] = '\0';
    }
    return result;
}

int iris_path_exists(const char* path) {
    if (!path) return 0;
#ifdef _WIN32
    return _access(path, 0) == 0;
#else
    return access(path, F_OK) == 0;
#endif
}

int iris_mkdir(const char* path) {
    if (!path) return -1;
#ifdef _WIN32
    return _mkdir(path) == 0 ? 0 : -1;
#else
    return mkdir(path, 0755) == 0 ? 0 : -1;
#endif
}

int iris_remove_file(const char* path) {
    if (!path) return -1;
    return remove(path) == 0 ? 0 : -1;
}

/* ======================================================================== */
/*  Type introspection                                                      */
/* ======================================================================== */

char* iris_type_of(IrisVal* val) {
    if (!val) return iris_box_str("unit")->str;
    const char* name;
    switch (val->tag) {
        case IRIS_TAG_I64:     name = "i64"; break;
        case IRIS_TAG_I32:     name = "i32"; break;
        case IRIS_TAG_F64:     name = "f64"; break;
        case IRIS_TAG_F32:     name = "f32"; break;
        case IRIS_TAG_BOOL:    name = "bool"; break;
        case IRIS_TAG_STR:     name = "str"; break;
        case IRIS_TAG_LIST:    name = "list"; break;
        case IRIS_TAG_MAP:     name = "map"; break;
        case IRIS_TAG_OPTION:  name = "option"; break;
        case IRIS_TAG_RESULT:  name = "result"; break;
        case IRIS_TAG_CLOSURE: name = "closure"; break;
        case IRIS_TAG_TUPLE:   name = "tuple"; break;
        case IRIS_TAG_STRUCT:  name = "struct"; break;
        case IRIS_TAG_CHAN:     name = "channel"; break;
        case IRIS_TAG_ATOMIC:  name = "atomic"; break;
        case IRIS_TAG_MUTEX:   name = "mutex"; break;
        case IRIS_TAG_GRAD:    name = "grad"; break;
        case IRIS_TAG_SPARSE:  name = "sparse"; break;
        case IRIS_TAG_UNIT:    name = "unit"; break;
        case IRIS_TAG_ENUM:    name = "enum"; break;
        default:               name = "unknown"; break;
    }
    size_t len = strlen(name);
    char* result = (char*)xmalloc(len + 1);
    memcpy(result, name, len + 1);
    return result;
}

/* ======================================================================== */
/*  Random                                                                  */
/* ======================================================================== */

/*
 * Deterministic, seedable RNG -- SplitMix64.
 *
 * This replaces libc `rand()`/`srand()`, which was wrong here in three
 * separate ways:
 *
 *   1. It could not be seeded from IRIS, so an evolved system could never be
 *      replayed. For a language whose pitch is a *verifiable* autonomy layer,
 *      a self-evolving system that cannot be reproduced cannot be audited --
 *      you cannot ask "how did it get here" and get an answer.
 *   2. `RAND_MAX` is 32767 on this toolchain, so `rand()/RAND_MAX` produced
 *      only 32768 distinct values. Evolutionary search over a genome of f64
 *      genes was quantising every draw to 15 bits.
 *   3. The interpreter used an entirely different generator (a chained
 *      `DefaultHasher`), so the two backends produced different sequences for
 *      the same program -- and kept two independent streams besides, one for
 *      `random` and one for `random_range`.
 *
 * SplitMix64 is used because it is exactly reproducible with plain 64-bit
 * wrapping arithmetic, which means the Rust interpreter can implement it bit
 * for bit. `src/interp/mod.rs` carries the identical constants; the two are
 * asserted equal in tests/test_random_determinism.iris. Changing one without
 * the other reintroduces the divergence this replaced.
 */

static uint64_t iris_rng_state = 0;
static int64_t  iris_rng_seed_value = 0;
static int      iris_rng_seeded = 0;

static void iris_rng_autoseed(void) {
    if (iris_rng_seeded) return;
    uint64_t seed = (uint64_t)time(NULL);
#ifdef _WIN32
    seed ^= ((uint64_t)GetCurrentProcessId() << 16);
    seed ^= ((uint64_t)GetTickCount() & 0xFFFFu);
#else
    seed ^= ((uint64_t)getpid() << 16);
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        seed ^= ((uint64_t)ts.tv_nsec & 0xFFFFu);
    }
#endif
    iris_rng_state = seed;
    iris_rng_seed_value = (int64_t)seed;
    iris_rng_seeded = 1;
}

static uint64_t iris_rng_next(void) {
    iris_rng_autoseed();
    iris_rng_state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = iris_rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Set the stream. Returns the seed, so a run can log what it used. */
int64_t iris_seed(int64_t seed) {
    iris_rng_state = (uint64_t)seed;
    iris_rng_seed_value = seed;
    iris_rng_seeded = 1;
    return seed;
}

/* The seed currently in effect, auto-generating one if never set. This is what
 * makes a run reproducible after the fact: print it, pass it back to seed(). */
int64_t iris_random_seed(void) {
    iris_rng_autoseed();
    return iris_rng_seed_value;
}

double iris_random(void) {
    /* Top 53 bits scaled into [0, 1) -- the full mantissa, no quantisation. */
    return (double)(iris_rng_next() >> 11) / 9007199254740992.0;
}

int64_t iris_random_range(int64_t lo, int64_t hi) {
    if (hi <= lo) return lo;
    uint64_t span = (uint64_t)(hi - lo);
    return lo + (int64_t)(iris_rng_next() % span);
}

/* ======================================================================== */
/*  Hashing / Encoding                                                      */
/* ======================================================================== */

int64_t iris_hash(const char* str) {
    if (!str) return 0;
    /* djb2 hash */
    uint64_t hash = 5381;
    int c;
    while ((c = (unsigned char)*str++))
        hash = ((hash << 5) + hash) + c;
    return (int64_t)hash;
}

static const char b64_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

char* iris_base64_encode(const char* str) {
    if (!str) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }
    size_t slen = strlen(str);
    size_t olen = 4 * ((slen + 2) / 3);
    char* out = (char*)xmalloc(olen + 1);
    size_t j = 0;
    for (size_t i = 0; i < slen; i += 3) {
        uint32_t a = (unsigned char)str[i];
        uint32_t b = (i + 1 < slen) ? (unsigned char)str[i+1] : 0;
        uint32_t c = (i + 2 < slen) ? (unsigned char)str[i+2] : 0;
        uint32_t triple = (a << 16) | (b << 8) | c;
        out[j++] = b64_table[(triple >> 18) & 0x3F];
        out[j++] = b64_table[(triple >> 12) & 0x3F];
        out[j++] = (i + 1 < slen) ? b64_table[(triple >> 6) & 0x3F] : '=';
        out[j++] = (i + 2 < slen) ? b64_table[triple & 0x3F] : '=';
    }
    out[j] = '\0';
    return out;
}

static int b64_decode_char(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

char* iris_base64_decode(const char* str) {
    if (!str) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }
    size_t slen = strlen(str);
    size_t olen = slen / 4 * 3;
    char* out = (char*)xmalloc(olen + 1);
    size_t j = 0;
    for (size_t i = 0; i < slen; i += 4) {
        int a = b64_decode_char(str[i]);
        int b = (i+1 < slen) ? b64_decode_char(str[i+1]) : 0;
        int c = (i+2 < slen) ? b64_decode_char(str[i+2]) : 0;
        int d = (i+3 < slen) ? b64_decode_char(str[i+3]) : 0;
        if (a < 0) a = 0; if (b < 0) b = 0; if (c < 0) c = 0; if (d < 0) d = 0;
        uint32_t triple = ((uint32_t)a << 18) | ((uint32_t)b << 12) | ((uint32_t)c << 6) | (uint32_t)d;
        if (j < olen) out[j++] = (triple >> 16) & 0xFF;
        if (j < olen && str[i+2] != '=') out[j++] = (triple >> 8) & 0xFF;
        if (j < olen && str[i+3] != '=') out[j++] = triple & 0xFF;
    }
    out[j] = '\0';
    return out;
}

/* ======================================================================== */
/*  String extras                                                           */
/* ======================================================================== */

char* iris_char_at(const char* str, int64_t idx) {
    if (!str) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }
    size_t len = strlen(str);
    if (idx < 0 || (size_t)idx >= len) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }
    char* result = (char*)xmalloc(2);
    result[0] = str[idx];
    result[1] = '\0';
    return result;
}

char* iris_str_reverse(const char* str) {
    if (!str) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }
    size_t len = strlen(str);
    char* result = (char*)xmalloc(len + 1);
    for (size_t i = 0; i < len; i++) result[i] = str[len - 1 - i];
    result[len] = '\0';
    return result;
}

/* ======================================================================== */
/*  Phase 105: Extended builtins                                            */
/* ======================================================================== */

/* -- String extras -- */

char* iris_str_pad_left(const char* str, int64_t width, const char* pad) {
    if (!str) str = "";
    if (!pad || pad[0] == '\0') pad = " ";
    size_t slen = strlen(str);
    if ((int64_t)slen >= width) {
        char* r = (char*)xmalloc(slen + 1);
        memcpy(r, str, slen + 1);
        return r;
    }
    size_t pad_needed = (size_t)(width - (int64_t)slen);
    char* r = (char*)xmalloc((size_t)width + 1);
    for (size_t i = 0; i < pad_needed; i++) r[i] = pad[0];
    memcpy(r + pad_needed, str, slen + 1);
    return r;
}

char* iris_str_pad_right(const char* str, int64_t width, const char* pad) {
    if (!str) str = "";
    if (!pad || pad[0] == '\0') pad = " ";
    size_t slen = strlen(str);
    if ((int64_t)slen >= width) {
        char* r = (char*)xmalloc(slen + 1);
        memcpy(r, str, slen + 1);
        return r;
    }
    size_t pad_needed = (size_t)(width - (int64_t)slen);
    char* r = (char*)xmalloc((size_t)width + 1);
    memcpy(r, str, slen);
    for (size_t i = 0; i < pad_needed; i++) r[slen + i] = pad[0];
    r[(size_t)width] = '\0';
    return r;
}

IrisList* iris_str_chars(const char* str) {
    if (!str) return iris_list_new();
    IrisList* l = iris_list_new();
    size_t len = strlen(str);
    for (size_t i = 0; i < len; i++) {
        char* c = (char*)xmalloc(2);
        c[0] = str[i]; c[1] = '\0';
        IrisVal* v = (IrisVal*)xmalloc(sizeof(IrisVal));
        v->tag = IRIS_TAG_STR;
        v->str = c;
        iris_list_push(l, v);
    }
    return l;
}

IrisList* iris_str_bytes(const char* str) {
    if (!str) return iris_list_new();
    IrisList* l = iris_list_new();
    size_t len = strlen(str);
    for (size_t i = 0; i < len; i++) {
        IrisVal* v = (IrisVal*)xmalloc(sizeof(IrisVal));
        v->tag = IRIS_TAG_I64;
        v->i64 = (int64_t)(unsigned char)str[i];
        iris_list_push(l, v);
    }
    return l;
}

int64_t iris_str_count(const char* str, const char* sub) {
    if (!str || !sub || sub[0] == '\0') return 0;
    int64_t count = 0;
    size_t sublen = strlen(sub);
    const char* p = str;
    while ((p = strstr(p, sub)) != NULL) {
        count++;
        p += sublen;
    }
    return count;
}

/* -- Math constants / predicates -- */

double iris_math_pi(void) { return 3.14159265358979323846; }
double iris_math_e(void)  { return 2.71828182845904523536; }
double iris_math_inf(void) {
    volatile double zero = 0.0;
    return 1.0 / zero; /* +Infinity */
}
int    iris_is_nan(double x) { return x != x; }
int    iris_is_inf(double x) {
    volatile double zero = 0.0;
    return (x == (1.0 / zero)) || (x == -(1.0 / zero));
}

/* -- OS / System -- */

char* iris_env_get(const char* key) {
    if (!key) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }
    const char* val = getenv(key);
    if (!val) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }
    size_t len = strlen(val);
    char* r = (char*)xmalloc(len + 1);
    memcpy(r, val, len + 1);
    return r;
}

void iris_env_set(const char* key, const char* val) {
    if (!key) return;
#ifdef _WIN32
    _putenv_s(key, val ? val : "");
#else
    setenv(key, val ? val : "", 1);
#endif
}

void iris_exit_code(int64_t code) {
    exit((int)code);
}

char* iris_exec_cmd(const char* cmd) {
#if defined(__wasm__)
    (void)cmd; { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }
#else
    if (!cmd) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }
#ifdef _WIN32
    FILE* fp = _popen(cmd, "r");
#else
    FILE* fp = popen(cmd, "r");
#endif
    if (!fp) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }
    size_t cap = 1024, len = 0;
    char* buf = (char*)xmalloc(cap);
    char tmp[256];
    while (fgets(tmp, sizeof(tmp), fp)) {
        size_t tlen = strlen(tmp);
        if (len + tlen >= cap) { cap *= 2; buf = (char*)realloc(buf, cap); }
        memcpy(buf + len, tmp, tlen);
        len += tlen;
    }
    buf[len] = '\0';
#ifdef _WIN32
    _pclose(fp);
#else
    pclose(fp);
#endif
    return buf;
#endif
}

int64_t iris_pid(void) {
#ifdef _WIN32
    return (int64_t)GetCurrentProcessId();
#else
    return (int64_t)getpid();
#endif
}

/* -- Crypto / UUID -- */

static uint64_t uuid_state = 0;

char* iris_uuid(void) {
    if (uuid_state == 0) {
        uuid_state = (uint64_t)time(NULL) ^ 0xDEADBEEFCAFEBABEULL;
    }
    /* xorshift64 */
    uuid_state ^= uuid_state << 13;
    uuid_state ^= uuid_state >> 7;
    uuid_state ^= uuid_state << 17;
    uint64_t a = uuid_state;
    uuid_state ^= uuid_state << 13;
    uuid_state ^= uuid_state >> 7;
    uuid_state ^= uuid_state << 17;
    uint64_t b = uuid_state;
    a = (a & 0xFFFFFFFFFFFF0FFFULL) | 0x4000ULL; /* version 4 */
    b = (b & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL; /* variant 1 */
    char* r = (char*)xmalloc(37);
    snprintf(r, 37, "%08x-%04x-%04x-%04x-%012llx",
        (uint32_t)(a >> 32),
        (uint16_t)((a >> 16) & 0xFFFF),
        (uint16_t)(a & 0xFFFF),
        (uint16_t)((b >> 48) & 0xFFFF),
        (unsigned long long)(b & 0xFFFFFFFFFFFFULL));
    return r;
}

/* Minimal SHA-256 */
static const uint32_t sha256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

static uint32_t sha256_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

char* iris_sha256(const char* input) {
    if (!input) { char* e = (char*)xmalloc(65); memset(e, '0', 64); e[64] = '\0'; return e; }
    size_t ilen = strlen(input);
    uint64_t bit_len = (uint64_t)ilen * 8;
    /* Padding */
    size_t padded = ilen + 1;
    while (padded % 64 != 56) padded++;
    padded += 8;
    uint8_t* msg = (uint8_t*)xmalloc(padded);
    memset(msg, 0, padded);
    memcpy(msg, input, ilen);
    msg[ilen] = 0x80;
    for (int i = 0; i < 8; i++) msg[padded - 1 - i] = (uint8_t)(bit_len >> (i * 8));
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                     0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for (size_t off = 0; off < padded; off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = ((uint32_t)msg[off+i*4]<<24)|((uint32_t)msg[off+i*4+1]<<16)|((uint32_t)msg[off+i*4+2]<<8)|msg[off+i*4+3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = sha256_rotr(w[i-15],7)^sha256_rotr(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1 = sha256_rotr(w[i-2],17)^sha256_rotr(w[i-2],19)^(w[i-2]>>10);
            w[i] = w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = sha256_rotr(e,6)^sha256_rotr(e,11)^sha256_rotr(e,25);
            uint32_t ch = (e&f)^((~e)&g);
            uint32_t t1 = hh+S1+ch+sha256_K[i]+w[i];
            uint32_t S0 = sha256_rotr(a,2)^sha256_rotr(a,13)^sha256_rotr(a,22);
            uint32_t maj = (a&b)^(a&c)^(b&c);
            uint32_t t2 = S0+maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    free(msg);
    char* out = (char*)xmalloc(65);
    snprintf(out, 65, "%08x%08x%08x%08x%08x%08x%08x%08x", h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
    return out;
}

char* iris_hex_encode(const char* input) {
    if (!input) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }
    size_t len = strlen(input);
    char* r = (char*)xmalloc(len * 2 + 1);
    for (size_t i = 0; i < len; i++) snprintf(r + i*2, 3, "%02x", (unsigned char)input[i]);
    r[len * 2] = '\0';
    return r;
}

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

char* iris_hex_decode(const char* input) {
    if (!input) { char* e = (char*)xmalloc(1); e[0] = '\0'; return e; }
    size_t len = strlen(input);
    size_t olen = len / 2;
    char* r = (char*)xmalloc(olen + 1);
    for (size_t i = 0; i < olen; i++)
        r[i] = (char)((hex_digit(input[i*2]) << 4) | hex_digit(input[i*2+1]));
    r[olen] = '\0';
    return r;
}

/* -- Deque (reuses IrisList) -- */

IrisList* iris_deque_new(void) { return iris_list_new(); }

IrisList* iris_deque_push_front(IrisList* dq, int64_t val) {
    if (!dq) return dq;
    IrisVal* boxed = iris_box_i64(val);
    /* shift elements right */
    if (dq->len >= dq->cap) {
        dq->cap = dq->cap ? dq->cap * 2 : 8;
        dq->data = (IrisVal**)xrealloc(dq->data, sizeof(IrisVal*) * dq->cap);
    }
    memmove(dq->data + 1, dq->data, sizeof(IrisVal*) * dq->len);
    iris_retain(boxed);
    dq->data[0] = boxed;
    dq->len++;
    return dq;
}

IrisList* iris_deque_push_back(IrisList* dq, int64_t val) {
    iris_list_push(dq, iris_box_i64(val));
    return dq;
}

int64_t iris_deque_pop_front(IrisList* dq) {
    if (!dq || dq->len == 0) return 0;
    IrisVal* v = dq->data[0];
    memmove(dq->data, dq->data + 1, sizeof(IrisVal*) * (dq->len - 1));
    dq->len--;
    int64_t res = v ? v->i64 : 0;
    if (v) iris_release(v);
    return res;
}

int64_t iris_deque_pop_back(IrisList* dq) {
    if (!dq || dq->len == 0) return 0;
    IrisVal* v = dq->data[--dq->len];
    int64_t res = v ? v->i64 : 0;
    if (v) iris_release(v);
    return res;
}

int64_t iris_deque_len(IrisList* dq) {
    return dq ? (int64_t)dq->len : 0;
}

int64_t iris_deque_front(IrisList* dq) {
    if (!dq || dq->len == 0) return 0;
    IrisVal* v = dq->data[0];
    return v ? v->i64 : 0;
}

int64_t iris_deque_back(IrisList* dq) {
    if (!dq || dq->len == 0) return 0;
    IrisVal* v = dq->data[dq->len - 1];
    return v ? v->i64 : 0;
}

/* -- BitSet (backed by IrisList of i64 words) -- */

IrisList* iris_bitset_new(int64_t nbits) {
    (void)nbits;
    return iris_list_new();
}

IrisList* iris_bitset_set(IrisList* bs, int64_t pos) {
    if (!bs) return bs;
    int64_t word_idx = pos / 64;
    int64_t bit_idx  = pos % 64;
    while ((int64_t)bs->len <= word_idx) {
        iris_list_push(bs, iris_box_i64(0));
    }
    IrisVal* w = bs->data[word_idx];
    int64_t wv = w ? w->i64 : 0;
    if (w) { iris_release(w); }
    bs->data[word_idx] = iris_box_i64(wv | (1LL << bit_idx));
    return bs;
}

int iris_bitset_get(IrisList* bs, int64_t pos) {
    if (!bs) return 0;
    int64_t word_idx = pos / 64;
    int64_t bit_idx  = pos % 64;
    if (word_idx >= (int64_t)bs->len) return 0;
    IrisVal* w = bs->data[word_idx];
    int64_t wv = w ? w->i64 : 0;
    return (wv >> bit_idx) & 1;
}

int64_t iris_bitset_count(IrisList* bs) {
    if (!bs) return 0;
    int64_t count = 0;
    for (size_t i = 0; i < bs->len; i++) {
        IrisVal* w = bs->data[i];
        if (w) {
            uint64_t v = (uint64_t)w->i64;
            /* popcount via Kernighan trick */
            while (v) { v &= v - 1; count++; }
        }
    }
    return count;
}

IrisList* iris_bitset_clear(IrisList* bs, int64_t pos) {
    if (!bs) return bs;
    int64_t word_idx = pos / 64;
    int64_t bit_idx  = pos % 64;
    if (word_idx < (int64_t)bs->len) {
        IrisVal* w = bs->data[word_idx];
        int64_t wv = w ? w->i64 : 0;
        if (w) { iris_release(w); }
        bs->data[word_idx] = iris_box_i64(wv & ~(1LL << bit_idx));
    }
    return bs;
}

/* -- FFI -- */

void* iris_ffi_open(const char* path) {
    if (!path) return NULL;
#ifdef _WIN32
    // Automatically inject known dependency paths to PATH for LoadLibrary search
    static int paths_injected = 0;
    if (!paths_injected) {
        paths_injected = 1;
        
        // 1. Ensure AMENT_PREFIX_PATH is set
        if (!getenv("AMENT_PREFIX_PATH")) {
            _putenv_s("AMENT_PREFIX_PATH", "C:\\dev\\ros2_humble\\ros2-windows");
            SetEnvironmentVariableA("AMENT_PREFIX_PATH", "C:\\dev\\ros2_humble\\ros2-windows");
        }
        
        // 2. Add dependencies to PATH
        char old_path[32768];
        DWORD len = GetEnvironmentVariableA("PATH", old_path, sizeof(old_path));
        if (len > 0 && len < sizeof(old_path)) {
            char new_path[32768];
            snprintf(new_path, sizeof(new_path),
                "C:\\dev\\ros2_humble\\ros2-windows\\bin;"
                "C:\\onnxruntime\\lib;"
                "C:\\tensorflow\\lib;"
                "C:\\libtorch\\lib;"
                "C:\\openblas\\bin;%s",
                old_path);
            _putenv_s("PATH", new_path);
            SetEnvironmentVariableA("PATH", new_path);
        }
    }

    void* h = (void*)LoadLibraryA(path);
    if (!h) {
        fprintf(stderr, "[iris_runtime] LoadLibraryA(\"%s\") failed with error code: %lu\n", path, GetLastError());
    }
    return h;
#elif defined(__unix__) || defined(__APPLE__)
    return dlopen(path, RTLD_LAZY);
#else
    return NULL;
#endif
}

int64_t iris_ffi_call(void* handle, const char* func_name) {
    if (!handle || !func_name) return -1;
#ifdef _WIN32
    typedef int64_t (*fn_t)(void);
    fn_t f = (fn_t)GetProcAddress((HMODULE)handle, func_name);
    if (!f) return -1;
    return f();
#elif defined(__unix__) || defined(__APPLE__)
    typedef int64_t (*fn_t)(void);
    fn_t f = (fn_t)dlsym(handle, func_name);
    if (!f) return -1;
    return f();
#else
    return -1;
#endif
}

int iris_ffi_close(void* handle) {
    if (!handle) return 0;
#ifdef _WIN32
    return FreeLibrary((HMODULE)handle) ? 1 : 0;
#elif defined(__unix__) || defined(__APPLE__)
    return dlclose(handle) == 0 ? 1 : 0;
#else
    return 0;
#endif
}

/* -- Expanded C FFI with typed arguments -- */

static void* ffi_get_sym(void* handle, const char* func_name) {
    if (!handle || !func_name) return NULL;
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)handle, func_name);
#elif defined(__unix__) || defined(__APPLE__)
    return dlsym(handle, func_name);
#else
    return NULL;
#endif
}

static int64_t ffi_dispatch_i64(void* fn_ptr, int64_t* args, int n) {
    typedef int64_t (*fn0)(void);
    typedef int64_t (*fn1)(int64_t);
    typedef int64_t (*fn2)(int64_t, int64_t);
    typedef int64_t (*fn3)(int64_t, int64_t, int64_t);
    typedef int64_t (*fn4)(int64_t, int64_t, int64_t, int64_t);
    typedef int64_t (*fn5)(int64_t, int64_t, int64_t, int64_t, int64_t);
    typedef int64_t (*fn6)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
    if (!fn_ptr) return -1;
    switch (n) {
        case 0: return ((fn0)fn_ptr)();
        case 1: return ((fn1)fn_ptr)(args[0]);
        case 2: return ((fn2)fn_ptr)(args[0], args[1]);
        case 3: return ((fn3)fn_ptr)(args[0], args[1], args[2]);
        case 4: return ((fn4)fn_ptr)(args[0], args[1], args[2], args[3]);
        case 5: return ((fn5)fn_ptr)(args[0], args[1], args[2], args[3], args[4]);
        default: return ((fn6)fn_ptr)(args[0], args[1], args[2], args[3], args[4], args[5]);
    }
}

int64_t iris_ffi_call_i64(void* handle, const char* func_name, int64_t* args, int nargs) {
    void* sym = ffi_get_sym(handle, func_name);
    return ffi_dispatch_i64(sym, args, nargs);
}

double iris_ffi_call_f64(void* handle, const char* func_name, int64_t* args, int nargs) {
    typedef double (*fn0)(void);
    typedef double (*fn1)(int64_t);
    typedef double (*fn2)(int64_t, int64_t);
    void* sym = ffi_get_sym(handle, func_name);
    if (!sym) return 0.0;
    switch (nargs) {
        case 0: return ((fn0)sym)();
        case 1: return ((fn1)sym)(args[0]);
        default: return ((fn2)sym)(args[0], args[1]);
    }
}

const char* iris_ffi_call_str(void* handle, const char* func_name, int64_t* args, int nargs) {
    typedef const char* (*fn0)(void);
    typedef const char* (*fn1)(int64_t);
    typedef const char* (*fn2)(int64_t, int64_t);
    void* sym = ffi_get_sym(handle, func_name);
    if (!sym) return "";
    switch (nargs) {
        case 0: return ((fn0)sym)();
        case 1: return ((fn1)sym)(args[0]);
        default: return ((fn2)sym)(args[0], args[1]);
    }
}

void iris_ffi_call_void(void* handle, const char* func_name, int64_t* args, int nargs) {
    typedef void (*fn0)(void);
    typedef void (*fn1)(int64_t);
    typedef void (*fn2)(int64_t, int64_t);
    void* sym = ffi_get_sym(handle, func_name);
    if (!sym) return;
    switch (nargs) {
        case 0: ((fn0)sym)(); break;
        case 1: ((fn1)sym)(args[0]); break;
        default: ((fn2)sym)(args[0], args[1]); break;
    }
}

/* ---- FFI out-parameter cells -------------------------------------------
 *
 * A great many C functions return their result through a pointer argument:
 *
 *     int64_t iris_rcl_take_twist(int64_t sub, double* out);   // writes 6 doubles
 *     int64_t iris_rcl_take_string(int64_t sub, char* buf, int32_t max);
 *
 * IRIS could not call any of those. `ffi_dispatch_i64` already passes an array
 * of int64 slots, and on every supported target a pointer fits in one -- so the
 * missing piece was never the calling convention. It was that IRIS had no way to
 * own a piece of memory and name its address.
 *
 * These give it one. `iris_ffi_out_new` returns the address of a zeroed block as
 * an int64, which is passed straight through as an ordinary argument; the
 * typed readers then pull values back out. Indexed reads matter because a single
 * out-pointer often receives several values (a Twist is six doubles).
 *
 * Ownership is explicit: the caller frees. That is deliberate -- these cells are
 * handed to foreign code, so tying them to the refcounting GC would mean
 * reasoning about a lifetime the GC cannot see.
 */

/* Every cell carries a 16-byte header immediately before the address handed
 * out: a magic word and the payload length.
 *
 * That buys two properties the interpreter already had, and the two backends
 * must agree or a policy behaves differently under `--emit eval` than in a
 * built binary. Bounds are checked on every read, so an out-param the callee
 * did not fill returns zero rather than garbage; and `out_free` is idempotent,
 * because the magic is cleared on release and a second free sees it gone.
 *
 * Idempotence is not a nicety here: the error path of a foreign call typically
 * frees on the way out, and a double free was corrupting the heap
 * (exit 0xC0000409) where the interpreter simply ignored it.
 *
 * 16 bytes keeps the payload 16-byte aligned, which is enough for any scalar a
 * C ABI will write through an out-pointer. */
#define IRIS_CELL_MAGIC 0x4952495343454C4CULL
#define IRIS_CELL_HDR   16

int64_t iris_ffi_out_new(int64_t nbytes) {
    if (nbytes <= 0) return 0;
    unsigned char* base = (unsigned char*)calloc(1, (size_t)nbytes + IRIS_CELL_HDR);
    if (!base) return 0;
    *(uint64_t*)base = IRIS_CELL_MAGIC;
    *(uint64_t*)(base + 8) = (uint64_t)nbytes;
    return (int64_t)(intptr_t)(base + IRIS_CELL_HDR);
}

/* Returns the payload length, or 0 if `cell` is not a live cell. */
static uint64_t iris_cell_len(int64_t cell) {
    if (!cell) return 0;
    unsigned char* base = (unsigned char*)(intptr_t)cell - IRIS_CELL_HDR;
    if (*(uint64_t*)base != IRIS_CELL_MAGIC) return 0;
    return *(uint64_t*)(base + 8);
}

void iris_ffi_out_free(int64_t cell) {
    if (!cell) return;
    unsigned char* base = (unsigned char*)(intptr_t)cell - IRIS_CELL_HDR;
    if (*(uint64_t*)base != IRIS_CELL_MAGIC) return;   /* already freed */
    *(uint64_t*)base = 0;
    free(base);
}

int64_t iris_ffi_out_sizeof_f64(void) { return (int64_t)sizeof(double); }
int64_t iris_ffi_out_sizeof_i64(void) { return (int64_t)sizeof(int64_t); }

double iris_ffi_out_get_f64(int64_t cell, int64_t index) {
    uint64_t len = iris_cell_len(cell);
    if (index < 0 || (uint64_t)(index + 1) * sizeof(double) > len) return 0.0;
    return ((double*)(intptr_t)cell)[index];
}

int64_t iris_ffi_out_get_i64(int64_t cell, int64_t index) {
    uint64_t len = iris_cell_len(cell);
    if (index < 0 || (uint64_t)(index + 1) * sizeof(int64_t) > len) return 0;
    return ((int64_t*)(intptr_t)cell)[index];
}

int32_t iris_ffi_out_get_i32(int64_t cell, int64_t index) {
    uint64_t len = iris_cell_len(cell);
    if (index < 0 || (uint64_t)(index + 1) * sizeof(int32_t) > len) return 0;
    return ((int32_t*)(intptr_t)cell)[index];
}

/* Read the cell as a NUL-terminated string, bounded by the payload length so an
 * unterminated buffer cannot run off the end. Returns a fresh copy, so the cell
 * may be freed immediately afterwards. */
char* iris_ffi_out_get_str(int64_t cell) {
    uint64_t len = iris_cell_len(cell);
    if (!len) return xstrdup("");
    const char* p = (const char*)(intptr_t)cell;
    uint64_t n = 0;
    while (n < len && p[n] != 0) n++;
    char* out = (char*)xmalloc((size_t)n + 1);
    memcpy(out, p, (size_t)n);
    out[n] = 0;
    return out;
}

void iris_ffi_out_set_f64(int64_t cell, int64_t index, double v) {
    uint64_t len = iris_cell_len(cell);
    if (index < 0 || (uint64_t)(index + 1) * sizeof(double) > len) return;
    ((double*)(intptr_t)cell)[index] = v;
}

void iris_ffi_out_set_i64(int64_t cell, int64_t index, int64_t v) {
    uint64_t len = iris_cell_len(cell);
    if (index < 0 || (uint64_t)(index + 1) * sizeof(int64_t) > len) return;
    ((int64_t*)(intptr_t)cell)[index] = v;
}

/* -- Python FFI -- */

static char python_buf[65536];

static const char* find_python_cmd(void) {
#ifdef _WIN32
    /* On Windows, try python first (py launcher), then python3. */
    if (system("python --version >nul 2>&1") == 0) return "python";
    if (system("python3 --version >nul 2>&1") == 0) return "python3";
#else
    if (system("python3 --version >/dev/null 2>&1") == 0) return "python3";
    if (system("python --version >/dev/null 2>&1") == 0) return "python";
#endif
    return NULL;
}

const char* iris_python_eval(const char* code) {
#if defined(__wasm__)
    (void)code; snprintf(python_buf, sizeof(python_buf), "error: python not available on WASM"); return python_buf;
#else
    const char* py = find_python_cmd();
    if (!py || !code) { snprintf(python_buf, sizeof(python_buf), "error: python not found"); return python_buf; }
    char cmd[8192];
    snprintf(cmd, sizeof(cmd), "%s -c \"import sys; sys.stdout.write(str(%s))\"", py, code);
#ifdef _WIN32
    FILE* fp = _popen(cmd, "r");
#else
    FILE* fp = popen(cmd, "r");
#endif
    if (!fp) { snprintf(python_buf, sizeof(python_buf), "error: popen failed"); return python_buf; }
    size_t n = fread(python_buf, 1, sizeof(python_buf) - 1, fp);
    python_buf[n] = '\0';
#ifdef _WIN32
    _pclose(fp);
#else
    pclose(fp);
#endif
    return python_buf;
#endif
}

int64_t iris_python_exec(const char* code_or_path) {
    const char* py = find_python_cmd();
    if (!py || !code_or_path) return -1;
    char cmd[8192];
    /* Check if it looks like a file path */
    FILE* test = fopen(code_or_path, "r");
    if (test) {
        fclose(test);
        snprintf(cmd, sizeof(cmd), "%s \"%s\"", py, code_or_path);
    } else {
        snprintf(cmd, sizeof(cmd), "%s -c \"%s\"", py, code_or_path);
    }
    return (int64_t)system(cmd);
}

const char* iris_python_call(const char* module, const char* func, const char* args_json) {
#if defined(__wasm__)
    (void)module;(void)func;(void)args_json; snprintf(python_buf, sizeof(python_buf), "error: python not available on WASM"); return python_buf;
#else
    const char* py = find_python_cmd();
    if (!py || !module || !func) { snprintf(python_buf, sizeof(python_buf), "error: python not found"); return python_buf; }
    char cmd[8192];
    const char* a = args_json ? args_json : "";
    snprintf(cmd, sizeof(cmd),
        /* The argument is quoted. Without the quotes it was interpolated as
         * a bare Python expression, so `py_call1("os.path", "basename",
         * "/a/b/c.txt")` generated `basename(/a/b/c.txt)` -- a SyntaxError.
         * The interpreter quotes it, so the two backends disagreed on every
         * non-numeric argument. See known-issues #44. */
        "%s -c \"import %s; print(%s.%s('%s'))\"",
        py, module, module, func, a);
#ifdef _WIN32
    FILE* fp = _popen(cmd, "r");
#else
    FILE* fp = popen(cmd, "r");
#endif
    if (!fp) { snprintf(python_buf, sizeof(python_buf), "error: popen failed"); return python_buf; }
    size_t n = fread(python_buf, 1, sizeof(python_buf) - 1, fp);
    python_buf[n] = '\0';
    /* Trim trailing newline */
    while (n > 0 && (python_buf[n-1] == '\n' || python_buf[n-1] == '\r')) { python_buf[--n] = '\0'; }
#ifdef _WIN32
    _pclose(fp);
#else
    pclose(fp);
#endif
    return python_buf;
#endif
}

const char* iris_python_version(void) {
#if defined(__wasm__)
    return "Python not available on WASM";
#else
    const char* py = find_python_cmd();
    if (!py) return "Python not found";
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "%s --version", py);
#ifdef _WIN32
    FILE* fp = _popen(cmd, "r");
#else
    FILE* fp = popen(cmd, "r");
#endif
    if (!fp) return "unknown";
    size_t n = fread(python_buf, 1, sizeof(python_buf) - 1, fp);
    python_buf[n] = '\0';
    while (n > 0 && (python_buf[n-1] == '\n' || python_buf[n-1] == '\r')) { python_buf[--n] = '\0'; }
#ifdef _WIN32
    _pclose(fp);
#else
    pclose(fp);
#endif
    return python_buf;
#endif
}

/* -- Rust FFI (aliases for C FFI — Rust cdylibs export extern "C") -- */

void* iris_rust_lib_open(const char* path) { return iris_ffi_open(path); }
int64_t iris_rust_call_i64(void* h, const char* fn_name, int64_t* args, int n) { return iris_ffi_call_i64(h, fn_name, args, n); }
double  iris_rust_call_f64(void* h, const char* fn_name, int64_t* args, int n) { return iris_ffi_call_f64(h, fn_name, args, n); }
void    iris_rust_call_void(void* h, const char* fn_name, int64_t* args, int n) { iris_ffi_call_void(h, fn_name, args, n); }

/* -- Functional list ops (numeric) -- */

double iris_list_sum(IrisList* list) {
    if (!list) return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < list->len; i++) {
        if (!list->data[i]) continue;
        switch (list->data[i]->tag) {
            case IRIS_TAG_I64:
                s += (double)list->data[i]->i64;
                break;
            case IRIS_TAG_I32:
                s += (double)list->data[i]->i32;
                break;
            case IRIS_TAG_F64:
                s += list->data[i]->f64;
                break;
            case IRIS_TAG_F32:
                s += (double)list->data[i]->f32;
                break;
            default:
                break;
        }
    }
    return s;
}

int64_t iris_list_min(IrisList* list) {
    if (!list || list->len == 0) return 0;
    int64_t m = INT64_MAX;
    for (size_t i = 0; i < list->len; i++) {
        if (list->data[i] && list->data[i]->tag == IRIS_TAG_I64 && list->data[i]->i64 < m)
            m = list->data[i]->i64;
    }
    return m;
}

int64_t iris_list_max(IrisList* list) {
    if (!list || list->len == 0) return 0;
    int64_t m = INT64_MIN;
    for (size_t i = 0; i < list->len; i++) {
        if (list->data[i] && list->data[i]->tag == IRIS_TAG_I64 && list->data[i]->i64 > m)
            m = list->data[i]->i64;
    }
    return m;
}

int64_t iris_list_index_of(IrisList* list, int64_t val) {
    if (!list) return -1;
    for (size_t i = 0; i < list->len; i++) {
        if (list->data[i] && list->data[i]->tag == IRIS_TAG_I64 && list->data[i]->i64 == val)
            return (int64_t)i;
    }
    return -1;
}

int64_t iris_list_count(IrisList* list, int64_t val) {
    if (!list) return 0;
    int64_t c = 0;
    for (size_t i = 0; i < list->len; i++) {
        if (list->data[i] && list->data[i]->tag == IRIS_TAG_I64 && list->data[i]->i64 == val)
            c++;
    }
    return c;
}

IrisList* iris_list_reverse(IrisList* list) {
    IrisList* r = iris_list_new();
    if (!list) return r;
    for (size_t i = list->len; i > 0; i--)
        iris_list_push(r, list->data[i-1]);
    return r;
}

IrisList* iris_list_take(IrisList* list, int64_t n) {
    IrisList* r = iris_list_new();
    if (!list) return r;
    size_t take = (n < 0) ? 0 : ((size_t)n > list->len ? list->len : (size_t)n);
    for (size_t i = 0; i < take; i++)
        iris_list_push(r, list->data[i]);
    return r;
}

IrisList* iris_list_drop(IrisList* list, int64_t n) {
    IrisList* r = iris_list_new();
    if (!list) return r;
    size_t start = (n < 0) ? 0 : ((size_t)n > list->len ? list->len : (size_t)n);
    for (size_t i = start; i < list->len; i++)
        iris_list_push(r, list->data[i]);
    return r;
}

/* -- Concurrency extras -- */

int64_t iris_thread_count(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int64_t)si.dwNumberOfProcessors;
#elif defined(__unix__) || defined(__APPLE__)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int64_t)n : 1;
#else
    return 1;
#endif
}

/* ====================================================================
 * Reference-Counting Garbage Collector
 * ====================================================================
 * Side-table approach: a hash map from pointer → refcount. This avoids
 * modifying the IrisVal layout while providing real reference counting.
 */

#define RC_TABLE_BUCKETS 4096

/* Bacon-Rajan cycle GC color codes. */
#define RC_COLOR_BLACK  0   /* live, not a candidate */
#define RC_COLOR_GRAY   1   /* possible cycle root */
#define RC_COLOR_WHITE  2   /* unreachable, collect me */

typedef struct RcEntry {
    void*               ptr;
    int32_t             kind;
    int64_t             count;
    int64_t             scan_count; /* scratch counter used by cycle collector */
    int                 color;
    int                 buffered;   /* 1 if in possible_roots */
    IrisWeakRef*        weak_refs;  /* list of active weak refs pointing to this ptr */
    struct RcEntry*     next;
} RcEntry;

static RcEntry* rc_table[RC_TABLE_BUCKETS];
static int64_t  gc_total_allocated = 0;
static int64_t  gc_total_freed = 0;
static int64_t  gc_cycles_collected = 0;
static int64_t  gc_weak_refs_invalidated = 0;
static int64_t  gc_auto_threshold = 10000;

/* Global mutex protecting the rc_table from concurrent retain/release calls. */
static pthread_mutex_t rc_global_mu = PTHREAD_MUTEX_INITIALIZER;

/* Possible cycle roots (Gray nodes). */
#define POSSIBLE_ROOTS_MAX 512
static void* possible_roots[POSSIBLE_ROOTS_MAX];
static int   possible_roots_count = 0;

static void iris_gc_cycle_collect_locked(void);

static void invalidate_weak_refs_locked(RcEntry* e) {
    if (!e) return;
    IrisWeakRef* w = e->weak_refs;
    while (w) {
        w->target = NULL;
        gc_weak_refs_invalidated++;
        w = w->next_weak;
    }
    e->weak_refs = NULL;
}

static size_t rc_hash(void* ptr) {
    uintptr_t v = (uintptr_t)ptr;
    v = (v >> 4) ^ (v >> 16);
    return (size_t)(v % RC_TABLE_BUCKETS);
}

/* Must be called with rc_global_mu held. */
static RcEntry* rc_find(void* ptr) {
    size_t h = rc_hash(ptr);
    RcEntry* e = rc_table[h];
    while (e) {
        if (e->ptr == ptr) return e;
        e = e->next;
    }
    return NULL;
}

/* Must be called with rc_global_mu held. */
static RcEntry* rc_insert(void* ptr, int32_t kind) {
    size_t h = rc_hash(ptr);
    RcEntry* e = xmalloc(sizeof(RcEntry));
    e->ptr = ptr;
    e->kind = kind;
    e->count = 1;
    e->scan_count = 0;
    e->color = RC_COLOR_BLACK;
    e->buffered = 0;
    e->weak_refs = NULL;
    e->next = rc_table[h];
    rc_table[h] = e;
    gc_total_allocated++;
    if (gc_auto_threshold > 0 && (gc_total_allocated % gc_auto_threshold == 0)) {
        iris_gc_cycle_collect_locked();
    }
    return e;
}

/* Must be called with rc_global_mu held. Removes and returns one live entry
 * from the RC table so cleanup can deep-free it outside the global lock. */
static RcEntry* rc_take_one_locked(void) {
    for (size_t h = 0; h < RC_TABLE_BUCKETS; h++) {
        RcEntry* e = rc_table[h];
        if (e) {
            rc_table[h] = e->next;
            e->next = NULL;
            return e;
        }
    }
    return NULL;
}

static void rc_free_list_payload(IrisList* list) {
    if (!list) return;
    for (size_t i = 0; i < list->len; i++) {
        if (list->data[i]) iris_release(list->data[i]);
    }
    free(list->data);
    free(list);
}

static void rc_free_map_payload(IrisMap* m) {
    if (!m) return;
    for (size_t i = 0; i < m->n_buckets; i++) {
        IrisMapEntry* e = m->buckets[i];
        while (e) {
            IrisMapEntry* next = e->next;
            free(e->key);
            if (e->val) iris_release(e->val);
            free(e);
            e = next;
        }
    }
    free(m->buckets);
    free(m);
}

static void rc_free_channel_payload(IrisChannel* c) {
    if (!c) return;
    for (size_t i = 0; i < c->count; i++) {
        size_t idx = (c->head + i) % c->cap;
        if (c->buf[idx]) iris_release(c->buf[idx]);
    }
    free(c->buf);
    pthread_mutex_destroy(&c->mu);
    pthread_cond_destroy(&c->not_empty);
    pthread_cond_destroy(&c->not_full);
    free(c);
}

static void rc_free_atomic_payload(IrisAtomic* a) {
    if (!a) return;
    if (a->val) iris_release(a->val);
    pthread_mutex_destroy(&a->mu);
    free(a);
}

static void rc_free_mutex_payload(IrisMutex* m) {
    if (!m) return;
    pthread_mutex_destroy(&m->mu);
    free(m);
}

static void rc_free_sparse_payload(IrisSparse* sp) {
    if (!sp) return;
    for (size_t i = 0; i < sp->len; i++) {
        if (sp->values[i]) iris_release(sp->values[i]);
    }
    free(sp->indices);
    free(sp->values);
    free(sp);
}

static void rc_deep_free_by_kind(void* ptr, int32_t kind) {
    if (!ptr) return;
    switch (kind) {
        case IRIS_RC_BOXED: {
            IrisVal* val = (IrisVal*)ptr;
            switch (val->tag) {
                case IRIS_TAG_STR:
                    /* Release the underlying string via the RC table so ownership
                     * and refcounts are consistent. If no RC entry exists the
                     * call is a no-op; otherwise the string will be freed when
                     * its count reaches zero. */
                    if (val->str) iris_release_kind(val->str, IRIS_RC_STR);
                    break;
                case IRIS_TAG_LIST:
                    iris_release_kind(val->ptr, IRIS_RC_LIST);
                    break;
                case IRIS_TAG_MAP:
                    iris_release_kind(val->ptr, IRIS_RC_MAP);
                    break;
                case IRIS_TAG_OPTION:
                    iris_release_kind(val->ptr, IRIS_RC_OPTION);
                    break;
                case IRIS_TAG_RESULT:
                    iris_release_kind(val->ptr, IRIS_RC_RESULT);
                    break;
                case IRIS_TAG_CHAN:
                    iris_release_kind(val->ptr, IRIS_RC_CHAN);
                    break;
                case IRIS_TAG_ATOMIC:
                    iris_release_kind(val->ptr, IRIS_RC_ATOMIC);
                    break;
                case IRIS_TAG_GRAD:
                    iris_release_kind(val->ptr, IRIS_RC_GRAD);
                    break;
                case IRIS_TAG_SPARSE:
                    iris_release_kind(val->ptr, IRIS_RC_SPARSE);
                    break;
                case IRIS_TAG_MUTEX:
                    iris_release_kind(val->ptr, IRIS_RC_MUTEX);
                    break;
                case IRIS_TAG_TUPLE:
                case IRIS_TAG_STRUCT:
                    rc_free_list_payload((IrisList*)val->ptr);
                    break;
                case IRIS_TAG_CLOSURE: {
                    IrisClosure* c = (IrisClosure*)val->ptr;
                    if (c) {
                        rc_free_list_payload(c->captures);
                        free(c);
                    }
                    break;
                }
                default:
                    break;
            }
            free(val);
            break;
        }
        case IRIS_RC_STR:
            if (ds_contains_and_remove(ptr)) {
                free((char*)ptr);
            }
            break;
        case IRIS_RC_LIST:
            rc_free_list_payload((IrisList*)ptr);
            break;
        case IRIS_RC_MAP:
            rc_free_map_payload((IrisMap*)ptr);
            break;
        case IRIS_RC_OPTION: {
            IrisOption* opt = (IrisOption*)ptr;
            if (opt) {
                if (opt->has_value && opt->value) iris_release(opt->value);
                free(opt);
            }
            break;
        }
        case IRIS_RC_RESULT: {
            IrisResult* res = (IrisResult*)ptr;
            if (res) {
                if (res->value) iris_release(res->value);
                free(res);
            }
            break;
        }
        case IRIS_RC_CHAN:
            rc_free_channel_payload((IrisChannel*)ptr);
            break;
        case IRIS_RC_ATOMIC:
            rc_free_atomic_payload((IrisAtomic*)ptr);
            break;
        case IRIS_RC_MUTEX:
            rc_free_mutex_payload((IrisMutex*)ptr);
            break;
        case IRIS_RC_GRAD:
            free((IrisGrad*)ptr);
            break;
        case IRIS_RC_SPARSE:
            rc_free_sparse_payload((IrisSparse*)ptr);
            break;
        default:
            free(ptr);
            break;
    }
}

void iris_retain_kind(void* ptr, int32_t kind) {
    if (!ptr) return;
    pthread_mutex_lock(&rc_global_mu);
    RcEntry* e = rc_find(ptr);
    if (e) {
        e->count++;
        e->color = RC_COLOR_BLACK;
    } else {
        rc_insert(ptr, kind);
    }
    pthread_mutex_unlock(&rc_global_mu);
}

void iris_retain(void* ptr) {
    iris_retain_kind(ptr, IRIS_RC_BOXED);
}

void iris_release_kind(void* ptr, int32_t kind) {
    if (!ptr) return;
    pthread_mutex_lock(&rc_global_mu);
    RcEntry* e = rc_find(ptr);
    if (!e) {
        pthread_mutex_unlock(&rc_global_mu);
        return;
    }
    e->count--;
    if (e->count <= 0) {
        int32_t free_kind = e->kind;
        e->color = RC_COLOR_BLACK;
        pthread_mutex_unlock(&rc_global_mu);
        rc_deep_free_by_kind(ptr, free_kind);
        pthread_mutex_lock(&rc_global_mu);
        gc_total_freed++;
        size_t h = rc_hash(ptr);
        RcEntry** pp = &rc_table[h];
        while (*pp) {
            if ((*pp)->ptr == ptr) {
                RcEntry* tmp = *pp;
                invalidate_weak_refs_locked(tmp);
                *pp = tmp->next;
                free(tmp);
                break;
            }
            pp = &((*pp)->next);
        }
    } else {
        if (e->color != RC_COLOR_GRAY && !e->buffered) {
            e->color = RC_COLOR_GRAY;
            e->buffered = 1;
            if (possible_roots_count < POSSIBLE_ROOTS_MAX) {
                possible_roots[possible_roots_count++] = ptr;
            }
        }
        if (kind != IRIS_RC_BOXED) {
            e->kind = kind;
        }
    }
    pthread_mutex_unlock(&rc_global_mu);
}

void iris_release(void* ptr) {
    iris_release_kind(ptr, IRIS_RC_BOXED);
}

int64_t iris_refcount(void* ptr) {
    if (!ptr) return 0;
    pthread_mutex_lock(&rc_global_mu);
    RcEntry* e = rc_find(ptr);
    int64_t c = e ? e->count : 0;
    pthread_mutex_unlock(&rc_global_mu);
    return c;
}

/* ── Bacon-Rajan cycle collector ─────────────────────────────────────────── */

/* Call fn(child_ptr) for every RC-tracked child owned by a node. */
typedef void (*ChildFn)(void*, void*);
static void rc_each_child(void* ptr, int32_t kind, ChildFn fn, void* ctx) {
    if (!ptr) return;
    switch (kind) {
        case IRIS_RC_BOXED: {
            IrisVal* val = (IrisVal*)ptr;
            switch (val->tag) {
                case IRIS_TAG_LIST:
                case IRIS_TAG_MAP:
                case IRIS_TAG_OPTION:
                case IRIS_TAG_RESULT:
                case IRIS_TAG_CHAN:
                case IRIS_TAG_ATOMIC:
                case IRIS_TAG_GRAD:
                case IRIS_TAG_SPARSE:
                case IRIS_TAG_MUTEX:
                    if (val->ptr) fn(val->ptr, ctx);
                    break;
                case IRIS_TAG_TUPLE:
                case IRIS_TAG_STRUCT: {
                    IrisList* list = (IrisList*)val->ptr;
                    if (list) for (size_t i = 0; i < list->len; i++) fn(list->data[i], ctx);
                    break;
                }
                case IRIS_TAG_CLOSURE: {
                    IrisClosure* c = (IrisClosure*)val->ptr;
                    if (c && c->captures) {
                        for (size_t i = 0; i < c->captures->len; i++) fn(c->captures->data[i], ctx);
                    }
                    break;
                }
                default:
                    break;
            }
            break;
        }
        case IRIS_RC_LIST: {
            IrisList* list = (IrisList*)ptr;
            if (list) for (size_t i = 0; i < list->len; i++) fn(list->data[i], ctx);
            break;
        }
        case IRIS_RC_MAP: {
            IrisMap* m = (IrisMap*)ptr;
            if (m) for (size_t i = 0; i < m->n_buckets; i++)
                for (IrisMapEntry* e = m->buckets[i]; e; e = e->next) fn(e->val, ctx);
            break;
        }
        case IRIS_RC_OPTION: {
            IrisOption* opt = (IrisOption*)ptr;
            if (opt && opt->has_value) fn(opt->value, ctx);
            break;
        }
        case IRIS_RC_RESULT: {
            IrisResult* res = (IrisResult*)ptr;
            if (res && res->value) fn(res->value, ctx);
            break;
        }
        case IRIS_RC_CHAN: {
            IrisChannel* c = (IrisChannel*)ptr;
            if (c) {
                for (size_t i = 0; i < c->count; i++) {
                    size_t idx = (c->head + i) % c->cap;
                    fn(c->buf[idx], ctx);
                }
            }
            break;
        }
        case IRIS_RC_ATOMIC: {
            IrisAtomic* a = (IrisAtomic*)ptr;
            if (a && a->val) fn(a->val, ctx);
            break;
        }
        case IRIS_RC_SPARSE: {
            IrisSparse* sp = (IrisSparse*)ptr;
            if (sp) for (size_t i = 0; i < sp->len; i++) fn(sp->values[i], ctx);
            break;
        }
        default:
            break;
    }
}

/* Phase 2 helper: decrement scan_count of child. */
static void rc_mark_gray_child(void* child, void* ctx) {
    (void)ctx;
    if (!child) return;
    RcEntry* e = rc_find(child);
    if (e) e->scan_count--;
}

/* Phase 3 helper: restore children to BLACK (external ref exists). */
static void rc_scan_black_child(void* child, void* ctx);
static void rc_scan_black_node(RcEntry* e) {
    if (!e || e->color == RC_COLOR_BLACK) return;
    e->color = RC_COLOR_BLACK;
    rc_each_child(e->ptr, e->kind, rc_scan_black_child, NULL);
}
static void rc_scan_black_child(void* child, void* ctx) {
    (void)ctx;
    if (!child) return;
    RcEntry* e = rc_find(child);
    if (e) {
        e->scan_count++;
        rc_scan_black_node(e);
    }
}

/* Cycle collection using the Bacon-Rajan trial-deletion algorithm.
 * Must be called with rc_global_mu held. */
static void iris_gc_cycle_collect_locked(void) {
    if (possible_roots_count == 0) return;

    /* Phase 1: Init scan_count from live count for all gray roots. */
    for (int i = 0; i < possible_roots_count; i++) {
        RcEntry* e = rc_find(possible_roots[i]);
        if (e && e->color == RC_COLOR_GRAY) e->scan_count = e->count;
    }

    /* Phase 2: For each gray root, decrement scan_count of each child pointer.
     * This simulates removing the internal (intra-cycle) references. */
    for (int i = 0; i < possible_roots_count; i++) {
        RcEntry* e = rc_find(possible_roots[i]);
        if (e && e->color == RC_COLOR_GRAY)
            rc_each_child(e->ptr, e->kind, rc_mark_gray_child, NULL);
    }

    /* Phase 3: Nodes with scan_count > 0 have external references — mark BLACK.
     * Nodes with scan_count == 0 are unreachable from outside — mark WHITE. */
    for (int i = 0; i < possible_roots_count; i++) {
        RcEntry* e = rc_find(possible_roots[i]);
        if (!e) continue;
        e->buffered = 0;
        if (e->color == RC_COLOR_GRAY) {
            if (e->scan_count > 0) {
                rc_scan_black_node(e); /* restore external refs transitively */
            } else {
                e->color = RC_COLOR_WHITE;
            }
        }
    }
    possible_roots_count = 0;

    /* Phase 4: Free all White (cyclic garbage) nodes. */
    for (size_t h = 0; h < RC_TABLE_BUCKETS; h++) {
        RcEntry** pp = &rc_table[h];
        while (*pp) {
            if ((*pp)->color == RC_COLOR_WHITE) {
                RcEntry* tmp = *pp;
                *pp = tmp->next;
                /* Deep-free outside the lock to avoid recursive acquire. */
                pthread_mutex_unlock(&rc_global_mu);
                rc_deep_free_by_kind(tmp->ptr, tmp->kind);
                pthread_mutex_lock(&rc_global_mu);
                gc_total_freed++;
                free(tmp);
            } else {
                pp = &((*pp)->next);
            }
        }
    }
}

void iris_gc_collect(void) {
    pthread_mutex_lock(&rc_global_mu);
    /* First run the cycle collector to break cycles, then sweep zero-count entries. */
    iris_gc_cycle_collect_locked();
    for (size_t h = 0; h < RC_TABLE_BUCKETS; h++) {
        RcEntry** pp = &rc_table[h];
        while (*pp) {
            if ((*pp)->count <= 0) {
                RcEntry* tmp = *pp;
                *pp = tmp->next;
                pthread_mutex_unlock(&rc_global_mu);
                rc_deep_free_by_kind(tmp->ptr, tmp->kind);
                pthread_mutex_lock(&rc_global_mu);
                gc_total_freed++;
                free(tmp);
            } else {
                pp = &((*pp)->next);
            }
        }
    }
    pthread_mutex_unlock(&rc_global_mu);
}

int64_t iris_gc_stats_allocated(void) {
    return gc_total_allocated;
}

int64_t iris_gc_stats_freed(void) {
    return gc_total_freed;
}

/* iris_runtime_cleanup — frees all GC-tracked live objects at program exit.
 * Also frees the RC side-table itself.  Registered via atexit() in the
 * constructor below so that sanitizers (ASAN/Valgrind) report a clean heap. */
static void iris_runtime_cleanup(void) {
    /* The main thread has no TLS destructor to run, so free its tape here. */
    iris_tape_thread_free();

    /* Run cycle collector one final time before cleanup. */
    pthread_mutex_lock(&rc_global_mu);
    iris_gc_cycle_collect_locked();
    possible_roots_count = 0;
    RcEntry* e = rc_take_one_locked();
    pthread_mutex_unlock(&rc_global_mu);
    while (e) {
        rc_deep_free_by_kind(e->ptr, e->kind);
        gc_total_freed++;
        free(e);
        pthread_mutex_lock(&rc_global_mu);
        e = rc_take_one_locked();
        pthread_mutex_unlock(&rc_global_mu);
    }
}

#ifdef _MSC_VER
/* MSVC: use a pragma section-based init instead of __attribute__((constructor)). */
static int _iris_runtime_init(void) { atexit(iris_runtime_cleanup); return 0; }
#pragma section(".CRT$XCU", read)
__declspec(allocate(".CRT$XCU")) static int (*_iris_init_ptr)(void) = _iris_runtime_init;
#else
__attribute__((constructor))
static void iris_runtime_init(void) {
    atexit(iris_runtime_cleanup);
}
#endif

/* ====================================================================
 * Sandbox / Security Policy (C runtime side)
 * ====================================================================
 * Simple global flags. The Rust-side SecurityPolicy is the authoritative
 * source; this C-side mirror is for native-compiled IRIS programs.
 */

static int sandbox_allow_fs = 1;
static int sandbox_allow_net = 1;
static int sandbox_allow_ffi = 1;

void iris_sandbox_set_policy(int allow_fs, int allow_net, int allow_ffi) {
    sandbox_allow_fs  = allow_fs;
    sandbox_allow_net = allow_net;
    sandbox_allow_ffi = allow_ffi;
}

int iris_sandbox_check_fs_read(const char* path) {
    (void)path;
    return sandbox_allow_fs ? 0 : 1;
}

int iris_sandbox_check_fs_write(const char* path) {
    (void)path;
    return sandbox_allow_fs ? 0 : 1;
}

int iris_sandbox_check_network(const char* host) {
    (void)host;
    return sandbox_allow_net ? 0 : 1;
}

/* ---------------------------------------------------------------------------
 * Enum Variant Helpers
 * --------------------------------------------------------------------------- */

IrisVal* iris_make_variant(int64_t tag, int32_t nfields, ...) {
    IrisVal* v = (IrisVal*)xmalloc(sizeof(IrisVal));
    v->tag = IRIS_TAG_ENUM;

    if (nfields == 0) {
        v->ptr = NULL;
    } else {
        IrisVal** fields = (IrisVal**)xmalloc(nfields * sizeof(IrisVal*));
        va_list args;
        va_start(args, nfields);
        for (int i = 0; i < nfields; i++) {
            fields[i] = va_arg(args, IrisVal*);
        }
        va_end(args);
        IrisEnum* e = (IrisEnum*)xmalloc(sizeof(IrisEnum));
        e->tag = tag;
        e->fields = fields;
        e->len = (size_t)nfields;
        v->ptr = e;
    }
    return v;
}

int64_t iris_get_variant_tag(IrisVal* v) {
    if (!v) return 0;

    // Hybrid enum representation:
    // 1) unit/tag-only enums may be passed as immediate tags cast to pointers
    // 2) payload enums are boxed IRIS_TAG_ENUM values
    uintptr_t raw = (uintptr_t)v;
    if (raw <= (uintptr_t)0xFFFF) {
        return (int64_t)raw;
    }

    if (v->tag == IRIS_TAG_ENUM) {
        IrisEnum* e = (IrisEnum*)v->ptr;
        return e ? e->tag : 0;
    }

    // Boxed integer fallback for tag-only values that traveled through containers.
    if (v->tag == IRIS_TAG_I64) {
        return v->i64;
    }
    if (v->tag == IRIS_TAG_I32) {
        return (int64_t)v->i32;
    }

    return 0;
}

IrisVal* iris_extract_variant_field(IrisVal* v, int64_t field_idx) {
    if (!v) return NULL;

    // Immediate tag-only variants carry no payload fields.
    if ((uintptr_t)v <= (uintptr_t)0xFFFF) return NULL;

    if (v->tag != IRIS_TAG_ENUM || !v->ptr) return NULL;
    IrisEnum* e = (IrisEnum*)v->ptr;
    if ((size_t)field_idx >= e->len) return NULL;
    return e->fields[field_idx];
}


// ── Adaptive AI runtime (std.adaptive) ────────────────────────────────────

struct IrisAdaptiveState {
    char*     name;
    double*   params;
    int64_t   n_params;
    double    learning_rate;
    double    risk_threshold;
    double    retrain_threshold;
    int64_t   min_obs_retrain;
    int64_t   obs_count;
    double    mean_err;
    double    m2;
    double    last_prediction;
    int       initialized;
};

// ── Internal implementations (IrisAdaptiveState*) ──────────────────────────

IrisAdaptiveState* iris_adaptive_new_impl(const char* name,
                                           int64_t n_params,
                                           double learning_rate,
                                           double risk_threshold) {
    IrisAdaptiveState* s = (IrisAdaptiveState*)xmalloc(sizeof(IrisAdaptiveState));
    s->name = name ? xstrdup(name) : xstrdup("adaptive");
    s->n_params = n_params;
    s->params = (double*)xcalloc((size_t)(n_params > 0 ? n_params : 1), sizeof(double));
    s->learning_rate = learning_rate > 0.0 ? learning_rate : 0.01;
    s->risk_threshold = risk_threshold > 0.0 ? risk_threshold : 2.0;
    s->retrain_threshold = 0.05;
    s->min_obs_retrain = 10;
    s->obs_count = 0;
    s->mean_err = 0.0;
    s->m2 = 0.0;
    s->last_prediction = 0.0;
    s->initialized = 1;
    return s;
}

void iris_adaptive_free_impl(IrisAdaptiveState* state) {
    if (!state) return;
    if (state->name) free(state->name);
    if (state->params) free(state->params);
    free(state);
}

const char* iris_adaptive_name_impl(IrisAdaptiveState* state) {
    return state ? state->name : "";
}

double iris_adaptive_get_param_impl(IrisAdaptiveState* state, int64_t idx) {
    if (!state || idx < 0 || idx >= state->n_params) return 0.0;
    return state->params[idx];
}

void iris_adaptive_set_param_impl(IrisAdaptiveState* state, int64_t idx, double value) {
    if (!state || idx < 0 || idx >= state->n_params) return;
    state->params[idx] = value;
}

int64_t iris_adaptive_n_params_impl(IrisAdaptiveState* state) {
    return state ? state->n_params : 0;
}

double iris_adaptive_learning_rate_impl(IrisAdaptiveState* state) {
    return state ? state->learning_rate : 0.0;
}

void iris_adaptive_set_learning_rate_impl(IrisAdaptiveState* state, double lr) {
    if (state) state->learning_rate = lr > 0.0 ? lr : 0.001;
}

void iris_adaptive_observe_impl(IrisAdaptiveState* state,
                                 const double* inputs, int64_t n_inputs,
                                 double target) {
    if (!state || !inputs || n_inputs <= 0) return;
    double pred = iris_adaptive_predict_impl(state, inputs, n_inputs);
    double error = pred - target;
    state->last_prediction = pred;
    iris_adaptive_record_error_impl(state, error);
    double lr = state->learning_rate;
    for (int64_t i = 0; i < n_inputs && i < state->n_params; i++) {
        state->params[i] -= lr * error * inputs[i];
    }
}

double iris_adaptive_predict_impl(IrisAdaptiveState* state,
                                  const double* inputs, int64_t n_inputs) {
    if (!state || !inputs) return 0.0;
    double pred = 0.0;
    for (int64_t i = 0; i < n_inputs && i < state->n_params; i++) {
        pred += state->params[i] * inputs[i];
    }
    return pred;
}

double iris_adaptive_train_batch_impl(IrisAdaptiveState* state,
                                       const double* inputs, int64_t n_samples,
                                       int64_t n_features, const double* targets) {
    if (!state || !inputs || !targets || n_samples <= 0) return 0.0;
    double total_loss = 0.0;
    for (int64_t b = 0; b < n_samples; b++) {
        const double* row = inputs + b * n_features;
        double pred = iris_adaptive_predict_impl(state, row, n_features);
        double error = pred - targets[b];
        total_loss += error * error;
        double lr = state->learning_rate;
        for (int64_t i = 0; i < n_features && i < state->n_params; i++) {
            state->params[i] -= lr * error * row[i];
        }
    }
    state->last_prediction = total_loss / (double)n_samples;
    return state->last_prediction;
}

void iris_adaptive_record_error_impl(IrisAdaptiveState* state, double error) {
    if (!state) return;
    state->obs_count++;
    double delta = error - state->mean_err;
    state->mean_err += delta / (double)state->obs_count;
    double delta2 = error - state->mean_err;
    state->m2 += delta * delta2;
}

IrisRiskMetrics iris_adaptive_get_risk_impl(IrisAdaptiveState* state) {
    IrisRiskMetrics m;
    m.mean_error = state ? state->mean_err : 0.0;
    m.max_error = 0.0;
    m.observations = state ? state->obs_count : 0;
    m.errors = 0;
    m.last_risk = 0.0;
    m.confidence = 1.0;
    if (state && state->obs_count > 1) {
        double variance = state->m2 / (double)(state->obs_count - 1);
        double std_dev = sqrt(variance);
        double risk_score = std_dev + fabs(state->mean_err);
        m.last_risk = risk_score > 1.0 ? 1.0 : risk_score;
        m.confidence = std_dev > 0.0 ? exp(-fabs(state->mean_err) / std_dev) : 1.0;
        if (m.confidence > 1.0) m.confidence = 1.0;
        if (m.last_risk > 0.5) m.errors = 1;
    }
    return m;
}

int iris_adaptive_is_unsafe_impl(IrisAdaptiveState* state) {
    if (!state) return 0;
    IrisRiskMetrics m = iris_adaptive_get_risk_impl(state);
    return m.last_risk > 0.5 || m.confidence < 0.3;
}

void iris_adaptive_set_risk_threshold_impl(IrisAdaptiveState* state, double threshold) {
    if (state) state->risk_threshold = threshold > 0.0 ? threshold : 1.0;
}

IrisUncertainty iris_adaptive_predict_with_uncertainty_impl(IrisAdaptiveState* state,
                                                             const double* inputs,
                                                             int64_t n_inputs) {
    IrisUncertainty u;
    u.mean = iris_adaptive_predict_impl(state, inputs, n_inputs);
    u.variance = 1.0;
    u.lower_95 = u.mean - 1.96;
    u.upper_95 = u.mean + 1.96;
    u.confidence = 0.5;
    if (state && state->obs_count > 1) {
        u.variance = state->m2 / (double)(state->obs_count - 1);
        double std_dev = sqrt(u.variance);
        u.lower_95 = u.mean - 1.96 * std_dev;
        u.upper_95 = u.mean + 1.96 * std_dev;
        u.confidence = std_dev > 0.0 ? exp(-fabs(u.mean) / std_dev) : 0.5;
        if (u.confidence > 1.0) u.confidence = 1.0;
    }
    return u;
}

double iris_adaptive_uncertainty_bayes_update_impl(IrisAdaptiveState* state,
                                                    double prior_mean, double prior_var,
                                                    double observation, double obs_var) {
    if (!state) return 0.0;
    /*
     * Normal-normal conjugate update, treating `observation` as one new datum
     * on top of the state's accumulated error statistics.
     *
     * This previously computed a posterior from the accumulated data only and
     * then returned `posterior_mean + observation`. That is not a posterior:
     * with no accumulated data it returned `prior_mean + observation`, where a
     * Bayesian update with no data must return the prior mean, and the result
     * grew without bound as observations arrived. The `observation` argument
     * was never actually folded into the update. Since this value is what an
     * adaptive system uses to decide whether it is confident enough to act, a
     * posterior that is not a posterior is a live hazard. See known-issues #30.
     */
    if (prior_var <= 0.0) prior_var = 1.0;
    if (obs_var <= 0.0) obs_var = 1.0;
    double n = (double)state->obs_count;
    double posterior_precision = 1.0 / prior_var + (n + 1.0) / obs_var;
    if (posterior_precision <= 0.0) posterior_precision = 1.0;
    double data_sum = n * state->mean_err + observation;
    return (prior_mean / prior_var + data_sum / obs_var) / posterior_precision;
}

int iris_adaptive_should_retrain_impl(IrisAdaptiveState* state) {
    if (!state) return 0;
    if (state->obs_count < state->min_obs_retrain) return 0;
    return fabs(state->mean_err) > state->retrain_threshold;
}

double iris_adaptive_auto_retrain_impl(IrisAdaptiveState* state,
                                       const double* inputs, int64_t n_samples,
                                       int64_t n_features, const double* targets) {
    if (!state) return 0.0;
    if (!iris_adaptive_should_retrain_impl(state)) {
        return state->mean_err;
    }
    double loss = iris_adaptive_train_batch_impl(state, inputs, n_samples, n_features, targets);
    iris_adaptive_reset_stats_impl(state);
    return loss;
}

void iris_adaptive_set_retrain_threshold_impl(IrisAdaptiveState* state, double threshold) {
    if (state) state->retrain_threshold = threshold > 0.0 ? threshold : 0.01;
}

void iris_adaptive_set_min_observations_for_retrain_impl(IrisAdaptiveState* state, int64_t n) {
    if (state) state->min_obs_retrain = n > 0 ? n : 1;
}

void iris_adaptive_adapt_threshold_impl(IrisAdaptiveState* state, double observed_error) {
    if (!state) return;
    double current = state->risk_threshold;
    double alpha = 0.1;
    double abs_err = fabs(observed_error);
    if (abs_err > current) {
        state->risk_threshold = current + alpha * (abs_err - current);
    } else {
        state->risk_threshold = current - alpha * (current - abs_err) * 0.5;
    }
    if (state->risk_threshold < 0.1) state->risk_threshold = 0.1;
    double error_rate = fabs(state->mean_err);
    if (state->obs_count > 0 && error_rate > 0.05) {
        state->learning_rate *= 1.01;
        if (state->learning_rate > 1.0) state->learning_rate = 1.0;
    } else if (state->obs_count > 10 && error_rate < 0.001) {
        state->learning_rate *= 0.99;
        if (state->learning_rate < 0.0001) state->learning_rate = 0.0001;
    }
}

double iris_adaptive_current_threshold_impl(IrisAdaptiveState* state) {
    return state ? state->risk_threshold : 0.0;
}

int64_t iris_adaptive_observation_count_impl(IrisAdaptiveState* state) {
    return state ? state->obs_count : 0;
}

double iris_adaptive_mean_error_impl(IrisAdaptiveState* state) {
    return state ? state->mean_err : 0.0;
}

void iris_adaptive_reset_stats_impl(IrisAdaptiveState* state) {
    if (!state) return;
    state->obs_count = 0;
    state->mean_err = 0.0;
    state->m2 = 0.0;
}

// ── Extern-compatible wrappers (int64_t handle) ───────────────────────────
static IrisAdaptiveState* ad_h(int64_t h) { return (IrisAdaptiveState*)(intptr_t)h; }

int64_t iris_adaptive_new(const char* name, int64_t n_params,
                           double learning_rate, double risk_threshold) {
    return (int64_t)(intptr_t)iris_adaptive_new_impl(name, n_params, learning_rate, risk_threshold);
}
int64_t iris_adaptive_free(int64_t handle) {
    iris_adaptive_free_impl(ad_h(handle));
    return 0;
}
const char* iris_adaptive_name(int64_t handle) {
    IrisAdaptiveState* state = ad_h(handle);
    if (state && state->name) return xstrdup(state->name);
    return xstrdup("");
}
double iris_adaptive_get_param(int64_t handle, int64_t idx) {
    return iris_adaptive_get_param_impl(ad_h(handle), idx);
}
int64_t iris_adaptive_set_param(int64_t handle, int64_t idx, double value) {
    iris_adaptive_set_param_impl(ad_h(handle), idx, value);
    return 0;
}
int64_t iris_adaptive_n_params(int64_t handle) {
    return iris_adaptive_n_params_impl(ad_h(handle));
}
double iris_adaptive_learning_rate(int64_t handle) {
    return iris_adaptive_learning_rate_impl(ad_h(handle));
}
int64_t iris_adaptive_set_learning_rate(int64_t handle, double lr) {
    iris_adaptive_set_learning_rate_impl(ad_h(handle), lr);
    return 0;
}
int64_t iris_adaptive_record_error(int64_t handle, double error) {
    iris_adaptive_record_error_impl(ad_h(handle), error);
    return 0;
}
int iris_adaptive_is_unsafe(int64_t handle) {
    return iris_adaptive_is_unsafe_impl(ad_h(handle));
}
int64_t iris_adaptive_set_risk_threshold(int64_t handle, double threshold) {
    iris_adaptive_set_risk_threshold_impl(ad_h(handle), threshold);
    return 0;
}
int iris_adaptive_should_retrain(int64_t handle) {
    return iris_adaptive_should_retrain_impl(ad_h(handle));
}
int64_t iris_adaptive_set_retrain_threshold(int64_t handle, double threshold) {
    iris_adaptive_set_retrain_threshold_impl(ad_h(handle), threshold);
    return 0;
}
int64_t iris_adaptive_set_min_observations_for_retrain(int64_t handle, int64_t n) {
    iris_adaptive_set_min_observations_for_retrain_impl(ad_h(handle), n);
    return 0;
}
int64_t iris_adaptive_adapt_threshold(int64_t handle, double observed_error) {
    iris_adaptive_adapt_threshold_impl(ad_h(handle), observed_error);
    return 0;
}
double iris_adaptive_current_threshold(int64_t handle) {
    return iris_adaptive_current_threshold_impl(ad_h(handle));
}
int64_t iris_adaptive_observation_count(int64_t handle) {
    return iris_adaptive_observation_count_impl(ad_h(handle));
}
double iris_adaptive_mean_error(int64_t handle) {
    return iris_adaptive_mean_error_impl(ad_h(handle));
}
int64_t iris_adaptive_reset_stats(int64_t handle) {
    iris_adaptive_reset_stats_impl(ad_h(handle));
    return 0;
}
double iris_adaptive_uncertainty_bayes_update(int64_t handle,
                                               double prior_mean, double prior_var,
                                               double observation, double obs_var) {
    return iris_adaptive_uncertainty_bayes_update_impl(ad_h(handle),
               prior_mean, prior_var, observation, obs_var);
}

int iris_sandbox_check_ffi(const char* lib_path) {
    (void)lib_path;
    return sandbox_allow_ffi ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Weak References, GC Stats, and Concurrency Timeouts
// ---------------------------------------------------------------------------

IrisWeakRef* iris_weak_ref_new(void* target) {
    IrisWeakRef* w = (IrisWeakRef*)xmalloc(sizeof(IrisWeakRef));
    w->target = target;
    w->next_weak = NULL;
    if (!target) return w;
    pthread_mutex_lock(&rc_global_mu);
    RcEntry* e = rc_find(target);
    if (e) {
        w->next_weak = e->weak_refs;
        e->weak_refs = w;
    } else {
        w->target = NULL;
    }
    pthread_mutex_unlock(&rc_global_mu);
    return w;
}

IrisOption* iris_weak_ref_upgrade(IrisWeakRef* w) {
    if (!w || !w->target) return iris_make_none();
    pthread_mutex_lock(&rc_global_mu);
    RcEntry* e = rc_find(w->target);
    if (e && e->count > 0) {
        e->count++;
        e->color = RC_COLOR_BLACK;
        void* ptr = w->target;
        pthread_mutex_unlock(&rc_global_mu);
        return iris_make_some((IrisVal*)ptr);
    }
    w->target = NULL;
    pthread_mutex_unlock(&rc_global_mu);
    return iris_make_none();
}

int32_t iris_weak_ref_alive(IrisWeakRef* w) {
    if (!w || !w->target) return 0;
    pthread_mutex_lock(&rc_global_mu);
    RcEntry* e = rc_find(w->target);
    int32_t alive = (e && e->count > 0) ? 1 : 0;
    if (!alive) w->target = NULL;
    pthread_mutex_unlock(&rc_global_mu);
    return alive;
}

void iris_gc_stats(int64_t* out_alloc, int64_t* out_freed, int64_t* out_cycles, int64_t* out_weak_inval) {
    pthread_mutex_lock(&rc_global_mu);
    if (out_alloc) *out_alloc = gc_total_allocated;
    if (out_freed) *out_freed = gc_total_freed;
    if (out_cycles) *out_cycles = gc_cycles_collected;
    if (out_weak_inval) *out_weak_inval = gc_weak_refs_invalidated;
    pthread_mutex_unlock(&rc_global_mu);
}

IrisOption* iris_chan_recv_timeout(IrisChannel* c, int64_t timeout_ms) {
    if (!c || timeout_ms <= 0) return iris_chan_try_recv(c);
    pthread_mutex_lock(&c->mu);
    if (c->count == 0) {
        struct timespec ts;
#if defined(_WIN32) || defined(_WIN64)
        SYSTEMTIME st;
        GetSystemTime(&st);
        FILETIME ft;
        SystemTimeToFileTime(&st, &ft);
        ULARGE_INTEGER uli;
        uli.LowPart = ft.dwLowDateTime;
        uli.HighPart = ft.dwHighDateTime;
        uint64_t ns_total = uli.QuadPart * 100 + (uint64_t)timeout_ms * 1000000;
        ts.tv_sec = (time_t)(ns_total / 1000000000);
        ts.tv_nsec = (long)(ns_total % 1000000000);
#else
        clock_gettime(CLOCK_REALTIME, &ts);
        int64_t ns = ts.tv_nsec + (timeout_ms % 1000) * 1000000;
        ts.tv_sec += (timeout_ms / 1000) + (ns / 1000000000);
        ts.tv_nsec = ns % 1000000000;
#endif
        while (c->count == 0) {
            int res = pthread_cond_timedwait(&c->not_empty, &c->mu, &ts);
            if (res != 0) break;
        }
    }
    if (c->count == 0) {
        pthread_mutex_unlock(&c->mu);
        return iris_make_none();
    }
    IrisVal* val = c->buf[c->head];
    c->head = (c->head + 1) % c->cap;
    c->count--;
    if (c->max_cap >= 0) pthread_cond_signal(&c->not_full);
    pthread_mutex_unlock(&c->mu);
    IrisOption* opt = iris_make_some(val);
    if (val) iris_release(val);
    return opt;
}

int32_t iris_task_group_join_timeout(IrisTaskGroup* tg, int64_t timeout_ms) {
    (void)timeout_ms;
    if (!tg) return 1;
    iris_task_group_join(tg);
    return 1;
}

// ---------------------------------------------------------------------------
// New builtins (called via generic BuiltinCall codegen path)
// ---------------------------------------------------------------------------

int64_t iris_list_remove(IrisList* l, int64_t idx) {
    if (idx < 0 || (size_t)idx >= l->len) {
        fprintf(stderr, "iris: list_remove index %ld out of bounds (len=%zu)\n", (long)idx, l->len);
        abort();
    }
    IrisVal* removed = l->data[idx];
    for (size_t j = (size_t)idx; j + 1 < l->len; j++)
        l->data[j] = l->data[j + 1];
    l->len--;
    return removed->i64;
}

int64_t iris_list_insert(IrisList* l, int64_t idx, IrisVal* val) {
    if (idx < 0 || (size_t)idx > l->len) {
        fprintf(stderr, "iris: list_insert index %ld out of bounds (len=%zu)\n", (long)idx, l->len);
        abort();
    }
    if (l->len == l->cap) {
        l->cap *= 2;
        l->data = xrealloc(l->data, sizeof(IrisVal*) * l->cap);
    }
    for (size_t j = l->len; j > (size_t)idx; j--)
        l->data[j] = l->data[j - 1];
    if (val) iris_retain(val);
    l->data[idx] = val;
    l->len++;
    return 0;
}

IrisVal* iris_map_entries(IrisVal* map_val) {
    IrisMap* m = iris_unbox_map(map_val);
    IrisList* entries = iris_list_new();
    for (size_t b = 0; b < m->n_buckets; b++) {
        for (IrisMapEntry* e = m->buckets[b]; e; e = e->next) {
            IrisVal* key = iris_box_str(e->key);
            IrisVal* pair = iris_make_struct(2, key, e->val);
            iris_list_push(entries, pair);
        }
    }
    return iris_box_list(entries);
}

IrisVal* iris_recv_timeout(IrisVal* chan_val, int64_t timeout_ms) {
    IrisChannel* c = iris_unbox_chan(chan_val);
    IrisOption* opt = iris_chan_recv_timeout(c, timeout_ms);
    return (IrisVal*)opt;
}

void iris_chan_send_b(IrisVal* chan_val, IrisVal* val) {
    IrisChannel* c = iris_unbox_chan(chan_val);
    iris_chan_send(c, val);
}

IrisWeakRef* iris_weak_ref(IrisVal* val) {
    return iris_weak_ref_new(val);
}

int32_t iris_weak_alive(IrisWeakRef* w) {
    return iris_weak_ref_alive(w);
}

IrisVal* iris_weak_upgrade(IrisWeakRef* w) {
    IrisOption* opt = iris_weak_ref_upgrade(w);
    return (IrisVal*)opt;
}

IrisVal* iris_gc_stats_map(void) {
    int64_t alloc = 0, freed = 0, cycles = 0, weak_inv = 0;
    iris_gc_stats(&alloc, &freed, &cycles, &weak_inv);
    IrisMap* m = iris_map_new();
    iris_map_set(m, iris_box_str("allocated"),   iris_box_i64(alloc));
    iris_map_set(m, iris_box_str("freed"),       iris_box_i64(freed));
    iris_map_set(m, iris_box_str("cycles"),      iris_box_i64(cycles));
    iris_map_set(m, iris_box_str("weak_invalidated"), iris_box_i64(weak_inv));
    return iris_box_map(m);
}

int32_t iris_gc_collect_call(void) {
    iris_gc_collect();
    return 0;
}

