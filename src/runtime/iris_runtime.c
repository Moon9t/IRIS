// iris_runtime.c — IRIS Language Runtime Library
//
// Implements all iris_* functions declared in iris_runtime.h.
// Memory model: malloc-based arena — allocations are never explicitly freed
// (suitable for scripting and ML workloads that run-and-exit).
// Concurrency: real pthreads for spawn, par_for, channels, atomics.

#include "iris_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>
#include <stdarg.h>
#include <errno.h>

// ---------------------------------------------------------------------------
// Internal memory helpers
// ---------------------------------------------------------------------------

static void* xmalloc(size_t n) {
    void* p = malloc(n);
    if (!p) { fprintf(stderr, "iris: out of memory\n"); abort(); }
    return p;
}

static void* xcalloc(size_t n, size_t sz) {
    void* p = calloc(n, sz);
    if (!p) { fprintf(stderr, "iris: out of memory\n"); abort(); }
    return p;
}

static void* xrealloc(void* p, size_t n) {
    void* q = realloc(p, n);
    if (!q) { fprintf(stderr, "iris: out of memory\n"); abort(); }
    return q;
}

static char* xstrdup(const char* s) {
    size_t n = strlen(s) + 1;
    char* d = xmalloc(n);
    memcpy(d, s, n);
    return d;
}

// ---------------------------------------------------------------------------
// Boxing / Unboxing
// ---------------------------------------------------------------------------

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
    return r;
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
        case IRIS_TAG_ENUM: printf("variant(%ld)", (long)v->i64);      break;
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
void iris_print_i64(int64_t v)  { printf("%ld\n",  (long)v); }
void iris_print_i32(int32_t v)  { printf("%d\n",   v); }
void iris_print_f64(double v) {
    /* Print integer-valued doubles without decimal to match interpreter output */
    if (v == (double)(long long)v && v > -1e15 && v < 1e15)
        printf("%lld\n", (long long)v);
    else
        printf("%g\n", v);
}
void iris_print_f32(float v)    { iris_print_f64((double)v); }
void iris_print_bool(int v)     { printf("%s\n", v ? "true" : "false"); }

void iris_panic(const char* msg) {
    fprintf(stderr, "panic: %s\n", msg);
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

int64_t iris_str_len(const char* s) { return (int64_t)strlen(s); }

char* iris_str_concat(const char* a, const char* b) {
    size_t la = strlen(a), lb = strlen(b);
    char* r = xmalloc(la + lb + 1);
    memcpy(r, a, la);
    memcpy(r + la, b, lb + 1);
    return r;
}

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
// Option
// ---------------------------------------------------------------------------

IrisOption* iris_make_some(IrisVal* val) {
    IrisOption* o = xmalloc(sizeof(IrisOption));
    o->has_value = 1;  o->value = val;
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
    r->is_ok = 1;  r->value = val;
    return r;
}
IrisResult* iris_make_err(IrisVal* val) {
    IrisResult* r = xmalloc(sizeof(IrisResult));
    r->is_ok = 0;  r->value = val;
    return r;
}
int      iris_is_ok(IrisResult* res)            { return res ? res->is_ok : 0; }
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

IrisList* iris_list_new(void) {
    IrisList* l = xcalloc(1, sizeof(IrisList));
    l->cap  = 8;
    l->data = xmalloc(sizeof(IrisVal*) * l->cap);
    return l;
}
void iris_list_push(IrisList* l, IrisVal* val) {
    if (l->len == l->cap) {
        l->cap *= 2;
        l->data = xrealloc(l->data, sizeof(IrisVal*) * l->cap);
    }
    l->data[l->len++] = val;
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
    if (idx < 0 || (size_t)idx >= l->len) {
        fprintf(stderr, "iris: list set index %ld out of bounds\n", (long)idx);
        abort();
    }
    l->data[idx] = val;
}
IrisVal* iris_list_pop(IrisList* l) {
    if (l->len == 0) { fprintf(stderr, "iris: pop on empty list\n"); abort(); }
    return l->data[--l->len];
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
void iris_map_set(IrisMap* m, const char* key, IrisVal* val) {
    size_t h = hash_str(key) % m->n_buckets;
    for (IrisMapEntry* e = m->buckets[h]; e; e = e->next) {
        if (strcmp(e->key, key) == 0) { e->val = val; return; }
    }
    IrisMapEntry* e = xmalloc(sizeof(IrisMapEntry));
    e->key = xstrdup(key);  e->val = val;  e->next = m->buckets[h];
    m->buckets[h] = e;  m->len++;
}
IrisVal* iris_map_get(IrisMap* m, const char* key) {
    size_t h = hash_str(key) % m->n_buckets;
    for (IrisMapEntry* e = m->buckets[h]; e; e = e->next)
        if (strcmp(e->key, key) == 0) return e->val;
    return NULL;
}
int iris_map_contains(IrisMap* m, const char* key) {
    size_t h = hash_str(key) % m->n_buckets;
    for (IrisMapEntry* e = m->buckets[h]; e; e = e->next)
        if (strcmp(e->key, key) == 0) return 1;
    return 0;
}
void iris_map_remove(IrisMap* m, const char* key) {
    size_t h = hash_str(key) % m->n_buckets;
    IrisMapEntry** pp = &m->buckets[h];
    while (*pp) {
        if (strcmp((*pp)->key, key) == 0) { *pp = (*pp)->next; m->len--; return; }
        pp = &(*pp)->next;
    }
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

void iris_list_sort(IrisList* l) {
    if (!l || l->len <= 1) return;
    /* Simple bubble sort for stability; replace with qsort if needed */
    for (size_t i = 0; i < l->len - 1; i++) {
        for (size_t j = 0; j < l->len - 1 - i; j++) {
            if (iris_val_compare(l->data[j], l->data[j+1]) > 0) {
                IrisVal* t = l->data[j];
                l->data[j] = l->data[j+1];
                l->data[j+1] = t;
            }
        }
    }
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

char* iris_file_read_all(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    size_t size = (size_t)sz;
    char* buf = xmalloc(size + 1);
    size_t n = fread(buf, 1, size, f);
    buf[n] = '\0';
    fclose(f);
    return buf;
}

char* iris_file_write_all(const char* path, const char* contents) {
    FILE* f = fopen(path, "wb");
    if (!f) return NULL;
    size_t len = strlen(contents);
    int ok = (fwrite(contents, 1, len, f) == len);
    fclose(f);
    return ok ? (char*)path : NULL;
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

char* iris_env_var(const char* key) {
    const char* v = getenv(key);
    return v ? xstrdup(v) : NULL;
}

// ---------------------------------------------------------------------------
// Channels and concurrency
// ---------------------------------------------------------------------------

#define CHAN_INIT_CAP 64u

IrisChannel* iris_chan_new(void) {
    IrisChannel* c = xmalloc(sizeof(IrisChannel));
    c->cap   = CHAN_INIT_CAP;
    c->buf   = xmalloc(sizeof(IrisVal*) * c->cap);
    c->head  = c->tail = c->count = 0;
    pthread_mutex_init(&c->mu,        NULL);
    pthread_cond_init (&c->not_empty, NULL);
    pthread_cond_init (&c->not_full,  NULL);
    return c;
}
void iris_chan_send(IrisChannel* c, IrisVal* val) {
    pthread_mutex_lock(&c->mu);
    while (c->count == c->cap) pthread_cond_wait(&c->not_full, &c->mu);
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
    pthread_cond_signal(&c->not_full);
    pthread_mutex_unlock(&c->mu);
    return val;
}
void iris_spawn_fn(void* fn) {
    pthread_t t;
    /* IRIS spawn functions return i64 but we run them on a detached thread
       and discard the return value — compatible on all LP64 platforms. */
    pthread_create(&t, NULL, (void*(*)(void*))fn, NULL);
    pthread_detach(t);
}

typedef struct { void (*fn)(int64_t); int64_t i; } ParArg;
static void* par_for_worker(void* arg) {
    ParArg* a = (ParArg*)arg;
    a->fn(a->i);
    free(a);
    return NULL;
}
void iris_par_for(void (*fn)(int64_t), int64_t start, int64_t end) {
    int64_t n = end - start;
    if (n <= 0) return;
    pthread_t* threads = xmalloc(sizeof(pthread_t) * (size_t)n);
    for (int64_t i = start; i < end; i++) {
        ParArg* a = xmalloc(sizeof(ParArg));
        a->fn = fn;  a->i = i;
        pthread_create(&threads[i - start], NULL, par_for_worker, a);
    }
    for (int64_t i = 0; i < n; i++) pthread_join(threads[i], NULL);
    free(threads);
}
void iris_barrier(void) { /* no-op outside par_for; par_for already joins all */ }

// ---------------------------------------------------------------------------
// Atomics and mutexes
// ---------------------------------------------------------------------------

IrisAtomic* iris_atomic_new(IrisVal* initial) {
    IrisAtomic* a = xmalloc(sizeof(IrisAtomic));
    pthread_mutex_init(&a->mu, NULL);
    a->val = initial;
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
IrisMutex* iris_mutex_new(void) {
    IrisMutex* m = xmalloc(sizeof(IrisMutex));
    pthread_mutex_init(&m->mu, NULL);
    return m;
}
IrisVal* iris_mutex_lock(IrisMutex* m) {
    pthread_mutex_lock(&m->mu);
    IrisVal* r = xmalloc(sizeof(IrisVal));
    r->tag = IRIS_TAG_UNIT;  r->i64 = 0;
    return r;
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
// Non-scalar array fallback (for complex / mixed-type arrays)
// ---------------------------------------------------------------------------

IrisList*  iris_alloc_array(void)                      { return iris_list_new(); }
IrisVal*   iris_array_load(IrisList* arr, int64_t idx) { return iris_list_get(arr, idx); }
void       iris_array_store(IrisList* arr, int64_t idx, IrisVal* val) { iris_list_set(arr, idx, val); }

// ---------------------------------------------------------------------------
// Tensor ops (shape-tracking stubs — not a real tensor kernel)
// ---------------------------------------------------------------------------

void* iris_tensor_op(void)                { return NULL; }
void* iris_tensor_load(void* t, ...)      { (void)t; return NULL; }
void  iris_tensor_store(void* t, ...)     { (void)t; }
