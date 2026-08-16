/* A tiny library with known-answer functions, so the FFI surface can be
 * asserted against values rather than "it did not crash". */
#include <string.h>
#include <stdint.h>
__declspec(dllexport) int64_t t_zero(void)                             { return 7; }
__declspec(dllexport) int64_t t_one(int64_t a)                         { return a * 2; }
__declspec(dllexport) int64_t t_two(int64_t a, int64_t b)              { return a + b; }
__declspec(dllexport) int64_t t_three(int64_t a, int64_t b, int64_t c) { return a + b + c; }
__declspec(dllexport) double  t_f64_one(int64_t a)                     { return (double)a / 2.0; }
__declspec(dllexport) double  t_f64_two(int64_t a, int64_t b)          { return (double)(a + b) / 4.0; }
__declspec(dllexport) const char* t_str(void)                          { return "hello-ffi"; }
__declspec(dllexport) int64_t t_fill(char* buf, int64_t max) {
    const char* src = "world";
    if (!buf || max < 6) return 0;
    strncpy(buf, src, (size_t)max - 1);
    buf[max - 1] = '\0';
    return 1;
}
