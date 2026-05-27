// iris_ml_kernels.c — IRIS ML Compute Engine
// v0.5.0 
//
// Real, high-performance ML kernels for native IRIS binaries.
// All functions operate on IrisTensor (contiguous f32 row-major buffers).
//
// Compile with -DIRIS_USE_BLAS -lopenblas for BLAS-accelerated matmul.
// Without BLAS: uses 32×32 blocked tiling for cache-friendly matmul.

#include "iris_ml_kernels.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

// Self-contained allocation wrappers (iris_runtime.c has its own static copies;
// since we compile as a separate translation unit we need our own).
static inline void* ml_malloc(size_t size) {
    void* p = malloc(size);
    if (!p && size) { fprintf(stderr, "IRIS ML: out of memory (%zu bytes)\n", size); abort(); }
    return p;
}
static inline void* ml_calloc(size_t count, size_t size) {
    void* p = calloc(count, size);
    if (!p && count && size) { fprintf(stderr, "IRIS ML: out of memory (%zu × %zu)\n", count, size); abort(); }
    return p;
}

#ifdef IRIS_USE_BLAS
#include <cblas.h>
#endif

// ===========================================================================
// Blocked Matmul with optional BLAS dispatch
// ===========================================================================

IrisTensor* iris_tensor_matmul_blocked(IrisTensor* a, IrisTensor* b) {
    if (!a || !b || a->ndim < 2 || b->ndim < 2) return NULL;
    int64_t m = a->shape[a->ndim - 2];
    int64_t k = a->shape[a->ndim - 1];
    int64_t n = b->shape[b->ndim - 1];
    if (b->shape[b->ndim - 2] != k) return NULL;

    int64_t out_shape[2] = { m, n };
    IrisTensor* out = iris_tensor_zeros(2, out_shape);

#ifdef IRIS_USE_BLAS
    // BLAS dispatch: peak CPU performance via cblas_sgemm
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)n, (int)k,
                1.0f, a->data, (int)k, b->data, (int)n,
                0.0f, out->data, (int)n);
#else
    // 32×32 blocked tiling for L1 cache friendliness
    #define TILE 32
    for (int64_t ii = 0; ii < m; ii += TILE) {
        for (int64_t jj = 0; jj < n; jj += TILE) {
            for (int64_t ll = 0; ll < k; ll += TILE) {
                int64_t ie = (ii + TILE < m) ? ii + TILE : m;
                int64_t je = (jj + TILE < n) ? jj + TILE : n;
                int64_t le = (ll + TILE < k) ? ll + TILE : k;
                for (int64_t i = ii; i < ie; i++) {
                    for (int64_t l = ll; l < le; l++) {
                        float a_il = a->data[i * k + l];
                        for (int64_t j = jj; j < je; j++) {
                            out->data[i * n + j] += a_il * b->data[l * n + j];
                        }
                    }
                }
            }
        }
    }
    #undef TILE
#endif
    return out;
}

// ===========================================================================
// Activation Functions
// ===========================================================================

IrisTensor* iris_tensor_softmax(IrisTensor* t, int32_t axis) {
    if (!t) return NULL;
    (void)axis;  // currently row-wise (last dim)
    IrisTensor* out = iris_tensor_alloc(t->ndim, t->shape);
    int64_t rows = (t->ndim >= 2) ? t->numel / t->shape[t->ndim - 1] : 1;
    int64_t cols = (t->ndim >= 2) ? t->shape[t->ndim - 1] : t->numel;

    for (int64_t r = 0; r < rows; r++) {
        float* src = t->data + r * cols;
        float* dst = out->data + r * cols;
        // Numerical stability: subtract max
        float mx = src[0];
        for (int64_t c = 1; c < cols; c++)
            if (src[c] > mx) mx = src[c];
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; c++) {
            dst[c] = expf(src[c] - mx);
            sum += dst[c];
        }
        if (sum > 0.0f)
            for (int64_t c = 0; c < cols; c++) dst[c] /= sum;
    }
    return out;
}

// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
IrisTensor* iris_tensor_gelu(IrisTensor* t) {
    if (!t) return NULL;
    IrisTensor* out = iris_tensor_alloc(t->ndim, t->shape);
    const float c = 0.7978845608f; // sqrt(2/π)
    for (int64_t i = 0; i < t->numel; i++) {
        float x = t->data[i];
        out->data[i] = 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
    }
    return out;
}

IrisTensor* iris_tensor_leaky_relu(IrisTensor* t, float alpha) {
    if (!t) return NULL;
    IrisTensor* out = iris_tensor_alloc(t->ndim, t->shape);
    for (int64_t i = 0; i < t->numel; i++) {
        float x = t->data[i];
        out->data[i] = (x > 0.0f) ? x : alpha * x;
    }
    return out;
}

// ===========================================================================
// Neural Network Layers
// ===========================================================================

// Conv2D — direct convolution (no im2col) with stride and padding.
// Input:  (N, C_in, H, W) or (C_in, H, W)  [auto-prepends N=1]
// Weight: (C_out, C_in, kH, kW)
// Bias:   (C_out,) or NULL

IrisTensor* iris_tensor_conv2d(IrisTensor* input, IrisTensor* weight,
                                IrisTensor* bias, int64_t stride, int64_t padding) {
    if (!input || !weight || input->ndim < 3 || weight->ndim != 4) return NULL;

    int has_batch = (input->ndim == 4);
    int64_t N    = has_batch ? input->shape[0] : 1;
    int64_t C_in = has_batch ? input->shape[1] : input->shape[0];
    int64_t H_in = has_batch ? input->shape[2] : input->shape[1];
    int64_t W_in = has_batch ? input->shape[3] : input->shape[2];

    int64_t C_out = weight->shape[0];
    if (weight->shape[1] != C_in) return NULL;
    int64_t kH    = weight->shape[2];
    int64_t kW    = weight->shape[3];

    int64_t H_out = (H_in + 2 * padding - kH) / stride + 1;
    int64_t W_out = (W_in + 2 * padding - kW) / stride + 1;
    if (H_out <= 0 || W_out <= 0) return NULL;

    int64_t out_shape[4] = { N, C_out, H_out, W_out };
    IrisTensor* out = iris_tensor_zeros(4, out_shape);

    for (int64_t n = 0; n < N; n++) {
        for (int64_t co = 0; co < C_out; co++) {
            for (int64_t oh = 0; oh < H_out; oh++) {
                for (int64_t ow = 0; ow < W_out; ow++) {
                    float val = (bias && co < bias->numel) ? bias->data[co] : 0.0f;

                    for (int64_t ci = 0; ci < C_in; ci++) {
                        for (int64_t kh = 0; kh < kH; kh++) {
                            for (int64_t kw = 0; kw < kW; kw++) {
                                int64_t ih = oh * stride - padding + kh;
                                int64_t iw = ow * stride - padding + kw;
                                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                    float in_val = input->data[
                                        n * (C_in * H_in * W_in) +
                                        ci * (H_in * W_in) +
                                        ih * W_in + iw];
                                    float w_val = weight->data[
                                        co * (C_in * kH * kW) +
                                        ci * (kH * kW) +
                                        kh * kW + kw];
                                    val += in_val * w_val;
                                }
                            }
                        }
                    }

                    out->data[n * (C_out * H_out * W_out) +
                              co * (H_out * W_out) +
                              oh * W_out + ow] = val;
                }
            }
        }
    }
    return out;
}

// MaxPool2D
IrisTensor* iris_tensor_maxpool2d(IrisTensor* input, int64_t kernel_size,
                                   int64_t stride) {
    if (!input || input->ndim < 3) return NULL;

    int has_batch = (input->ndim == 4);
    int64_t N = has_batch ? input->shape[0] : 1;
    int64_t C = has_batch ? input->shape[1] : input->shape[0];
    int64_t H = has_batch ? input->shape[2] : input->shape[1];
    int64_t W = has_batch ? input->shape[3] : input->shape[2];

    int64_t H_out = (H - kernel_size) / stride + 1;
    int64_t W_out = (W - kernel_size) / stride + 1;
    if (H_out <= 0 || W_out <= 0) return NULL;

    int64_t out_shape[4] = { N, C, H_out, W_out };
    IrisTensor* out = iris_tensor_alloc(4, out_shape);

    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            for (int64_t oh = 0; oh < H_out; oh++) {
                for (int64_t ow = 0; ow < W_out; ow++) {
                    float max_val = -INFINITY;
                    for (int64_t kh = 0; kh < kernel_size; kh++) {
                        for (int64_t kw = 0; kw < kernel_size; kw++) {
                            int64_t ih = oh * stride + kh;
                            int64_t iw = ow * stride + kw;
                            float v = input->data[
                                n * (C * H * W) +
                                c * (H * W) +
                                ih * W + iw];
                            if (v > max_val) max_val = v;
                        }
                    }
                    out->data[n * (C * H_out * W_out) +
                              c * (H_out * W_out) +
                              oh * W_out + ow] = max_val;
                }
            }
        }
    }
    return out;
}

// BatchNorm (inference mode)
// y = (x - mean) / sqrt(var + eps) * gamma + beta
// Assumes NCHW layout, channels at dim 1.
IrisTensor* iris_tensor_batchnorm(IrisTensor* input, IrisTensor* gamma,
                                   IrisTensor* beta, IrisTensor* running_mean,
                                   IrisTensor* running_var, float eps) {
    if (!input || input->ndim < 2) return NULL;
    IrisTensor* out = iris_tensor_alloc(input->ndim, input->shape);

    int64_t C = input->shape[1];
    int64_t spatial = input->numel / (input->shape[0] * C);

    for (int64_t n = 0; n < input->shape[0]; n++) {
        for (int64_t c = 0; c < C; c++) {
            float mean = (running_mean && c < running_mean->numel)
                         ? running_mean->data[c] : 0.0f;
            float var  = (running_var && c < running_var->numel)
                         ? running_var->data[c] : 1.0f;
            float g = (gamma && c < gamma->numel) ? gamma->data[c] : 1.0f;
            float b = (beta  && c < beta->numel)  ? beta->data[c]  : 0.0f;
            float inv_std = 1.0f / sqrtf(var + eps);

            for (int64_t s = 0; s < spatial; s++) {
                int64_t idx = n * C * spatial + c * spatial + s;
                out->data[idx] = (input->data[idx] - mean) * inv_std * g + b;
            }
        }
    }
    return out;
}

// ===========================================================================
// Loss Functions
// ===========================================================================

// MSE Loss: mean((pred - target)²)
float iris_tensor_mse_loss(IrisTensor* pred, IrisTensor* target) {
    if (!pred || !target || pred->numel != target->numel || pred->numel == 0)
        return 0.0f;
    float sum = 0.0f;
    for (int64_t i = 0; i < pred->numel; i++) {
        float d = pred->data[i] - target->data[i];
        sum += d * d;
    }
    return sum / (float)pred->numel;
}

// Cross-entropy loss with numerically stable log-softmax
// logits: (N, C), targets: (N,) with integer class indices as floats
float iris_tensor_cross_entropy_loss(IrisTensor* logits, IrisTensor* targets) {
    if (!logits || !targets || logits->ndim != 2) return 0.0f;
    int64_t N = logits->shape[0];
    int64_t C = logits->shape[1];
    float total_loss = 0.0f;

    for (int64_t n = 0; n < N; n++) {
        float* row = logits->data + n * C;
        int64_t label = (int64_t)targets->data[n];
        if (label < 0 || label >= C) continue;

        // log_softmax = x_k - log(sum(exp(x - max)))
        float mx = row[0];
        for (int64_t c = 1; c < C; c++)
            if (row[c] > mx) mx = row[c];
        float log_sum_exp = 0.0f;
        for (int64_t c = 0; c < C; c++)
            log_sum_exp += expf(row[c] - mx);
        log_sum_exp = mx + logf(log_sum_exp);

        total_loss -= (row[label] - log_sum_exp);
    }
    return (N > 0) ? total_loss / (float)N : 0.0f;
}

// Binary cross-entropy: -mean(t*log(p) + (1-t)*log(1-p))
float iris_tensor_binary_ce_loss(IrisTensor* pred, IrisTensor* target) {
    if (!pred || !target || pred->numel != target->numel || pred->numel == 0)
        return 0.0f;
    const float eps = 1e-7f;
    float sum = 0.0f;
    for (int64_t i = 0; i < pred->numel; i++) {
        float p = pred->data[i];
        float t = target->data[i];
        if (p < eps) p = eps;
        if (p > 1.0f - eps) p = 1.0f - eps;
        sum -= t * logf(p) + (1.0f - t) * logf(1.0f - p);
    }
    return sum / (float)pred->numel;
}

// ===========================================================================
// Optimizers
// ===========================================================================

// SGD: params -= lr * grads (in-place)
void iris_tensor_sgd_step(IrisTensor* params, IrisTensor* grads, float lr) {
    if (!params || !grads || params->numel != grads->numel) return;
    for (int64_t i = 0; i < params->numel; i++) {
        params->data[i] -= lr * grads->data[i];
    }
}

// Adam optimizer
IrisAdamState* iris_adam_create(int64_t numel, float lr, float beta1,
                                 float beta2, float eps) {
    IrisAdamState* s = (IrisAdamState*)ml_calloc(1, sizeof(IrisAdamState));
    s->numel = numel;
    s->m = (float*)ml_calloc((size_t)numel, sizeof(float));
    s->v = (float*)ml_calloc((size_t)numel, sizeof(float));
    s->t = 0;
    s->lr    = lr;
    s->beta1 = (beta1 > 0.0f) ? beta1 : 0.9f;
    s->beta2 = (beta2 > 0.0f) ? beta2 : 0.999f;
    s->eps   = (eps   > 0.0f) ? eps   : 1e-8f;
    return s;
}

void iris_adam_step(IrisAdamState* state, IrisTensor* params, IrisTensor* grads) {
    if (!state || !params || !grads) return;
    if (params->numel != state->numel || grads->numel != state->numel) return;

    state->t++;
    float bc1 = 1.0f - powf(state->beta1, (float)state->t);
    float bc2 = 1.0f - powf(state->beta2, (float)state->t);

    for (int64_t i = 0; i < state->numel; i++) {
        float g = grads->data[i];
        state->m[i] = state->beta1 * state->m[i] + (1.0f - state->beta1) * g;
        state->v[i] = state->beta2 * state->v[i] + (1.0f - state->beta2) * g * g;
        float m_hat = state->m[i] / bc1;
        float v_hat = state->v[i] / bc2;
        params->data[i] -= state->lr * m_hat / (sqrtf(v_hat) + state->eps);
    }
}

void iris_adam_free(IrisAdamState* state) {
    if (!state) return;
    free(state->m);
    free(state->v);
    free(state);
}

// ===========================================================================
// Tensor Utilities
// ===========================================================================

IrisTensor* iris_tensor_clone(IrisTensor* t) {
    if (!t) return NULL;
    IrisTensor* out = iris_tensor_alloc(t->ndim, t->shape);
    memcpy(out->data, t->data, (size_t)t->numel * sizeof(float));
    return out;
}

IrisTensor* iris_tensor_flatten(IrisTensor* t) {
    if (!t) return NULL;
    int64_t flat_shape[1] = { t->numel };
    return iris_tensor_reshape(t, 1, flat_shape);
}

IrisTensor* iris_tensor_cat(IrisTensor* a, IrisTensor* b) {
    if (!a || !b) return NULL;
    if (a->ndim != b->ndim) return NULL;
    // All dims except dim 0 must match
    for (int32_t d = 1; d < a->ndim; d++) {
        if (a->shape[d] != b->shape[d]) return NULL;
    }
    int64_t* new_shape = (int64_t*)ml_malloc((size_t)a->ndim * sizeof(int64_t));
    new_shape[0] = a->shape[0] + b->shape[0];
    for (int32_t d = 1; d < a->ndim; d++) new_shape[d] = a->shape[d];

    IrisTensor* out = iris_tensor_alloc(a->ndim, new_shape);
    memcpy(out->data, a->data, (size_t)a->numel * sizeof(float));
    memcpy(out->data + a->numel, b->data, (size_t)b->numel * sizeof(float));
    free(new_shape);
    return out;
}

IrisTensor* iris_tensor_rand(int32_t ndim, const int64_t* shape) {
    IrisTensor* t = iris_tensor_alloc(ndim, shape);
    for (int64_t i = 0; i < t->numel; i++) {
        t->data[i] = (float)rand() / (float)RAND_MAX;
    }
    return t;
}

// Random normal via Box-Muller transform
IrisTensor* iris_tensor_randn(int32_t ndim, const int64_t* shape) {
    IrisTensor* t = iris_tensor_alloc(ndim, shape);
    for (int64_t i = 0; i < t->numel; i += 2) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = (float)rand() / (float)RAND_MAX;
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 6.28318530718f * u2;
        t->data[i] = r * cosf(theta);
        if (i + 1 < t->numel)
            t->data[i + 1] = r * sinf(theta);
    }
    return t;
}

IrisTensor* iris_tensor_scale(IrisTensor* t, float s) {
    if (!t) return NULL;
    IrisTensor* out = iris_tensor_alloc(t->ndim, t->shape);
    for (int64_t i = 0; i < t->numel; i++)
        out->data[i] = t->data[i] * s;
    return out;
}

IrisTensor* iris_tensor_add_scalar(IrisTensor* t, float s) {
    if (!t) return NULL;
    IrisTensor* out = iris_tensor_alloc(t->ndim, t->shape);
    for (int64_t i = 0; i < t->numel; i++)
        out->data[i] = t->data[i] + s;
    return out;
}
