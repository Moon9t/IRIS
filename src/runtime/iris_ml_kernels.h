// iris_ml_kernels.h — IRIS ML Compute Kernels
// High-performance ML operations for the "C of Machine Learning" vision.
//
// v0.5.0 — Requires iris_runtime.h for IrisTensor and allocation helpers.
//
// Build with -DIRIS_USE_BLAS and link -lopenblas/-lmkl for accelerated matmul.
// Build with -DIRIS_USE_BLAS -I/path/to/openblas/include to enable BLAS dispatch.

#ifndef IRIS_ML_KERNELS_H
#define IRIS_ML_KERNELS_H

#include "iris_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Blocked Matmul with optional BLAS dispatch
// ---------------------------------------------------------------------------
// Uses 32×32 tiles for L1 cache friendliness.  When compiled with
// -DIRIS_USE_BLAS, dispatches to cblas_sgemm for peak CPU performance.
IrisTensor* iris_tensor_matmul_blocked(IrisTensor* a, IrisTensor* b);

// ---------------------------------------------------------------------------
// Activation Functions
// ---------------------------------------------------------------------------
IrisTensor* iris_tensor_softmax(IrisTensor* t, int32_t axis);
IrisTensor* iris_tensor_gelu(IrisTensor* t);
IrisTensor* iris_tensor_leaky_relu(IrisTensor* t, float alpha);

// ---------------------------------------------------------------------------
// Neural Network Layers
// ---------------------------------------------------------------------------
// Conv2D: input (N,C,H,W), weight (C_out,C_in,kH,kW), bias (C_out) or NULL
IrisTensor* iris_tensor_conv2d(IrisTensor* input, IrisTensor* weight,
                                IrisTensor* bias, int64_t stride, int64_t padding);

// MaxPool2D: input (N,C,H,W)
IrisTensor* iris_tensor_maxpool2d(IrisTensor* input, int64_t kernel_size,
                                   int64_t stride);

// BatchNorm (inference mode): y = (x - mean) / sqrt(var + eps) * gamma + beta
IrisTensor* iris_tensor_batchnorm(IrisTensor* input, IrisTensor* gamma,
                                   IrisTensor* beta, IrisTensor* running_mean,
                                   IrisTensor* running_var, float eps);

// ---------------------------------------------------------------------------
// Loss Functions
// ---------------------------------------------------------------------------
// MSE: mean((pred - target)^2)
float iris_tensor_mse_loss(IrisTensor* pred, IrisTensor* target);

// Cross-entropy: logits (N,C), targets (N,) with class indices
float iris_tensor_cross_entropy_loss(IrisTensor* logits, IrisTensor* targets);

// Binary cross-entropy: pred/target in [0,1]
float iris_tensor_binary_ce_loss(IrisTensor* pred, IrisTensor* target);

// ---------------------------------------------------------------------------
// Optimizers
// ---------------------------------------------------------------------------
// SGD: params -= lr * grads (in-place)
void iris_tensor_sgd_step(IrisTensor* params, IrisTensor* grads, float lr);

// Adam optimizer state
typedef struct {
    float*  m;       // first moment estimate
    float*  v;       // second moment estimate
    int64_t numel;   // number of parameters
    int64_t t;       // timestep counter
    float   lr;      // learning rate
    float   beta1;   // first moment decay (default 0.9)
    float   beta2;   // second moment decay (default 0.999)
    float   eps;     // numerical stability (default 1e-8)
} IrisAdamState;

IrisAdamState* iris_adam_create(int64_t numel, float lr, float beta1,
                                 float beta2, float eps);
void iris_adam_step(IrisAdamState* state, IrisTensor* params, IrisTensor* grads);
void iris_adam_free(IrisAdamState* state);

// ---------------------------------------------------------------------------
// Tensor Utilities
// ---------------------------------------------------------------------------
IrisTensor* iris_tensor_clone(IrisTensor* t);
IrisTensor* iris_tensor_flatten(IrisTensor* t);
IrisTensor* iris_tensor_cat(IrisTensor* a, IrisTensor* b);    // concat dim 0
IrisTensor* iris_tensor_rand(int32_t ndim, const int64_t* shape);   // U[0,1)
IrisTensor* iris_tensor_randn(int32_t ndim, const int64_t* shape);  // N(0,1)
IrisTensor* iris_tensor_scale(IrisTensor* t, float s);         // t * scalar
IrisTensor* iris_tensor_add_scalar(IrisTensor* t, float s);    // t + scalar

#ifdef __cplusplus
}
#endif

#endif /* IRIS_ML_KERNELS_H */
