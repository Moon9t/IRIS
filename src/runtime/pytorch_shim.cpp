#include "iris_runtime.h"
#include <stdio.h>

#if defined(LIBTORCH_ENABLED)
#include <torch/script.h>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

struct PyTorchModel {
    std::shared_ptr<torch::jit::script::Module> module;
};

extern "C" void* iris_pytorch_load(const char* model_path) {
    if (!model_path) return NULL;
#ifdef _WIN32
    DWORD attrs = GetFileAttributesA(model_path);
    if (attrs == INVALID_FILE_ATTRIBUTES) {
        return NULL;
    }
#else
    struct stat st;
    if (stat(model_path, &st) != 0) {
        return NULL;
    }
#endif

    try {
        auto module = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path));
        PyTorchModel* m = new PyTorchModel();
        m->module = module;
        return (void*)m;
    } catch (const std::exception& e) {
        fprintf(stderr, "iris: libtorch load error: %s\n", e.what());
        return NULL;
    }
}

static torch::Tensor iris_tensor_to_torch(IrisTensor* t) {
    // Create a CPU tensor from IrisTensor data (copies)
    std::vector<int64_t> sizes(t->ndim);
    for (int i = 0; i < t->ndim; ++i) sizes[i] = (int64_t)t->shape[i];
    auto options = torch::TensorOptions().dtype(c10::ScalarType::Float).device(c10::Device(c10::DeviceType::CPU));
    return torch::from_blob(t->data, sizes, options).clone(); // clone to own memory
}

static IrisTensor* torch_to_iris_tensor(const torch::Tensor& tt) {
    auto t = tt.contiguous().to(c10::Device(c10::DeviceType::CPU));
    int ndim = t.dim();
    IrisTensor* out = iris_tensor_alloc((int32_t)ndim, (const int64_t*)t.sizes().data());
    // copy data
    float* src = (float*)t.data_ptr();
    for (int64_t i = 0; i < out->numel; ++i) {
        out->data[i] = src[i];
    }
    return out;
}

extern "C" int iris_pytorch_run(void* model, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs) {
    if (!model) return -1;
    PyTorchModel* m = (PyTorchModel*)model;
    try {
        std::vector<torch::jit::IValue> args;
        for (size_t i = 0; i < n_inputs; ++i) {
            args.push_back(iris_tensor_to_torch(inputs[i]));
        }
        auto result = m->module->forward(args);
        std::vector<torch::Tensor> outs;
        if (result.isTensor()) {
            outs.push_back(result.toTensor());
        } else if (result.isTuple()) {
            auto tup = result.toTuple();
            for (const auto& v : tup->elements()) {
                if (v.isTensor()) outs.push_back(v.toTensor());
            }
        }
        *n_outputs = outs.size();
        *outputs = (IrisTensor**)malloc(sizeof(IrisTensor*) * (*n_outputs));
        for (size_t i = 0; i < *n_outputs; ++i) {
            (*outputs)[i] = torch_to_iris_tensor(outs[i]);
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "iris: libtorch run error: %s\n", e.what());
        return -1;
    }
}

extern "C" double iris_pytorch_train_step(void* model, IrisTensor** inputs, size_t n_inputs, IrisTensor** targets, size_t n_targets, double lr) {
    if (!model || n_inputs == 0 || n_targets == 0) return 0.0;
    PyTorchModel* m = (PyTorchModel*)model;
    try {
        std::vector<torch::jit::IValue> args;
        for (size_t i = 0; i < n_inputs; ++i) {
            args.push_back(iris_tensor_to_torch(inputs[i]));
        }
        
        auto out_val = m->module->forward(args);
        torch::Tensor out_tensor;
        if (out_val.isTensor()) {
            out_tensor = out_val.toTensor();
        } else if (out_val.isTuple()) {
            auto tup = out_val.toTuple();
            if (!tup->elements().empty() && tup->elements()[0].isTensor()) {
                out_tensor = tup->elements()[0].toTensor();
            }
        }
        if (!out_tensor.defined()) {
            fprintf(stderr, "iris: train_step forward pass did not return a valid tensor\n");
            return 0.0;
        }

        auto target_tensor = iris_tensor_to_torch(targets[0]);
        auto loss = torch::mse_loss(out_tensor, target_tensor);
        loss.backward();

        for (auto param : m->module->parameters()) {
            if (param.requires_grad() && param.grad().defined()) {
                param.data().sub_(param.grad().data() * lr);
                param.grad().zero_();
            }
        }

        return (double)loss.item<float>();
    } catch (const std::exception& e) {
        fprintf(stderr, "iris: libtorch train_step error: %s\n", e.what());
        return 0.0;
    }
}

extern "C" void iris_pytorch_free(void* model) {
    if (!model) return;
    PyTorchModel* m = (PyTorchModel*)model;
    delete m;
}

#else

extern "C" void* iris_pytorch_load(const char* model_path) {
    (void)model_path; fprintf(stderr, "iris: libtorch support not enabled at build time\n"); return NULL;
}
extern "C" int iris_pytorch_run(void* model, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs) {
    (void)model;(void)inputs;(void)n_inputs;(void)outputs;(void)n_outputs; fprintf(stderr, "iris: libtorch support not enabled at build time\n"); return -1;
}
extern "C" double iris_pytorch_train_step(void* model, IrisTensor** inputs, size_t n_inputs, IrisTensor** targets, size_t n_targets, double lr) {
    (void)model; (void)inputs; (void)n_inputs; (void)targets; (void)n_targets; (void)lr;
    fprintf(stderr, "iris: libtorch support not enabled at build time\n");
    return 0.0;
}
extern "C" void iris_pytorch_free(void* model) { (void)model; }

#endif

