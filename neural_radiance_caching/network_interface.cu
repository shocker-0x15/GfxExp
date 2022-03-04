#include "network_interface.h"

#include <cuda_runtime.h>

#define TCNN_MIN_GPU_ARCH 86
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#include <memory>

using namespace tcnn;
using precision_t = network_precision_t;

// Position: 3
// Scattered Direction: 2
// Normal: 2
// Roughness: 1
// Diffuse Reflectance: 3
// Specular Reflectance: 3
constexpr static uint32_t numInputDims = 14;
// RGB Radiance: 3
constexpr static uint32_t numOutputDims = 3;

class NeuralRadianceCache::Priv {
    std::shared_ptr<Loss<precision_t>> loss;
    std::shared_ptr<Optimizer<precision_t>> optimizer;
    std::shared_ptr<NetworkWithInputEncoding<precision_t>> network;

    std::shared_ptr<Trainer<float, precision_t, precision_t>> trainer;

public:
    friend class NeuralRadianceCache;

    Priv() {}
};



NeuralRadianceCache::NeuralRadianceCache() {
    m = new Priv();
}

NeuralRadianceCache::~NeuralRadianceCache() {
    delete m;
}

void NeuralRadianceCache::initialize() {
    json config = {
        {"loss", {
            {"otype", "RelativeL2Luminance"}
        }},
        {"optimizer", {
            {"otype", "EMA"},
            {"decay", 0.99f},
            {"nesteed", {
                {"otype", "Adam"},
                {"learning_rate", 1e-2f},
                {"beta1", 0.9f},
                {"beta2", 0.99f},
                {"l2_reg", 0.0f}
            }}
        }},
        {"encoding", {
            {"otype", "NRC"}
        }},
        {"network", {
            {"otype", "FullyFusedMLP"},
            {"n_neurons", 64},
            {"n_hidden_layers", 5},
            {"activation", "ReLU"},
            {"output_activation", "None"},
        }}
    };

    m->loss.reset(create_loss<precision_t>(config.value("loss", json::object())));
    m->optimizer.reset(create_optimizer<precision_t>(config.value("optimizer", json::object())));
    m->network = std::make_shared<NetworkWithInputEncoding<precision_t>>(
        numInputDims, numOutputDims,
        config.value("encoding", json::object()),
        config.value("network", json::object()));

    m->trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(
        m->network, m->optimizer, m->loss);
}

void NeuralRadianceCache::finalize() {
    m->trainer = nullptr;
    m->network = nullptr;
    m->optimizer = nullptr;
    m->loss = nullptr;
}

void NeuralRadianceCache::infer(
    CUstream stream, float* inputData, uint32_t numData, float* predictionData) {
    uint32_t numDataPadded = (numData + 255) / 256 * 256;
    GPUMatrix<float> inputs(inputData, numInputDims, numDataPadded);
    GPUMatrix<float> predictions(predictionData, numOutputDims, numDataPadded);
    m->network->inference(stream, inputs, predictions);
}

void NeuralRadianceCache::train(
    CUstream stream, float* inputData, float* targetData, uint32_t numData) {
    uint32_t numDataPadded = (numData + 255) / 256 * 256;
    GPUMatrix<float> inputs(inputData, numInputDims, numDataPadded);
    GPUMatrix<float> targets(targetData, numOutputDims, numDataPadded);
    m->trainer->training_step(stream, inputs, targets, nullptr);
}
