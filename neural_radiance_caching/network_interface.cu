#include "network_interface.h"
#include "../common/common_shared.h"

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

void NeuralRadianceCache::initialize(PositionEncoding posEnc, uint32_t numHiddenLayers, float learningRate) {
    json config = {
        {"loss", {
            {"otype", "RelativeL2Luminance"}
        }},
        {"optimizer", {
            {"otype", "EMA"},
            {"decay", 0.99f},
            {"nested", {
                {"otype", "Adam"},
                {"learning_rate", learningRate},
                {"beta1", 0.9f},
                {"beta2", 0.99f},
                //{"l2_reg", 1e-6f},
            }}
        }},
        {"network", {
            {"otype", "FullyFusedMLP"},
            {"n_neurons", 64},
            {"n_hidden_layers", numHiddenLayers},
            {"activation", "ReLU"},
            {"output_activation", "None"},
        }}
    };

    if (posEnc == PositionEncoding::TriangleWave) {
        //config["encoding"] = { {"otype", "NRC"} };
        config["encoding"] = {
            {"otype", "Composite"},
            {"nested", {
                {
                    {"n_dims_to_encode", 3},
                    {"otype", "TriangleWave"},
                    {"n_frequencies", 12},
                },
                {
                    {"n_dims_to_encode", 5},
                    {"otype", "OneBlob"},
                    {"n_bins", 4},
                },
                {
                    {"otype", "Identity"}
                },
            }}
        };
    }
    else if (posEnc == PositionEncoding::HashGrid) {
        config["encoding"] = {
            {"otype", "Composite"},
            {"nested", {
                {
                    {"n_dims_to_encode", 3},
                    {"otype", "HashGrid"},
                    {"n_levels", 16},
                    {"n_features_per_level", 2},
                    {"log2_hashmap_size", 19},
                    {"base_resolution", 16},
                    {"per_level_scale", 1.5},
                },
                {
                    {"n_dims_to_encode", 5},
                    {"otype", "OneBlob"},
                    {"n_bins", 4},
                },
                {
                    {"otype", "Identity"}
                },
            }}
        };
        config["optimizer"]["nested"]["epsilon"] = 1e-15f;
    }

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
    Assert((numData & 0x7F) == 0, "numData must be a multiple of 128.");
    GPUMatrix<float> inputs(inputData, numInputDims, numData);
    GPUMatrix<float> predictions(predictionData, numOutputDims, numData);
    m->network->inference(stream, inputs, predictions);
}

void NeuralRadianceCache::train(
    CUstream stream, float* inputData, float* targetData, uint32_t numData) {
    Assert((numData & 0x7F) == 0, "numData must be a multiple of 128.");
    GPUMatrix<float> inputs(inputData, numInputDims, numData);
    GPUMatrix<float> targets(targetData, numOutputDims, numData);
    m->trainer->training_step(stream, inputs, targets, nullptr);
}
