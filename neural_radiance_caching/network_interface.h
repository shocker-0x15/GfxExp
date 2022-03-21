#pragma once

#include <cuda.h>

enum class PositionEncoding {
    TriangleWave,
    HashGrid,
};

// JP: サンプルプログラム全体をnvcc経由でコンパイルしないといけない状況を避けるため、
//     pimplイディオムによってtiny-cuda-nnをcpp側に隔離する。
// EN: Isolate the tiny-cuda-nn into the cpp side by pimpl idiom to avoid the situation where
//     the entire sample program needs to be compiled via nvcc.
class NeuralRadianceCache {
    class Priv;
    Priv* m = nullptr;

public:
    NeuralRadianceCache();
    ~NeuralRadianceCache();

    void initialize(PositionEncoding posEnc, uint32_t numHiddenLayers, float learningRate);
    void finalize();

    void infer(CUstream stream, float* inputData, uint32_t numData, float* predictionData);
    void train(CUstream stream, float* inputData, float* targetData, uint32_t numData,
               float* lossOnCPU = nullptr);
};
