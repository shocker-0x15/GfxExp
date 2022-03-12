#pragma once

#include <cuda.h>

enum class PositionEncoding {
    OneBlob,
    Hash,
};

class NeuralRadianceCache {
public:
    class Priv;
private:
    Priv* m = nullptr;

public:
    NeuralRadianceCache();
    ~NeuralRadianceCache();

    void initialize(PositionEncoding posEnc, uint32_t numHiddenLayers, float learningRate);
    void finalize();

    void infer(CUstream stream, float* inputData, uint32_t numData, float* predictionData);
    void train(CUstream stream, float* inputData, float* targetData, uint32_t numData);
};
