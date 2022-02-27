#pragma once

#include <cuda.h>
//#include "neural_radiance_caching_shared.h"

class NeuralRadianceCache {
public:
    class Priv;
private:
    Priv* m = nullptr;

public:
    NeuralRadianceCache();
    ~NeuralRadianceCache();

    void initialize();
    void finalize();

    void infer(CUstream stream, float* inputData, uint32_t numData, float* predictionData);
    void train(CUstream stream, float* inputData, float* targetData, uint32_t numData);
};
