#define PURE_CUDA
#include "../nrtdsm_shared.h"

using namespace shared;

CUDA_DEVICE_FUNCTION float2 computeTexelMinMax(
    const CUtexObject heightMap, const int32_t mipLevel, const int2 &imgSize, const int2 &pixIdx) {
    const auto sample = [&](float px, float py) {
        return tex2DLod<float>(heightMap, px / imgSize.x, py / imgSize.y, mipLevel);
    };

    // TODO?: テクセルコーナー間の補間ではなくテクセルセンター間の補間とすることで
    //        Bilinearサンプル4点じゃなくてPointサンプル4点にできる？
    const float cornerHeightUL = sample(pixIdx.x - 0.0f, pixIdx.y - 0.0f);
    const float cornerHeightUR = sample(pixIdx.x + 1.0f, pixIdx.y - 0.0f);
    const float cornerHeightBL = sample(pixIdx.x - 0.0f, pixIdx.y + 1.0f);
    const float cornerHeightBR = sample(pixIdx.x + 1.0f, pixIdx.y + 1.0f);
    const float minHeight = std::fmin(std::fmin(std::fmin(
        cornerHeightUL, cornerHeightUR), cornerHeightBL), cornerHeightBR);
    const float maxHeight = std::fmax(std::fmax(std::fmax(
        cornerHeightUL, cornerHeightUR), cornerHeightBL), cornerHeightBR);

    return make_float2(minHeight, maxHeight);
}



CUDA_DEVICE_KERNEL void generateFirstMinMaxMipMap(const MaterialData* const material) {
    const int2 pixIdx(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 imgSize = material->heightMapSize;
    if (pixIdx.x >= imgSize.x || pixIdx.y >= imgSize.y)
        return;

    material->minMaxMipMap[0].write(
        pixIdx, 
        computeTexelMinMax(material->heightMap, 0, imgSize, pixIdx));
}



CUDA_DEVICE_KERNEL void generateMinMaxMipMap(
    const MaterialData* material, const uint32_t srcMipLevel) {
    const int2 dstPixIdx(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 srcImageSize = material->heightMapSize >> srcMipLevel;
    const int2 dstImageSize = srcImageSize / 2;
    if (dstPixIdx.x >= dstImageSize.x || dstPixIdx.y >= dstImageSize.y)
        return;

    const int2 basePixIdx = 2 * dstPixIdx;
    float minHeight = INFINITY;
    float maxHeight = -INFINITY;
    float2 minMax;

    const optixu::NativeBlockBuffer2D<float2> &prevMinMaxMip = material->minMaxMipMap[srcMipLevel];

    minMax = prevMinMaxMip.read(basePixIdx + int2(0, 0));
    minHeight = std::fmin(minMax.x, minHeight);
    maxHeight = std::fmax(minMax.y, maxHeight);

    minMax = prevMinMaxMip.read(basePixIdx + int2(1, 0));
    minHeight = std::fmin(minMax.x, minHeight);
    maxHeight = std::fmax(minMax.y, maxHeight);

    minMax = prevMinMaxMip.read(basePixIdx + int2(0, 1));
    minHeight = std::fmin(minMax.x, minHeight);
    maxHeight = std::fmax(minMax.y, maxHeight);

    minMax = prevMinMaxMip.read(basePixIdx + int2(1, 1));
    minHeight = std::fmin(minMax.x, minHeight);
    maxHeight = std::fmax(minMax.y, maxHeight);

    material->minMaxMipMap[srcMipLevel + 1].write(dstPixIdx, make_float2(minHeight, maxHeight));
}
