#define PURE_CUDA
#include "../regir_shared.h"

CUDA_DEVICE_KERNEL void copyToLinearBuffers(
    optixu::NativeBlockBuffer2D<float4> colorAccumBuffer,
    optixu::NativeBlockBuffer2D<float4> albedoAccumBuffer,
    optixu::NativeBlockBuffer2D<float4> normalAccumBuffer,
    optixu::NativeBlockBuffer2D<float4> motionVectorBuffer,
    float4* linearColorBuffer,
    float4* linearAlbedoBuffer,
    float4* linearNormalBuffer,
    float2* linearMotionVectorBuffer,
    uint2 imageSize) {
    uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                   blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= imageSize.x ||
        launchIndex.y >= imageSize.y)
        return;

    uint32_t linearIndex = launchIndex.y * imageSize.x + launchIndex.x;
    linearColorBuffer[linearIndex] = colorAccumBuffer.read(launchIndex);
    linearAlbedoBuffer[linearIndex] = albedoAccumBuffer.read(launchIndex);
    float3 normal = getXYZ(normalAccumBuffer.read(launchIndex));
    if (normal.x != 0 || normal.y != 0 || normal.z != 0)
        normal = normalize(normal);
    linearNormalBuffer[linearIndex] = make_float4(normal, 1.0f);
    float4 motionVector = motionVectorBuffer.read(launchIndex);
    linearMotionVectorBuffer[linearIndex] = make_float2(motionVector.x, motionVector.y);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t calcCellLinearIndex(
    const float3 &gridOrigin, const float3 &gridCellSize, const uint3 &gridDimension,
    const float3 &positionInWorld) {
    float3 relPos = positionInWorld - gridOrigin;
    uint32_t ix = min(max(static_cast<uint32_t>(relPos.x / gridCellSize.x), 0u),
                      gridDimension.x - 1);
    uint32_t iy = min(max(static_cast<uint32_t>(relPos.y / gridCellSize.y), 0u),
                      gridDimension.y - 1);
    uint32_t iz = min(max(static_cast<uint32_t>(relPos.z / gridCellSize.z), 0u),
                      gridDimension.z - 1);
    return iz * gridDimension.x * gridDimension.y
        + iy * gridDimension.x
        + ix;
}

template <typename T>
CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t getFNV1Hash32(const T &x) {
    static const uint32_t FNV_OFFSET_BASIS_32 = 2166136261U;
    //static const uint64_t FNV_OFFSET_BASIS_64 = 14695981039346656037U;

    static const uint32_t FNV_PRIME_32 = 16777619U;
    //static const uint64_t FNV_PRIME_64 = 1099511628211LLU;

    uint32_t hash = FNV_OFFSET_BASIS_32;
    auto bytes = reinterpret_cast<const uint8_t*>(&x);
    for (int i = 0; i < sizeof(T); ++i)
        hash = (FNV_PRIME_32 * hash) ^ (bytes[i]);

    return hash;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float3 calcCellColor(
    const float3 &gridOrigin, const float3 &gridCellSize, const uint3 &gridDimension,
    const float3 &positionInWorld) {
    uint32_t cellLinearIndex = calcCellLinearIndex(gridOrigin, gridCellSize, gridDimension, positionInWorld);

    uint32_t hash = getFNV1Hash32(cellLinearIndex);
    shared::PCG32RNG rng;
    rng.setState((static_cast<uint64_t>(hash) << 32) | 3018421212);

    return HSVtoRGB(rng.getFloat0cTo1o(),
                    0.5f + 0.5f * rng.getFloat0cTo1o(),
                    0.5f + 0.5f * rng.getFloat0cTo1o());
}

CUDA_DEVICE_KERNEL void visualizeToOutputBuffer(
    optixu::NativeBlockBuffer2D<shared::GBuffer0> gBuffer0, uint32_t visualizeCell,
    float3 gridOrigin, float3 gridCellSize, uint3 gridDimension,
    void* linearBuffer,
    shared::BufferToDisplay bufferTypeToDisplay,
    float motionVectorOffset, float motionVectorScale,
    optixu::NativeBlockBuffer2D<float4> outputBuffer,
    uint2 imageSize) {
    uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                   blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= imageSize.x ||
        launchIndex.y >= imageSize.y)
        return;

    uint32_t linearIndex = launchIndex.y * imageSize.x + launchIndex.x;
    float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    switch (bufferTypeToDisplay) {
    case shared::BufferToDisplay::NoisyBeauty:
    case shared::BufferToDisplay::DenoisedBeauty: {
        auto typedLinearBuffer = reinterpret_cast<const float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        if (visualizeCell) {
            shared::GBuffer0 gb0 = gBuffer0.read(launchIndex);
            float3 cellColor = calcCellColor(gridOrigin, gridCellSize, gridDimension, gb0.positionInWorld);
            value.x *= cellColor.x;
            value.y *= cellColor.y;
            value.z *= cellColor.z;
        }
        break;
    }
    case shared::BufferToDisplay::Albedo: {
        auto typedLinearBuffer = reinterpret_cast<const float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        break;
    }
    case shared::BufferToDisplay::Normal: {
        auto typedLinearBuffer = reinterpret_cast<const float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        value.x = 0.5f + 0.5f * value.x;
        value.y = 0.5f + 0.5f * value.y;
        value.z = 0.5f + 0.5f * value.z;
        break;
    }
    case shared::BufferToDisplay::Flow: {
        auto typedLinearBuffer = reinterpret_cast<const float2*>(linearBuffer);
        float2 f2Value = typedLinearBuffer[linearIndex];
        value = make_float4(fminf(fmaxf(motionVectorScale * f2Value.x + motionVectorOffset, 0.0f), 1.0f),
                            fminf(fmaxf(motionVectorScale * f2Value.y + motionVectorOffset, 0.0f), 1.0f),
                            motionVectorOffset, 1.0f);
        break;
    }
    default:
        break;
    }
    outputBuffer.write(launchIndex, value);
}
