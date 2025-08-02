#define PURE_CUDA
#include "../regir_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void copyToLinearBuffers(
    float4* linearColorBuffer,
    float4* linearAlbedoBuffer,
    float4* linearNormalBuffer,
    float2* linearMotionVectorBuffer)
{
    const uint2 launchIndex = make_uint2(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= plp.s->imageSize.x ||
        launchIndex.y >= plp.s->imageSize.y)
        return;

    const uint32_t linearIndex = launchIndex.y * plp.s->imageSize.x + launchIndex.x;
    linearColorBuffer[linearIndex] = plp.s->beautyAccumBuffer.read(launchIndex);
    linearAlbedoBuffer[linearIndex] = plp.s->albedoAccumBuffer.read(launchIndex);
    Normal3D normal(getXYZ(plp.s->normalAccumBuffer.read(launchIndex)));
    if (normal.x != 0 || normal.y != 0 || normal.z != 0)
        normal.normalize();
    linearNormalBuffer[linearIndex] = make_float4(normal.toNative(), 1.0f);
    const GBuffer1Elements gb1Elems = plp.s->GBuffer1[plp.f->bufferIndex].read(launchIndex);
    linearMotionVectorBuffer[linearIndex] = gb1Elems.motionVector.toNative();
}

CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t calcCellLinearIndex(
    const Point3D &gridOrigin, const Vector3D &gridCellSize, const uint3 &gridDimension,
    const Point3D &positionInWorld)
{
    const Point3D relPos(positionInWorld - gridOrigin);
    const uint32_t ix = min(max(
        static_cast<uint32_t>(relPos.x / gridCellSize.x), 0u), gridDimension.x - 1);
    const uint32_t iy = min(max(
        static_cast<uint32_t>(relPos.y / gridCellSize.y), 0u), gridDimension.y - 1);
    const uint32_t iz = min(max(
        static_cast<uint32_t>(relPos.z / gridCellSize.z), 0u), gridDimension.z - 1);
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
    const auto bytes = reinterpret_cast<const uint8_t*>(&x);
    for (int i = 0; i < sizeof(T); ++i)
        hash = (FNV_PRIME_32 * hash) ^ (bytes[i]);

    return hash;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE RGB calcCellColor(
    const Point3D &gridOrigin, const Vector3D &gridCellSize, const uint3 &gridDimension,
    const Point3D &positionInWorld)
{
    const uint32_t cellLinearIndex = calcCellLinearIndex(gridOrigin, gridCellSize, gridDimension, positionInWorld);

    const uint32_t hash = getFNV1Hash32(cellLinearIndex);
    PCG32RNG rng;
    rng.setState((static_cast<uint64_t>(hash) << 32) | 3018421212);

    return HSVtoRGB(
        rng.getFloat0cTo1o(),
        0.5f + 0.5f * rng.getFloat0cTo1o(),
        0.5f + 0.5f * rng.getFloat0cTo1o());
}

CUDA_DEVICE_KERNEL void visualizeToOutputBuffer(
    uint32_t visualizeCell,
    Point3D gridOrigin, Vector3D gridCellSize, uint3 gridDimension,
    void* linearBuffer,
    BufferToDisplay bufferTypeToDisplay,
    float motionVectorOffset, float motionVectorScale,
    optixu::NativeBlockBuffer2D<float4> outputBuffer)
{
    const uint32_t bufIdx = plp.f->bufferIndex;
    const uint2 launchIndex = make_uint2(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= plp.s->imageSize.x ||
        launchIndex.y >= plp.s->imageSize.y)
        return;

    const uint32_t linearIndex = launchIndex.y * plp.s->imageSize.x + launchIndex.x;
    float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    switch (bufferTypeToDisplay) {
    case BufferToDisplay::NoisyBeauty:
    case BufferToDisplay::DenoisedBeauty: {
        const auto typedLinearBuffer = reinterpret_cast<const float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        if (visualizeCell) {
            const GBuffer2Elements gb2Elems = plp.s->GBuffer2[bufIdx].read(launchIndex);
            const RGB cellColor = calcCellColor(gridOrigin, gridCellSize, gridDimension, gb2Elems.positionInWorld);
            value.x *= cellColor.r;
            value.y *= cellColor.g;
            value.z *= cellColor.b;
        }
        break;
    }
    case BufferToDisplay::Albedo: {
        const auto typedLinearBuffer = reinterpret_cast<const float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        break;
    }
    case BufferToDisplay::Normal: {
        const auto typedLinearBuffer = reinterpret_cast<const float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        value.x = 0.5f + 0.5f * value.x;
        value.y = 0.5f + 0.5f * value.y;
        value.z = 0.5f + 0.5f * value.z;
        break;
    }
    case BufferToDisplay::Flow: {
        const auto typedLinearBuffer = reinterpret_cast<const float2*>(linearBuffer);
        const float2 f2Value = typedLinearBuffer[linearIndex];
        value = make_float4(
            fminf(fmaxf(motionVectorScale * f2Value.x + motionVectorOffset, 0.0f), 1.0f),
            fminf(fmaxf(motionVectorScale * f2Value.y + motionVectorOffset, 0.0f), 1.0f),
            motionVectorOffset, 1.0f);
        break;
    }
    default:
        break;
    }
    outputBuffer.write(launchIndex, value);
}
