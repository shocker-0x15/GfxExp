#define PURE_CUDA
#include "../tfdm_shared.h"

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
    uint2 launchIndex = make_uint2(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= imageSize.x ||
        launchIndex.y >= imageSize.y)
        return;

    uint32_t linearIndex = launchIndex.y * imageSize.x + launchIndex.x;
    linearColorBuffer[linearIndex] = colorAccumBuffer.read(launchIndex);
    linearAlbedoBuffer[linearIndex] = albedoAccumBuffer.read(launchIndex);
    Normal3D normal(getXYZ(normalAccumBuffer.read(launchIndex)));
    if (normal.x != 0 || normal.y != 0 || normal.z != 0)
        normal.normalize();
    linearNormalBuffer[linearIndex] = make_float4(normal.toNative(), 1.0f);
    float4 motionVector = motionVectorBuffer.read(launchIndex);
    linearMotionVectorBuffer[linearIndex] = make_float2(motionVector.x, motionVector.y);
}



CUDA_DEVICE_KERNEL void visualizeToOutputBuffer(
    optixu::NativeBlockBuffer2D<shared::GBuffer0> gBuffer0,
    optixu::NativeBlockBuffer2D<shared::GBuffer1> gBuffer1,
    void* linearBuffer,
    shared::BufferToDisplay bufferTypeToDisplay,
    float motionVectorOffset, float motionVectorScale,
    optixu::NativeBlockBuffer2D<float4> outputBuffer,
    uint2 imageSize) {
    uint2 launchIndex = make_uint2(
        blockDim.x * blockIdx.x + threadIdx.x,
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
    case shared::BufferToDisplay::TexCoord: {
        Point2D texCoord;
        texCoord.x = gBuffer0.read(launchIndex).texCoord_x;
        texCoord.y = gBuffer1.read(launchIndex).texCoord_y;
#if STORE_BARYCENTRICS
        value.x = 1 - texCoord.x - texCoord.y;
        value.y = texCoord.x;
        value.z = texCoord.y;
#else
        value.x = std::fmod(texCoord.x, 1.0f);
        value.y = std::fmod(texCoord.y, 1.0f);
        value.z = 0.5f * std::fmod((texCoord.x - value.x) / 10.0f, 1.0f)
            + 0.5f * std::fmod(texCoord.y - value.y, 2.0f);
#endif
        break;
    }
    case shared::BufferToDisplay::Flow: {
        auto typedLinearBuffer = reinterpret_cast<const float2*>(linearBuffer);
        float2 f2Value = typedLinearBuffer[linearIndex];
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
