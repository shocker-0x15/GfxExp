#include "path_tracing_shared.h"

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



CUDA_DEVICE_KERNEL void visualizeToOutputBuffer(
    optixu::NativeBlockBuffer2D<shared::GBuffer0> gBuffer0,
    void* linearBuffer,
    shared::BufferToDisplay bufferTypeToDisplay,
    float brightness,
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
        //value = brightness * typedLinearBuffer[linearIndex];
        //// simple tone-map
        //value.x = 1 - std::exp(-value.x);
        //value.y = 1 - std::exp(-value.y);
        //value.z = 1 - std::exp(-value.z);

        value = typedLinearBuffer[linearIndex];
        float lum = sRGB_calcLuminance(make_float3(value));
        if (lum > 0.0f) {
            float lumT = 1 - std::exp(-brightness * lum);
            // simple tone-map
            value = value * (lumT / lum);
        }
        else {
            value.x = value.y = value.z = 0.0f;
        }
        value.w = 1.0f;

        //const auto reinhard = [](float x, float Lw) {
        //    return x / (1 + x) * (1 + x / (Lw * Lw));
        //};
        //value = brightness * typedLinearBuffer[linearIndex];
        //value.x = reinhard(value.x, 1.0f);
        //value.y = reinhard(value.y, 1.0f);
        //value.z = reinhard(value.z, 1.0f);
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
