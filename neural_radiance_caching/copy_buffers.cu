#define PURE_CUDA
#include "neural_radiance_caching_shared.h"

using namespace shared;

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
    bool visualizeTrainingPath,
    void* linearBuffer, BufferToDisplay bufferTypeToDisplay,
    float brightness,
    float motionVectorOffset, float motionVectorScale,
    optixu::NativeBlockBuffer2D<float4> outputBuffer) {
    uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                   blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= plp.s->imageSize.x ||
        launchIndex.y >= plp.s->imageSize.y)
        return;

    uint32_t linearIndex = launchIndex.y * plp.s->imageSize.x + launchIndex.x;
    float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    switch (bufferTypeToDisplay) {
    case BufferToDisplay::NoisyBeauty:
    case BufferToDisplay::DirectlyVisualizedPrediction:
    case BufferToDisplay::DenoisedBeauty: {
        if (bufferTypeToDisplay == BufferToDisplay::DirectlyVisualizedPrediction) {
            const TerminalInfo &terminalInfo = plp.s->inferenceTerminalInfoBuffer[linearIndex];

            float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
            if (terminalInfo.hasQuery) {
                radiance = max(plp.s->inferredRadianceBuffer[linearIndex], make_float3(0.0f, 0.0f, 0.0f));

                const RadianceQuery &terminalQuery = plp.s->inferenceRadianceQueryBuffer[linearIndex];
                radiance *= (terminalQuery.diffuseReflectance + terminalQuery.specularReflectance);
            }
            value = make_float4(terminalInfo.alpha * radiance, 1.0f);
        }
        else {
            auto typedLinearBuffer = reinterpret_cast<const float4*>(linearBuffer);
            //value = brightness * typedLinearBuffer[linearIndex];
            //// simple tone-map
            //value.x = 1 - std::exp(-value.x);
            //value.y = 1 - std::exp(-value.y);
            //value.z = 1 - std::exp(-value.z);

            value = typedLinearBuffer[linearIndex];
        }
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
    case BufferToDisplay::Albedo: {
        auto typedLinearBuffer = reinterpret_cast<const float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        break;
    }
    case BufferToDisplay::Normal: {
        auto typedLinearBuffer = reinterpret_cast<const float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        value.x = 0.5f + 0.5f * value.x;
        value.y = 0.5f + 0.5f * value.y;
        value.z = 0.5f + 0.5f * value.z;
        break;
    }
    case BufferToDisplay::Flow: {
        auto typedLinearBuffer = reinterpret_cast<const float2*>(linearBuffer);
        float2 f2Value = typedLinearBuffer[linearIndex];
        value = make_float4(fminf(fmaxf(motionVectorScale * f2Value.x + motionVectorOffset, 0.0f), 1.0f),
                            fminf(fmaxf(motionVectorScale * f2Value.y + motionVectorOffset, 0.0f), 1.0f),
                            motionVectorOffset, 1.0f);
        break;
    }
    case BufferToDisplay::RenderingPathLength: {
        const TerminalInfo &terminalInfo = plp.s->inferenceTerminalInfoBuffer[linearIndex];
        float t = fminf(terminalInfo.pathLength / 5.0f, 1.0f);
        const float3 R = make_float3(1.0f, 0.0f, 0.0f);
        const float3 G = make_float3(0.0f, 1.0f, 0.0f);
        const float3 B = make_float3(0.0f, 0.0f, 1.0f);
        float3 v = make_float3(0.0f);
        if (t < 0.5f) {
            float tt = 2 * t;
            v = B * (1.0f - tt) + G * tt;
        }
        else {
            float tt = 2 * (t - 0.5f);
            v = G * (1.0f - tt) + R * tt;
        }
        value = make_float4(v, 1.0f);
        break;
    }
    default:
        break;
    }

    if (visualizeTrainingPath) {
        const TerminalInfo &terminalInfo = plp.s->inferenceTerminalInfoBuffer[linearIndex];
        if (terminalInfo.isTrainingPixel) {
            value *= make_float4(1.0f, 0.25f, 1.0f, 1.0f);
        }
        else {
            if (terminalInfo.isUnbiasedTile)
                value *= make_float4(0.25f, 1.0f, 0.25f, 1.0f);
        }
    }

    outputBuffer.write(launchIndex, value);
}
