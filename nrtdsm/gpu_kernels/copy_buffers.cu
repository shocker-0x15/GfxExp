#define PURE_CUDA
#include "../nrtdsm_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void copyToLinearBuffers(
    float4* linearColorBuffer,
    float4* linearAlbedoBuffer,
    float4* linearNormalBuffer,
    float2* linearMotionVectorBuffer) {
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



CUDA_DEVICE_KERNEL void visualizeToOutputBuffer(
    void* linearBuffer,
    BufferToDisplay bufferTypeToDisplay,
    float motionVectorOffset, float motionVectorScale,
    optixu::NativeBlockBuffer2D<float4> outputBuffer) {
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
    case BufferToDisplay::TexCoord: {
        const GBuffer0Elements gb0Elems = plp.s->GBuffer0[bufIdx].read(launchIndex);
        Point2D texCoord;
        const float bcB = decodeBarycentric(gb0Elems.qbcB);
        const float bcC = decodeBarycentric(gb0Elems.qbcC);
        if (gb0Elems.instSlot != 0xFFFFFFFF) {
            const InstanceData &inst = plp.s->instanceDataBufferArray[bufIdx][gb0Elems.instSlot];
            const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[gb0Elems.geomInstSlot];
            const Triangle &tri = geomInst.triangleBuffer[gb0Elems.primIndex];
            const Vertex &vA = geomInst.vertexBuffer[tri.index0];
            const Vertex &vB = geomInst.vertexBuffer[tri.index1];
            const Vertex &vC = geomInst.vertexBuffer[tri.index2];
            const float bcA = 1.0f - (bcB + bcC);
            texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;
        }
        else {
            texCoord.x = bcB;
            texCoord.y = bcC;
        }
        value.x = std::fmod(texCoord.x, 1.0f);
        value.y = std::fmod(texCoord.y, 1.0f);
        value.z = 0.5f * std::fmod((texCoord.x - value.x) / 10.0f, 1.0f)
            + 0.5f * std::fmod(texCoord.y - value.y, 2.0f);
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
