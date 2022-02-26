#define PURE_CUDA
#include "neural_radiance_caching_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void resetNRCBuffers(
    uint32_t offsetToSelectTrainingPath,
    uint32_t frameIndex) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex >= plp.s->maxNumTrainingSuffixes)
        return;

    if (linearIndex == 0) {
        // Adjust tile size based on the number of training data generated in the previous frame.
        if (frameIndex > 0) {
            uint32_t prevNumTrainingData = *plp.s->numTrainingData;
            float r = std::sqrt(static_cast<float>(prevNumTrainingData) / maxNumTrainingDataPerFrame);
            uint2 curTileSize = *plp.s->tileSize;
            uint2 newTileSize = make_uint2(static_cast<uint32_t>(curTileSize.x * r),
                                           static_cast<uint32_t>(curTileSize.y * r));
            newTileSize = make_uint2(min(max(newTileSize.x, 1u), 128u),
                                     min(max(newTileSize.y, 1u), 128u));
            *plp.s->tileSize = newTileSize;
        }
        else {
            *plp.s->tileSize = make_uint2(8, 8);
        }

        *plp.s->numTrainingData = 0;
        *plp.s->offsetToSelectTrainingPath = offsetToSelectTrainingPath;
    }

    TrainingSuffixTerminalInfo terminalInfo;
    terminalInfo.prevVertexDataIndex = invalidVertexDataIndex;
    terminalInfo.hasQuery = false;
    plp.s->trainSuffixTerminalInfoBuffer[linearIndex] = terminalInfo;
}

CUDA_DEVICE_KERNEL void accumulateInferredRadianceValues() {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t numPixels = plp.s->imageSize.x * plp.s->imageSize.y;
    if (linearIndex >= numPixels)
        return;

    const TerminalInfo &terminalInfo = plp.s->inferenceTerminalInfoBuffer[linearIndex];
    if (!terminalInfo.hasQuery)
        return;

    float3 directCont = plp.s->perFrameContributionBuffer[linearIndex];
    float3 radiance = max(plp.s->inferredRadianceBuffer[linearIndex], make_float3(0.0f, 0.0f, 0.0f));
    float3 indirectCont = terminalInfo.alpha * radiance;
    float3 contribution = directCont + indirectCont;

    uint2 pixelIndex = make_uint2(linearIndex % plp.s->imageSize.x,
                                  linearIndex / plp.s->imageSize.x);
    float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.f->numAccumFrames > 0)
        prevColorResult = getXYZ(plp.s->beautyAccumBuffer.read(pixelIndex));
    float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
    float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * contribution;
    plp.s->beautyAccumBuffer.write(pixelIndex, make_float4(colorResult, 1.0f));
}

CUDA_DEVICE_KERNEL void propagateRadianceValues() {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex >= plp.s->maxNumTrainingSuffixes)
        return;

    const TrainingSuffixTerminalInfo &terminalInfo = plp.s->trainSuffixTerminalInfoBuffer[linearIndex];
    if (terminalInfo.prevVertexDataIndex == invalidVertexDataIndex)
        return;

    float3 contribution = make_float3(0.0f, 0.0f, 0.0f);
    if (terminalInfo.hasQuery) {
        uint32_t offset = plp.s->imageSize.x * plp.s->imageSize.y;
        contribution = max(plp.s->inferredRadianceBuffer[offset + linearIndex], make_float3(0.0f, 0.0f, 0.0f));
    }

    uint32_t lastTrainDataIndex = terminalInfo.prevVertexDataIndex;
    while (lastTrainDataIndex != invalidVertexDataIndex) {
        const TrainingVertexInfo &vertexInfo = plp.s->trainVertexInfoBuffer[lastTrainDataIndex];
        float3 &targetValue = plp.s->trainTargetBuffer[lastTrainDataIndex];
        float3 indirectCont = vertexInfo.localThroughput * contribution;
        contribution = targetValue + indirectCont;
        targetValue += contribution;
        
        lastTrainDataIndex = vertexInfo.prevVertexDataIndex;
    }
}

CUDA_DEVICE_KERNEL void permuteTrainingData() {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
}