#define PURE_CUDA
#include "neural_radiance_caching_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void preprocessNRC(
    uint32_t offsetToSelectUnbiasedTile,
    uint32_t offsetToSelectTrainingPath,
    bool isNewSequence) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex >= plp.s->maxNumTrainingSuffixes)
        return;

    uint32_t prevBufIdx = (plp.f->bufferIndex + 1) % 2;
    uint32_t bufIdx = plp.f->bufferIndex;
    if (linearIndex == 0) {
        // JP: 前フレームで生成された訓練データ数に基づいてタイルサイズを調整する。
        // EN: Adjust tile size based on the number of training data generated in the previous frame.
        uint2 newTileSize;
        if (isNewSequence) {
            newTileSize = make_uint2(8, 8);
        }
        else {
            uint32_t prevNumTrainingData = *plp.s->numTrainingData[prevBufIdx];
            float r = std::sqrt(static_cast<float>(prevNumTrainingData) / numTrainingDataPerFrame);
            uint2 curTileSize = *plp.s->tileSize[prevBufIdx];
            newTileSize = make_uint2(static_cast<uint32_t>(curTileSize.x * r),
                                     static_cast<uint32_t>(curTileSize.y * r));
            newTileSize = make_uint2(min(max(newTileSize.x, 4u), 128u),
                                     min(max(newTileSize.y, 4u), 128u));
        }
        *plp.s->tileSize[bufIdx] = newTileSize;

        *plp.s->numTrainingData[bufIdx] = 0;
        *plp.s->offsetToSelectUnbiasedTile = offsetToSelectUnbiasedTile;
        *plp.s->offsetToSelectTrainingPath = offsetToSelectTrainingPath;

        plp.s->targetMinMax[bufIdx][0] = float3AsOrderedInt(make_float3(INFINITY)); //min
        plp.s->targetMinMax[bufIdx][1] = float3AsOrderedInt(make_float3(-INFINITY)); // max
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

    // JP: Rendering Pathはこの時点では終端前の各頂点における完全な推定値(直接光 + 間接光)の累積値と
    //     終端点における直接光の推定値を累積している。ネットワークから推定された終端点における間接光
    //     をスループットを乗じて累積することでピクセルを完成させる。
    //     パスがロシアンルーレットで終了した場合や無限遠に飛んだ場合はネットワークの推定は使われない。
    // EN: Each rendering path have accumulated complete estimates (direct + indirect light) at vertices
    //     preceded to the terminal and a direct light estimate at the terminal so far.
    //     Accumulate the predicted indirect light from the network multiplied by a throughput to
    //     complete a pixel.
    //     Network prediction is not used in the case where the path ended with Russian roulette or traced
    //     to infinity.
    float3 directCont = plp.s->perFrameContributionBuffer[linearIndex];
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    if (terminalInfo.hasQuery) {
        radiance = max(plp.s->inferredRadianceBuffer[linearIndex], make_float3(0.0f, 0.0f, 0.0f));

        if constexpr (useReflectanceFactorization) {
            const RadianceQuery &terminalQuery = plp.s->inferenceRadianceQueryBuffer[linearIndex];
            radiance *= (terminalQuery.diffuseReflectance + terminalQuery.specularReflectance);
        }
    }
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

        if constexpr (useReflectanceFactorization) {
            const RadianceQuery &terminalQuery = plp.s->inferenceRadianceQueryBuffer[offset + linearIndex];
            contribution *= (terminalQuery.diffuseReflectance + terminalQuery.specularReflectance);
        }
    }

    // JP: 各Training Vertexのローカルスループットを乗じながら再帰的にネットワークから与えられた輝度を
    //     伝播させることで訓練データを完成させる。
    // EN: Recursively propagate the radiance value from the network while multiplying a local throughput
    //     at each training vertex to complete training data.
    uint32_t lastTrainDataIndex = terminalInfo.prevVertexDataIndex;
    while (lastTrainDataIndex != invalidVertexDataIndex) {
        const TrainingVertexInfo &vertexInfo = plp.s->trainVertexInfoBuffer[lastTrainDataIndex];
        float3 &targetValue = plp.s->trainTargetBuffer[0][lastTrainDataIndex];
        float3 indirectCont = vertexInfo.localThroughput * contribution;
        contribution = targetValue + indirectCont;

        targetValue = contribution;

        if constexpr (useReflectanceFactorization) {
            const RadianceQuery &query = plp.s->trainRadianceQueryBuffer[0][lastTrainDataIndex];
            float3 refFactor = query.diffuseReflectance + query.specularReflectance;
            targetValue = safeDivide(targetValue, refFactor);
        }
        
        lastTrainDataIndex = vertexInfo.prevVertexDataIndex;
    }
}

CUDA_DEVICE_KERNEL void shuffleTrainingData() {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t bufIdx = plp.f->bufferIndex;

    uint32_t numTrainingData = *plp.s->numTrainingData[bufIdx];
    if (numTrainingData > 0) {
        LinearCongruentialGenerator &shuffler = plp.s->dataShufflerBuffer[linearIndex];
        static_assert((numTrainingDataPerFrame & (numTrainingDataPerFrame - 1)) == 0,
                      "The number of traing data is assumed to be the power of 2 here.");
        uint32_t dstIdx = shuffler.next() % numTrainingDataPerFrame;

        // JP: パストレーサーが生成したサンプル数が足りないときはラップアラウンドする。
        // EN: Wrap around for the case where the path tracer generates too few samples.
        uint32_t srcIdx = linearIndex % numTrainingData;
        RadianceQuery query = plp.s->trainRadianceQueryBuffer[0][srcIdx];
        float3 targetValue = plp.s->trainTargetBuffer[0][srcIdx];

        if (!query.isValid()) {
            printf("p: (%g, %g, %g), n: (%g, %g), v: (%g, %g), "
                   "r: %g, d: (%g, %g, %g), s: (%g, %g, %g)\n",
                   query.position.x, query.position.y, query.position.z,
                   query.normal_phi, query.normal_theta,
                   query.vOut_phi, query.vOut_theta,
                   query.roughness,
                   query.diffuseReflectance.x, query.diffuseReflectance.y, query.diffuseReflectance.z,
                   query.specularReflectance.x, query.specularReflectance.y, query.specularReflectance.z);
            query.position = make_float3(0.0f);
            query.normal_phi = 0.0f;
            query.normal_theta = 0.0f;
            query.vOut_phi = 0.0f;
            query.vOut_theta = 0.0f;
            query.roughness = 0.0f;
            query.diffuseReflectance = query.specularReflectance = make_float3(0.0f);
        }
        if (!allFinite(targetValue)) {
            printf("tgt: (%g, %g, %g)\n", targetValue.x, targetValue.y, targetValue.z);
            targetValue = make_float3(0.0f);
        }

        CUDA_SHARED_MEM uint32_t sm_pool[2 * sizeof(float3) / sizeof(uint32_t)];
        auto &sm_minRadiance = reinterpret_cast<float3AsOrderedInt &>(sm_pool[0]);
        auto &sm_maxRadiance = reinterpret_cast<float3AsOrderedInt &>(sm_pool[3]);
        if (threadIdx.x == 0) {
            sm_minRadiance = float3AsOrderedInt(make_float3(INFINITY));
            sm_maxRadiance = float3AsOrderedInt(make_float3(-INFINITY));
        }
        __syncthreads();
        atomicMin_float3_block(&sm_minRadiance, targetValue);
        atomicMax_float3_block(&sm_maxRadiance, targetValue);
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicMin_float3(&plp.s->targetMinMax[bufIdx][0], sm_minRadiance);
            atomicMax_float3(&plp.s->targetMinMax[bufIdx][1], sm_maxRadiance);
        }

        plp.s->trainRadianceQueryBuffer[1][dstIdx] = query;
        plp.s->trainTargetBuffer[1][dstIdx] = targetValue;
    }
    else {
        RadianceQuery query = {};
        plp.s->trainRadianceQueryBuffer[1][linearIndex] = query;
        plp.s->trainTargetBuffer[1][linearIndex] = make_float3(0.0f, 0.0f, 0.0f);
    }
}
