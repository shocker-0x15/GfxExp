#define PURE_CUDA
#include "regir_shared.h"

using namespace shared;

// TODO: セルの中央だけのサンプリングだと、セルの中央が光源の裏側に回ってしまっている場合に、
//       寄与の可能性のあるサンプルを棄却してしまう。代表点をランダムに決定するなどで解決できそうだが、
//       PDFが毎回変わるのでそれを考慮する必要あり？
CUDA_DEVICE_FUNCTION float3 sampleIntensity(
    const float3 &shadingPoint, float minSquaredDistance,
    float uLight, bool sampleEnvLight, float uPos0, float uPos1,
    LightSample* lightSample, float* probDensity) {
    sampleLight<false>(
        shadingPoint,
        uLight, sampleEnvLight, uPos0, uPos1,
        lightSample, probDensity);

    float3 shadowRayDir = lightSample->atInfinity ?
        lightSample->position :
        (lightSample->position - shadingPoint);
    float dist2 = sqLength(shadowRayDir);
    float dist = std::sqrt(dist2);
    shadowRayDir /= dist;

    float lpCos = dot(-shadowRayDir, lightSample->normal);

    if (lpCos > 0) {
        float3 Le = lightSample->emittance / Pi;
        float3 ret = Le * (lpCos / dist2);
        return ret;
    }
    else {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
}

template <bool useTemporalReuse>
CUDA_DEVICE_FUNCTION void buildCellReservoirsAndTemporalReuse(uint32_t frameIndex) {
    uint32_t linearThreadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t cellLinearIndex = linearThreadIndex / kNumLightSlotsPerCell;
    uint32_t lastAccessFrameIndex = plp.s->lastAccessFrameIndices[cellLinearIndex];
    if (linearThreadIndex == 0)
        *plp.f->numActiveCells = 0;
    plp.s->perCellNumAccesses[cellLinearIndex] = 0;
    if (frameIndex - lastAccessFrameIndex > 8)
        return;

    //uint32_t lightSlotIndex = linearThreadIndex % kNumLightSlotsPerCell;
    uint32_t iz = cellLinearIndex / (plp.s->gridDimension.x * plp.s->gridDimension.y);
    uint32_t iy = (cellLinearIndex % (plp.s->gridDimension.x * plp.s->gridDimension.y)) / plp.s->gridDimension.x;
    uint32_t ix = cellLinearIndex % plp.s->gridDimension.x;
    float3 cellCenter = plp.s->gridOrigin + make_float3(
        (ix + 0.5f) * plp.s->gridCellSize.x,
        (iy + 0.5f) * plp.s->gridCellSize.y,
        (iz + 0.5f) * plp.s->gridCellSize.z);
    const float minSquaredDistance = sqLength(0.5f * plp.s->gridCellSize);

    uint32_t bufferIndex = plp.f->bufferIndex;
    Reservoir<LightSample>* curReservoirs = plp.s->reservoirs[bufferIndex];
    ReservoirInfo* curReservoirInfos = plp.s->reservoirInfos[bufferIndex];

    PCG32RNG rng = plp.s->lightSlotRngs[linearThreadIndex];

    float selectedTargetPDensity = 0.0f;
    Reservoir<LightSample> reservoir;
    reservoir.initialize();

    // JP: セルの代表点に到達する光度をターゲットPDFとしてStreaming RISを実行。
    // EN: Perform streaming RIS with luminous intensity reaching to a cell's representative point
    //     as the target PDF.
    const uint32_t numCandidates = 1 << plp.f->log2NumCandidatesPerLightSlot;
    for (int candIdx = 0; candIdx < numCandidates; ++candIdx) {
        // JP: 環境光テクスチャーが設定されている場合は一定の確率でサンプルする。
        //     ダイバージェンスを抑えるために、ループの最初とそれ以外で環境光かそれ以外のサンプリングを分ける。
        //     ただし、そもそもReGIRは2段階のRISにおいてVisibilityを一切考慮していないため、環境光は(特に高いエネルギーの場合)、
        //     Reservoir中のサンプルに無駄なものを増やしてしまい、むしろ分散が増える傾向にある。
        //     環境光のサンプリングは別で行うほうが良いかもしれない。
        // EN: Sample an environmental light texture with a fixed probability if it is set.
        //     Separate sampling from the environmental light and the others to
        //     the beginning of the loop and the rest to avoid divergence.
        //     However in the first place, ReGIR doesn't take visibility into account at all during two-stage RIS,
        //     therefore an environmental light (particularly with a high-energy case) tends to increase useless
        //     samples in reservoirs, resulting in high variance.
        //     Separated environmental light sampling may be preferred.
        float uLight = rng.getFloat0cTo1o();
        bool sampleEnvLight = false;
        float probToSampleCurLightType = 1.0f;
        if (plp.s->envLightTexture && plp.f->enableEnvLight) {
            float prob = min(max(probToSampleEnvLight * numCandidates - candIdx, 0.0f), 1.0f);
            if (uLight < prob) {
                probToSampleCurLightType = probToSampleEnvLight;
                uLight = uLight / prob;
                sampleEnvLight = true;
            }
            else {
                probToSampleCurLightType = 1.0f - probToSampleEnvLight;
                uLight = (uLight - prob) / (1 - prob);
            }
        }

        // JP: 候補サンプルを生成して、ターゲットPDFを計算する。
        //     ターゲットPDFは正規化されていなくても良い。
        // EN: Generate a candidate sample then calculate the target PDF for it.
        //     Target PDF doesn't require to be normalized.
        LightSample lightSample;
        float areaPDensity;
        float3 cont = sampleIntensity(
            cellCenter, minSquaredDistance,
            uLight, sampleEnvLight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
            &lightSample, &areaPDensity);
        areaPDensity *= probToSampleCurLightType;
        float targetPDensity = convertToWeight(cont);

        // JP: 候補サンプル生成用のPDFとターゲットPDFは異なるためサンプルにはウェイトがかかる。
        // EN: The sample has a weight since the PDF to generate the candidate sample and the target PDF are
        //     different.
        float weight = targetPDensity / areaPDensity;
        //if (ix == 7 && iz == 7 && iy == 0) {
        //    printf("%2u, %2u, %2u, %3u, %u: %g, %g\n", ix, iy, iz, lightSlotIndex, candIdx,
        //           areaPDensity, targetPDensity);
        //}
        if (reservoir.update(lightSample, weight, rng.getFloat0cTo1o()))
            selectedTargetPDensity = targetPDensity;
    }

    // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
    // EN: Calculate the estimate of the reciprocal of the probability density that the current sample suvives.
    float recPDFEstimate = reservoir.getSumWeights() / (selectedTargetPDensity * reservoir.getStreamLength());
    if (!isfinite(recPDFEstimate)) {
        recPDFEstimate = 0.0f;
        selectedTargetPDensity = 0.0f;
    }

    // JP: 元の文献では過去数フレーム分のストリーム長で正規化されたReservoirを保持して、それらを結合しているが、
    //     ここでは正規化は行わず現在フレームと過去フレームの累積Reservoirの2つを結合する。
    // EN: The original literature suggests using stream length normalized reservoirs of several previous
    //     frames, then combine them, but here it doesn't use normalization and combines two reservoirs, one from
    //     the current frame and the other is the accumulation of the previous frames.
    if constexpr (useTemporalReuse) {
        uint32_t prevBufferIndex = (bufferIndex + 1) % 2;
        const Reservoir<LightSample>* prevReservoirs = plp.s->reservoirs[prevBufferIndex];
        const ReservoirInfo* prevReservoirInfos = plp.s->reservoirInfos[prevBufferIndex];

        uint32_t selfStreamLength = reservoir.getStreamLength();
        if (recPDFEstimate == 0.0f)
            reservoir.initialize();
        uint32_t combinedStreamLength = selfStreamLength;
        uint32_t maxNumPrevSamples = 20 * selfStreamLength;

        // JP: 際限なく過去フレームで得たサンプルがウェイトを増やさないように、
        //     前フレームのストリーム長を、現在フレームのReservoirに対して20倍までに制限する。
        // EN: Limit the stream length of the previous frame by 20 times of that of the current frame
        //     in order to avoid a sample obtained in the past getting a unlimited weight.
        // TODO: 光源アニメーションがある場合には前フレームと今のフレームでターゲットPDFが異なるので
        //       ウェイトを調整するべき？
        const Reservoir<LightSample> &prevReservoir = prevReservoirs[linearThreadIndex];
        const ReservoirInfo &prevResInfo = prevReservoirInfos[linearThreadIndex];
        const LightSample &prevLightSample = prevReservoir.getSample();
        float prevTargetDensity = prevResInfo.targetDensity;
        uint32_t prevStreamLength = min(prevReservoir.getStreamLength(), maxNumPrevSamples);
        float lengthCorrection = static_cast<float>(prevStreamLength) / prevReservoir.getStreamLength();
        float weight = lengthCorrection * prevReservoir.getSumWeights(); // New target PDF and prev target PDF are the same here.
        if (reservoir.update(prevLightSample, weight, rng.getFloat0cTo1o()))
            selectedTargetPDensity = prevTargetDensity;
        combinedStreamLength += prevStreamLength;
        reservoir.setStreamLength(combinedStreamLength);

        // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
        // EN: Calculate the estimate of the reciprocal of the probability density that the current sample suvives.
        float weightForEstimate = 1.0f / reservoir.getStreamLength();
        recPDFEstimate = weightForEstimate * reservoir.getSumWeights() / selectedTargetPDensity;
        if (!isfinite(recPDFEstimate)) {
            recPDFEstimate = 0.0f;
            selectedTargetPDensity = 0.0f;
        }
    }

    ReservoirInfo resInfo;
    resInfo.recPDFEstimate = recPDFEstimate;
    resInfo.targetDensity = selectedTargetPDensity;

    plp.s->lightSlotRngs[linearThreadIndex] = rng;
    curReservoirs[linearThreadIndex] = reservoir;
    curReservoirInfos[linearThreadIndex] = resInfo;
}

CUDA_DEVICE_KERNEL void buildCellReservoirs(uint32_t frameIndex) {
    buildCellReservoirsAndTemporalReuse<false>(frameIndex);
}

CUDA_DEVICE_KERNEL void buildCellReservoirsAndTemporalReuse(uint32_t frameIndex) {
    buildCellReservoirsAndTemporalReuse<true>(frameIndex);
}

CUDA_DEVICE_KERNEL void updateLastAccessFrameIndices(uint32_t frameIndex) {
    // JP: 現在のフレーム中でアクセスされたセルにフレーム番号を記録する。
    // EN: Record the frame number to cells that accessed in the current frame.
    uint32_t linearThreadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t cellLinearIndex = linearThreadIndex;
    uint32_t perCellNumAccesses = plp.s->perCellNumAccesses[cellLinearIndex];
    if (perCellNumAccesses > 0)
        plp.s->lastAccessFrameIndices[cellLinearIndex] = frameIndex;

    uint32_t numActiveCellsInGroup = __popc(__ballot_sync(0xFFFFFFFF, perCellNumAccesses > 0));
    if (threadIdx.x == 0 && numActiveCellsInGroup > 0)
        atomicAdd(plp.f->numActiveCells, numActiveCellsInGroup);
}
