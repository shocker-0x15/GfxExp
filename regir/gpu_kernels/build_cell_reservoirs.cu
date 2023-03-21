#define PURE_CUDA
#include "../regir_shared.h"

using namespace shared;

CUDA_DEVICE_FUNCTION CUDA_INLINE RGB sampleIntensity(
    const Point3D &cellCenter, const Vector3D &halfCellSize, float minSquaredDistance,
    float uLight, bool sampleEnvLight, float uPos0, float uPos1,
    LightSample* lightSample, float* probDensity) {
    sampleLight<false>(
        cellCenter,
        uLight, sampleEnvLight, uPos0, uPos1,
        lightSample, probDensity);

    float dist2 = minSquaredDistance;
    float lpCos = 1;
    bool isOutsideCell =
        lightSample->atInfinity ||
        lightSample->position.x < cellCenter.x - halfCellSize.x ||
        lightSample->position.x > cellCenter.x + halfCellSize.x ||
        lightSample->position.y < cellCenter.y - halfCellSize.y ||
        lightSample->position.y > cellCenter.y + halfCellSize.y ||
        lightSample->position.z < cellCenter.z - halfCellSize.z ||
        lightSample->position.z > cellCenter.z + halfCellSize.z;
    if (isOutsideCell) {
        Vector3D shadowRayDir = lightSample->atInfinity ?
            Vector3D(lightSample->position) :
            (lightSample->position - cellCenter);
        // JP: 光源点を含む平面への垂直距離を求める。
        // EN: Calculate the perpendicular distance to a plane on which the light point is.
        float perpDistance = dot(-shadowRayDir, lightSample->normal);

        dist2 = shadowRayDir.sqLength();
        float dist = std::sqrt(dist2);

        /*
        JP: セルを囲むバウンディングスフィアが「光源点を含む平面が法線側につくる半空間」に完全に
            含まれる場合は通常どおりcos項と距離を計算する。
            逆に「法線と反対側の半空間」に完全に含まれる場合は、サンプルした光源点がセルに寄与することは
            ありえないのでcos項をゼロとして評価する。
            どちらとも言えない場合はcos項は常に1として評価する。
        EN: Calculate the cosine term and the distance as usual when the bounding sphere for the cell
            is completely encompassed in "the half-space spanned for the normal direction side of the plane
            on which the light point is".
            Contrary, when the sphere is completely encompassed in "the half-space for the opposite side of
            the normal", evaluate the cosine term as zero since the sampled light point will never
            contribute to the cell.
            Always evaluate the cosine term as 1 for the unknown case.
        */
        bool cellIsInValidHalfSpace = lpCos > minSquaredDistance || lightSample->atInfinity;
        bool cellIsInInvalidHalfSpace = lpCos < -minSquaredDistance;
        if (cellIsInValidHalfSpace)
            lpCos = perpDistance / dist;
        else if (cellIsInInvalidHalfSpace)
            lpCos = 0.0f;
        else
            ;// Unknown case
    }

    if (lpCos > 0.0f) {
        RGB Le = lightSample->emittance / Pi;
        RGB ret = Le * (lpCos / dist2);
        return ret;
    }
    else {
        return RGB(0.0f, 0.0f, 0.0f);
    }
}

template <bool useTemporalReuse>
CUDA_DEVICE_FUNCTION CUDA_INLINE void buildCellReservoirsAndTemporalReuse(uint32_t frameIndex) {
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
    Point3D cellCenter = plp.s->gridOrigin + Vector3D(
        (ix + 0.5f) * plp.s->gridCellSize.x,
        (iy + 0.5f) * plp.s->gridCellSize.y,
        (iz + 0.5f) * plp.s->gridCellSize.z);
    const Vector3D halfCellSize = 0.5f * plp.s->gridCellSize;
    const float minSquaredDistance = (0.5f * plp.s->gridCellSize).sqLength();

    uint32_t bufferIndex = plp.f->bufferIndex;
    RWBuffer<Reservoir<LightSample>> curReservoirs = plp.s->reservoirs[bufferIndex];
    RWBuffer<ReservoirInfo> curReservoirInfos = plp.s->reservoirInfos[bufferIndex];

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
            if (plp.s->lightInstDist.integral() > 0.0f) {
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
            else {
                sampleEnvLight = true;
            }
        }

        // JP: 候補サンプルを生成して、ターゲットPDFを計算する。
        //     ターゲットPDFは正規化されていなくても良い。
        // EN: Generate a candidate sample then calculate the target PDF for it.
        //     Target PDF doesn't require to be normalized.
        LightSample lightSample;
        float areaPDensity;
        RGB cont = sampleIntensity(
            cellCenter, halfCellSize, minSquaredDistance,
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
    // EN: Calculate the estimate of the reciprocal of the probability density that the current sample survives.
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
        RWBuffer<Reservoir<LightSample>> prevReservoirs = plp.s->reservoirs[prevBufferIndex];
        RWBuffer<ReservoirInfo> prevReservoirInfos = plp.s->reservoirInfos[prevBufferIndex];

        uint32_t selfStreamLength = reservoir.getStreamLength();
        if (recPDFEstimate == 0.0f)
            reservoir.initialize();
        uint32_t combinedStreamLength = selfStreamLength;
        uint32_t maxNumPrevSamples = 20 * selfStreamLength;

        // JP: 際限なく過去フレームで得たサンプルがウェイトを増やさないように、
        //     前フレームのストリーム長を、現在フレームのReservoirに対して20倍までに制限する。
        // EN: Limit the stream length of the previous frame by 20 times of that of the current frame
        //     in order to avoid a sample obtained in the past getting an unlimited weight.
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
        // EN: Calculate the estimate of the reciprocal of the probability density that the current sample survives.
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
