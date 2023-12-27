#include "../restir_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility);
}

static constexpr bool useMIS_RIS = true;



template <bool withTemporalRIS, bool useUnbiasedEstimator>
CUDA_DEVICE_FUNCTION CUDA_INLINE void performInitialAndTemporalRIS() {
    static_assert(withTemporalRIS || !useUnbiasedEstimator, "Invalid combination.");

    const int2 launchIndex = make_int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t curBufIdx = plp.f->bufferIndex;

    const GBuffer0Elements gb0Elems = plp.s->GBuffer0[curBufIdx].read(launchIndex);
    const uint32_t instSlot = gb0Elems.instSlot;
    const float bcB = decodeBarycentric(gb0Elems.qbcB);
    const float bcC = decodeBarycentric(gb0Elems.qbcC);

    if (instSlot == 0xFFFFFFFF)
        return;

    const PerspectiveCamera &camera = plp.f->camera;

    const GBuffer2Elements gb2Elems = plp.s->GBuffer2[curBufIdx].read(launchIndex);
    const GBuffer3Elements gb3Elems = plp.s->GBuffer3[curBufIdx].read(launchIndex);

    Point3D positionInWorld = gb2Elems.positionInWorld;
    const Normal3D geometricNormalInWorld = decodeNormal(gb2Elems.qGeometricNormal);

    const MaterialData &mat = plp.s->materialDataBuffer[gb3Elems.matSlot];

    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

    Vector3D vOut = camera.position - positionInWorld;
    const float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
    // Offsetting assumes BRDF.
    positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);
    const float dist = length(vOut);
    vOut /= dist;

    const Normal3D shadingNormalInWorld = decodeNormal(gb3Elems.qShadingNormal);
    const Vector3D shadingTangentInWorld = decodeVector(gb3Elems.qShadingTangent);
    const ReferenceFrame shadingFrame(shadingNormalInWorld, shadingTangentInWorld);
    const Vector3D vOutLocal = shadingFrame.toLocal(vOut);

    const Point2D texCoord = decodeTexCoords(gb3Elems.qTexCoord);
    BSDF bsdf;
    bsdf.setup(mat, texCoord, 0.0f);

    const uint32_t curResIndex = plp.currentReservoirIndex;
    Reservoir<LightSample> reservoir;
    reservoir.initialize(LightSample());

    // JP: Unshadowed ContributionをターゲットPDFとしてStreaming RISを実行。
    // EN: Perform streaming RIS with unshadowed contribution as the target PDF.
    float selectedTargetDensity = 0.0f;
    const uint32_t numCandidates = 1 << plp.f->log2NumCandidateSamples;
    for (int i = 0; i < numCandidates; ++i) {
        // JP: 環境光テクスチャーが設定されている場合は一定の確率でサンプルする。
        //     ダイバージェンスを抑えるために、ループの最初とそれ以外で環境光かそれ以外のサンプリングを分ける。
        // EN: Sample an environmental light texture with a fixed probability if it is set.
        //     Separate sampling from the environmental light and the others to
        //     the beginning of the loop and the rest to avoid divergence.
        float ul = rng.getFloat0cTo1o();
        float probToSampleCurLightType = 1.0f;
        bool sampleEnvLight = false;
        if (plp.s->envLightTexture && plp.f->enableEnvLight) {
            if (plp.s->lightInstDist.integral() > 0.0f) {
                float prob = min(max(probToSampleEnvLight * numCandidates - i, 0.0f), 1.0f);
                if (ul < prob) {
                    probToSampleCurLightType = probToSampleEnvLight;
                    ul = ul / prob;
                    sampleEnvLight = true;
                }
                else {
                    probToSampleCurLightType = 1.0f - probToSampleEnvLight;
                    ul = (ul - prob) / (1 - prob);
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
        float probDensity;
        sampleLight<false>(
            positionInWorld,
            ul, sampleEnvLight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
            &lightSample, &probDensity);
        const RGB cont = performDirectLighting<ReSTIRRayType, false>(
            positionInWorld, vOutLocal, shadingFrame, bsdf,
            lightSample);
        probDensity *= probToSampleCurLightType;
        const float targetDensity = convertToWeight(cont);

        // JP: 候補サンプル生成用のPDFとターゲットPDFは異なるためサンプルにはウェイトがかかる。
        // EN: The sample has a weight since the PDF to generate the candidate sample and the target PDF are
        //     different.
        const float weight = targetDensity / probDensity;
        if (reservoir.update(lightSample, weight, rng.getFloat0cTo1o()))
            selectedTargetDensity = targetDensity;
    }

    // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
    // EN: Calculate the estimate of the reciprocal of the probability density that the current sample survives.
    float recPDFEstimate = reservoir.getSumWeights() / (selectedTargetDensity * reservoir.getStreamLength());
    if (!stc::isfinite(recPDFEstimate)) {
        recPDFEstimate = 0.0f;
        selectedTargetDensity = 0.0f;
    }

    // JP: サンプルが遮蔽されていて寄与を持たない場合に、隣接ピクセルにサンプルが伝播しないよう、
    //     Reservoirのウェイトをゼロにする。
    // EN: Set the reservoir's weight to zero so that the occluded sample which has no contribution
    //     will not propagate to neighboring pixels.
    if (plp.f->reuseVisibility && selectedTargetDensity > 0.0f) {
        if (!evaluateVisibility<ReSTIRRayType>(positionInWorld, reservoir.getSample())) {
            recPDFEstimate = 0.0f;
            selectedTargetDensity = 0.0f;
        }
    }

    if constexpr (withTemporalRIS) {
        const uint32_t prevBufIdx = (curBufIdx + 1) % 2;
        const uint32_t prevResIndex = (curResIndex + 1) % 2;

        bool neighborIsSelected;
        if constexpr (useUnbiasedEstimator)
            neighborIsSelected = false;
        else
            (void)neighborIsSelected;
        const uint32_t selfStreamLength = reservoir.getStreamLength();
        if (recPDFEstimate == 0.0f)
            reservoir.initialize(LightSample());
        uint32_t combinedStreamLength = selfStreamLength;
        const uint32_t maxPrevStreamLength = 20 * selfStreamLength;

        const GBuffer1Elements gb1Elems = plp.s->GBuffer1[curBufIdx].read(launchIndex);
        const int2 nbCoord = make_int2(
            launchIndex.x + 0.5f - gb1Elems.motionVector.x,
            launchIndex.y + 0.5f - gb1Elems.motionVector.y);

        // JP: 隣接ピクセルのジオメトリ・マテリアルがあまりに異なる場合に候補サンプルを再利用すると
        //     バイアスが増えてしまうため、そのようなピクセルは棄却する。
        // EN: Reusing candidates from neighboring pixels with substantially different geometry/material
        //     leads to increased bias. Reject such a pixel.
        const bool acceptedNeighbor = testNeighbor<!useUnbiasedEstimator>(
            prevBufIdx, nbCoord, dist, shadingNormalInWorld);
        if (acceptedNeighbor) {
            const Reservoir<LightSample> neighbor = plp.s->reservoirBufferArray[prevResIndex][nbCoord];
            const ReservoirInfo neighborInfo = plp.s->reservoirInfoBufferArray[prevResIndex].read(nbCoord);

            // JP: 隣接ピクセルが持つ候補サンプルの「現在の」ピクセルにおける確率密度を計算する。
            // EN: Calculate the probability density at the "current" pixel of the candidate sample
            //     the neighboring pixel holds.
            // TODO: アニメーションやジッタリングがある場合には前フレームの対応ピクセルのターゲットPDFは
            //       変わってしまっているはず。この場合にはUnbiasedにするにはもうちょっと工夫がいる？
            const LightSample nbLightSample = neighbor.getSample();
            const RGB cont = performDirectLighting<ReSTIRRayType, false>(
                positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
            const float targetDensity = convertToWeight(cont);

            // JP: 際限なく過去フレームで得たサンプルがウェイトを増やさないように、
            //     前フレームのストリーム長を、現在フレームのReservoirに対して20倍までに制限する。
            // EN: Limit the stream length of the previous frame by 20 times of that of the current frame
            //     in order to avoid a sample obtained in the past getting an unlimited weight.
            const uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
            const float weight = targetDensity * neighborInfo.recPDFEstimate * nbStreamLength;
            if (reservoir.update(nbLightSample, weight, rng.getFloat0cTo1o())) {
                selectedTargetDensity = targetDensity;
                if constexpr (useUnbiasedEstimator)
                    neighborIsSelected = true;
            }

            combinedStreamLength += nbStreamLength;
        }
        reservoir.setStreamLength(combinedStreamLength);

        float weightForEstimate;
        if constexpr (useUnbiasedEstimator) {
            // JP: 推定関数をunbiasedとするための、生き残ったサンプルのウェイトを計算する。
            //     ここではReservoirの結合時とは逆に、サンプルは生き残った1つだが、
            //     ターゲットPDFは隣接ピクセルのものを評価する。
            // EN: Compute a weight for the survived sample to make the estimator unbiased.
            //     In contrast to the case where we combine reservoirs, the sample is only one survived and
            //     Evaluate target PDFs at the neighboring pixels here.
            const LightSample selectedLightSample = reservoir.getSample();

            float numWeight;
            float denomWeight;

            // JP: まずは現在のピクセルのターゲットPDFに対応する量を計算。
            // EN: First, calculate a quantity corresponding to the current pixel's target PDF.
            {
                const RGB cont = performDirectLighting<ReSTIRRayType, false>(
                    positionInWorld, vOutLocal, shadingFrame, bsdf, selectedLightSample);
                const float targetDensityForSelf = convertToWeight(cont);
                if constexpr (useMIS_RIS) {
                    numWeight = targetDensityForSelf;
                    denomWeight = targetDensityForSelf * selfStreamLength;
                }
                else {
                    numWeight = 1.0f;
                    denomWeight = 0.0f;
                    if (targetDensityForSelf > 0.0f)
                        denomWeight = selfStreamLength;
                }
            }

            // JP: 続いて隣接ピクセルのターゲットPDFに対応する量を計算。
            // EN: Next, calculate a quantity corresponding to the neighboring pixel's target PDF.
            if (acceptedNeighbor) {
                const GBuffer2Elements nbGb2Elems = plp.s->GBuffer2[prevBufIdx].read(nbCoord);
                const GBuffer3Elements nbGb3Elems = plp.s->GBuffer3[prevBufIdx].read(nbCoord);

                Point3D nbPositionInWorld = nbGb2Elems.positionInWorld;
                const Normal3D nbGeometricNormalInWorld = decodeNormal(nbGb2Elems.qGeometricNormal);
                const Vector3D nbVOut = normalize(plp.f->prevCamera.position - nbPositionInWorld);
                const float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
                nbPositionInWorld = offsetRayOrigin(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);

                const MaterialData &nbMat = plp.s->materialDataBuffer[nbGb3Elems.matSlot];
                const Point2D nbTexCoord = decodeTexCoords(nbGb3Elems.qTexCoord);
                BSDF nbBsdf;
                nbBsdf.setup(nbMat, nbTexCoord, 0.0f);

                const Normal3D nbShadingNormalInWorld = decodeNormal(nbGb3Elems.qShadingNormal);
                const Vector3D nbShadingTangentInWorld = decodeVector(nbGb3Elems.qShadingTangent);
                const ReferenceFrame nbShadingFrame(nbShadingNormalInWorld, nbShadingTangentInWorld);
                const Vector3D nbVOutLocal = nbShadingFrame.toLocal(nbVOut);

                const Reservoir<LightSample> neighbor = plp.s->reservoirBufferArray[prevResIndex][nbCoord];

                // JP: 際限なく過去フレームのウェイトが高まってしまうのを防ぐため、
                //     Temporal Reuseでは前フレームのストリーム長を現在のピクセルの20倍に制限する。
                // EN: To prevent the weight for previous frames to grow unlimitedly,
                //     limit the previous frame's weight by 20x of the current pixel's one.
                const RGB cont = performDirectLighting<ReSTIRRayType, false>(
                    nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, selectedLightSample);
                const float nbTargetDensity = convertToWeight(cont);
                const uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
                if constexpr (useMIS_RIS) {
                    denomWeight += nbTargetDensity * nbStreamLength;
                    if (neighborIsSelected)
                        numWeight = nbTargetDensity;
                }
                else {
                    if (nbTargetDensity > 0.0f)
                        denomWeight += nbStreamLength;
                }
            }

            weightForEstimate = numWeight / denomWeight;
        }
        else {
            weightForEstimate = 1.0f / reservoir.getStreamLength();
        }

        // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
        // EN: Calculate the estimate of the reciprocal of the probability density that the current sample survives.
        recPDFEstimate = weightForEstimate * reservoir.getSumWeights() / selectedTargetDensity;
        if (!stc::isfinite(recPDFEstimate)) {
            recPDFEstimate = 0.0f;
            selectedTargetDensity = 0.0f;
        }
    }

    ReservoirInfo reservoirInfo;
    reservoirInfo.recPDFEstimate = recPDFEstimate;
    reservoirInfo.targetDensity = selectedTargetDensity;

    plp.s->rngBuffer.write(launchIndex, rng);
    plp.s->reservoirBufferArray[curResIndex][launchIndex] = reservoir;
    plp.s->reservoirInfoBufferArray[curResIndex].write(launchIndex, reservoirInfo);
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(performInitialRIS)() {
    performInitialAndTemporalRIS<false, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(performInitialAndTemporalRISBiased)() {
    performInitialAndTemporalRIS<true, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(performInitialAndTemporalRISUnbiased)() {
    performInitialAndTemporalRIS<true, true>();
}



template <bool useUnbiasedEstimator>
CUDA_DEVICE_FUNCTION CUDA_INLINE void performSpatialRIS() {
    const int2 launchIndex = make_int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = plp.f->bufferIndex;

    const GBuffer0Elements gb0Elems = plp.s->GBuffer0[bufIdx].read(launchIndex);
    if (gb0Elems.instSlot == 0xFFFFFFFF)
        return;

    const GBuffer2Elements gb2Elems = plp.s->GBuffer2[bufIdx].read(launchIndex);
    const GBuffer3Elements gb3Elems = plp.s->GBuffer3[bufIdx].read(launchIndex);

    Point3D positionInWorld = gb2Elems.positionInWorld;
    const Normal3D geometricNormalInWorld = decodeNormal(gb2Elems.qGeometricNormal);

    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

    Vector3D vOut = plp.f->camera.position - positionInWorld;
    const float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
    // Offsetting assumes BRDF.
    positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);
    const float dist = length(vOut);
    vOut /= dist;

    const ReferenceFrame shadingFrame(
        decodeNormal(gb3Elems.qShadingNormal), decodeVector(gb3Elems.qShadingTangent));
    const Vector3D vOutLocal = shadingFrame.toLocal(vOut);

    const MaterialData &mat = plp.s->materialDataBuffer[gb3Elems.matSlot];
    const Point2D texCoord = decodeTexCoords(gb3Elems.qTexCoord);
    BSDF bsdf;
    bsdf.setup(mat, texCoord, 0.0f);

    const uint32_t srcResIndex = plp.currentReservoirIndex;
    const uint32_t dstResIndex = (srcResIndex + 1) % 2;

    Reservoir<LightSample> combinedReservoir;
    combinedReservoir.initialize(LightSample());
    float selectedTargetDensity = 0.0f;
    int32_t selectedNeighborIndex;
    if constexpr (useUnbiasedEstimator)
        selectedNeighborIndex = -1;
    else
        (void)selectedNeighborIndex;

    // JP: まず現在のピクセルのReservoirを結合する。
    // EN: First combine the reservoir for the current pixel.
    const Reservoir<LightSample> self = plp.s->reservoirBufferArray[srcResIndex][launchIndex];
    const ReservoirInfo selfResInfo = plp.s->reservoirInfoBufferArray[srcResIndex].read(launchIndex);
    if (selfResInfo.recPDFEstimate > 0.0f) {
        combinedReservoir = self;
        selectedTargetDensity = selfResInfo.targetDensity;
    }
    uint32_t combinedStreamLength = self.getStreamLength();

    for (int nIdx = 0; nIdx < plp.f->numSpatialNeighbors; ++nIdx) {
        // JP: 周辺ピクセルの座標をランダムに決定。
        // EN: Randomly determine the coordinates of a neighboring pixel.
        float radius = plp.f->spatialNeighborRadius;
        float deltaX, deltaY;
        if (plp.f->useLowDiscrepancyNeighbors) {
            Vector2D delta = plp.s->spatialNeighborDeltas[(plp.spatialNeighborBaseIndex + nIdx) % 1024];
            deltaX = radius * delta.x;
            deltaY = radius * delta.y;
        }
        else {
            radius *= std::sqrt(rng.getFloat0cTo1o());
            const float angle = 2 * Pi * rng.getFloat0cTo1o();
            deltaX = radius * std::cos(angle);
            deltaY = radius * std::sin(angle);
        }
        const int2 nbCoord = make_int2(
            launchIndex.x + 0.5f + deltaX,
            launchIndex.y + 0.5f + deltaY);

        // JP: 隣接ピクセルのジオメトリ・マテリアルがあまりに異なる場合に候補サンプルを再利用すると
        //     バイアスが増えてしまうため、そのようなピクセルは棄却する。
        // EN: Reusing candidates from neighboring pixels with substantially different geometry/material
        //     leads to increased bias. Reject such a pixel.
        const bool acceptedNeighbor = testNeighbor<!useUnbiasedEstimator>(
            bufIdx, nbCoord, dist, shadingFrame.normal)
            && (nbCoord.x != launchIndex.x || nbCoord.y != launchIndex.y);
        if (acceptedNeighbor) {
            const Reservoir<LightSample> neighbor = plp.s->reservoirBufferArray[srcResIndex][nbCoord];
            const ReservoirInfo neighborInfo = plp.s->reservoirInfoBufferArray[srcResIndex].read(nbCoord);

            // JP: 隣接ピクセルが持つ候補サンプルの「現在の」ピクセルにおける確率密度を計算する。
            // EN: Calculate the probability density at the "current" pixel of the candidate sample
            //     the neighboring pixel holds.
            const LightSample nbLightSample = neighbor.getSample();
            const RGB cont = performDirectLighting<ReSTIRRayType, false>(
                positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
            const float targetDensity = convertToWeight(cont);

            // JP: 隣接ピクセルと現在のピクセルではターゲットPDFが異なるためサンプルはウェイトを持つ。
            // EN: The sample has a weight since the target PDFs of the neighboring pixel and the current
            //     are the different.
            const uint32_t nbStreamLength = neighbor.getStreamLength();
            const float weight = targetDensity * neighborInfo.recPDFEstimate * nbStreamLength;
            if (combinedReservoir.update(nbLightSample, weight, rng.getFloat0cTo1o())) {
                selectedTargetDensity = targetDensity;
                if constexpr (useUnbiasedEstimator)
                    selectedNeighborIndex = nIdx;
            }

            combinedStreamLength += nbStreamLength;
        }
    }
    combinedReservoir.setStreamLength(combinedStreamLength);

    float weightForEstimate = 0.0f;
    if constexpr (useUnbiasedEstimator) {
        if (selectedTargetDensity > 0.0f) {
            // JP: 推定関数をunbiasedとするための、生き残ったサンプルのウェイトを計算する。
            //     ここではReservoirの結合時とは逆に、サンプルは生き残った1つだが、
            //     ターゲットPDFは隣接ピクセルのものを評価する。
            // EN: Compute a weight for the survived sample to make the estimator unbiased.
            //     In contrast to the case where we combine reservoirs, the sample is only one survived and
            //     Evaluate target PDFs at the neighboring pixels here.
            const LightSample selectedLightSample = combinedReservoir.getSample();

            float numWeight;
            float denomWeight;

            // JP: まずは現在のピクセルのターゲットPDFに対応する量を計算。
            // EN: First, calculate a quantity corresponding to the current pixel's target PDF.
            bool visibility = true;
            {
                RGB cont;
                if (plp.f->reuseVisibility)
                    cont = performDirectLighting<ReSTIRRayType, true>(
                        positionInWorld, vOutLocal, shadingFrame, bsdf, selectedLightSample);
                else
                    cont = performDirectLighting<ReSTIRRayType, false>(
                        positionInWorld, vOutLocal, shadingFrame, bsdf, selectedLightSample);
                const float targetDensityForSelf = convertToWeight(cont);
                if (plp.f->reuseVisibility)
                    visibility = targetDensityForSelf > 0.0f;
                if constexpr (useMIS_RIS) {
                    numWeight = targetDensityForSelf;
                    denomWeight = targetDensityForSelf * self.getStreamLength();
                }
                else {
                    numWeight = 1.0f;
                    denomWeight = 0.0f;
                    if (targetDensityForSelf > 0.0f)
                        denomWeight = self.getStreamLength();
                }
            }

            // JP: 続いて隣接ピクセルのターゲットPDFに対応する量を計算。
            // EN: Next, calculate quantities corresponding to the neighboring pixels' target PDFs.
            for (int nIdx = 0; nIdx < plp.f->numSpatialNeighbors; ++nIdx) {
                float radius = plp.f->spatialNeighborRadius;
                float deltaX, deltaY;
                if (plp.f->useLowDiscrepancyNeighbors) {
                    const Vector2D delta = plp.s->spatialNeighborDeltas[(plp.spatialNeighborBaseIndex + nIdx) % 1024];
                    deltaX = radius * delta.x;
                    deltaY = radius * delta.y;
                }
                else {
                    radius *= std::sqrt(rng.getFloat0cTo1o());
                    const float angle = 2 * Pi * rng.getFloat0cTo1o();
                    deltaX = radius * std::cos(angle);
                    deltaY = radius * std::sin(angle);
                }
                const int2 nbCoord = make_int2(
                    launchIndex.x + 0.5f + deltaX,
                    launchIndex.y + 0.5f + deltaY);

                const bool acceptedNeighbor =
                    (nbCoord.x >= 0 && nbCoord.x < plp.s->imageSize.x &&
                     nbCoord.y >= 0 && nbCoord.y < plp.s->imageSize.y)
                    && (nbCoord.x != launchIndex.x || nbCoord.y != launchIndex.y);
                if (acceptedNeighbor) {
                    const GBuffer0Elements nbGb0Elems = plp.s->GBuffer0[bufIdx].read(nbCoord);
                    if (nbGb0Elems.instSlot == 0xFFFFFFFF)
                        continue;

                    const GBuffer2Elements nbGb2Elems = plp.s->GBuffer2[bufIdx].read(nbCoord);
                    const GBuffer3Elements nbGb3Elems = plp.s->GBuffer3[bufIdx].read(nbCoord);

                    Point3D nbPositionInWorld = nbGb2Elems.positionInWorld;
                    const Normal3D nbGeometricNormalInWorld = decodeNormal(nbGb2Elems.qGeometricNormal);
                    const Vector3D nbVOut = normalize(plp.f->prevCamera.position - nbPositionInWorld);
                    const float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
                    nbPositionInWorld = offsetRayOrigin(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);

                    const MaterialData &nbMat = plp.s->materialDataBuffer[nbGb3Elems.matSlot];
                    const Point2D nbTexCoord = decodeTexCoords(nbGb3Elems.qTexCoord);
                    BSDF nbBsdf;
                    nbBsdf.setup(nbMat, nbTexCoord, 0.0f);

                    const ReferenceFrame nbShadingFrame(
                        decodeNormal(nbGb3Elems.qShadingNormal), decodeVector(nbGb3Elems.qShadingTangent));
                    const Vector3D nbVOutLocal = nbShadingFrame.toLocal(nbVOut);

                    const Reservoir<LightSample> neighbor = plp.s->reservoirBufferArray[srcResIndex][nbCoord];

                    // TODO: ウェイトの条件さえ満たしていれば、MISウェイト計算にはVisibilityはなくても良い？
                    //       要検討。
                    RGB cont;
                    if (plp.f->reuseVisibility)
                        cont = performDirectLighting<ReSTIRRayType, true>(
                            nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, selectedLightSample);
                    else
                        cont = performDirectLighting<ReSTIRRayType, false>(
                            nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, selectedLightSample);
                    const float nbTargetDensity = convertToWeight(cont);
                    const uint32_t nbStreamLength = neighbor.getStreamLength();
                    if constexpr (useMIS_RIS) {
                        denomWeight += nbTargetDensity * nbStreamLength;
                        if (nIdx == selectedNeighborIndex)
                            numWeight = nbTargetDensity;
                    }
                    else {
                        if (nbTargetDensity > 0.0f)
                            denomWeight += nbStreamLength;
                    }
                }
            }

            weightForEstimate = numWeight / denomWeight;
            if (plp.f->reuseVisibility && !visibility)
                weightForEstimate = 0.0f;
        }
    }
    else {
        weightForEstimate = 1.0f / combinedReservoir.getStreamLength();
    }

    // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
    // EN: Calculate the estimate of the reciprocal of the probability density that the current sample survives.
    ReservoirInfo reservoirInfo;
    reservoirInfo.recPDFEstimate = weightForEstimate * combinedReservoir.getSumWeights() / selectedTargetDensity;
    reservoirInfo.targetDensity = selectedTargetDensity;
    if (!stc::isfinite(reservoirInfo.recPDFEstimate)) {
        reservoirInfo.recPDFEstimate = 0.0f;
        reservoirInfo.targetDensity = 0.0f;
    }

    plp.s->rngBuffer.write(launchIndex, rng);
    plp.s->reservoirBufferArray[dstResIndex][launchIndex] = combinedReservoir;
    plp.s->reservoirInfoBufferArray[dstResIndex].write(launchIndex, reservoirInfo);
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(performSpatialRISBiased)() {
    performSpatialRIS<false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(performSpatialRISUnbiased)() {
    performSpatialRIS<true>();
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(shading)() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = plp.f->bufferIndex;

    const GBuffer0Elements gb0Elems = plp.s->GBuffer0[bufIdx].read(launchIndex);
    const GBuffer3Elements gb3Elems = plp.s->GBuffer3[bufIdx].read(launchIndex);

    const Point2D texCoord = decodeTexCoords(gb3Elems.qTexCoord);

    RGB contribution(0.01f, 0.01f, 0.01f);
    if (gb0Elems.instSlot != 0xFFFFFFFF) {
        const GBuffer2Elements gb2Elems = plp.s->GBuffer2[bufIdx].read(launchIndex);
        Point3D positionInWorld = gb2Elems.positionInWorld;
        const Normal3D geometricNormalInWorld = decodeNormal(gb2Elems.qGeometricNormal);

        const Vector3D vOut = normalize(plp.f->camera.position - positionInWorld);
        const float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
        // Offsetting assumes BRDF.
        positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);

        const ReferenceFrame shadingFrame(
            decodeNormal(gb3Elems.qShadingNormal), decodeVector(gb3Elems.qShadingTangent));
        const Vector3D vOutLocal = shadingFrame.toLocal(vOut);

        const MaterialData &mat = plp.s->materialDataBuffer[gb3Elems.matSlot];
        BSDF bsdf;
        bsdf.setup(mat, texCoord, 0.0f);

        const uint32_t curResIndex = plp.currentReservoirIndex;
        const Reservoir<LightSample> reservoir = plp.s->reservoirBufferArray[curResIndex][launchIndex];
        const ReservoirInfo reservoirInfo = plp.s->reservoirInfoBufferArray[curResIndex].read(launchIndex);

        // JP: 光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from a light source directly seeing.
        contribution = RGB(0.0f);
        if (vOutLocal.z > 0) {
            RGB emittance(0.0f, 0.0f, 0.0f);
            if (mat.emittance) {
                const float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
                emittance = RGB(getXYZ(texValue));
            }
            contribution += emittance / Pi;
        }

        // JP: 最終的に残ったサンプルとそのウェイトを使ってシェーディングを実行する。
        // EN: Perform shading using the sample survived in the end and its weight.
        const LightSample lightSample = reservoir.getSample();
        RGB directCont(0.0f);
        const float recPDFEstimate = reservoirInfo.recPDFEstimate;
        if (recPDFEstimate > 0 && stc::isfinite(recPDFEstimate)) {
            const bool visDone = plp.f->reuseVisibility &&
                (!plp.f->enableTemporalReuse || (plp.f->enableSpatialReuse && plp.f->useUnbiasedEstimator));
            if (visDone)
                directCont = performDirectLighting<ReSTIRRayType, false>(
                    positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample);
            else
                directCont = performDirectLighting<ReSTIRRayType, true>(
                    positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample);
        }

        contribution += recPDFEstimate * directCont;
    }
    else {
        // JP: 環境光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from the environmental light source directly seeing.
        if (plp.s->envLightTexture && plp.f->enableEnvLight) {
            const float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, texCoord.x, texCoord.y, 0.0f);
            const RGB luminance = plp.f->envLightPowerCoeff * RGB(getXYZ(texValue));
            contribution = luminance;
        }
    }

    RGB prevColorResult(0.0f, 0.0f, 0.0f);
    if (plp.f->numAccumFrames > 0)
        prevColorResult = RGB(getXYZ(plp.s->beautyAccumBuffer.read(launchIndex)));
    const float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
    const RGB colorResult = (1 - curWeight) * prevColorResult + curWeight * contribution;
    plp.s->beautyAccumBuffer.write(launchIndex, make_float4(colorResult.toNative(), 1.0f));
}
