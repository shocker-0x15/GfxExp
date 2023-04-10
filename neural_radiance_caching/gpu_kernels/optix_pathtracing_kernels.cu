#include "../neural_radiance_caching_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility);
}



CUDA_DEVICE_FUNCTION CUDA_INLINE void convertToPolar(const float3 &dir, float* phi, float* theta) {
    float z = std::fmin(std::fmax(dir.z, -1.0f), 1.0f);
    *theta = std::acos(z);
    *phi = std::atan2(dir.y, dir.x);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void createRadianceQuery(
    const float3 &positionInWorld, const float3 &normalInWorld, const float3 &scatteredDirInWorld,
    float roughness, const float3 &diffuseReflectance, const float3 &specularReflectance,
    RadianceQuery* query) {
    float phi, theta;
    query->position = plp.s->sceneAABB->normalize(positionInWorld);
    convertToPolar(normalInWorld, &phi, &theta);
    query->normal_phi = phi;
    query->normal_theta = theta;
    convertToPolar(scatteredDirInWorld, &phi, &theta);
    query->vOut_phi = phi;
    query->vOut_theta = theta;
    query->roughness = 1 - std::exp(-roughness);
    query->diffuseReflectance = diffuseReflectance;
    query->specularReflectance = specularReflectance;
}

static constexpr bool useSolidAngleSampling = false;

CUDA_DEVICE_FUNCTION CUDA_INLINE float3 performNextEventEstimation(
    const float3 &shadingPoint, const float3 &vOutLocal, const ReferenceFrame &shadingFrame, const BSDF &bsdf,
    PCG32RNG &rng) {
    float uLight = rng.getFloat0cTo1o();
    bool selectEnvLight = false;
    float probToSampleCurLightType = 1.0f;
    if (plp.s->envLightTexture && plp.f->enableEnvLight) {
        if (plp.s->lightInstDist.integral() > 0.0f) {
            if (uLight < probToSampleEnvLight) {
                probToSampleCurLightType = probToSampleEnvLight;
                uLight /= probToSampleCurLightType;
                selectEnvLight = true;
            }
            else {
                probToSampleCurLightType = 1.0f - probToSampleEnvLight;
                uLight = (uLight - probToSampleEnvLight) / probToSampleCurLightType;
            }
        }
        else {
            selectEnvLight = true;
        }
    }
    LightSample lightSample;
    float areaPDensity;
    sampleLight<useSolidAngleSampling>(
        shadingPoint,
        uLight, selectEnvLight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &lightSample, &areaPDensity);
    areaPDensity *= probToSampleCurLightType;

    float3 shadowRay = lightSample.atInfinity ?
        lightSample.position :
        (lightSample.position - shadingPoint);
    float dist2 = sqLength(shadowRay);
    shadowRay /= std::sqrt(dist2);
    float3 vInLocal = shadingFrame.toLocal(shadowRay);
    float lpCos = std::fabs(dot(shadowRay, lightSample.normal));
    float bsdfPDensity = bsdf.evaluatePDF(vOutLocal, vInLocal) * lpCos / dist2;
    if (!isfinite(bsdfPDensity))
        bsdfPDensity = 0.0f;
    float lightPDensity = areaPDensity;
    float misWeight = pow2(lightPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
    float3 ret = make_float3(0.0f);
    if (areaPDensity > 0.0f)
        ret = performDirectLighting<PathTracingRayType, true>(
            shadingPoint, vOutLocal, shadingFrame, bsdf, lightSample) * (misWeight / areaPDensity);
    //if (!allFinite(ret)) {
    //    printf("mis: %g / %g, p:(%g, %g, %g), v:(%g, %g, %g)\n",
    //           misWeight, areaPDensity,
    //           shadingPoint.x, shadingPoint.y, shadingPoint.z,
    //           vOutLocal.x, vOutLocal.y, vOutLocal.z);
    //}

    return ret;
}



static constexpr bool useMIS_RIS = true;



template <bool withTemporalRIS, bool useUnbiasedEstimator>
CUDA_DEVICE_FUNCTION CUDA_INLINE void performInitialAndTemporalRIS() {
    static_assert(withTemporalRIS || !useUnbiasedEstimator, "Invalid combination.");

    int2 launchIndex = make_int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    uint32_t curBufIdx = plp.f->bufferIndex;

    GBuffer2 gBuffer2 = plp.s->GBuffer2[curBufIdx].read(launchIndex);
    uint32_t materialSlot = gBuffer2.materialSlot;

    if (materialSlot == 0xFFFFFFFF)
        return;

    GBuffer0 gBuffer0 = plp.s->GBuffer0[curBufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[curBufIdx].read(launchIndex);
    float3 positionInWorld = gBuffer0.positionInWorld;
    float3 shadingNormalInWorld = gBuffer1.normalInWorld;
    float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);

    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

    const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

    // TODO?: Use true geometric normal.
    float3 geometricNormalInWorld = shadingNormalInWorld;
    float3 vOut = plp.f->camera.position - positionInWorld;
    float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

    BSDF bsdf;
    bsdf.setup(mat, texCoord);
    ReferenceFrame shadingFrame(shadingNormalInWorld);
    positionInWorld = offsetRayOriginNaive(positionInWorld, frontHit * geometricNormalInWorld);
    float dist = length(vOut);
    vOut /= dist;
    float3 vOutLocal = shadingFrame.toLocal(vOut);

    uint32_t curResIndex = plp.currentReservoirIndex;
    Reservoir<LightSample> reservoir;
    reservoir.initialize();

    // JP: Unshadowed ContributionをターゲットPDFとしてStreaming RISを実行。
    // EN: Perform streaming RIS with unshadowed contribution as the target PDF.
    float selectedTargetDensity = 0.0f;
    uint32_t numCandidates = 1 << plp.f->log2NumCandidateSamples;
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
        float3 cont = performDirectLighting<PathTracingRayType, false>(
            positionInWorld, vOutLocal, shadingFrame, bsdf,
            lightSample);
        probDensity *= probToSampleCurLightType;
        float targetDensity = convertToWeight(cont);

        // JP: 候補サンプル生成用のPDFとターゲットPDFは異なるためサンプルにはウェイトがかかる。
        // EN: The sample has a weight since the PDF to generate the candidate sample and the target PDF are
        //     different.
        float weight = targetDensity / probDensity;
        if (reservoir.update(lightSample, weight, rng.getFloat0cTo1o()))
            selectedTargetDensity = targetDensity;
    }

    // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
    // EN: Calculate the estimate of the reciprocal of the probability density that the current sample survives.
    float recPDFEstimate = reservoir.getSumWeights() / (selectedTargetDensity * reservoir.getStreamLength());
    if (!isfinite(recPDFEstimate)) {
        recPDFEstimate = 0.0f;
        selectedTargetDensity = 0.0f;
    }

    // JP: サンプルが遮蔽されていて寄与を持たない場合に、隣接ピクセルにサンプルが伝播しないよう、
    //     Reservoirのウェイトをゼロにする。
    // EN: Set the reservoir's weight to zero so that the occluded sample which has no contribution
    //     will not propagate to neighboring pixels.
    if (plp.f->reuseVisibility && selectedTargetDensity > 0.0f) {
        if (!evaluateVisibility<PathTracingRayType>(positionInWorld, reservoir.getSample())) {
            recPDFEstimate = 0.0f;
            selectedTargetDensity = 0.0f;
        }
    }

    if constexpr (withTemporalRIS) {
        uint32_t prevBufIdx = (curBufIdx + 1) % 2;
        uint32_t prevResIndex = (curResIndex + 1) % 2;

        bool neighborIsSelected;
        if constexpr (useUnbiasedEstimator)
            neighborIsSelected = false;
        else
            (void)neighborIsSelected;
        uint32_t selfStreamLength = reservoir.getStreamLength();
        if (recPDFEstimate == 0.0f)
            reservoir.initialize();
        uint32_t combinedStreamLength = selfStreamLength;
        uint32_t maxPrevStreamLength = 20 * selfStreamLength;

        float2 motionVector = gBuffer2.motionVector;
        int2 nbCoord = make_int2(launchIndex.x + 0.5f - motionVector.x,
                                 launchIndex.y + 0.5f - motionVector.y);

        // JP: 隣接ピクセルのジオメトリ・マテリアルがあまりに異なる場合に候補サンプルを再利用すると
        //     バイアスが増えてしまうため、そのようなピクセルは棄却する。
        // EN: Reusing candidates from neighboring pixels with substantially different geometry/material
        //     leads to increased bias. Reject such a pixel.
        bool acceptedNeighbor = testNeighbor<!useUnbiasedEstimator>(prevBufIdx, nbCoord, dist, shadingNormalInWorld);
        if (acceptedNeighbor) {
            const Reservoir<LightSample> /*&*/neighbor = plp.s->reservoirBuffer[prevResIndex][nbCoord];
            const ReservoirInfo neighborInfo = plp.s->reservoirInfoBuffer[prevResIndex].read(nbCoord);

            // JP: 隣接ピクセルが持つ候補サンプルの「現在の」ピクセルにおける確率密度を計算する。
            // EN: Calculate the probability density at the "current" pixel of the candidate sample
            //     the neighboring pixel holds.
            // TODO: アニメーションやジッタリングがある場合には前フレームの対応ピクセルのターゲットPDFは
            //       変わってしまっているはず。この場合にはUnbiasedにするにはもうちょっと工夫がいる？
            LightSample nbLightSample = neighbor.getSample();
            float3 cont = performDirectLighting<PathTracingRayType, false>(
                positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
            float targetDensity = convertToWeight(cont);

            // JP: 際限なく過去フレームで得たサンプルがウェイトを増やさないように、
            //     前フレームのストリーム長を、現在フレームのReservoirに対して20倍までに制限する。
            // EN: Limit the stream length of the previous frame by 20 times of that of the current frame
            //     in order to avoid a sample obtained in the past getting an unlimited weight.
            uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
            float weight = targetDensity * neighborInfo.recPDFEstimate * nbStreamLength;
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
            LightSample selectedLightSample = reservoir.getSample();

            float numWeight;
            float denomWeight;

            // JP: まずは現在のピクセルのターゲットPDFに対応する量を計算。
            // EN: First, calculate a quantity corresponding to the current pixel's target PDF.
            {
                float3 cont = performDirectLighting<PathTracingRayType, false>(
                    positionInWorld, vOutLocal, shadingFrame, bsdf, selectedLightSample);
                float targetDensityForSelf = convertToWeight(cont);
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
                GBuffer2 nbGBuffer2 = plp.s->GBuffer2[prevBufIdx].read(nbCoord);
                uint32_t nbMaterialSlot = nbGBuffer2.materialSlot;
                if (nbMaterialSlot != 0xFFFFFFFF) {
                    GBuffer0 nbGBuffer0 = plp.s->GBuffer0[prevBufIdx].read(nbCoord);
                    GBuffer1 nbGBuffer1 = plp.s->GBuffer1[prevBufIdx].read(nbCoord);
                    float3 nbPositionInWorld = nbGBuffer0.positionInWorld;
                    float3 nbShadingNormalInWorld = nbGBuffer1.normalInWorld;
                    float2 nbTexCoord = make_float2(nbGBuffer0.texCoord_x, nbGBuffer1.texCoord_y);

                    const MaterialData &nbMat = plp.s->materialDataBuffer[nbMaterialSlot];

                    // TODO?: Use true geometric normal.
                    float3 nbGeometricNormalInWorld = nbShadingNormalInWorld;
                    float3 nbVOut = plp.f->camera.position - nbPositionInWorld;
                    float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

                    BSDF nbBsdf;
                    nbBsdf.setup(nbMat, nbTexCoord);
                    ReferenceFrame nbShadingFrame(nbShadingNormalInWorld);
                    nbPositionInWorld = offsetRayOriginNaive(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);
                    float nbDist = length(nbVOut);
                    nbVOut /= nbDist;
                    float3 nbVOutLocal = nbShadingFrame.toLocal(nbVOut);

                    const Reservoir<LightSample> /*&*/neighbor = plp.s->reservoirBuffer[prevResIndex][nbCoord];

                    // JP: 際限なく過去フレームのウェイトが高まってしまうのを防ぐため、
                    //     Temporal Reuseでは前フレームのストリーム長を現在のピクセルの20倍に制限する。
                    // EN: To prevent the weight for previous frames to grow unlimitedly,
                    //     limit the previous frame's weight by 20x of the current pixel's one.
                    float3 cont = performDirectLighting<PathTracingRayType, false>(
                        nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, selectedLightSample);
                    float nbTargetDensity = convertToWeight(cont);
                    uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
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
            }

            weightForEstimate = numWeight / denomWeight;
        }
        else {
            weightForEstimate = 1.0f / reservoir.getStreamLength();
        }

        // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
        // EN: Calculate the estimate of the reciprocal of the probability density that the current sample survives.
        recPDFEstimate = weightForEstimate * reservoir.getSumWeights() / selectedTargetDensity;
        if (!isfinite(recPDFEstimate)) {
            recPDFEstimate = 0.0f;
            selectedTargetDensity = 0.0f;
        }
    }

    ReservoirInfo reservoirInfo;
    reservoirInfo.recPDFEstimate = recPDFEstimate;
    reservoirInfo.targetDensity = selectedTargetDensity;

    plp.s->rngBuffer.write(launchIndex, rng);
    plp.s->reservoirBuffer[curResIndex][launchIndex] = reservoir;
    plp.s->reservoirInfoBuffer[curResIndex].write(launchIndex, reservoirInfo);
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
    int2 launchIndex = make_int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    uint32_t bufIdx = plp.f->bufferIndex;

    GBuffer2 gBuffer2 = plp.s->GBuffer2[bufIdx].read(launchIndex);
    uint32_t materialSlot = gBuffer2.materialSlot;

    if (materialSlot == 0xFFFFFFFF)
        return;

    GBuffer0 gBuffer0 = plp.s->GBuffer0[bufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[bufIdx].read(launchIndex);
    float3 positionInWorld = gBuffer0.positionInWorld;
    float3 shadingNormalInWorld = gBuffer1.normalInWorld;
    float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);

    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

    const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

    // TODO?: Use true geometric normal.
    float3 geometricNormalInWorld = shadingNormalInWorld;
    float3 vOut = plp.f->camera.position - positionInWorld;
    float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

    BSDF bsdf;
    bsdf.setup(mat, texCoord);
    ReferenceFrame shadingFrame(shadingNormalInWorld);
    positionInWorld = offsetRayOriginNaive(positionInWorld, frontHit * geometricNormalInWorld);
    float dist = length(vOut);
    vOut /= dist;
    float3 vOutLocal = shadingFrame.toLocal(vOut);

    uint32_t srcResIndex = plp.currentReservoirIndex;
    uint32_t dstResIndex = (srcResIndex + 1) % 2;

    Reservoir<LightSample> combinedReservoir;
    combinedReservoir.initialize();
    float selectedTargetDensity = 0.0f;
    int32_t selectedNeighborIndex;
    if constexpr (useUnbiasedEstimator)
        selectedNeighborIndex = -1;
    else
        (void)selectedNeighborIndex;

    // JP: まず現在のピクセルのReservoirを結合する。
    // EN: First combine the reservoir for the current pixel.
    const Reservoir<LightSample> /*&*/self = plp.s->reservoirBuffer[srcResIndex][launchIndex];
    const ReservoirInfo selfResInfo = plp.s->reservoirInfoBuffer[srcResIndex].read(launchIndex);
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
            float2 delta = plp.s->spatialNeighborDeltas[(plp.spatialNeighborBaseIndex + nIdx) % 1024];
            deltaX = radius * delta.x;
            deltaY = radius * delta.y;
        }
        else {
            radius *= std::sqrt(rng.getFloat0cTo1o());
            float angle = 2 * Pi * rng.getFloat0cTo1o();
            deltaX = radius * std::cos(angle);
            deltaY = radius * std::sin(angle);
        }
        int2 nbCoord = make_int2(launchIndex.x + 0.5f + deltaX,
                                    launchIndex.y + 0.5f + deltaY);

        // JP: 隣接ピクセルのジオメトリ・マテリアルがあまりに異なる場合に候補サンプルを再利用すると
        //     バイアスが増えてしまうため、そのようなピクセルは棄却する。
        // EN: Reusing candidates from neighboring pixels with substantially different geometry/material
        //     leads to increased bias. Reject such a pixel.
        bool acceptedNeighbor = testNeighbor<!useUnbiasedEstimator>(bufIdx, nbCoord, dist, shadingNormalInWorld);
        acceptedNeighbor &= nbCoord.x != launchIndex.x || nbCoord.y != launchIndex.y;
        if (acceptedNeighbor) {
            const Reservoir<LightSample> /*&*/neighbor = plp.s->reservoirBuffer[srcResIndex][nbCoord];
            const ReservoirInfo neighborInfo = plp.s->reservoirInfoBuffer[srcResIndex].read(nbCoord);

            // JP: 隣接ピクセルが持つ候補サンプルの「現在の」ピクセルにおける確率密度を計算する。
            // EN: Calculate the probability density at the "current" pixel of the candidate sample
            //     the neighboring pixel holds.
            LightSample nbLightSample = neighbor.getSample();
            float3 cont = performDirectLighting<PathTracingRayType, false>(
                positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
            float targetDensity = convertToWeight(cont);

            // JP: 隣接ピクセルと現在のピクセルではターゲットPDFが異なるためサンプルはウェイトを持つ。
            // EN: The sample has a weight since the target PDFs of the neighboring pixel and the current
            //     are the different.
            uint32_t nbStreamLength = neighbor.getStreamLength();
            float weight = targetDensity * neighborInfo.recPDFEstimate * nbStreamLength;
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
        // printf("unbiased restir rendering!\n");
        if (selectedTargetDensity > 0.0f) {
            // JP: 推定関数をunbiasedとするための、生き残ったサンプルのウェイトを計算する。
            //     ここではReservoirの結合時とは逆に、サンプルは生き残った1つだが、
            //     ターゲットPDFは隣接ピクセルのものを評価する。
            // EN: Compute a weight for the survived sample to make the estimator unbiased.
            //     In contrast to the case where we combine reservoirs, the sample is only one survived and
            //     Evaluate target PDFs at the neighboring pixels here.
            LightSample selectedLightSample = combinedReservoir.getSample();

            float numWeight;
            float denomWeight;

            // JP: まずは現在のピクセルのターゲットPDFに対応する量を計算。
            // EN: First, calculate a quantity corresponding to the current pixel's target PDF.
            bool visibility = true;
            {
                float3 cont;
                if (plp.f->reuseVisibility)
                    cont = performDirectLighting<PathTracingRayType, true>(
                        positionInWorld, vOutLocal, shadingFrame, bsdf, selectedLightSample);
                else
                    cont = performDirectLighting<PathTracingRayType, false>(
                        positionInWorld, vOutLocal, shadingFrame, bsdf, selectedLightSample);
                float targetDensityForSelf = convertToWeight(cont);
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
                    float2 delta = plp.s->spatialNeighborDeltas[(plp.spatialNeighborBaseIndex + nIdx) % 1024];
                    deltaX = radius * delta.x;
                    deltaY = radius * delta.y;
                }
                else {
                    radius *= std::sqrt(rng.getFloat0cTo1o());
                    float angle = 2 * Pi * rng.getFloat0cTo1o();
                    deltaX = radius * std::cos(angle);
                    deltaY = radius * std::sin(angle);
                }
                int2 nbCoord = make_int2(launchIndex.x + 0.5f + deltaX,
                                         launchIndex.y + 0.5f + deltaY);

                bool acceptedNeighbor =
                    nbCoord.x >= 0 && nbCoord.x < plp.s->imageSize.x &&
                    nbCoord.y >= 0 && nbCoord.y < plp.s->imageSize.y;
                acceptedNeighbor &= nbCoord.x != launchIndex.x || nbCoord.y != launchIndex.y;
                if (acceptedNeighbor) {
                    GBuffer2 nbGBuffer2 = plp.s->GBuffer2[bufIdx].read(nbCoord);

                    uint32_t nbMaterialSlot = nbGBuffer2.materialSlot;
                    if (nbMaterialSlot == 0xFFFFFFFF)
                        continue;

                    GBuffer0 nbGBuffer0 = plp.s->GBuffer0[bufIdx].read(nbCoord);
                    GBuffer1 nbGBuffer1 = plp.s->GBuffer1[bufIdx].read(nbCoord);
                    float3 nbPositionInWorld = nbGBuffer0.positionInWorld;
                    float3 nbShadingNormalInWorld = nbGBuffer1.normalInWorld;
                    float2 nbTexCoord = make_float2(nbGBuffer0.texCoord_x, nbGBuffer1.texCoord_y);

                    const MaterialData &nbMat = plp.s->materialDataBuffer[nbMaterialSlot];

                    // TODO?: Use true geometric normal.
                    float3 nbGeometricNormalInWorld = nbShadingNormalInWorld;
                    float3 nbVOut = plp.f->camera.position - nbPositionInWorld;
                    float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

                    BSDF nbBsdf;
                    nbBsdf.setup(nbMat, nbTexCoord);
                    ReferenceFrame nbShadingFrame(nbShadingNormalInWorld);
                    nbPositionInWorld = offsetRayOriginNaive(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);
                    float nbDist = length(nbVOut);
                    nbVOut /= nbDist;
                    float3 nbVOutLocal = nbShadingFrame.toLocal(nbVOut);

                    const Reservoir<LightSample> /*&*/neighbor = plp.s->reservoirBuffer[srcResIndex][nbCoord];

                    // TODO: ウェイトの条件さえ満たしていれば、MISウェイト計算にはVisibilityはなくても良い？
                    //       要検討。
                    float3 cont;
                    if (plp.f->reuseVisibility)
                        cont = performDirectLighting<PathTracingRayType, true>(
                            nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, selectedLightSample);
                    else
                        cont = performDirectLighting<PathTracingRayType, false>(
                            nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, selectedLightSample);
                    float nbTargetDensity = convertToWeight(cont);
                    uint32_t nbStreamLength = neighbor.getStreamLength();
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
    if (!isfinite(reservoirInfo.recPDFEstimate)) {
        reservoirInfo.recPDFEstimate = 0.0f;
        reservoirInfo.targetDensity = 0.0f;
    }

    plp.s->rngBuffer.write(launchIndex, rng);
    plp.s->reservoirBuffer[dstResIndex][launchIndex] = combinedReservoir;
    plp.s->reservoirInfoBuffer[dstResIndex].write(launchIndex, reservoirInfo);
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(performSpatialRISBiased)() {
    performSpatialRIS<false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(performSpatialRISUnbiased)() {
    performSpatialRIS<true>();
}



template <bool useNRC, bool useReSTIR>
CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_raygen_generic() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    uint32_t bufIdx = plp.f->bufferIndex;
    GBuffer0 gBuffer0 = plp.s->GBuffer0[bufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[bufIdx].read(launchIndex);
    GBuffer2 gBuffer2 = plp.s->GBuffer2[bufIdx].read(launchIndex);

    float3 positionInWorld = gBuffer0.positionInWorld;
    float3 shadingNormalInWorld = gBuffer1.normalInWorld;
    float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);
    uint32_t materialSlot = gBuffer2.materialSlot;

    const PerspectiveCamera &camera = plp.f->camera;

    uint32_t linearTileIndex;
    bool isTrainingPath;
    bool isUnbiasedTrainingTile;
    if constexpr (useNRC) {
        const uint2 tileSize = *plp.s->tileSize[bufIdx];
        const uint32_t numPixelsInTile = tileSize.x * tileSize.y;

        // JP: 動的サイズのタイルごとに1つトレーニングパスを選ぶ。
        // EN: choose a training path for each dynamic-sized tile.
        uint2 localIndex = launchIndex % tileSize;
        uint32_t localLinearIndex = localIndex.y * tileSize.x + localIndex.x;
        isTrainingPath = (localLinearIndex + *plp.s->offsetToSelectTrainingPath) % numPixelsInTile == 0;

        uint2 numTiles = (plp.s->imageSize + tileSize - 1) / tileSize;
        uint2 tileIndex = launchIndex / tileSize;
        linearTileIndex = tileIndex.y * numTiles.x + tileIndex.x;

        // JP: トレーニングパスの16本に1本はセルフトレーニングを使用しないUnbiasedパスとする。
        // EN: Make one path out of every 16 training paths not use self-training and unbiased.
        const uint2 tileGroupSize = make_uint2(4, 4);
        uint2 localTileIndex = tileIndex % tileGroupSize;
        uint32_t localLinearTileIndex = localTileIndex.y * tileGroupSize.x + localTileIndex.x;
        isUnbiasedTrainingTile = (localLinearTileIndex + *plp.s->offsetToSelectUnbiasedTile) % 16 == 0;
    }
    else {
        (void)linearTileIndex;
        (void)isTrainingPath;
        (void)isUnbiasedTrainingTile;
    }

    bool useEnvLight = plp.s->envLightTexture && plp.f->enableEnvLight;
    float3 contribution = make_float3(0.001f, 0.001f, 0.001f);
    bool renderingPathEndsWithCache = false;
    uint32_t pathLength = 1;
    if (materialSlot != 0xFFFFFFFF) {
        float3 alpha = make_float3(1.0f);
        float initImportance = sRGB_calcLuminance(alpha);
        PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

        // JP: 最初の交点におけるシェーディング。
        // EN: Shading on the first hit.
        float3 vIn;
        float dirPDensity;
        float primaryPathSpread;
        float3 localThroughput;
        uint32_t trainDataIndex;
        {
            const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

            // TODO?: Use true geometric normal.
            float3 geometricNormalInWorld = shadingNormalInWorld;
            float3 vOut = camera.position - positionInWorld;
            float primaryDist2 = sqLength(vOut);
            vOut /= std::sqrt(primaryDist2);
            float primaryDotVN = dot(vOut, geometricNormalInWorld);
            float frontHit = primaryDotVN >= 0.0f ? 1.0f : -1.0f;

            if constexpr (useNRC)
                primaryPathSpread = primaryDist2 / (4 * Pi * std::fabs(primaryDotVN));

            ReferenceFrame shadingFrame(shadingNormalInWorld);
            positionInWorld = offsetRayOriginNaive(positionInWorld, frontHit * geometricNormalInWorld);
            float3 vOutLocal = shadingFrame.toLocal(vOut);

            // JP: 光源を直接見ている場合の寄与を蓄積。
            // EN: Accumulate the contribution from a light source directly seeing.
            contribution = make_float3(0.0f);
            if (vOutLocal.z > 0 && mat.emittance) {
                float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
                float3 emittance = make_float3(texValue);
                contribution += alpha * emittance / Pi;
            }

            BSDF bsdf;
            bsdf.setup(mat, texCoord);


            float3 directContNEE = make_float3(0.0f);
            if constexpr (useReSTIR) {
                uint32_t curResIndex = plp.currentReservoirIndex;
                const Reservoir<LightSample> /*&*/reservoir = plp.s->reservoirBuffer[curResIndex][launchIndex];
                const ReservoirInfo reservoirInfo = plp.s->reservoirInfoBuffer[curResIndex].read(launchIndex);
                // JP: 最終的に残ったサンプルとそのウェイトを使ってシェーディングを実行する。
                // EN: Perform shading using the sample survived in the end and its weight.
                const LightSample &lightSample = reservoir.getSample();
                float recPDFEstimate = reservoirInfo.recPDFEstimate;
                if (recPDFEstimate > 0 && isfinite(recPDFEstimate)) {
                    bool visDone = plp.f->reuseVisibility &&
                        (!plp.f->enableTemporalReuse || (plp.f->enableSpatialReuse && plp.f->useUnbiasedEstimator));
                    if (visDone)
                        directContNEE = performDirectLighting<PathTracingRayType, false>(
                            positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample);
                    else
                        directContNEE = performDirectLighting<PathTracingRayType, true>(
                            positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample);
                }
                directContNEE *= recPDFEstimate;
                contribution += alpha * directContNEE;
            } else {
                // Next event estimation (explicit light sampling) on the first hit.
                directContNEE = performNextEventEstimation(
                    positionInWorld, vOutLocal, shadingFrame, bsdf, rng);
                contribution += alpha * directContNEE;
            }

            // generate a next ray.
            float3 vInLocal;
            localThroughput = bsdf.sampleThroughput(
                vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
                &vInLocal, &dirPDensity);
            alpha *= localThroughput;
            vIn = shadingFrame.fromLocal(vInLocal);

            if constexpr (useNRC) {
                // JP: 訓練データエントリーの確保。
                // EN: Allocate a training data entry.
                if (isTrainingPath) {
                    trainDataIndex = atomicAdd(plp.s->numTrainingData[bufIdx], 1u);

                    if (trainDataIndex < trainBufferSize) {
                        float roughness;
                        float3 diffuseReflectance, specularReflectance;
                        bsdf.getSurfaceParameters(
                            &diffuseReflectance, &specularReflectance, &roughness);

                        RadianceQuery radQuery;
                        createRadianceQuery(
                            positionInWorld, shadingFrame.normal, vOut,
                            roughness, diffuseReflectance, specularReflectance,
                            &radQuery);
                        plp.s->trainRadianceQueryBuffer[0][trainDataIndex] = radQuery;

                        TrainingVertexInfo vertInfo;
                        vertInfo.localThroughput = localThroughput;
                        vertInfo.prevVertexDataIndex = invalidVertexDataIndex;
                        vertInfo.pathLength = pathLength;
                        plp.s->trainVertexInfoBuffer[trainDataIndex] = vertInfo;

                        // JP: 現在の頂点に対する直接照明(NEE)によるScattered Radianceでターゲット値を初期化。
                        // EN: Initialize a target value by scattered radiance at the current vertex
                        //     by direct lighting (NEE).
                        plp.s->trainTargetBuffer[0][trainDataIndex] = directContNEE;
                        //if (!allFinite(directContNEE))
                        //    printf("NEE: (%g, %g, %g)\n",
                        //           directContNEE.x, directContNEE.y, directContNEE.z);
                    }
                    else {
                        trainDataIndex = invalidVertexDataIndex;
                    }
                }
            }
            else {
                (void)primaryPathSpread;
                (void)trainDataIndex;
            }
        }

        // Path extension loop
        PathTraceWriteOnlyPayload woPayload = {};
        PathTraceWriteOnlyPayload* woPayloadPtr = &woPayload;
        PathTraceReadWritePayload<useNRC> rwPayload = {};
        PathTraceReadWritePayload<useNRC>* rwPayloadPtr = &rwPayload;
        rwPayload.rng = rng;
        rwPayload.initImportance = initImportance;
        rwPayload.alpha = alpha;
        rwPayload.contribution = contribution;
        rwPayload.prevDirPDensity = dirPDensity;
        if constexpr (useNRC) {
            rwPayload.linearTileIndex = linearTileIndex;
            rwPayload.primaryPathSpread = primaryPathSpread;
            rwPayload.curSqrtPathSpread = 0.0f;
            rwPayload.prevLocalThroughput = localThroughput;
            rwPayload.prevTrainDataIndex = trainDataIndex;
            rwPayload.renderingPathEndsWithCache = false;
            rwPayload.isTrainingPath = isTrainingPath;
            rwPayload.isUnbiasedTrainingTile = isUnbiasedTrainingTile;
            rwPayload.trainingSuffixEndsWithCache = false;
        }
        rwPayload.pathLength = pathLength;
        float3 rayOrg = positionInWorld;
        float3 rayDir = vIn;
        while (true) {
            bool isValidSampling = rwPayload.prevDirPDensity > 0.0f && isfinite(rwPayload.prevDirPDensity);
            if (!isValidSampling)
                break;

            ++rwPayload.pathLength;
            // JP: 通常のパストレーシングとNRCを正しく比較するには(特に通常のパストレーシングにおいて)
            //     反射回数制限を解除する必要がある。
            // EN: Disabling the limitation in the number of bounces (particularly for the base path tracing)
            //     is required to properly compare the base path tracing and NRC.
            if (rwPayload.pathLength >= plp.f->maxPathLength && plp.f->maxPathLength > 0)
                rwPayload.maxLengthTerminate = true;
            rwPayload.terminate = true;

            constexpr PathTracingRayType pathTraceRayType = useNRC ?
                PathTracingRayType::NRC : PathTracingRayType::Baseline;
            PathTraceRayPayloadSignature<useNRC>::trace(
                plp.f->travHandle, rayOrg, rayDir,
                0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
                pathTraceRayType, maxNumRayTypes, pathTraceRayType,
                woPayloadPtr, rwPayloadPtr);
            if (rwPayload.terminate)
                break;
            rayOrg = woPayload.nextOrigin;
            rayDir = woPayload.nextDirection;
        }
        contribution = rwPayload.contribution;

        plp.s->rngBuffer.write(launchIndex, rwPayload.rng);

        if constexpr (useNRC) {
            renderingPathEndsWithCache = rwPayload.renderingPathEndsWithCache;
            pathLength = rwPayload.pathLength;
            if (rwPayload.isTrainingPath && !rwPayload.trainingSuffixEndsWithCache) {
                TrainingSuffixTerminalInfo terminalInfo;
                terminalInfo.prevVertexDataIndex = rwPayload.prevTrainDataIndex;
                terminalInfo.hasQuery = false;
                terminalInfo.pathLength = rwPayload.pathLength;
                plp.s->trainSuffixTerminalInfoBuffer[rwPayload.linearTileIndex] = terminalInfo;
            }
        }
    }
    else {
        // JP: 環境光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from the environmental light source directly seeing.
        if (useEnvLight) {
            float u = texCoord.x, v = texCoord.y;
            float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
            float3 luminance = plp.f->envLightPowerCoeff * make_float3(texValue);
            contribution = luminance;
        }
    }

    if constexpr (useNRC) {
        uint32_t linearIndex = launchIndex.y * plp.s->imageSize.x + launchIndex.x;

        // JP: 無限遠にレイが飛んだか、ロシアンルーレットによってパストレースが完了したケース。
        // EN: When a ray goes infinity or the path ends with Russain roulette.
        if (!renderingPathEndsWithCache) {
            TerminalInfo terminalInfo;
            terminalInfo.alpha = make_float3(0.0f, 0.0f, 0.0f);
            terminalInfo.pathLength = pathLength;
            terminalInfo.hasQuery = false;
            terminalInfo.isTrainingPixel = isTrainingPath;
            terminalInfo.isUnbiasedTile = isUnbiasedTrainingTile;
            plp.s->inferenceTerminalInfoBuffer[linearIndex] = terminalInfo;
        }

        plp.s->perFrameContributionBuffer[linearIndex] = contribution;
    }
    else {
        (void)renderingPathEndsWithCache;
        (void)pathLength;

        float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
        if (plp.f->numAccumFrames > 0)
            prevColorResult = getXYZ(plp.s->beautyAccumBuffer.read(launchIndex));
        float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
        float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * contribution;
        plp.s->beautyAccumBuffer.write(launchIndex, make_float4(colorResult, 1.0f));
    }
}

template <bool useNRC>
CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_closestHit_generic() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    uint32_t bufIdx = plp.f->bufferIndex;

    auto sbtr = HitGroupSBTRecordData::get();
    const InstanceData &inst = plp.f->instanceDataBuffer[optixGetInstanceId()];
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];

    PathTraceWriteOnlyPayload* woPayload;
    PathTraceReadWritePayload<useNRC>* rwPayload;
    PathTraceRayPayloadSignature<useNRC>::get(&woPayload, &rwPayload);
    PCG32RNG &rng = rwPayload->rng;

    const float3 rayOrigin = optixGetWorldRayOrigin();

    auto hp = HitPointParameter::get();
    float3 positionInWorld;
    float3 shadingNormalInWorld;
    float3 texCoord0DirInWorld;
    float3 geometricNormalInWorld;
    float2 texCoord;
    float hypAreaPDensity;
    computeSurfacePoint<true, useSolidAngleSampling>(
        inst, geomInst, hp.primIndex, hp.b1, hp.b2,
        rayOrigin,
        &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
        &geometricNormalInWorld, &texCoord, &hypAreaPDensity);

    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    float3 vOut = normalize(-optixGetWorldRayDirection());
    float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

    ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
    float3 modLocalNormal = mat.readModifiedNormal(mat.normal, mat.normalDimInfo, texCoord);
    if (plp.f->enableBumpMapping)
        applyBumpMapping(modLocalNormal, &shadingFrame);
    positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);
    float3 vOutLocal = shadingFrame.toLocal(vOut);
    //if (!allFinite(vOutLocal)) {
    //    printf("(%g, %g, %g), (%g, %g, %g), (%g, %g, %g)\n",
    //           shadingFrame.tangent.x, shadingFrame.tangent.y, shadingFrame.tangent.z,
    //           shadingFrame.bitangent.x, shadingFrame.bitangent.y, shadingFrame.bitangent.z,
    //           shadingFrame.normal.x, shadingFrame.normal.y, shadingFrame.normal.z);
    //}

    float dist2 = squaredDistance(rayOrigin, positionInWorld);
    if constexpr (useNRC)
        rwPayload->curSqrtPathSpread += std::sqrt(dist2 / (rwPayload->prevDirPDensity * std::fabs(vOutLocal.z)));

    // Implicit Light Sampling
    if (vOutLocal.z > 0 && mat.emittance) {
        float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
        float3 emittance = make_float3(texValue);
        float lightPDensity = hypAreaPDensity * dist2 / vOutLocal.z;
        float bsdfPDensity = rwPayload->prevDirPDensity;
        float misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
        float3 directContImplicit = emittance * (misWeight / Pi);
        rwPayload->contribution += rwPayload->alpha * directContImplicit;

        if constexpr (useNRC) {
            // JP: 1つ前の頂点に対する直接照明(Implicit)によるScattered Radianceをターゲット値に加算。
            // EN: Accumulate scattered radiance at the previous vertex by direct lighting (implicit)
            //     to the target value.
            if (rwPayload->isTrainingPath && rwPayload->prevTrainDataIndex != invalidVertexDataIndex) {
                plp.s->trainTargetBuffer[0][rwPayload->prevTrainDataIndex] +=
                    rwPayload->prevLocalThroughput * directContImplicit;
                //if (!allFinite(rwPayload->prevLocalThroughput) ||
                //    !allFinite(directContImplicit))
                //    printf("Implicit: (%g, %g, %g), (%g, %g, %g)\n",
                //           rwPayload->prevLocalThroughput.x,
                //           rwPayload->prevLocalThroughput.y,
                //           rwPayload->prevLocalThroughput.z,
                //           directContImplicit.x,
                //           directContImplicit.y,
                //           directContImplicit.z);
            }
        }
    }

    // Russian roulette
    bool performRR = true;
    bool terminatedByRR = false;
    float recContinueProb = 1.0f;
    if constexpr (useNRC) {
        if (rwPayload->isTrainingPath)
            performRR = rwPayload->pathLength > 2;
    }
    if (performRR) {
        float continueProb = std::fmin(sRGB_calcLuminance(rwPayload->alpha) / rwPayload->initImportance, 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb || rwPayload->maxLengthTerminate) {
            if constexpr (useNRC) {
                if (rwPayload->renderingPathEndsWithCache &&
                    rwPayload->isTrainingPath && rwPayload->isUnbiasedTrainingTile)
                    return;
                terminatedByRR = true;
            }
            else {
                return;
            }
        }
        recContinueProb = 1.0f / continueProb;
    }

    BSDF bsdf;
    bsdf.setup(mat, texCoord);

    if constexpr (useNRC) {
        bool endsWithCache = false;
        bool pathIsSpreadEnough =
            pow2(rwPayload->curSqrtPathSpread) > pathTerminationFactor * rwPayload->primaryPathSpread;
        endsWithCache |= pathIsSpreadEnough;
        if (rwPayload->renderingPathEndsWithCache &&
            rwPayload->isTrainingPath && rwPayload->isUnbiasedTrainingTile)
            endsWithCache = false;

        if (endsWithCache) {
            uint32_t linearIndex = launchIndex.y * plp.s->imageSize.x + launchIndex.x;

            float roughness;
            float3 diffuseReflectance, specularReflectance;
            bsdf.getSurfaceParameters(
                &diffuseReflectance, &specularReflectance, &roughness);

            // JP: Radianceクエリーのための情報を記録する。
            // EN: Store information for radiance query.
            RadianceQuery radQuery;
            createRadianceQuery(
                positionInWorld, shadingFrame.normal, vOut,
                roughness, diffuseReflectance, specularReflectance,
                &radQuery);

            if (!rwPayload->renderingPathEndsWithCache) {
                plp.s->inferenceRadianceQueryBuffer[linearIndex] = radQuery;

                TerminalInfo terminalInfo;
                terminalInfo.alpha = rwPayload->alpha;
                terminalInfo.pathLength = rwPayload->pathLength;
                terminalInfo.hasQuery = true;
                terminalInfo.isTrainingPixel = rwPayload->isTrainingPath;
                terminalInfo.isUnbiasedTile = rwPayload->isUnbiasedTrainingTile;
                plp.s->inferenceTerminalInfoBuffer[linearIndex] = terminalInfo;

                rwPayload->renderingPathEndsWithCache = true;
                if (rwPayload->isTrainingPath)
                    rwPayload->curSqrtPathSpread = 0;
                else
                    return;
            }
            else {
                // JP: 訓練データバッファーがフルの場合は既にTraining Suffixは終了したことになっている。
                // EN: The training suffix should have been ended if the training data buffer is full.
                if (!rwPayload->trainingSuffixEndsWithCache) {
                    uint32_t offset = plp.s->imageSize.x * plp.s->imageSize.y;
                    plp.s->inferenceRadianceQueryBuffer[offset + rwPayload->linearTileIndex] = radQuery;

                    // JP: 直前のTraining VertexへのリンクとともにTraining Suffixを終了させる。
                    // EN: Finish the training suffix with the link to the previous training vertex.
                    TrainingSuffixTerminalInfo terminalInfo;
                    terminalInfo.prevVertexDataIndex = rwPayload->prevTrainDataIndex;
                    terminalInfo.hasQuery = true;
                    terminalInfo.pathLength = rwPayload->pathLength;
                    plp.s->trainSuffixTerminalInfoBuffer[rwPayload->linearTileIndex] = terminalInfo;

                    rwPayload->trainingSuffixEndsWithCache = true;
                }
                return;
            }
        }
    }

    if constexpr (useNRC) {
        if (terminatedByRR)
            return;
    }
    rwPayload->alpha *= recContinueProb;
    if constexpr (useNRC) {
        if (rwPayload->isTrainingPath && rwPayload->prevTrainDataIndex != invalidVertexDataIndex)
            plp.s->trainVertexInfoBuffer[rwPayload->prevTrainDataIndex].localThroughput *= recContinueProb;
    }

    // Next Event Estimation (Explicit Light Sampling)
    float3 directContNEE = performNextEventEstimation(
        positionInWorld, vOutLocal, shadingFrame, bsdf, rng);
    rwPayload->contribution += rwPayload->alpha * directContNEE;

    // generate a next ray.
    float3 vInLocal;
    float dirPDensity;
    float3 localThroughput = bsdf.sampleThroughput(
        vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &vInLocal, &dirPDensity);
    rwPayload->alpha *= localThroughput;
    float3 vIn = shadingFrame.fromLocal(vInLocal);

    woPayload->nextOrigin = positionInWorld;
    woPayload->nextDirection = vIn;
    rwPayload->prevDirPDensity = dirPDensity;
    if constexpr (useNRC)
        rwPayload->prevLocalThroughput = localThroughput;
    rwPayload->terminate = false;

    if constexpr (useNRC) {
        // JP: 訓練データエントリーの確保。
        // EN: Allocate a training data entry.
        if (rwPayload->isTrainingPath && !rwPayload->trainingSuffixEndsWithCache) {
            uint32_t trainDataIndex = atomicAdd(plp.s->numTrainingData[bufIdx], 1u);

            // TODO?: 訓練データ数の正確な推定のためにtrainingSuffixEndsWithCacheのチェックをここに持ってくる？

            float roughness;
            float3 diffuseReflectance, specularReflectance;
            bsdf.getSurfaceParameters(
                &diffuseReflectance, &specularReflectance, &roughness);

            RadianceQuery radQuery;
            createRadianceQuery(
                positionInWorld, shadingFrame.normal, vOut,
                roughness, diffuseReflectance, specularReflectance,
                &radQuery);

            if (trainDataIndex < trainBufferSize) {
                plp.s->trainRadianceQueryBuffer[0][trainDataIndex] = radQuery;

                // JP: ローカルスループットと前のTraining Vertexへのリンクを記録。
                // EN: Record the local throughput and the link to the previous training vertex.
                TrainingVertexInfo vertInfo;
                vertInfo.localThroughput = localThroughput;
                vertInfo.prevVertexDataIndex = rwPayload->prevTrainDataIndex;
                vertInfo.pathLength = rwPayload->pathLength;
                plp.s->trainVertexInfoBuffer[trainDataIndex] = vertInfo;

                // JP: 現在の頂点に対する直接照明(NEE)によるScattered Radianceでターゲット値を初期化。
                // EN: Initialize a target value by scattered radiance at the current vertex by
                //     direct lighting (NEE).
                plp.s->trainTargetBuffer[0][trainDataIndex] = directContNEE;
                //if (!allFinite(directContNEE))
                //    printf("NEE: (%g, %g, %g)\n",
                //           directContNEE.x, directContNEE.y, directContNEE.z);

                rwPayload->prevTrainDataIndex = trainDataIndex;
            }
            // JP: 訓練データがバッファーを溢れた場合は強制的にTraining Suffixを終了させる。
            // EN: Forcefully end the training suffix if the training data buffer become full.
            else {
                uint32_t offset = plp.s->imageSize.x * plp.s->imageSize.y;
                plp.s->inferenceRadianceQueryBuffer[offset + rwPayload->linearTileIndex] = radQuery;

                TrainingSuffixTerminalInfo terminalInfo;
                terminalInfo.prevVertexDataIndex = rwPayload->prevTrainDataIndex;
                terminalInfo.hasQuery = true;
                terminalInfo.pathLength = rwPayload->pathLength;
                plp.s->trainSuffixTerminalInfoBuffer[rwPayload->linearTileIndex] = terminalInfo;

                rwPayload->trainingSuffixEndsWithCache = true;
            }
        }
    }
}

template <bool useNRC>
CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_miss_generic() {
    if (!plp.s->envLightTexture || !plp.f->enableEnvLight)
        return;

    PathTraceReadWritePayload<useNRC>* rwPayload;
    PathTraceRayPayloadSignature<useNRC>::get(nullptr, &rwPayload);

    float3 rayDir = normalize(optixGetWorldRayDirection());
    float posPhi, theta;
    toPolarYUp(rayDir, &posPhi, &theta);

    float phi = posPhi + plp.f->envLightRotation;
    phi = phi - floorf(phi / (2 * Pi)) * 2 * Pi;
    float2 texCoord = make_float2(phi / (2 * Pi), theta / Pi);

    // Implicit Light Sampling
    float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, texCoord.x, texCoord.y, 0.0f);
    float3 luminance = plp.f->envLightPowerCoeff * make_float3(texValue);
    float uvPDF = plp.s->envLightImportanceMap.evaluatePDF(texCoord.x, texCoord.y);
    float hypAreaPDensity = uvPDF / (2 * Pi * Pi * std::sin(theta));
    float lightPDensity = probToSampleEnvLight * hypAreaPDensity;
    float bsdfPDensity = rwPayload->prevDirPDensity;
    float misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
    float3 directContImplicit = misWeight * luminance;
    rwPayload->contribution += rwPayload->alpha * directContImplicit;

    if constexpr (useNRC) {
        // JP: 1つ前の頂点に対する直接照明(Implicit)によるScattered Radianceをターゲット値に加算。
        // EN: Accumulate scattered radiance at the previous vertex by direct lighting (implicit)
        //     to the target value.
        if (rwPayload->isTrainingPath && rwPayload->prevTrainDataIndex != invalidVertexDataIndex) {
            plp.s->trainTargetBuffer[0][rwPayload->prevTrainDataIndex] +=
                rwPayload->prevLocalThroughput * directContImplicit;
            //if (!allFinite(rwPayload->prevLocalThroughput) ||
            //    !allFinite(directContImplicit))
            //    printf("Implicit: (%g, %g, %g), (%g, %g, %g)\n",
            //           rwPayload->prevLocalThroughput.x,
            //           rwPayload->prevLocalThroughput.y,
            //           rwPayload->prevLocalThroughput.z,
            //           directContImplicit.x,
            //           directContImplicit.y,
            //           directContImplicit.z);
        }
    }
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTraceBaseline)() {
    pathTrace_raygen_generic<false, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTraceBaselineReSTIR)() {
    pathTrace_raygen_generic<false, true>();
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTraceBaseline)() {
    pathTrace_closestHit_generic<false>();
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(pathTraceBaseline)() {
    pathTrace_miss_generic<false>();
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTraceNRC)() {
    pathTrace_raygen_generic<true, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTraceNRCReSTIR)() {
    pathTrace_raygen_generic<true, true>();
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTraceNRC)() {
    pathTrace_closestHit_generic<true>();
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(pathTraceNRC)() {
    pathTrace_miss_generic<true>();
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(visualizePrediction)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    uint32_t linearIndex = launchIndex.y * plp.s->imageSize.x + launchIndex.x;

    uint32_t bufIdx = plp.f->bufferIndex;
    GBuffer0 gBuffer0 = plp.s->GBuffer0[bufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[bufIdx].read(launchIndex);
    GBuffer2 gBuffer2 = plp.s->GBuffer2[bufIdx].read(launchIndex);

    float3 positionInWorld = gBuffer0.positionInWorld;
    float3 shadingNormalInWorld = gBuffer1.normalInWorld;
    float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);
    uint32_t materialSlot = gBuffer2.materialSlot;

    const PerspectiveCamera &camera = plp.f->camera;

    if (materialSlot != 0xFFFFFFFF) {
        const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

        // TODO?: Use true geometric normal.
        float3 geometricNormalInWorld = shadingNormalInWorld;
        float3 vOut = camera.position - positionInWorld;
        float primaryDist2 = sqLength(vOut);
        vOut /= std::sqrt(primaryDist2);
        float primaryDotVN = dot(vOut, geometricNormalInWorld);
        float frontHit = primaryDotVN >= 0.0f ? 1.0f : -1.0f;

        ReferenceFrame shadingFrame(shadingNormalInWorld);
        positionInWorld = offsetRayOriginNaive(positionInWorld, frontHit * geometricNormalInWorld);

        BSDF bsdf;
        bsdf.setup(mat, texCoord);

        float roughness;
        float3 diffuseReflectance, specularReflectance;
        bsdf.getSurfaceParameters(
            &diffuseReflectance, &specularReflectance, &roughness);

        RadianceQuery radQuery;
        createRadianceQuery(
            positionInWorld, shadingFrame.normal, vOut,
            roughness, diffuseReflectance, specularReflectance,
            &radQuery);

        plp.s->inferenceRadianceQueryBuffer[linearIndex] = radQuery;
    }
    else {
        //// JP: 環境光源を直接見ている場合の寄与を蓄積。
        //// EN: Accumulate the contribution from the environmental light source directly seeing.
        //if (useEnvLight) {
        //    float u = texCoord.x, v = texCoord.y;
        //    float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
        //    float3 luminance = plp.f->envLightPowerCoeff * make_float3(texValue);
        //    contribution = luminance;
        //}
    }

    TerminalInfo terminalInfo;
    terminalInfo.alpha = make_float3(1.0f);
    terminalInfo.pathLength = 1;
    terminalInfo.hasQuery = materialSlot != 0xFFFFFFFF;
    terminalInfo.isTrainingPixel = false;
    terminalInfo.isUnbiasedTile = false;
    plp.s->inferenceTerminalInfoBuffer[linearIndex] = terminalInfo;
}
