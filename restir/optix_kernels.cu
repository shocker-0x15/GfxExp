#include "restir_shared.h"

using namespace shared;

struct HitPointParameter {
    float b1, b2;
    int32_t primIndex;

    CUDA_DEVICE_FUNCTION static HitPointParameter get() {
        HitPointParameter ret;
        float2 bc = optixGetTriangleBarycentrics();
        ret.b1 = bc.x;
        ret.b2 = bc.y;
        ret.primIndex = optixGetPrimitiveIndex();
        return ret;
    }
};

struct HitGroupSBTRecordData {
    GeometryInstanceData geomInstData;

    CUDA_DEVICE_FUNCTION static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    optixu::setPayloads<VisibilityRayPayloadSignature>(&visibility);
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(setupGBuffers)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    const PerspectiveCamera &camera = plp.f->camera;
    float jx = 0.5f;
    float jy = 0.5f;
    if (plp.f->enableJittering) {
        // JP: ジッターをかけると現状の実装ではUnbiased要件を満たさないかもしれない。要検討。
        // EN: Jittering may break the requirements for unbiasedness with the current implementation.
        //     Need more consideration.
        PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);
        jx = rng.getFloat0cTo1o();
        jy = rng.getFloat0cTo1o();
        plp.s->rngBuffer.write(launchIndex, rng);
    }
    float x = (launchIndex.x + jx) / plp.s->imageSize.x;
    float y = (launchIndex.y + jy) / plp.s->imageSize.y;
    float vh = 2 * std::tan(camera.fovY * 0.5f);
    float vw = camera.aspect * vh;

    float3 origin = camera.position;
    float3 direction = normalize(camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    HitPointParams hitPointParams;
    hitPointParams.positionInWorld = make_float3(NAN);
    hitPointParams.prevPositionInWorld = make_float3(NAN);
    hitPointParams.normalInWorld = make_float3(NAN);
    hitPointParams.texCoord = make_float2(NAN);
    hitPointParams.materialSlot = 0xFFFFFFFF;

    PickInfo pickInfo = {};

    HitPointParams* hitPointParamsPtr = &hitPointParams;
    PickInfo* pickInfoPtr = &pickInfo;
    optixu::trace<PrimaryRayPayloadSignature>(
        plp.f->travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        hitPointParamsPtr, pickInfoPtr);



    float2 curRasterPos = make_float2(launchIndex.x + 0.5f, launchIndex.y + 0.5f);
    float2 prevRasterPos =
        plp.f->prevCamera.calcScreenPosition(hitPointParams.prevPositionInWorld)
        * make_float2(plp.s->imageSize.x, plp.s->imageSize.y);
    float2 motionVector = curRasterPos - prevRasterPos;
    if (plp.f->resetFlowBuffer || isnan(hitPointParams.prevPositionInWorld.x))
        motionVector = make_float2(0.0f, 0.0f);

    GBuffer0 gBuffer0;
    gBuffer0.positionInWorld = hitPointParams.positionInWorld;
    gBuffer0.texCoord_x = hitPointParams.texCoord.x;
    GBuffer1 gBuffer1;
    gBuffer1.normalInWorld = hitPointParams.normalInWorld;
    gBuffer1.texCoord_y = hitPointParams.texCoord.y;
    GBuffer2 gBuffer2;
    gBuffer2.motionVector = motionVector;
    gBuffer2.materialSlot = hitPointParams.materialSlot;

    uint32_t bufIdx = plp.f->bufferIndex;
    plp.s->GBuffer0[bufIdx].write(launchIndex, gBuffer0);
    plp.s->GBuffer1[bufIdx].write(launchIndex, gBuffer1);
    plp.s->GBuffer2[bufIdx].write(launchIndex, gBuffer2);

    if (launchIndex.x == plp.f->mousePosition.x &&
        launchIndex.y == plp.f->mousePosition.y)
        *plp.f->pickInfo = pickInfo;

    // JP: デノイザーに必要な情報を出力。
    // EN: Output information required for the denoiser.
    float3 firstHitNormal = transpose(camera.orientation) * hitPointParams.normalInWorld;
    firstHitNormal.x *= -1;
    float3 prevAlbedoResult = make_float3(0.0f, 0.0f, 0.0f);
    float3 prevNormalResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.f->numAccumFrames > 0) {
        prevAlbedoResult = getXYZ(plp.s->albedoAccumBuffer.read(launchIndex));
        prevNormalResult = getXYZ(plp.s->normalAccumBuffer.read(launchIndex));
    }
    float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
    float3 albedoResult = (1 - curWeight) * prevAlbedoResult + curWeight * hitPointParams.albedo;
    float3 normalResult = (1 - curWeight) * prevNormalResult + curWeight * firstHitNormal;
    plp.s->albedoAccumBuffer.write(launchIndex, make_float4(albedoResult, 1.0f));
    plp.s->normalAccumBuffer.write(launchIndex, make_float4(normalResult, 1.0f));
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(setupGBuffers)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    auto sbtr = HitGroupSBTRecordData::get();
    const InstanceData &inst = plp.f->instanceDataBuffer[optixGetInstanceId()];
    const GeometryInstanceData &geomInst = sbtr.geomInstData;

    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    optixu::getPayloads<PrimaryRayPayloadSignature>(&hitPointParams, &pickInfo);

    auto hp = HitPointParameter::get();
    float3 positionInWorld;
    float3 prevPositionInWorld;
    float3 shadingNormalInWorld;
    float3 texCoord0DirInWorld;
    //float3 geometricNormalInWorld;
    float2 texCoord;
    {
        const Triangle &tri = geomInst.triangleBuffer[hp.primIndex];
        const Vertex &v0 = geomInst.vertexBuffer[tri.index0];
        const Vertex &v1 = geomInst.vertexBuffer[tri.index1];
        const Vertex &v2 = geomInst.vertexBuffer[tri.index2];
        float b1 = hp.b1;
        float b2 = hp.b2;
        float b0 = 1 - (b1 + b2);
        float3 localP = b0 * v0.position + b1 * v1.position + b2 * v2.position;
        shadingNormalInWorld = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
        texCoord0DirInWorld = b0 * v0.texCoord0Dir + b1 * v1.texCoord0Dir + b2 * v2.texCoord0Dir;
        //geometricNormalInWorld = cross(v1.position - v0.position, v2.position - v0.position);
        texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

        positionInWorld = optixTransformPointFromObjectToWorldSpace(localP);
        prevPositionInWorld = inst.prevTransform * localP;
        shadingNormalInWorld = normalize(optixTransformNormalFromObjectToWorldSpace(shadingNormalInWorld));
        texCoord0DirInWorld = normalize(optixTransformVectorFromObjectToWorldSpace(texCoord0DirInWorld));
        //geometricNormalInWorld = normalize(optixTransformNormalFromObjectToWorldSpace(geometricNormalInWorld));
        if (!allFinite(shadingNormalInWorld)) {
            shadingNormalInWorld = make_float3(0, 0, 1);
            texCoord0DirInWorld = make_float3(1, 0, 0);
        }
    }

    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    BSDF bsdf;
    bsdf.setup(mat, texCoord);
    ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
    float3 modLocalNormal = mat.readModifiedNormal(mat.normal, texCoord, mat.normalDimension);
    if (plp.f->enableBumpMapping)
        applyBumpMapping(modLocalNormal, &shadingFrame);
    float3 vOut = -optixGetWorldRayDirection();
    float3 vOutLocal = shadingFrame.toLocal(normalize(vOut));

    hitPointParams->albedo = bsdf.evaluateDHReflectanceEstimate(vOutLocal);
    hitPointParams->positionInWorld = positionInWorld;
    hitPointParams->prevPositionInWorld = prevPositionInWorld;
    hitPointParams->normalInWorld = shadingFrame.normal;
    hitPointParams->texCoord = texCoord;
    hitPointParams->materialSlot = geomInst.materialSlot;

    // JP: マウスが乗っているピクセルの情報を出力する。
    // EN: Export the information of the pixel on which the mouse is.
    if (launchIndex.x == plp.f->mousePosition.x &&
        launchIndex.y == plp.f->mousePosition.y) {
        pickInfo->hit = true;
        pickInfo->instSlot = optixGetInstanceId();
        pickInfo->geomInstSlot = geomInst.geomInstSlot;
        pickInfo->matSlot = geomInst.materialSlot;
        pickInfo->primIndex = hp.primIndex;
        pickInfo->positionInWorld = positionInWorld;
        pickInfo->normalInWorld = shadingFrame.normal;
        pickInfo->albedo = hitPointParams->albedo;
        float3 emittance = make_float3(0.0f, 0.0f, 0.0f);
        if (mat.emittance) {
            float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
            emittance = make_float3(texValue);
        }
        pickInfo->emittance = emittance;
    }
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(setupGBuffers)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    float3 vOut = -optixGetWorldRayDirection();
    float3 p = -vOut;

    float posPhi, posTheta;
    toPolarYUp(p, &posPhi, &posTheta);

    float phi = posPhi + plp.f->envLightRotation;

    float u = phi / (2 * Pi);
    u -= floorf(u);
    float v = posTheta / Pi;

    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    optixu::getPayloads<PrimaryRayPayloadSignature>(&hitPointParams, &pickInfo);

    hitPointParams->albedo = make_float3(0.0f, 0.0f, 0.0f);
    hitPointParams->positionInWorld = p;
    hitPointParams->prevPositionInWorld = p;
    hitPointParams->normalInWorld = vOut;
    hitPointParams->texCoord = make_float2(u, v);
    hitPointParams->materialSlot = 0xFFFFFFFF;

    // JP: マウスが乗っているピクセルの情報を出力する。
    // EN: Export the information of the pixel on which the mouse is.
    if (launchIndex.x == plp.f->mousePosition.x &&
        launchIndex.y == plp.f->mousePosition.y) {
        pickInfo->hit = true;
        pickInfo->instSlot = 0xFFFFFFFF;
        pickInfo->geomInstSlot = 0xFFFFFFFF;
        pickInfo->matSlot = 0xFFFFFFFF;
        pickInfo->primIndex = 0xFFFFFFFF;
        pickInfo->positionInWorld = p;
        pickInfo->albedo = make_float3(0.0f, 0.0f, 0.0f);
        float3 emittance = make_float3(0.0f, 0.0f, 0.0f);
        if (plp.s->envLightTexture && plp.f->enableEnvLight) {
            float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
            emittance = make_float3(texValue);
            emittance *= Pi * plp.f->envLightPowerCoeff;
        }
        pickInfo->emittance = emittance;
        pickInfo->normalInWorld = vOut;
    }
}



template <bool testGeometry>
CUDA_DEVICE_FUNCTION bool testNeighbor(
    uint32_t nbBufIdx, int2 nbCoord, float dist, const float3 &normalInWorld) {
    if (nbCoord.x < 0 || nbCoord.x >= plp.s->imageSize.x ||
        nbCoord.y < 0 || nbCoord.y >= plp.s->imageSize.y)
        return false;

    GBuffer2 nbGBuffer2 = plp.s->GBuffer2[nbBufIdx].read(nbCoord);
    if (nbGBuffer2.materialSlot == 0xFFFFFFFF)
        return false;

    if constexpr (testGeometry) {
        GBuffer0 nbGBuffer0 = plp.s->GBuffer0[nbBufIdx].read(nbCoord);
        GBuffer1 nbGBuffer1 = plp.s->GBuffer1[nbBufIdx].read(nbCoord);
        float3 nbPositionInWorld = nbGBuffer0.positionInWorld;
        float3 nbNormalInWorld = nbGBuffer1.normalInWorld;
        float nbDist = length(plp.f->camera.position - nbPositionInWorld);
        if (abs(nbDist - dist) / dist > 0.1f || dot(normalInWorld, nbNormalInWorld) < 0.9f)
            return false;
    }

    return true;
}

static constexpr bool useMIS_RIS = true;



template <bool withTemporalRIS, bool useUnbiasedEstimator>
CUDA_DEVICE_FUNCTION void performInitialAndTemporalRIS() {
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

        // JP: 候補サンプルを生成して、ターゲットPDFを計算する。
        //     ターゲットPDFは正規化されていなくても良い。
        // EN: Generate a candidate sample then calculate the target PDF for it.
        //     Target PDF doesn't require to be normalized.
        LightSample lightSample;
        float probDensity;
        sampleLight(ul, sampleEnvLight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
                    &lightSample, &probDensity);
        float3 cont = performDirectLighting<false>(
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
    // EN: Calculate the estimate of the reciprocal of the probability density that the current sample suvives.
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
        if (!evaluateVisibility(positionInWorld, reservoir.getSample())) {
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
            float3 cont = performDirectLighting<false>(
                positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
            float targetDensity = convertToWeight(cont);

            // JP: 際限なく過去フレームで得たサンプルがウェイトを増やさないように、
            //     前フレームのストリーム長を、現在フレームのReservoirに対して20倍までに制限する。
            // EN: Limit the stream length of the previous frame by 20 times of that of the current frame
            //     in order to avoid a sample obtained in the past getting a unlimited weight.
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
                float3 cont = performDirectLighting<false>(
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
                    float3 cont = performDirectLighting<false>(
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
        // EN: Calculate the estimate of the reciprocal of the probability density that the current sample suvives.
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
CUDA_DEVICE_FUNCTION void performSpatialRIS() {
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
            float3 cont = performDirectLighting<false>(
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
                    cont = performDirectLighting<true>(
                        positionInWorld, vOutLocal, shadingFrame, bsdf, selectedLightSample);
                else
                    cont = performDirectLighting<false>(
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
                        cont = performDirectLighting<true>(
                            nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, selectedLightSample);
                    else
                        cont = performDirectLighting<false>(
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
    // EN: Calculate the estimate of the reciprocal of the probability density that the current sample suvives.
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



CUDA_DEVICE_KERNEL void RT_RG_NAME(shading)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    uint32_t bufIdx = plp.f->bufferIndex;

    GBuffer0 gBuffer0 = plp.s->GBuffer0[bufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[bufIdx].read(launchIndex);
    GBuffer2 gBuffer2 = plp.s->GBuffer2[bufIdx].read(launchIndex);
    float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);
    uint32_t materialSlot = gBuffer2.materialSlot;

    const PerspectiveCamera &camera = plp.f->camera;

    float3 contribution = make_float3(0.01f, 0.01f, 0.01f);
    if (materialSlot != 0xFFFFFFFF) {
        float3 positionInWorld = gBuffer0.positionInWorld;
        float3 shadingNormalInWorld = gBuffer1.normalInWorld;

        const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

        // TODO?: Use true geometric normal.
        float3 geometricNormalInWorld = shadingNormalInWorld;
        float3 vOut = normalize(camera.position - positionInWorld);
        float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

        BSDF bsdf;
        bsdf.setup(mat, texCoord);
        ReferenceFrame shadingFrame(shadingNormalInWorld);
        positionInWorld = offsetRayOriginNaive(positionInWorld, frontHit * geometricNormalInWorld);
        float3 vOutLocal = shadingFrame.toLocal(vOut);

        uint32_t curResIndex = plp.currentReservoirIndex;
        const Reservoir<LightSample> /*&*/reservoir = plp.s->reservoirBuffer[curResIndex][launchIndex];
        const ReservoirInfo reservoirInfo = plp.s->reservoirInfoBuffer[curResIndex].read(launchIndex);

        // JP: 光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from a light source directly seeing.
        contribution = make_float3(0.0f);
        if (vOutLocal.z > 0) {
            float3 emittance = make_float3(0.0f, 0.0f, 0.0f);
            if (mat.emittance) {
                float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
                emittance = make_float3(texValue);
            }
            contribution += emittance / Pi;
        }

        // JP: 最終的に残ったサンプルとそのウェイトを使ってシェーディングを実行する。
        // EN: Perform shading using the sample survived in the end and its weight.
        const LightSample &lightSample = reservoir.getSample();
        float3 directCont = make_float3(0.0f);
        float recPDFEstimate = reservoirInfo.recPDFEstimate;
        if (recPDFEstimate > 0 && isfinite(recPDFEstimate)) {
            bool visDone = plp.f->reuseVisibility &&
                (!plp.f->enableTemporalReuse || (plp.f->enableSpatialReuse && plp.f->useUnbiasedEstimator));
            if (visDone)
                directCont = performDirectLighting<false>(positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample);
            else
                directCont = performDirectLighting<true>(positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample);
        }

        contribution += recPDFEstimate * directCont;
    }
    else {
        // JP: 環境光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from the environmental light source directly seeing.
        if (plp.s->envLightTexture && plp.f->enableEnvLight) {
            float u = texCoord.x, v = texCoord.y;
            float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
            float3 luminance = plp.f->envLightPowerCoeff * make_float3(texValue);
            contribution = luminance;
        }
    }

    float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.f->numAccumFrames > 0)
        prevColorResult = getXYZ(plp.s->beautyAccumBuffer.read(launchIndex));
    float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
    float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * contribution;
    plp.s->beautyAccumBuffer.write(launchIndex, make_float4(colorResult, 1.0f));
}



// ----------------------------------------------------------------
// Rearchitected ReSTIR

template <bool withTemporalRIS, bool withSpatialRIS, bool useUnbiasedEstimator>
CUDA_DEVICE_FUNCTION void traceShadowRays() {
    int2 launchIndex = make_int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    uint32_t curBufIdx = plp.f->bufferIndex;
    uint32_t curResIndex = plp.currentReservoirIndex;
    uint32_t prevBufIdx;
    uint32_t prevResIndex;
    GBuffer0 gBuffer0 = plp.s->GBuffer0[curBufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[curBufIdx].read(launchIndex);
    GBuffer2 gBuffer2 = plp.s->GBuffer2[curBufIdx].read(launchIndex);
    if constexpr (withTemporalRIS || withSpatialRIS) {
        prevBufIdx = (curBufIdx + 1) % 2;
        prevResIndex = (curResIndex + 1) % 2;
    }
    else {
        (void)prevBufIdx;
        (void)prevResIndex;
    }

    float3 positionInWorld = gBuffer0.positionInWorld;
    float3 shadingNormalInWorld = gBuffer1.normalInWorld;
    uint32_t materialSlot = gBuffer2.materialSlot;

    if (materialSlot == 0xFFFFFFFF)
        return;

    // TODO?: Use true geometric normal.
    float3 geometricNormalInWorld = shadingNormalInWorld;
    float3 vOut = plp.f->camera.position - positionInWorld;
    float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

    ReferenceFrame shadingFrame(shadingNormalInWorld);
    positionInWorld = offsetRayOriginNaive(positionInWorld, frontHit * geometricNormalInWorld);
    float dist = length(vOut);

    SampleVisibility sampleVis;

    // New sample for the current pixel
    LightSample newSample;
    bool newSampleIsValid;
    {
        const Reservoir<LightSample> /*&*/reservoir = plp.s->reservoirBuffer[curResIndex][launchIndex];
        newSample = reservoir.getSample();
        newSampleIsValid = reservoir.getSumWeights() > 0.0f;
        if (newSampleIsValid)
            sampleVis.newSample = evaluateVisibility(positionInWorld, newSample);
    }

    // Temporal Sample
    int2 tNbCoord;
    float3 tNbPositionInWorld;
    bool temporalSampleIsValid;
    if constexpr (withTemporalRIS) {
        float2 motionVector = gBuffer2.motionVector;
        tNbCoord = make_int2(launchIndex.x + 0.5f - motionVector.x,
                             launchIndex.y + 0.5f - motionVector.y);
        // JP: 隣接ピクセルのジオメトリ・マテリアルがあまりに異なる場合に候補サンプルを再利用すると
        //     バイアスが増えてしまうため、そのようなピクセルは棄却する。
        // EN: Reusing candidates from neighboring pixels with substantially different geometry/material
        //     leads to increased bias. Reject such a pixel.
        sampleVis.temporalPassedHeuristic = testNeighbor<true>(prevBufIdx, tNbCoord, dist, shadingNormalInWorld);
        if (sampleVis.temporalPassedHeuristic) {
            Reservoir<LightSample> neighbor;
            LightSample temporalSample;
            if (plp.f->reuseVisibilityForTemporal && !useUnbiasedEstimator) {
                SampleVisibility prevSampleVis = plp.s->sampleVisibilityBuffer[prevBufIdx].read(tNbCoord);
                sampleVis.temporalSample = prevSampleVis.selectedSample;
            }
            else {
                neighbor = plp.s->reservoirBuffer[prevResIndex][tNbCoord];
                temporalSample = neighbor.getSample();
                temporalSampleIsValid = neighbor.getSumWeights() > 0.0f;
                if (temporalSampleIsValid)
                    sampleVis.temporalSample = evaluateVisibility(positionInWorld, temporalSample);
            }

            if constexpr (useUnbiasedEstimator) {
                GBuffer0 nbGBuffer0 = plp.s->GBuffer0[prevBufIdx].read(tNbCoord);
                GBuffer1 nbGBuffer1 = plp.s->GBuffer1[prevBufIdx].read(tNbCoord);
                float3 nbPositionInWorld = nbGBuffer0.positionInWorld;
                float3 nbShadingNormalInWorld = nbGBuffer1.normalInWorld;

                // TODO?: Use true geometric normal.
                float3 nbGeometricNormalInWorld = nbShadingNormalInWorld;
                float3 nbVOut = plp.f->prevCamera.position - nbPositionInWorld;
                float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
                tNbPositionInWorld = offsetRayOriginNaive(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);

                if (newSampleIsValid)
                    sampleVis.newSampleOnTemporal = evaluateVisibility(tNbPositionInWorld, newSample);

                if (temporalSampleIsValid)
                    sampleVis.temporalSampleOnCurrent = evaluateVisibility(positionInWorld, temporalSample);
            }
            else {
                (void)tNbPositionInWorld;
            }
        }
    }
    else {
        (void)tNbCoord;
        (void)tNbPositionInWorld;
        (void)temporalSampleIsValid;
    }

    // Spatiotemporal Sample
    int2 stNbCoord;
    float3 stNbPositionInWorld;
    bool spatiotemporalSampleIsValid;
    if constexpr (withSpatialRIS) {
        // JP: 周辺ピクセルの座標をランダムに決定。
        // EN: Randomly determine the coordinates of a neighboring pixel.
        float radius = plp.f->spatialNeighborRadius;
        float deltaX, deltaY;
        if (plp.f->useLowDiscrepancyNeighbors) {
            uint32_t deltaIndex = plp.spatialNeighborBaseIndex +
                5 * launchIndex.x + 7 * launchIndex.y;
            float2 delta = plp.s->spatialNeighborDeltas[deltaIndex % 1024];
            deltaX = radius * delta.x;
            deltaY = radius * delta.y;
        }
        else {
            PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);
            radius *= std::sqrt(rng.getFloat0cTo1o());
            float angle = 2 * Pi * rng.getFloat0cTo1o();
            deltaX = radius * std::cos(angle);
            deltaY = radius * std::sin(angle);
            // JP: シェーディング時に同じ近傍を得るためにRNGのステート変化は保存しない。
            // EN: Not store RNG's state changes to get the same neighbor when shading.
        }
        stNbCoord = make_int2(launchIndex.x + 0.5f + deltaX,
                              launchIndex.y + 0.5f + deltaY);

        // JP: 隣接ピクセルのジオメトリ・マテリアルがあまりに異なる場合に候補サンプルを再利用すると
        //     バイアスが増えてしまうため、そのようなピクセルは棄却する。
        // EN: Reusing candidates from neighboring pixels with substantially different geometry/material
        //     leads to increased bias. Reject such a pixel.
        sampleVis.spatiotemporalPassedHeuristic = testNeighbor<true>(prevBufIdx, stNbCoord, dist, shadingNormalInWorld);
        sampleVis.spatiotemporalPassedHeuristic &= stNbCoord.x != launchIndex.x || stNbCoord.y != launchIndex.y;
        if (sampleVis.spatiotemporalPassedHeuristic) {
            bool reused = false;
            if (plp.f->reuseVisibilityForSpatiotemporal && !useUnbiasedEstimator) {
                float threshold2 = pow2(plp.f->radiusThresholdForSpatialVisReuse);
                float dist2 = pow2(deltaX) + pow2(deltaY);
                reused = dist2 < threshold2;
            }

            Reservoir<LightSample> neighbor;
            LightSample spatiotemporalSample;
            if (reused) {
                SampleVisibility prevSampleVis = plp.s->sampleVisibilityBuffer[prevBufIdx].read(stNbCoord);
                sampleVis.spatiotemporalSample = prevSampleVis.selectedSample;
            }
            else {
                neighbor = plp.s->reservoirBuffer[prevResIndex][stNbCoord];
                spatiotemporalSample = neighbor.getSample();
                spatiotemporalSampleIsValid = neighbor.getSumWeights() > 0.0f;
                if (spatiotemporalSampleIsValid)
                    sampleVis.spatiotemporalSample = evaluateVisibility(positionInWorld, spatiotemporalSample);
            }

            if constexpr (useUnbiasedEstimator) {
                GBuffer0 nbGBuffer0 = plp.s->GBuffer0[prevBufIdx].read(stNbCoord);
                GBuffer1 nbGBuffer1 = plp.s->GBuffer1[prevBufIdx].read(stNbCoord);
                float3 nbPositionInWorld = nbGBuffer0.positionInWorld;
                float3 nbShadingNormalInWorld = nbGBuffer1.normalInWorld;

                // TODO?: Use true geometric normal.
                float3 nbGeometricNormalInWorld = nbShadingNormalInWorld;
                float3 nbVOut = plp.f->prevCamera.position - nbPositionInWorld;
                float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
                stNbPositionInWorld = offsetRayOriginNaive(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);

                if (newSampleIsValid)
                    sampleVis.newSampleOnSpatiotemporal = evaluateVisibility(stNbPositionInWorld, newSample);

                if (spatiotemporalSampleIsValid)
                    sampleVis.spatiotemporalSampleOnCurrent = evaluateVisibility(positionInWorld, spatiotemporalSample);
            }
            else {
                (void)stNbPositionInWorld;
            }
        }
    }
    else {
        (void)stNbCoord;
        (void)stNbPositionInWorld;
        (void)spatiotemporalSampleIsValid;
    }

    if constexpr (useUnbiasedEstimator && withTemporalRIS && withSpatialRIS) {
        if (sampleVis.temporalPassedHeuristic && sampleVis.spatiotemporalPassedHeuristic) {
            if (temporalSampleIsValid) {
                const Reservoir<LightSample> /*&*/tNeighbor = plp.s->reservoirBuffer[prevResIndex][tNbCoord];
                sampleVis.temporalSampleOnSpatiotemporal =
                    evaluateVisibility(stNbPositionInWorld, tNeighbor.getSample());
            }
            if (spatiotemporalSampleIsValid) {
                const Reservoir<LightSample> /*&*/stNeighbor = plp.s->reservoirBuffer[prevResIndex][stNbCoord];
                sampleVis.spatiotemporalSampleOnTemporal =
                    evaluateVisibility(tNbPositionInWorld, stNeighbor.getSample());
            }
        }
    }

    plp.s->sampleVisibilityBuffer[curBufIdx].write(launchIndex, sampleVis);
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRays)() {
    traceShadowRays<false, false, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRaysWithTemporalReuseBiased)() {
    traceShadowRays<true, false, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRaysWithSpatialReuseBiased)() {
    traceShadowRays<false, true, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRaysWithSpatioTemporalReuseBiased)() {
    traceShadowRays<true, true, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRaysWithTemporalReuseUnbiased)() {
    traceShadowRays<true, false, true>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRaysWithSpatialReuseUnbiased)() {
    traceShadowRays<false, true, true>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRaysWithSpatioTemporalReuseUnbiased)() {
    traceShadowRays<true, true, true>();
}



enum class SampleType {
    New = 0,
    Temporal,
    Spatiotemporal
};

template <SampleType sampleType, bool withTemporalRIS, bool withSpatialRIS>
CUDA_DEVICE_FUNCTION float computeMISWeight(
    const int2 &launchIndex, uint32_t prevBufIdx, uint32_t prevResIndex,
    uint32_t maxPrevStreamLength, const SampleVisibility &sampleVis,
    uint32_t selfStreamLength, const float3 &positionInWorld, const float3 &vOutLocal,
    const ReferenceFrame &shadingFrame, const BSDF &bsdf,
    const int2 &tNbCoord, const int2 &stNbCoord,
    uint32_t streamLength, const LightSample &lightSample, float sampleTargetDensity) {
    float numMisWeight = sampleTargetDensity;
    float denomMisWeight = numMisWeight * streamLength;

    if constexpr (sampleType != SampleType::New) {
        // JP: 与えられたサンプルを現在のシェーディング点で得る確率密度を計算する。
        // EN: Compute a probability density to get the given sample at the current shading point.
        float3 cont = performDirectLighting<false>(
            positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample);
        float targetDensity = convertToWeight(cont);
        if (plp.f->useUnbiasedEstimator) {
            targetDensity *= sampleType == SampleType::Temporal ?
                sampleVis.temporalSampleOnCurrent :
                sampleVis.spatiotemporalSampleOnCurrent;
        }

        if constexpr (useMIS_RIS) {
            denomMisWeight += targetDensity * selfStreamLength;
        }
        else {
            if (targetDensity > 0.0f)
                denomMisWeight += selfStreamLength;
        }
    }

    if constexpr (sampleType != SampleType::Temporal && withTemporalRIS) {
        if (sampleVis.temporalPassedHeuristic) {
            GBuffer0 nbGBuffer0 = plp.s->GBuffer0[prevBufIdx].read(tNbCoord);
            GBuffer1 nbGBuffer1 = plp.s->GBuffer1[prevBufIdx].read(tNbCoord);
            GBuffer2 nbGBuffer2 = plp.s->GBuffer2[prevBufIdx].read(tNbCoord);
            float3 nbPositionInWorld = nbGBuffer0.positionInWorld;
            float3 nbShadingNormalInWorld = nbGBuffer1.normalInWorld;
            float2 nbTexCoord = make_float2(nbGBuffer0.texCoord_x, nbGBuffer1.texCoord_y);
            uint32_t nbMaterialSlot = nbGBuffer2.materialSlot;

            const MaterialData &nbMat = plp.s->materialDataBuffer[nbMaterialSlot];

            // TODO?: Use true geometric normal.
            float3 nbGeometricNormalInWorld = nbShadingNormalInWorld;
            float3 nbVOut = plp.f->prevCamera.position - nbPositionInWorld;
            float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

            BSDF nbBsdf;
            nbBsdf.setup(nbMat, nbTexCoord);
            ReferenceFrame nbShadingFrame(nbShadingNormalInWorld);
            nbPositionInWorld = offsetRayOriginNaive(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);
            float nbDist = length(nbVOut);
            nbVOut /= nbDist;
            float3 nbVOutLocal = nbShadingFrame.toLocal(nbVOut);

            // JP: 与えられたサンプルを前のフレームで対応するシェーディング点で得る確率密度を計算する。
            // EN: Compute a probability density to get the given sample at the corresponding shading point
            //     in the previous frame.
            float3 cont = performDirectLighting<false>(
                nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, lightSample);
            float nbTargetDensity = convertToWeight(cont);
            if (plp.f->useUnbiasedEstimator) {
                nbTargetDensity *= sampleType == SampleType::New ?
                    sampleVis.newSampleOnTemporal :
                    sampleVis.spatiotemporalSampleOnTemporal;
            }

            const Reservoir<LightSample> /*&*/neighbor = plp.s->reservoirBuffer[prevResIndex][tNbCoord];
            uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
            if constexpr (useMIS_RIS) {
                denomMisWeight += nbTargetDensity * nbStreamLength;
            }
            else {
                if (nbTargetDensity > 0.0f)
                    denomMisWeight += nbStreamLength;
            }
        }
    }

    if constexpr (sampleType != SampleType::Spatiotemporal && withSpatialRIS) {
        if (sampleVis.spatiotemporalPassedHeuristic) {
            GBuffer0 nbGBuffer0 = plp.s->GBuffer0[prevBufIdx].read(stNbCoord);
            GBuffer1 nbGBuffer1 = plp.s->GBuffer1[prevBufIdx].read(stNbCoord);
            GBuffer2 nbGBuffer2 = plp.s->GBuffer2[prevBufIdx].read(stNbCoord);
            float3 nbPositionInWorld = nbGBuffer0.positionInWorld;
            float3 nbShadingNormalInWorld = nbGBuffer1.normalInWorld;
            float2 nbTexCoord = make_float2(nbGBuffer0.texCoord_x, nbGBuffer1.texCoord_y);
            uint32_t nbMaterialSlot = nbGBuffer2.materialSlot;

            const MaterialData &nbMat = plp.s->materialDataBuffer[nbMaterialSlot];

            // TODO?: Use true geometric normal.
            float3 nbGeometricNormalInWorld = nbShadingNormalInWorld;
            float3 nbVOut = plp.f->prevCamera.position - nbPositionInWorld;
            float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

            BSDF nbBsdf;
            nbBsdf.setup(nbMat, nbTexCoord);
            ReferenceFrame nbShadingFrame(nbShadingNormalInWorld);
            nbPositionInWorld = offsetRayOriginNaive(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);
            float nbDist = length(nbVOut);
            nbVOut /= nbDist;
            float3 nbVOutLocal = nbShadingFrame.toLocal(nbVOut);

            // JP: 与えられたサンプルを近傍のシェーディング点(前フレーム)で得る確率密度を計算する。
            // EN: Compute a probability density to get the given sample at a shading point on the neighbor
            //     (in the previous frame).
            float3 cont = performDirectLighting<false>(
                nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, lightSample);
            float nbTargetDensity = convertToWeight(cont);
            if (plp.f->useUnbiasedEstimator) {
                nbTargetDensity *= sampleType == SampleType::New ?
                    sampleVis.newSampleOnSpatiotemporal :
                    sampleVis.temporalSampleOnSpatiotemporal;
            }

            const Reservoir<LightSample> /*&*/neighbor = plp.s->reservoirBuffer[prevResIndex][stNbCoord];
            uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
            if constexpr (useMIS_RIS) {
                denomMisWeight += nbTargetDensity * nbStreamLength;
            }
            else {
                if (nbTargetDensity > 0.0f)
                    denomMisWeight += nbStreamLength;
            }
        }
    }

    return numMisWeight / denomMisWeight;
}

template <bool withTemporalRIS, bool withSpatialRIS>
CUDA_DEVICE_FUNCTION void shadeAndResample() {
    int2 launchIndex = make_int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    uint32_t curBufIdx = plp.f->bufferIndex;
    uint32_t prevBufIdx;
    uint32_t curResIndex = plp.currentReservoirIndex;
    uint32_t prevResIndex;
    GBuffer0 gBuffer0 = plp.s->GBuffer0[curBufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[curBufIdx].read(launchIndex);
    GBuffer2 gBuffer2 = plp.s->GBuffer2[curBufIdx].read(launchIndex);
    if constexpr (withTemporalRIS || withSpatialRIS) {
        prevBufIdx = (curBufIdx + 1) % 2;
        prevResIndex = (curResIndex + 1) % 2;
    }
    else {
        (void)prevBufIdx;
        (void)prevResIndex;
    }

    float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);
    uint32_t materialSlot = gBuffer2.materialSlot;

    float3 contribution = make_float3(0.01f, 0.01f, 0.01f);
    if (materialSlot != 0xFFFFFFFF) {
        PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

        float3 positionInWorld = gBuffer0.positionInWorld;
        float3 shadingNormalInWorld = gBuffer1.normalInWorld;
        float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);

        int2 tNbCoord;
        if constexpr (withTemporalRIS) {
            float2 motionVector = gBuffer2.motionVector;
            tNbCoord = make_int2(launchIndex.x + 0.5f - motionVector.x,
                                 launchIndex.y + 0.5f - motionVector.y);
        }
        else {
            (void)tNbCoord;
        }
        int2 stNbCoord;
        if constexpr (withSpatialRIS) {
            // JP: 周辺ピクセルの座標をランダムに決定。
            // EN: Randomly determine the coordinates of a neighboring pixel.
            float radius = plp.f->spatialNeighborRadius;
            float deltaX, deltaY;
            if (plp.f->useLowDiscrepancyNeighbors) {
                uint32_t deltaIndex = plp.spatialNeighborBaseIndex +
                    5 * launchIndex.x + 7 * launchIndex.y;
                float2 delta = plp.s->spatialNeighborDeltas[deltaIndex % 1024];
                deltaX = radius * delta.x;
                deltaY = radius * delta.y;
            }
            else {
                radius *= std::sqrt(rng.getFloat0cTo1o());
                float angle = 2 * Pi * rng.getFloat0cTo1o();
                deltaX = radius * std::cos(angle);
                deltaY = radius * std::sin(angle);
            }
            stNbCoord = make_int2(launchIndex.x + 0.5f + deltaX,
                                  launchIndex.y + 0.5f + deltaY);
        }
        else {
            (void)stNbCoord;
        }

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

        // JP: 光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from a light source directly seeing.
        contribution = make_float3(0.0f);
        if (vOutLocal.z > 0) {
            float3 emittance = make_float3(0.0f, 0.0f, 0.0f);
            if (mat.emittance) {
                float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
                emittance = make_float3(texValue);
            }
            contribution += emittance / Pi;
        }

        SampleVisibility sampleVis = plp.s->sampleVisibilityBuffer[curResIndex].read(launchIndex);

        float selectedTargetDensity = 0.0f;
        Reservoir<LightSample> combinedReservoir;
        uint32_t combinedStreamLength = 0;
        combinedReservoir.initialize();

        float3 directCont = make_float3(0.0f, 0.0f, 0.0f);
        float selectedMisWeight = 0.0f;

        const Reservoir<LightSample> /*&*/selfRes = plp.s->reservoirBuffer[curResIndex][launchIndex];
        const ReservoirInfo selfResInfo = plp.s->reservoirInfoBuffer[curResIndex].read(launchIndex);
        uint32_t selfStreamLength = selfRes.getStreamLength();
        uint32_t maxPrevStreamLength;
        if constexpr (withTemporalRIS || withSpatialRIS)
            maxPrevStreamLength = 20 * selfStreamLength;
        else
            (void)maxPrevStreamLength;

        // New sample for the current pixel.
        {
            if (selfResInfo.recPDFEstimate > 0.0f && sampleVis.newSample) {
                LightSample lightSample = selfRes.getSample();
                float3 cont = performDirectLighting<false>(
                    positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample);
                float misWeight;
                if constexpr (withTemporalRIS || withSpatialRIS)
                    misWeight = computeMISWeight<SampleType::New, withTemporalRIS, withSpatialRIS>(
                        launchIndex, prevBufIdx, prevResIndex,
                        maxPrevStreamLength, sampleVis,
                        selfStreamLength, positionInWorld, vOutLocal, shadingFrame, bsdf,
                        tNbCoord, stNbCoord,
                        selfStreamLength, lightSample, selfResInfo.targetDensity);
                else
                    misWeight = 1.0f / selfStreamLength;
                selectedMisWeight = misWeight;
                float contWeight = selfStreamLength * misWeight * selfResInfo.recPDFEstimate;
                float3 rawDirectCont = sampleVis.newSample * contWeight * cont;
                float weight = selfRes.getSumWeights();
                directCont += /*weight * */rawDirectCont;
                combinedReservoir = selfRes;
                selectedTargetDensity = selfResInfo.targetDensity;
            }
            sampleVis.selectedSample = sampleVis.newSample;
            combinedStreamLength = selfStreamLength;
        }

        // Temporal Sample
        if constexpr (withTemporalRIS) {
            if (sampleVis.temporalPassedHeuristic) {
                const Reservoir<LightSample> /*&*/neighbor = plp.s->reservoirBuffer[prevResIndex][tNbCoord];
                const ReservoirInfo neighborInfo = plp.s->reservoirInfoBuffer[prevResIndex].read(tNbCoord);
                // JP: 際限なく過去フレームで得たサンプルがウェイトを増やさないように、
                //     前フレームのストリーム長を、現在フレームのReservoirに対して20倍までに制限する。
                // EN: Limit the stream length of the previous frame by 20 times of that of the current frame
                //     in order to avoid a sample obtained in the past getting a unlimited weight.
                uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
                if (neighborInfo.recPDFEstimate > 0.0f) {
                    // JP: 隣接ピクセルが持つ候補サンプルの「現在の」ピクセルにおける確率密度を計算する。
                    // EN: Calculate the probability density at the "current" pixel of the candidate sample
                    //     the neighboring pixel holds.
                    // TODO: アニメーションやジッタリングがある場合には前フレームの対応ピクセルのターゲットPDFは
                    //       変わってしまっているはず。この場合にはUnbiasedにするにはもうちょっと工夫がいる？
                    LightSample nbLightSample = neighbor.getSample();
                    float3 cont = performDirectLighting<false>(
                        positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
                    float targetDensity = convertToWeight(cont);

                    float misWeight = computeMISWeight<SampleType::Temporal, withTemporalRIS, withSpatialRIS>(
                        launchIndex, prevBufIdx, prevResIndex,
                        maxPrevStreamLength, sampleVis,
                        selfStreamLength, positionInWorld, vOutLocal, shadingFrame, bsdf,
                        tNbCoord, stNbCoord,
                        nbStreamLength, nbLightSample, neighborInfo.targetDensity);
                    float contWeight = nbStreamLength * misWeight * neighborInfo.recPDFEstimate;
                    float3 rawDirectCont = (sampleVis.temporalSample * contWeight) * cont;
                    float weight = targetDensity * neighborInfo.recPDFEstimate * nbStreamLength;
                    directCont += /*weight * */rawDirectCont;
                    if (combinedReservoir.update(nbLightSample, weight, rng.getFloat0cTo1o())) {
                        selectedTargetDensity = targetDensity;
                        selectedMisWeight = misWeight;
                        sampleVis.selectedSample = sampleVis.temporalSample;
                    }
                }
                combinedStreamLength += nbStreamLength;
            }
        }

        // Spatiotemporal Sample
        if constexpr (withSpatialRIS) {
            if (sampleVis.spatiotemporalPassedHeuristic) {
                const Reservoir<LightSample> /*&*/neighbor = plp.s->reservoirBuffer[prevResIndex][stNbCoord];
                const ReservoirInfo neighborInfo = plp.s->reservoirInfoBuffer[prevResIndex].read(stNbCoord);
                // JP: 際限なく過去フレームで得たサンプルがウェイトを増やさないように、
                //     前フレームのストリーム長を、現在フレームのReservoirに対して20倍までに制限する。
                // EN: Limit the stream length of the previous frame by 20 times of that of the current frame
                //     in order to avoid a sample obtained in the past getting a unlimited weight.
                uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
                if (neighborInfo.recPDFEstimate > 0.0f) {
                    // JP: 隣接ピクセルが持つ候補サンプルの「現在の」ピクセルにおける確率密度を計算する。
                    // EN: Calculate the probability density at the "current" pixel of the candidate sample
                    //     the neighboring pixel holds.
                    LightSample nbLightSample = neighbor.getSample();
                    float3 cont = performDirectLighting<false>(
                        positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
                    float targetDensity = convertToWeight(cont);

                    // JP: 隣接ピクセルと現在のピクセルではターゲットPDFが異なるためサンプルはウェイトを持つ。
                    // EN: The sample has a weight since the target PDFs of the neighboring pixel and the current
                    //     are the different.
                    float misWeight = computeMISWeight<SampleType::Spatiotemporal, withTemporalRIS, withSpatialRIS>(
                        launchIndex, prevBufIdx, prevResIndex,
                        maxPrevStreamLength, sampleVis,
                        selfStreamLength, positionInWorld, vOutLocal, shadingFrame, bsdf,
                        tNbCoord, stNbCoord,
                        nbStreamLength, nbLightSample, neighborInfo.targetDensity);
                    float contWeight = nbStreamLength * misWeight * neighborInfo.recPDFEstimate;
                    float3 rawDirectCont = (sampleVis.spatiotemporalSample * contWeight) * cont;
                    float weight = targetDensity * neighborInfo.recPDFEstimate * nbStreamLength;
                    directCont += /*weight * */rawDirectCont;
                    if (combinedReservoir.update(nbLightSample, weight, rng.getFloat0cTo1o())) {
                        selectedTargetDensity = targetDensity;
                        selectedMisWeight = misWeight;
                        sampleVis.selectedSample = sampleVis.spatiotemporalSample;
                    }
                }
                combinedStreamLength += nbStreamLength;
            }
        }

        combinedReservoir.setStreamLength(combinedStreamLength);
        //if (combinedReservoir.getSumWeights() > 0.0f)
        //    directCont /= combinedReservoir.getSumWeights();
        contribution += directCont;

        // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
        // EN: Calculate the estimate of the reciprocal of the probability density that the current sample suvives.
        float recPDFEstimate = selectedMisWeight * combinedReservoir.getSumWeights() / selectedTargetDensity;
        if (!isfinite(recPDFEstimate) || (plp.f->reuseVisibility && !sampleVis.selectedSample)) {
            recPDFEstimate = 0.0f;
            selectedTargetDensity = 0.0f;
        }

        ReservoirInfo reservoirInfo;
        reservoirInfo.recPDFEstimate = recPDFEstimate;
        reservoirInfo.targetDensity = selectedTargetDensity;

        plp.s->sampleVisibilityBuffer[curResIndex].write(launchIndex, sampleVis);
        plp.s->reservoirBuffer[curResIndex][launchIndex] = combinedReservoir;
        plp.s->reservoirInfoBuffer[curResIndex].write(launchIndex, reservoirInfo);
        plp.s->rngBuffer.write(launchIndex, rng);
    }
    else {
        // JP: 環境光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from the environmental light source directly seeing.
        if (plp.s->envLightTexture && plp.f->enableEnvLight) {
            float u = texCoord.x, v = texCoord.y;
            float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
            float3 luminance = plp.f->envLightPowerCoeff * make_float3(texValue);
            contribution = luminance;
        }
    }

    float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.f->numAccumFrames > 0)
        prevColorResult = getXYZ(plp.s->beautyAccumBuffer.read(launchIndex));
    float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
    float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * contribution;
    plp.s->beautyAccumBuffer.write(launchIndex, make_float4(colorResult, 1.0f));
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(shadeAndResample)() {
    shadeAndResample<false, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(shadeAndResampleWithTemporalReuse)() {
    shadeAndResample<true, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(shadeAndResampleWithSpatialReuse)() {
    shadeAndResample<false, true>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(shadeAndResampleWithSpatioTemporalReuse)() {
    shadeAndResample<true, true>();
}
