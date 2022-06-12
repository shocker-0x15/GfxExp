#include "../svgf_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility);
}



struct PreviousResult {
    float3 noisyLighting;
    float firstMoment;
    float secondMoment;
    SampleInfo sampleInfo;
};

struct PreviousNeighbor {
    float3 position;
    float3 normal;
    uint32_t instSlot;
    uint32_t matSlot;

    float3 noisyLighting;
    float firstMoment;
    float secondMoment;
    SampleInfo sampleInfo;

    CUDA_DEVICE_FUNCTION PreviousNeighbor(const int2 &pix) {
        uint32_t prevBufIdx = (plp.f->bufferIndex + 1) % 2;
        const StaticPipelineLaunchParameters::TemporalSet &staticTemporalSet =
            plp.s->temporalSets[prevBufIdx];
        const PerFramePipelineLaunchParameters::TemporalSet &perFrameTemporalSet =
            plp.f->temporalSets[prevBufIdx];
        float depth = perFrameTemporalSet.depthBuffer.read(glPix(pix));
        GBuffer0 gBuffer0 = perFrameTemporalSet.GBuffer0.read(glPix(pix));
        GBuffer1 gBuffer1 = perFrameTemporalSet.GBuffer1.read(glPix(pix));
        GBuffer2 gBuffer2 = perFrameTemporalSet.GBuffer2.read(glPix(pix));
        position = gBuffer0.positionInWorld;
        normal = gBuffer1.normalInWorld;
        instSlot = gBuffer2.instSlot;
        matSlot = gBuffer2.materialSlot;

        Lighting_Variance lighting_var = plp.s->prevNoisyLightingBuffer.read(pix);
        noisyLighting = lighting_var.noisyLighting;
        MomentPair_SampleInfo momentPair_sampleInfo =
            staticTemporalSet.momentPair_sampleInfo_buffer.read(pix);
        firstMoment = momentPair_sampleInfo.firstMoment;
        secondMoment = momentPair_sampleInfo.secondMoment;
        sampleInfo = momentPair_sampleInfo.sampleInfo;
    }
};

CUDA_DEVICE_FUNCTION CUDA_INLINE void reprojectPreviousAccumulation(
    const float3 &curPosInWorld, const float3 &curNormalInWorld, uint32_t curInstSlot, uint32_t curMatSlot,
    float2 prevScreenPos,
    PreviousResult* prevResult) {
    prevResult->noisyLighting = make_float3(0.0f, 0.0f, 0.0f);
    prevResult->firstMoment = 0.0f;
    prevResult->secondMoment = 0.0f;
    prevResult->sampleInfo = SampleInfo(0, 0);

    bool outOfScreen = (prevScreenPos.x < 0.0f || prevScreenPos.y < 0.0f ||
                        prevScreenPos.x >= 1.0f || prevScreenPos.y >= 1.0f);
    if (outOfScreen)
        return;

    int2 imageSize = plp.s->imageSize;

    float2 prevViewportPos = make_float2(imageSize.x * prevScreenPos.x, imageSize.y * prevScreenPos.y);
    int2 prevPixPos = make_int2(prevViewportPos);

    int2 ulPos = make_int2(prevPixPos.x, prevPixPos.y);
    int2 urPos = make_int2(min(prevPixPos.x + 1, imageSize.x - 1), prevPixPos.y);
    int2 llPos = make_int2(prevPixPos.x, min(prevPixPos.y + 1, imageSize.y - 1));
    int2 lrPos = make_int2(min(prevPixPos.x + 1, imageSize.x - 1),
                           min(prevPixPos.y + 1, imageSize.y - 1));

    PreviousNeighbor prevNeighbors[] = {
        PreviousNeighbor(ulPos),
        PreviousNeighbor(urPos),
        PreviousNeighbor(llPos),
        PreviousNeighbor(lrPos),
    };

    float sumWeights = 0.0f;
    float prevFloatSampleCount = 0;
    uint32_t acceptableFlags = 0;
    float s = clamp((prevViewportPos.x - 0.5f) - prevPixPos.x, 0.0f, 1.0f);
    float t = clamp((prevViewportPos.y - 0.5f) - prevPixPos.y, 0.0f, 1.0f);

    //{
    //    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    //    if (launchIndex == plp.f->mousePosition) {
    //        printf("m: %4u, %4u, prev: %6.1f, %6.1f: %.3f, %.3f\n",
    //               vector2Arg(launchIndex), vector2Arg(prevViewportPos),
    //               s, t);
    //    }
    //}

    const auto testAndAccumulate = [&](uint32_t i, float weight) {
        const PreviousNeighbor &prevNeighbor = prevNeighbors[i];
        if (prevNeighbor.instSlot != curInstSlot || prevNeighbor.matSlot != curMatSlot)
            return;
        if (dot(prevNeighbor.normal, curNormalInWorld) <= 0.85)
            return;
        if (sqLength(prevNeighbor.position - curPosInWorld) > 0.1f)
            return;

        prevResult->noisyLighting += weight * prevNeighbor.noisyLighting;
        prevResult->firstMoment += weight * prevNeighbor.firstMoment;
        prevResult->secondMoment += weight * prevNeighbor.secondMoment;
        prevFloatSampleCount += weight * prevNeighbor.sampleInfo.count;
        sumWeights += weight;
        acceptableFlags |= (1 << i);
    };

    testAndAccumulate(0, (1 - s) * (1 - t));
    testAndAccumulate(1, s * (1 - t));
    testAndAccumulate(2, (1 - s) * t);
    testAndAccumulate(3, s * t);

    if (sumWeights > 0) {
        prevResult->noisyLighting /= sumWeights;
        prevResult->firstMoment /= sumWeights;
        prevResult->secondMoment /= sumWeights;
        prevResult->sampleInfo.count = static_cast<uint32_t>(roundf(prevFloatSampleCount / sumWeights));
        prevResult->sampleInfo.acceptFlags = acceptableFlags;
    }
}



static constexpr bool useSolidAngleSampling = false;
static constexpr bool useImplicitLightSampling = true;
static constexpr bool useExplicitLightSampling = true;
static constexpr bool useMultipleImportanceSampling = useImplicitLightSampling && useExplicitLightSampling;
static_assert(useImplicitLightSampling || useExplicitLightSampling, "Invalid configuration for light sampling.");

CUDA_DEVICE_FUNCTION CUDA_INLINE float3 performNextEventEstimation(
    const float3 &shadingPoint, const float3 &vOutLocal, const ReferenceFrame &shadingFrame, const BSDF &bsdf,
    PCG32RNG &rng) {
    float3 ret = make_float3(0.0f);
    if constexpr (useExplicitLightSampling) {
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
        float misWeight = 1.0f;
        if constexpr (useMultipleImportanceSampling) {
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
            misWeight = pow2(lightPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
        }
        if (areaPDensity > 0.0f)
            ret = performDirectLighting<PathTracingRayType, true>(
                shadingPoint, vOutLocal, shadingFrame, bsdf, lightSample) * (misWeight / areaPDensity);
    }

    return ret;
}

template <bool enableTemporalAccumulation>
CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_rayGen_generic() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    uint32_t curBufIdx = plp.f->bufferIndex;
    const StaticPipelineLaunchParameters::TemporalSet &staticTemporalSet =
        plp.s->temporalSets[curBufIdx];
    const PerFramePipelineLaunchParameters::TemporalSet &perFrameTemporalSet =
        plp.f->temporalSets[curBufIdx];

    GBuffer0 gBuffer0 = perFrameTemporalSet.GBuffer0.read(glPix(launchIndex));
    GBuffer1 gBuffer1 = perFrameTemporalSet.GBuffer1.read(glPix(launchIndex));
    GBuffer2 gBuffer2 = perFrameTemporalSet.GBuffer2.read(glPix(launchIndex));

    float3 positionInWorld = gBuffer0.positionInWorld;
    float3 shadingNormalInWorld = gBuffer1.normalInWorld;
    float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);
    float2 prevScreenPos = gBuffer2.prevScreenPos;
    uint32_t instSlot = gBuffer2.instSlot;
    uint32_t materialSlot = gBuffer2.materialSlot;

    const PerspectiveCamera &camera = perFrameTemporalSet.camera;

    float3 dhReflectance = make_float3(0.0f, 0.0f, 0.0f);
    bool useEnvLight = plp.s->envLightTexture && plp.f->enableEnvLight;
    float3 contribution = make_float3(0.001f, 0.001f, 0.001f);
    if (materialSlot != 0xFFFFFFFF) {
        float3 alpha = make_float3(1.0f);
        float initImportance = sRGB_calcLuminance(alpha);
        PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

        // JP: 最初の交点におけるシェーディング。
        // EN: Shading on the first hit.
        float3 vIn;
        float dirPDensity;
        {
            const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

            // TODO?: Use true geometric normal.
            float3 geometricNormalInWorld = shadingNormalInWorld;
            float3 vOut = normalize(camera.position - positionInWorld);
            float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

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

            dhReflectance = bsdf.evaluateDHReflectanceEstimate(vOutLocal);

            // Next event estimation (explicit light sampling) on the first hit.
            contribution += alpha * performNextEventEstimation(
                positionInWorld, vOutLocal, shadingFrame, bsdf, rng);

            // generate a next ray.
            float3 vInLocal;
            alpha *= bsdf.sampleThroughput(
                vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
                &vInLocal, &dirPDensity);
            vIn = shadingFrame.fromLocal(vInLocal);
        }

        // Path extension loop
        PathTraceWriteOnlyPayload woPayload = {};
        PathTraceWriteOnlyPayload* woPayloadPtr = &woPayload;
        PathTraceReadWritePayload rwPayload = {};
        PathTraceReadWritePayload* rwPayloadPtr = &rwPayload;
        rwPayload.rng = rng;
        rwPayload.initImportance = initImportance;
        rwPayload.alpha = alpha;
        rwPayload.prevDirPDensity = dirPDensity;
        rwPayload.contribution = contribution;
        rwPayload.pathLength = 1;
        float3 rayOrg = positionInWorld;
        float3 rayDir = vIn;
        while (true) {
            bool isValidSampling = rwPayload.prevDirPDensity > 0.0f && isfinite(rwPayload.prevDirPDensity);
            if (!isValidSampling)
                break;

            ++rwPayload.pathLength;
            if (rwPayload.pathLength >= plp.f->maxPathLength)
                rwPayload.maxLengthTerminate = true;
            rwPayload.terminate = true;
            // JP: 経路長制限に到達したときに、implicit light samplingを使わない場合はClosest-hit program内
            //     で行うことが無いので終了する。
            // EN: Nothing to do in the closest-hit program when reaching the path length limit
            //     in the case implicit light sampling is unused.
            if constexpr (!useImplicitLightSampling) {
                if (rwPayload.maxLengthTerminate)
                    break;
                // Russian roulette
                float continueProb = std::fmin(sRGB_calcLuminance(rwPayload.alpha) / rwPayload.initImportance, 1.0f);
                if (rwPayload.rng.getFloat0cTo1o() >= continueProb)
                    break;
                rwPayload.alpha /= continueProb;
            }

            constexpr PathTracingRayType pathTraceRayType = PathTracingRayType::Baseline;
            optixu::trace<PathTraceRayPayloadSignature>(
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

    /*
    JP: Directional-Hemisperical Reflectance (DHref)の推定値が非常に小さな場合、
        demodulationされたライティングの値が異常に大きくなってしまう不安定性があるため
        十分に小さい場合はDHrefをゼロとして扱う。
    EN: Treat directional-hemispherical reflectance (DHref) as zero when the estimate of DHref is very small
        because there is instability in demodulated lighting values that the value become so huge in this case.
    */
    dhReflectance.x = dhReflectance.x < 0.001f ? 0.0f : dhReflectance.x;
    dhReflectance.y = dhReflectance.y < 0.001f ? 0.0f : dhReflectance.y;
    dhReflectance.z = dhReflectance.z < 0.001f ? 0.0f : dhReflectance.z;

    Albedo albedo = {};
    albedo.dhReflectance = dhReflectance;
    plp.s->albedoBuffer.write(launchIndex, albedo);

    PreviousResult prevResult;
    if constexpr (enableTemporalAccumulation) {
        reprojectPreviousAccumulation(
            positionInWorld, shadingNormalInWorld, instSlot, materialSlot,
            prevScreenPos, &prevResult);
    }

    float3 demCont = safeDivide(contribution, albedo.dhReflectance);
    float luminance = sRGB_calcLuminance(demCont);
    float sqLuminance = pow2(luminance);

    if (plp.f->isFirstFrame || !plp.f->enableTemporalAccumulation)
        prevResult.sampleInfo.asUInt32 = 0;
    uint32_t sampleCount = min(prevResult.sampleInfo.count + 1, 65535u);
    if constexpr (enableTemporalAccumulation) {
        if (sampleCount > 1) {
            float curWeight = 1.0f / 5; // Exponential Moving Average
            if (sampleCount < 5) // Cumulative Moving Average
                curWeight = 1.0f / sampleCount;
            float prevWeight = 1.0f - curWeight;
            demCont = prevWeight * prevResult.noisyLighting + curWeight * demCont;
            luminance = prevWeight * prevResult.firstMoment + curWeight * luminance;
            sqLuminance = prevWeight * prevResult.secondMoment + curWeight * sqLuminance;
        }
    }

    //if (plp.f->mousePosition == launchIndex) {
    //    printf("%2u (%4u, %4u): norm: (%g, %g, %g) cont: (%g, %g, %g), dem: (%g, %g, %g), %g, %g, %u\n",
    //           plp.f->frameIndex, launchIndex.x, launchIndex.y,
    //           vector3Arg(shadingNormalInWorld), vector3Arg(contribution), vector3Arg(demCont),
    //           luminance, sqLuminance, sampleCount);
    //}

    Lighting_Variance lighting_var;
    lighting_var.noisyLighting = demCont;
    lighting_var.variance = 0.0f;
    plp.s->lighting_variance_buffers[0].write(launchIndex, lighting_var);

    MomentPair_SampleInfo moment_sampleInfo = {};
    moment_sampleInfo.firstMoment = luminance;
    moment_sampleInfo.secondMoment = sqLuminance;
    moment_sampleInfo.sampleInfo = SampleInfo(sampleCount, prevResult.sampleInfo.acceptFlags);
    staticTemporalSet.momentPair_sampleInfo_buffer.write(launchIndex, moment_sampleInfo);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_closestHit_generic() {
    auto sbtr = HitGroupSBTRecordData::get();
    const InstanceData &inst = plp.f->instanceDataBuffer[optixGetInstanceId()];
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];

    PathTraceWriteOnlyPayload* woPayload;
    PathTraceReadWritePayload* rwPayload;
    PathTraceRayPayloadSignature::get(&woPayload, &rwPayload);
    PCG32RNG &rng = rwPayload->rng;

    const float3 rayOrigin = optixGetWorldRayOrigin();

    auto hp = HitPointParameter::get();
    float3 positionInWorld;
    float3 shadingNormalInWorld;
    float3 texCoord0DirInWorld;
    float3 geometricNormalInWorld;
    float2 texCoord;
    float hypAreaPDensity;
    computeSurfacePoint<useMultipleImportanceSampling, useSolidAngleSampling>(
        inst, geomInst, hp.primIndex, hp.b1, hp.b2,
        rayOrigin,
        &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
        &geometricNormalInWorld, &texCoord, &hypAreaPDensity);
    if constexpr (!useMultipleImportanceSampling)
        (void)hypAreaPDensity;

    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    float3 vOut = normalize(-optixGetWorldRayDirection());
    float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

    ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
    if (plp.f->enableBumpMapping) {
        float3 modLocalNormal = mat.readModifiedNormal(mat.normal, mat.normalDimInfo, texCoord);
        applyBumpMapping(modLocalNormal, &shadingFrame);
    }
    positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);
    float3 vOutLocal = shadingFrame.toLocal(vOut);

    if constexpr (useImplicitLightSampling) {
        // Implicit Light Sampling
        if (vOutLocal.z > 0 && mat.emittance) {
            float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
            float3 emittance = make_float3(texValue);
            float misWeight = 1.0f;
            if constexpr (useMultipleImportanceSampling) {
                float dist2 = squaredDistance(rayOrigin, positionInWorld);
                float lightPDensity = hypAreaPDensity * dist2 / vOutLocal.z;
                float bsdfPDensity = rwPayload->prevDirPDensity;
                misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
            }
            rwPayload->contribution += rwPayload->alpha * emittance * (misWeight / Pi);
        }

        // Russian roulette
        float continueProb = std::fmin(sRGB_calcLuminance(rwPayload->alpha) / rwPayload->initImportance, 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb || rwPayload->maxLengthTerminate)
            return;
        rwPayload->alpha /= continueProb;
    }

    // JP: ファイアフライを抑えるために二次反射以降のスペキュラー面のラフネスを抑える。
    // EN: Mollify specular roughness after secondary reflection to avoid fire flies.
    BSDF bsdf;
    BSDFFlags bsdfFlags = plp.f->mollifySpecular ? BSDFFlags::Regularize : BSDFFlags::None;
    bsdf.setup(mat, texCoord, bsdfFlags);

    // Next Event Estimation (Explicit Light Sampling)
    rwPayload->contribution += rwPayload->alpha * performNextEventEstimation(
        positionInWorld, vOutLocal, shadingFrame, bsdf, rng);

    // generate a next ray.
    float3 vInLocal;
    float dirPDensity;
    rwPayload->alpha *= bsdf.sampleThroughput(
        vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &vInLocal, &dirPDensity);
    float3 vIn = shadingFrame.fromLocal(vInLocal);

    woPayload->nextOrigin = positionInWorld;
    woPayload->nextDirection = vIn;
    rwPayload->prevDirPDensity = dirPDensity;
    rwPayload->terminate = false;
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTraceWithoutTemporalAccumulation)() {
    pathTrace_rayGen_generic<false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTraceWithTemporalAccumulation)() {
    pathTrace_rayGen_generic<true>();
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTrace)() {
    pathTrace_closestHit_generic();
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(pathTrace)() {
    if constexpr (useImplicitLightSampling) {
        if (!plp.s->envLightTexture || !plp.f->enableEnvLight)
            return;

        PathTraceReadWritePayload* rwPayload;
        PathTraceRayPayloadSignature::get(nullptr, &rwPayload);

        float3 rayDir = normalize(optixGetWorldRayDirection());
        float posPhi, theta;
        toPolarYUp(rayDir, &posPhi, &theta);

        float phi = posPhi + plp.f->envLightRotation;
        phi = phi - floorf(phi / (2 * Pi)) * 2 * Pi;
        float2 texCoord = make_float2(phi / (2 * Pi), theta / Pi);

        // Implicit Light Sampling
        float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, texCoord.x, texCoord.y, 0.0f);
        float3 luminance = plp.f->envLightPowerCoeff * make_float3(texValue);
        float misWeight = 1.0f;
        if constexpr (useMultipleImportanceSampling) {
            float uvPDF = plp.s->envLightImportanceMap.evaluatePDF(texCoord.x, texCoord.y);
            float hypAreaPDensity = uvPDF / (2 * Pi * Pi * std::sin(theta));
            float lightPDensity =
                (plp.s->lightInstDist.integral() > 0.0f ? probToSampleEnvLight : 1.0f) *
                hypAreaPDensity;
            float bsdfPDensity = rwPayload->prevDirPDensity;
            misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
        }
        rwPayload->contribution += rwPayload->alpha * luminance * misWeight;
    }
}
