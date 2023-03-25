#include "../svgf_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility);
}



struct PreviousResult {
    RGB noisyLighting;
    float firstMoment;
    float secondMoment;
    SampleInfo sampleInfo;
};

struct PreviousNeighbor {
    Point3D position;
    Normal3D normal;
    uint32_t instSlot;
    uint32_t matSlot;

    RGB noisyLighting;
    float firstMoment;
    float secondMoment;
    SampleInfo sampleInfo;

    CUDA_DEVICE_FUNCTION PreviousNeighbor(const int2 &pix) {
        uint32_t prevBufIdx = (plp.f->bufferIndex + 1) % 2;
        const StaticPipelineLaunchParameters::TemporalSet &staticTemporalSet =
            plp.s->temporalSets[prevBufIdx];
        const PerFramePipelineLaunchParameters::TemporalSet &perFrameTemporalSet =
            plp.f->temporalSets[prevBufIdx];
        //float depth = perFrameTemporalSet.depthBuffer.read(glPix(pix));
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
    const Point3D &curPosInWorld, const Normal3D &curNormalInWorld, uint32_t curInstSlot, uint32_t curMatSlot,
    Point2D prevScreenPos,
    PreviousResult* prevResult) {
    prevResult->noisyLighting = RGB(0.0f, 0.0f, 0.0f);
    prevResult->firstMoment = 0.0f;
    prevResult->secondMoment = 0.0f;
    prevResult->sampleInfo = SampleInfo(0, 0);

    bool outOfScreen = (prevScreenPos.x < 0.0f || prevScreenPos.y < 0.0f ||
                        prevScreenPos.x >= 1.0f || prevScreenPos.y >= 1.0f);
    if (outOfScreen)
        return;

    int2 imageSize = plp.s->imageSize;

    Point2D prevViewportPos(imageSize.x * prevScreenPos.x, imageSize.y * prevScreenPos.y);
    int2 prevPixPos = make_int2(prevViewportPos.x, prevViewportPos.y);
    Vector2D fDelta = prevViewportPos - (Point2D(prevPixPos.x, prevPixPos.y) + Vector2D(0.5f));
    int2 delta = make_int2(fDelta.x < 0 ? -1 : 1,
                           fDelta.y < 0 ? -1 : 1);

    int2 basePos = make_int2(prevPixPos.x, prevPixPos.y);
    int2 dxPos = make_int2(clamp(prevPixPos.x + delta.x, 0, imageSize.x - 1), prevPixPos.y);
    int2 dyPos = make_int2(prevPixPos.x, clamp(prevPixPos.y + delta.y, 0, imageSize.y - 1));
    int2 dxdyPos = make_int2(clamp(prevPixPos.x + delta.x, 0, imageSize.x - 1),
                             clamp(prevPixPos.y + delta.y, 0, imageSize.y - 1));

    PreviousNeighbor prevNeighbors[] = {
        PreviousNeighbor(basePos),
        PreviousNeighbor(dxPos),
        PreviousNeighbor(dyPos),
        PreviousNeighbor(dxdyPos),
    };

    float sumWeights = 0.0f;
    float prevFloatSampleCount = 0;
    uint32_t acceptableFlags = 0;
    float s = std::fabs(fDelta.x);
    float t = std::fabs(fDelta.y);

    const auto testAndAccumulate = [&](uint32_t i, float weight) {
        const PreviousNeighbor &prevNeighbor = prevNeighbors[i];
        if (prevNeighbor.instSlot != curInstSlot || prevNeighbor.matSlot != curMatSlot)
            return;
        if (dot(prevNeighbor.normal, curNormalInWorld) <= 0.85f)
            return;
        if ((prevNeighbor.position - curPosInWorld).sqLength() > 0.1f) // TODO: シーンスケールに対して相対的な指標にする。
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

CUDA_DEVICE_FUNCTION CUDA_INLINE RGB performNextEventEstimation(
    const Point3D &shadingPoint, const Vector3D &vOutLocal, const ReferenceFrame &shadingFrame, const BSDF &bsdf,
    PCG32RNG &rng) {
    RGB ret(0.0f);
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
            Vector3D shadowRay = lightSample.atInfinity ?
                Vector3D(lightSample.position) :
                (lightSample.position - shadingPoint);
            float dist2 = shadowRay.sqLength();
            shadowRay /= std::sqrt(dist2);
            Vector3D vInLocal = shadingFrame.toLocal(shadowRay);
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

    Point3D positionInWorld = gBuffer0.positionInWorld;
    Normal3D shadingNormalInWorld = gBuffer1.normalInWorld;
    Point2D texCoord(gBuffer0.texCoord_x, gBuffer1.texCoord_y);
    Point2D prevScreenPos = gBuffer2.prevScreenPos;
    uint32_t instSlot = gBuffer2.instSlot;
    uint32_t materialSlot = gBuffer2.materialSlot;
    if (materialSlot == 0xFFFFFFFF) {
        Lighting_Variance lighting_var;
        lighting_var.noisyLighting = RGB(0.0f, 0.0f, 0.0f);
        lighting_var.variance = 0.0f;
        plp.s->lighting_variance_buffers[0].write(launchIndex, lighting_var);
        return;
    }

    const PerspectiveCamera &camera = perFrameTemporalSet.camera;

    RGB dhReflectance(0.0f, 0.0f, 0.0f);
    RGB contribution(0.0f, 0.0f, 0.0f);
    RGB alpha(1.0f);
    float initImportance = sRGB_calcLuminance(alpha);
    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

    // JP: 最初の交点におけるシェーディング。
    // EN: Shading on the first hit.
    Vector3D vIn;
    float dirPDensity;
    {
        const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

        // TODO?: Use true geometric normal.
        Normal3D geometricNormalInWorld = shadingNormalInWorld;
        Vector3D vOut = normalize(camera.position - positionInWorld);
        float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

        ReferenceFrame shadingFrame(shadingNormalInWorld);
        positionInWorld = offsetRayOriginNaive(positionInWorld, frontHit * geometricNormalInWorld);
        Vector3D vOutLocal = shadingFrame.toLocal(vOut);

        // JP: 光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from a light source directly seeing.
        if (vOutLocal.z > 0 && mat.emittance) {
            float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
            RGB emittance = RGB(getXYZ(texValue));
            contribution += alpha * emittance / Pi;
        }

        BSDF bsdf;
        bsdf.setup(mat, texCoord);

        dhReflectance = bsdf.evaluateDHReflectanceEstimate(vOutLocal);

        // Next event estimation (explicit light sampling) on the first hit.
        contribution += alpha * performNextEventEstimation(
            positionInWorld, vOutLocal, shadingFrame, bsdf, rng);

        // generate a next ray.
        Vector3D vInLocal;
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
    Point3D rayOrg = positionInWorld;
    Vector3D rayDir = vIn;
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
        PathTraceRayPayloadSignature::trace(
            plp.f->travHandle, rayOrg.toNative(), rayDir.toNative(),
            0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
            pathTraceRayType, maxNumRayTypes, pathTraceRayType,
            woPayloadPtr, rwPayloadPtr);
        if (rwPayload.terminate)
            break;
        rayOrg = woPayload.nextOrigin;
        rayDir = woPayload.nextDirection;
    }
    contribution = rwPayload.contribution;
    contribution = max(contribution, RGB(0.0f));

    plp.s->rngBuffer.write(launchIndex, rwPayload.rng);

    /*
    JP: Directional-Hemisperical Reflectance (DHref)の推定値が非常に小さな場合、
        demodulationされたライティングの値が異常に大きくなってしまう不安定性があるため
        十分に小さい場合はDHrefをゼロとして扱う。
    EN: Treat directional-hemispherical reflectance (DHref) as zero when the estimate of DHref
        is very small because there is instability in demodulated lighting values that
        the value become so huge in this case.
    */
    dhReflectance.r = dhReflectance.r < 0.001f ? 0.0f : dhReflectance.r;
    dhReflectance.g = dhReflectance.g < 0.001f ? 0.0f : dhReflectance.g;
    dhReflectance.b = dhReflectance.b < 0.001f ? 0.0f : dhReflectance.b;

    Albedo albedo = {};
    albedo.dhReflectance = dhReflectance;
    plp.s->albedoBuffer.write(launchIndex, albedo);

    // JP: Temporal Reprojectionを行って前フレームの対応サンプルを取得する。
    // EN: Perform temporal reprojection to obtain the corresponding sample from the previous frame.
    PreviousResult prevResult;
    if constexpr (enableTemporalAccumulation) {
        reprojectPreviousAccumulation(
            positionInWorld, shadingNormalInWorld, instSlot, materialSlot,
            prevScreenPos, &prevResult);
    }

    RGB demCont = safeDivide(contribution, albedo.dhReflectance);
    float luminance = sRGB_calcLuminance(demCont);
    float sqLuminance = pow2(luminance);

    // JP: 前フレームの結果と現在のフレームの結果をブレンドする。
    // EN: Blend the current frame result adn the previous frame result.
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

    const Point3D rayOrigin(optixGetWorldRayOrigin());

    auto hp = HitPointParameter::get();
    Point3D positionInWorld;
    Normal3D shadingNormalInWorld;
    Vector3D texCoord0DirInWorld;
    Normal3D geometricNormalInWorld;
    Point2D texCoord;
    float hypAreaPDensity;
    computeSurfacePoint<useMultipleImportanceSampling, useSolidAngleSampling>(
        inst, geomInst, hp.primIndex, hp.b1, hp.b2,
        rayOrigin,
        &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
        &geometricNormalInWorld, &texCoord, &hypAreaPDensity);
    if constexpr (!useMultipleImportanceSampling)
        (void)hypAreaPDensity;

    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    Vector3D vOut = normalize(-Vector3D(optixGetWorldRayDirection()));
    float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

    ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
    if (plp.f->enableBumpMapping) {
        Normal3D modLocalNormal = mat.readModifiedNormal(mat.normal, mat.normalDimInfo, texCoord);
        applyBumpMapping(modLocalNormal, &shadingFrame);
    }
    positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);
    Vector3D vOutLocal = shadingFrame.toLocal(vOut);

    if constexpr (useImplicitLightSampling) {
        // Implicit Light Sampling
        if (vOutLocal.z > 0 && mat.emittance) {
            float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
            RGB emittance = RGB(getXYZ(texValue));
            float misWeight = 1.0f;
            if constexpr (useMultipleImportanceSampling) {
                float dist2 = sqDistance(rayOrigin, positionInWorld);
                float lightPDensity = hypAreaPDensity * dist2 / vOutLocal.z;
                float bsdfPDensity = rwPayload->prevDirPDensity;
                misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
            }
            rwPayload->contribution += rwPayload->alpha * emittance * (misWeight / Pi);
        }

        // Russian roulette
        float continueProb = 1.0f;
        if (rwPayload->pathLength > 2)
          continueProb = std::fmin(sRGB_calcLuminance(rwPayload->alpha) / rwPayload->initImportance, 1.0f);
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
    Vector3D vInLocal;
    float dirPDensity;
    rwPayload->alpha *= bsdf.sampleThroughput(
        vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &vInLocal, &dirPDensity);
    Vector3D vIn = shadingFrame.fromLocal(vInLocal);

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

        Vector3D rayDir = normalize(Vector3D(optixGetWorldRayDirection()));
        float posPhi, theta;
        toPolarYUp(rayDir, &posPhi, &theta);

        float phi = posPhi + plp.f->envLightRotation;
        phi = phi - floorf(phi / (2 * Pi)) * 2 * Pi;
        Point2D texCoord(phi / (2 * Pi), theta / Pi);

        // Implicit Light Sampling
        float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, texCoord.x, texCoord.y, 0.0f);
        RGB luminance = plp.f->envLightPowerCoeff * RGB(getXYZ(texValue));
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
