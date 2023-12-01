#include "../regir_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility);
}



static constexpr bool useSolidAngleSampling = false;
static constexpr bool useImplicitLightSampling = true;
static constexpr bool useExplicitLightSampling = true;
static constexpr bool useMultipleImportanceSampling = useImplicitLightSampling && useExplicitLightSampling;
static_assert(useImplicitLightSampling || useExplicitLightSampling, "Invalid configuration for light sampling.");

CUDA_DEVICE_FUNCTION CUDA_INLINE RGB sampleFromCell(
    const Point3D &shadingPoint, const Vector3D &vOutLocal, const ReferenceFrame &shadingFrame, const BSDF &bsdf,
    uint32_t frameIndex, PCG32RNG &rng,
    LightSample* lightSample, float* recProbDensityEstimate) {
    Vector3D randomOffset;
    if (plp.f->enableCellRandomization) {
        randomOffset = plp.s->gridCellSize
            * Vector3D(-0.5f + rng.getFloat0cTo1o(),
                       -0.5f + rng.getFloat0cTo1o(),
                       -0.5f + rng.getFloat0cTo1o());
    }
    else {
        randomOffset = Vector3D(0.0f);
    }
    const uint32_t cellLinearIndex = calcCellLinearIndex(shadingPoint + randomOffset);
    const uint32_t resStartIndex = kNumLightSlotsPerCell * cellLinearIndex;

    // JP: セルに触れたフラグを建てておく。
    // EN: Set the flag indicating the cell is touched.
    atomicAdd(&plp.s->perCellNumAccesses[cellLinearIndex], 1u);

    // JP: セルごとに保持している複数のReservoirからリサンプリングを行う。
    // EN: Resample from multiple reservoirs held by each cell.
    const uint32_t numResampling = 1 << plp.f->log2NumCandidatesPerCell;
    Reservoir<LightSample> combinedReservoir;
    combinedReservoir.initialize(LightSample());
    uint32_t combinedStreamLength = 0;
    RGB selectedContribution(0.0f);
    float selectedTargetPDensity = 0.0f;
    for (int i = 0; i < numResampling; ++i) {
        const uint32_t lightSlotIdx = resStartIndex + mapPrimarySampleToDiscrete(rng.getFloat0cTo1o(), kNumLightSlotsPerCell);
        const Reservoir<LightSample> &r = plp.s->reservoirs[plp.f->bufferIndex][lightSlotIdx];
        const ReservoirInfo &rInfo = plp.s->reservoirInfos[plp.f->bufferIndex][lightSlotIdx];
        const LightSample lightSample = r.getSample();
        const uint32_t streamLength = r.getStreamLength();
        combinedStreamLength += streamLength;
        if (rInfo.recPDFEstimate == 0.0f)
            continue;

        // JP: Unshadowed ContributionをターゲットPDFとする。
        // EN: Use unshadowed constribution as the target PDF.
        const RGB cont = performDirectLighting<PathTracingRayType, false>(
            shadingPoint, vOutLocal, shadingFrame, bsdf, lightSample);
        const float targetPDensity = convertToWeight(cont);

        // JP: ソースのターゲットPDFとここでのターゲットPDFは異なるためサンプルにはウェイトがかかる。
        // EN: The sample has a weight since the source PDF and the target PDF hre are different.
        const float weight = targetPDensity * rInfo.recPDFEstimate * streamLength;
        if (combinedReservoir.update(lightSample, weight, rng.getFloat0cTo1o())) {
            selectedContribution = cont;
            selectedTargetPDensity = targetPDensity;
        }
    }
    combinedReservoir.setStreamLength(combinedStreamLength);

    *lightSample = combinedReservoir.getSample();

    const float weightForEstimate = 1.0f / combinedReservoir.getStreamLength();
    *recProbDensityEstimate = weightForEstimate * combinedReservoir.getSumWeights() / selectedTargetPDensity;
    if (!isfinite(*recProbDensityEstimate))
        *recProbDensityEstimate = 0.0f;

    return selectedContribution;
}

template <bool useReGIR>
CUDA_DEVICE_FUNCTION CUDA_INLINE RGB performNextEventEstimation(
    const Point3D &shadingPoint, const Vector3D &vOutLocal, const ReferenceFrame &shadingFrame, const BSDF &bsdf,
    PCG32RNG &rng) {
    RGB ret(0.0f);
    if constexpr (useReGIR) {
        LightSample lightSample;
        float recProbDensityEstimate;
        const RGB unshadowedContribution = sampleFromCell(
            shadingPoint, vOutLocal, shadingFrame, bsdf,
            plp.f->frameIndex, rng,
            &lightSample, &recProbDensityEstimate);
        if (recProbDensityEstimate > 0.0f) {
            const float visibility = evaluateVisibility<PathTracingRayType>(shadingPoint, lightSample);
            ret = unshadowedContribution * (visibility * recProbDensityEstimate);
        }
    }
    else {
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
                const float dist2 = shadowRay.sqLength();
                shadowRay /= std::sqrt(dist2);
                const Vector3D vInLocal = shadingFrame.toLocal(shadowRay);
                const float lpCos = std::fabs(dot(shadowRay, lightSample.normal));
                float bsdfPDensity = bsdf.evaluatePDF(vOutLocal, vInLocal) * lpCos / dist2;
                if (!isfinite(bsdfPDensity))
                    bsdfPDensity = 0.0f;
                const float lightPDensity = areaPDensity;
                misWeight = pow2(lightPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
            }
            if (areaPDensity > 0.0f)
                ret = performDirectLighting<PathTracingRayType, true>(
                    shadingPoint, vOutLocal, shadingFrame, bsdf, lightSample) * (misWeight / areaPDensity);
        }
    }

    return ret;
}

template <bool useReGIR>
CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_rayGen_generic() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = plp.f->bufferIndex;

    const GBuffer0Elements gb0Elems = plp.s->GBuffer0[bufIdx].read(launchIndex);
    const uint32_t instSlot = gb0Elems.instSlot;
    const float bcB = decodeBarycentric(gb0Elems.qbcB);
    const float bcC = decodeBarycentric(gb0Elems.qbcC);

    const PerspectiveCamera &camera = plp.f->camera;

    const bool useEnvLight = plp.s->envLightTexture && plp.f->enableEnvLight;
    RGB contribution(0.001f, 0.001f, 0.001f);
    if (instSlot != 0xFFFFFFFF) {
        const uint32_t geomInstSlot = gb0Elems.geomInstSlot;
        const InstanceData &inst = plp.s->instanceDataBufferArray[bufIdx][instSlot];
        const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[geomInstSlot];
        Point3D positionInWorld;
        Normal3D geometricNormalInWorld;
        Normal3D shadingNormalInWorld;
        Vector3D texCoord0DirInWorld;
        Point2D texCoord;
        computeSurfacePoint(
            inst, geomInst,
            gb0Elems.primIndex, bcB, bcC,
            &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
            &geometricNormalInWorld, &texCoord);

        RGB alpha(1.0f);
        const float initImportance = sRGB_calcLuminance(alpha);
        PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

        // JP: 最初の交点におけるシェーディング。
        // EN: Shading on the first hit.
        Vector3D vIn;
        float dirPDensity;
        {
            const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

            const Vector3D vOut = normalize(camera.position - positionInWorld);
            const float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
            // Offsetting assumes BRDF.
            positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);

            ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
            if (plp.f->enableBumpMapping) {
                const Normal3D modLocalNormal = mat.readModifiedNormal(mat.normal, mat.normalDimInfo, texCoord, 0.0f);
                applyBumpMapping(modLocalNormal, &shadingFrame);
            }
            const Vector3D vOutLocal = shadingFrame.toLocal(vOut);

            // JP: 光源を直接見ている場合の寄与を蓄積。
            // EN: Accumulate the contribution from a light source directly seeing.
            contribution = RGB(0.0f);
            if (vOutLocal.z > 0 && mat.emittance) {
                const float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
                const RGB emittance(getXYZ(texValue));
                contribution += alpha * emittance / Pi;
            }

            BSDF bsdf;
            bsdf.setup(mat, texCoord, 0.0f);

            // Next event estimation (explicit light sampling) on the first hit.
            contribution += alpha * performNextEventEstimation<useReGIR>(
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
            const bool isValidSampling = rwPayload.prevDirPDensity > 0.0f && isfinite(rwPayload.prevDirPDensity);
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
            if constexpr (useReGIR || !useImplicitLightSampling) {
                if (rwPayload.maxLengthTerminate)
                    break;
                // Russian roulette
                const float continueProb =
                    std::fmin(sRGB_calcLuminance(rwPayload.alpha) / rwPayload.initImportance, 1.0f);
                if (rwPayload.rng.getFloat0cTo1o() >= continueProb)
                    break;
                rwPayload.alpha /= continueProb;
            }

            constexpr PathTracingRayType pathTraceRayType = useReGIR ?
                PathTracingRayType::ReGIR : PathTracingRayType::Baseline;
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

        plp.s->rngBuffer.write(launchIndex, rwPayload.rng);
    }
    else {
        // JP: 環境光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from the environmental light source directly seeing.
        if (useEnvLight) {
            const float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, bcB, bcC, 0.0f);
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

template <bool useReGIR>
CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_closestHit_generic() {
    const uint32_t bufIdx = plp.f->bufferIndex;
    const auto sbtr = HitGroupSBTRecordData::get();
    const InstanceData &inst = plp.s->instanceDataBufferArray[bufIdx][optixGetInstanceId()];
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];

    PathTraceWriteOnlyPayload* woPayload;
    PathTraceReadWritePayload* rwPayload;
    PathTraceRayPayloadSignature::get(&woPayload, &rwPayload);
    PCG32RNG &rng = rwPayload->rng;

    const Point3D rayOrigin(optixGetWorldRayOrigin());

    const auto hp = HitPointParameter::get();
    Point3D positionInWorld;
    Normal3D shadingNormalInWorld;
    Vector3D texCoord0DirInWorld;
    Normal3D geometricNormalInWorld;
    Point2D texCoord;
    float hypAreaPDensity;
    computeSurfacePoint<useMultipleImportanceSampling && !useReGIR, useSolidAngleSampling>(
        inst, geomInst, hp.primIndex, hp.bcB, hp.bcC,
        rayOrigin,
        &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
        &geometricNormalInWorld, &texCoord, &hypAreaPDensity);
    if constexpr (!useMultipleImportanceSampling || useReGIR)
        (void)hypAreaPDensity;

    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    const Vector3D vOut = normalize(-Vector3D(optixGetWorldRayDirection()));
    const float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

    ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
    if (plp.f->enableBumpMapping) {
        const Normal3D modLocalNormal = mat.readModifiedNormal(mat.normal, mat.normalDimInfo, texCoord, 0.0f);
        applyBumpMapping(modLocalNormal, &shadingFrame);
    }
    positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);
    const Vector3D vOutLocal = shadingFrame.toLocal(vOut);

    if constexpr (useImplicitLightSampling || !useReGIR) {
        // Implicit Light Sampling
        if (vOutLocal.z > 0 && mat.emittance) {
            const float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
            const RGB emittance(getXYZ(texValue));
            float misWeight = 1.0f;
            if constexpr (useMultipleImportanceSampling) {
                const float dist2 = sqDistance(rayOrigin, positionInWorld);
                const float lightPDensity = hypAreaPDensity * dist2 / vOutLocal.z;
                const float bsdfPDensity = rwPayload->prevDirPDensity;
                misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
            }
            rwPayload->contribution += rwPayload->alpha * emittance * (misWeight / Pi);
        }

        // Russian roulette
        const float continueProb = std::fmin(sRGB_calcLuminance(rwPayload->alpha) / rwPayload->initImportance, 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb || rwPayload->maxLengthTerminate)
            return;
        rwPayload->alpha /= continueProb;
    }

    BSDF bsdf;
    bsdf.setup(mat, texCoord, 0.0f);

    // Next Event Estimation (Explicit Light Sampling)
    rwPayload->contribution += rwPayload->alpha * performNextEventEstimation<useReGIR>(
        positionInWorld, vOutLocal, shadingFrame, bsdf, rng);

    // generate a next ray.
    Vector3D vInLocal;
    float dirPDensity;
    rwPayload->alpha *= bsdf.sampleThroughput(
        vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &vInLocal, &dirPDensity);
    const Vector3D vIn = shadingFrame.fromLocal(vInLocal);

    woPayload->nextOrigin = positionInWorld;
    woPayload->nextDirection = vIn;
    rwPayload->prevDirPDensity = dirPDensity;
    rwPayload->terminate = false;
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTraceBaseline)() {
    pathTrace_rayGen_generic<false>();
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTraceBaseline)() {
    pathTrace_closestHit_generic<false>();
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(pathTraceBaseline)() {
    if constexpr (useImplicitLightSampling) {
        if (!plp.s->envLightTexture || !plp.f->enableEnvLight)
            return;

        PathTraceReadWritePayload* rwPayload;
        PathTraceRayPayloadSignature::get(nullptr, &rwPayload);

        const Vector3D rayDir = normalize(Vector3D(optixGetWorldRayDirection()));
        float posPhi, theta;
        toPolarYUp(rayDir, &posPhi, &theta);

        float phi = posPhi + plp.f->envLightRotation;
        phi = phi - floorf(phi / (2 * Pi)) * 2 * Pi;
        const Point2D texCoord(phi / (2 * Pi), theta / Pi);

        // Implicit Light Sampling
        const float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, texCoord.x, texCoord.y, 0.0f);
        const RGB luminance = plp.f->envLightPowerCoeff * RGB(getXYZ(texValue));
        float misWeight = 1.0f;
        if constexpr (useMultipleImportanceSampling) {
            const float uvPDF = plp.s->envLightImportanceMap.evaluatePDF(texCoord.x, texCoord.y);
            const float hypAreaPDensity = uvPDF / (2 * Pi * Pi * std::sin(theta));
            const float lightPDensity =
                (plp.s->lightInstDist.integral() > 0.0f ? probToSampleEnvLight : 1.0f) *
                hypAreaPDensity;
            const float bsdfPDensity = rwPayload->prevDirPDensity;
            misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
        }
        rwPayload->contribution += rwPayload->alpha * luminance * misWeight;
    }
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTraceReGIR)() {
    pathTrace_rayGen_generic<true>();
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTraceReGIR)() {
    pathTrace_closestHit_generic<true>();
}
