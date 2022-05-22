#include "../path_tracing_shared.h"

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

CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_rayGen_generic() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    uint32_t bufIdx = plp.f->bufferIndex;
    GBuffer0 gBuffer0 = plp.s->GBuffer0[bufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[bufIdx].read(launchIndex);
    GBuffer2 gBuffer2 = plp.s->GBuffer2[bufIdx].read(launchIndex);

    float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);
    uint32_t materialSlot = gBuffer2.materialSlot;

    bool useEnvLight = plp.s->envLightTexture && plp.f->enableEnvLight;
    float3 contribution = make_float3(0.0f, 0.0f, 0.0f);
    if (materialSlot != 0xFFFFFFFF) {
        const PerspectiveCamera &camera = plp.f->camera;

        float3 alpha = make_float3(1.0f);
        float initImportance = sRGB_calcLuminance(alpha);
        PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

        float3 positionInWorld = gBuffer0.positionInWorld;
        ReferenceFrame shadingFrame(gBuffer1.normalInWorld);
        // TODO?: Use true geometric normal.
        float3 geometricNormalInWorld = shadingFrame.normal;

        float3 vOutLocal;
        {
            float3 vOut = normalize(camera.position - positionInWorld);
            float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
            positionInWorld = offsetRayOriginNaive(positionInWorld, frontHit * geometricNormalInWorld);
            vOutLocal = shadingFrame.toLocal(vOut);
        }

        if constexpr (useImplicitLightSampling) {
            // Implicit Light Sampling
            const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];
            if (vOutLocal.z > 0 && mat.emittance) {
                float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
                float3 emittance = make_float3(texValue);
                contribution += alpha * emittance / Pi;
            }
        }

        // Path extension loop
        uint32_t pathLength = 1;
        while (true) {
            const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

            BSDF bsdf;
            bsdf.setup(mat, texCoord);

            if constexpr (useExplicitLightSampling) {
                // Next Event Estimation (Explicit Light Sampling)
                contribution += alpha * performNextEventEstimation(
                    positionInWorld, vOutLocal, shadingFrame, bsdf, rng);
            }

            if constexpr (!useImplicitLightSampling) {
                if (pathLength + 1 >= plp.f->maxPathLength)
                    break;
            }

            // sample a next ray.
            float3 vInLocal;
            float dirPDensity;
            alpha *= bsdf.sampleThroughput(
                vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
                &vInLocal, &dirPDensity);
            if (dirPDensity <= 0.0f || !isfinite(dirPDensity))
                break;
            if constexpr (!useImplicitLightSampling) {
                // Russian roulette
                float continueProb = std::fmin(sRGB_calcLuminance(alpha) / initImportance, 1.0f);
                if (rng.getFloat0cTo1o() >= continueProb)
                    break;
                alpha /= continueProb;
            }
            float3 vIn = shadingFrame.fromLocal(vInLocal);
            ++pathLength;

            float3 rayOrg = positionInWorld;
            uint32_t instSlot;
            uint32_t geomInstSlot;
            HitPointParameter hp;
            constexpr PathTracingRayType pathTraceRayType = PathTracingRayType::Baseline;
            optixu::trace<PathTraceRayPayloadSignature>(
                plp.f->travHandle, rayOrg, vIn,
                0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
                pathTraceRayType, maxNumRayTypes, pathTraceRayType,
                instSlot, geomInstSlot, hp);

            if (instSlot != 0xFFFFFFFF) {
                const InstanceData &inst = plp.f->instanceDataBuffer[instSlot];
                const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[geomInstSlot];
                materialSlot = geomInst.materialSlot;

                float3 shadingNormalInWorld;
                float3 texCoord0DirInWorld;
                float hypAreaPDensity;
                computeSurfacePoint<useMultipleImportanceSampling, useSolidAngleSampling>(
                    inst, geomInst, hp.primIndex, hp.b1, hp.b2,
                    rayOrg,
                    &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
                    &geometricNormalInWorld, &texCoord, &hypAreaPDensity);
                if constexpr (!useMultipleImportanceSampling)
                    (void)hypAreaPDensity;

                shadingFrame = ReferenceFrame(shadingNormalInWorld, texCoord0DirInWorld);
                float3 modLocalNormal = mat.readModifiedNormal(mat.normal, texCoord, mat.normalDimension);
                if (plp.f->enableBumpMapping)
                    applyBumpMapping(modLocalNormal, &shadingFrame);

                float3 vOut = -vIn;
                float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
                positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);
                vOutLocal = shadingFrame.toLocal(vOut);

                if constexpr (useImplicitLightSampling) {
                    // Implicit Light Sampling
                    const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];
                    if (vOutLocal.z > 0 && mat.emittance) {
                        float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
                        float3 emittance = make_float3(texValue);
                        float misWeight = 1.0f;
                        if constexpr (useMultipleImportanceSampling) {
                            float dist2 = squaredDistance(rayOrg, positionInWorld);
                            float lightPDensity = hypAreaPDensity * dist2 / vOutLocal.z;
                            float bsdfPDensity = dirPDensity;
                            misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
                        }
                        contribution += alpha * emittance * (misWeight / Pi);
                    }

                    // Russian roulette
                    float continueProb = std::fmin(sRGB_calcLuminance(alpha) / initImportance, 1.0f);
                    if (rng.getFloat0cTo1o() >= continueProb || pathLength >= plp.f->maxPathLength)
                        break;
                    alpha /= continueProb;
                }
            }
            else {
                // JP: 無限遠光源のImplicit Light Sampling。
                // EN: Implicit light sampling for the infinitely distant light.
                if constexpr (useImplicitLightSampling) {
                    if (!plp.s->envLightTexture || !plp.f->enableEnvLight)
                        break;

                    float posPhi, theta;
                    toPolarYUp(vIn, &posPhi, &theta);

                    float phi = posPhi + plp.f->envLightRotation;
                    phi = phi - floorf(phi / (2 * Pi)) * 2 * Pi;
                    float2 texCoord = make_float2(phi / (2 * Pi), theta / Pi);

                    // Implicit Light Sampling
                    float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, texCoord.x, texCoord.y, 0.0f);
                    float3 luminance = plp.f->envLightPowerCoeff * make_float3(texValue);
                    float misWeight = 1.0f;
                    if constexpr (useMultipleImportanceSampling) {
                        if (pathLength > 1) {
                            float uvPDF = plp.s->envLightImportanceMap.evaluatePDF(texCoord.x, texCoord.y);
                            float hypAreaPDensity = uvPDF / (2 * Pi * Pi * std::sin(theta));
                            float lightPDensity =
                                (plp.s->lightInstDist.integral() > 0.0f ? probToSampleEnvLight : 1.0f) *
                                hypAreaPDensity;
                            float bsdfPDensity = dirPDensity;
                            misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
                        }
                    }
                    contribution += alpha * luminance * misWeight;
                }
                break;
            }
        }

        plp.s->rngBuffer.write(launchIndex, rng);
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
        else {
            contribution = make_float3(0.001f, 0.001f, 0.001f);
        }
    }

    float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.f->numAccumFrames > 0)
        prevColorResult = getXYZ(plp.s->beautyAccumBuffer.read(launchIndex));
    float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
    float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * contribution;
    plp.s->beautyAccumBuffer.write(launchIndex, make_float4(colorResult, 1.0f));
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_closestHit_generic() {
    auto sbtr = HitGroupSBTRecordData::get();
    uint32_t instSlot = optixGetInstanceId();
    uint32_t geomInstSlot = sbtr.geomInstSlot;
    auto hp = HitPointParameter::get();
    PathTraceRayPayloadSignature::set(&instSlot, &geomInstSlot, &hp);
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTraceBaseline)() {
    pathTrace_rayGen_generic();
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTraceBaseline)() {
    pathTrace_closestHit_generic();
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(pathTraceBaseline)() {
    constexpr uint32_t instSlot = 0xFFFFFFFF;
    PathTraceRayPayloadSignature::set(&instSlot, nullptr, nullptr);
}
