#include "../tfdm_shared.h"
#include "tfdm_intersection_kernels.h"

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

CUDA_DEVICE_FUNCTION CUDA_INLINE RGB performNextEventEstimation(
    const Point3D &shadingPoint, const Vector3D &vOutLocal, const ReferenceFrame &shadingFrame,
    const BSDF &bsdf, PCG32RNG &rng) {
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
            const float dist2 = shadowRay.sqLength();
            shadowRay /= std::sqrt(dist2);
            const Vector3D vInLocal = shadingFrame.toLocal(shadowRay);
            const float lpCos = std::fabs(dot(shadowRay, lightSample.normal));
            float bsdfPDensity = bsdf.evaluatePDF(vOutLocal, vInLocal) * lpCos / dist2;
            if (!stc::isfinite(bsdfPDensity))
                bsdfPDensity = 0.0f;
            const float lightPDensity = areaPDensity;
            misWeight = pow2(lightPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
        }
        if (areaPDensity > 0.0f)
            ret = performDirectLighting<PathTracingRayType, true>(
                shadingPoint, vOutLocal, shadingFrame, bsdf, lightSample) * (misWeight / areaPDensity);
    }

    return ret;
}

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
        const float bcA = 1.0f - (bcB + bcC);
        Point3D positionInWorld;
        Normal3D geometricNormalInWorld;
        Normal3D shadingNormalInWorld;
        Vector3D texCoord0DirInWorld;
        Point2D texCoord;
        // JP: ディスプレイスメントを適用したサーフェスに関してGバッファーから交点や法線を復元するのは
        //     不可能ではないが難がある。
        //     そこでディスプレイスメントサーフェスでは追加のGバッファーを使用してそれらの情報を記録している。
        // EN: There is difficulty (not impossible though) in restoring the hit position and normals
        //     from the G-buffers for displaced surfaces.
        //     Therefore displaced surfaces use additional G-buffers to store those information.
        if (gb0Elems.isDisplacedMesh) {
            const GBuffer1Elements gb1Elems = plp.s->GBuffer1[bufIdx].read(launchIndex);
            geometricNormalInWorld = decodeNormal(gb1Elems.qGeometricNormal);
            shadingNormalInWorld = decodeNormal(gb1Elems.qShadingNormal);

            const GBuffer2Elements gb2Elems = plp.s->GBuffer2[bufIdx].read(launchIndex);
            positionInWorld = gb2Elems.positionInWorld;
        }
        computeSurfacePoint(
            inst, geomInst, gb0Elems.isDisplacedMesh,
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
            //if (plp.f->enableBumpMapping) {
            //    const Normal3D modLocalNormal = mat.readModifiedNormal(mat.normal, mat.normalDimInfo, texCoord, 0.0f);
            //    applyBumpMapping(modLocalNormal, &shadingFrame);
            //}
            const Vector3D vOutLocal = shadingFrame.toLocal(vOut);

            // JP: 光源を直接見ている場合の寄与を蓄積。
            // EN: Accumulate the contribution from a light source directly seeing.
            contribution = RGB(0.0f);
            if (vOutLocal.z > 0 && mat.emittance) {
                const float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
                const RGB emittance(getXYZ(texValue));
                contribution += alpha * emittance / Pi;
            }

            float targetMipLevel = 0.0f;
            if (gb0Elems.isDisplacedMesh) {
                const GeometryInstanceDataForTFDM &tfdmGeomInst = plp.s->geomInstTfdmDataBuffer[geomInstSlot];
                targetMipLevel = tfdmGeomInst.params.targetMipLevel;
                // JP: ベース三角形のエッジに着色する。
                // EN: Color the edges of the base triangle.
                if (plp.f->showBaseEdges) {
                    if (bcA < 0.01f || bcB < 0.01f || bcC < 0.01f)
                        alpha *= RGB(1.0f, 0.25f, 0.0f);
                }
            }

            BSDF bsdf;
            bsdf.setup(mat, texCoord, targetMipLevel);

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
            const bool isValidSampling = rwPayload.prevDirPDensity > 0.0f && stc::isfinite(rwPayload.prevDirPDensity);
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
                const float continueProb =
                    std::fmin(sRGB_calcLuminance(rwPayload.alpha) / rwPayload.initImportance, 1.0f);
                if (rwPayload.rng.getFloat0cTo1o() >= continueProb)
                    break;
                rwPayload.alpha /= continueProb;
            }

            constexpr PathTracingRayType pathTraceRayType = PathTracingRayType::Closest;
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
    plp.s->beautyAccumBuffer.write(launchIndex, make_float4(colorResult.r, colorResult.g, colorResult.b, 1.0f));
}

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
    float targetMipLevel = 0;

    const uint32_t hitKind = optixGetHitKind();
    const bool isTriangleHit =
        hitKind == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE
        || hitKind == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE;
    const bool isDisplacedTriangleHit =
        hitKind == CustomHitKind_DisplacedSurfaceFrontFace
        || hitKind == CustomHitKind_DisplacedSurfaceBackFace;
    if (isTriangleHit || isDisplacedTriangleHit) {
        computeSurfacePoint<useMultipleImportanceSampling, useSolidAngleSampling>(
            inst, geomInst, isDisplacedTriangleHit, hp.primIndex, hp.bcB, hp.bcC,
            rayOrigin,
            &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
            &geometricNormalInWorld, &texCoord, &hypAreaPDensity);

        if (isDisplacedTriangleHit) {
            const GeometryInstanceDataForTFDM &tfdmGeomInst = plp.s->geomInstTfdmDataBuffer[sbtr.geomInstSlot];
            targetMipLevel = tfdmGeomInst.params.targetMipLevel;
        }
    }
    else { // for AABB debugging
        const GeometryInstanceDataForTFDM &tfdm = plp.s->geomInstTfdmDataBuffer[sbtr.geomInstSlot];
        const AABB &aabb = tfdm.aabbBuffer[hp.primIndex];
        Normal3D normalInObj;
        const Point3D positionInObj = aabb.restoreHitPoint(hp.bcB, hp.bcC, &normalInObj);
        positionInWorld = transformPointFromObjectToWorldSpace(positionInObj);
        geometricNormalInWorld = normalize(transformNormalFromObjectToWorldSpace(normalInObj));
        shadingNormalInWorld = geometricNormalInWorld;
        texCoord = Point2D(0.0f, 0.0f);
        Vector3D bitangent;
        makeCoordinateSystem(shadingNormalInWorld, &texCoord0DirInWorld, &bitangent);

        hypAreaPDensity = 0.0f;
    }
    if constexpr (!useMultipleImportanceSampling)
        (void)hypAreaPDensity;

    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    const Vector3D vOut = normalize(-Vector3D(optixGetWorldRayDirection()));
    const float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

    const ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
    // Offsetting assumes BRDF.
    positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);
    const Vector3D vOutLocal = shadingFrame.toLocal(vOut);

    if constexpr (useImplicitLightSampling) {
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
    bsdf.setup(mat, texCoord, targetMipLevel);

    // Next Event Estimation (Explicit Light Sampling)
    rwPayload->contribution += rwPayload->alpha * performNextEventEstimation(
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

CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTrace)() {
    pathTrace_rayGen_generic();
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
