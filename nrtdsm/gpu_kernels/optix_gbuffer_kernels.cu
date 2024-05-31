#include "../nrtdsm_shared.h"
#include "nrtdsm_intersection_kernels.h"

using namespace shared;

CUDA_DEVICE_KERNEL void RT_IS_NAME(primaryDisplacementMappedSurface)() {
#if OUTPUT_TRAVERSAL_STATS
    TraversalStats travStats;
    PrimaryRayPayloadSignature::get(nullptr, nullptr, &travStats);
    detailedSurface_generic<true, false>(&travStats);
    PrimaryRayPayloadSignature::set(nullptr, nullptr, &travStats);
#else
    detailedSurface_generic<false, false>(nullptr);
#endif
}

CUDA_DEVICE_KERNEL void RT_IS_NAME(primaryShellMappedSurface)() {
#if OUTPUT_TRAVERSAL_STATS
    TraversalStats travStats;
    PrimaryRayPayloadSignature::get(nullptr, nullptr, &travStats);
    detailedSurface_generic<true, true>(&travStats);
    PrimaryRayPayloadSignature::set(nullptr, nullptr, &travStats);
#else
    detailedSurface_generic<false, true>(nullptr);
#endif
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(setupGBuffers)() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = plp.f->bufferIndex;

    const PerspectiveCamera &camera = plp.f->camera;
    float jx = 0.5f;
    float jy = 0.5f;
    if (plp.f->enableJittering) {
        PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);
        jx = rng.getFloat0cTo1o();
        jy = rng.getFloat0cTo1o();
        plp.s->rngBuffer.write(launchIndex, rng);
    }
    const float x = (launchIndex.x + jx) / plp.s->imageSize.x;
    const float y = (launchIndex.y + jy) / plp.s->imageSize.y;
    const float vh = 2 * std::tan(camera.fovY * 0.5f);
    const float vw = camera.aspect * vh;

    const Point3D origin = camera.position;
    const Vector3D direction = normalize(camera.orientation * Vector3D(vw * (0.5f - x), vh * (0.5f - y), 1));

    HitPointParams hitPointParams;
    hitPointParams.albedo = RGB(0.0f);
    hitPointParams.positionInWorld = Point3D(NAN);
    hitPointParams.prevPositionInWorld = Point3D(NAN);
    hitPointParams.geometricNormalInWorld = Normal3D(NAN);
    hitPointParams.shadingNormalInWorld = Normal3D(NAN);
    hitPointParams.instSlot = invalidInstIndex;
    hitPointParams.meshType = 0;
    hitPointParams.geomInstSlot = invalidGeomInstIndex;
    hitPointParams.shellBvhGeomIndex = 0;
    hitPointParams.primIndex = invalidPrimIndex;
    hitPointParams.qbcB = 0;
    hitPointParams.qbcC = 0;
#if OUTPUT_TRAVERSAL_STATS
    TraversalStats travStats;
    travStats.numAabbTests = 0;
    travStats.numLeafTests = 0;
#endif

    PickInfo pickInfo = {};

    HitPointParams* hitPointParamsPtr = &hitPointParams;
    PickInfo* pickInfoPtr = &pickInfo;
    PrimaryRayPayloadSignature::trace(
        plp.f->travHandle, origin.toNative(), direction.toNative(),
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        GBufferRayType::Primary, maxNumRayTypes, GBufferRayType::Primary,
        hitPointParamsPtr, pickInfoPtr
#if OUTPUT_TRAVERSAL_STATS
        , travStats
#endif
    );



    const Point2D curRasterPos(launchIndex.x + 0.5f, launchIndex.y + 0.5f);
    const Point2D prevRasterPos =
        plp.f->prevCamera.calcScreenPosition(hitPointParams.prevPositionInWorld)
        * Point2D(plp.s->imageSize.x, plp.s->imageSize.y);
    Vector2D motionVector = curRasterPos - prevRasterPos;
    if (plp.f->resetFlowBuffer || isnan(hitPointParams.prevPositionInWorld.x))
        motionVector = Vector2D(0.0f, 0.0f);

    GBuffer0Elements gb0Elems = {};
    gb0Elems.instSlot = hitPointParams.instSlot;
    gb0Elems.meshType = hitPointParams.meshType;
    gb0Elems.geomInstSlot = hitPointParams.geomInstSlot;
    gb0Elems.shellBvhGeomIndex = hitPointParams.shellBvhGeomIndex;
    gb0Elems.primIndex = hitPointParams.primIndex;
    gb0Elems.qbcB = hitPointParams.qbcB;
    gb0Elems.qbcC = hitPointParams.qbcC;
    GBuffer1Elements gb1Elems = {};
    gb1Elems.motionVector = motionVector;
    gb1Elems.qGeometricNormal = encodeNormal(hitPointParams.geometricNormalInWorld);
    gb1Elems.qShadingNormal = encodeNormal(hitPointParams.shadingNormalInWorld);
    GBuffer2Elements gb2Elems = {};
    gb2Elems.positionInWorld = hitPointParams.positionInWorld;

    plp.s->GBuffer0[bufIdx].write(launchIndex, gb0Elems);
    plp.s->GBuffer1[bufIdx].write(launchIndex, gb1Elems);
    plp.s->GBuffer2[bufIdx].write(launchIndex, gb2Elems);

#if OUTPUT_TRAVERSAL_STATS
    plp.s->numTravStatsBuffer.write(launchIndex, travStats);
#endif

    if (launchIndex.x == plp.f->mousePosition.x &&
        launchIndex.y == plp.f->mousePosition.y) {
        pickInfo.instSlot = hitPointParams.instSlot;
        pickInfo.geomInstSlot = hitPointParams.geomInstSlot;
        pickInfo.primIndex = hitPointParams.primIndex;
        pickInfo.positionInWorld = hitPointParams.positionInWorld;
        pickInfo.normalInWorld = hitPointParams.shadingNormalInWorld;
        pickInfo.albedo = hitPointParams.albedo;
        *plp.s->pickInfos[bufIdx] = pickInfo;
    }

    // JP: デノイザーに必要な情報を出力。
    // EN: Output information required for the denoiser.
    RGB prevAlbedoResult(0.0f, 0.0f, 0.0f);
    Normal3D prevNormalResult(0.0f, 0.0f, 0.0f);
    if (plp.f->numAccumFrames > 0) {
        prevAlbedoResult = RGB(getXYZ(plp.s->albedoAccumBuffer.read(launchIndex)));
        prevNormalResult = Normal3D(getXYZ(plp.s->normalAccumBuffer.read(launchIndex)));
    }
    const float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
    const RGB albedoResult = (1 - curWeight) * prevAlbedoResult + curWeight * hitPointParams.albedo;
    const Normal3D normalResult = (1 - curWeight) * prevNormalResult + curWeight * hitPointParams.shadingNormalInWorld;
    plp.s->albedoAccumBuffer.write(launchIndex, make_float4(albedoResult.toNative(), 1.0f));
    plp.s->normalAccumBuffer.write(launchIndex, make_float4(normalResult.toNative(), 1.0f));
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(setupGBuffers)() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = plp.f->bufferIndex;

    const auto sbtr = HitGroupSBTRecordData::get();
    const InstanceData &inst = plp.s->instanceDataBufferArray[bufIdx][optixGetInstanceId()];
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];

    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    PrimaryRayPayloadSignature::get(
        &hitPointParams, &pickInfo
#if OUTPUT_TRAVERSAL_STATS
        , nullptr
#endif
    );

    const auto hp = HitPointParameter::get();
    const uint32_t hitKind = optixGetHitKind();
    const bool isTriangleHit =
        hitKind == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE ||
        hitKind == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE;
    const bool isDispMapTriangleHit =
        hitKind == CustomHitKind_DisplacementMappedSurfaceFrontFace ||
        hitKind == CustomHitKind_DisplacementMappedSurfaceBackFace;
    const bool isShellMapTriangleHit =
        hitKind == CustomHitKind_ShellMappedSurfaceFrontFace ||
        hitKind == CustomHitKind_ShellMappedSurfaceBackFace;
    hitPointParams->instSlot = optixGetInstanceId();
    hitPointParams->meshType =
        isTriangleHit ? 0 :
        isDispMapTriangleHit ? 1 :
        2;
    hitPointParams->geomInstSlot = sbtr.geomInstSlot;
    hitPointParams->primIndex = hp.primIndex;

    Point3D positionInWorld;
    Point3D prevPositionInWorld;
    Normal3D geometricNormalInWorld;
    Normal3D shadingNormalInWorld;
    Vector3D texCoord0DirInWorld;
    Point2D texCoord;
    uint32_t materialSlot = geomInst.materialSlot;
    if (isTriangleHit || isDispMapTriangleHit || isShellMapTriangleHit) {
        const Triangle &tri = geomInst.triangleBuffer[hp.primIndex];
        const Vertex &vA = geomInst.vertexBuffer[tri.index0];
        const Vertex &vB = geomInst.vertexBuffer[tri.index1];
        const Vertex &vC = geomInst.vertexBuffer[tri.index2];
        const float bcB = hp.bcB;
        const float bcC = hp.bcC;
        const float bcA = 1 - (bcB + bcC);
        hitPointParams->qbcB = encodeBarycentric(bcB);
        hitPointParams->qbcC = encodeBarycentric(bcC);
        Normal3D geometricNormalInObj;
        Normal3D shadingNormalInObj;
        const Vector3D texCoord0DirInObj = bcA * vA.texCoord0Dir + bcB * vB.texCoord0Dir + bcC * vC.texCoord0Dir;
        if (isTriangleHit) {
            const Point3D positionInObj = bcA * vA.position + bcB * vB.position + bcC * vC.position;
            positionInWorld = transformPointFromObjectToWorldSpace(positionInObj);
            geometricNormalInObj = Normal3D(cross(vB.position - vA.position, vC.position - vA.position));
            shadingNormalInObj = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
        }
        else {
            positionInWorld = Point3D(optixGetWorldRayOrigin())
                + optixGetRayTmax() * Vector3D(optixGetWorldRayDirection());
            DisplacedSurfaceAttributes hitAttrs;
            DisplacedSurfaceAttributeSignature::get(nullptr, nullptr, &hitAttrs);
            geometricNormalInObj = hitAttrs.normalInObj;
            shadingNormalInObj = hitAttrs.normalInObj;

            if (isShellMapTriangleHit) {
                const GeometryInstanceDataForNRTDSM &nrtdsmGeomInst =
                    plp.s->geomInstNrtdsmDataBuffer[sbtr.geomInstSlot];
                hitPointParams->shellBvhGeomIndex = hitAttrs.geomIndex;
                materialSlot = nrtdsmGeomInst.materialSlots[hitAttrs.geomIndex];
            }
        }
        texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;
        prevPositionInWorld = inst.curToPrevTransform * positionInWorld;

        geometricNormalInWorld = normalize(transformNormalFromObjectToWorldSpace(geometricNormalInObj));
        shadingNormalInWorld = normalize(transformNormalFromObjectToWorldSpace(shadingNormalInObj));
        texCoord0DirInWorld = transformVectorFromObjectToWorldSpace(texCoord0DirInObj);
        texCoord0DirInWorld = normalize(
            texCoord0DirInWorld - dot(shadingNormalInWorld, texCoord0DirInWorld) * shadingNormalInWorld);
        if (!shadingNormalInWorld.allFinite()) {
            geometricNormalInWorld = Normal3D(0, 0, 1);
            shadingNormalInWorld = Normal3D(0, 0, 1);
            texCoord0DirInWorld = Vector3D(1, 0, 0);
        }
    }
    else { // for Prism debugging
        const GeometryInstanceDataForNRTDSM &nrtdsm = plp.s->geomInstNrtdsmDataBuffer[sbtr.geomInstSlot];
        const NRTDSMTriangleAuxInfo &dispTriAuxInfo = nrtdsm.dispTriAuxInfoBuffer[hp.primIndex];
        const Triangle &tri = geomInst.triangleBuffer[hp.primIndex];
        const Vertex &vA = geomInst.vertexBuffer[tri.index0];
        const Vertex &vB = geomInst.vertexBuffer[tri.index1];
        const Vertex &vC = geomInst.vertexBuffer[tri.index2];
        const float minHeight = dispTriAuxInfo.minHeight;
        const float maxHeight = minHeight + dispTriAuxInfo.amplitude;
        const Point3D pA = vA.position + minHeight * vA.normal;
        const Point3D pB = vB.position + minHeight * vB.normal;
        const Point3D pC = vC.position + minHeight * vC.normal;
        const Point3D pD = vA.position + maxHeight * vA.normal;
        const Point3D pE = vB.position + maxHeight * vB.normal;
        const Point3D pF = vC.position + maxHeight * vC.normal;
        Normal3D normalInObj;
        const Point3D positionInObj = restorePrismHitPoint(
            pA, pB, pC,
            pD, pE, pF,
            hp.bcB, hp.bcC, &normalInObj);
        positionInWorld = transformPointFromObjectToWorldSpace(positionInObj);
        geometricNormalInWorld = normalize(transformNormalFromObjectToWorldSpace(normalInObj));
        shadingNormalInWorld = geometricNormalInWorld;
        texCoord = Point2D(0.0f, 0.0f);
        Vector3D bitangent;
        makeCoordinateSystem(shadingNormalInWorld, &texCoord0DirInWorld, &bitangent);
    }

    hitPointParams->positionInWorld = positionInWorld;
    hitPointParams->prevPositionInWorld = prevPositionInWorld;
    hitPointParams->geometricNormalInWorld = geometricNormalInWorld;

    const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

    BSDF bsdf;
    bsdf.setup(mat, texCoord, 0.0f);
    const ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
    //if (plp.f->enableBumpMapping) {
    //    const Normal3D modLocalNormal = mat.readModifiedNormal(mat.normal, mat.normalDimInfo, texCoord, 0.0f);
    //    applyBumpMapping(modLocalNormal, &shadingFrame);
    //}
    const Vector3D vOut(-Vector3D(optixGetWorldRayDirection()));
    const Vector3D vOutLocal = shadingFrame.toLocal(normalize(vOut));

    hitPointParams->shadingNormalInWorld = shadingFrame.normal;
    hitPointParams->albedo = bsdf.evaluateDHReflectanceEstimate(vOutLocal);

    // JP: マウスが乗っているピクセルの情報を出力する。
    // EN: Export the information of the pixel on which the mouse is.
    if (launchIndex.x == plp.f->mousePosition.x &&
        launchIndex.y == plp.f->mousePosition.y) {
        pickInfo->hit = true;
        pickInfo->matSlot = geomInst.materialSlot;
        RGB emittance(0.0f, 0.0f, 0.0f);
        if (mat.emittance) {
            float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
            emittance = RGB(getXYZ(texValue));
        }
        pickInfo->emittance = emittance;
    }
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(setupGBuffers)() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    const Vector3D vOut(-Vector3D(optixGetWorldRayDirection()));
    const Point3D p(-vOut);

    float posPhi, posTheta;
    toPolarYUp(Vector3D(p), &posPhi, &posTheta);

    const float phi = posPhi + plp.f->envLightRotation;

    float u = phi / (2 * pi_v<float>);
    u -= floorf(u);
    const float v = posTheta / pi_v<float>;

    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    PrimaryRayPayloadSignature::get(
        &hitPointParams, &pickInfo
#if OUTPUT_TRAVERSAL_STATS
        , nullptr
#endif
    );

    hitPointParams->positionInWorld = p;
    hitPointParams->prevPositionInWorld = p;
    hitPointParams->geometricNormalInWorld = Normal3D(vOut);
    hitPointParams->shadingNormalInWorld = Normal3D(vOut);
    hitPointParams->qbcB = encodeBarycentric(u);
    hitPointParams->qbcC = encodeBarycentric(v);

    // JP: マウスが乗っているピクセルの情報を出力する。
    // EN: Export the information of the pixel on which the mouse is.
    if (launchIndex.x == plp.f->mousePosition.x &&
        launchIndex.y == plp.f->mousePosition.y) {
        pickInfo->hit = true;
        pickInfo->matSlot = 0xFFFFFFFF;
        RGB emittance(0.0f, 0.0f, 0.0f);
        if (plp.s->envLightTexture && plp.f->enableEnvLight) {
            float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
            emittance = RGB(getXYZ(texValue));
            emittance *= pi_v<float> * plp.f->envLightPowerCoeff;
        }
        pickInfo->emittance = emittance;
    }
}
