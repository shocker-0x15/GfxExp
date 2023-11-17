#include "../tfdm_shared.h"
#include "tfdm_intersection_kernels.h"

using namespace shared;

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

    Point3D origin = camera.position;
    Vector3D direction = normalize(camera.orientation * Vector3D(vw * (0.5f - x), vh * (0.5f - y), 1));

    HitPointParams hitPointParams;
    hitPointParams.positionInWorld = Point3D(NAN);
    hitPointParams.prevPositionInWorld = Point3D(NAN);
    hitPointParams.normalInWorld = Normal3D(NAN);
    hitPointParams.texCoord = Point2D(NAN);
    hitPointParams.materialSlot = 0xFFFFFFFF;
    hitPointParams.geomInstSlot = 0x7FFFFFFF;
    hitPointParams.isTfdmMesh = false;
    hitPointParams.primIndex = 0xFFFFFFFF;
#if OUTPUT_TRAVERSAL_STATS
    hitPointParams.numTravIterations = 0;
#endif

    PickInfo pickInfo = {};

    HitPointParams* hitPointParamsPtr = &hitPointParams;
    PickInfo* pickInfoPtr = &pickInfo;
    PrimaryRayPayloadSignature::trace(
        plp.f->travHandle, origin.toNative(), direction.toNative(),
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        GBufferRayType::Primary, maxNumRayTypes, GBufferRayType::Primary,
        hitPointParamsPtr, pickInfoPtr);



    Point2D curRasterPos(launchIndex.x + 0.5f, launchIndex.y + 0.5f);
    Point2D prevRasterPos =
        plp.f->prevCamera.calcScreenPosition(hitPointParams.prevPositionInWorld)
        * Point2D(plp.s->imageSize.x, plp.s->imageSize.y);
    Vector2D motionVector = curRasterPos - prevRasterPos;
    if (plp.f->resetFlowBuffer || isnan(hitPointParams.prevPositionInWorld.x))
        motionVector = Vector2D(0.0f, 0.0f);

    GBuffer0 gBuffer0;
    gBuffer0.positionInWorld = hitPointParams.positionInWorld;
    gBuffer0.texCoord_x = hitPointParams.texCoord.x;
    GBuffer1 gBuffer1;
    gBuffer1.normalInWorld = hitPointParams.normalInWorld;
    gBuffer1.texCoord_y = hitPointParams.texCoord.y;
    GBuffer2 gBuffer2;
    gBuffer2.motionVector = __float22half2_rn(make_float2(motionVector.x, motionVector.y));
    gBuffer2.materialSlot = hitPointParams.materialSlot;
    gBuffer2.geomInstSlot = hitPointParams.geomInstSlot;
    gBuffer2.isTfdmMesh = hitPointParams.isTfdmMesh;
    gBuffer2.primIndex = hitPointParams.primIndex;

    uint32_t bufIdx = plp.f->bufferIndex;
    plp.s->GBuffer0[bufIdx].write(launchIndex, gBuffer0);
    plp.s->GBuffer1[bufIdx].write(launchIndex, gBuffer1);
    plp.s->GBuffer2[bufIdx].write(launchIndex, gBuffer2);

#if OUTPUT_TRAVERSAL_STATS
    plp.s->numTravItrsBuffer.write(launchIndex, hitPointParams.numTravIterations);
#endif

    if (launchIndex.x == plp.f->mousePosition.x &&
        launchIndex.y == plp.f->mousePosition.y)
        *plp.f->pickInfo = pickInfo;

    // JP: デノイザーに必要な情報を出力。
    // EN: Output information required for the denoiser.
    Normal3D firstHitNormal = transpose(camera.orientation) * hitPointParams.normalInWorld;
    firstHitNormal.x *= -1;
    RGB prevAlbedoResult(0.0f, 0.0f, 0.0f);
    Normal3D prevNormalResult(0.0f, 0.0f, 0.0f);
    if (plp.f->numAccumFrames > 0) {
        prevAlbedoResult = RGB(getXYZ(plp.s->albedoAccumBuffer.read(launchIndex)));
        prevNormalResult = Normal3D(getXYZ(plp.s->normalAccumBuffer.read(launchIndex)));
    }
    float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
    RGB albedoResult = (1 - curWeight) * prevAlbedoResult + curWeight * hitPointParams.albedo;
    Normal3D normalResult = (1 - curWeight) * prevNormalResult + curWeight * firstHitNormal;
    plp.s->albedoAccumBuffer.write(launchIndex, make_float4(albedoResult.toNative(), 1.0f));
    plp.s->normalAccumBuffer.write(launchIndex, make_float4(normalResult.toNative(), 1.0f));
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(setupGBuffers)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    auto sbtr = HitGroupSBTRecordData::get();
    const InstanceData &inst = plp.f->instanceDataBuffer[optixGetInstanceId()];
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];
    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    PrimaryRayPayloadSignature::get(&hitPointParams, &pickInfo);

    auto hp = HitPointParameter::get();
    Point3D positionInWorld;
    Point3D prevPositionInWorld;
    Normal3D shadingNormalInWorld;
    Vector3D texCoord0DirInWorld;
    //Normal3D geometricNormalInWorld;
#if STORE_BARYCENTRICS
    Point2D texCoord(hp.b1, hp.b2);
#else
    Point2D texCoord;
#endif
    float targetMipLevel = 0;
#if OUTPUT_TRAVERSAL_STATS
    uint32_t numTravIterations = 0;
#endif
    const uint32_t hitKind = optixGetHitKind();
    const bool isTriangleHit =
        hitKind == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE
        || hitKind == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE;
    const bool isDisplacedTriangleHit =
        hitKind == CustomHitKind_DisplacedSurfaceFrontFace
        || hitKind == CustomHitKind_DisplacedSurfaceBackFace;
    if (isTriangleHit || isDisplacedTriangleHit) {
        const Triangle &tri = geomInst.triangleBuffer[hp.primIndex];
        const Vertex &v0 = geomInst.vertexBuffer[tri.index0];
        const Vertex &v1 = geomInst.vertexBuffer[tri.index1];
        const Vertex &v2 = geomInst.vertexBuffer[tri.index2];
        float b1 = hp.b1;
        float b2 = hp.b2;
        float b0 = 1 - (b1 + b2);
        texCoord0DirInWorld = b0 * v0.texCoord0Dir + b1 * v1.texCoord0Dir + b2 * v2.texCoord0Dir;
        if (isTriangleHit) {
            Point3D localP = b0 * v0.position + b1 * v1.position + b2 * v2.position;
            positionInWorld = transformPointFromObjectToWorldSpace(localP);
            shadingNormalInWorld = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
        }
        else {
            positionInWorld = Point3D(optixGetWorldRayOrigin())
                + optixGetRayTmax() * Vector3D(optixGetWorldRayDirection());
            DisplacedSurfaceAttributes hitAttrs;
            DisplacedSurfaceAttributeSignature::get(nullptr, nullptr, &hitAttrs);
            shadingNormalInWorld = hitAttrs.normalInObj;
#if OUTPUT_TRAVERSAL_STATS
            numTravIterations = hitAttrs.numIterations;
#endif
            texCoord0DirInWorld += -dot(shadingNormalInWorld, texCoord0DirInWorld) * shadingNormalInWorld;

            const GeometryInstanceDataForTFDM &tfdmGeomInst = plp.s->geomInstTfdmDataBuffer[sbtr.geomInstSlot];
            targetMipLevel = tfdmGeomInst.params.targetMipLevel;

            hitPointParams->isTfdmMesh = true;
        }
        prevPositionInWorld = inst.curToPrevTransform * positionInWorld;
        //geometricNormalInWorld = Normal3D(cross(v1.position - v0.position, v2.position - v0.position));
#if !STORE_BARYCENTRICS
        texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;
#endif

        shadingNormalInWorld = normalize(transformNormalFromObjectToWorldSpace(shadingNormalInWorld));
        texCoord0DirInWorld = normalize(transformVectorFromObjectToWorldSpace(texCoord0DirInWorld));
        //geometricNormalInWorld = normalize(transformNormalFromObjectToWorldSpace(geometricNormalInWorld));
        if (!shadingNormalInWorld.allFinite()) {
            shadingNormalInWorld = Normal3D(0, 0, 1);
            texCoord0DirInWorld = Vector3D(1, 0, 0);
        }
    }
    else { // for AABB debugging
        const GeometryInstanceDataForTFDM &tfdm = plp.s->geomInstTfdmDataBuffer[sbtr.geomInstSlot];
        const AABB &aabb = tfdm.aabbBuffer[hp.primIndex];
        Normal3D n;
        Point3D p = aabb.restoreHitPoint(hp.b1, hp.b2, &n);
        positionInWorld = transformPointFromObjectToWorldSpace(p);
        shadingNormalInWorld = normalize(transformNormalFromObjectToWorldSpace(n));
        //geometricNormalInWorld = shadingNormalInWorld;
        texCoord = Point2D(0.0f, 0.0f);
        Vector3D bitangent;
        makeCoordinateSystem(shadingNormalInWorld, &texCoord0DirInWorld, &bitangent);
    }

    BSDF bsdf;
    bsdf.setup(mat, texCoord, targetMipLevel);
    ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
    if (plp.f->enableBumpMapping) {
        Normal3D modLocalNormal = mat.readModifiedNormal(
            mat.normal, mat.normalDimInfo, texCoord, targetMipLevel);
        applyBumpMapping(modLocalNormal, &shadingFrame);
    }
    Vector3D vOut(-Vector3D(optixGetWorldRayDirection()));
    Vector3D vOutLocal = shadingFrame.toLocal(normalize(vOut));

    hitPointParams->albedo = bsdf.evaluateDHReflectanceEstimate(vOutLocal);
    hitPointParams->positionInWorld = positionInWorld;
    hitPointParams->prevPositionInWorld = prevPositionInWorld;
    hitPointParams->normalInWorld = shadingFrame.normal;
    hitPointParams->texCoord = texCoord;
    hitPointParams->materialSlot = geomInst.materialSlot;
    hitPointParams->geomInstSlot = geomInst.geomInstSlot;
    hitPointParams->primIndex = hp.primIndex;
#if OUTPUT_TRAVERSAL_STATS
    hitPointParams->numTravIterations = numTravIterations;
#endif

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
        RGB emittance(0.0f, 0.0f, 0.0f);
        if (mat.emittance) {
            float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
            emittance = RGB(getXYZ(texValue));
        }
        pickInfo->emittance = emittance;
    }
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(setupGBuffers)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    Vector3D vOut(-Vector3D(optixGetWorldRayDirection()));
    Point3D p(-vOut);

    float posPhi, posTheta;
    toPolarYUp(Vector3D(p), &posPhi, &posTheta);

    float phi = posPhi + plp.f->envLightRotation;

    float u = phi / (2 * Pi);
    u -= floorf(u);
    float v = posTheta / Pi;

    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    PrimaryRayPayloadSignature::get(&hitPointParams, &pickInfo);

    hitPointParams->albedo = RGB(0.0f, 0.0f, 0.0f);
    hitPointParams->positionInWorld = p;
    hitPointParams->prevPositionInWorld = p;
    hitPointParams->normalInWorld = Normal3D(vOut);
    hitPointParams->texCoord = Point2D(u, v);
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
        pickInfo->albedo = RGB(0.0f, 0.0f, 0.0f);
        RGB emittance(0.0f, 0.0f, 0.0f);
        if (plp.s->envLightTexture && plp.f->enableEnvLight) {
            float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
            emittance = RGB(getXYZ(texValue));
            emittance *= Pi * plp.f->envLightPowerCoeff;
        }
        pickInfo->emittance = emittance;
        pickInfo->normalInWorld = Normal3D(vOut);
    }
}
