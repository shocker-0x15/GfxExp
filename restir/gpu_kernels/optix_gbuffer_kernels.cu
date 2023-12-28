#include "../restir_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void RT_RG_NAME(setupGBuffers)() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = plp.f->bufferIndex;

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
    hitPointParams.qGeometricNormalInWorld = 0;
    hitPointParams.shadingNormalInWorld = Normal3D(NAN);
    hitPointParams.qTexCoord0DirInWorld = 0;
    hitPointParams.qTexCoord = 0;
    hitPointParams.matSlot = 0xFFFFFFFF;
    hitPointParams.instSlot = 0xFFFFFFFF;
    hitPointParams.geomInstSlot = 0xFFFFFFFF;
    hitPointParams.primIndex = 0xFFFFFFFF;
    hitPointParams.qbcB = 0;
    hitPointParams.qbcC = 0;

    PickInfo pickInfo = {};

    HitPointParams* hitPointParamsPtr = &hitPointParams;
    PickInfo* pickInfoPtr = &pickInfo;
    PrimaryRayPayloadSignature::trace(
        plp.f->travHandle, origin.toNative(), direction.toNative(),
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        GBufferRayType::Primary, maxNumRayTypes, GBufferRayType::Primary,
        hitPointParamsPtr, pickInfoPtr);



    const Point2D curRasterPos(launchIndex.x + 0.5f, launchIndex.y + 0.5f);
    const Point2D prevRasterPos =
        plp.f->prevCamera.calcScreenPosition(hitPointParams.prevPositionInWorld)
        * Point2D(plp.s->imageSize.x, plp.s->imageSize.y);
    Vector2D motionVector = curRasterPos - prevRasterPos;
    if (plp.f->resetFlowBuffer || isnan(hitPointParams.prevPositionInWorld.x))
        motionVector = Vector2D(0.0f, 0.0f);

    GBuffer0Elements gb0Elems = {};
    gb0Elems.instSlot = hitPointParams.instSlot;
    gb0Elems.geomInstSlot = hitPointParams.geomInstSlot;
    gb0Elems.primIndex = hitPointParams.primIndex;
    gb0Elems.qbcB = hitPointParams.qbcB;
    gb0Elems.qbcC = hitPointParams.qbcC;
    GBuffer1Elements gb1Elems = {};
    gb1Elems.motionVector = motionVector;
    GBuffer2Elements gb2Elems = {};
    gb2Elems.positionInWorld = hitPointParams.positionInWorld;
    gb2Elems.qGeometricNormal = hitPointParams.qGeometricNormalInWorld;
    GBuffer3Elements gb3Elems = {};
    gb3Elems.qShadingNormal = encodeNormal(hitPointParams.shadingNormalInWorld);
    gb3Elems.qShadingTangent = hitPointParams.qTexCoord0DirInWorld;
    gb3Elems.qTexCoord = hitPointParams.qTexCoord;
    gb3Elems.matSlot = hitPointParams.matSlot;

    plp.s->GBuffer0[bufIdx].write(launchIndex, gb0Elems);
    plp.s->GBuffer1[bufIdx].write(launchIndex, gb1Elems);
    plp.s->GBuffer2[bufIdx].write(launchIndex, gb2Elems);
    plp.s->GBuffer3[bufIdx].write(launchIndex, gb3Elems);

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
    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    PrimaryRayPayloadSignature::get(&hitPointParams, &pickInfo);

    const auto hp = HitPointParameter::get();
    hitPointParams->matSlot = geomInst.materialSlot;
    hitPointParams->instSlot = optixGetInstanceId();
    hitPointParams->geomInstSlot = sbtr.geomInstSlot;
    hitPointParams->primIndex = hp.primIndex;

    Point3D positionInWorld;
    Point3D prevPositionInWorld;
    Normal3D geometricNormalInWorld;
    Normal3D shadingNormalInWorld;
    Vector3D texCoord0DirInWorld;
    Point2D texCoord;
    {
        const Triangle &tri = geomInst.triangleBuffer[hp.primIndex];
        const Vertex &vA = geomInst.vertexBuffer[tri.index0];
        const Vertex &vB = geomInst.vertexBuffer[tri.index1];
        const Vertex &vC = geomInst.vertexBuffer[tri.index2];
        const float bcB = hp.bcB;
        const float bcC = hp.bcC;
        const float bcA = 1 - (bcB + bcC);
        hitPointParams->qbcB = encodeBarycentric(bcB);
        hitPointParams->qbcC = encodeBarycentric(bcC);
        const Point3D positionInObj = bcA * vA.position + bcB * vB.position + bcC * vC.position;
        const Normal3D shadingNormalInObj = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
        const Vector3D texCoord0DirInObj = bcA * vA.texCoord0Dir + bcB * vB.texCoord0Dir + bcC * vC.texCoord0Dir;
        texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;
        const Normal3D geometricNormalInObj = Normal3D(cross(vB.position - vA.position, vC.position - vA.position));

        positionInWorld = transformPointFromObjectToWorldSpace(positionInObj);
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

    hitPointParams->positionInWorld = positionInWorld;
    hitPointParams->prevPositionInWorld = prevPositionInWorld;
    hitPointParams->qGeometricNormalInWorld = encodeNormal(geometricNormalInWorld);
    hitPointParams->qTexCoord = encodeTexCoords(texCoord);

    BSDF bsdf;
    bsdf.setup(mat, texCoord, 0.0f);
    ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
    if (plp.f->enableBumpMapping) {
        const Normal3D modLocalNormal = mat.readModifiedNormal(mat.normal, mat.normalDimInfo, texCoord, 0.0f);
        applyBumpMapping(modLocalNormal, &shadingFrame);
    }
    const Vector3D vOut(-Vector3D(optixGetWorldRayDirection()));
    const Vector3D vOutLocal = shadingFrame.toLocal(normalize(vOut));

    hitPointParams->shadingNormalInWorld = shadingFrame.normal;
    hitPointParams->qTexCoord0DirInWorld = encodeVector(shadingFrame.tangent);
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
    PrimaryRayPayloadSignature::get(&hitPointParams, &pickInfo);

    hitPointParams->positionInWorld = p;
    hitPointParams->prevPositionInWorld = p;
    hitPointParams->qGeometricNormalInWorld = encodeNormal(Normal3D(vOut));
    hitPointParams->shadingNormalInWorld = Normal3D(vOut);
    hitPointParams->qTexCoord0DirInWorld = encodeVector(Vector3D(-std::cos(posPhi), 0, -std::sin(posPhi)));
    hitPointParams->qTexCoord = encodeTexCoords(Point2D(u, v));
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
