#include "../svgf_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void RT_RG_NAME(pick)() {
    const uint32_t curBufIdx = plp.f->bufferIndex;
    const PerFramePipelineLaunchParameters::TemporalSet &curPerFrameTemporalSet =
        plp.f->temporalSets[curBufIdx];

    const PerspectiveCamera &camera = curPerFrameTemporalSet.camera;
    const Point2D fPix(
        plp.f->mousePosition.x + camera.subPixelOffset.x,
        plp.f->mousePosition.y + 1 - camera.subPixelOffset.y);
    const float x = fPix.x / plp.s->imageSize.x;
    const float y = fPix.y / plp.s->imageSize.y;
    const float vh = 2 * std::tan(camera.fovY * 0.5f);
    const float vw = camera.aspect * vh;

    const Point3D origin = camera.position;
    const Vector3D direction = normalize(camera.orientation * Vector3D(vw * (0.5f - x), vh * (0.5f - y), 1));

    HitPointParams hitPointParams;
    hitPointParams.positionInWorld = Point3D(NAN);
    hitPointParams.prevPositionInWorld = Point3D(NAN);
    hitPointParams.normalInWorld = Normal3D(NAN);
    hitPointParams.texCoord = Point2D(NAN);
    hitPointParams.materialSlot = 0xFFFFFFFF;

    PickInfo pickInfo = {};

    PickInfo* pickInfoPtr = &pickInfo;
    PickRayPayloadSignature::trace(
        plp.f->travHandle, origin.toNative(), direction.toNative(),
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        PickRayType::Primary, maxNumRayTypes, PickRayType::Primary,
        pickInfoPtr);

    *plp.s->pickInfos[curBufIdx] = pickInfo;
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pick)() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = plp.f->bufferIndex;

    const auto sbtr = HitGroupSBTRecordData::get();
    const InstanceData &inst = plp.s->instanceDataBufferArray[bufIdx][optixGetInstanceId()];
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];

    PickInfo* pickInfo;
    PickRayPayloadSignature::get(&pickInfo);

    const auto hp = HitPointParameter::get();
    Point3D positionInWorld;
    Normal3D shadingNormalInWorld;
    Vector3D texCoord0DirInWorld;
    //Normal3D geometricNormalInWorld;
    Point2D texCoord;
    {
        const Triangle &tri = geomInst.triangleBuffer[hp.primIndex];
        const Vertex &vA = geomInst.vertexBuffer[tri.index0];
        const Vertex &vB = geomInst.vertexBuffer[tri.index1];
        const Vertex &vC = geomInst.vertexBuffer[tri.index2];
        const float bcB = hp.bcB;
        const float bcC = hp.bcC;
        const float bcA = 1 - (bcB + bcC);
        const Point3D localP = bcA * vA.position + bcB * vB.position + bcC * vC.position;
        shadingNormalInWorld = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
        texCoord0DirInWorld = bcA * vA.texCoord0Dir + bcB * vB.texCoord0Dir + bcC * vC.texCoord0Dir;
        //geometricNormalInWorld = cross(vB.position - vA.position, vC.position - vA.position);
        texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;

        positionInWorld = transformPointFromObjectToWorldSpace(localP);
        shadingNormalInWorld = normalize(transformNormalFromObjectToWorldSpace(shadingNormalInWorld));
        texCoord0DirInWorld = normalize(transformVectorFromObjectToWorldSpace(texCoord0DirInWorld));
        //geometricNormalInWorld = normalize(transformNormalFromObjectToWorldSpace(geometricNormalInWorld));
        if (!shadingNormalInWorld.allFinite()) {
            shadingNormalInWorld = Normal3D(0, 0, 1);
            texCoord0DirInWorld = Vector3D(1, 0, 0);
        }
    }

    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    BSDF bsdf;
    bsdf.setup(mat, texCoord, 0.0f);
    ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
    if (plp.f->enableBumpMapping) {
        Normal3D modLocalNormal = mat.readModifiedNormal(mat.normal, mat.normalDimInfo, texCoord, 0.0f);
        applyBumpMapping(modLocalNormal, &shadingFrame);
    }
    const Vector3D vOut(-Vector3D(optixGetWorldRayDirection()));
    const Vector3D vOutLocal = shadingFrame.toLocal(normalize(vOut));
    const RGB dhReflectance = bsdf.evaluateDHReflectanceEstimate(vOutLocal);

    pickInfo->hit = true;
    pickInfo->instSlot = optixGetInstanceId();
    pickInfo->geomInstSlot = geomInst.geomInstSlot;
    pickInfo->matSlot = geomInst.materialSlot;
    pickInfo->primIndex = hp.primIndex;
    pickInfo->positionInWorld = positionInWorld;
    pickInfo->normalInWorld = shadingFrame.normal;
    pickInfo->albedo = dhReflectance;
    RGB emittance(0.0f, 0.0f, 0.0f);
    if (mat.emittance) {
        const float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
        emittance = RGB(getXYZ(texValue));
    }
    pickInfo->emittance = emittance;
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(pick)() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    const Vector3D vOut(-Vector3D(optixGetWorldRayDirection()));
    const Point3D p(-vOut);

    float posPhi, posTheta;
    toPolarYUp(Vector3D(p), &posPhi, &posTheta);

    float phi = posPhi + plp.f->envLightRotation;

    float u = phi / (2 * Pi);
    u -= floorf(u);
    const float v = posTheta / Pi;

    PickInfo* pickInfo;
    PickRayPayloadSignature::get(&pickInfo);

    pickInfo->hit = true;
    pickInfo->instSlot = 0xFFFFFFFF;
    pickInfo->geomInstSlot = 0xFFFFFFFF;
    pickInfo->matSlot = 0xFFFFFFFF;
    pickInfo->primIndex = 0xFFFFFFFF;
    pickInfo->positionInWorld = p;
    pickInfo->albedo = RGB(0.0f, 0.0f, 0.0f);
    RGB emittance(0.0f, 0.0f, 0.0f);
    if (plp.s->envLightTexture && plp.f->enableEnvLight) {
        const float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
        emittance = RGB(getXYZ(texValue));
        emittance *= Pi * plp.f->envLightPowerCoeff;
    }
    pickInfo->emittance = emittance;
    pickInfo->normalInWorld = Normal3D(vOut);
}
