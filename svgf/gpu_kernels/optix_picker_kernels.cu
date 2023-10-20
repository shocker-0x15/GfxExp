#include "../svgf_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void RT_RG_NAME(pick)() {
    uint32_t curBufIdx = plp.f->bufferIndex;
    const PerFramePipelineLaunchParameters::TemporalSet &curPerFrameTemporalSet =
        plp.f->temporalSets[curBufIdx];

    const PerspectiveCamera &camera = curPerFrameTemporalSet.camera;
    Point2D fPix(plp.f->mousePosition.x + camera.subPixelOffset.x,
                 plp.f->mousePosition.y + 1 - camera.subPixelOffset.y);
    float x = fPix.x / plp.s->imageSize.x;
    float y = fPix.y / plp.s->imageSize.y;
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

    PickInfo pickInfo = {};

    PickInfo* pickInfoPtr = &pickInfo;
    PickRayPayloadSignature::trace(
        plp.f->travHandle, origin.toNative(), direction.toNative(),
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        PickRayType::Primary, maxNumRayTypes, PickRayType::Primary,
        pickInfoPtr);

    *plp.f->pickInfo = pickInfo;
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pick)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    auto sbtr = HitGroupSBTRecordData::get();
    const InstanceData &inst = plp.f->instanceDataBuffer[optixGetInstanceId()];
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];

    PickInfo* pickInfo;
    PickRayPayloadSignature::get(&pickInfo);

    auto hp = HitPointParameter::get();
    Point3D positionInWorld;
    Normal3D shadingNormalInWorld;
    Vector3D texCoord0DirInWorld;
    //Normal3D geometricNormalInWorld;
    Point2D texCoord;
    {
        const Triangle &tri = geomInst.triangleBuffer[hp.primIndex];
        const Vertex &v0 = geomInst.vertexBuffer[tri.index0];
        const Vertex &v1 = geomInst.vertexBuffer[tri.index1];
        const Vertex &v2 = geomInst.vertexBuffer[tri.index2];
        float b1 = hp.b1;
        float b2 = hp.b2;
        float b0 = 1 - (b1 + b2);
        Point3D localP = b0 * v0.position + b1 * v1.position + b2 * v2.position;
        shadingNormalInWorld = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
        texCoord0DirInWorld = b0 * v0.texCoord0Dir + b1 * v1.texCoord0Dir + b2 * v2.texCoord0Dir;
        //geometricNormalInWorld = cross(v1.position - v0.position, v2.position - v0.position);
        texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

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
    Vector3D vOut(-Vector3D(optixGetWorldRayDirection()));
    Vector3D vOutLocal = shadingFrame.toLocal(normalize(vOut));
    RGB dhReflectance = bsdf.evaluateDHReflectanceEstimate(vOutLocal);

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
        float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
        emittance = RGB(getXYZ(texValue));
    }
    pickInfo->emittance = emittance;
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(pick)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    Vector3D vOut(-Vector3D(optixGetWorldRayDirection()));
    Point3D p(-vOut);

    float posPhi, posTheta;
    toPolarYUp(Vector3D(p), &posPhi, &posTheta);

    float phi = posPhi + plp.f->envLightRotation;

    float u = phi / (2 * Pi);
    u -= floorf(u);
    float v = posTheta / Pi;

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
        float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
        emittance = RGB(getXYZ(texValue));
        emittance *= Pi * plp.f->envLightPowerCoeff;
    }
    pickInfo->emittance = emittance;
    pickInfo->normalInWorld = Normal3D(vOut);
}
