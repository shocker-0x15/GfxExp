#include "../svgf_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void RT_RG_NAME(pick)() {
    uint32_t curBufIdx = plp.f->bufferIndex;
    const PerFramePipelineLaunchParameters::TemporalSet &curPerFrameTemporalSet =
        plp.f->temporalSets[curBufIdx];

    const PerspectiveCamera &camera = curPerFrameTemporalSet.camera;
    float2 fPix = make_float2(plp.f->mousePosition.x + camera.subPixelOffset.x,
                              plp.f->mousePosition.y + camera.subPixelOffset.y);
    float x = fPix.x / plp.s->imageSize.x;
    float y = fPix.y / plp.s->imageSize.y;
    float vh = 2 * std::tan(camera.fovY * 0.5f);
    float vw = camera.aspect * vh;

    float3 origin = camera.position;
    float3 direction = normalize(camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    HitPointParams hitPointParams;
    hitPointParams.positionInWorld = make_float3(NAN);
    hitPointParams.prevPositionInWorld = make_float3(NAN);
    hitPointParams.normalInWorld = make_float3(NAN);
    hitPointParams.texCoord = make_float2(NAN);
    hitPointParams.materialSlot = 0xFFFFFFFF;

    PickInfo pickInfo = {};

    PickInfo* pickInfoPtr = &pickInfo;
    optixu::trace<PickRayPayloadSignature>(
        plp.f->travHandle, origin, direction,
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
    float3 positionInWorld;
    float3 shadingNormalInWorld;
    float3 texCoord0DirInWorld;
    //float3 geometricNormalInWorld;
    float2 texCoord;
    {
        const Triangle &tri = geomInst.triangleBuffer[hp.primIndex];
        const Vertex &v0 = geomInst.vertexBuffer[tri.index0];
        const Vertex &v1 = geomInst.vertexBuffer[tri.index1];
        const Vertex &v2 = geomInst.vertexBuffer[tri.index2];
        float b1 = hp.b1;
        float b2 = hp.b2;
        float b0 = 1 - (b1 + b2);
        float3 localP = b0 * v0.position + b1 * v1.position + b2 * v2.position;
        shadingNormalInWorld = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
        texCoord0DirInWorld = b0 * v0.texCoord0Dir + b1 * v1.texCoord0Dir + b2 * v2.texCoord0Dir;
        //geometricNormalInWorld = cross(v1.position - v0.position, v2.position - v0.position);
        texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

        positionInWorld = optixTransformPointFromObjectToWorldSpace(localP);
        shadingNormalInWorld = normalize(optixTransformNormalFromObjectToWorldSpace(shadingNormalInWorld));
        texCoord0DirInWorld = normalize(optixTransformVectorFromObjectToWorldSpace(texCoord0DirInWorld));
        //geometricNormalInWorld = normalize(optixTransformNormalFromObjectToWorldSpace(geometricNormalInWorld));
        if (!allFinite(shadingNormalInWorld)) {
            shadingNormalInWorld = make_float3(0, 0, 1);
            texCoord0DirInWorld = make_float3(1, 0, 0);
        }
    }

    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    BSDF bsdf;
    bsdf.setup(mat, texCoord);
    ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
    if (plp.f->enableBumpMapping) {
        float3 modLocalNormal = mat.readModifiedNormal(mat.normal, mat.normalDimInfo, texCoord);
        applyBumpMapping(modLocalNormal, &shadingFrame);
    }
    float3 vOut = -optixGetWorldRayDirection();
    float3 vOutLocal = shadingFrame.toLocal(normalize(vOut));
    float3 dhReflectance = bsdf.evaluateDHReflectanceEstimate(vOutLocal);

    pickInfo->hit = true;
    pickInfo->instSlot = optixGetInstanceId();
    pickInfo->geomInstSlot = geomInst.geomInstSlot;
    pickInfo->matSlot = geomInst.materialSlot;
    pickInfo->primIndex = hp.primIndex;
    pickInfo->positionInWorld = positionInWorld;
    pickInfo->normalInWorld = shadingFrame.normal;
    pickInfo->albedo = dhReflectance;
    float3 emittance = make_float3(0.0f, 0.0f, 0.0f);
    if (mat.emittance) {
        float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
        emittance = make_float3(texValue);
    }
    pickInfo->emittance = emittance;
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(pick)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    float3 vOut = -optixGetWorldRayDirection();
    float3 p = -vOut;

    float posPhi, posTheta;
    toPolarYUp(p, &posPhi, &posTheta);

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
    pickInfo->albedo = make_float3(0.0f, 0.0f, 0.0f);
    float3 emittance = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.s->envLightTexture && plp.f->enableEnvLight) {
        float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
        emittance = make_float3(texValue);
        emittance *= Pi * plp.f->envLightPowerCoeff;
    }
    pickInfo->emittance = emittance;
    pickInfo->normalInWorld = vOut;
}
