#include "regir_shared.h"
#include "../common/common_device.cuh"

using namespace shared;

RT_PIPELINE_LAUNCH_PARAMETERS PipelineLaunchParameters plp;



struct BSDF {
    static constexpr uint32_t NumDwords = 16;
    BSDFSampleThroughput m_sampleThroughput;
    BSDFEvaluate m_evaluate;
    BSDFEvaluatePDF m_evaluatePDF;
    BSDFEvaluateDHReflectanceEstimate m_evaluateDHReflectanceEstimate;
    uint32_t m_data[NumDwords];

    CUDA_DEVICE_FUNCTION float3 sampleThroughput(const float3 &vGiven, float uDir0, float uDir1,
                                                 float3* vSampled, float* dirPDensity) const {
        return m_sampleThroughput(m_data, vGiven, uDir0, uDir1, vSampled, dirPDensity);
    }
    CUDA_DEVICE_FUNCTION float3 evaluate(const float3 &vGiven, const float3 &vSampled) const {
        return m_evaluate(m_data, vGiven, vSampled);
    }
    CUDA_DEVICE_FUNCTION float evaluatePDF(const float3 &vGiven, const float3 &vSampled) const {
        return m_evaluatePDF(m_data, vGiven, vSampled);
    }
    CUDA_DEVICE_FUNCTION float3 evaluateDHReflectanceEstimate(const float3 &vGiven) const {
        return m_evaluateDHReflectanceEstimate(m_data, vGiven);
    }
};

template <typename BSDFType>
CUDA_DEVICE_FUNCTION void setupBSDF(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf);

template<>
CUDA_DEVICE_FUNCTION void setupBSDF<LambertBRDF>(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf) {
    float4 reflectance = tex2DLod<float4>(matData.asLambert.reflectance, texCoord.x, texCoord.y, 0.0f);
    auto &bsdfBody = *reinterpret_cast<LambertBRDF*>(bsdf->m_data);
    bsdfBody = LambertBRDF(make_float3(reflectance.x, reflectance.y, reflectance.z));
}

template<>
CUDA_DEVICE_FUNCTION void setupBSDF<DiffuseAndSpecularBRDF>(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf) {
    float4 diffuseColor = tex2DLod<float4>(matData.asDiffuseAndSpecular.diffuse, texCoord.x, texCoord.y, 0.0f);
    float4 specularF0Color = tex2DLod<float4>(matData.asDiffuseAndSpecular.specular, texCoord.x, texCoord.y, 0.0f);
    float smoothness = tex2DLod<float>(matData.asDiffuseAndSpecular.smoothness, texCoord.x, texCoord.y, 0.0f);
    auto &bsdfBody = *reinterpret_cast<DiffuseAndSpecularBRDF*>(bsdf->m_data);
    bsdfBody = DiffuseAndSpecularBRDF(make_float3(diffuseColor),
                                      make_float3(specularF0Color),
                                      min(smoothness, 0.999f));
}



#define DEFINE_BSDF_CALLABLES(BSDFType) \
    RT_CALLABLE_PROGRAM void RT_DC_NAME(setup ## BSDFType)(\
        const MaterialData &matData, const float2 &texCoord, BSDF* bsdf) {\
        bsdf->m_sampleThroughput = matData.bsdfSampleThroughput;\
        bsdf->m_evaluate = matData.bsdfEvaluate;\
        bsdf->m_evaluatePDF = matData.bsdfEvaluatePDF;\
        bsdf->m_evaluateDHReflectanceEstimate = matData.bsdfEvaluateDHReflectanceEstimate;\
        return setupBSDF<BSDFType>(matData, texCoord, bsdf);\
    }\
    RT_CALLABLE_PROGRAM float3 RT_DC_NAME(BSDFType ## _sampleThroughput)(\
        const uint32_t* data, const float3 &vGiven, float uDir0, float uDir1,\
        float3* vSampled, float* dirPDensity) {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.sampleThroughput(vGiven, uDir0, uDir1, vSampled, dirPDensity);\
    }\
    RT_CALLABLE_PROGRAM float3 RT_DC_NAME(BSDFType ## _evaluate)(\
        const uint32_t* data, const float3 &vGiven, const float3 &vSampled) {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluate(vGiven, vSampled);\
    }\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(BSDFType ## _evaluatePDF)(\
        const uint32_t* data, const float3 &vGiven, const float3 &vSampled) {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluatePDF(vGiven, vSampled);\
    }\
    RT_CALLABLE_PROGRAM float3 RT_DC_NAME(BSDFType ## _evaluateDHReflectanceEstimate)(\
        const uint32_t* data, const float3 &vGiven) {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluateDHReflectanceEstimate(vGiven);\
    }

DEFINE_BSDF_CALLABLES(LambertBRDF);
DEFINE_BSDF_CALLABLES(DiffuseAndSpecularBRDF);



struct HitPointParameter {
    float b1, b2;
    int32_t primIndex;

    CUDA_DEVICE_FUNCTION static HitPointParameter get() {
        HitPointParameter ret;
        float2 bc = optixGetTriangleBarycentrics();
        ret.b1 = bc.x;
        ret.b2 = bc.y;
        ret.primIndex = optixGetPrimitiveIndex();
        return ret;
    }
};

struct HitGroupSBTRecordData {
    GeometryInstanceData geomInstData;

    CUDA_DEVICE_FUNCTION static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    optixu::setPayloads<VisibilityRayPayloadSignature>(&visibility);
}



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

    float3 origin = camera.position;
    float3 direction = normalize(camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    HitPointParams hitPointParams;
    hitPointParams.positionInWorld = make_float3(NAN);
    hitPointParams.prevPositionInWorld = make_float3(NAN);
    hitPointParams.normalInWorld = make_float3(NAN);
    hitPointParams.texCoord = make_float2(NAN);
    hitPointParams.materialSlot = 0xFFFFFFFF;

    PickInfo pickInfo = {};

    HitPointParams* hitPointParamsPtr = &hitPointParams;
    PickInfo* pickInfoPtr = &pickInfo;
    optixu::trace<PrimaryRayPayloadSignature>(
        plp.f->travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        hitPointParamsPtr, pickInfoPtr);



    float2 curRasterPos = make_float2(launchIndex.x + 0.5f, launchIndex.y + 0.5f);
    float2 prevRasterPos =
        plp.f->prevCamera.calcScreenPosition(hitPointParams.prevPositionInWorld)
        * make_float2(plp.s->imageSize.x, plp.s->imageSize.y);
    float2 motionVector = curRasterPos - prevRasterPos;
    if (plp.f->resetFlowBuffer || isnan(hitPointParams.prevPositionInWorld.x))
        motionVector = make_float2(0.0f, 0.0f);

    GBuffer0 gBuffer0;
    gBuffer0.positionInWorld = hitPointParams.positionInWorld;
    gBuffer0.texCoord_x = hitPointParams.texCoord.x;
    GBuffer1 gBuffer1;
    gBuffer1.normalInWorld = hitPointParams.normalInWorld;
    gBuffer1.texCoord_y = hitPointParams.texCoord.y;
    GBuffer2 gBuffer2;
    gBuffer2.motionVector = motionVector;
    gBuffer2.materialSlot = hitPointParams.materialSlot;

    uint32_t bufIdx = plp.f->bufferIndex;
    plp.s->GBuffer0[bufIdx].write(launchIndex, gBuffer0);
    plp.s->GBuffer1[bufIdx].write(launchIndex, gBuffer1);
    plp.s->GBuffer2[bufIdx].write(launchIndex, gBuffer2);

    if (launchIndex.x == plp.f->mousePosition.x &&
        launchIndex.y == plp.f->mousePosition.y)
        *plp.f->pickInfo = pickInfo;
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(setupGBuffers)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    auto sbtr = HitGroupSBTRecordData::get();
    const InstanceData &inst = plp.f->instanceDataBuffer[optixGetInstanceId()];
    const GeometryInstanceData &geomInst = sbtr.geomInstData;

    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    optixu::getPayloads<PrimaryRayPayloadSignature>(&hitPointParams, &pickInfo);

    auto hp = HitPointParameter::get();
    float3 p;
    float3 prevP;
    float3 sn;
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
        sn = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
        //sn = cross(v1.position - v0.position,
        //           v2.position - v0.position);
        texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

        p = optixTransformPointFromObjectToWorldSpace(localP);
        prevP = inst.prevTransform * localP;
        sn = normalize(optixTransformNormalFromObjectToWorldSpace(sn));
        if (!allFinite(sn))
            sn = make_float3(0, 0, 1);
    }

    hitPointParams->positionInWorld = p;
    hitPointParams->prevPositionInWorld = prevP;
    hitPointParams->normalInWorld = sn;
    hitPointParams->texCoord = texCoord;
    hitPointParams->materialSlot = geomInst.materialSlot;

    // JP: マウスが乗っているピクセルの情報を出力する。
    // EN: Export the information of the pixel on which the mouse is.
    if (launchIndex.x == plp.f->mousePosition.x &&
        launchIndex.y == plp.f->mousePosition.y) {
        float3 vOut = -optixGetWorldRayDirection();

        const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];
        BSDF bsdf;
        mat.setupBSDF(mat, texCoord, &bsdf);
        ReferenceFrame shadingFrame(sn);
        float3 vOutLocal = shadingFrame.toLocal(normalize(vOut));

        pickInfo->hit = true;
        pickInfo->instSlot = optixGetInstanceId();
        pickInfo->geomInstSlot = geomInst.geomInstSlot;
        pickInfo->matSlot = geomInst.materialSlot;
        pickInfo->primIndex = hp.primIndex;
        pickInfo->positionInWorld = p;
        pickInfo->albedo = bsdf.evaluateDHReflectanceEstimate(vOutLocal);
        float3 emittance = make_float3(0.0f, 0.0f, 0.0f);
        if (mat.emittance) {
            float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
            emittance = make_float3(texValue);
        }
        pickInfo->emittance = emittance;
        pickInfo->normalInWorld = sn;
    }
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(setupGBuffers)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    float3 vOut = -optixGetWorldRayDirection();
    float3 p = -vOut;

    float posPhi, posTheta;
    toPolarYUp(p, &posPhi, &posTheta);

    float phi = posPhi + plp.f->envLightRotation;

    float u = phi / (2 * Pi);
    u -= floorf(u);
    float v = posTheta / Pi;

    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    optixu::getPayloads<PrimaryRayPayloadSignature>(&hitPointParams, &pickInfo);

    hitPointParams->positionInWorld = p;
    hitPointParams->prevPositionInWorld = p;
    hitPointParams->normalInWorld = vOut;
    hitPointParams->texCoord = make_float2(u, v);
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
}



CUDA_DEVICE_FUNCTION float3 sampleLight(
    float ul, bool sampleEnvLight, float u0, float u1,
    LightSample* lightSample, float3* lightPosition, float3* lightNormal, float* areaPDensity) {
    CUtexObject texEmittance = 0;
    float3 emittance = make_float3(0.0f, 0.0f, 0.0f);
    float2 texCoord;
    if (sampleEnvLight) {
        lightSample->instIndex = 0xFFFFFFFF;
        lightSample->geomInstIndex = 0xFFFFFFFF;
        lightSample->primIndex = 0xFFFFFFFF;

        float u, v;
        float uvPDF;
        plp.s->envLightImportanceMap.sample(u0, u1, &u, &v, &uvPDF);
        float phi = 2 * Pi * u;
        float theta = Pi * v;
        lightSample->b1 = phi;
        lightSample->b2 = theta;

        float posPhi = phi - plp.f->envLightRotation;
        posPhi = posPhi - floorf(posPhi / (2 * Pi)) * 2 * Pi;

        float3 direction = fromPolarYUp(posPhi, theta);
        float3 position = make_float3(direction.x, direction.y, direction.z);
        *lightPosition = position;

        *lightNormal = -position;

        // JP: テクスチャー空間中のPDFを面積に関するものに変換する。
        // EN: convert the PDF in texture space to one with respect to area.
        // The true value is: lim_{l to inf} uvPDF / (2 * Pi * Pi * std::sin(theta)) / l^2
        *areaPDensity = uvPDF / (2 * Pi * Pi * std::sin(theta));

        texEmittance = plp.s->envLightTexture;
        emittance = make_float3(Pi * plp.f->envLightPowerCoeff);
        texCoord.x = u;
        texCoord.y = v;
    }
    else {
        float lightProb = 1.0f;

        // JP: まずはインスタンスをサンプルする。
        // EN: First, sample an instance.
        float instProb;
        float uGeomInst;
        uint32_t instIndex = plp.s->lightInstDist.sample(ul, &instProb, &uGeomInst);
        lightProb *= instProb;
        const InstanceData &inst = plp.f->instanceDataBuffer[instIndex];
        lightSample->instIndex = instIndex;

        // JP: 次にサンプルしたインスタンスに属するジオメトリインスタンスをサンプルする。
        // EN: Next, sample a geometry instance which belongs to the sampled instance.
        float geomInstProb;
        float uPrim;
        uint32_t geomInstIndexInInst = inst.lightGeomInstDist.sample(uGeomInst, &geomInstProb, &uPrim);
        uint32_t geomInstIndex = inst.geomInstSlots[geomInstIndexInInst];
        lightProb *= geomInstProb;
        const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[geomInstIndex];
        lightSample->geomInstIndex = geomInstIndex;

        // JP: 最後に、サンプルしたジオメトリインスタンスに属するプリミティブをサンプルする。
        // EN: Finally, sample a primitive which belongs to the sampled geometry instance.
        float primProb;
        uint32_t primIndex = geomInst.emitterPrimDist.sample(uPrim, &primProb);
        lightProb *= primProb;
        lightSample->primIndex = primIndex;

        // Uniform sampling on unit triangle
        // A Low-Distortion Map Between Triangle and Square
        float t0 = 0.5f * u0;
        float t1 = 0.5f * u1;
        float offset = t1 - t0;
        if (offset > 0)
            t1 += offset;
        else
            t0 -= offset;
        float t2 = 1 - (t0 + t1);

        lightSample->b1 = t1;
        lightSample->b2 = t2;

        //printf("%u-%u-%u: %g\n", instIndex, geomInstIndex, primIndex, lightProb);

        const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

        const shared::Triangle &tri = geomInst.triangleBuffer[primIndex];
        const shared::Vertex (&v)[3] = {
            geomInst.vertexBuffer[tri.index0],
            geomInst.vertexBuffer[tri.index1],
            geomInst.vertexBuffer[tri.index2]
        };

        *lightPosition = t0 * v[0].position + t1 * v[1].position + t2 * v[2].position;
        *lightPosition = inst.transform * *lightPosition;
        *lightNormal = cross(v[1].position - v[0].position,
                             v[2].position - v[0].position);
        float area = length(*lightNormal);
        *lightNormal = (inst.normalMatrix * *lightNormal) / area;
        area *= 0.5f;
        *areaPDensity = lightProb / area;

        //printf("%u-%u-%u: (%g, %g, %g), PDF: %g\n", instIndex, geomInstIndex, primIndex,
        //       mat.emittance.x, mat.emittance.y, mat.emittance.z, *areaPDensity);

        //printf("%u-%u-%u: (%g, %g, %g), (%g, %g, %g)\n", instIndex, geomInstIndex, primIndex,
        //       lightPosition->x, lightPosition->y, lightPosition->z,
        //       lightNormal->x, lightNormal->y, lightNormal->z);

        if (mat.emittance) {
            texEmittance = mat.emittance;
            emittance = make_float3(1.0f, 1.0f, 1.0f);
            texCoord = t0 * v[0].texCoord + t1 * v[1].texCoord + t2 * v[2].texCoord;
        }
    }

    if (texEmittance) {
        float4 texValue = tex2DLod<float4>(texEmittance, texCoord.x, texCoord.y, 0.0f);
        emittance *= make_float3(texValue);
    }

    return emittance;
}

CUDA_DEVICE_FUNCTION float3 evaluateLight(const LightSample &lightSample, float3* lightPosition, float3* lightNormal) {
    if (lightSample.atInfinity()) {
        float phi = lightSample.b1;
        float theta = lightSample.b2;
        float u = phi / (2 * Pi);
        float v = theta / Pi;

        float posPhi = phi - plp.f->envLightRotation;
        posPhi = posPhi - floorf(posPhi / (2 * Pi)) * 2 * Pi;

        float3 direction = fromPolarYUp(posPhi, theta);
        float3 position = make_float3(direction.x, direction.y, direction.z);
        *lightPosition = position;

        *lightNormal = -position;

        float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
        float3 emittance = make_float3(texValue);
        emittance *= Pi * plp.f->envLightPowerCoeff;

        return emittance;
    }
    else {
        const InstanceData &inst = plp.f->instanceDataBuffer[lightSample.instIndex];
        const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[lightSample.geomInstIndex];
        const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

        const shared::Triangle &tri = geomInst.triangleBuffer[lightSample.primIndex];
        const shared::Vertex (&v)[3] = {
            geomInst.vertexBuffer[tri.index0],
            geomInst.vertexBuffer[tri.index1],
            geomInst.vertexBuffer[tri.index2]
        };

        float t1 = lightSample.b1;
        float t2 = lightSample.b2;
        float t0 = 1.0f - (t1 + t2);

        *lightPosition = t0 * v[0].position + t1 * v[1].position + t2 * v[2].position;
        *lightPosition = inst.transform * *lightPosition;
        *lightNormal = cross(v[1].position - v[0].position,
                             v[2].position - v[0].position);
        float area = length(*lightNormal);
        *lightNormal = (inst.normalMatrix * *lightNormal) / area;

        float3 emittance = make_float3(0.0f, 0.0f, 0.0f);
        if (mat.emittance) {
            float2 texCoord = t0 * v[0].texCoord + t1 * v[1].texCoord + t2 * v[2].texCoord;
            float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
            emittance = make_float3(texValue);
        }

        return emittance;
    }
}

CUDA_DEVICE_FUNCTION float3 sampleUnshadowedContribution(
    const float3 &shadingPoint, const float3 &vOutLocal, const ReferenceFrame &shadingFrame, const BSDF &bsdf,
    float uLight, bool sampleEnvLight, float uPos0, float uPos1, LightSample* lightSample, float* probDensity) {
    float3 lp;
    float3 lpn;
    float3 M = sampleLight(uLight, sampleEnvLight, uPos0, uPos1,
                           lightSample, &lp, &lpn, probDensity);
    bool atInfinity = lightSample->atInfinity();

    float3 offsetOrigin = shadingPoint + shadingFrame.normal * RayEpsilon;
    float3 shadowRayDir = atInfinity ? lp : (lp - offsetOrigin);
    float dist2 = sqLength(shadowRayDir);
    float dist = std::sqrt(dist2);
    shadowRayDir /= dist;
    float3 shadowRayDirLocal = shadingFrame.toLocal(shadowRayDir);

    float lpCos = dot(-shadowRayDir, lpn);
    float spCos = shadowRayDirLocal.z;

    if (lpCos > 0) {
        float3 Le = M / Pi;
        float3 fsValue = bsdf.evaluate(vOutLocal, shadowRayDirLocal);
        float G = lpCos * std::fabs(spCos) / dist2;
        float3 ret = fsValue * Le * G;
        return ret;
    }
    else {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
}

template <bool withVisibility>
CUDA_DEVICE_FUNCTION float3 performDirectLighting(
    const float3 &shadingPoint, const float3 &vOutLocal, const ReferenceFrame &shadingFrame, const BSDF &bsdf,
    const LightSample &lightSample) {
    float3 lp;
    float3 lpn;
    float3 M = evaluateLight(lightSample, &lp, &lpn);
    bool atInfinity = lightSample.atInfinity();

    float3 offsetOrigin = shadingPoint + shadingFrame.normal * RayEpsilon;
    float3 shadowRayDir = atInfinity ? lp : (lp - offsetOrigin);
    float dist2 = sqLength(shadowRayDir);
    float dist = std::sqrt(dist2);
    shadowRayDir /= dist;
    if (atInfinity)
        dist = 1e+10f;
    float3 shadowRayDirLocal = shadingFrame.toLocal(shadowRayDir);

    float lpCos = dot(-shadowRayDir, lpn);
    float spCos = shadowRayDirLocal.z;

    float visibility = 1.0f;
    if constexpr (withVisibility) {
        optixu::trace<VisibilityRayPayloadSignature>(
            plp.f->travHandle,
            offsetOrigin, shadowRayDir, 0.0f, dist * 0.9999f, 0.0f,
            0xFF, OPTIX_RAY_FLAG_NONE,
            RayType_Visibility, NumRayTypes, RayType_Visibility,
            visibility);
    }

    if (visibility > 0 && lpCos > 0) {
        float3 Le = M / Pi;
        float3 fsValue = bsdf.evaluate(vOutLocal, shadowRayDirLocal);
        float G = lpCos * std::fabs(spCos) / dist2;
        float3 ret = fsValue * Le * G;
        return ret;
    }
    else {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
}

CUDA_DEVICE_FUNCTION bool evaluateVisibility(
    const float3 &shadingPoint, const ReferenceFrame &shadingFrame, const LightSample &lightSample) {
    float3 lp;
    float3 lpn;
    evaluateLight(lightSample, &lp, &lpn);
    bool atInfinity = lightSample.atInfinity();

    float3 offsetOrigin = shadingPoint + shadingFrame.normal * RayEpsilon;
    float3 shadowRayDir = atInfinity ? lp : (lp - offsetOrigin);
    float dist2 = sqLength(shadowRayDir);
    float dist = std::sqrt(dist2);
    shadowRayDir /= dist;
    if (atInfinity)
        dist = 1e+10f;

    float visibility = 1.0f;
    optixu::trace<VisibilityRayPayloadSignature>(
        plp.f->travHandle,
        offsetOrigin, shadowRayDir, 0.0f, dist * 0.9999f, 0.0f,
        0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Visibility, NumRayTypes, RayType_Visibility,
        visibility);

    return visibility > 0.0f;
}



CUDA_DEVICE_FUNCTION uint32_t calcCellLinearIndex(const float3 &positionInWorld) {
    float3 relPos = positionInWorld - plp.s->gridOrigin;
    uint32_t ix = min(max(static_cast<uint32_t>(relPos.x / plp.s->gridCellSize.x), 0u),
                      plp.s->gridDimension.x - 1);
    uint32_t iy = min(max(static_cast<uint32_t>(relPos.y / plp.s->gridCellSize.y), 0u),
                      plp.s->gridDimension.y - 1);
    uint32_t iz = min(max(static_cast<uint32_t>(relPos.z / plp.s->gridCellSize.z), 0u),
                      plp.s->gridDimension.z - 1);
    return iz * plp.s->gridDimension.x * plp.s->gridDimension.y
        + iy * plp.s->gridDimension.x
        + ix;
}

CUDA_DEVICE_FUNCTION float3 sampleFromCell(
    const float3 &shadingPoint, const float3 &vOutLocal, const ReferenceFrame &shadingFrame, const BSDF &bsdf,
    uint32_t frameIndex, PCG32RNG &rng,
    LightSample* lightSample, float* recProbDensityEstimate) {
    float3 randomOffset = plp.s->gridCellSize
        * make_float3(-0.5f + rng.getFloat0cTo1o(),
                      -0.5f + rng.getFloat0cTo1o(),
                      -0.5f + rng.getFloat0cTo1o());
    uint32_t cellLinearIndex = calcCellLinearIndex(shadingPoint + randomOffset);
    uint32_t resStartIndex = kNumLightSlotsPerCell * cellLinearIndex;

    // JP: 
    // EN: 
    atomicOr(&plp.s->cellTouchFlags[cellLinearIndex], 1u);

    // JP: 
    // EN: 
    constexpr uint32_t numResampling = 4;
    Reservoir<LightSample> combinedReservoir;
    combinedReservoir.initialize();
    uint32_t combinedStreamLength = 0;
    float3 selectedContribution = make_float3(0.0f);
    float selectedTargetPDensity = 0.0f;
    for (int i = 0; i < numResampling; ++i) {
        uint32_t lightSlotIdx = resStartIndex + mapPrimarySampleToDiscrete(rng.getFloat0cTo1o(), kNumLightSlotsPerCell);
        const Reservoir<LightSample> &r = plp.s->reservoirs[plp.f->bufferIndex][lightSlotIdx];
        const ReservoirInfo &rInfo = plp.s->reservoirInfos[plp.f->bufferIndex][lightSlotIdx];
        const LightSample &lightSample = r.getSample();
        uint32_t streamLength = r.getStreamLength();
        if (rInfo.recPDFEstimate == 0.0f)
            continue;
        float3 cont = performDirectLighting<false>(shadingPoint, vOutLocal, shadingFrame, bsdf, lightSample);
        float targetPDensity = convertToWeight(cont);
        float weight = targetPDensity * rInfo.recPDFEstimate * streamLength;
        if (combinedReservoir.update(lightSample, weight, rng.getFloat0cTo1o())) {
            selectedContribution = cont;
            selectedTargetPDensity = targetPDensity;
        }
        combinedStreamLength += streamLength;
    }
    combinedReservoir.setStreamLength(combinedStreamLength);

    *lightSample = combinedReservoir.getSample();

    float weightForEstimate = 1.0f / combinedReservoir.getStreamLength();
    *recProbDensityEstimate = weightForEstimate * combinedReservoir.getSumWeights() / selectedTargetPDensity;
    if (!isfinite(*recProbDensityEstimate))
        *recProbDensityEstimate = 0.0f;

    return selectedContribution;
}

static constexpr bool useImplicitLightSampling = true;
static constexpr bool useExplicitLightSampling = true;
static constexpr bool useMultipleImportanceSampling = useImplicitLightSampling && useExplicitLightSampling;
static_assert(useImplicitLightSampling || useExplicitLightSampling, "Invalid configuration for light sampling.");

template <bool useReGIR>
CUDA_DEVICE_FUNCTION void pathTrace_rayGen_generic() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    uint32_t bufIdx = plp.f->bufferIndex;
    GBuffer0 gBuffer0 = plp.s->GBuffer0[bufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[bufIdx].read(launchIndex);
    GBuffer2 gBuffer2 = plp.s->GBuffer2[bufIdx].read(launchIndex);

    float3 positionInWorld = gBuffer0.positionInWorld;
    float3 normalInWorld = gBuffer1.normalInWorld;
    float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);
    uint32_t materialSlot = gBuffer2.materialSlot;

    const PerspectiveCamera &camera = plp.f->camera;

    float3 albedo = make_float3(0.0f);
    float3 contribution = make_float3(0.01f, 0.01f, 0.01f);
    if (materialSlot != 0xFFFFFFFF) {
        float3 alpha = make_float3(1.0f);

        const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

        ReferenceFrame shadingFrame(normalInWorld);
        float3 vOut = normalize(camera.position - positionInWorld);
        float3 vOutLocal = shadingFrame.toLocal(vOut);

        // JP: 光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from a light source directly seeing.
        contribution = make_float3(0.0f);
        if (vOutLocal.z > 0 && mat.emittance) {
            float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
            float3 emittance = make_float3(texValue);
            contribution += alpha * emittance / Pi;
        }

        PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

        BSDF bsdf;
        mat.setupBSDF(mat, texCoord, &bsdf);

        albedo = bsdf.evaluateDHReflectanceEstimate(vOutLocal);

        // ----------------------------------------------------------------
        // Next Event Estimation

        if constexpr (useReGIR) {
            LightSample lightSample;
            float recProbDensityEstimate;
            float3 unshadowedContribution = sampleFromCell(
                positionInWorld, vOutLocal, shadingFrame, bsdf,
                plp.f->frameIndex, rng,
                &lightSample, &recProbDensityEstimate);
            if (recProbDensityEstimate > 0.0f) {
                float visibility = evaluateVisibility(positionInWorld, shadingFrame, lightSample);
                contribution += alpha * unshadowedContribution * (visibility * recProbDensityEstimate);
            }
        }
        else {
            if constexpr (useExplicitLightSampling) {
                LightSample lightSample;
                float uLight = rng.getFloat0cTo1o();
                bool selectEnvLight = false;
                float probToSampleCurLightType = 1.0f;
                if (plp.s->envLightTexture && plp.f->enableEnvLight) {
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
                float3 lightPosition;
                float3 lightNormal;
                float areaPDensity;
                float3 M = sampleLight(uLight, selectEnvLight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
                                       &lightSample, &lightPosition, &lightNormal, &areaPDensity);
                areaPDensity *= probToSampleCurLightType;
                float misWeight = 1.0f;
                if constexpr (useMultipleImportanceSampling) {
                    float3 shadowRay = lightPosition - positionInWorld;
                    float dist2 = sqLength(shadowRay);
                    shadowRay /= std::sqrt(dist2);
                    float3 vInLocal = shadingFrame.toLocal(shadowRay);
                    float lpCos = std::fabs(dot(shadowRay, lightNormal));
                    float bsdfPDensity = bsdf.evaluatePDF(vOutLocal, vInLocal) * lpCos / dist2;
                    float lightPDensity = areaPDensity;
                    misWeight = pow2(lightPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
                }
                contribution += alpha * performDirectLighting<true>(
                    positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample) * (misWeight / areaPDensity);
            }
        }

        // ----------------------------------------------------------------

        float initImportance = sRGB_calcLuminance(alpha);

        // generate a next ray.
        float3 vInLocal;
        float dirPDensity;
        alpha *= bsdf.sampleThroughput(
            vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
            &vInLocal, &dirPDensity);
        float3 vIn = shadingFrame.fromLocal(vInLocal);

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
        float3 rayOrg = positionInWorld + RayEpsilon * normalInWorld;
        float3 rayDir = vIn;
        constexpr uint32_t maxPathLength = 5;
        while (true) {
            bool isValidSampling = rwPayload.prevDirPDensity > 0.0f && isfinite(rwPayload.prevDirPDensity);
            if (!isValidSampling)
                break;

            ++rwPayload.pathLength;
            if (rwPayload.pathLength >= maxPathLength)
                rwPayload.maxLengthTerminate = true;
            rwPayload.terminate = true;
            if constexpr (!useImplicitLightSampling || useReGIR) {
                float continueProb = std::fmin(sRGB_calcLuminance(rwPayload.alpha) / rwPayload.initImportance, 1.0f);
                if (rwPayload.rng.getFloat0cTo1o() >= continueProb || rwPayload.maxLengthTerminate)
                    break;
                rwPayload.alpha /= continueProb;
            }
            constexpr RayType pathTraceRayType = useReGIR ? RayType_PathTraceReGIR : RayType_PathTraceBaseline;
            optixu::trace<PathTraceRayPayloadSignature>(
                plp.f->travHandle, rayOrg, rayDir,
                0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
                pathTraceRayType, NumRayTypes, pathTraceRayType,
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
        if (plp.s->envLightTexture && plp.f->enableEnvLight) {
            float u = texCoord.x, v = texCoord.y;
            float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
            float3 luminance = make_float3(texValue);
            luminance *= plp.f->envLightPowerCoeff;

            contribution = luminance;

            albedo = make_float3(0.0f, 0.0f, 0.0f);
        }
    }



    // JP: デノイザーに必要な情報を出力。
    // EN: Output information required for the denoiser.
    float3 firstHitNormal = transpose(camera.orientation) * normalInWorld;
    float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
    float3 prevAlbedoResult = make_float3(0.0f, 0.0f, 0.0f);
    float3 prevNormalResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.f->numAccumFrames > 0) {
        prevColorResult = getXYZ(plp.s->beautyAccumBuffer.read(launchIndex));
        prevAlbedoResult = getXYZ(plp.s->albedoAccumBuffer.read(launchIndex));
        prevNormalResult = getXYZ(plp.s->normalAccumBuffer.read(launchIndex));
    }
    float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
    float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * contribution;
    float3 albedoResult = (1 - curWeight) * prevAlbedoResult + curWeight * albedo;
    float3 normalResult = (1 - curWeight) * prevNormalResult + curWeight * firstHitNormal;
    plp.s->beautyAccumBuffer.write(launchIndex, make_float4(colorResult, 1.0f));
    plp.s->albedoAccumBuffer.write(launchIndex, make_float4(albedoResult, 1.0f));
    plp.s->normalAccumBuffer.write(launchIndex, make_float4(normalResult, 1.0f));
}

template <bool useReGIR>
CUDA_DEVICE_FUNCTION void pathTrace_closestHit_generic() {
    auto sbtr = HitGroupSBTRecordData::get();
    const InstanceData &inst = plp.f->instanceDataBuffer[optixGetInstanceId()];
    const GeometryInstanceData &geomInst = sbtr.geomInstData;

    PathTraceWriteOnlyPayload* woPayload;
    PathTraceReadWritePayload* rwPayload;
    optixu::getPayloads<PathTraceRayPayloadSignature>(&woPayload, &rwPayload);
    PCG32RNG &rng = rwPayload->rng;

    auto hp = HitPointParameter::get();
    float3 positionInWorld;
    float3 normalInWorld;
    float2 texCoord;
    float hypAreaPDensity;
    {
        const Triangle &tri = geomInst.triangleBuffer[hp.primIndex];
        const Vertex &v0 = geomInst.vertexBuffer[tri.index0];
        const Vertex &v1 = geomInst.vertexBuffer[tri.index1];
        const Vertex &v2 = geomInst.vertexBuffer[tri.index2];
        float b1 = hp.b1;
        float b2 = hp.b2;
        float b0 = 1 - (b1 + b2);
        float3 localP = b0 * v0.position + b1 * v1.position + b2 * v2.position;
        normalInWorld = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
        if constexpr (useMultipleImportanceSampling) {
            float primProb = geomInst.emitterPrimDist.evaluatePMF(hp.primIndex);
            float3 geomNormal = cross(v1.position - v0.position, v2.position - v0.position);
            float area = 0.5f * length(geomNormal);
            hypAreaPDensity = primProb / area;
        }
        texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

        positionInWorld = optixTransformPointFromObjectToWorldSpace(localP);
        normalInWorld = normalize(optixTransformNormalFromObjectToWorldSpace(normalInWorld));
        if (!allFinite(normalInWorld))
            normalInWorld = make_float3(0, 0, 1);
    }

    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    ReferenceFrame shadingFrame(normalInWorld);
    float3 vOut = normalize(-optixGetWorldRayDirection());
    float3 vOutLocal = shadingFrame.toLocal(vOut);

    if constexpr (useImplicitLightSampling && !useReGIR) {
        // Implicit Light Sampling
        if (vOutLocal.z > 0 && mat.emittance) {
            float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
            float3 emittance = make_float3(texValue);
            float misWeight = 1.0f;
            if constexpr (useMultipleImportanceSampling) {
                float dist2 = squaredDistance(optixGetWorldRayOrigin(), positionInWorld);
                float instProb = inst.lightGeomInstDist.integral() / plp.s->lightInstDist.integral();
                float geomInstProb = geomInst.emitterPrimDist.integral() / inst.lightGeomInstDist.integral();
                float lightPDensity = instProb * geomInstProb * hypAreaPDensity * dist2 / vOutLocal.z;
                float bsdfPDensity = rwPayload->prevDirPDensity;
                misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
            }
            rwPayload->contribution += rwPayload->alpha * emittance * (misWeight / Pi);
        }

        // Russian roulette
        float continueProb = std::fmin(sRGB_calcLuminance(rwPayload->alpha) / rwPayload->initImportance, 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb || rwPayload->maxLengthTerminate)
            return;
        rwPayload->alpha /= continueProb;
    }

    BSDF bsdf;
    mat.setupBSDF(mat, texCoord, &bsdf);

    // ----------------------------------------------------------------
    // Next Event Estimation

    if constexpr (useReGIR) {
        LightSample lightSample;
        float recProbDensityEstimate;
        float3 unshadowedContribution = sampleFromCell(
            positionInWorld, vOutLocal, shadingFrame, bsdf,
            plp.f->frameIndex, rng,
            &lightSample, &recProbDensityEstimate);
        if (recProbDensityEstimate > 0.0f) {
            float visibility = evaluateVisibility(positionInWorld, shadingFrame, lightSample);
            rwPayload->contribution += rwPayload->alpha * unshadowedContribution * (visibility * recProbDensityEstimate);
        }
    }
    else {
        if constexpr (useExplicitLightSampling) {
            LightSample lightSample;
            float uLight = rng.getFloat0cTo1o();
            bool selectEnvLight = false;
            float probToSampleCurLightType = 1.0f;
            if (plp.s->envLightTexture && plp.f->enableEnvLight) {
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
            float3 lightPosition;
            float3 lightNormal;
            float areaPDensity;
            float3 M = sampleLight(uLight, selectEnvLight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
                                   &lightSample, &lightPosition, &lightNormal, &areaPDensity);
            areaPDensity *= probToSampleCurLightType;
            float misWeight = 1.0f;
            if constexpr (useMultipleImportanceSampling) {
                float3 shadowRay = lightPosition - positionInWorld;
                float dist2 = sqLength(shadowRay);
                shadowRay /= std::sqrt(dist2);
                float3 vInLocal = shadingFrame.toLocal(shadowRay);
                float lpCos = std::fabs(dot(shadowRay, lightNormal));
                float bsdfPDensity = bsdf.evaluatePDF(vOutLocal, vInLocal) * lpCos / dist2;
                float lightPDensity = areaPDensity;
                misWeight = pow2(lightPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
            }
            rwPayload->contribution += rwPayload->alpha * performDirectLighting<true>(
                positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample) * (misWeight / areaPDensity);
        }
    }

    // ----------------------------------------------------------------

    // generate a next ray.
    float3 vInLocal;
    float dirPDensity;
    rwPayload->alpha *= bsdf.sampleThroughput(
        vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &vInLocal, &dirPDensity);
    float3 vIn = shadingFrame.fromLocal(vInLocal);

    woPayload->nextOrigin = positionInWorld + RayEpsilon * normalInWorld;
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
        optixu::getPayloads<PathTraceRayPayloadSignature>(nullptr, &rwPayload);

        float3 rayDir = normalize(optixGetWorldRayDirection());
        float posPhi, theta;
        toPolarYUp(rayDir, &posPhi, &theta);

        float phi = posPhi + plp.f->envLightRotation;
        phi = phi - floorf(phi / (2 * Pi)) * 2 * Pi;
        float2 texCoord = make_float2(phi / (2 * Pi), theta / Pi);

        // Implicit Light Sampling
        float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, texCoord.x, texCoord.y, 0.0f);
        float3 emittance = (Pi * plp.f->envLightPowerCoeff) * make_float3(texValue);
        float misWeight = 1.0f;
        if constexpr (useMultipleImportanceSampling) {
            float uvPDF = plp.s->envLightImportanceMap.evaluatePDF(texCoord.x, texCoord.y);
            float hypAreaPDensity = uvPDF / (2 * Pi * Pi * std::sin(theta));
            float lightPDensity = probToSampleEnvLight * hypAreaPDensity;
            float bsdfPDensity = rwPayload->prevDirPDensity;
            misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
        }
        rwPayload->contribution += rwPayload->alpha * emittance * (misWeight / Pi);
    }
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTraceRegir)() {
    pathTrace_rayGen_generic<true>();
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTraceRegir)() {
    pathTrace_closestHit_generic<true>();
}
