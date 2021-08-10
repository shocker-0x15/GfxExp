#include "restir_shared.h"

using namespace shared;

RT_PIPELINE_LAUNCH_PARAMETERS PipelineLaunchParameters plp;

#define pixelprintf(idx, px, py, fmt, ...) do { if (idx.x == px && idx.y == py) printf(fmt, ##__VA_ARGS__); } while (0)



CUDA_DEVICE_FUNCTION float pow2(float x) {
    return x * x;
}
CUDA_DEVICE_FUNCTION float pow3(float x) {
    return x * x * x;
}
CUDA_DEVICE_FUNCTION float pow4(float x) {
    return x * x * x * x;
}
CUDA_DEVICE_FUNCTION float pow5(float x) {
    return x * x * x * x * x;
}

template <typename T>
CUDA_DEVICE_FUNCTION T lerp(const T &v0, const T &v1, float t) {
    return (1 - t) * v0 + t * v1;
}



// ( 0, 0,  1) <=> phi:      0
// (-1, 0,  0) <=> phi: 1/2 pi
// ( 0, 0, -1) <=> phi:   1 pi
// ( 1, 0,  0) <=> phi: 3/2 pi
CUDA_DEVICE_FUNCTION float3 fromPolarYUp(float phi, float theta) {
    float sinPhi, cosPhi;
    float sinTheta, cosTheta;
    sincosf(phi, &sinPhi, &cosPhi);
    sincosf(theta, &sinTheta, &cosTheta);
    return make_float3(-sinPhi * sinTheta, cosTheta, cosPhi * sinTheta);
}
CUDA_DEVICE_FUNCTION void toPolarYUp(const float3 &v, float* phi, float* theta) {
    *theta = std::acos(min(max(v.y, -1.0f), 1.0f));
    *phi = std::fmod(std::atan2(-v.x, v.z) + 2 * Pi,
                     2 * Pi);
}

CUDA_DEVICE_FUNCTION float3 halfVector(const float3 &a, const float3 &b) {
    return normalize(a + b);
}

CUDA_DEVICE_FUNCTION float absDot(const float3 &a, const float3 &b) {
    return std::fabs(dot(a, b));
}

CUDA_DEVICE_FUNCTION void makeCoordinateSystem(const float3 &normal, float3* tangent, float3* bitangent) {
    float sign = normal.z >= 0 ? 1 : -1;
    const float a = -1 / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    *tangent = make_float3(1 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    *bitangent = make_float3(b, sign + normal.y * normal.y * a, -normal.y);
}

struct ReferenceFrame {
    float3 tangent;
    float3 bitangent;
    float3 normal;

    CUDA_DEVICE_FUNCTION ReferenceFrame(const float3 &_tangent, const float3 &_bitangent, const float3 &_normal) :
        tangent(_tangent), bitangent(_bitangent), normal(_normal) {}
    CUDA_DEVICE_FUNCTION ReferenceFrame(const float3 &_normal) : normal(_normal) {
        makeCoordinateSystem(normal, &tangent, &bitangent);
    }

    CUDA_DEVICE_FUNCTION float3 toLocal(const float3 &v) const {
        return make_float3(dot(tangent, v), dot(bitangent, v), dot(normal, v));
    }
    CUDA_DEVICE_FUNCTION float3 fromLocal(const float3 &v) const {
        return make_float3(dot(make_float3(tangent.x, bitangent.x, normal.x), v),
                           dot(make_float3(tangent.y, bitangent.y, normal.y), v),
                           dot(make_float3(tangent.z, bitangent.z, normal.z), v));
    }
};



struct BSDF {
    static constexpr uint32_t NumDwords = 16;
    GetBaseColor m_getBaseColor;
    EvaluateBSDF m_evaluate;
    uint32_t m_data[NumDwords];

    CUDA_DEVICE_FUNCTION float3 getBaseColor(const float3 &vout) const {
        return m_getBaseColor(m_data, vout);
    }
    CUDA_DEVICE_FUNCTION float3 evaluate(const float3 &vin, const float3 &vout) const {
        return m_evaluate(m_data, vin, vout);
    }
};

template <typename BSDFType>
CUDA_DEVICE_FUNCTION void setupBSDF(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf);

template <typename BSDFType>
CUDA_DEVICE_FUNCTION float3 bsdf_getBaseColor(const uint32_t* data, const float3 &vout) {
    auto &bsdf = *reinterpret_cast<const BSDFType*>(data);
    return bsdf.getBaseColor(vout);
}

template <typename BSDFType>
CUDA_DEVICE_FUNCTION float3 bsdf_evaluate(const uint32_t* data, const float3 &vin, const float3 &vout) {
    auto &bsdf = *reinterpret_cast<const BSDFType*>(data);
    return bsdf.evaluate(vin, vout);
}



class LambertBRDF {
    float3 m_reflectance;

public:
    CUDA_DEVICE_FUNCTION LambertBRDF(const float3 &reflectance) :
        m_reflectance(reflectance) {}

    CUDA_DEVICE_FUNCTION float3 getBaseColor(const float3 &vout) const {
        return m_reflectance;
    }

    CUDA_DEVICE_FUNCTION float3 evaluate(const float3 &vin, const float3 &vout) const {
        if (vin.z * vout.z > 0)
            return m_reflectance;
        else
            return make_float3(0.0f, 0.0f, 0.0f);
    }
};

template<>
CUDA_DEVICE_FUNCTION void setupBSDF<LambertBRDF>(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf) {
    float4 reflectance = tex2DLod<float4>(matData.asLambert.reflectance, texCoord.x, texCoord.y, 0.0f);
    auto &bsdfBody = *reinterpret_cast<LambertBRDF*>(bsdf->m_data);
    bsdfBody = LambertBRDF(make_float3(reflectance.x, reflectance.y, reflectance.z));
}



// JP: ホストコード側でまだ対応させていない。
// EN: Not supported yet on the host code.
class DiffuseAndSpecularBRDF {
    struct GGXMicrofacetDistribution {
        float alpha_g;

        CUDA_DEVICE_FUNCTION float evaluate(const float3 &m) const {
            if (m.z <= 0.0f)
                return 0.0f;
            float temp = pow2(m.x) + pow2(m.y) + pow2(m.z * alpha_g);
            return pow2(alpha_g) / (Pi * pow2(temp));
        }
        CUDA_DEVICE_FUNCTION float evaluateSmithG1(const float3 &v, const float3 &m) const {
            if (dot(v, m) * v.z <= 0)
                return 0.0f;
            float temp = pow2(alpha_g) * (pow2(v.x) + pow2(v.y)) / pow2(v.z);
            return 2 / (1 + std::sqrt(1 + temp));
        }
        CUDA_DEVICE_FUNCTION float sample(const float3 &v, float u0, float u1,
                                          float3* m, float* mPDensity) const {
            // stretch view
            float3 sv = normalize(make_float3(alpha_g * v.x, alpha_g * v.y, v.z));

            // orthonormal basis
            float distIn2D = std::sqrt(sv.x * sv.x + sv.y * sv.y);
            float recDistIn2D = 1.0f / distIn2D;
            float3 T1 = (sv.z < 0.9999f) ?
                make_float3(sv.y * recDistIn2D, -sv.x * recDistIn2D, 0) :
                make_float3(1.0f, 0.0f, 0.0f);
            float3 T2 = make_float3(T1.y * sv.z, -T1.x * sv.z, distIn2D);

            // sample point with polar coordinates (r, phi)
            float a = 1.0f / (1.0f + sv.z);
            float r = std::sqrt(u0);
            float phi = Pi * ((u1 < a) ? u1 / a : 1 + (u1 - a) / (1.0f - a));
            float sinPhi, cosPhi;
            sincosf(phi, &sinPhi, &cosPhi);
            float P1 = r * cosPhi;
            float P2 = r * sinPhi * ((u1 < a) ? 1.0f : sv.z);

            // compute normal
            *m = P1 * T1 + P2 * T2 + std::sqrt(1.0f - P1 * P1 - P2 * P2) * sv;

            // unstretch
            *m = normalize(make_float3(alpha_g * m->x, alpha_g * m->y, m->z));

            float D = evaluate(*m);
            *mPDensity = evaluateSmithG1(v, *m) * absDot(v, *m) * D / std::fabs(v.z);

            return D;
        }
        CUDA_DEVICE_FUNCTION float evaluatePDF(const float3 &v, const float3 &m) {
            return evaluateSmithG1(v, m) * absDot(v, m) * evaluate(m) / std::fabs(v.z);
        }
    };

    float3 m_diffuseColor;
    float3 m_specularF0Color;
    float m_roughness;

public:
    CUDA_DEVICE_FUNCTION DiffuseAndSpecularBRDF(const float3 &diffuseColor, const float3 &specularF0Color, float smoothness) {
        m_diffuseColor = diffuseColor;
        m_specularF0Color = specularF0Color;
        m_roughness = 1 - smoothness;
    }

    CUDA_DEVICE_FUNCTION DiffuseAndSpecularBRDF(const float3 &diffuseColor, float reflectance, float smoothness, float metallic) {
        m_diffuseColor = diffuseColor * (1 - metallic);
        m_specularF0Color = make_float3(0.16f * pow2(reflectance) * (1 - metallic)) + diffuseColor * metallic;
        m_roughness = 1 - smoothness;
    }

    CUDA_DEVICE_FUNCTION float3 getBaseColor(const float3 &vout) const {
        bool entering = vout.z >= 0.0f;
        float3 dirV = entering ? vout : -vout;

        float expectedCosTheta_d = dirV.z;
        float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * pow2(expectedCosTheta_d);
        float oneMinusDotVN5 = pow5(1 - dirV.z);
        float expectedDiffFGiven = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
        float expectedDiffFSampled = 1.0f; // ad-hoc
        float3 diffuseDHR = m_diffuseColor * expectedDiffFGiven * expectedDiffFSampled * lerp(1.0f, 1.0f / 1.51f, m_roughness);

        //float expectedOneMinusDotVH5 = oneMinusDotVN5;
        // (1 - roughness) is an ad-hoc adjustment.
        float expectedOneMinusDotVH5 = pow5(1 - dirV.z) * (1 - m_roughness);

        float3 specularDHR = lerp(m_specularF0Color, make_float3(1.0f), expectedOneMinusDotVH5);

        return min(diffuseDHR + specularDHR, make_float3(1.0f));
    }

    CUDA_DEVICE_FUNCTION float3 evaluate(const float3 &vGiven, const float3 &vSampled) const {
        GGXMicrofacetDistribution ggx;
        ggx.alpha_g = m_roughness * m_roughness;

        if (vSampled.z * vGiven.z <= 0)
            return make_float3(0.0f, 0.0f, 0.0f);

        bool entering = vGiven.z >= 0.0f;
        float3 dirV = entering ? vGiven : -vGiven;
        float3 dirL = entering ? vSampled : -vSampled;

        float3 m = halfVector(dirL, dirV);
        float dotLH = dot(dirL, m);

        float oneMinusDotLH5 = pow5(1 - dotLH);

        float D = ggx.evaluate(m);
#if defined(USE_HEIGHT_CORRELATED_SMITH)
        float G = ggx.evaluateHeightCorrelatedSmithG(dirL, dirV, m);
#else
        float G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
#endif
        constexpr float F90 = 1.0f;
        float3 F = lerp(m_specularF0Color, make_float3(F90), oneMinusDotLH5);

        float microfacetDenom = 4 * dirL.z * dirV.z;
        float3 specularValue = F * ((D * G) / microfacetDenom);
        if (G == 0)
            specularValue = make_float3(0.0f);

        float F_D90 = 0.5f * m_roughness + 2 * m_roughness * dotLH * dotLH;
        float oneMinusDotVN5 = pow5(1 - dirV.z);
        float oneMinusDotLN5 = pow5(1 - dirL.z);
        float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
        float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);

        float3 diffuseValue = m_diffuseColor * (diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, m_roughness) / Pi);

        float3 ret = diffuseValue + specularValue;

        return ret;
    }
};

template<>
CUDA_DEVICE_FUNCTION void setupBSDF<DiffuseAndSpecularBRDF>(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf) {
    float4 diffuseColor = tex2DLod<float4>(matData.asDiffuseAndSpecular.diffuse, texCoord.x, texCoord.y, 0.0f);
    float4 specularF0Color = tex2DLod<float4>(matData.asDiffuseAndSpecular.specular, texCoord.x, texCoord.y, 0.0f);
    float smoothness = tex2DLod<float>(matData.asDiffuseAndSpecular.smoothness, texCoord.x, texCoord.y, 0.0f);
    auto &bsdfBody = *reinterpret_cast<DiffuseAndSpecularBRDF*>(bsdf->m_data);
    bsdfBody = DiffuseAndSpecularBRDF(make_float3(diffuseColor),
                                      make_float3(specularF0Color),
                                      smoothness);
}



#define DEFINE_BSDF_CALLABLES(BSDFType) \
    RT_CALLABLE_PROGRAM void RT_DC_NAME(setup ## BSDFType)(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf) {\
        bsdf->m_getBaseColor = matData.getBaseColor;\
        bsdf->m_evaluate = matData.evaluateBSDF;\
        return setupBSDF<BSDFType>(matData, texCoord, bsdf);\
    }\
    RT_CALLABLE_PROGRAM float3 RT_DC_NAME(BSDFType ## _getBaseColor)(const uint32_t* data, const float3 &vout) {\
        return bsdf_getBaseColor<BSDFType>(data, vout);\
    }\
    RT_CALLABLE_PROGRAM float3 RT_DC_NAME(BSDFType ## _evaluate)(const uint32_t* data, const float3 &vin, const float3 &vout) {\
        return bsdf_evaluate<BSDFType>(data, vin, vout);\
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
        pickInfo->albedo = bsdf.getBaseColor(vOutLocal);
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
        if (plp.s->envLightTexture) {
            float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
            emittance = make_float3(texValue);
            emittance *= Pi * plp.f->envLightPowerCoeff;
        }
        pickInfo->emittance = emittance;
        pickInfo->normalInWorld = vOut;
    }
}



CUDA_DEVICE_FUNCTION float3 sampleLight(
    float ul, float u0, float u1,
    LightSample* lightSample, float3* lightPosition, float3* lightNormal, float* areaPDensity) {
    float lightProb = 1.0f;

    // JP: 環境光テクスチャーが設定されている場合は一定の確率でサンプルする。
    // EN: Sample an environmental texture with a fixed probability if it is set.
    bool sampleEnvLight = false;
    if (plp.s->envLightTexture) {
        if (ul < 0.25f) {
            lightProb = 0.25f;
            ul = ul / 0.25f;
            sampleEnvLight = true;
        }
        else {
            lightProb = 0.75f;
            ul = (ul - 0.25f) / 0.75f;
        }
    }

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

        float sinPhi, cosPhi;
        sincosf(posPhi, &sinPhi, &cosPhi);
        float3 texCoord0Dir = normalize(make_float3(-cosPhi, 0.0f, -sinPhi));

        *lightNormal = -position;

        // JP: テクスチャー空間中のPDFを面積に関するものに変換する。
        // EN: convert the PDF in texture space to one with respect to area.
        // The true value is: lim_{l to inf} uvPDF / (2 * Pi * Pi * std::sin(theta)) / l^2
        *areaPDensity = uvPDF / (2 * Pi * Pi * std::sin(theta));
        *areaPDensity *= lightProb;

        float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
        float3 emittance = make_float3(texValue);
        emittance *= Pi * plp.f->envLightPowerCoeff;

        return emittance;
    }
    else {
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
        uint32_t geomInstIndex = inst.geomInstSlots[inst.lightGeomInstDist.sample(uGeomInst, &geomInstProb, &uPrim)];
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
        const shared::Vertex(&v)[3] = {
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

        float3 emittance = make_float3(0.0f, 0.0f, 0.0f);
        if (mat.emittance) {
            float2 texCoord = t0 * v[0].texCoord + t1 * v[1].texCoord + t2 * v[2].texCoord;
            float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
            emittance = make_float3(texValue);
        }

        return emittance;
    }
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

        float sinPhi, cosPhi;
        sincosf(posPhi, &sinPhi, &cosPhi);
        float3 texCoord0Dir = normalize(make_float3(-cosPhi, 0.0f, -sinPhi));

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
    float uLight, float uPos0, float uPos1, LightSample* lightSample, float* probDensity) {
    float3 lp;
    float3 lpn;
    float3 M = sampleLight(uLight, uPos0, uPos1,
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



CUDA_DEVICE_KERNEL void RT_RG_NAME(generateInitialCandidates)() {
    int2 launchIndex = make_int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

    uint32_t curBufIdx = plp.f->bufferIndex;
    GBuffer0 gBuffer0 = plp.s->GBuffer0[curBufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[curBufIdx].read(launchIndex);
    GBuffer2 gBuffer2 = plp.s->GBuffer2[curBufIdx].read(launchIndex);

    float3 positionInWorld = gBuffer0.positionInWorld;
    float3 normalInWorld = gBuffer1.normalInWorld;
    float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);
    uint32_t materialSlot = gBuffer2.materialSlot;

    if (materialSlot != 0xFFFFFFFF) {
        PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

        const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

        BSDF bsdf;
        mat.setupBSDF(mat, texCoord, &bsdf);
        ReferenceFrame shadingFrame(normalInWorld);
        float3 vOut = plp.f->camera.position - positionInWorld;
        float dist = length(vOut);
        vOut /= dist;
        float3 vOutLocal = shadingFrame.toLocal(vOut);

        uint32_t curResIndex = plp.currentReservoirIndex;
        Reservoir<LightSample> reservoir = plp.s->reservoirBuffer[curResIndex][launchIndex];
        reservoir.initialize();

        // JP: Unshadowed ContributionをターゲットPDFとしてStreaming RISを実行。
        // EN: Perform streaming RIS with unshadowed contribution as the target PDF.
        float selectedTargetDensity = 0.0f;
        uint32_t numCandidates = 1 << plp.f->log2NumCandidateSamples;
        for (int i = 0; i < numCandidates; ++i) {
            // JP: 候補サンプルを生成して、ターゲットPDFを計算する。
            //     ターゲットPDFは正規化されていなくても良い。
            // EN: Generate a candidate sample then calculate the target PDF for it.
            //     Target PDF doesn't require to be normalized.
            LightSample lightSample;
            float probDensity;
            float3 cont = sampleUnshadowedContribution(
                positionInWorld, vOutLocal, shadingFrame, bsdf,
                rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
                &lightSample, &probDensity);
            float targetDensity = convertToWeight(cont);

            // JP: 候補サンプル生成用のPDFとターゲットPDFは異なるためサンプルにはウェイトがかかる。
            // EN: The sample has a weight since the PDF to generate the candidate sample and the target PDF are
            //     different.
            float weight = targetDensity / probDensity;
            if (reservoir.update(lightSample, weight, rng.getFloat0cTo1o()))
                selectedTargetDensity = targetDensity;
        }

        // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
        // EN: Calculate the estimate of the reciprocal of the probability density that the current sample suvives.
        ReservoirInfo reservoirInfo;
        reservoirInfo.recPDFEstimate = reservoir.getSumWeights() / (selectedTargetDensity * reservoir.getStreamLength());
        reservoirInfo.targetDensity = selectedTargetDensity;
        if (!isfinite(reservoirInfo.recPDFEstimate)) {
            reservoirInfo.recPDFEstimate = 0.0f;
            reservoirInfo.targetDensity = 0.0f;
        }

        // JP: サンプルが遮蔽されていて寄与を持たない場合に、隣接ピクセルにサンプルが伝播しないよう、
        //     Reservoirのウェイトをゼロにする。
        // EN: Set the reservoir's weight to zero so that the occluded sample which has no contribution
        //     will not propagate to neighboring pixels.
        if (plp.f->reuseVisibility) {
            if (!evaluateVisibility(positionInWorld, shadingFrame, reservoir.getSample())) {
                reservoirInfo.recPDFEstimate = 0.0f;
                reservoirInfo.targetDensity = 0.0f;
            }
        }

        plp.s->rngBuffer.write(launchIndex, rng);
        plp.s->reservoirBuffer[curResIndex][launchIndex] = reservoir;
        plp.s->reservoirInfoBuffer[curResIndex].write(launchIndex, reservoirInfo);
    }
}



CUDA_DEVICE_FUNCTION bool testNeighbor(
    uint32_t nbBufIdx, int2 nbCoord, float dist, const float3 &normalInWorld) {
    if (nbCoord.x < 0 || nbCoord.x >= plp.s->imageSize.x ||
        nbCoord.y < 0 || nbCoord.y >= plp.s->imageSize.y)
        return false;

    GBuffer2 nbGBuffer2 = plp.s->GBuffer2[nbBufIdx].read(nbCoord);
    if (nbGBuffer2.materialSlot == 0xFFFFFFFF)
        return false;

    GBuffer0 nbGBuffer0 = plp.s->GBuffer0[nbBufIdx].read(nbCoord);
    GBuffer1 nbGBuffer1 = plp.s->GBuffer1[nbBufIdx].read(nbCoord);
    float3 nbPositionInWorld = nbGBuffer0.positionInWorld;
    float3 nbNormalInWorld = nbGBuffer1.normalInWorld;
    float nbDist = length(plp.f->camera.position - nbPositionInWorld);
    if (abs(nbDist - dist) / dist > 0.1f || dot(normalInWorld, nbNormalInWorld) < 0.9f)
        return false;

    return true;
}



static constexpr bool useMIS_RIS = true;

template <bool useUnbiasedEstimator>
CUDA_DEVICE_FUNCTION void combineTemporalNeighbors() {
    int2 launchIndex = make_int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

    uint32_t curBufIdx = plp.f->bufferIndex;
    uint32_t prevBufIdx = (curBufIdx + 1) % 2;
    GBuffer0 gBuffer0 = plp.s->GBuffer0[curBufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[curBufIdx].read(launchIndex);
    GBuffer2 gBuffer2 = plp.s->GBuffer2[curBufIdx].read(launchIndex);

    float3 positionInWorld = gBuffer0.positionInWorld;
    float3 normalInWorld = gBuffer1.normalInWorld;
    float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);
    float2 motionVector = gBuffer2.motionVector;
    uint32_t materialSlot = gBuffer2.materialSlot;

    if (materialSlot != 0xFFFFFFFF) {
        const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

        BSDF bsdf;
        mat.setupBSDF(mat, texCoord, &bsdf);
        ReferenceFrame shadingFrame(normalInWorld);
        float3 vOut = plp.f->camera.position - positionInWorld;
        float dist = length(vOut);
        vOut /= dist;
        float3 vOutLocal = shadingFrame.toLocal(vOut);

        uint32_t curResIndex = plp.currentReservoirIndex;
        uint32_t prevResIndex = (curResIndex + 1) % 2;

        float selectedTargetDensity = 0.0f;
        bool neighborIsSelected = false;
        Reservoir<LightSample> combinedReservoir;
        combinedReservoir.initialize();

        // JP: まず現在のピクセルのReservoirを結合する。
        // EN: First combine the reservoir for the current pixel.
        const Reservoir<LightSample> /*&*/self = plp.s->reservoirBuffer[curResIndex][launchIndex];
        const ReservoirInfo selfResInfo = plp.s->reservoirInfoBuffer[curResIndex].read(launchIndex);
        if (selfResInfo.recPDFEstimate > 0.0f) {
            combinedReservoir = self;
            selectedTargetDensity = selfResInfo.targetDensity;
        }
        uint32_t combinedStreamLength = self.getStreamLength();
        uint32_t maxNumPrevSamples = 20 * self.getStreamLength();

        int2 nbCoord = make_int2(launchIndex.x + 0.5f - motionVector.x,
                                 launchIndex.y + 0.5f - motionVector.y);
        bool acceptedNeighbor;
        if constexpr (useUnbiasedEstimator) {
            acceptedNeighbor =
                nbCoord.x >= 0 && nbCoord.x < plp.s->imageSize.x &&
                nbCoord.y >= 0 && nbCoord.y < plp.s->imageSize.y;
        }
        else {
            // JP: 隣接ピクセルのジオメトリ・マテリアルがあまりに異なる場合に候補サンプルを再利用すると
            //     バイアスが増えてしまうため、そのようなピクセルは棄却する。
            // EN: Reusing candidates from neighboring pixels with substantially different geometry/material
            //     leads to increased bias. Reject such a pixel.
            acceptedNeighbor = testNeighbor(prevBufIdx, nbCoord, dist, normalInWorld);
        }
        if (acceptedNeighbor) {
            const Reservoir<LightSample> /*&*/neighbor = plp.s->reservoirBuffer[prevResIndex][nbCoord];
            const ReservoirInfo neighborInfo = plp.s->reservoirInfoBuffer[prevResIndex].read(nbCoord);

            // JP: 隣接ピクセルが持つ候補サンプルの「現在の」ピクセルにおける確率密度を計算する。
            // EN: Calculate the probability density at the "current" pixel of the candidate sample
            //     the neighboring pixel holds.
            // TODO: アニメーションやジッタリングがある場合には前フレームの対応ピクセルのターゲットPDFは
            //       変わってしまっているはず。この場合にはUnbiasedにするにはもうちょっと工夫がいる？
            LightSample nbLightSample = neighbor.getSample();
            float3 cont = performDirectLighting<false>(
                positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
            float targetDensity = convertToWeight(cont);

            uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxNumPrevSamples);
            float weight = targetDensity * neighborInfo.recPDFEstimate * nbStreamLength;
            if (combinedReservoir.update(nbLightSample, weight, rng.getFloat0cTo1o())) {
                selectedTargetDensity = targetDensity;
                if constexpr (useUnbiasedEstimator)
                    neighborIsSelected = true;
                else
                    (void)neighborIsSelected;
            }

            combinedStreamLength += nbStreamLength;
        }
        combinedReservoir.setStreamLength(combinedStreamLength);

        float weightForEstimate;
        if constexpr (useUnbiasedEstimator) {
            // JP: 推定関数をunbiasedとするための、生き残ったサンプルのウェイトを計算する。
            //     ここではReservoirの結合時とは逆に、サンプルは生き残った1つだが、
            //     ターゲットPDFは隣接ピクセルのものを評価する。
            // EN: Compute a weight for the survived sample to make the estimator unbiased.
            //     In contrast to the case where we combine reservoirs, the sample is only one survived and
            //     Evaluate target PDFs at the neighboring pixels here.
            LightSample selectedLightSample = combinedReservoir.getSample();

            float numWeight;
            float denomWeight;

            // JP: まずは現在のピクセルのターゲットPDFに対応する量を計算。
            // EN: First, calculate a quantity corresponding to the current pixel's target PDF.
            {
                float3 cont = performDirectLighting<false>(
                    positionInWorld, vOutLocal, shadingFrame, bsdf, selectedLightSample);
                float targetDensityForSelf = convertToWeight(cont);
                if constexpr (useMIS_RIS) {
                    numWeight = targetDensityForSelf;
                    denomWeight = targetDensityForSelf * self.getStreamLength();
                }
                else {
                    numWeight = 1.0f;
                    denomWeight = 0.0f;
                    if (targetDensityForSelf > 0.0f)
                        denomWeight = self.getStreamLength();
                }
            }

            // JP: 続いて隣接ピクセルのターゲットPDFに対応する量を計算。
            // EN: Next, calculate a quantity corresponding to the neighboring pixel's target PDF.
            if (acceptedNeighbor) {
                GBuffer2 nbGBuffer2 = plp.s->GBuffer2[prevBufIdx].read(nbCoord);
                uint32_t nbMaterialSlot = nbGBuffer2.materialSlot;
                if (nbMaterialSlot != 0xFFFFFFFF) {
                    GBuffer0 nbGBuffer0 = plp.s->GBuffer0[prevBufIdx].read(nbCoord);
                    GBuffer1 nbGBuffer1 = plp.s->GBuffer1[prevBufIdx].read(nbCoord);
                    float3 nbPositionInWorld = nbGBuffer0.positionInWorld;
                    float3 nbNormalInWorld = nbGBuffer1.normalInWorld;
                    float2 nbTexCoord = make_float2(nbGBuffer0.texCoord_x, nbGBuffer1.texCoord_y);

                    const MaterialData &nbMat = plp.s->materialDataBuffer[nbMaterialSlot];

                    BSDF nbBsdf;
                    nbMat.setupBSDF(nbMat, nbTexCoord, &nbBsdf);
                    ReferenceFrame nbShadingFrame(nbNormalInWorld);
                    float3 nbVOut = plp.f->camera.position - nbPositionInWorld;
                    float nbDist = length(nbVOut);
                    nbVOut /= nbDist;
                    float3 nbVOutLocal = nbShadingFrame.toLocal(nbVOut);

                    const Reservoir<LightSample> /*&*/neighbor = plp.s->reservoirBuffer[prevResIndex][nbCoord];

                    // JP: 際限なく過去フレームのウェイトが高まってしまうのを防ぐため、
                    //     Temporal Reuseでは前フレームのストリーム長を現在のピクセルの20倍に制限する。
                    // EN: To prevent the weight for previous frames to grow unlimitedly,
                    //     limit the previous frame's weight by 20x of the current pixel's one.
                    float3 cont = performDirectLighting<false>(
                        nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, selectedLightSample);
                    float nbTargetDensity = convertToWeight(cont);
                    uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxNumPrevSamples);
                    if constexpr (useMIS_RIS) {
                        denomWeight += nbTargetDensity * nbStreamLength;
                        if (neighborIsSelected)
                            numWeight = nbTargetDensity;
                    }
                    else {
                        if (nbTargetDensity > 0.0f)
                            denomWeight += nbStreamLength;
                    }
                }
            }

            weightForEstimate = numWeight / denomWeight;
        }
        else {
            weightForEstimate = 1.0f / combinedReservoir.getStreamLength();
        }
        // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
        // EN: Calculate the estimate of the reciprocal of the probability density that the current sample suvives.
        ReservoirInfo reservoirInfo;
        reservoirInfo.recPDFEstimate = weightForEstimate * combinedReservoir.getSumWeights() / selectedTargetDensity;
        reservoirInfo.targetDensity = selectedTargetDensity;
        if (!isfinite(reservoirInfo.recPDFEstimate)) {
            reservoirInfo.recPDFEstimate = 0.0f;
            reservoirInfo.targetDensity = 0.0f;
        }

        plp.s->rngBuffer.write(launchIndex, rng);
        plp.s->reservoirBuffer[curResIndex][launchIndex] = combinedReservoir;
        plp.s->reservoirInfoBuffer[curResIndex].write(launchIndex, reservoirInfo);
    }
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(combineTemporalNeighborsBiased)() {
    combineTemporalNeighbors<false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(combineTemporalNeighborsUnbiased)() {
    combineTemporalNeighbors<true>();
}



template <bool useUnbiasedEstimator>
CUDA_DEVICE_FUNCTION void combineSpatialNeighbors() {
    int2 launchIndex = make_int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

    uint32_t bufIdx = plp.f->bufferIndex;
    GBuffer0 gBuffer0 = plp.s->GBuffer0[bufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[bufIdx].read(launchIndex);
    GBuffer2 gBuffer2 = plp.s->GBuffer2[bufIdx].read(launchIndex);

    float3 positionInWorld = gBuffer0.positionInWorld;
    float3 normalInWorld = gBuffer1.normalInWorld;
    float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);
    uint32_t materialSlot = gBuffer2.materialSlot;

    if (materialSlot != 0xFFFFFFFF) {
        const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

        BSDF bsdf;
        mat.setupBSDF(mat, texCoord, &bsdf);
        ReferenceFrame shadingFrame(normalInWorld);
        float3 vOut = plp.f->camera.position - positionInWorld;
        float dist = length(vOut);
        vOut /= dist;
        float3 vOutLocal = shadingFrame.toLocal(vOut);

        uint32_t srcResIndex = plp.currentReservoirIndex;
        uint32_t dstResIndex = (srcResIndex + 1) % 2;

        float selectedTargetDensity = 0.0f;
        int32_t selectedNeighborIndex = -1;
        Reservoir<LightSample> combinedReservoir;
        combinedReservoir.initialize();

        // JP: まず現在のピクセルのReservoirを結合する。
        // EN: First combine the reservoir for the current pixel.
        const Reservoir<LightSample> /*&*/self = plp.s->reservoirBuffer[srcResIndex][launchIndex];
        const ReservoirInfo selfResInfo = plp.s->reservoirInfoBuffer[srcResIndex].read(launchIndex);
        if (selfResInfo.recPDFEstimate > 0.0f) {
            combinedReservoir = self;
            selectedTargetDensity = selfResInfo.targetDensity;
        }
        uint32_t combinedStreamLength = self.getStreamLength();

        for (int nIdx = 0; nIdx < plp.f->numSpatialNeighbors; ++nIdx) {
            // JP: 周辺ピクセルの座標をランダムに決定。
            // EN: Randomly determine the coordinates of a neighboring pixel.
            float radius = plp.f->spatialNeighborRadius;
            float deltaX, deltaY;
            if (plp.f->useLowDiscrepancyNeighbors) {
                float2 delta = plp.s->spatialNeighborDeltas[(plp.spatialNeighborBaseIndex + nIdx) % 1024];
                deltaX = radius * delta.x;
                deltaY = radius * delta.y;
            }
            else {
                radius *= std::sqrt(rng.getFloat0cTo1o());
                float angle = 2 * Pi * rng.getFloat0cTo1o();
                deltaX = radius * std::cos(angle);
                deltaY = radius * std::sin(angle);
            }
            int2 nbCoord = make_int2(launchIndex.x + 0.5f + deltaX,
                                     launchIndex.y + 0.5f + deltaY);

            bool acceptedNeighbor;
            if constexpr (useUnbiasedEstimator) {
                acceptedNeighbor =
                    nbCoord.x >= 0 && nbCoord.x < plp.s->imageSize.x &&
                    nbCoord.y >= 0 && nbCoord.y < plp.s->imageSize.y;
                if (acceptedNeighbor) {
                    GBuffer2 nbGBuffer2 = plp.s->GBuffer2[bufIdx].read(nbCoord);
                    uint32_t nbMaterialSlot = nbGBuffer2.materialSlot;
                    acceptedNeighbor &= nbMaterialSlot != 0xFFFFFFFF;
                }
            }
            else {
                // JP: 隣接ピクセルのジオメトリ・マテリアルがあまりに異なる場合に候補サンプルを再利用すると
                //     バイアスが増えてしまうため、そのようなピクセルは棄却する。
                // EN: Reusing candidates from neighboring pixels with substantially different geometry/material
                //     leads to increased bias. Reject such a pixel.
                acceptedNeighbor = testNeighbor(bufIdx, nbCoord, dist, normalInWorld);
            }
            acceptedNeighbor &= nbCoord.x != launchIndex.x || nbCoord.y != launchIndex.y;
            if (acceptedNeighbor) {
                const Reservoir<LightSample> /*&*/neighbor = plp.s->reservoirBuffer[srcResIndex][nbCoord];
                const ReservoirInfo neighborInfo = plp.s->reservoirInfoBuffer[srcResIndex].read(nbCoord);

                // JP: 隣接ピクセルが持つ候補サンプルの「現在の」ピクセルにおける確率密度を計算する。
                // EN: Calculate the probability density at the "current" pixel of the candidate sample
                //     the neighboring pixel holds.
                LightSample nbLightSample = neighbor.getSample();
                float3 cont;
                if constexpr (useUnbiasedEstimator) {
                    if (plp.f->reuseVisibility) // ?
                        cont = performDirectLighting<true>(
                            positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
                    else
                        cont = performDirectLighting<false>(
                            positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
                }
                else {
                    cont = performDirectLighting<false>(positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
                }
                float targetDensity = convertToWeight(cont);

                // JP: 隣接ピクセルと現在のピクセルではターゲットPDFが異なるためサンプルはウェイトを持つ。
                // EN: The sample has a weight since the target PDFs of the neighboring pixel and the current
                //     are the different.
                uint32_t nbStreamLength = neighbor.getStreamLength();
                float weight = targetDensity * neighborInfo.recPDFEstimate * nbStreamLength;
                if (combinedReservoir.update(nbLightSample, weight, rng.getFloat0cTo1o())) {
                    selectedTargetDensity = targetDensity;
                    if constexpr (useUnbiasedEstimator)
                        selectedNeighborIndex = nIdx;
                    else
                        (void)selectedNeighborIndex;
                }

                combinedStreamLength += nbStreamLength;
            }
        }
        combinedReservoir.setStreamLength(combinedStreamLength);

        float weightForEstimate;
        if constexpr (useUnbiasedEstimator) {
            // JP: 推定関数をunbiasedとするための、生き残ったサンプルのウェイトを計算する。
            //     ここではReservoirの結合時とは逆に、サンプルは生き残った1つだが、
            //     ターゲットPDFは隣接ピクセルのものを評価する。
            // EN: Compute a weight for the survived sample to make the estimator unbiased.
            //     In contrast to the case where we combine reservoirs, the sample is only one survived and
            //     Evaluate target PDFs at the neighboring pixels here.
            LightSample selectedLightSample = combinedReservoir.getSample();

            float numWeight;
            float denomWeight;

            // JP: まずは現在のピクセルのターゲットPDFに対応する量を計算。
            // EN: First, calculate a quantity corresponding to the current pixel's target PDF.
            {
                float3 cont;
                if (plp.f->reuseVisibility) // ?
                    cont = performDirectLighting<true>(
                        positionInWorld, vOutLocal, shadingFrame, bsdf, selectedLightSample);
                else
                    cont = performDirectLighting<false>(
                        positionInWorld, vOutLocal, shadingFrame, bsdf, selectedLightSample);
                float targetDensityForSelf = convertToWeight(cont);
                if constexpr (useMIS_RIS) {
                    numWeight = targetDensityForSelf;
                    denomWeight = targetDensityForSelf * self.getStreamLength();
                }
                else {
                    numWeight = 1.0f;
                    denomWeight = 0.0f;
                    if (targetDensityForSelf > 0.0f)
                        denomWeight = self.getStreamLength();
                }
            }

            // JP: 続いて隣接ピクセルのターゲットPDFに対応する量を計算。
            // EN: Next, calculate quantities corresponding to the neighboring pixels' target PDFs.
            for (int nIdx = 0; nIdx < plp.f->numSpatialNeighbors; ++nIdx) {
                float radius = plp.f->spatialNeighborRadius;
                float deltaX, deltaY;
                if (plp.f->useLowDiscrepancyNeighbors) {
                    float2 delta = plp.s->spatialNeighborDeltas[(plp.spatialNeighborBaseIndex + nIdx) % 1024];
                    deltaX = radius * delta.x;
                    deltaY = radius * delta.y;
                }
                else {
                    radius *= std::sqrt(rng.getFloat0cTo1o());
                    float angle = 2 * Pi * rng.getFloat0cTo1o();
                    deltaX = radius * std::cos(angle);
                    deltaY = radius * std::sin(angle);
                }
                int2 nbCoord = make_int2(launchIndex.x + 0.5f + deltaX,
                                         launchIndex.y + 0.5f + deltaY);

                bool acceptedNeighbor =
                    nbCoord.x >= 0 && nbCoord.x < plp.s->imageSize.x &&
                    nbCoord.y >= 0 && nbCoord.y < plp.s->imageSize.y;
                acceptedNeighbor &= nbCoord.x != launchIndex.x || nbCoord.y != launchIndex.y;
                if (acceptedNeighbor) {
                    GBuffer2 nbGBuffer2 = plp.s->GBuffer2[bufIdx].read(nbCoord);

                    uint32_t nbMaterialSlot = nbGBuffer2.materialSlot;
                    if (nbMaterialSlot == 0xFFFFFFFF)
                        continue;

                    GBuffer0 nbGBuffer0 = plp.s->GBuffer0[bufIdx].read(nbCoord);
                    GBuffer1 nbGBuffer1 = plp.s->GBuffer1[bufIdx].read(nbCoord);
                    float3 nbPositionInWorld = nbGBuffer0.positionInWorld;
                    float3 nbNormalInWorld = nbGBuffer1.normalInWorld;
                    float2 nbTexCoord = make_float2(nbGBuffer0.texCoord_x, nbGBuffer1.texCoord_y);

                    const MaterialData &nbMat = plp.s->materialDataBuffer[nbMaterialSlot];

                    BSDF nbBsdf;
                    nbMat.setupBSDF(nbMat, nbTexCoord, &nbBsdf);
                    ReferenceFrame nbShadingFrame(nbNormalInWorld);
                    float3 nbVOut = plp.f->camera.position - nbPositionInWorld;
                    float nbDist = length(nbVOut);
                    nbVOut /= nbDist;
                    float3 nbVOutLocal = nbShadingFrame.toLocal(nbVOut);

                    const Reservoir<LightSample> /*&*/neighbor = plp.s->reservoirBuffer[srcResIndex][nbCoord];

                    // TODO: ウェイトの条件さえ満たしていれば、MISウェイト計算にはVisibilityはなくても良い？
                    //       要検討。
                    float3 cont;
                    if (plp.f->reuseVisibility) // ?
                        cont = performDirectLighting<true>(
                            nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, selectedLightSample);
                    else
                        cont = performDirectLighting<false>(
                            nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, selectedLightSample);
                    float nbTargetDensity = convertToWeight(cont);
                    uint32_t nbStreamLength = neighbor.getStreamLength();
                    if constexpr (useMIS_RIS) {
                        denomWeight += nbTargetDensity * nbStreamLength;
                        if (nIdx == selectedNeighborIndex)
                            numWeight = nbTargetDensity;
                    }
                    else {
                        if (nbTargetDensity > 0.0f)
                            denomWeight += nbStreamLength;
                    }
                }
            }

            weightForEstimate = numWeight / denomWeight;
        }
        else {
            weightForEstimate = 1.0f / combinedReservoir.getStreamLength();
        }
        // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
        // EN: Calculate the estimate of the reciprocal of the probability density that the current sample suvives.
        ReservoirInfo reservoirInfo;
        reservoirInfo.recPDFEstimate = weightForEstimate * combinedReservoir.getSumWeights() / selectedTargetDensity;
        reservoirInfo.targetDensity = selectedTargetDensity;
        if (!isfinite(reservoirInfo.recPDFEstimate)) {
            reservoirInfo.recPDFEstimate = 0.0f;
            reservoirInfo.targetDensity = 0.0f;
        }

        plp.s->rngBuffer.write(launchIndex, rng);
        plp.s->reservoirBuffer[dstResIndex][launchIndex] = combinedReservoir;
        plp.s->reservoirInfoBuffer[dstResIndex].write(launchIndex, reservoirInfo);
    }
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(combineSpatialNeighborsBiased)() {
    combineSpatialNeighbors<false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(combineSpatialNeighborsUnbiased)() {
    combineSpatialNeighbors<true>();
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(shading)() {
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
        const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

        BSDF bsdf;
        mat.setupBSDF(mat, texCoord, &bsdf);
        ReferenceFrame shadingFrame(normalInWorld);
        float3 vOut = normalize(camera.position - positionInWorld);
        float3 vOutLocal = shadingFrame.toLocal(vOut);

        uint32_t curResIndex = plp.currentReservoirIndex;
        const Reservoir<LightSample> /*&*/reservoir = plp.s->reservoirBuffer[curResIndex][launchIndex];
        const ReservoirInfo reservoirInfo = plp.s->reservoirInfoBuffer[curResIndex].read(launchIndex);

        // JP: 光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from a light source directly seeing.
        contribution = make_float3(0.0f);
        if (vOutLocal.z > 0) {
            float3 emittance = make_float3(0.0f, 0.0f, 0.0f);
            if (mat.emittance) {
                float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
                emittance = make_float3(texValue);
            }
            contribution += emittance / Pi;
        }

        // JP: 最終的に残ったサンプルとそのウェイトを使ってシェーディングを実行する。
        // EN: Perform shading using the sample survived in the end and its weight.
        const LightSample &lightSample = reservoir.getSample();
        float3 directCont = make_float3(0.0f);
        float recPDFEstimate = reservoirInfo.recPDFEstimate;
        if (recPDFEstimate > 0 && isfinite(recPDFEstimate))
            directCont = recPDFEstimate * performDirectLighting<true>(positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample);

        contribution += directCont;

        albedo = bsdf.getBaseColor(vOutLocal);
    }
    else {
        if (plp.s->envLightTexture) {
            float u = texCoord.x, v = texCoord.y;
            float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
            float3 emittance = make_float3(texValue);
            emittance *= plp.f->envLightPowerCoeff;

            contribution = emittance / Pi;

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
