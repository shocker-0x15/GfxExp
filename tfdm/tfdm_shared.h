#pragma once

#include "../common/common_shared.h"
#include "affine_arithmetic.h"
#if defined(__CUDA_ARCH__)
#include <cuda_fp16.h>
#endif

#define USE_DISPLACED_SURFACES 1
#define USE_WORKAROUND_FOR_CUDA_BC_TEX 1
#define STORE_BARYCENTRICS 0
#define OUTPUT_TRAVERSAL_STATS 1

namespace shared {
    static constexpr bool useMultipleRootOptimization = 1;

    static constexpr float probToSampleEnvLight = 0.25f;



    enum class LocalIntersectionType {
        Box = 0,
        TwoTriangle,
        Bilinear,
        BSpline
    };

    enum CustomHitKind : uint8_t {
        CustomHitKind_AABBFrontFace = 0,
        CustomHitKind_AABBBackFace,
        CustomHitKind_DisplacedSurfaceFrontFace,
        CustomHitKind_DisplacedSurfaceBackFace,
    };



    struct DisplacedSurfaceAttributes {
        Normal3D normalInObj;
#if OUTPUT_TRAVERSAL_STATS
        uint32_t numIterations;
#endif
    };



    struct GBufferRayType {
        enum Value {
            Primary,
            NumTypes
        } value;

        CUDA_DEVICE_FUNCTION constexpr GBufferRayType(Value v = Primary) : value(v) {}

        CUDA_DEVICE_FUNCTION operator uint32_t() const {
            return static_cast<uint32_t>(value);
        }
    };

    struct PathTracingRayType {
        enum Value {
            Closest,
            Visibility,
            NumTypes
        } value;

        CUDA_DEVICE_FUNCTION constexpr PathTracingRayType(Value v = Closest) : value(v) {}

        CUDA_DEVICE_FUNCTION operator uint32_t() const {
            return static_cast<uint32_t>(value);
        }
    };

    constexpr uint32_t maxNumRayTypes = 2;



    struct PerspectiveCamera {
        float aspect;
        float fovY;
        Point3D position;
        Matrix3x3 orientation;

        CUDA_COMMON_FUNCTION Point2D calcScreenPosition(const Point3D &posInWorld) const {
            Matrix3x3 invOri = invert(orientation);
            Point3D posInView(invOri * (posInWorld - position));
            Point2D posAtZ1(posInView.x / posInView.z, posInView.y / posInView.z);
            float h = 2 * std::tan(fovY / 2);
            float w = aspect * h;
            return Point2D(1 - (posAtZ1.x + 0.5f * w) / w,
                           1 - (posAtZ1.y + 0.5f * h) / h);
        }
    };



    struct HitPointParams {
        RGB albedo;
        Point3D positionInWorld;
        Point3D prevPositionInWorld;
        Normal3D normalInWorld;
        Point2D texCoord;
        uint32_t materialSlot;
        uint32_t geomInstSlot : 31;
        uint32_t isTfdmMesh : 1;
        uint32_t primIndex;
#if OUTPUT_TRAVERSAL_STATS
        uint32_t numTravIterations;
#endif
    };



    struct LightSample {
        RGB emittance;
        Point3D position;
        Normal3D normal;
        unsigned int atInfinity : 1;
    };



    struct PickInfo {
        uint32_t instSlot;
        uint32_t geomInstSlot;
        uint32_t primIndex;
        uint32_t matSlot;
        Point3D positionInWorld;
        Normal3D normalInWorld;
        RGB albedo;
        RGB emittance;
        unsigned int hit : 1;
    };



    struct GBuffer0 {
        Point3D positionInWorld;
        float texCoord_x;
    };

    struct GBuffer1 {
        Normal3D normalInWorld;
        float texCoord_y;
    };

    struct GBuffer2 {
#if defined(__CUDA_ARCH__)
        __half2 motionVector;
#else
        struct {
            uint16_t motionVectorX;
            uint16_t motionVectorY;
        };
#endif
        uint32_t materialSlot;
        uint32_t geomInstSlot : 31;
        uint32_t isTfdmMesh : 1;
        uint32_t primIndex;
    };



    struct PathTraceWriteOnlyPayload {
        Point3D nextOrigin;
        Vector3D nextDirection;
    };

    struct PathTraceReadWritePayload {
        PCG32RNG rng;
        float initImportance;
        RGB alpha;
        RGB contribution;
        float prevDirPDensity;
        unsigned int maxLengthTerminate : 1;
        unsigned int terminate : 1;
        unsigned int pathLength : 6;
    };

    
    
    struct StaticPipelineLaunchParameters {
        int2 imageSize;
        optixu::NativeBlockBuffer2D<PCG32RNG> rngBuffer;

        optixu::NativeBlockBuffer2D<GBuffer0> GBuffer0[2];
        optixu::NativeBlockBuffer2D<GBuffer1> GBuffer1[2];
        optixu::NativeBlockBuffer2D<GBuffer2> GBuffer2[2];

#if OUTPUT_TRAVERSAL_STATS
        optixu::NativeBlockBuffer2D<uint32_t> numTravItrsBuffer;
#endif

        ROBuffer<MaterialData> materialDataBuffer;
        ROBuffer<GeometryInstanceData> geometryInstanceDataBuffer;
        ROBuffer<GeometryInstanceDataForTFDM> geomInstTfdmDataBuffer;
        LightDistribution lightInstDist;
        RegularConstantContinuousDistribution2D envLightImportanceMap;
        CUtexObject envLightTexture;

        optixu::NativeBlockBuffer2D<float4> beautyAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> albedoAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> normalAccumBuffer;
    };

    struct PerFramePipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        uint32_t numAccumFrames;
        uint32_t frameIndex;

        ROBuffer<InstanceData> instanceDataBuffer;

        PerspectiveCamera camera;
        PerspectiveCamera prevCamera;

        float envLightPowerCoeff;
        float envLightRotation;

        int2 mousePosition;
        PickInfo* pickInfo;

        unsigned int maxPathLength : 4;
        unsigned int bufferIndex : 1;
        unsigned int resetFlowBuffer : 1;
        unsigned int enableJittering : 1;
        unsigned int enableEnvLight : 1;
        unsigned int enableDebugPrint : 1;
        unsigned int showBaseEdges : 1;

        uint32_t debugSwitches;
        void setDebugSwitch(int32_t idx, bool b) {
            debugSwitches &= ~(1 << idx);
            debugSwitches |= b << idx;
        }
        CUDA_COMMON_FUNCTION bool getDebugSwitch(int32_t idx) const {
            return (debugSwitches >> idx) & 0b1;
        }
    };
    
    struct PipelineLaunchParameters {
        StaticPipelineLaunchParameters* s;
        PerFramePipelineLaunchParameters* f;
    };



    enum class BufferToDisplay {
        NoisyBeauty = 0,
        Albedo,
        Normal,
        TexCoord,
        Flow,
        TraversalIterations,
        DenoisedBeauty,
    };



    using AABBAttributeSignature = optixu::AttributeSignature<float, float>;
    using DisplacedSurfaceAttributeSignature = optixu::AttributeSignature<float, float, DisplacedSurfaceAttributes>;

    using PrimaryRayPayloadSignature =
        optixu::PayloadSignature<shared::HitPointParams*, shared::PickInfo*>;
    using PathTraceRayPayloadSignature =
        optixu::PayloadSignature<shared::PathTraceWriteOnlyPayload*, shared::PathTraceReadWritePayload*>;
    using VisibilityRayPayloadSignature =
        optixu::PayloadSignature<float>;
}



#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

#if defined(PURE_CUDA)
CUDA_CONSTANT_MEM shared::PipelineLaunchParameters plp;
#else
RT_PIPELINE_LAUNCH_PARAMETERS shared::PipelineLaunchParameters plp;
#endif

#include "../common/common_device.cuh"

template <bool useSolidAngleSampling>
CUDA_DEVICE_FUNCTION CUDA_INLINE void sampleLight(
    const Point3D &shadingPoint,
    float ul, bool sampleEnvLight, float u0, float u1,
    shared::LightSample* lightSample, float* areaPDensity) {
    using namespace shared;
    CUtexObject texEmittance = 0;
    RGB emittance(0.0f, 0.0f, 0.0f);
    Point2D texCoord;
    if (sampleEnvLight) {
        float u, v;
        float uvPDF;
        plp.s->envLightImportanceMap.sample(u0, u1, &u, &v, &uvPDF);
        float phi = 2 * Pi * u;
        float theta = Pi * v;

        float posPhi = phi - plp.f->envLightRotation;
        posPhi = posPhi - floorf(posPhi / (2 * Pi)) * 2 * Pi;

        Vector3D direction = fromPolarYUp(posPhi, theta);
        Point3D position(direction.x, direction.y, direction.z);
        lightSample->position = position;
        lightSample->atInfinity = true;

        lightSample->normal = Normal3D(-position);

        // JP: テクスチャー空間中のPDFを面積に関するものに変換する。
        // EN: convert the PDF in texture space to one with respect to area.
        // The true value is: lim_{l to inf} uvPDF / (2 * Pi * Pi * std::sin(theta)) / l^2
        *areaPDensity = uvPDF / (2 * Pi * Pi * std::sin(theta));

        texEmittance = plp.s->envLightTexture;
        // JP: 環境マップテクスチャーの値に係数をかけて、通常の光源と同じように返り値を光束発散度
        //     として扱えるようにする。
        // EN: Multiply a coefficient to make the return value possible to be handled as luminous emittance.
        emittance = RGB(Pi * plp.f->envLightPowerCoeff);
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
        if (instProb == 0.0f) {
            *areaPDensity = 0.0f;
            return;
        }
        //Assert(inst.lightGeomInstDist.integral() > 0.0f,
        //       "Non-emissive inst %u, prob %g, u: %g(0x%08x).", instIndex, instProb, ul, *(uint32_t*)&ul);


        // JP: 次にサンプルしたインスタンスに属するジオメトリインスタンスをサンプルする。
        // EN: Next, sample a geometry instance which belongs to the sampled instance.
        float geomInstProb;
        float uPrim;
        uint32_t geomInstIndexInInst = inst.lightGeomInstDist.sample(uGeomInst, &geomInstProb, &uPrim);
        uint32_t geomInstIndex = inst.geomInstSlots[geomInstIndexInInst];
        lightProb *= geomInstProb;
        const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[geomInstIndex];
        if (geomInstProb == 0.0f) {
            *areaPDensity = 0.0f;
            return;
        }
        //Assert(geomInst.emitterPrimDist.integral() > 0.0f,
        //       "Non-emissive geom inst %u, prob %g, u: %g.", geomInstIndex, geomInstProb, uGeomInst);

        // JP: 最後に、サンプルしたジオメトリインスタンスに属するプリミティブをサンプルする。
        // EN: Finally, sample a primitive which belongs to the sampled geometry instance.
        float primProb;
        uint32_t primIndex = geomInst.emitterPrimDist.sample(uPrim, &primProb);
        lightProb *= primProb;

        //printf("%u-%u-%u: %g\n", instIndex, geomInstIndex, primIndex, lightProb);

        const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

        const shared::Triangle &tri = geomInst.triangleBuffer[primIndex];
        const shared::Vertex (&v)[3] = {
            geomInst.vertexBuffer[tri.index0],
            geomInst.vertexBuffer[tri.index1],
            geomInst.vertexBuffer[tri.index2]
        };
        const Point3D p[3] = {
            inst.transform * v[0].position,
            inst.transform * v[1].position,
            inst.transform * v[2].position,
        };

        Normal3D geomNormal(cross(p[1] - p[0], p[2] - p[0]));

        float t0, t1, t2;
        if constexpr (useSolidAngleSampling) {
            // Uniform sampling in solid angle subtended by the triangle for the shading point.
            float dist;
            Vector3D dir;
            float dirPDF;
            {
                const auto project = [](const Vector3D &vA, const Vector3D &vB) {
                    return normalize(vA - dot(vA, vB) * vB);
                };

                // TODO: ? compute in the local coordinates.
                Vector3D A = normalize(p[0] - shadingPoint);
                Vector3D B = normalize(p[1] - shadingPoint);
                Vector3D C = normalize(p[2] - shadingPoint);
                Vector3D cAB = normalize(cross(A, B));
                Vector3D cBC = normalize(cross(B, C));
                Vector3D cCA = normalize(cross(C, A));
                //float cos_a = dot(B, C);
                //float cos_b = dot(C, A);
                float cos_c = dot(A, B);
                float cosAlpha = -dot(cAB, cCA);
                float cosBeta = -dot(cBC, cAB);
                float cosGamma = -dot(cCA, cBC);
                float alpha = std::acos(cosAlpha);
                float sinAlpha = std::sqrt(1 - pow2(cosAlpha));
                float sphArea = alpha + std::acos(cosBeta) + std::acos(cosGamma) - Pi;

                float sphAreaHat = sphArea * u0;
                float s = std::sin(sphAreaHat - alpha);
                float t = std::cos(sphAreaHat - alpha);
                float uu = t - cosAlpha;
                float vv = s + sinAlpha * cos_c;
                float q = ((vv * t - uu * s) * cosAlpha - vv) / ((vv * s + uu * t) * sinAlpha);

                Vector3D cHat = q * A + std::sqrt(1 - pow2(q)) * project(C, A);
                float z = 1 - u1 * (1 - dot(cHat, B));
                Vector3D P = z * B + std::sqrt(1 - pow2(z)) * project(cHat, B);

                const auto restoreBarycentrics = [&geomNormal]
                (const Point3D &org, const Vector3D &dir,
                 const Point3D &pA, const Point3D &pB, const Point3D &pC,
                 float* dist, float* b1, float* b2) {
                     Vector3D eAB = pB - pA;
                     Vector3D eAC = pC - pA;
                     Vector3D pVec = cross(dir, eAC);
                     float recDet = 1.0f / dot(eAB, pVec);
                     Vector3D tVec = org - pA;
                     *b1 = dot(tVec, pVec) * recDet;
                     Vector3D qVec = cross(tVec, eAB);
                     *b2 = dot(dir, qVec) * recDet;
                     *dist = dot(eAC, qVec) * recDet;
                };
                dir = P;
                restoreBarycentrics(shadingPoint, dir, p[0], p[1], p[2], &dist, &t1, &t2);
                t0 = 1 - t1 - t2;
                dirPDF = 1 / sphArea;
            }

            geomNormal = normalize(geomNormal);
            float lpCos = -dot(dir, geomNormal);
            if (lpCos > 0 && isfinite(dirPDF))
                *areaPDensity = lightProb * (dirPDF * lpCos / pow2(dist));
            else
                *areaPDensity = 0.0f;
        }
        else {
            // Uniform sampling on unit triangle
            // A Low-Distortion Map Between Triangle and Square
            t0 = 0.5f * u0;
            t1 = 0.5f * u1;
            float offset = t1 - t0;
            if (offset > 0)
                t1 += offset;
            else
                t0 -= offset;
            t2 = 1 - (t0 + t1);

            float recArea = 2.0f / length(geomNormal);
            *areaPDensity = lightProb * recArea;
        }
        lightSample->position = t0 * p[0] + t1 * p[1] + t2 * p[2];
        lightSample->atInfinity = false;
        lightSample->normal = t0 * v[0].normal + t1 * v[1].normal + t2 * v[2].normal;
        lightSample->normal = normalize(inst.normalMatrix * lightSample->normal);

        if (mat.emittance) {
            texEmittance = mat.emittance;
            emittance = RGB(1.0f, 1.0f, 1.0f);
            texCoord = t0 * v[0].texCoord + t1 * v[1].texCoord + t2 * v[2].texCoord;
        }
    }

    if (texEmittance) {
        float4 texValue = tex2DLod<float4>(texEmittance, texCoord.x, texCoord.y, 0.0f);
        emittance *= RGB(texValue.x, texValue.y, texValue.z);
    }
    lightSample->emittance = emittance;
}

template <typename RayType, bool withVisibility>
CUDA_DEVICE_FUNCTION CUDA_INLINE RGB performDirectLighting(
    const Point3D &shadingPoint, const Vector3D &vOutLocal, const ReferenceFrame &shadingFrame,
    const BSDF &bsdf, const shared::LightSample &lightSample) {
    using namespace shared;
    Vector3D shadowRayDir = lightSample.atInfinity ?
        Vector3D(lightSample.position) :
        (lightSample.position - shadingPoint);
    float dist2 = shadowRayDir.sqLength();
    float dist = std::sqrt(dist2);
    shadowRayDir /= dist;
    Vector3D shadowRayDirLocal = shadingFrame.toLocal(shadowRayDir);

    float lpCos = dot(-shadowRayDir, lightSample.normal);
    float spCos = shadowRayDirLocal.z;

    float visibility = 1.0f;
    if constexpr (withVisibility) {
        if (lightSample.atInfinity)
            dist = 1e+10f;
        shared::VisibilityRayPayloadSignature::trace(
            plp.f->travHandle,
            shadingPoint.toNative(), shadowRayDir.toNative(), 0.0f, dist * 0.9999f, 0.0f,
            0xFF, OPTIX_RAY_FLAG_NONE,
            RayType::Visibility, shared::maxNumRayTypes, RayType::Visibility,
            visibility);
    }

    if (visibility > 0 && lpCos > 0) {
        RGB Le = lightSample.emittance / Pi; // assume diffuse emitter.
        RGB fsValue = bsdf.evaluate(vOutLocal, shadowRayDirLocal);
        float G = lpCos * std::fabs(spCos) / dist2;
        RGB ret = fsValue * Le * G;
        return ret;
    }
    else {
        return RGB(0.0f, 0.0f, 0.0f);
    }
}

template <typename RayType>
CUDA_DEVICE_FUNCTION CUDA_INLINE bool evaluateVisibility(
    const Point3D &shadingPoint, const shared::LightSample &lightSample) {
    using namespace shared;
    Vector3D shadowRayDir = lightSample.atInfinity ?
        Vector3D(lightSample.position) :
        (lightSample.position - shadingPoint);
    float dist2 = shadowRayDir.sqLength();
    float dist = std::sqrt(dist2);
    shadowRayDir /= dist;
    if (lightSample.atInfinity)
        dist = 1e+10f;

    float visibility = 1.0f;
    shared::VisibilityRayPayloadSignature::trace(
        plp.f->travHandle,
        shadingPoint.toNative(), shadowRayDir.toNative(), 0.0f, dist * 0.9999f, 0.0f,
        0xFF, OPTIX_RAY_FLAG_NONE,
        RayType::Visibility, shared::maxNumRayTypes, RayType::Visibility,
        visibility);

    return visibility > 0.0f;
}

template <bool computeHypotheticalAreaPDensity, bool useSolidAngleSampling>
CUDA_DEVICE_FUNCTION CUDA_INLINE void computeSurfacePoint(
    const shared::InstanceData &inst,
    const shared::GeometryInstanceData &geomInst,
    bool isDisplacedTriangleHit,
    uint32_t primIndex, float b1, float b2,
    const Point3D &referencePoint,
    Point3D* positionInWorld, Normal3D* shadingNormalInWorld, Vector3D* texCoord0DirInWorld,
    Normal3D* geometricNormalInWorld, Point2D* texCoord,
    float* hypAreaPDensity) {
    using namespace shared;
    const Triangle &tri = geomInst.triangleBuffer[primIndex];
    const Vertex &v0 = geomInst.vertexBuffer[tri.index0];
    const Vertex &v1 = geomInst.vertexBuffer[tri.index1];
    const Vertex &v2 = geomInst.vertexBuffer[tri.index2];
    const Point3D p[3] = {
        transformPointFromObjectToWorldSpace(v0.position),
        transformPointFromObjectToWorldSpace(v1.position),
        transformPointFromObjectToWorldSpace(v2.position),
    };
    float b0 = 1 - (b1 + b2);

    // JP: ヒットポイントのローカル座標中の各値を計算する。
    // EN: Compute hit point properties in the local coordinates.
    Normal3D shadingNormal;
    Normal3D geometricNormal;
    Vector3D texCoord0Dir = b0 * v0.texCoord0Dir + b1 * v1.texCoord0Dir + b2 * v2.texCoord0Dir;
    if (isDisplacedTriangleHit) {
        DisplacedSurfaceAttributes hitAttrs;
        DisplacedSurfaceAttributeSignature::get(nullptr, nullptr, &hitAttrs);
        shadingNormal = hitAttrs.normalInObj;
        geometricNormal = transformNormalFromObjectToWorldSpace(shadingNormal);
        *positionInWorld = Point3D(optixGetWorldRayOrigin())
            + optixGetRayTmax() * Vector3D(optixGetWorldRayDirection());
        texCoord0Dir += -dot(shadingNormal, texCoord0Dir) * shadingNormal;
    }
    else {
        shadingNormal = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
        geometricNormal = Normal3D(cross(p[1] - p[0], p[2] - p[0]));
        *positionInWorld = b0 * p[0] + b1 * p[1] + b2 * p[2];
    }

    float area;
    if constexpr (computeHypotheticalAreaPDensity && !useSolidAngleSampling)
        area = 0.5f * length(geometricNormal);
    else
        (void)area;
    *texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

    // JP: ローカル座標中の値をワールド座標中の値へと変換する。
    // EN: Convert the local properties to ones in world coordinates.
    *shadingNormalInWorld = normalize(transformNormalFromObjectToWorldSpace(shadingNormal));
    *texCoord0DirInWorld = normalize(transformVectorFromObjectToWorldSpace(texCoord0Dir));
    *geometricNormalInWorld = normalize(geometricNormal);
    if (!shadingNormalInWorld->allFinite()) {
        *shadingNormalInWorld = Normal3D(0, 0, 1);
        *texCoord0DirInWorld = Vector3D(1, 0, 0);
    }
    if (!texCoord0DirInWorld->allFinite()) {
        Vector3D bitangent;
        makeCoordinateSystem(*shadingNormalInWorld, texCoord0DirInWorld, &bitangent);
    }

    if constexpr (computeHypotheticalAreaPDensity) {
        // JP: 交点をExplicit Light Samplingでサンプルする場合の仮想的な確率密度を求める。
        // EN: Compute a hypothetical probability density with which the intersection point
        //     is sampled by explicit light sampling.
        float lightProb = 1.0f;
        if (plp.s->envLightTexture && plp.f->enableEnvLight)
            lightProb *= (1 - probToSampleEnvLight);
        float instImportance = inst.lightGeomInstDist.integral();
        lightProb *= (pow2(inst.uniformScale) * instImportance) / plp.s->lightInstDist.integral();
        lightProb *= geomInst.emitterPrimDist.integral() / instImportance;
        if (!isfinite(lightProb)) {
            *hypAreaPDensity = 0.0f;
            return;
        }
        lightProb *= geomInst.emitterPrimDist.evaluatePMF(primIndex);
        if constexpr (useSolidAngleSampling) {
            // TODO: ? compute in the local coordinates.
            Vector3D A = normalize(p[0] - referencePoint);
            Vector3D B = normalize(p[1] - referencePoint);
            Vector3D C = normalize(p[2] - referencePoint);
            Vector3D cAB = normalize(cross(A, B));
            Vector3D cBC = normalize(cross(B, C));
            Vector3D cCA = normalize(cross(C, A));
            float cosAlpha = -dot(cAB, cCA);
            float cosBeta = -dot(cBC, cAB);
            float cosGamma = -dot(cCA, cBC);
            float sphArea = std::acos(cosAlpha) + std::acos(cosBeta) + std::acos(cosGamma) - Pi;
            float dirPDF = 1.0f / sphArea;
            Vector3D refDir = referencePoint - *positionInWorld;
            float dist2ToRefPoint = sqLength(refDir);
            refDir /= std::sqrt(dist2ToRefPoint);
            float lpCos = dot(refDir, *geometricNormalInWorld);
            if (lpCos > 0 && isfinite(dirPDF))
                *hypAreaPDensity = lightProb * (dirPDF * lpCos / dist2ToRefPoint);
            else
                *hypAreaPDensity = 0.0f;
        }
        else {
            *hypAreaPDensity = lightProb / area;
        }
        Assert(isfinite(*hypAreaPDensity), "hypP: %g, area: %g", *hypAreaPDensity, area);
    }
    else {
        (void)*hypAreaPDensity;
    }
}



struct HitPointParameter {
    float b1, b2;
    int32_t primIndex;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static HitPointParameter get() {
        using namespace shared;

        HitPointParameter ret;
        OptixPrimitiveType primType = optixGetPrimitiveType();
        if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
            float2 bc = optixGetTriangleBarycentrics();
            ret.b1 = bc.x;
            ret.b2 = bc.y;
        }
        else {
#if USE_DISPLACED_SURFACES
            DisplacedSurfaceAttributeSignature::get(&ret.b1, &ret.b2, nullptr);
#else
            AABBAttributeSignature::get(&ret.b1, &ret.b2);
#endif
        }
        ret.primIndex = optixGetPrimitiveIndex();
        return ret;
    }
};

struct HitGroupSBTRecordData {
    uint32_t geomInstSlot;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



struct Texel {
    int16_t x;
    int16_t y;
    int16_t lod;

    CUDA_DEVICE_FUNCTION bool operator==(const Texel &r) const {
        return x == r.x && y == r.y && lod == r.lod;
    }
    CUDA_DEVICE_FUNCTION bool operator!=(const Texel &r) const {
        return x != r.x || y != r.y || lod != r.lod;
    }
};

CUDA_DEVICE_FUNCTION CUDA_INLINE void up(Texel &texel) {
    ++texel.lod;
    texel.x = floorDiv(texel.x, 2);
    texel.y = floorDiv(texel.y, 2);
    //texel.x /= 2;
    //texel.y /= 2;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void down(Texel &texel) {
    --texel.lod;
    texel.x *= 2;
    texel.y *= 2;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void down(Texel &texel, bool signX, bool signY) {
    --texel.lod;
    texel.x = 2 * texel.x + signX;
    texel.y = 2 * texel.y + signY;
    //texel.x = 2 * texel.x + signX;
    //texel.y = 2 * texel.y + signY;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void next(Texel &texel, int32_t maxDepth) {
    while (true) {
        switch (2 * floorMod(texel.x, 2) + floorMod(texel.y, 2)) {
        //switch (2 * (texel.x % 2) + texel.y % 2) {
        case 1:
            --texel.y;
            ++texel.x;
            return;
        case 3:
            up(texel);
            if (texel.lod > maxDepth)
                return;
            break;
        default:
            ++texel.y;
            return;
        }
    }
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void next(Texel &texel, bool signX, bool signY, int32_t maxDepth) {
    while (true) {
        switch (2 * floorMod(texel.x + signX, 2) + floorMod(texel.y + signY, 2)) {
        //switch (2 * ((texel.x + signX) % 2) + (texel.y + signY) % 2) {
        case 1:
            texel.y += signY ? 1 : -1;
            texel.x += signX ? -1 : 1;
            return;
        case 3:
            up(texel);
            if (texel.lod > maxDepth)
                return;
            break;
        default:
            texel.y += signY ? -1 : 1;
            return;
        }
    }
}

enum class TriangleSquareIntersection2DResult {
    SquareOutsideTriangle = 0,
    SquareInsideTriangle,
    SquareOverlappingTriangle
};

CUDA_DEVICE_FUNCTION CUDA_INLINE TriangleSquareIntersection2DResult testTriangleSquareIntersection2D(
    const Point2D triPs[3], bool tcFlipped, const Vector2D triEdgeNormals[3],
    const Point2D &triAabbMinP, const Point2D &triAabbMaxP,
    const Point2D &squareCenter, float squareHalfWidth) {
    const Vector2D vSquareCenter = static_cast<Vector2D>(squareCenter);
    const Point2D relTriPs[] = {
        triPs[0] - vSquareCenter,
        triPs[1] - vSquareCenter,
        triPs[2] - vSquareCenter,
    };

    // JP: テクセルのAABBと三角形のAABBのIntersectionを計算する。
    // EN: Test intersection between the texel AABB and the triangle AABB.
    if (any(min(Point2D(squareHalfWidth), triAabbMaxP - vSquareCenter) <=
            max(Point2D(-squareHalfWidth), triAabbMinP - vSquareCenter)))
        return TriangleSquareIntersection2DResult::SquareOutsideTriangle;

    // JP: いずれかの三角形のエッジの法線方向にテクセルがあるならテクセルは三角形の外にある。
    // EN: Texel is outside of the triangle if the texel is in the normal direction of any edge.
    for (int eIdx = 0; eIdx < 3; ++eIdx) {
        Vector2D eNormal = (tcFlipped ? -1 : 1) * triEdgeNormals[eIdx];
        Bool2D b = eNormal >= Vector2D(0.0f);
        Vector2D e = static_cast<Vector2D>(relTriPs[eIdx]) +
            Vector2D((b.x ? 1 : -1) * squareHalfWidth,
                     (b.y ? 1 : -1) * squareHalfWidth);
        if (dot(eNormal, e) <= 0)
            return TriangleSquareIntersection2DResult::SquareOutsideTriangle;
    }

    // JP: テクセルが三角形のエッジとかぶっているかどうかを調べる。
    // EN: Test if the texel is overlapping with some edges of the triangle.
    for (int i = 0; i < 4; ++i) {
        Point2D corner(
            (i % 2 ? -1 : 1) * squareHalfWidth,
            (i / 2 ? -1 : 1) * squareHalfWidth);
        for (int eIdx = 0; eIdx < 3; ++eIdx) {
            const Point2D &o = relTriPs[eIdx];
            const Vector2D &e1 = relTriPs[(eIdx + 1) % 3] - o;
            Vector2D e2 = corner - o;
            if ((tcFlipped ? -1 : 1) * cross(e1, e2) < 0)
                return TriangleSquareIntersection2DResult::SquareOverlappingTriangle;
        }
    }

    // JP: それ以外の場合はテクセルは三角形に囲まれている。
    // EN: Otherwise, the texel is encompassed by the triangle.
    return TriangleSquareIntersection2DResult::SquareInsideTriangle;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void findRoots(
    const Point2D &triAabbMinP, const Point2D &triAabbMaxP, const int32_t maxDepth, uint32_t targetMipLevel,
    Texel* const roots, uint32_t* const numRoots) {
    using namespace shared;
    static_assert(useMultipleRootOptimization, "Naive method is not implemented.");
    const Vector2D d = triAabbMaxP - triAabbMinP;
    const uint32_t largerDim = d.y > d.x;
    int32_t startMipLevel = maxDepth - prevPowOf2Exponent(static_cast<uint32_t>(1.0f / d[largerDim])) - 1;
    startMipLevel = /*std::*/max(startMipLevel, 0);
    while (true) {
        const float res = std::pow(2.0f, static_cast<float>(maxDepth - startMipLevel));
        const int32_t minTexelX = static_cast<int32_t>(std::floor(res * triAabbMinP.x));
        const int32_t minTexelY = static_cast<int32_t>(std::floor(res * triAabbMinP.y));
        const int32_t maxTexelX = static_cast<int32_t>(std::floor(res * triAabbMaxP.x));
        const int32_t maxTexelY = static_cast<int32_t>(std::floor(res * triAabbMaxP.y));
        if ((maxTexelX - minTexelX) < 2 && (maxTexelY - minTexelY) < 2 &&
            startMipLevel >= targetMipLevel) {
            *numRoots = 0;
            for (int y = minTexelY; y <= maxTexelY; ++y) {
                for (int x = minTexelX; x <= maxTexelX; ++x) {
                    Texel &root = roots[(*numRoots)++];
                    root.x = x;
                    root.y = y;
                    root.lod = startMipLevel;
                }
            }
            break;
        }
        ++startMipLevel;
    }
}



#if !defined(PURE_CUDA) || defined(CUDAU_CODE_COMPLETION)

CUDA_DEVICE_FUNCTION bool isCursorPixel() {
    return plp.f->mousePosition == make_int2(optixGetLaunchIndex());
}

CUDA_DEVICE_FUNCTION bool getDebugPrintEnabled() {
    return plp.f->enableDebugPrint;
}

#endif

#endif // #if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
