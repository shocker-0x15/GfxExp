﻿#pragma once

#include "../common/common_shared.h"

namespace shared {
    static constexpr float probToSampleEnvLight = 0.25f;
    static constexpr float pathTerminationFactor = 0.01f; // c in the paper.
    static constexpr uint32_t numTrainingDataPerFrame = 1 << 16;
    static constexpr uint32_t trainBufferSize = 2 * numTrainingDataPerFrame;
    static constexpr bool useReflectanceFactorization = true;



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
            Baseline,
            NRC,
            Visibility,
            NumTypes
        } value;

        CUDA_DEVICE_FUNCTION constexpr PathTracingRayType(Value v = Baseline) : value(v) {}

        CUDA_DEVICE_FUNCTION operator uint32_t() const {
            return static_cast<uint32_t>(value);
        }
    };

    constexpr uint32_t maxNumRayTypes = 3;



    struct PerspectiveCamera {
        float aspect;
        float fovY;
        Point3D position;
        Matrix3x3 orientation;

        CUDA_COMMON_FUNCTION Point2D calcScreenPosition(const Point3D &posInWorld) const {
            const Matrix3x3 invOri = invert(orientation);
            const Point3D posInView(invOri * (posInWorld - position));
            const Point2D posAtZ1(posInView.x / posInView.z, posInView.y / posInView.z);
            const float h = 2 * std::tan(fovY / 2);
            const float w = aspect * h;
            return Point2D(1 - (posAtZ1.x + 0.5f * w) / w,
                           1 - (posAtZ1.y + 0.5f * h) / h);
        }
    };



    struct HitPointParams {
        RGB albedo;
        Point3D positionInWorld;
        Point3D prevPositionInWorld;
        Normal3D shadingNormalInWorld;
        uint32_t instSlot;
        uint32_t geomInstSlot;
        uint32_t primIndex;
        uint16_t qbcB;
        uint16_t qbcC;
    };



    struct LightSample {
        RGB emittance;
        Point3D position;
        Normal3D normal;
        uint32_t atInfinity : 1;

        CUDA_COMMON_FUNCTION LightSample() : atInfinity(false) {}
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
        uint32_t hit : 1;
    };



    struct GBuffer0Elements {
        uint32_t instSlot;
        uint32_t geomInstSlot;
        uint32_t primIndex;
        uint16_t qbcB;
        uint16_t qbcC;
    };

    struct GBuffer1Elements {
        Vector2D motionVector;
    };



    struct RadianceQuery {
        Point3D position;
        float normal_phi;
        float normal_theta;
        float vOut_phi;
        float vOut_theta;
        float roughness;
        RGB diffuseReflectance;
        RGB specularReflectance;

        CUDA_DEVICE_FUNCTION bool isValid() const {
            return
                position.allFinite() &&
                stc::isfinite(normal_phi) && stc::isfinite(normal_theta) &&
                stc::isfinite(vOut_phi) && stc::isfinite(vOut_theta) &&
                stc::isfinite(roughness) &&
                diffuseReflectance.allFinite() &&
                specularReflectance.allFinite();
        }
    };
    
    struct TerminalInfo {
        RGB alpha;
        uint32_t hasQuery : 1;
        // for stats/debug
        uint32_t pathLength : 8;
        uint32_t isTrainingPixel : 1;
        uint32_t isUnbiasedTile : 1;
    };

    static constexpr uint32_t invalidVertexDataIndex = 0x007FFFFF;

    struct TrainingVertexInfo {
        RGB localThroughput;
        uint32_t prevVertexDataIndex : 23;
        // for stats/debug
        uint32_t pathLength : 8;
    };

    struct TrainingSuffixTerminalInfo {
        uint32_t prevVertexDataIndex : 23;
        uint32_t hasQuery : 1;
        // for stats/debug
        uint32_t pathLength : 8;
    };

    class LinearCongruentialGenerator {
        static constexpr uint32_t a = 1103515245;
        static constexpr uint32_t c = 12345;
        static constexpr uint32_t m = 1u << 31;
        uint32_t m_state;

    public:
        LinearCongruentialGenerator() : m_state(0) {}

        CUDA_COMMON_FUNCTION void setState(uint32_t seed) {
            m_state = seed;
        }

        CUDA_COMMON_FUNCTION uint32_t next() {
            m_state = ((m_state * a) + c) % m;
            return m_state;
        }
    };



    struct PathTraceWriteOnlyPayload {
        Point3D nextOrigin;
        Vector3D nextDirection;
    };

    template <bool useNRC>
    struct PathTraceReadWritePayload;

    template <>
    struct PathTraceReadWritePayload<false> {
        PCG32RNG rng;
        float initImportance;
        RGB alpha;
        RGB contribution;
        float prevDirPDensity;
        uint32_t maxLengthTerminate : 1;
        uint32_t terminate : 1;
        uint32_t pathLength : 6;
    };

    template <>
    struct PathTraceReadWritePayload<true> {
        PCG32RNG rng;
        float initImportance;
        RGB alpha;
        RGB contribution;
        float prevDirPDensity;

        uint32_t linearTileIndex;
        float primaryPathSpread;
        float curSqrtPathSpread;
        RGB prevLocalThroughput;
        uint32_t prevTrainDataIndex;
        uint32_t renderingPathEndsWithCache : 1;
        uint32_t isTrainingPath : 1;
        uint32_t isUnbiasedTrainingTile : 1;
        uint32_t trainingSuffixEndsWithCache : 1;

        uint32_t maxLengthTerminate : 1;
        uint32_t terminate : 1;
        uint32_t pathLength : 6;
    };

    
    
    struct StaticPipelineLaunchParameters {
        int2 imageSize;
        optixu::NativeBlockBuffer2D<PCG32RNG> rngBuffer;

        optixu::NativeBlockBuffer2D<GBuffer0Elements> GBuffer0[2];
        optixu::NativeBlockBuffer2D<GBuffer1Elements> GBuffer1[2];

        ROBuffer<MaterialData> materialDataBuffer;
        ROBuffer<InstanceData> instanceDataBufferArray[2];
        ROBuffer<GeometryInstanceData> geometryInstanceDataBuffer;
        LightDistribution lightInstDist;
        RegularConstantContinuousDistribution2D envLightImportanceMap;
        CUtexObject envLightTexture;

        AABB* sceneAABB;

        uint32_t maxNumTrainingSuffixes;
        uint32_t* numTrainingData[2];
        uint2* tileSize[2];
        RGBAsOrderedInt* targetMinMax[2];
        RGB* targetAvg[2];
        uint32_t* offsetToSelectUnbiasedTile;
        uint32_t* offsetToSelectTrainingPath;
        RWBuffer<RadianceQuery> inferenceRadianceQueryBuffer; // image size + #(training suffix)
        RWBuffer<TerminalInfo> inferenceTerminalInfoBuffer; // image size
        RWBuffer<RGB> inferredRadianceBuffer; // image size + #(training suffix)
        RWBuffer<RGB> perFrameContributionBuffer; // image size
        RWBuffer<RadianceQuery> trainRadianceQueryBuffer[2]; // #(training vertex)
        RWBuffer<RGB> trainTargetBuffer[2]; // #(training vertex)
        RWBuffer<TrainingVertexInfo> trainVertexInfoBuffer; // #(training vertex)
        RWBuffer<TrainingSuffixTerminalInfo> trainSuffixTerminalInfoBuffer; // #(training suffix)
        RWBuffer<LinearCongruentialGenerator> dataShufflerBuffer; // numTrainingDataPerFrame

        optixu::NativeBlockBuffer2D<float4> beautyAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> albedoAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> normalAccumBuffer;

        PickInfo* pickInfos[2];
    };

    struct PerFramePipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        uint32_t numAccumFrames;
        uint32_t frameIndex;

        PerspectiveCamera camera;
        PerspectiveCamera prevCamera;

        float radianceScale;

        float envLightPowerCoeff;
        float envLightRotation;

        int2 mousePosition;

        uint32_t maxPathLength : 4;
        uint32_t bufferIndex : 1;
        uint32_t resetFlowBuffer : 1;
        uint32_t enableJittering : 1;
        uint32_t enableEnvLight : 1;
        uint32_t enableBumpMapping : 1;
        uint32_t enableDebugPrint : 1;

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
        Flow,
        RenderingPathLength,
        DirectlyVisualizedPrediction,
        DenoisedBeauty,
    };



    using PrimaryRayPayloadSignature =
        optixu::PayloadSignature<shared::HitPointParams*, shared::PickInfo*>;
    template <bool useNRC>
    using PathTraceRayPayloadSignature =
        optixu::PayloadSignature<shared::PathTraceWriteOnlyPayload*, shared::PathTraceReadWritePayload<useNRC>*>;
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
    shared::LightSample* lightSample, float* areaPDensity)
{
    using namespace shared;
    CUtexObject texEmittance = 0;
    RGB emittance(0.0f, 0.0f, 0.0f);
    Point2D texCoord;
    if (sampleEnvLight) {
        float u, v;
        float uvPDF;
        plp.s->envLightImportanceMap.sample(u0, u1, &u, &v, &uvPDF);
        const float phi = 2 * pi_v<float> * u;
        const float theta = pi_v<float> * v;

        float posPhi = phi - plp.f->envLightRotation;
        posPhi = posPhi - floorf(posPhi / (2 * pi_v<float>)) * 2 * pi_v<float>;

        const Vector3D direction = fromPolarYUp(posPhi, theta);
        const Point3D position(direction.x, direction.y, direction.z);
        lightSample->position = position;
        lightSample->atInfinity = true;

        lightSample->normal = Normal3D(-position);

        // JP: テクスチャー空間中のPDFを面積に関するものに変換する。
        // EN: convert the PDF in texture space to one with respect to area.
        // The true value is: lim_{l to inf} uvPDF / (2 * Pi * Pi * sin(theta)) / l^2
        const float sinTheta = std::sin(theta);
        if (sinTheta == 0.0f) {
            *areaPDensity = 0.0f;
            return;
        }
        *areaPDensity = uvPDF / (2 * pi_v<float> * pi_v<float> * sinTheta);

        texEmittance = plp.s->envLightTexture;
        // JP: 環境マップテクスチャーの値に係数をかけて、通常の光源と同じように返り値を光束発散度
        //     として扱えるようにする。
        // EN: Multiply a coefficient to make the return value possible to be handled as luminous emittance.
        emittance = RGB(pi_v<float> * plp.f->envLightPowerCoeff);
        texCoord.x = u;
        texCoord.y = v;
    }
    else {
        float lightProb = 1.0f;

        // JP: まずはインスタンスをサンプルする。
        // EN: First, sample an instance.
        float instProb;
        float uGeomInst;
        const uint32_t instSlot = plp.s->lightInstDist.sample(ul, &instProb, &uGeomInst);
        lightProb *= instProb;
        const InstanceData &inst = plp.s->instanceDataBufferArray[plp.f->bufferIndex][instSlot];
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
        const uint32_t geomInstIndexInInst = inst.lightGeomInstDist.sample(uGeomInst, &geomInstProb, &uPrim);
        const uint32_t geomInstSlot = inst.geomInstSlots[geomInstIndexInInst];
        lightProb *= geomInstProb;
        const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[geomInstSlot];
        if (geomInstProb == 0.0f) {
            *areaPDensity = 0.0f;
            return;
        }
        //Assert(geomInst.emitterPrimDist.integral() > 0.0f,
        //       "Non-emissive geom inst %u, prob %g, u: %g.", geomInstIndex, geomInstProb, uGeomInst);

        // JP: 最後に、サンプルしたジオメトリインスタンスに属するプリミティブをサンプルする。
        // EN: Finally, sample a primitive which belongs to the sampled geometry instance.
        float primProb;
        const uint32_t primIndex = geomInst.emitterPrimDist.sample(uPrim, &primProb);
        lightProb *= primProb;

        //printf("%u-%u-%u: %g\n", instIndex, geomInstIndex, primIndex, lightProb);

        const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

        const shared::Triangle &tri = geomInst.triangleBuffer[primIndex];
        const shared::Vertex &vA = geomInst.vertexBuffer[tri.index0];
        const shared::Vertex &vB = geomInst.vertexBuffer[tri.index1];
        const shared::Vertex &vC = geomInst.vertexBuffer[tri.index2];
        const Point3D pA = inst.transform * vA.position;
        const Point3D pB = inst.transform * vB.position;
        const Point3D pC = inst.transform * vC.position;

        Normal3D geomNormal(cross(pB - pA, pC - pA));

        float bcA, bcB, bcC;
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
                const Vector3D A = normalize(pA - shadingPoint);
                const Vector3D B = normalize(pB - shadingPoint);
                const Vector3D C = normalize(pC - shadingPoint);
                const Vector3D cAB = normalize(cross(A, B));
                const Vector3D cBC = normalize(cross(B, C));
                const Vector3D cCA = normalize(cross(C, A));
                //float cos_a = dot(B, C);
                //float cos_b = dot(C, A);
                const float cos_c = dot(A, B);
                const float cosAlpha = -dot(cAB, cCA);
                const float cosBeta = -dot(cBC, cAB);
                const float cosGamma = -dot(cCA, cBC);
                const float alpha = std::acos(cosAlpha);
                const float sinAlpha = std::sqrt(1 - pow2(cosAlpha));
                const float sphArea = alpha + std::acos(cosBeta) + std::acos(cosGamma) - pi_v<float>;

                const float sphAreaHat = sphArea * u0;
                const float s = std::sin(sphAreaHat - alpha);
                const float t = std::cos(sphAreaHat - alpha);
                const float uu = t - cosAlpha;
                const float vv = s + sinAlpha * cos_c;
                const float q = ((vv * t - uu * s) * cosAlpha - vv) / ((vv * s + uu * t) * sinAlpha);

                const Vector3D cHat = q * A + std::sqrt(1 - pow2(q)) * project(C, A);
                const float z = 1 - u1 * (1 - dot(cHat, B));
                const Vector3D P = z * B + std::sqrt(1 - pow2(z)) * project(cHat, B);

                const auto restoreBarycentrics = [&geomNormal]
                (const Point3D &org, const Vector3D &dir,
                 const Point3D &pA, const Point3D &pB, const Point3D &pC,
                 float* dist, float* bcB, float* bcC) {
                    const Vector3D eAB = pB - pA;
                    const Vector3D eAC = pC - pA;
                    const Vector3D pVec = cross(dir, eAC);
                    const float recDet = 1.0f / dot(eAB, pVec);
                    const Vector3D tVec = org - pA;
                    *bcB = dot(tVec, pVec) * recDet;
                    const Vector3D qVec = cross(tVec, eAB);
                    *bcC = dot(dir, qVec) * recDet;
                    *dist = dot(eAC, qVec) * recDet;
                };
                dir = P;
                restoreBarycentrics(shadingPoint, dir, pA, pB, pC, &dist, &bcB, &bcC);
                bcA = 1 - (bcB + bcC);
                dirPDF = 1 / sphArea;
            }

            geomNormal = normalize(geomNormal);
            const float lpCos = -dot(dir, geomNormal);
            if (lpCos > 0 && stc::isfinite(dirPDF))
                *areaPDensity = lightProb * (dirPDF * lpCos / pow2(dist));
            else
                *areaPDensity = 0.0f;
        }
        else {
            // Uniform sampling on unit triangle
            // A Low-Distortion Map Between Triangle and Square
            bcA = 0.5f * u0;
            bcB = 0.5f * u1;
            const float offset = bcB - bcA;
            if (offset > 0)
                bcB += offset;
            else
                bcA -= offset;
            bcC = 1 - (bcA + bcB);

            const float recArea = 2.0f / length(geomNormal);
            *areaPDensity = lightProb * recArea;
        }
        lightSample->position = bcA * pA + bcB * pB + bcC * pC;
        lightSample->atInfinity = false;
        lightSample->normal = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
        lightSample->normal = normalize(inst.normalMatrix * lightSample->normal);

        if (mat.emittance) {
            texEmittance = mat.emittance;
            emittance = RGB(1.0f, 1.0f, 1.0f);
            texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;
        }
    }

    if (texEmittance) {
        const float4 texValue = tex2DLod<float4>(texEmittance, texCoord.x, texCoord.y, 0.0f);
        emittance *= RGB(getXYZ(texValue));
    }
    lightSample->emittance = emittance;
}

template <typename RayType, bool withVisibility>
CUDA_DEVICE_FUNCTION CUDA_INLINE RGB performDirectLighting(
    const Point3D &shadingPoint, const Vector3D &vOutLocal, const ReferenceFrame &shadingFrame,
    const BSDF &bsdf, const shared::LightSample &lightSample)
{
    using namespace shared;
    Vector3D shadowRayDir = lightSample.atInfinity ?
        Vector3D(lightSample.position) :
        (lightSample.position - shadingPoint);
    const float dist2 = shadowRayDir.sqLength();
    float dist = std::sqrt(dist2);
    shadowRayDir /= dist;
    const Vector3D shadowRayDirLocal = shadingFrame.toLocal(shadowRayDir);

    const float lpCos = dot(-shadowRayDir, lightSample.normal);
    const float spCos = shadowRayDirLocal.z;

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
        const RGB Le = lightSample.emittance / pi_v<float>; // assume diffuse emitter.
        const RGB fsValue = bsdf.evaluate(vOutLocal, shadowRayDirLocal);
        const float G = lpCos * std::fabs(spCos) / dist2;
        const RGB ret = fsValue * Le * G;
        return ret;
    }
    else {
        return RGB(0.0f, 0.0f, 0.0f);
    }
}

template <typename RayType>
CUDA_DEVICE_FUNCTION CUDA_INLINE bool evaluateVisibility(
    const Point3D &shadingPoint, const shared::LightSample &lightSample)
{
    using namespace shared;
    Vector3D shadowRayDir = lightSample.atInfinity ?
        Vector3D(lightSample.position) :
        (lightSample.position - shadingPoint);
    const float dist2 = shadowRayDir.sqLength();
    float dist = std::sqrt(dist2);
    shadowRayDir /= dist;
    if (lightSample.atInfinity)
        dist = 1e+10f;

    float visibility = 1.0f;
    VisibilityRayPayloadSignature::trace(
        plp.f->travHandle,
        shadingPoint.toNative(), shadowRayDir.toNative(), 0.0f, dist * 0.9999f, 0.0f,
        0xFF, OPTIX_RAY_FLAG_NONE,
        RayType::Visibility, maxNumRayTypes, RayType::Visibility,
        visibility);

    return visibility > 0.0f;
}

template <bool computeHypotheticalAreaPDensity, bool useSolidAngleSampling>
CUDA_DEVICE_FUNCTION CUDA_INLINE void computeSurfacePoint(
    const shared::InstanceData &inst,
    const shared::GeometryInstanceData &geomInst,
    uint32_t primIndex, float bcB, float bcC,
    const Point3D &referencePoint,
    Point3D* positionInWorld, Normal3D* shadingNormalInWorld, Vector3D* texCoord0DirInWorld,
    Normal3D* geometricNormalInWorld, Point2D* texCoord,
    float* hypAreaPDensity)
{
    using namespace shared;
    const Triangle &tri = geomInst.triangleBuffer[primIndex];
    const Vertex &vA = geomInst.vertexBuffer[tri.index0];
    const Vertex &vB = geomInst.vertexBuffer[tri.index1];
    const Vertex &vC = geomInst.vertexBuffer[tri.index2];
    const Point3D pA = transformPointFromObjectToWorldSpace(vA.position);
    const Point3D pB = transformPointFromObjectToWorldSpace(vB.position);
    const Point3D pC = transformPointFromObjectToWorldSpace(vC.position);
    const float bcA = 1 - (bcB + bcC);

    // JP: ヒットポイントのローカル座標中の各値を計算する。
    // EN: Compute hit point properties in the local coordinates.
    *positionInWorld = bcA * pA + bcB * pB + bcC * pC;
    const Normal3D shadingNormalInObj = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
    const Vector3D texCoord0DirInObj = bcA * vA.texCoord0Dir + bcB * vB.texCoord0Dir + bcC * vC.texCoord0Dir;
    *texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;

    *geometricNormalInWorld = Normal3D(cross(pB - pA, pC - pA));
    float area;
    if constexpr (computeHypotheticalAreaPDensity && !useSolidAngleSampling) {
        area = 0.5f * length(*geometricNormalInWorld);
        *geometricNormalInWorld = *geometricNormalInWorld / (2 * area);
    }
    else {
        *geometricNormalInWorld = normalize(*geometricNormalInWorld);
        (void)area;
    }

    // JP: ローカル座標中の値をワールド座標中の値へと変換する。
    // EN: Convert the local properties to ones in world coordinates.
    *shadingNormalInWorld = normalize(transformNormalFromObjectToWorldSpace(shadingNormalInObj));
    *texCoord0DirInWorld = normalize(transformVectorFromObjectToWorldSpace(texCoord0DirInObj));
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
        const float instImportance = inst.lightGeomInstDist.integral();
        lightProb *= (pow2(inst.uniformScale) * instImportance) / plp.s->lightInstDist.integral();
        lightProb *= geomInst.emitterPrimDist.integral() / instImportance;
        if (!stc::isfinite(lightProb)) {
            *hypAreaPDensity = 0.0f;
            return;
        }
        lightProb *= geomInst.emitterPrimDist.evaluatePMF(primIndex);
        if constexpr (useSolidAngleSampling) {
            // TODO: ? compute in the local coordinates.
            const Vector3D A = normalize(pA - referencePoint);
            const Vector3D B = normalize(pB - referencePoint);
            const Vector3D C = normalize(pC - referencePoint);
            const Vector3D cAB = normalize(cross(A, B));
            const Vector3D cBC = normalize(cross(B, C));
            const Vector3D cCA = normalize(cross(C, A));
            const float cosAlpha = -dot(cAB, cCA);
            const float cosBeta = -dot(cBC, cAB);
            const float cosGamma = -dot(cCA, cBC);
            const float sphArea = std::acos(cosAlpha) + std::acos(cosBeta) + std::acos(cosGamma) - pi_v<float>;
            const float dirPDF = 1.0f / sphArea;
            Vector3D refDir = referencePoint - *positionInWorld;
            const float dist2ToRefPoint = sqLength(refDir);
            refDir /= std::sqrt(dist2ToRefPoint);
            const float lpCos = dot(refDir, *geometricNormalInWorld);
            if (lpCos > 0 && stc::isfinite(dirPDF))
                *hypAreaPDensity = lightProb * (dirPDF * lpCos / dist2ToRefPoint);
            else
                *hypAreaPDensity = 0.0f;
        }
        else {
            *hypAreaPDensity = lightProb / area;
        }
        Assert(stc::isfinite(*hypAreaPDensity), "hypP: %g, area: %g", *hypAreaPDensity, area);
    }
    else {
        (void)*hypAreaPDensity;
    }
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void computeSurfacePoint(
    const shared::InstanceData &inst,
    const shared::GeometryInstanceData &geomInst,
    uint32_t primIndex, float bcB, float bcC,
    Point3D* positionInWorld, Normal3D* shadingNormalInWorld, Vector3D* texCoord0DirInWorld,
    Normal3D* geometricNormalInWorld, Point2D* texCoord)
{
    using namespace shared;
    const Triangle &tri = geomInst.triangleBuffer[primIndex];
    const Vertex &vA = geomInst.vertexBuffer[tri.index0];
    const Vertex &vB = geomInst.vertexBuffer[tri.index1];
    const Vertex &vC = geomInst.vertexBuffer[tri.index2];
    const float bcA = 1 - (bcB + bcC);

    // JP: ヒットポイントのローカル座標中の各値を計算する。
    // EN: Compute hit point properties in the local coordinates.
    const Point3D positionInObj = bcA * vA.position + bcB * vB.position + bcC * vC.position;
    *positionInWorld = inst.transform * positionInObj;
    *geometricNormalInWorld = normalize(
        inst.normalMatrix * Normal3D(cross(vB.position - vA.position, vC.position - vA.position)));
    const Normal3D shadingNormalInObj = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
    const Vector3D texCoord0DirInObj = bcA * vA.texCoord0Dir + bcB * vB.texCoord0Dir + bcC * vC.texCoord0Dir;
    *texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;

    // JP: ローカル座標中の値をワールド座標中の値へと変換する。
    // EN: Convert the local properties to ones in world coordinates.
    *shadingNormalInWorld = normalize(inst.normalMatrix * shadingNormalInObj);
    *texCoord0DirInWorld = inst.transform * texCoord0DirInObj;
    *texCoord0DirInWorld = normalize(
        *texCoord0DirInWorld - dot(*shadingNormalInWorld, *texCoord0DirInWorld) * *shadingNormalInWorld);
    if (!shadingNormalInWorld->allFinite()) {
        *geometricNormalInWorld = Normal3D(0, 0, 1);
        *shadingNormalInWorld = Normal3D(0, 0, 1);
        *texCoord0DirInWorld = Vector3D(1, 0, 0);
    }
    if (!texCoord0DirInWorld->allFinite()) {
        Vector3D bitangent;
        makeCoordinateSystem(*shadingNormalInWorld, texCoord0DirInWorld, &bitangent);
    }
}



struct HitPointParameter {
    float bcB, bcC;
    int32_t primIndex;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static HitPointParameter get() {
        HitPointParameter ret;
        const float2 bc = optixGetTriangleBarycentrics();
        ret.bcB = bc.x;
        ret.bcC = bc.y;
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

#if !defined(PURE_CUDA) || defined(CUDAU_CODE_COMPLETION)

CUDA_DEVICE_FUNCTION bool isCursorPixel() {
    return plp.f->mousePosition == make_int2(optixGetLaunchIndex());
}

#endif

CUDA_DEVICE_FUNCTION bool getDebugPrintEnabled() {
    return plp.f->enableDebugPrint;
}

#endif // #if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
