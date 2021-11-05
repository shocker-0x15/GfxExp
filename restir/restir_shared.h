#pragma once

#include "../common/common_shared.h"

// JP: Callable Programや関数ポインターによる動的な関数呼び出しを
//     無くした場合の性能を見たい場合にこのマクロを有効化する。
// EN: Enable this switch when you want to see performance
//     without dynamic function calls by callable programs or function pointers.
//#define USE_HARD_CODED_BSDF_FUNCTIONS
#define HARD_CODED_BSDF DiffuseAndSpecularBRDF

enum CallableProgram {
    CallableProgram_ReadModifiedNormalFromNormalMap = 0,
    CallableProgram_ReadModifiedNormalFromNormalMap2ch,
    CallableProgram_ReadModifiedNormalFromHeightMap,
    CallableProgram_SetupLambertBRDF,
    CallableProgram_LambertBRDF_sampleThroughput,
    CallableProgram_LambertBRDF_evaluate,
    CallableProgram_LambertBRDF_evaluatePDF,
    CallableProgram_LambertBRDF_evaluateDHReflectanceEstimate,
    CallableProgram_SetupDiffuseAndSpecularBRDF,
    CallableProgram_SetupSimplePBR_BRDF,
    CallableProgram_DiffuseAndSpecularBRDF_sampleThroughput,
    CallableProgram_DiffuseAndSpecularBRDF_evaluate,
    CallableProgram_DiffuseAndSpecularBRDF_evaluatePDF,
    CallableProgram_DiffuseAndSpecularBRDF_evaluateDHReflectanceEstimate,
    NumCallablePrograms
};

#if (defined(__CUDA_ARCH__) && defined(PURE_CUDA)) || defined(OPTIXU_Platform_CodeCompletion)
CUDA_CONSTANT_MEM void* c_callableToPointerMap[NumCallablePrograms];
#endif

constexpr const char* callableProgramEntryPoints[] = {
    RT_DC_NAME_STR("readModifiedNormalFromNormalMap"),
    RT_DC_NAME_STR("readModifiedNormalFromNormalMap2ch"),
    RT_DC_NAME_STR("readModifiedNormalFromHeightMap"),
    RT_DC_NAME_STR("setupLambertBRDF"),
    RT_DC_NAME_STR("LambertBRDF_sampleThroughput"),
    RT_DC_NAME_STR("LambertBRDF_evaluate"),
    RT_DC_NAME_STR("LambertBRDF_evaluatePDF"),
    RT_DC_NAME_STR("LambertBRDF_evaluateDHReflectanceEstimate"),
    RT_DC_NAME_STR("setupDiffuseAndSpecularBRDF"),
    RT_DC_NAME_STR("setupSimplePBR_BRDF"),
    RT_DC_NAME_STR("DiffuseAndSpecularBRDF_sampleThroughput"),
    RT_DC_NAME_STR("DiffuseAndSpecularBRDF_evaluate"),
    RT_DC_NAME_STR("DiffuseAndSpecularBRDF_evaluatePDF"),
    RT_DC_NAME_STR("DiffuseAndSpecularBRDF_evaluateDHReflectanceEstimate"),
};

constexpr const char* callableProgramPointerNames[] = {
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("readModifiedNormalFromNormalMap"),
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("readModifiedNormalFromNormalMap2ch"),
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("readModifiedNormalFromHeightMap"),
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("setupLambertBRDF"),
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("LambertBRDF_sampleThroughput"),
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("LambertBRDF_evaluate"),
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("LambertBRDF_evaluatePDF"),
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("LambertBRDF_evaluateDHReflectanceEstimate"),
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("setupDiffuseAndSpecularBRDF"),
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("setupSimplePBR_BRDF"),
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("DiffuseAndSpecularBRDF_sampleThroughput"),
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("DiffuseAndSpecularBRDF_evaluate"),
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("DiffuseAndSpecularBRDF_evaluatePDF"),
    CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR("DiffuseAndSpecularBRDF_evaluateDHReflectanceEstimate"),
};

namespace shared {
    static constexpr float probToSampleEnvLight = 0.25f;

    static constexpr uint32_t numLightSubsets = 128;
    static constexpr uint32_t lightSubsetSize = 1024;
    static constexpr int tileSizeX = 8;
    static constexpr int tileSizeY = 8;



    enum RayType {
        RayType_Primary = 0,
        RayType_Visibility,
        NumRayTypes
    };



    struct Vertex {
        float3 position;
        float3 normal;
        float3 texCoord0Dir;
        float2 texCoord;
    };

    struct Triangle {
        uint32_t index0, index1, index2;
    };



    struct PerspectiveCamera {
        float aspect;
        float fovY;
        float3 position;
        Matrix3x3 orientation;

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION float2 calcScreenPosition(const float3 &posInWorld) const {
            Matrix3x3 invOri = inverse(orientation);
            float3 posInView = invOri * (posInWorld - position);
            float2 posAtZ1 = make_float2(posInView.x / posInView.z, posInView.y / posInView.z);
            float h = 2 * std::tan(fovY / 2);
            float w = aspect * h;
            return make_float2(1 - (posAtZ1.x + 0.5f * w) / w,
                               1 - (posAtZ1.y + 0.5f * h) / h);
        }
#endif
    };



    struct MaterialData;

    using SetupBSDFBody = DynamicFunction<
        void(const MaterialData &matData, const float2 &texCoord, uint32_t* bodyData)>;

    struct MaterialData {
        union {
            struct {
                CUtexObject reflectance;
            } asLambert;
            struct {
                CUtexObject diffuse;
                CUtexObject specular;
                CUtexObject smoothness;
            } asDiffuseAndSpecular;
            struct {
                CUtexObject baseColor_opacity;
                CUtexObject occlusion_roughness_metallic;
            } asSimplePBR;
        };
        CUtexObject normal;
        CUtexObject emittance;
        union {
            struct {
                unsigned int normalWidth : 16;
                unsigned int normalHeight : 16;
            };
            uint32_t normalDimension;
        };

        ReadModifiedNormal readModifiedNormal;

        SetupBSDFBody setupBSDFBody;
        BSDFSampleThroughput bsdfSampleThroughput; // Not used in this sample
        BSDFEvaluate bsdfEvaluate;
        BSDFEvaluatePDF bsdfEvaluatePDF; // Not used in this sample
        BSDFEvaluateDHReflectanceEstimate bsdfEvaluateDHReflectanceEstimate;
    };

    struct GeometryInstanceData {
        const Vertex* vertexBuffer;
        const Triangle* triangleBuffer;
        DiscreteDistribution1D emitterPrimDist;
        uint32_t materialSlot;
        uint32_t geomInstSlot;
    };

    struct InstanceData {
        Matrix4x4 transform;
        Matrix4x4 prevTransform;
        Matrix3x3 normalMatrix;

        const uint32_t* geomInstSlots;
        uint32_t numGeomInsts;
        DiscreteDistribution1D lightGeomInstDist;
    };



    struct HitPointParams {
        float3 albedo;
        float3 positionInWorld;
        float3 prevPositionInWorld;
        float3 normalInWorld;
        float2 texCoord;
        uint32_t materialSlot;
    };



    CUDA_DEVICE_FUNCTION float convertToWeight(const float3 &color) {
        //return sRGB_calcLuminance(color);
        return (color.x + color.y + color.z) / 3;
    }



    struct LightSample {
        float3 emittance;
        float3 position;
        float3 normal;
        unsigned int atInfinity : 1;
    };

    struct PreSampledLight {
        LightSample sample;
        float areaPDensity;
    };

    using WeightSum = float;
    //using WeightSum = FloatSum;

    template <typename SampleType>
    class Reservoir {
        SampleType m_sample;
        WeightSum m_sumWeights;
        uint32_t m_streamLength;

    public:
        CUDA_DEVICE_FUNCTION void initialize() {
            m_sumWeights = 0;
            m_streamLength = 0;
        }
        CUDA_DEVICE_FUNCTION bool update(const SampleType &newSample, float weight, float u) {
            m_sumWeights += weight;
            bool accepted = u < weight / m_sumWeights;
            if (accepted)
                m_sample = newSample;
            ++m_streamLength;
            return accepted;
        }

        CUDA_DEVICE_FUNCTION LightSample getSample() const {
            return m_sample;
        }
        CUDA_DEVICE_FUNCTION float getSumWeights() const {
            return m_sumWeights;
        }
        CUDA_DEVICE_FUNCTION uint32_t getStreamLength() const {
            return m_streamLength;
        }
        CUDA_DEVICE_FUNCTION void setStreamLength(uint32_t length) {
            m_streamLength = length;
        }
    };

    struct ReservoirInfo {
        float recPDFEstimate;
        float targetDensity;
    };

    union SampleVisibility {
        uint32_t asUInt;
        struct {
            unsigned int newSample : 1;
            unsigned int newSampleOnTemporal : 1;
            unsigned int newSampleOnSpatiotemporal : 1;
            unsigned int hasValidTemporalSample : 1;
            unsigned int temporalSample : 1;
            unsigned int temporalSampleOnCurrent : 1;
            unsigned int temporalSampleOnSpatiotemporal : 1;
            unsigned int hasValidSpatiotemporalSample : 1;
            unsigned int spatiotemporalSample : 1;
            unsigned int spatiotemporalSampleOnCurrent : 1;
            unsigned int spatiotemporalSampleOnTemporal : 1;
            unsigned int selectedSample : 1;
        };

        CUDA_DEVICE_FUNCTION SampleVisibility() : asUInt(0) {}
    };



    struct PickInfo {
        uint32_t instSlot;
        uint32_t geomInstSlot;
        uint32_t primIndex;
        uint32_t matSlot;
        float3 positionInWorld;
        float3 normalInWorld;
        float3 albedo;
        float3 emittance;
        unsigned int hit : 1;
    };



    struct GBuffer0 {
        float3 positionInWorld;
        float texCoord_x;
    };

    struct GBuffer1 {
        float3 normalInWorld;
        float texCoord_y;
    };

    struct GBuffer2 {
        float2 motionVector;
        uint32_t materialSlot;
    };

    
    
    struct StaticPipelineLaunchParameters {
        int2 imageSize;
        int2 numTiles;

        // only for rearchitected ver.
        PCG32RNG* lightPreSamplingRngs;
        PreSampledLight* preSampledLights;

        optixu::NativeBlockBuffer2D<PCG32RNG> rngBuffer;

        optixu::NativeBlockBuffer2D<GBuffer0> GBuffer0[2];
        optixu::NativeBlockBuffer2D<GBuffer1> GBuffer1[2];
        optixu::NativeBlockBuffer2D<GBuffer2> GBuffer2[2];

        optixu::BlockBuffer2D<Reservoir<LightSample>, 1> reservoirBuffer[2];
        optixu::NativeBlockBuffer2D<ReservoirInfo> reservoirInfoBuffer[2];
        optixu::NativeBlockBuffer2D<SampleVisibility> sampleVisibilityBuffer[2];
        const float2* spatialNeighborDeltas; // only for rearchitected ver.

        const MaterialData* materialDataBuffer;
        const GeometryInstanceData* geometryInstanceDataBuffer;
        DiscreteDistribution1D lightInstDist;
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

        const InstanceData* instanceDataBuffer;

        PerspectiveCamera camera;
        PerspectiveCamera prevCamera;

        float envLightPowerCoeff;
        float envLightRotation;

        float spatialNeighborRadius;
        float radiusThresholdForSpatialVisReuse; // only for rearchitected ver.

        int2 mousePosition;
        PickInfo* pickInfo;

        unsigned int log2NumCandidateSamples : 4;
        unsigned int numSpatialNeighbors : 4;
        unsigned int useLowDiscrepancyNeighbors : 1;
        unsigned int reuseVisibility : 1;
        unsigned int reuseVisibilityForTemporal : 1; // only for rearchitected ver.
        unsigned int reuseVisibilityForSpatiotemporal : 1; // only for rearchitected ver.
        unsigned int enableTemporalReuse : 1;
        unsigned int enableSpatialReuse : 1;
        unsigned int useUnbiasedEstimator : 1;
        unsigned int bufferIndex : 1;
        unsigned int resetFlowBuffer : 1;
        unsigned int enableJittering : 1;
        unsigned int enableEnvLight : 1;
        unsigned int enableBumpMapping : 1;

        uint32_t debugSwitches;
        void setDebugSwitch(int32_t idx, bool b) {
            debugSwitches &= ~(1 << idx);
            debugSwitches |= b << idx;
        }
        CUDA_DEVICE_FUNCTION bool getDebugSwitch(int32_t idx) const {
            return (debugSwitches >> idx) & 0b1;
        }
    };
    
    struct PipelineLaunchParameters {
        StaticPipelineLaunchParameters* s;
        PerFramePipelineLaunchParameters* f;
        unsigned int currentReservoirIndex : 1;
        unsigned int spatialNeighborBaseIndex : 10;
    };



    enum class BufferToDisplay {
        NoisyBeauty = 0,
        Albedo,
        Normal,
        Flow,
        DenoisedBeauty,
    };
}

#define PrimaryRayPayloadSignature shared::HitPointParams*, shared::PickInfo*
#define VisibilityRayPayloadSignature float



#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

#include "../common/common_device.cuh"

#if defined(PURE_CUDA)
CUDA_CONSTANT_MEM shared::PipelineLaunchParameters plp;
#else
RT_PIPELINE_LAUNCH_PARAMETERS shared::PipelineLaunchParameters plp;
#endif

namespace shared {
    template <typename BSDFType>
    CUDA_DEVICE_FUNCTION void setupBSDFBody(const MaterialData &matData, const float2 &texCoord, uint32_t* bodyData);

    template<>
    CUDA_DEVICE_FUNCTION void setupBSDFBody<LambertBRDF>(
        const MaterialData &matData, const float2 &texCoord, uint32_t* bodyData) {
        float4 reflectance = tex2DLod<float4>(matData.asLambert.reflectance, texCoord.x, texCoord.y, 0.0f);
        auto &bsdfBody = *reinterpret_cast<LambertBRDF*>(bodyData);
        bsdfBody = LambertBRDF(make_float3(reflectance.x, reflectance.y, reflectance.z));
    }

    template<>
    CUDA_DEVICE_FUNCTION void setupBSDFBody<DiffuseAndSpecularBRDF>(
        const MaterialData &matData, const float2 &texCoord, uint32_t* bodyData) {
        float4 diffuseColor = tex2DLod<float4>(matData.asDiffuseAndSpecular.diffuse, texCoord.x, texCoord.y, 0.0f);
        float4 specularF0Color = tex2DLod<float4>(matData.asDiffuseAndSpecular.specular, texCoord.x, texCoord.y, 0.0f);
        float smoothness = tex2DLod<float>(matData.asDiffuseAndSpecular.smoothness, texCoord.x, texCoord.y, 0.0f);
        auto &bsdfBody = *reinterpret_cast<DiffuseAndSpecularBRDF*>(bodyData);
        bsdfBody = DiffuseAndSpecularBRDF(make_float3(diffuseColor),
                                          make_float3(specularF0Color),
                                          min(smoothness, 0.999f));
    }

    template<>
    CUDA_DEVICE_FUNCTION void setupBSDFBody<SimplePBR_BRDF>(
        const MaterialData &matData, const float2 &texCoord, uint32_t* bodyData) {
        float4 baseColor_opacity = tex2DLod<float4>(matData.asSimplePBR.baseColor_opacity, texCoord.x, texCoord.y, 0.0f);
        float4 occlusion_roughness_metallic = tex2DLod<float4>(matData.asSimplePBR.occlusion_roughness_metallic, texCoord.x, texCoord.y, 0.0f);
        float3 baseColor = make_float3(baseColor_opacity);
        float smoothness = min(1.0f - occlusion_roughness_metallic.y, 0.999f);
        float metallic = occlusion_roughness_metallic.z;
        auto &bsdfBody = *reinterpret_cast<SimplePBR_BRDF*>(bodyData);
        bsdfBody = SimplePBR_BRDF(baseColor, 0.5f, smoothness, metallic);
    }



    struct BSDF {
#if defined(USE_HARD_CODED_BSDF_FUNCTIONS)
        uint32_t m_data[sizeof(DiffuseAndSpecularBRDF) / 4];
#else
        static constexpr uint32_t NumDwords = 16;
        BSDFSampleThroughput m_sampleThroughput;
        BSDFEvaluate m_evaluate;
        BSDFEvaluatePDF m_evaluatePDF;
        BSDFEvaluateDHReflectanceEstimate m_evaluateDHReflectanceEstimate;
        uint32_t m_data[NumDwords];
#endif

        CUDA_DEVICE_FUNCTION void setup(const MaterialData &matData, const float2 &texCoord) {
#if defined(USE_HARD_CODED_BSDF_FUNCTIONS)
            setupBSDFBody<HARD_CODED_BSDF>(matData, texCoord, m_data);
#else
            m_sampleThroughput = matData.bsdfSampleThroughput;
            m_evaluate = matData.bsdfEvaluate;
            m_evaluatePDF = matData.bsdfEvaluatePDF;
            m_evaluateDHReflectanceEstimate = matData.bsdfEvaluateDHReflectanceEstimate;
            matData.setupBSDFBody(matData, texCoord, m_data);
#endif
        }
        CUDA_DEVICE_FUNCTION float3 sampleThroughput(const float3 &vGiven, float uDir0, float uDir1,
                                                     float3* vSampled, float* dirPDensity) const {
#if defined(USE_HARD_CODED_BSDF_FUNCTIONS)
            auto &bsdf = *reinterpret_cast<const HARD_CODED_BSDF*>(m_data);
            return bsdf.sampleThroughput(vGiven, uDir0, uDir1, vSampled, dirPDensity);
#else
            return m_sampleThroughput(m_data, vGiven, uDir0, uDir1, vSampled, dirPDensity);
#endif
        }
        CUDA_DEVICE_FUNCTION float3 evaluate(const float3 &vGiven, const float3 &vSampled) const {
#if defined(USE_HARD_CODED_BSDF_FUNCTIONS)
            auto &bsdf = *reinterpret_cast<const HARD_CODED_BSDF*>(m_data);
            return bsdf.evaluate(vGiven, vSampled);
#else
            return m_evaluate(m_data, vGiven, vSampled);
#endif
        }
        CUDA_DEVICE_FUNCTION float evaluatePDF(const float3 &vGiven, const float3 &vSampled) const {
#if defined(USE_HARD_CODED_BSDF_FUNCTIONS)
            auto &bsdf = *reinterpret_cast<const HARD_CODED_BSDF*>(m_data);
            return bsdf.evaluatePDF(vGiven, vSampled);
#else
            return m_evaluatePDF(m_data, vGiven, vSampled);
#endif
        }
        CUDA_DEVICE_FUNCTION float3 evaluateDHReflectanceEstimate(const float3 &vGiven) const {
#if defined(USE_HARD_CODED_BSDF_FUNCTIONS)
            auto &bsdf = *reinterpret_cast<const HARD_CODED_BSDF*>(m_data);
            return bsdf.evaluateDHReflectanceEstimate(vGiven);
#else
            return m_evaluateDHReflectanceEstimate(m_data, vGiven);
#endif
        }
    };



#define DEFINE_BSDF_CALLABLES(BSDFType) \
    RT_CALLABLE_PROGRAM void RT_DC_NAME(setup ## BSDFType)(\
        const MaterialData &matData, const float2 &texCoord, uint32_t* bodyData) {\
        setupBSDFBody<BSDFType>(matData, texCoord, bodyData);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(setup ## BSDFType)

    DEFINE_BSDF_CALLABLES(LambertBRDF);
    DEFINE_BSDF_CALLABLES(DiffuseAndSpecularBRDF);
    DEFINE_BSDF_CALLABLES(SimplePBR_BRDF);

#undef DEFINE_BSDF_CALLABLES



    CUDA_DEVICE_FUNCTION void sampleLight(
        float ul, bool sampleEnvLight, float u0, float u1,
        LightSample* lightSample, float* areaPDensity) {
        CUtexObject texEmittance = 0;
        float3 emittance = make_float3(0.0f, 0.0f, 0.0f);
        float2 texCoord;
        if (sampleEnvLight) {
            float u, v;
            float uvPDF;
            plp.s->envLightImportanceMap.sample(u0, u1, &u, &v, &uvPDF);
            float phi = 2 * Pi * u;
            float theta = Pi * v;

            float posPhi = phi - plp.f->envLightRotation;
            posPhi = posPhi - floorf(posPhi / (2 * Pi)) * 2 * Pi;

            float3 direction = fromPolarYUp(posPhi, theta);
            float3 position = make_float3(direction.x, direction.y, direction.z);
            lightSample->position = position;
            lightSample->atInfinity = true;

            lightSample->normal = -position;

            // JP: テクスチャー空間中のPDFを面積に関するものに変換する。
            // EN: convert the PDF in texture space to one with respect to area.
            // The true value is: lim_{l to inf} uvPDF / (2 * Pi * Pi * std::sin(theta)) / l^2
            *areaPDensity = uvPDF / (2 * Pi * Pi * std::sin(theta));

            texEmittance = plp.s->envLightTexture;
            // JP: 環境マップテクスチャーの値に係数をかけて、通常の光源と同じように返り値を光束発散度
            //     として扱えるようにする。
            // EN: Multiply a coefficient to make the return value possible to be handled as luminous emittance.
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

            // JP: 次にサンプルしたインスタンスに属するジオメトリインスタンスをサンプルする。
            // EN: Next, sample a geometry instance which belongs to the sampled instance.
            float geomInstProb;
            float uPrim;
            uint32_t geomInstIndexInInst = inst.lightGeomInstDist.sample(uGeomInst, &geomInstProb, &uPrim);
            uint32_t geomInstIndex = inst.geomInstSlots[geomInstIndexInInst];
            lightProb *= geomInstProb;
            const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[geomInstIndex];

            // JP: 最後に、サンプルしたジオメトリインスタンスに属するプリミティブをサンプルする。
            // EN: Finally, sample a primitive which belongs to the sampled geometry instance.
            float primProb;
            uint32_t primIndex = geomInst.emitterPrimDist.sample(uPrim, &primProb);
            lightProb *= primProb;

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

            //printf("%u-%u-%u: %g\n", instIndex, geomInstIndex, primIndex, lightProb);

            const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

            const shared::Triangle &tri = geomInst.triangleBuffer[primIndex];
            const shared::Vertex(&v)[3] = {
                geomInst.vertexBuffer[tri.index0],
                geomInst.vertexBuffer[tri.index1],
                geomInst.vertexBuffer[tri.index2]
            };
            float3 p[3] = {
                inst.transform * v[0].position,
                inst.transform * v[1].position,
                inst.transform * v[2].position,
            };

            float3 geomNormal = cross(p[1] - p[0], p[2] - p[0]);
            lightSample->position = t0 * p[0] + t1 * p[1] + t2 * p[2];
            lightSample->atInfinity = false;
            float recArea = 1.0f / length(geomNormal);
            //lightSample->normal = geomNormal * recArea;
            lightSample->normal = t0 * v[0].normal + t1 * v[1].normal + t2 * v[2].normal;
            lightSample->normal = normalize(inst.normalMatrix * lightSample->normal);
            recArea *= 2;
            *areaPDensity = lightProb * recArea;

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
        lightSample->emittance = emittance;
    }

    template <bool withVisibility>
    CUDA_DEVICE_FUNCTION float3 performDirectLighting(
        const float3 &shadingPoint, const float3 &vOutLocal, const ReferenceFrame &shadingFrame, const BSDF &bsdf,
        const LightSample &lightSample) {
        float3 shadowRayDir = lightSample.atInfinity ?
            lightSample.position :
            (lightSample.position - shadingPoint);
        float dist2 = sqLength(shadowRayDir);
        float dist = std::sqrt(dist2);
        shadowRayDir /= dist;
        float3 shadowRayDirLocal = shadingFrame.toLocal(shadowRayDir);

        float lpCos = dot(-shadowRayDir, lightSample.normal);
        float spCos = shadowRayDirLocal.z;

        float visibility = 1.0f;
        if constexpr (withVisibility) {
            if (lightSample.atInfinity)
                dist = 1e+10f;
            optixu::trace<VisibilityRayPayloadSignature>(
                plp.f->travHandle,
                shadingPoint, shadowRayDir, 0.0f, dist * 0.9999f, 0.0f,
                0xFF, OPTIX_RAY_FLAG_NONE,
                RayType_Visibility, NumRayTypes, RayType_Visibility,
                visibility);
        }

        if (visibility > 0 && lpCos > 0) {
            float3 Le = lightSample.emittance / Pi; // assume diffuse emitter.
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
        const float3 &shadingPoint, const LightSample &lightSample) {
        float3 shadowRayDir = lightSample.atInfinity ?
            lightSample.position :
            (lightSample.position - shadingPoint);
        float dist2 = sqLength(shadowRayDir);
        float dist = std::sqrt(dist2);
        shadowRayDir /= dist;
        if (lightSample.atInfinity)
            dist = 1e+10f;

        float visibility = 1.0f;
        optixu::trace<VisibilityRayPayloadSignature>(
            plp.f->travHandle,
            shadingPoint, shadowRayDir, 0.0f, dist * 0.9999f, 0.0f,
            0xFF, OPTIX_RAY_FLAG_NONE,
            RayType_Visibility, NumRayTypes, RayType_Visibility,
            visibility);

        return visibility > 0.0f;
    }
}

#endif
