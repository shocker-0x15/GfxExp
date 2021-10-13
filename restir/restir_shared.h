#pragma once

#include "../common/common_shared.h"

struct BSDF;

namespace shared {
    static constexpr float probToSampleEnvLight = 0.25f;



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

    using SetupBSDF = optixu::DirectCallableProgramID<
        void(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf)>;
    using BSDFSampleThroughput = optixu::DirectCallableProgramID<
        float3(const uint32_t* data, const float3 &vGiven, float uDir0, float uDir1,
               float3* vSampled, float* dirPDensity)>;
    using BSDFEvaluate = optixu::DirectCallableProgramID<
        float3(const uint32_t* data, const float3 &vGiven, const float3 &vSampled)>;
    using BSDFEvaluatePDF = optixu::DirectCallableProgramID<
        float(const uint32_t* data, const float3 &vGiven, const float3 &vSampled)>;
    using BSDFEvaluateDHReflectanceEstimate = optixu::DirectCallableProgramID<
        float3(const uint32_t* data, const float3 &vGiven)>;

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

        SetupBSDF setupBSDF;
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
        uint32_t instIndex;
        uint32_t geomInstIndex;
        uint32_t primIndex;
        float b1;
        float b2;

        CUDA_DEVICE_FUNCTION bool atInfinity() const {
            return instIndex == 0xFFFFFFFF;
        }
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
        optixu::NativeBlockBuffer2D<PCG32RNG> rngBuffer;

        optixu::NativeBlockBuffer2D<GBuffer0> GBuffer0[2];
        optixu::NativeBlockBuffer2D<GBuffer1> GBuffer1[2];
        optixu::NativeBlockBuffer2D<GBuffer2> GBuffer2[2];

        optixu::BlockBuffer2D<Reservoir<LightSample>, 1> reservoirBuffer[2];
        optixu::NativeBlockBuffer2D<ReservoirInfo> reservoirInfoBuffer[2];
        const float2* spatialNeighborDeltas;

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

        const InstanceData* instanceDataBuffer;

        PerspectiveCamera camera;
        PerspectiveCamera prevCamera;

        float envLightPowerCoeff;
        float envLightRotation;

        float spatialNeighborRadius;

        int2 mousePosition;
        PickInfo* pickInfo;

        unsigned int log2NumCandidateSamples : 4;
        unsigned int numSpatialNeighbors : 4;
        unsigned int useLowDiscrepancyNeighbors : 1;
        unsigned int reuseVisibility : 1;
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
