#pragma once

#include "../common/common.h"

struct BSDF;

namespace Shared {
    static constexpr float Pi = 3.14159265358979323846f;
    static constexpr float RayEpsilon = 1e-4;



    template <typename RealType>
    struct CompensatedSum {
        RealType result;
        RealType comp;

        CUDA_DEVICE_FUNCTION CompensatedSum(const RealType &value = RealType(0)) : result(value), comp(0.0) { };

        CUDA_DEVICE_FUNCTION CompensatedSum &operator=(const RealType &value) {
            result = value;
            comp = 0;
            return *this;
        }

        CUDA_DEVICE_FUNCTION CompensatedSum &operator+=(const RealType &value) {
            RealType cInput = value - comp;
            RealType sumTemp = result + cInput;
            comp = (sumTemp - result) - cInput;
            result = sumTemp;
            return *this;
        }

        CUDA_DEVICE_FUNCTION operator RealType() const { return result; };
    };

    //using FloatSum = float;
    using FloatSum = CompensatedSum<float>;



    class PCG32RNG {
        uint64_t state;

    public:
        CUDA_DEVICE_FUNCTION PCG32RNG() {}

        void setState(uint64_t _state) { state = _state; }

        CUDA_DEVICE_FUNCTION uint32_t operator()() {
            uint64_t oldstate = state;
            // Advance internal state
            state = oldstate * 6364136223846793005ULL + 1;
            // Calculate output function (XSH RR), uses old state for max ILP
            uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-static_cast<int32_t>(rot)) & 31));
        }

        CUDA_DEVICE_FUNCTION float getFloat0cTo1o() {
            uint32_t fractionBits = ((*this)() >> 9) | 0x3f800000;
            return *(float*)&fractionBits - 1.0f;
        }
    };



    template <typename RealType>
    class DiscreteDistribution1DTemplate {
        const RealType* m_PMF;
        const RealType* m_CDF;
        RealType m_integral;
        uint32_t m_numValues;

    public:
        DiscreteDistribution1DTemplate(const RealType* PMF, const RealType* CDF, RealType integral, uint32_t numValues) :
            m_PMF(PMF), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {}

        CUDA_DEVICE_FUNCTION DiscreteDistribution1DTemplate() {}

        CUDA_DEVICE_FUNCTION uint32_t sample(RealType u, RealType* prob) const {
            Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
            int idx = 0;
            for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
                if (idx + d >= m_numValues)
                    continue;
                if (m_CDF[idx + d] <= u)
                    idx += d;
            }
            Assert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
            *prob = m_PMF[idx];
            return idx;
        }

        CUDA_DEVICE_FUNCTION uint32_t sample(RealType u, RealType* prob, RealType* remapped) const {
            Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
            int idx = 0;
            for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
                if (idx + d >= m_numValues)
                    continue;
                if (m_CDF[idx + d] <= u)
                    idx += d;
            }
            Assert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
            *prob = m_PMF[idx];
            *remapped = (u - m_CDF[idx]) / (m_CDF[idx + 1] - m_CDF[idx]);
            Assert(isfinite(*remapped), "Remapped value is indefinite %g.", *remapped);
            return idx;
        }

        CUDA_DEVICE_FUNCTION RealType evaluatePMF(uint32_t idx) const {
            Assert(idx >= 0 && idx < m_numValues, "\"idx\" is out of range [0, %u)", m_numValues);
            return m_PMF[idx];
        }

        CUDA_DEVICE_FUNCTION RealType integral() const { return m_integral; }

        CUDA_DEVICE_FUNCTION uint32_t numValues() const { return m_numValues; }
    };

    using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;



    enum RayType {
        RayType_Primary = 0,
        RayType_Visibility,
        NumRayTypes
    };



    struct Vertex {
        float3 position;
        float3 normal;
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

    using SetupBSDF = optixu::DirectCallableProgramID<void(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf)>;
    using GetBaseColor = optixu::DirectCallableProgramID<float3(const uint32_t* data, const float3 &vout)>;
    using EvaluateBSDF = optixu::DirectCallableProgramID<float3(const uint32_t* data, const float3 &vin, const float3 &vout)>;

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
        };
        CUtexObject emittance;

        SetupBSDF setupBSDF;
        GetBaseColor getBaseColor;
        EvaluateBSDF evaluateBSDF;
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
    };

    template <typename SampleType>
    class Reservoir {
        SampleType m_sample;
        FloatSum m_sumWeights;
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

#define PrimaryRayPayloadSignature Shared::HitPointParams*, Shared::PickInfo*
#define VisibilityRayPayloadSignature float
