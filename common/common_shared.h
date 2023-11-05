#pragma once

#include "basic_types.h"

// JP: Callable Programや関数ポインターによる動的な関数呼び出しを
//     無くした場合の性能を見たい場合にこのマクロを有効化する。
// EN: Enable this switch when you want to see performance
//     without dynamic function calls by callable programs or function pointers.
//#define USE_HARD_CODED_BSDF_FUNCTIONS
#define HARD_CODED_BSDF DiffuseAndSpecularBRDF
//#define HARD_CODED_BSDF SimplePBR_BRDF
//#define HARD_CODED_BSDF LambertBRDF

#define USE_PROBABILITY_TEXTURE 0

// Use Walker's alias method with initialization by Vose's algorithm
//#define USE_WALKER_ALIAS_METHOD

#define PROCESS_DYNAMIC_FUNCTIONS \
    PROCESS_DYNAMIC_FUNCTION(readModifiedNormalFromNormalMap), \
    PROCESS_DYNAMIC_FUNCTION(readModifiedNormalFromNormalMap2ch), \
    PROCESS_DYNAMIC_FUNCTION(readModifiedNormalFromHeightMap), \
    PROCESS_DYNAMIC_FUNCTION(setupLambertBRDF), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_getSurfaceParameters), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_sampleThroughput), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluate), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluatePDF), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluateDHReflectanceEstimate), \
    PROCESS_DYNAMIC_FUNCTION(setupDiffuseAndSpecularBRDF), \
    PROCESS_DYNAMIC_FUNCTION(setupSimplePBR_BRDF), \
    PROCESS_DYNAMIC_FUNCTION(DiffuseAndSpecularBRDF_getSurfaceParameters), \
    PROCESS_DYNAMIC_FUNCTION(DiffuseAndSpecularBRDF_sampleThroughput), \
    PROCESS_DYNAMIC_FUNCTION(DiffuseAndSpecularBRDF_evaluate), \
    PROCESS_DYNAMIC_FUNCTION(DiffuseAndSpecularBRDF_evaluatePDF), \
    PROCESS_DYNAMIC_FUNCTION(DiffuseAndSpecularBRDF_evaluateDHReflectanceEstimate),

enum CallableProgram {
#define PROCESS_DYNAMIC_FUNCTION(Func) CallableProgram_ ## Func
    PROCESS_DYNAMIC_FUNCTIONS
#undef PROCESS_DYNAMIC_FUNCTION
    NumCallablePrograms
};

constexpr const char* callableProgramEntryPoints[] = {
#define PROCESS_DYNAMIC_FUNCTION(Func) RT_DC_NAME_STR(#Func)
    PROCESS_DYNAMIC_FUNCTIONS
#undef PROCESS_DYNAMIC_FUNCTION
};

#define CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR(name) "ptr_" #name
constexpr const char* callableProgramPointerNames[] = {
#define PROCESS_DYNAMIC_FUNCTION(Func) CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR(Func)
    PROCESS_DYNAMIC_FUNCTIONS
#undef PROCESS_DYNAMIC_FUNCTION
};
#undef CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR

#undef PROCESS_DYNAMIC_FUNCTIONS

#if (defined(__CUDA_ARCH__) && defined(PURE_CUDA)) || defined(OPTIXU_Platform_CodeCompletion)
CUDA_CONSTANT_MEM void* c_callableToPointerMap[NumCallablePrograms];
#endif

#if defined(PURE_CUDA)
#   define CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(name) \
        extern "C" CUDA_DEVICE_MEM auto ptr_ ## name = RT_DC_NAME(name)
#else
#   define CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(name)
#endif



namespace shared {
    template <typename FuncType>
    class DynamicFunction;

    template <typename ReturnType, typename... ArgTypes>
    class DynamicFunction<ReturnType(ArgTypes...)> {
        using Signature = ReturnType (*)(ArgTypes...);
        optixu::DirectCallableProgramID<ReturnType(ArgTypes...)> m_callableHandle;

    public:
        CUDA_COMMON_FUNCTION DynamicFunction() {}
        CUDA_COMMON_FUNCTION DynamicFunction(uint32_t sbtIndex) : m_callableHandle(sbtIndex) {}

        CUDA_COMMON_FUNCTION explicit operator uint32_t() const { return static_cast<uint32_t>(m_callableHandle); }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION ReturnType operator()(const ArgTypes &... args) const {
#   if defined(PURE_CUDA)
            void* ptr = c_callableToPointerMap[static_cast<uint32_t>(m_callableHandle)];
            auto func = reinterpret_cast<Signature>(ptr);
            return func(args...);
#   else
            return m_callableHandle(args...);
#   endif
        }
#endif
    };



    static constexpr bool enableBufferOobCheck = true;
    template <typename T>
    using ROBuffer = cudau::ROBufferTemplate<T, enableBufferOobCheck>;
    template <typename T>
    using RWBuffer = cudau::RWBufferTemplate<T, enableBufferOobCheck>;



    class PCG32RNG {
        uint64_t state;

    public:
        CUDA_COMMON_FUNCTION PCG32RNG() {}

        CUDA_COMMON_FUNCTION void setState(uint64_t _state) { state = _state; }

        CUDA_COMMON_FUNCTION uint32_t operator()() {
            uint64_t oldstate = state;
            // Advance internal state
            state = oldstate * 6364136223846793005ULL + 1;
            // Calculate output function (XSH RR), uses old state for max ILP
            uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-static_cast<int32_t>(rot)) & 31));
        }

        CUDA_COMMON_FUNCTION float getFloat0cTo1o() {
            uint32_t fractionBits = ((*this)() >> 9) | 0x3f800000;
            return *(float*)&fractionBits - 1.0f;
        }
    };



    CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t mapPrimarySampleToDiscrete(
        float u01, uint32_t numValues, float* uRemapped = nullptr) {
#if defined(__CUDA_ARCH__)
        uint32_t idx = min(static_cast<uint32_t>(u01 * numValues), numValues - 1);
#else
        uint32_t idx = std::min(static_cast<uint32_t>(u01 * numValues), numValues - 1);
#endif
        if (uRemapped)
            *uRemapped = u01 * numValues - idx;
        return idx;
    }



    template <typename RealType>
    struct AliasTableEntry {
        uint32_t secondIndex;
        RealType probToPickFirst;

        CUDA_COMMON_FUNCTION AliasTableEntry() {}
        CUDA_COMMON_FUNCTION AliasTableEntry(uint32_t _secondIndex, RealType _probToPickFirst) :
            secondIndex(_secondIndex), probToPickFirst(_probToPickFirst) {}
    };

    template <typename RealType>
    struct AliasValueMap {
        RealType scaleForFirst;
        RealType scaleForSecond;
        RealType offsetForSecond;
    };



    template <typename RealType>
    class DiscreteDistribution1DTemplate {
        RealType* m_weights;
#if defined(USE_WALKER_ALIAS_METHOD)
        const AliasTableEntry<RealType>* m_aliasTable;
        const AliasValueMap<RealType>* m_valueMaps;
#else
        RealType* m_CDF;
#endif
        RealType m_integral;
        uint32_t m_numValues;

    public:
#if defined(USE_WALKER_ALIAS_METHOD)
        DiscreteDistribution1DTemplate(
            RealType* weights, AliasTableEntry<RealType>* aliasTable, AliasValueMap<RealType>* valueMaps,
            RealType integral, uint32_t numValues) :
            m_weights(weights), m_aliasTable(aliasTable), m_valueMaps(valueMaps),
            m_integral(integral), m_numValues(numValues) {}
#else
        DiscreteDistribution1DTemplate(
            RealType* weights, RealType* CDF, RealType integral, uint32_t numValues) :
            m_weights(weights), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {}
#endif

        CUDA_COMMON_FUNCTION DiscreteDistribution1DTemplate() {}

        CUDA_COMMON_FUNCTION uint32_t sample(RealType u, RealType* prob, RealType* remapped = nullptr) const {
            Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
#if defined(USE_WALKER_ALIAS_METHOD)
            uint32_t idx = mapPrimarySampleToDiscrete(u, m_numValues, &u);
            const AliasTableEntry<RealType> &entry = m_aliasTable[idx];
            const AliasValueMap<RealType> &valueMap = m_valueMaps[idx];
            if (u < entry.probToPickFirst) {
                if (remapped)
                    *remapped = valueMap.scaleForFirst * u;
            }
            else {
                idx = entry.secondIndex;
                if (remapped)
                    *remapped = valueMap.scaleForSecond * u + valueMap.offsetForSecond;
            }
#else
            u *= m_integral;
            int idx = 0;
            for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
                if (idx + d >= m_numValues)
                    continue;
                if (m_CDF[idx + d] <= u)
                    idx += d;
            }
            Assert(idx < m_numValues, "Invalid Index!: %u >= %u, u: %g, integ: %g",
                   idx, m_numValues, u, m_integral);
            if (remapped) {
                RealType lCDF = m_CDF[idx];
                RealType rCDF = m_integral;
                if (idx < m_numValues - 1)
                    rCDF = m_CDF[idx + 1];
                *remapped = (u - lCDF) / (rCDF - lCDF);
                Assert(isfinite(*remapped), "Remapped value is not a finite value %g.",
                       *remapped);
            }
#endif
            *prob = m_weights[idx] / m_integral;
            return idx;
        }

        CUDA_COMMON_FUNCTION RealType evaluatePMF(uint32_t idx) const {
            if (!m_weights || m_integral == 0.0f)
                return 0.0f;
            Assert(idx < m_numValues, "\"idx\" is out of range [0, %u)", m_numValues);
            return m_weights[idx] / m_integral;
        }

        CUDA_COMMON_FUNCTION RealType integral() const { return m_integral; }

        CUDA_COMMON_FUNCTION uint32_t numValues() const { return m_numValues; }

        CUDA_COMMON_FUNCTION void setNumValues(uint32_t numValues) {
            m_numValues = numValues;
        }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION void setWeightAt(uint32_t index, RealType value) const {
            m_weights[index] = value;
        }

#   if !defined(USE_WALKER_ALIAS_METHOD)
        CUDA_DEVICE_FUNCTION void finalize() {
            uint32_t lastIndex = m_numValues - 1;
            m_integral = m_CDF[lastIndex] + m_weights[lastIndex];
        }
#   endif
#endif
    };

    using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;



    template <typename RealType>
    class RegularConstantContinuousDistribution1DTemplate {
        const RealType* m_PDF;
#if defined(USE_WALKER_ALIAS_METHOD)
        const AliasTableEntry<RealType>* m_aliasTable;
        const AliasValueMap<RealType>* m_valueMaps;
#else
        const RealType* m_CDF;
#endif
        RealType m_integral;
        uint32_t m_numValues;

    public:
#if defined(USE_WALKER_ALIAS_METHOD)
        RegularConstantContinuousDistribution1DTemplate(
            const RealType* PDF, const AliasTableEntry<RealType>* aliasTable, const AliasValueMap<RealType>* valueMaps,
            RealType integral, uint32_t numValues) :
            m_PDF(PDF), m_aliasTable(aliasTable), m_valueMaps(valueMaps),
            m_integral(integral), m_numValues(numValues) {}
#else
        RegularConstantContinuousDistribution1DTemplate(
            const RealType* PDF, const RealType* CDF, RealType integral, uint32_t numValues) :
            m_PDF(PDF), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {}
#endif

        CUDA_COMMON_FUNCTION RegularConstantContinuousDistribution1DTemplate() {}

        CUDA_COMMON_FUNCTION RealType sample(RealType u, RealType* probDensity) const {
            Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
#if defined(USE_WALKER_ALIAS_METHOD)
            uint32_t idx = mapPrimarySampleToDiscrete(u, m_numValues, &u);
            const AliasTableEntry<RealType> &entry = m_aliasTable[idx];
            const AliasValueMap<RealType> &valueMap = m_valueMaps[idx];
            RealType t;
            if (u < entry.probToPickFirst) {
                t = valueMap.scaleForFirst * u;
            }
            else {
                idx = entry.secondIndex;
                t = valueMap.scaleForSecond * u + valueMap.offsetForSecond;
            }
#else
            int idx = 0;
            for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
                if (idx + d >= m_numValues)
                    continue;
                if (m_CDF[idx + d] <= u)
                    idx += d;
            }
            Assert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
            RealType t = (u - m_CDF[idx]) / (m_CDF[idx + 1] - m_CDF[idx]);
#endif
            *probDensity = m_PDF[idx];
            return (idx + t) / m_numValues;
        }
        CUDA_COMMON_FUNCTION RealType evaluatePDF(RealType smp) const {
            Assert(smp >= 0 && smp < 1.0, "\"smp\": %g is out of range [0, 1).", smp);
            int32_t idx = min(m_numValues - 1, static_cast<uint32_t>(smp * m_numValues));
            return m_PDF[idx];
        }
        CUDA_COMMON_FUNCTION RealType integral() const { return m_integral; }

        CUDA_COMMON_FUNCTION uint32_t numValues() const { return m_numValues; }
    };

    using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float>;



    template <typename RealType>
    class RegularConstantContinuousDistribution2DTemplate {
        const RegularConstantContinuousDistribution1DTemplate<RealType>* m_1DDists;
        RegularConstantContinuousDistribution1DTemplate<RealType> m_top1DDist;

    public:
        RegularConstantContinuousDistribution2DTemplate(
            const RegularConstantContinuousDistribution1DTemplate<RealType>* _1DDists,
            const RegularConstantContinuousDistribution1DTemplate<RealType> &top1DDist) :
            m_1DDists(_1DDists), m_top1DDist(top1DDist) {}

        CUDA_COMMON_FUNCTION RegularConstantContinuousDistribution2DTemplate() {}

        CUDA_COMMON_FUNCTION void sample(
            RealType u0, RealType u1, RealType* d0, RealType* d1, RealType* probDensity) const {
            RealType topPDF;
            *d1 = m_top1DDist.sample(u1, &topPDF);
            uint32_t idx1D = mapPrimarySampleToDiscrete(*d1, m_top1DDist.numValues());
            *d0 = m_1DDists[idx1D].sample(u0, probDensity);
            *probDensity *= topPDF;
        }
        CUDA_COMMON_FUNCTION RealType evaluatePDF(RealType d0, RealType d1) const {
            uint32_t idx1D = mapPrimarySampleToDiscrete(d1, m_top1DDist.numValues());
            return m_top1DDist.evaluatePDF(d1) * m_1DDists[idx1D].evaluatePDF(d0);
        }
    };

    using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float>;



    CUDA_COMMON_FUNCTION CUDA_INLINE uint2 computeProbabilityTextureDimentions(uint32_t maxNumElems) {
#if !defined(__CUDA_ARCH__)
        using std::max;
#endif
        uint2 dims = make_uint2(max(nextPowerOf2(maxNumElems), 2u), 1u);
        while ((dims.x != dims.y) && (dims.x != 2 * dims.y)) {
            dims.x /= 2;
            dims.y *= 2;
        }
        return dims;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE uint2 compute2DFrom1D(const uint2 &dims, uint32_t index1D) {
        return make_uint2(index1D % dims.x, index1D / dims.x);
    }

    class ProbabilityTexture {
        CUtexObject m_cuTexObj;
        unsigned int m_maxDimX : 16;
        unsigned int m_maxDimY : 16;
        unsigned int m_dimX : 16;
        unsigned int m_dimY : 16;
        float m_integral;

    public:
        CUDA_COMMON_FUNCTION void setTexObject(CUtexObject texObj, uint2 maxDims) {
            m_cuTexObj = texObj;
            m_maxDimX = maxDims.x;
            m_maxDimY = maxDims.y;
        }

        CUDA_COMMON_FUNCTION void setDimensions(const uint2 &dims) {
            m_dimX = dims.x;
            m_dimY = dims.y;
        }

        CUDA_COMMON_FUNCTION uint2 getDimensions() const {
            return make_uint2(m_dimX, m_dimY);
        }

        CUDA_COMMON_FUNCTION uint32_t calcNumMipLevels() const {
            return nextPowOf2Exponent(m_dimX) + 1;
        }
        CUDA_COMMON_FUNCTION uint32_t calcMaxNumMipLevels() const {
            return nextPowOf2Exponent(m_maxDimX) + 1;
        }

        CUDA_COMMON_FUNCTION uint2 compute2DFrom1D(uint32_t index1D) const {
            return make_uint2(index1D % m_dimX, index1D / m_dimX);
        }
        CUDA_COMMON_FUNCTION uint32_t compute1DFrom2D(const uint2 &index2D) const {
            return index2D.y * m_dimX + index2D.x;
        }

        CUDA_COMMON_FUNCTION float integral() const {
            if (m_cuTexObj == 0 || m_integral == 0.0f)
                return 0.0f;
            return m_integral;
        }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION uint32_t sample(float u, float* prob, float* remapped = nullptr) const {
            Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
            uint2 index2D = make_uint2(0, 0);
            uint32_t numMipLevels = calcNumMipLevels();
            *prob = 1;
            Vector2D recCurActualDims;
            {
                uint2 curActualDims = make_uint2(2, m_maxDimX > m_maxDimY ? 1 : 2);
                curActualDims <<= calcMaxNumMipLevels() - numMipLevels;
                recCurActualDims = Vector2D(1.0f / curActualDims.x, 1.0f / curActualDims.y);
            }
            uint2 curDims = make_uint2(2, m_dimX > m_dimY ? 1 : 2);
            for (uint32_t mipLevel = numMipLevels - 2; mipLevel != UINT32_MAX; --mipLevel) {
                index2D = 2 * index2D;
                Vector2D tc(index2D.x + 0.5f, index2D.y + 0.5f);
                Vector2D ll = tc + Vector2D(0, 1);
                Vector2D lr = tc + Vector2D(1, 1);
                Vector2D ur = tc + Vector2D(1, 0);
                Vector2D ul = tc + Vector2D(0, 0);
                Vector2D nll = ll * recCurActualDims;
                Vector2D nlr = lr * recCurActualDims;
                Vector2D nur = ur * recCurActualDims;
                Vector2D nul = ul * recCurActualDims;
                Vector4D neighbors;
                neighbors.x = ll.y < curDims.y ?
                    tex2DLod<float>(m_cuTexObj, nll.x, nll.y, mipLevel) : 0.0f;
                neighbors.y = (lr.x < curDims.x && lr.y < curDims.y) ?
                    tex2DLod<float>(m_cuTexObj, nlr.x, nlr.y, mipLevel) : 0.0f;
                neighbors.z = ur.x < curDims.x ?
                    tex2DLod<float>(m_cuTexObj, nur.x, nur.y, mipLevel) : 0.0f;
                neighbors.w = tex2DLod<float>(m_cuTexObj, nul.x, nul.y, mipLevel);
                float sumProbs = neighbors.x + neighbors.y + neighbors.z + neighbors.w;
                u *= sumProbs;
                float accProb = 0;
                float stepProb;
                if ((accProb + neighbors.x) > u) {
                    stepProb = neighbors.x;
                    index2D.y += 1;
                }
                else {
                    accProb += neighbors.x;
                    if ((accProb + neighbors.y) > u) {
                        stepProb = neighbors.y;
                        u -= accProb;
                        index2D.x += 1;
                        index2D.y += 1;
                    }
                    else {
                        accProb += neighbors.y;
                        if ((accProb + neighbors.z) > u) {
                            stepProb = neighbors.z;
                            u -= accProb;
                            index2D.x += 1;
                        }
                        else {
                            accProb += neighbors.z;
                            stepProb = neighbors.w;
                            u -= accProb;
                        }
                    }
                }
                *prob *= stepProb / sumProbs;
                u /= stepProb;
                recCurActualDims /= 2.0f;
                curDims *= 2;
            }
            if (remapped)
                *remapped = u;
            return compute1DFrom2D(index2D);
        }

        CUDA_DEVICE_FUNCTION void setIntegral(float v) {
            m_integral = v;
        }
#endif
    };

    using LightDistribution =
#if USE_PROBABILITY_TEXTURE
        ProbabilityTexture;
#else
        DiscreteDistribution1D;
#endif



    // Reference:
    // Long-Period Hash Functions for Procedural Texturing
    // combined permutation table of the hash function of period 739,024 = lcm(11, 13, 16, 17, 19)
#if defined(__CUDA_ARCH__)
    CUDA_CONSTANT_MEM
#endif
    static uint8_t PermutationTable[] = {
        // table 0: 11 numbers
        0, 10, 2, 7, 3, 5, 6, 4, 8, 1, 9,
        // table 1: 13 numbers
        5, 11, 6, 8, 1, 10, 12, 9, 3, 7, 0, 4, 2,
        // table 2: 16 numbers = the range of the hash function required by Perlin noise.
        13, 10, 11, 5, 6, 9, 4, 3, 8, 7, 14, 2, 0, 1, 15, 12,
        // table 3: 17 numbers
        1, 13, 5, 14, 12, 3, 6, 16, 0, 8, 9, 2, 11, 4, 15, 7, 10,
        // table 4: 19 numbers
        10, 6, 5, 8, 15, 0, 17, 7, 14, 18, 13, 16, 2, 9, 12, 1, 11, 4, 3,
        //// table 6: 23 numbers
        //20, 21, 4, 5, 0, 18, 14, 2, 6, 22, 10, 17, 3, 7, 8, 16, 19, 11, 9, 13, 1, 15, 12
    };

    // References
    // Improving Noise
    // This code is based on the web site: adrian's soapbox
    // http://flafla2.github.io/2014/08/09/perlinnoise.html
    class PerlinNoise3D {
        int32_t m_repeat;

        CUDA_COMMON_FUNCTION CUDA_INLINE static uint8_t hash(int32_t x, int32_t y, int32_t z) {
            uint32_t sum = 0;
            sum += PermutationTable[0 + (PermutationTable[0 + (PermutationTable[0 + x % 11] + y) % 11] + z) % 11];
            sum += PermutationTable[11 + (PermutationTable[11 + (PermutationTable[11 + x % 13] + y) % 13] + z) % 13];
            sum += PermutationTable[24 + (PermutationTable[24 + (PermutationTable[24 + x % 16] + y) % 16] + z) % 16];
            sum += PermutationTable[40 + (PermutationTable[40 + (PermutationTable[40 + x % 17] + y) % 17] + z) % 17];
            sum += PermutationTable[57 + (PermutationTable[57 + (PermutationTable[57 + x % 19] + y) % 19] + z) % 19];
            return sum % 16;
        }

        CUDA_COMMON_FUNCTION CUDA_INLINE static float gradient(
            uint32_t hash, float xu, float yu, float zu) {
            switch (hash & 0xF) {
                // Dot products with 12 vectors defined by the directions from the center of a cube to its edges.
            case 0x0: return  xu + yu; // ( 1,  1,  0)
            case 0x1: return -xu + yu; // (-1,  1,  0)
            case 0x2: return  xu - yu; // ( 1, -1,  0)
            case 0x3: return -xu - yu; // (-1, -1,  0)
            case 0x4: return  xu + zu; // ( 1,  0,  1)
            case 0x5: return -xu + zu; // (-1,  0,  1)
            case 0x6: return  xu - zu; // ( 1,  0, -1)
            case 0x7: return -xu - zu; // (-1,  0, -1)
            case 0x8: return  yu + zu; // ( 0,  1,  1)
            case 0x9: return -yu + zu; // ( 0, -1,  1)
            case 0xA: return  yu - zu; // ( 0,  1, -1)
            case 0xB: return -yu - zu; // ( 0, -1, -1)

                // To avoid the cost of dividing by 12, we pad to 16 gradient directions.
                // These form a regular tetrahedron, so adding them redundantly introduces no visual bias in the texture.
            case 0xC: return  xu + yu; // ( 1,  1,  0)
            case 0xD: return -yu + zu; // ( 0, -1,  1)
            case 0xE: return -xu + yu; // (-1 , 1,  0)
            case 0xF: return -yu - zu; // ( 0, -1, -1)

            default: return 0; // never happens
            }
        }

    public:
        CUDA_COMMON_FUNCTION PerlinNoise3D(int32_t repeat) : m_repeat(repeat) {}

        CUDA_COMMON_FUNCTION float evaluate(const Point3D &p, float frequency) const {
            float x = frequency * p.x;
            float y = frequency * p.y;
            float z = frequency * p.z;
            const uint32_t repeat = static_cast<uint32_t>(m_repeat * frequency);

            // If we have any repeat on, change the coordinates to their "local" repetitions.
            if (repeat > 0) {
#if defined(__CUDA_ARCH__)
                x = fmodf(x, repeat);
                y = fmodf(y, repeat);
                z = fmodf(z, repeat);
#else
                x = std::fmod(x, static_cast<float>(repeat));
                y = std::fmod(y, static_cast<float>(repeat));
                z = std::fmod(z, static_cast<float>(repeat));
#endif
                if (x < 0)
                    x += repeat;
                if (y < 0)
                    y += repeat;
                if (z < 0)
                    z += repeat;
            }

            // Calculate the "unit cube" that the point asked will be located in.
            // The left bound is ( |_x_|,|_y_|,|_z_| ) and the right bound is that plus 1.
#if defined(__CUDA_ARCH__)
            int32_t xi = floorf(x);
            int32_t yi = floorf(y);
            int32_t zi = floorf(z);
#else
            int32_t xi = static_cast<int32_t>(std::floor(x));
            int32_t yi = static_cast<int32_t>(std::floor(y));
            int32_t zi = static_cast<int32_t>(std::floor(z));
#endif

            const auto fade = [](float t) {
                // Fade function as defined by Ken Perlin.
                // This eases coordinate values so that they will "ease" towards integral values.
                // This ends up smoothing the final output.
                // 6t^5 - 15t^4 + 10t^3
                return t * t * t * (t * (t * 6 - 15) + 10);
            };

            // Next we calculate the location (from 0.0 to 1.0) in that cube.
            // We also fade the location to smooth the result.
            float xu = x - xi;
            float yu = y - yi;
            float zu = z - zi;
            float u = fade(xu);
            float v = fade(yu);
            float w = fade(zu);

            const auto inc = [this, repeat](int32_t num) {
                ++num;
                if (repeat > 0)
                    num %= repeat;
                return num;
            };

            uint8_t lll, llu, lul, luu, ull, ulu, uul, uuu;
            lll = hash(xi, yi, zi);
            ull = hash(inc(xi), yi, zi);
            lul = hash(xi, inc(yi), zi);
            uul = hash(inc(xi), inc(yi), zi);
            llu = hash(xi, yi, inc(zi));
            ulu = hash(inc(xi), yi, inc(zi));
            luu = hash(xi, inc(yi), inc(zi));
            uuu = hash(inc(xi), inc(yi), inc(zi));

            const auto lerp = [](float v0, float v1, float t) {
                return v0 * (1 - t) + v1 * t;
            };

            // The gradient function calculates the dot product between a pseudorandom gradient vector and 
            // the vector from the input coordinate to the 8 surrounding points in its unit cube.
            // This is all then lerped together as a sort of weighted average based on the faded (u,v,w) values we made earlier.
            float _llValue = lerp(gradient(lll, xu, yu, zu), gradient(ull, xu - 1, yu, zu), u);
            float _ulValue = lerp(gradient(lul, xu, yu - 1, zu), gradient(uul, xu - 1, yu - 1, zu), u);
            float __lValue = lerp(_llValue, _ulValue, v);

            float _luValue = lerp(gradient(llu, xu, yu, zu - 1), gradient(ulu, xu - 1, yu, zu - 1), u);
            float _uuValue = lerp(gradient(luu, xu, yu - 1, zu - 1), gradient(uuu, xu - 1, yu - 1, zu - 1), u);
            float __uValue = lerp(_luValue, _uuValue, v);

            float ret = lerp(__lValue, __uValue, w);
            return ret;
        }
    };

    class MultiOctavePerlinNoise3D {
        PerlinNoise3D m_primaryNoiseGen;
        uint32_t m_numOctaves;
        float m_initialFrequency;
        float m_initialAmplitude;
        float m_frequencyMultiplier;
        float m_persistence;
        float m_supValue;

    public:
        CUDA_COMMON_FUNCTION MultiOctavePerlinNoise3D(
            uint32_t numOctaves, float initialFrequency, float supValueOrInitialAmplitude, bool supSpecified,
            float frequencyMultiplier, float persistence, uint32_t repeat) :
            m_primaryNoiseGen(repeat),
            m_numOctaves(numOctaves),
            m_initialFrequency(initialFrequency),
            m_frequencyMultiplier(frequencyMultiplier), m_persistence(persistence) {
            if (supSpecified) {
                float amplitude = 1.0f;
                float tempSupValue = 0;
                for (int i = 0; i < static_cast<int32_t>(m_numOctaves); ++i) {
                    tempSupValue += amplitude;
                    amplitude *= m_persistence;
                }
                m_initialAmplitude = supValueOrInitialAmplitude / tempSupValue;
                m_supValue = supValueOrInitialAmplitude;
            }
            else {
                m_initialAmplitude = supValueOrInitialAmplitude;
                float amplitude = m_initialAmplitude;
                m_supValue = 0;
                for (int i = 0; i < static_cast<int32_t>(m_numOctaves); ++i) {
                    m_supValue += amplitude;
                    amplitude *= m_persistence;
                }
            }
        }

        CUDA_COMMON_FUNCTION float evaluate(const Point3D &p) const {
            float total = 0;
            float frequency = m_initialFrequency;
            float amplitude = m_initialAmplitude;
            for (int i = 0; i < static_cast<int32_t>(m_numOctaves); ++i) {
                total += m_primaryNoiseGen.evaluate(p, frequency) * amplitude;

                amplitude *= m_persistence;
                frequency *= m_frequencyMultiplier;
            }

            return total;
        }
    };



    struct TexDimInfo {
        uint32_t dimX : 14;
        uint32_t dimY : 14;
        uint32_t isNonPowerOfTwo : 1;
        uint32_t isBCTexture : 1;
        uint32_t isLeftHanded : 1; // for normal map
    };



    using ReadModifiedNormal = DynamicFunction<
        Normal3D(CUtexObject texture, TexDimInfo dimInfo, Point2D texCoord, float mipLevel)>;

    using BSDFGetSurfaceParameters = DynamicFunction<
        void(const uint32_t* data, RGB* diffuseReflectance, RGB* specularReflectance, float* roughness)>;
    using BSDFSampleThroughput = DynamicFunction<
        RGB(const uint32_t* data, const Vector3D &vGiven, float uDir0, float uDir1,
            Vector3D* vSampled, float* dirPDensity)>;
    using BSDFEvaluate = DynamicFunction<
        RGB(const uint32_t* data, const Vector3D &vGiven, const Vector3D &vSampled)>;
    using BSDFEvaluatePDF = DynamicFunction<
        float(const uint32_t* data, const Vector3D &vGiven, const Vector3D &vSampled)>;
    using BSDFEvaluateDHReflectanceEstimate = DynamicFunction<
        RGB(const uint32_t* data, const Vector3D &vGiven)>;



    struct Vertex {
        Point3D position;
        Normal3D normal;
        Vector3D texCoord0Dir;
        Point2D texCoord;
    };

    struct Triangle {
        uint32_t index0, index1, index2;
    };

    struct MaterialData;

    struct BSDFFlags {
        enum Value {
            None = 0,
            Regularize = 1 << 0,
        } value;

        CUDA_DEVICE_FUNCTION constexpr BSDFFlags(Value v = None) : value(v) {}

        CUDA_DEVICE_FUNCTION operator uint32_t() const {
            return static_cast<uint32_t>(value);
        }
    };

    using SetupBSDFBody = DynamicFunction<
        void(const MaterialData &matData, Point2D texCoord, float mipLevel,
             uint32_t* bodyData, BSDFFlags flags)>;

    struct MaterialData {
        union {
            struct {
                CUtexObject reflectance;
                TexDimInfo reflectanceDimInfo;
            } asLambert;
            struct {
                CUtexObject diffuse;
                CUtexObject specular;
                CUtexObject smoothness;
                TexDimInfo diffuseDimInfo;
                TexDimInfo specularDimInfo;
                TexDimInfo smoothnessDimInfo;
            } asDiffuseAndSpecular;
            struct {
                CUtexObject baseColor_opacity;
                CUtexObject occlusion_roughness_metallic;
                TexDimInfo baseColor_opacity_dimInfo;
                TexDimInfo occlusion_roughness_metallic_dimInfo;
            } asSimplePBR;
        };
        CUtexObject normal;
        CUtexObject emittance;
        TexDimInfo normalDimInfo;

        ReadModifiedNormal readModifiedNormal;

        SetupBSDFBody setupBSDFBody;
        BSDFGetSurfaceParameters bsdfGetSurfaceParameters;
        BSDFSampleThroughput bsdfSampleThroughput;
        BSDFEvaluate bsdfEvaluate;
        BSDFEvaluatePDF bsdfEvaluatePDF;
        BSDFEvaluateDHReflectanceEstimate bsdfEvaluateDHReflectanceEstimate;

        // for TFDM
        int2 heightMapSize;
        CUtexObject heightMap;
        optixu::NativeBlockBuffer2D<float2>* minMaxMipMap;
    };

    struct GeometryInstanceData {
        ROBuffer<Vertex> vertexBuffer;
        ROBuffer<Triangle> triangleBuffer;
        LightDistribution emitterPrimDist;
        uint32_t materialSlot;
        uint32_t geomInstSlot;
    };

    // for TFDM
    struct DisplacedTriangleAuxInfo {
        Matrix4x4 matObjToTc;
        Matrix3x3 matTcToBc;
        Matrix3x3 matTcToNInObj;
        Matrix3x3 matTcToObj;
    };

    // for TFDM
    struct GeometryInstanceDataForTFDM {
        ROBuffer<DisplacedTriangleAuxInfo> dispTriAuxInfoBuffer;
        ROBuffer<AABB> aabbBuffer;
        float hOffset;
        float hScale;
        float hBias;
        Matrix3x3 textureTransform;
    };

    struct InstanceData {
        Matrix4x4 transform;
        Matrix4x4 curToPrevTransform;
        Matrix3x3 normalMatrix;
        float uniformScale;

        ROBuffer<uint32_t> geomInstSlots;
        LightDistribution lightGeomInstDist;
    };
}
