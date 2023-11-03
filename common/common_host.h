#pragma once

#include "common_shared.h"

#include "../utils/gl_util.h"
#include <numbers>
#include <fstream>
#include <sstream>
#include <array>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <random>
#include <filesystem>
#include <functional>
#include <thread>
#include <chrono>
#include <variant>

#include "../ext/cubd/cubd.h"
#include "stopwatch.h"

#define ENABLE_VDB 0

#if ENABLE_VDB
#include "../common/vdb_interface.h"

inline void setColor(float r, float g, float b) {
    vdb_color(r, g, b);
}

inline void setColor(const RGB &c) {
    vdb_color(c.r, c.g, c.b);
}

inline void drawPoint(const Point3D &p) {
    vdb_point(p.x, p.y, p.z);
}

inline void drawLine(const Point3D &a, const Point3D &b) {
    vdb_line(a.x, a.y, a.z, b.x, b.y, b.z);
}

template <bool isNormal>
inline void drawVector(const Point3D &p, const Vector3D_T<float, isNormal> &v, float scale = 1.0f) {
    Point3D b = p + scale * v;
    vdb_line(p.x, p.y, p.z, b.x, b.y, b.z);
}

inline void drawTriangle(const Point3D &a, const Point3D &b, const Point3D &c) {
    vdb_triangle(
        a.x, a.y, a.z,
        b.x, b.y, b.z,
        c.x, c.y, c.z);
}

inline void drawAabb(const AABB &aabb) {
    drawLine(Point3D(aabb.minP.x, aabb.minP.y, aabb.minP.z), Point3D(aabb.maxP.x, aabb.minP.y, aabb.minP.z));
    drawLine(Point3D(aabb.minP.x, aabb.maxP.y, aabb.minP.z), Point3D(aabb.maxP.x, aabb.maxP.y, aabb.minP.z));
    drawLine(Point3D(aabb.minP.x, aabb.minP.y, aabb.maxP.z), Point3D(aabb.maxP.x, aabb.minP.y, aabb.maxP.z));
    drawLine(Point3D(aabb.minP.x, aabb.maxP.y, aabb.maxP.z), Point3D(aabb.maxP.x, aabb.maxP.y, aabb.maxP.z));

    drawLine(Point3D(aabb.minP.x, aabb.minP.y, aabb.minP.z), Point3D(aabb.minP.x, aabb.maxP.y, aabb.minP.z));
    drawLine(Point3D(aabb.maxP.x, aabb.minP.y, aabb.minP.z), Point3D(aabb.maxP.x, aabb.maxP.y, aabb.minP.z));
    drawLine(Point3D(aabb.minP.x, aabb.minP.y, aabb.maxP.z), Point3D(aabb.minP.x, aabb.maxP.y, aabb.maxP.z));
    drawLine(Point3D(aabb.maxP.x, aabb.minP.y, aabb.maxP.z), Point3D(aabb.maxP.x, aabb.maxP.y, aabb.maxP.z));

    drawLine(Point3D(aabb.minP.x, aabb.minP.y, aabb.minP.z), Point3D(aabb.minP.x, aabb.minP.y, aabb.maxP.z));
    drawLine(Point3D(aabb.maxP.x, aabb.minP.y, aabb.minP.z), Point3D(aabb.maxP.x, aabb.minP.y, aabb.maxP.z));
    drawLine(Point3D(aabb.minP.x, aabb.maxP.y, aabb.minP.z), Point3D(aabb.minP.x, aabb.maxP.y, aabb.maxP.z));
    drawLine(Point3D(aabb.maxP.x, aabb.maxP.y, aabb.minP.z), Point3D(aabb.maxP.x, aabb.maxP.y, aabb.maxP.z));
}

#endif

template <std::floating_point T>
static constexpr T pi_v = std::numbers::pi_v<T>;

#if 1
#   define hpprintf(fmt, ...) do { devPrintf(fmt, ##__VA_ARGS__); printf(fmt, ##__VA_ARGS__); } while (0)
#else
#   define hpprintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif

template <typename T, typename Deleter, typename ...ArgTypes>
std::shared_ptr<T> make_shared_with_deleter(const Deleter &deleter, ArgTypes&&... args) {
    return std::shared_ptr<T>(new T(std::forward<ArgTypes>(args)...),
                              deleter);
}

std::filesystem::path getExecutableDirectory();

std::string readTxtFile(const std::filesystem::path &filepath);

std::vector<char> readBinaryFile(const std::filesystem::path &filepath);



template <uint32_t numBuffers>
class StreamChain {
    std::array<CUstream, numBuffers> m_streams;
    std::array<CUevent, numBuffers> m_endEvents;
    uint32_t m_curBufIdx;

public:
    StreamChain() {
        for (int i = 0; i < numBuffers; ++i) {
            m_streams[i] = nullptr;
            m_endEvents[i] = nullptr;
        }
    }

    void initialize(CUcontext cuContext) {
        for (int i = 0; i < numBuffers; ++i) {
            CUDADRV_CHECK(cuStreamCreate(&m_streams[i], 0));
            CUDADRV_CHECK(cuEventCreate(&m_endEvents[i], 0));
        }
        m_curBufIdx = 0;
    }

    void finalize() {
        for (int i = 1; i >= 0; --i) {
            CUDADRV_CHECK(cuStreamSynchronize(m_streams[i]));
            CUDADRV_CHECK(cuEventDestroy(m_endEvents[i]));
            CUDADRV_CHECK(cuStreamDestroy(m_streams[i]));
            m_endEvents[i] = nullptr;
            m_streams[i] = nullptr;
        }
    }

    void swap() {
        CUstream curStream = m_streams[m_curBufIdx];
        CUevent curEvent = m_endEvents[m_curBufIdx];
        CUDADRV_CHECK(cuEventRecord(curEvent, curStream));
        m_curBufIdx = (m_curBufIdx + 1) % numBuffers;
    }

    CUstream waitAvailableAndGetCurrentStream() const {
        CUstream curStream = m_streams[m_curBufIdx];
        CUevent prevStreamEndEvent = m_endEvents[(m_curBufIdx + numBuffers - 1) % numBuffers];
        CUDADRV_CHECK(cuStreamSynchronize(curStream));
        CUDADRV_CHECK(cuStreamWaitEvent(curStream, prevStreamEndEvent, 0));
        return curStream;
    }

    void waitAllWorkDone() const {
        for (int i = 0; i < numBuffers; ++i)
            CUDADRV_CHECK(cuStreamSynchronize(m_streams[i]));
    }
};



template <typename RealType>
class DiscreteDistribution1DTemplate {
    cudau::TypedBuffer<RealType> m_weights;
#if defined(USE_WALKER_ALIAS_METHOD)
    cudau::TypedBuffer<shared::AliasTableEntry<RealType>> m_aliasTable;
    cudau::TypedBuffer<shared::AliasValueMap<RealType>> m_valueMaps;
#else
    cudau::TypedBuffer<RealType> m_CDF;
#endif
    RealType m_integral;
    uint32_t m_numValues;
    unsigned int m_isInitialized : 1;

public:
    DiscreteDistribution1DTemplate() :
        m_integral(0.0f), m_numValues(0), m_isInitialized(false) {}
    void initialize(
        CUcontext cuContext, cudau::BufferType type,
        const RealType* values, uint32_t numValues);
    void finalize() {
        if (!m_isInitialized)
            return;
#if defined(USE_WALKER_ALIAS_METHOD)
        if (m_valueMaps.isInitialized() && m_aliasTable.isInitialized() && m_weights.isInitialized()) {
            m_valueMaps.finalize();
            m_aliasTable.finalize();
            m_weights.finalize();
        }
#else
        if (m_CDF.isInitialized() && m_weights.isInitialized()) {
            m_CDF.finalize();
            m_weights.finalize();
        }
#endif
    }

    DiscreteDistribution1DTemplate &operator=(DiscreteDistribution1DTemplate &&v) {
        m_weights = std::move(v.m_weights);
#if defined(USE_WALKER_ALIAS_METHOD)
        m_aliasTable = std::move(v.m_aliasTable);
        m_valueMaps = std::move(v.m_valueMaps);
#else
        m_CDF = std::move(v.m_CDF);
#endif
        m_integral = v.m_integral;
        m_numValues = v.m_numValues;
        return *this;
    }

    RealType getIntengral() const {
        return m_integral;
    }

    bool isInitialized() const { return m_isInitialized; }

    void getDeviceType(shared::DiscreteDistribution1DTemplate<RealType>* instance) const {
#if defined(USE_WALKER_ALIAS_METHOD)
        new (instance) shared::DiscreteDistribution1DTemplate<RealType>(
            m_weights.isInitialized() ? m_weights.getDevicePointer() : nullptr,
            m_aliasTable.isInitialized() ? m_aliasTable.getDevicePointer() : nullptr,
            m_valueMaps.isInitialized() ? m_valueMaps.getDevicePointer() : nullptr,
            m_integral, m_numValues);
#else
        new (instance) shared::DiscreteDistribution1DTemplate<RealType>(
            m_weights.isInitialized() ? m_weights.getDevicePointer() : nullptr,
            m_CDF.isInitialized() ? m_CDF.getDevicePointer() : nullptr,
            m_integral, m_numValues);
#endif
    }

    RealType* weightsOnDevice() const {
        return m_weights.getDevicePointer();
    }

#if !defined(USE_WALKER_ALIAS_METHOD)
    RealType* cdfOnDevice() const {
        return m_CDF.getDevicePointer();
    }
#endif
};



template <typename RealType>
class RegularConstantContinuousDistribution1DTemplate {
    cudau::TypedBuffer<RealType> m_PDF;
#if defined(USE_WALKER_ALIAS_METHOD)
    cudau::TypedBuffer<shared::AliasTableEntry<RealType>> m_aliasTable;
    cudau::TypedBuffer<shared::AliasValueMap<RealType>> m_valueMaps;
#else
    cudau::TypedBuffer<RealType> m_CDF;
#endif
    RealType m_integral;
    uint32_t m_numValues;
    unsigned int m_isInitialized : 1;

public:
    RegularConstantContinuousDistribution1DTemplate() : m_isInitialized(false) {}

    void initialize(
        CUcontext cuContext, cudau::BufferType type,
        const RealType* values, uint32_t numValues);
    void finalize(CUcontext cuContext) {
        if (!m_isInitialized)
            return;
#if defined(USE_WALKER_ALIAS_METHOD)
        if (m_valueMaps.isInitialized() && m_aliasTable.isInitialized() && m_PDF.isInitialized()) {
            m_valueMaps.finalize();
            m_aliasTable.finalize();
            m_PDF.finalize();
        }
#else
        if (m_CDF.isInitialized() && m_PDF.isInitialized()) {
            m_CDF.finalize();
            m_PDF.finalize();
        }
#endif
    }

    RegularConstantContinuousDistribution1DTemplate &operator=(RegularConstantContinuousDistribution1DTemplate &&v) {
        m_PDF = std::move(v.m_PDF);
#if defined(USE_WALKER_ALIAS_METHOD)
        m_aliasTable = std::move(v.m_aliasTable);
        m_valueMaps = std::move(v.m_valueMaps);
#else
        m_CDF = std::move(v.m_CDF);
#endif
        m_integral = v.m_integral;
        m_numValues = v.m_numValues;
        return *this;
    }

    RealType getIntegral() const { return m_integral; }
    uint32_t getNumValues() const { return m_numValues; }

    bool isInitialized() const { return m_isInitialized; }

    void getDeviceType(shared::RegularConstantContinuousDistribution1DTemplate<RealType>* instance) const {
#if defined(USE_WALKER_ALIAS_METHOD)
        new (instance) shared::RegularConstantContinuousDistribution1DTemplate<RealType>(
            m_PDF.getDevicePointer(), m_aliasTable.getDevicePointer(), m_valueMaps.getDevicePointer(),
            m_integral, m_numValues);
#else
        new (instance) shared::RegularConstantContinuousDistribution1DTemplate<RealType>(
            m_PDF.getDevicePointer(), m_CDF.getDevicePointer(), m_integral, m_numValues);
#endif
    }
};



template <typename RealType>
class RegularConstantContinuousDistribution2DTemplate {
    cudau::TypedBuffer<shared::RegularConstantContinuousDistribution1DTemplate<RealType>> m_raw1DDists;
    RegularConstantContinuousDistribution1DTemplate<RealType>* m_1DDists;
    RegularConstantContinuousDistribution1DTemplate<RealType> m_top1DDist;
    unsigned int m_isInitialized : 1;

public:
    RegularConstantContinuousDistribution2DTemplate() : m_1DDists(nullptr), m_isInitialized(false) {}

    RegularConstantContinuousDistribution2DTemplate &operator=(RegularConstantContinuousDistribution2DTemplate &&v) {
        m_raw1DDists = std::move(v.m_raw1DDists);
        m_1DDists = std::move(v.m_1DDists);
        m_top1DDist = std::move(v.m_top1DDist);
        return *this;
    }

    void initialize(
        CUcontext cuContext, cudau::BufferType type,
        const RealType* values, uint32_t numD1, uint32_t numD2);
    void finalize(CUcontext cuContext) {
        if (!m_isInitialized)
            return;

        m_top1DDist.finalize(cuContext);

        for (int i = m_top1DDist.getNumValues() - 1; i >= 0; --i) {
            m_1DDists[i].finalize(cuContext);
        }

        m_raw1DDists.finalize();
        delete[] m_1DDists;
        m_1DDists = nullptr;
    }

    bool isInitialized() const { return m_isInitialized; }

    void getDeviceType(shared::RegularConstantContinuousDistribution2DTemplate<RealType>* instance) const {
        shared::RegularConstantContinuousDistribution1DTemplate<RealType> top1DDist;
        m_top1DDist.getDeviceType(&top1DDist);
        new (instance) shared::RegularConstantContinuousDistribution2DTemplate<RealType>(
            m_raw1DDists.getDevicePointer(), top1DDist);
    }
};



using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;
using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float>;
using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float>;



class ProbabilityTexture {
    cudau::Array m_cuArray;
    CUtexObject m_cuTexObj;
    unsigned int m_isInitialized : 1;

public:
    ProbabilityTexture() : m_cuTexObj(0), m_isInitialized(false) {}

    void initialize(CUcontext cuContext, uint32_t numValues);
    void finalize() {
        if (!m_isInitialized)
            return;
        CUDADRV_CHECK(cuTexObjectDestroy(m_cuTexObj));
        m_cuArray.finalize();
    }

    bool isInitialized() const { return m_isInitialized; }

    CUsurfObject getSurfaceObject(uint32_t mipLevel) const {
        return m_cuArray.getSurfaceObject(mipLevel);
    }

    void getDeviceType(shared::ProbabilityTexture* probTex) const {
        probTex->setTexObject(
            m_cuTexObj,
            uint2(static_cast<uint32_t>(m_cuArray.getWidth()),
                  static_cast<uint32_t>(m_cuArray.getHeight())));
    }
};

using LightDistribution =
#if USE_PROBABILITY_TEXTURE
    ProbabilityTexture;
#else
    DiscreteDistribution1D;
#endif



struct MovingAverageTime {
    float values[60];
    uint32_t index;
    uint32_t numValidValues;
    MovingAverageTime() : index(0), numValidValues(0) {}
    void append(float value) {
        values[index] = value;
        index = (index + 1) % lengthof(values);
        numValidValues = std::min<uint32_t>(numValidValues + 1, static_cast<uint32_t>(lengthof(values)));
    }
    float getAverage() const {
        float sum = 0.0f;
        for (uint32_t i = 0; i < numValidValues; ++i)
            sum += values[(index - 1 - i + lengthof(values)) % lengthof(values)];
        return numValidValues > 0 ? sum / numValidValues : 0.0f;
    }
};



class SlotFinder {
    uint32_t m_numLayers;
    uint32_t m_numLowestFlagBins;
    uint32_t m_numTotalCompiledFlagBins;
    uint32_t* m_flagBins;
    uint32_t* m_offsetsToOR_AND;
    uint32_t* m_numUsedFlagsUnderBinList;
    uint32_t* m_offsetsToNumUsedFlags;
    uint32_t* m_numFlagsInLayerList;

    SlotFinder(const SlotFinder &) = delete;
    SlotFinder &operator=(const SlotFinder &) = delete;

    void aggregate();

    uint32_t getNumLayers() const {
        return m_numLayers;
    }

    const uint32_t* getOffsetsToOR_AND() const {
        return m_offsetsToOR_AND;
    }

    const uint32_t* getOffsetsToNumUsedFlags() const {
        return m_offsetsToNumUsedFlags;
    }

    const uint32_t* getNumFlagsInLayerList() const {
        return m_numFlagsInLayerList;
    }

public:
    static constexpr uint32_t InvalidSlotIndex = 0xFFFFFFFF;

    SlotFinder() :
        m_numLayers(0), m_numLowestFlagBins(0), m_numTotalCompiledFlagBins(0),
        m_flagBins(nullptr), m_offsetsToOR_AND(nullptr),
        m_numUsedFlagsUnderBinList(nullptr), m_offsetsToNumUsedFlags(nullptr),
        m_numFlagsInLayerList(nullptr) {
    }
    ~SlotFinder() {
    }

    void initialize(uint32_t numSlots);

    void finalize();

    SlotFinder &operator=(SlotFinder &&inst) {
        finalize();

        m_numLayers = inst.m_numLayers;
        m_numLowestFlagBins = inst.m_numLowestFlagBins;
        m_numTotalCompiledFlagBins = inst.m_numTotalCompiledFlagBins;
        m_flagBins = inst.m_flagBins;
        m_offsetsToOR_AND = inst.m_offsetsToOR_AND;
        m_numUsedFlagsUnderBinList = inst.m_numUsedFlagsUnderBinList;
        m_offsetsToNumUsedFlags = inst.m_offsetsToNumUsedFlags;
        m_numFlagsInLayerList = inst.m_numFlagsInLayerList;
        inst.m_flagBins = nullptr;
        inst.m_offsetsToOR_AND = nullptr;
        inst.m_numUsedFlagsUnderBinList = nullptr;
        inst.m_offsetsToNumUsedFlags = nullptr;
        inst.m_numFlagsInLayerList = nullptr;

        return *this;
    }
    SlotFinder(SlotFinder &&inst) {
        *this = std::move(inst);
    }

    void resize(uint32_t numSlots);

    void reset() {
        std::fill_n(m_flagBins, m_numLowestFlagBins + m_numTotalCompiledFlagBins, 0);
        std::fill_n(m_numUsedFlagsUnderBinList, m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2, 0);
    }



    void setInUse(uint32_t slotIdx);

    void setNotInUse(uint32_t slotIdx);

    bool getUsage(uint32_t slotIdx) const {
        uint32_t binIdx = slotIdx / 32;
        uint32_t flagIdxInBin = slotIdx % 32;
        uint32_t flagBin = m_flagBins[binIdx];

        return (bool)((flagBin >> flagIdxInBin) & 0x1);
    }

    uint32_t getFirstAvailableSlot() const;

    uint32_t getFirstUsedSlot() const;

    uint32_t find_nthUsedSlot(uint32_t n) const;

    uint32_t getNumSlots() const {
        return m_numFlagsInLayerList[0];
    }

    uint32_t getNumUsed() const {
        return m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[m_numLayers - 1]];
    }

    void debugPrint() const;
};



enum class MaterialConvention {
    Traditional = 0,
    SimplePBR,
};

enum class BumpMapTextureType {
    NormalMap = 0,
    NormalMap_BC,
    NormalMap_BC_2ch,
    HeightMap,
    HeightMap_BC,
};

struct Material {
    struct Lambert {
        const cudau::Array* reflectance;
        CUtexObject texReflectance;
        Lambert() :
            reflectance(nullptr), texReflectance(0) {}
    };
    struct DiffuseAndSpecular {
        const cudau::Array* diffuse;
        CUtexObject texDiffuse;
        const cudau::Array* specular;
        CUtexObject texSpecular;
        const cudau::Array* smoothness;
        CUtexObject texSmoothness;
        DiffuseAndSpecular() :
            diffuse(nullptr), texDiffuse(0),
            specular(nullptr), texSpecular(0),
            smoothness(nullptr), texSmoothness(0) {}
    };
    struct SimplePBR {
        const cudau::Array* baseColor_opacity;
        CUtexObject texBaseColor_opacity;
        const cudau::Array* occlusion_roughness_metallic;
        CUtexObject texOcclusion_roughness_metallic;

        SimplePBR() :
            baseColor_opacity(nullptr), texBaseColor_opacity(0),
            occlusion_roughness_metallic(nullptr), texOcclusion_roughness_metallic(0) {}
    };
    std::variant<
        Lambert,
        DiffuseAndSpecular,
        SimplePBR> body;
    const cudau::Array* normal;
    CUtexObject texNormal;
    const glu::Texture2D* gfxNormal;
    glu::Sampler gfxSampler;
    BumpMapTextureType bumpMapType;
    const cudau::Array* emittance;
    CUtexObject texEmittance;

    // for TFDM
    const cudau::Array* heightMap;
    CUtexObject heightMapTex;
    cudau::Array minMaxMipMap;
    cudau::TypedBuffer<optixu::NativeBlockBuffer2D<float2>> minMaxMipMapSurfs;

    uint32_t materialSlot;

    Material() :
        normal(nullptr), texNormal(0),
        emittance(nullptr), texEmittance(0),
        heightMap(nullptr), heightMapTex(0),
        materialSlot(0) {}
};

struct GeometryInstance {
    const Material* mat;

    glu::Buffer gfxVertexBuffer;
    glu::Buffer gfxTriangleBuffer;
    glu::VertexArray gfxVertexArray;
    cudau::TypedBuffer<shared::Vertex> vertexBuffer;
    cudau::TypedBuffer<shared::Triangle> triangleBuffer;
    LightDistribution emitterPrimDist;
    uint32_t geomInstSlot;
    optixu::GeometryInstance optixGeomInst;
    AABB aabb;
    optixu::GeometryType geometryType;
    // for TFDM
    cudau::TypedBuffer<shared::DisplacedTriangleAuxInfo> dispTriAuxInfoBuffer;
    cudau::TypedBuffer<AABB> aabbBuffer;

    void draw() const {
        glUniform1ui(9, mat->materialSlot);
        glBindTextureUnit(0, mat->gfxNormal->getHandle());
        glBindSampler(0, mat->gfxSampler.getHandle());
        uint32_t flags = 0;
        if (mat->bumpMapType == BumpMapTextureType::NormalMap ||
            mat->bumpMapType == BumpMapTextureType::NormalMap_BC)
            flags |= 0 << 0;
        else if (mat->bumpMapType == BumpMapTextureType::NormalMap_BC_2ch)
            flags |= 1 << 0;
        else
            flags |= 2 << 0;
        glUniform1ui(11, flags);

        glBindVertexArray(gfxVertexArray.getHandle());
        glDrawElements(
            GL_TRIANGLES, 3 * static_cast<GLsizei>(triangleBuffer.numElements()),
            GL_UNSIGNED_INT, nullptr);
    }
};

struct GeometryGroup {
    std::set<const GeometryInstance*> geomInsts;

    optixu::GeometryAccelerationStructure optixGas;
    cudau::Buffer optixGasMem;
    uint32_t numEmitterPrimitives;
    AABB aabb;
    uint32_t needsRebuild : 1;
    uint32_t refittable : 1;

    void draw() const {
        for (const GeometryInstance* geomInst : geomInsts)
            geomInst->draw();
    }
};

struct Mesh {
    struct GeometryGroupInstance {
        const GeometryGroup* geomGroup;
        Matrix4x4 transform;
    };
    std::vector<GeometryGroupInstance> groupInsts;
};

struct Instance {
    Mesh::GeometryGroupInstance geomGroupInst;

    cudau::TypedBuffer<uint32_t> geomInstSlots;
    LightDistribution lightGeomInstDist;
    uint32_t instSlot;
    optixu::Instance optixInst;

    Matrix4x4 prevMatM2W;
    Matrix4x4 matM2W;
    Matrix3x3 nMatM2W;

    void draw() const {
        glUniformMatrix4fv(5, 1, false, reinterpret_cast<const float*>(&prevMatM2W));
        glUniformMatrix4fv(6, 1, false, reinterpret_cast<const float*>(&matM2W));
        glUniformMatrix3fv(7, 1, false, reinterpret_cast<const float*>(&nMatM2W));
        glUniform1ui(8, instSlot);
        geomGroupInst.geomGroup->draw();
    }
};

struct InstanceController {
    Instance* inst;

    float curScale;
    Quaternion curOrientation;
    Point3D curPosition;

    float beginScale;
    Quaternion beginOrientation;
    Point3D beginPosition;
    float endScale;
    Quaternion endOrientation;
    Point3D endPosition;
    float time;
    float frequency;

    InstanceController(
        Instance* _inst,
        float _beginScale, const Quaternion &_beginOrienatation, const Point3D &_beginPosition,
        float _endScale, const Quaternion &_endOrienatation, const Point3D &_endPosition,
        float _frequency, float initTime) :
        inst(_inst),
        beginScale(_beginScale), beginOrientation(_beginOrienatation), beginPosition(_beginPosition),
        endScale(_endScale), endOrientation(_endOrienatation), endPosition(_endPosition),
        time(initTime), frequency(_frequency) {
    }

    void updateBody(float dt) {
        time = std::fmod(time + dt, frequency);
        float t = 0.5f - 0.5f * std::cos(2 * pi_v<float> * time / frequency);
        curScale = (1 - t) * beginScale + t * endScale;
        curOrientation = Slerp(t, beginOrientation, endOrientation);
        curPosition = (1 - t) * beginPosition + t * endPosition;
    }

    void update(shared::InstanceData* instDataBuffer, float dt) {
        Matrix4x4 prevTransform = inst->matM2W;
        updateBody(dt);
        Matrix3x3 matRot = curOrientation.toMatrix3x3();

        inst->matM2W =
            Matrix4x4(matRot * curScale, curPosition) *
            inst->geomGroupInst.transform;
        inst->nMatM2W = matRot / curScale;

        Matrix4x4 tMatM2W = transpose(inst->matM2W);
        inst->optixInst.setTransform(reinterpret_cast<const float*>(&tMatM2W));

        Vector3D scale;
        inst->matM2W.decompose(&scale, nullptr, nullptr);
        float uniformScale = scale.x;

        shared::InstanceData &instData = instDataBuffer[inst->instSlot];
        instData.curToPrevTransform = prevTransform * invert(inst->matM2W);
        instData.transform = inst->matM2W;
        instData.normalMatrix = inst->nMatM2W;
        instData.uniformScale = uniformScale;
    }
};

// TODO: シーンまわり綺麗にしたい。
struct Scene {
    static constexpr cudau::BufferType bufferType = cudau::BufferType::Device;

    static constexpr uint32_t maxNumMaterials = 1024;
    static constexpr uint32_t maxNumGeometryInstances = 65536;
    static constexpr uint32_t maxNumInstances = 16384;

    struct ComputeProbTex {
        CUmodule cudaModule;
        cudau::Kernel computeFirstMip;
        cudau::Kernel computeTriangleProbTexture;
        cudau::Kernel computeGeomInstProbTexture;
        cudau::Kernel computeInstProbTexture;
        cudau::Kernel computeMip;
        cudau::Kernel computeTriangleProbBuffer;
        cudau::Kernel computeGeomInstProbBuffer;
        cudau::Kernel computeInstProbBuffer;
        cudau::Kernel finalizeDiscreteDistribution1D;
        cudau::Kernel test;
    } computeProbTex;

    optixu::Scene optixScene;
    uint32_t numRayTypes;

    SlotFinder materialSlotFinder;
    SlotFinder geomInstSlotFinder;
    SlotFinder instSlotFinder;
    cudau::TypedBuffer<shared::MaterialData> materialDataBuffer;
    cudau::TypedBuffer<shared::GeometryInstanceData> geomInstDataBuffer;
    cudau::TypedBuffer<shared::InstanceData> instDataBuffer[2];

    std::vector<Material*> materials;
    std::vector<GeometryInstance*> geomInsts;
    std::vector<GeometryGroup*> geomGroups;

    std::map<std::string, Mesh*> meshes;

    std::vector<Instance*> insts;
    std::vector<InstanceController*> instControllers;
    AABB initialSceneAabb;

    LightDistribution lightInstDist;

    optixu::InstanceAccelerationStructure ias;
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> iasInstanceBuffer;

    cudau::Buffer asScratchMem;
    cudau::Buffer scanScratchMem; // TODO: unify

    size_t hitGroupSbtSize;

    void initialize(
        const std::filesystem::path &ptxDir, CUcontext cuContext, optixu::Context optixContext,
        uint32_t _numRayTypes) {
        CUDADRV_CHECK(cuModuleLoad(
            &computeProbTex.cudaModule,
            (ptxDir / "compute_light_probs.ptx").string().c_str()));
        computeProbTex.computeFirstMip =
            cudau::Kernel(computeProbTex.cudaModule, "computeProbabilityTextureFirstMip", cudau::dim3(32), 0);
        computeProbTex.computeTriangleProbTexture =
            cudau::Kernel(computeProbTex.cudaModule, "computeTriangleProbTexture", cudau::dim3(32), 0);
        computeProbTex.computeGeomInstProbTexture =
            cudau::Kernel(computeProbTex.cudaModule, "computeGeomInstProbTexture", cudau::dim3(32), 0);
        computeProbTex.computeInstProbTexture =
            cudau::Kernel(computeProbTex.cudaModule, "computeInstProbTexture", cudau::dim3(32), 0);
        computeProbTex.computeMip =
            cudau::Kernel(computeProbTex.cudaModule, "computeProbabilityTextureMip", cudau::dim3(8, 8), 0);
        computeProbTex.computeTriangleProbBuffer =
            cudau::Kernel(computeProbTex.cudaModule, "computeTriangleProbBuffer", cudau::dim3(32), 0);
        computeProbTex.computeGeomInstProbBuffer =
            cudau::Kernel(computeProbTex.cudaModule, "computeGeomInstProbBuffer", cudau::dim3(32), 0);
        computeProbTex.computeInstProbBuffer =
            cudau::Kernel(computeProbTex.cudaModule, "computeInstProbBuffer", cudau::dim3(32), 0);
        computeProbTex.finalizeDiscreteDistribution1D =
            cudau::Kernel(computeProbTex.cudaModule, "finalizeDiscreteDistribution1D", cudau::dim3(32), 0);
        computeProbTex.test =
            cudau::Kernel(computeProbTex.cudaModule, "testProbabilityTexture", cudau::dim3(32), 0);

        optixScene = optixContext.createScene();
        numRayTypes = _numRayTypes;

        materialSlotFinder.initialize(maxNumMaterials);
        geomInstSlotFinder.initialize(maxNumGeometryInstances);
        instSlotFinder.initialize(maxNumInstances);

        materialDataBuffer.initialize(cuContext, bufferType, maxNumMaterials);
        geomInstDataBuffer.initialize(cuContext, bufferType, maxNumGeometryInstances);
        instDataBuffer[0].initialize(cuContext, bufferType, maxNumInstances);
        instDataBuffer[1].initialize(cuContext, bufferType, maxNumInstances);

        ias = optixScene.createInstanceAccelerationStructure();

#if USE_PROBABILITY_TEXTURE
        lightInstDist.initialize(cuContext, maxNumInstances);
#else
        lightInstDist.initialize(cuContext, bufferType, nullptr, maxNumInstances);
#endif

        size_t scanScratchSize;
        constexpr int32_t maxScanSize = std::max<int32_t>({
            maxNumMaterials,
            maxNumGeometryInstances,
            maxNumInstances });
        CUDADRV_CHECK(cubd::DeviceScan::ExclusiveSum(
            nullptr, scanScratchSize,
            static_cast<float*>(nullptr), static_cast<float*>(nullptr), maxScanSize));
        scanScratchMem.initialize(cuContext, bufferType, scanScratchSize, 1u);
    }

    void finalize() {
        scanScratchMem.finalize();

        lightInstDist.finalize();

        asScratchMem.finalize();
        iasInstanceBuffer.finalize();
        iasMem.finalize();
        for (int i = static_cast<int>(geomGroups.size()) - 1; i >= 0; --i) {
            GeometryGroup* geomGroup = geomGroups[i];
            geomGroup->optixGasMem.finalize();
        }
        ias.destroy();

        for (int i = static_cast<int>(insts.size()) - 1; i >= 0; --i) {
            Instance* inst = insts[i];
            inst->optixInst.destroy();
            inst->lightGeomInstDist.finalize();
            inst->lightGeomInstDist.finalize();
        }
        for (int i = static_cast<int>(geomGroups.size()) - 1; i >= 0; --i) {
            GeometryGroup* geomGroup = geomGroups[i];
            geomGroup->optixGas.destroy();
        }
        for (int i = static_cast<int>(geomInsts.size()) - 1; i >= 0; --i) {
            GeometryInstance* geomInst = geomInsts[i];
            geomInst->aabbBuffer.finalize();
            geomInst->dispTriAuxInfoBuffer.finalize();
            geomInst->optixGeomInst.destroy();
            geomInst->emitterPrimDist.finalize();
            geomInst->triangleBuffer.finalize();
            geomInst->vertexBuffer.finalize();
            geomInst->gfxVertexArray.finalize();
            geomInst->gfxTriangleBuffer.finalize();
            geomInst->gfxVertexBuffer.finalize();
        }
        for (int i = static_cast<int>(materials.size()) - 1; i >= 0; --i) {
            Material* material = materials[i];
        }

        instDataBuffer[1].finalize();
        instDataBuffer[0].finalize();
        geomInstDataBuffer.finalize();
        materialDataBuffer.finalize();

        instSlotFinder.finalize();
        geomInstSlotFinder.finalize();
        materialSlotFinder.finalize();

        optixScene.destroy();

        CUDADRV_CHECK(cuModuleUnload(computeProbTex.cudaModule));
    }

    void map() {
        materialDataBuffer.map();
        geomInstDataBuffer.map();
        instDataBuffer[0].map();
    }
    void unmap() {
        instDataBuffer[0].unmap();
        geomInstDataBuffer.unmap();
        materialDataBuffer.unmap();
    }

    void setupASs(CUcontext cuContext) {
        for (int i = 0; i < insts.size(); ++i) {
            const Instance* inst = insts[i];
            ias.addChild(inst->optixInst);
        }

        OptixAccelBufferSizes asSizes;
        size_t asScratchSize = 0;
        for (int i = 0; i < geomGroups.size(); ++i) {
            GeometryGroup* geomGroup = geomGroups[i];
            geomGroup->optixGas.setConfiguration(
                optixu::ASTradeoff::PreferFastTrace,
                optixu::AllowUpdate::No, optixu::AllowCompaction::No, optixu::AllowRandomVertexAccess::No);
            geomGroup->optixGas.prepareForBuild(&asSizes);
            geomGroup->optixGasMem.initialize(
                cuContext, bufferType, asSizes.outputSizeInBytes, 1);
            asScratchSize = std::max(asSizes.tempSizeInBytes, asScratchSize);
        }

        ias.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No, optixu::AllowCompaction::No, optixu::AllowRandomInstanceAccess::No);
        ias.prepareForBuild(&asSizes);
        iasMem.initialize(
            cuContext, bufferType, asSizes.outputSizeInBytes, 1);
        asScratchSize = std::max(asSizes.tempSizeInBytes, asScratchSize);
        iasInstanceBuffer.initialize(cuContext, bufferType, ias.getNumChildren());

        asScratchMem.initialize(cuContext, bufferType, asScratchSize, 1);

        optixScene.generateShaderBindingTableLayout(&hitGroupSbtSize);

        for (int bufIdx = 0; bufIdx < 2; ++bufIdx) {
            cudau::TypedBuffer<shared::InstanceData> &curInstDataBuffer = instDataBuffer[bufIdx];
            shared::InstanceData* instDataBufferOnHost = curInstDataBuffer.map();
            for (int i = 0; i < instControllers.size(); ++i) {
                InstanceController* controller = instControllers[i];
                Instance* inst = controller->inst;
                shared::InstanceData &instData = instDataBufferOnHost[inst->instSlot];
                controller->update(instDataBufferOnHost, 0.0f);
                // TODO: まとめて送る。
                CUDADRV_CHECK(cuMemcpyHtoDAsync(
                    curInstDataBuffer.getCUdeviceptrAt(inst->instSlot),
                    &instData, sizeof(instData), 0));
            }
            curInstDataBuffer.unmap();
        }
    }

    OptixTraversableHandle updateASs(CUstream stream) {
        for (int i = 0; i < geomGroups.size(); ++i) {
            GeometryGroup* geomGroup = geomGroups[i];
            if (geomGroup->needsRebuild)
                geomGroup->optixGas.rebuild(0, geomGroup->optixGasMem, asScratchMem);
            geomGroup->needsRebuild = false;
        }

        return ias.rebuild(0, iasInstanceBuffer, iasMem, asScratchMem);
    }

    void setupLightGeomDistributions() {
        CUstream cuStream = 0;

        for (int geomInstIdx = 0; geomInstIdx < geomInsts.size(); ++geomInstIdx) {
            const GeometryInstance* geomInst = geomInsts[geomInstIdx];
            if (!geomInst->emitterPrimDist.isInitialized())
                continue;
            shared::GeometryInstanceData* geomInstData =
                geomInstDataBuffer.getDevicePointerAt(geomInst->geomInstSlot);
            uint32_t numTriangles = static_cast<uint32_t>(geomInst->triangleBuffer.numElements());
#if USE_PROBABILITY_TEXTURE
            uint2 dims = shared::computeProbabilityTextureDimentions(numTriangles);
            uint32_t numMipLevels = nextPowOf2Exponent(dims.x) + 1;
            computeProbTex.computeTriangleProbTexture(
                cuStream, computeProbTex.computeTriangleProbTexture.calcGridDim(dims.x * dims.y),
                geomInstData, numTriangles,
                materialDataBuffer.getDevicePointer(),
                geomInst->emitterPrimDist.getSurfaceObject(0));
#else
            computeProbTex.computeTriangleProbBuffer(
                cuStream, computeProbTex.computeTriangleProbBuffer.calcGridDim(numTriangles),
                geomInstData, numTriangles,
                materialDataBuffer.getDevicePointer());
#endif
            //hpprintf("%5u: %4u tris\n", geomInstIdx, numTriangles);
            //CUDADRV_CHECK(cuStreamSynchronize(cuStream));
        }

        for (int geomInstIdx = 0; geomInstIdx < geomInsts.size(); ++geomInstIdx) {
            const GeometryInstance* geomInst = geomInsts[geomInstIdx];
            if (!geomInst->emitterPrimDist.isInitialized())
                continue;
            shared::GeometryInstanceData* geomInstData =
                geomInstDataBuffer.getDevicePointerAt(geomInst->geomInstSlot);
            uint32_t numTriangles = static_cast<uint32_t>(geomInst->triangleBuffer.numElements());
#if USE_PROBABILITY_TEXTURE
            uint2 curDims = shared::computeProbabilityTextureDimentions(numTriangles);
            uint32_t numMipLevels = nextPowOf2Exponent(curDims.x) + 1;
            for (int dstMipLevel = 1; dstMipLevel < numMipLevels; ++dstMipLevel) {
                curDims = (curDims + uint2(1, 1)) / 2;
                computeProbTex.computeMip(
                    cuStream, computeProbTex.computeMip.calcGridDim(curDims.x, curDims.y),
                    &geomInstData->emitterPrimDist, dstMipLevel,
                    geomInst->emitterPrimDist.getSurfaceObject(dstMipLevel - 1),
                    geomInst->emitterPrimDist.getSurfaceObject(dstMipLevel));
                //hpprintf("%5u-%u: %3u x %3u\n", geomInstIdx, dstMipLevel, curDims.x, curDims.y);
            }
#else
            size_t scratchMemSize = scanScratchMem.sizeInBytes();
            CUDADRV_CHECK(cubd::DeviceScan::ExclusiveSum(
                scanScratchMem.getDevicePointer(), scratchMemSize,
                geomInst->emitterPrimDist.weightsOnDevice(),
                geomInst->emitterPrimDist.cdfOnDevice(),
                numTriangles, cuStream));

            computeProbTex.finalizeDiscreteDistribution1D(
                cuStream, computeProbTex.finalizeDiscreteDistribution1D.calcGridDim(1),
                &geomInstData->emitterPrimDist);
#endif
            //CUDADRV_CHECK(cuStreamSynchronize(cuStream));
        }

        //{
        //    CUDADRV_CHECK(cuStreamSynchronize(cuStream));
        //    for (int geomInstIdx = 0; geomInstIdx < geomInsts.size(); ++geomInstIdx) {
        //        GeometryInstance* geomInst = geomInsts[geomInstIdx];
        //        if (!geomInst->emitterPrimDist.isInitialized())
        //            continue;
        //        shared::GeometryInstanceData* geomInstData =
        //            geomInstDataBuffer.getDevicePointerAt(geomInst->geomInstSlot);
        //        shared::LightDistribution lightDistOnHost;
        //        CUDADRV_CHECK(cuMemcpyDtoH(
        //            &lightDistOnHost, reinterpret_cast<CUdeviceptr>(&geomInstData->emitterPrimDist),
        //            sizeof(lightDistOnHost)));
        //        hpprintf("%5u: %g\n", geomInstIdx, lightDistOnHost.integral());
        //    }
        //}

        for (int instIdx = 0; instIdx < insts.size(); ++instIdx) {
            const Instance* inst = insts[instIdx];
            if (!inst->lightGeomInstDist.isInitialized())
                continue;
            shared::InstanceData* instData = instDataBuffer[0].getDevicePointerAt(inst->instSlot);
            uint32_t numGeomInsts = static_cast<uint32_t>(inst->geomGroupInst.geomGroup->geomInsts.size());
#if USE_PROBABILITY_TEXTURE
            uint2 dims = shared::computeProbabilityTextureDimentions(numGeomInsts);
            uint32_t numMipLevels = nextPowOf2Exponent(dims.x) + 1;
            computeProbTex.computeGeomInstProbTexture(
                cuStream, computeProbTex.computeGeomInstProbTexture.calcGridDim(dims.x * dims.y),
                instData, instIdx, numGeomInsts,
                geomInstDataBuffer.getDevicePointer(),
                inst->lightGeomInstDist.getSurfaceObject(0));
#else
            computeProbTex.computeGeomInstProbBuffer(
                cuStream, computeProbTex.computeGeomInstProbBuffer.calcGridDim(numGeomInsts),
                instData, instIdx, numGeomInsts,
                geomInstDataBuffer.getDevicePointer());
#endif
            //hpprintf("%5u: %4u geomInsts\n", instIdx, numGeomInsts);
            //CUDADRV_CHECK(cuStreamSynchronize(cuStream));
        }

        for (int instIdx = 0; instIdx < insts.size(); ++instIdx) {
            const Instance* inst = insts[instIdx];
            if (!inst->lightGeomInstDist.isInitialized())
                continue;
            shared::InstanceData* instData = instDataBuffer[0].getDevicePointerAt(inst->instSlot);
            uint32_t numGeomInsts = static_cast<uint32_t>(inst->geomGroupInst.geomGroup->geomInsts.size());
#if USE_PROBABILITY_TEXTURE
            uint2 curDims = shared::computeProbabilityTextureDimentions(numGeomInsts);
            uint32_t numMipLevels = nextPowOf2Exponent(curDims.x) + 1;
            for (int dstMipLevel = 1; dstMipLevel < numMipLevels; ++dstMipLevel) {
                curDims = (curDims + uint2(1, 1)) / 2;
                computeProbTex.computeMip(
                    cuStream, computeProbTex.computeMip.calcGridDim(curDims.x, curDims.y),
                    &instData->lightGeomInstDist, dstMipLevel,
                    inst->lightGeomInstDist.getSurfaceObject(dstMipLevel - 1),
                    inst->lightGeomInstDist.getSurfaceObject(dstMipLevel));
                //hpprintf("%5u-%u: %3u x %3u\n", instIdx, dstMipLevel, curDims.x, curDims.y);
            }
#else
            size_t scratchMemSize = scanScratchMem.sizeInBytes();
            CUDADRV_CHECK(cubd::DeviceScan::ExclusiveSum(
                asScratchMem.getDevicePointer(), scratchMemSize,
                inst->lightGeomInstDist.weightsOnDevice(),
                inst->lightGeomInstDist.cdfOnDevice(),
                numGeomInsts, cuStream));

            computeProbTex.finalizeDiscreteDistribution1D(
                cuStream, computeProbTex.finalizeDiscreteDistribution1D.calcGridDim(1),
                &instData->lightGeomInstDist);
#endif
            //CUDADRV_CHECK(cuStreamSynchronize(cuStream));
        }

        //{
        //    CUDADRV_CHECK(cuStreamSynchronize(cuStream));
        //    for (int instIdx = 0; instIdx < insts.size(); ++instIdx) {
        //        Instance* inst = insts[instIdx];
        //        if (!inst->lightGeomInstDist.isInitialized())
        //            continue;
        //        shared::InstanceData* instData =
        //            instDataBuffer[0].getDevicePointerAt(inst->instSlot);
        //        shared::LightDistribution lightDistOnHost;
        //        CUDADRV_CHECK(cuMemcpyDtoH(
        //            &lightDistOnHost, reinterpret_cast<CUdeviceptr>(&instData->lightGeomInstDist),
        //            sizeof(lightDistOnHost)));
        //        hpprintf("%5u: %g\n", instIdx, lightDistOnHost.integral());
        //    }
        //}

        CUDADRV_CHECK(cuMemcpyDtoDAsync(
            instDataBuffer[1].getCUdeviceptr(), instDataBuffer[0].getCUdeviceptr(),
            instDataBuffer[1].sizeInBytes(), cuStream));

        CUDADRV_CHECK(cuStreamSynchronize(cuStream));
    }

    void setupLightInstDistribution(
        CUstream cuStream, CUdeviceptr lightInstDistAddr, uint32_t instBufferIndex) {
        shared::LightDistribution dLightInstDist;
        lightInstDist.getDeviceType(&dLightInstDist);
        CUDADRV_CHECK(cuMemcpyHtoDAsync(
            lightInstDistAddr, &dLightInstDist, sizeof(dLightInstDist), cuStream));

        uint32_t numInsts = static_cast<uint32_t>(insts.size());
#if USE_PROBABILITY_TEXTURE
        uint2 dims = shared::computeProbabilityTextureDimentions(numInsts);
        uint32_t numMipLevels = nextPowOf2Exponent(dims.x) + 1;
        computeProbTex.computeInstProbTexture(
            cuStream, computeProbTex.computeInstProbTexture.calcGridDim(dims.x * dims.y),
            lightInstDistAddr, numInsts,
            instDataBuffer[instBufferIndex].getDevicePointer(),
            lightInstDist.getSurfaceObject(0));

        uint2 curDims = dims;
        for (int dstMipLevel = 1; dstMipLevel < numMipLevels; ++dstMipLevel) {
            curDims = (curDims + uint2(1, 1)) / 2;
            computeProbTex.computeMip(
                cuStream, computeProbTex.computeMip.calcGridDim(curDims.x, curDims.y),
                lightInstDistAddr, dstMipLevel,
                lightInstDist.getSurfaceObject(dstMipLevel - 1),
                lightInstDist.getSurfaceObject(dstMipLevel));
        }

        //CUDADRV_CHECK(cuStreamSynchronize(cuStream));
        //CUDADRV_CHECK(cuMemcpyDtoH(&lightInstDist, probTeXAddr, sizeof(lightInstDist)));
        //auto values = lightInstDistArray.map<float>(numMipLevels - 1);
        //hpprintf("%g\n", values[0]);
        //lightInstDistArray.unmap(numMipLevels - 1);
#else
        computeProbTex.computeInstProbBuffer(
            cuStream, computeProbTex.computeInstProbBuffer.calcGridDim(numInsts),
            lightInstDistAddr, numInsts,
            instDataBuffer[instBufferIndex].getDevicePointer());
        //CUDADRV_CHECK(cuStreamSynchronize(cuStream));

        size_t scratchMemSize = scanScratchMem.sizeInBytes();
        CUDADRV_CHECK(cubd::DeviceScan::ExclusiveSum(
            scanScratchMem.getDevicePointer(), scratchMemSize,
            lightInstDist.weightsOnDevice(),
            lightInstDist.cdfOnDevice(),
            numInsts, cuStream));
        //CUDADRV_CHECK(cuStreamSynchronize(cuStream));

        computeProbTex.finalizeDiscreteDistribution1D(
            cuStream, computeProbTex.finalizeDiscreteDistribution1D.calcGridDim(1),
            lightInstDistAddr);
        //CUDADRV_CHECK(cuStreamSynchronize(cuStream));

        //CUDADRV_CHECK(cuStreamSynchronize(cuStream));
        //shared::LightDistribution lightDistOnHost;
        //CUDADRV_CHECK(cuMemcpyDtoH(
        //    &lightDistOnHost, lightInstDistAddr,
        //    sizeof(lightDistOnHost)));
        //hpprintf("%g\n", lightDistOnHost.integral());

        //std::vector<float> weights(lightDistOnHost.m_numValues);
        //std::vector<float> CDF(lightDistOnHost.m_numValues);
        //CUDADRV_CHECK(cuMemcpyDtoH(
        //    weights.data(), reinterpret_cast<CUdeviceptr>(lightDistOnHost.m_weights),
        //    sizeof(float) * lightDistOnHost.m_numValues));
        //CUDADRV_CHECK(cuMemcpyDtoH(
        //    CDF.data(), reinterpret_cast<CUdeviceptr>(lightDistOnHost.m_CDF),
        //    sizeof(float) * lightDistOnHost.m_numValues));

        //{
        //    uint32_t m_numValues = lightDistOnHost.m_numValues;
        //    for (int i = 0; i < m_numValues; ++i) {
        //        hpprintf("%4u: %g, %g\n", i, weights[i], CDF[i]);
        //    }
        //    hpprintf("\n");

        //    float m_integral = lightDistOnHost.m_integral;
        //    uint32_t uu = 0x3f485b70;
        //    //uint32_t uu = 0x3f3ed174;
        //    float u = /*0.782645f*/*(float*)&uu;
        //    u *= m_integral;
        //    int idx = 0;
        //    for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
        //        if (idx + d >= m_numValues)
        //            continue;
        //        if (CDF[idx + d] <= u)
        //            idx += d;
        //    }
        //    hpprintf("");
        //}
        //hpprintf("");
#endif
    }

    void draw() const {
        for (int instIdx = 0; instIdx < insts.size(); ++instIdx) {
            const Instance* inst = insts[instIdx];
            inst->draw();
        }
    }
};

void finalizeTextureCaches();

template <typename T>
void createImmTexture(
    CUcontext cuContext,
    const T &immValue,
    bool isNormalized,
    const cudau::Array** texture,
    const glu::Texture2D** gfxTexture = nullptr);

template <typename T, bool useSurface = false>
bool loadTexture(
    const std::filesystem::path &filePath, const T &fallbackValue,
    CUcontext cuContext,
    const cudau::Array** texture,
    bool* needsDegamma,
    bool* isHDR = nullptr);

bool loadNormalTexture(
    const std::filesystem::path &filePath,
    CUcontext cuContext,
    const cudau::Array** texture,
    const glu::Texture2D** gfxTexture,
    BumpMapTextureType* bumpMapType);

void createNormalTexture(
    CUcontext cuContext,
    const std::filesystem::path &normalPath,
    Material* mat);

void createEmittanceTexture(
    CUcontext cuContext,
    const std::filesystem::path &emittancePath, const RGB &immEmittance,
    Material* mat,
    bool* needsDegamma, bool* isHDR);

void createLambertMaterial(
    CUcontext cuContext, Scene* scene,
    const std::filesystem::path &reflectancePath, const RGB &immReflectance,
    const std::filesystem::path &normalPath,
    const std::filesystem::path &emittancePath, const RGB &immEmittance);

void createDiffuseAndSpecularMaterial(
    CUcontext cuContext, Scene* scene,
    const std::filesystem::path &diffuseColorPath, const RGB &immDiffuseColor,
    const std::filesystem::path &specularColorPath, const RGB &immSpecularColor,
    float immSmoothness,
    const std::filesystem::path &normalPath,
    const std::filesystem::path &emittancePath, const RGB &immEmittance);

void createSimplePBRMaterial(
    CUcontext cuContext, Scene* scene,
    const std::filesystem::path &baseColor_opacityPath, const float4 &immBaseColor_opacity,
    const std::filesystem::path &occlusion_roughness_metallicPath,
    const float3 &immOcclusion_roughness_metallic,
    const std::filesystem::path &normalPath,
    const std::filesystem::path &emittancePath, const RGB &immEmittance);

GeometryInstance* createGeometryInstance(
    CUcontext cuContext, Scene* scene,
    const std::vector<shared::Vertex> &vertices,
    const std::vector<shared::Triangle> &triangles,
    const Material* mat, optixu::Material optixMat,
    bool allocateGfxResource);

GeometryInstance* createTFDMGeometryInstance(
    CUcontext cuContext, Scene* scene,
    const std::vector<shared::Vertex> &vertices,
    const std::vector<shared::Triangle> &triangles,
    const Material* mat, optixu::Material optixMat);

GeometryGroup* createGeometryGroup(
    Scene* scene,
    const std::set<const GeometryInstance*> &geomInsts);

void createTriangleMeshes(
    const std::string &meshName,
    const std::filesystem::path &filePath,
    MaterialConvention matConv,
    const Matrix4x4 &preTransform,
    CUcontext cuContext, Scene* scene, optixu::Material optixMat,
    bool allocateGfxResource = false);

void createRectangleLight(
    const std::string &meshName,
    float width, float depth,
    const RGB &reflectance,
    const std::filesystem::path &emittancePath,
    const RGB &immEmittance,
    const Matrix4x4 &transform,
    CUcontext cuContext, Scene* scene, optixu::Material optixMat,
    bool allocateGfxResource = false);

void createSphereLight(
    const std::string &meshName,
    float radius,
    const RGB &reflectance,
    const std::filesystem::path &emittancePath,
    const RGB &immEmittance,
    const Point3D &position,
    CUcontext cuContext, Scene* scene, optixu::Material optixMat,
    bool allocateGfxResource = false);

Instance* createInstance(
    CUcontext cuContext, Scene* scene,
    const Mesh::GeometryGroupInstance &geomGroupInst,
    const Matrix4x4 &transform);

void loadEnvironmentalTexture(
    const std::filesystem::path &filePath,
    CUcontext cuContext,
    cudau::Array* envLightArray, CUtexObject* envLightTexture,
    RegularConstantContinuousDistribution2D* envLightImportanceMap);



void saveImage(const std::filesystem::path &filepath, uint32_t width, uint32_t height, const uint32_t* data);
void saveImageHDR(
    const std::filesystem::path &filepath, uint32_t width, uint32_t height,
    float brightnessScale,
    const float* data, bool flipY = false);
void saveImageHDR(
    const std::filesystem::path &filepath, uint32_t width, uint32_t height,
    float brightnessScale,
    const float4* data, bool flipY = false);

struct SDRImageSaverConfig {
    float alphaForOverride;
    float brightnessScale;
    unsigned int applyToneMap : 1;
    unsigned int apply_sRGB_gammaCorrection : 1;
    unsigned int flipY : 1;

    SDRImageSaverConfig() :
        brightnessScale(1.0f),
        applyToneMap(false), apply_sRGB_gammaCorrection(false),
        flipY(false),
        alphaForOverride(-1) {}
};

void saveImage(
    const std::filesystem::path &filepath, uint32_t width, uint32_t height, const float4* data,
    const SDRImageSaverConfig &config);

void saveImage(
    const std::filesystem::path &filepath,
    uint32_t width, cudau::TypedBuffer<float4> &buffer,
    const SDRImageSaverConfig &config);

void saveImage(
    const std::filesystem::path &filepath,
    cudau::Array &array,
    const SDRImageSaverConfig &config);

template <uint32_t log2BlockWidth>
void saveImage(
    const std::filesystem::path &filepath,
    optixu::HostBlockBuffer2D<float4, log2BlockWidth> &buffer,
    const SDRImageSaverConfig &config) {
    uint32_t width = buffer.getWidth();
    uint32_t height = buffer.getHeight();
    auto data = new float4[width * height];
    buffer.map();
    for (int y = 0; y < static_cast<int32_t>(height); ++y) {
        for (int x = 0; x < static_cast<int32_t>(width); ++x) {
            data[y * width + x] = buffer(x, y);
        }
    }
    buffer.unmap();
    saveImage(filepath, width, height, data, config);
    delete[] data;
}
