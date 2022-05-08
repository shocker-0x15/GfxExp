#pragma once

#include "common_shared.h"

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

#include "stopwatch.h"

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

std::string readTxtFile(const std::filesystem::path& filepath);



template <typename RealType>
class DiscreteDistribution1DTemplate {
    cudau::TypedBuffer<RealType> m_PMF;
#if defined(USE_WALKER_ALIAS_METHOD)
    cudau::TypedBuffer<shared::AliasTableEntry<RealType>> m_aliasTable;
    cudau::TypedBuffer<shared::AliasValueMap<RealType>> m_valueMaps;
#else
    cudau::TypedBuffer<RealType> m_CDF;
#endif
    RealType m_integral;
    uint32_t m_numValues;

public:
    void initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues);
    void finalize() {
#if defined(USE_WALKER_ALIAS_METHOD)
        if (m_valueMaps.isInitialized() && m_aliasTable.isInitialized() && m_PMF.isInitialized()) {
            m_valueMaps.finalize();
            m_aliasTable.finalize();
            m_PMF.finalize();
        }
#else
        if (m_CDF.isInitialized() && m_PMF.isInitialized()) {
            m_CDF.finalize();
            m_PMF.finalize();
        }
#endif
    }

    DiscreteDistribution1DTemplate &operator=(DiscreteDistribution1DTemplate &&v) {
        m_PMF = std::move(v.m_PMF);
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

    void getDeviceType(shared::DiscreteDistribution1DTemplate<RealType>* instance) const {
#if defined(USE_WALKER_ALIAS_METHOD)
        new (instance) shared::DiscreteDistribution1DTemplate<RealType>(
            m_PMF.isInitialized() ? m_PMF.getDevicePointer() : nullptr,
            m_aliasTable.isInitialized() ? m_aliasTable.getDevicePointer() : nullptr,
            m_valueMaps.isInitialized() ? m_valueMaps.getDevicePointer() : nullptr,
            m_integral, m_numValues);
#else
        new (instance) shared::DiscreteDistribution1DTemplate<RealType>(
            m_PMF.isInitialized() ? m_PMF.getDevicePointer() : nullptr,
            m_CDF.isInitialized() ? m_CDF.getDevicePointer() : nullptr,
            m_integral, m_numValues);
#endif
    }
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

public:
    void initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues);
    void finalize(CUcontext cuContext) {
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

public:
    RegularConstantContinuousDistribution2DTemplate() : m_1DDists(nullptr) {}

    RegularConstantContinuousDistribution2DTemplate &operator=(RegularConstantContinuousDistribution2DTemplate &&v) {
        m_raw1DDists = std::move(v.m_raw1DDists);
        m_1DDists = std::move(v.m_1DDists);
        m_top1DDist = std::move(v.m_top1DDist);
        return *this;
    }

    void initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numD1, size_t numD2);
    void finalize(CUcontext cuContext) {
        m_top1DDist.finalize(cuContext);

        for (int i = m_top1DDist.getNumValues() - 1; i >= 0; --i) {
            m_1DDists[i].finalize(cuContext);
        }

        m_raw1DDists.finalize();
        delete[] m_1DDists;
        m_1DDists = nullptr;
    }

    bool isInitialized() const { return m_1DDists != nullptr; }

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
    const cudau::Array* emittance;
    CUtexObject texEmittance;
    uint32_t materialSlot;

    Material() :
        normal(nullptr), texNormal(0),
        emittance(nullptr), texEmittance(0),
        materialSlot(0) {}
};

struct GeometryInstance {
    const Material* mat;

    cudau::TypedBuffer<shared::Vertex> vertexBuffer;
    cudau::TypedBuffer<shared::Triangle> triangleBuffer;
    cudau::Array emitterPrimDist;
    CUtexObject emitterPrimDistTex;
    uint32_t geomInstSlot;
    optixu::GeometryInstance optixGeomInst;
    AABB aabb;
};

struct GeometryGroup {
    std::set<const GeometryInstance*> geomInsts;

    optixu::GeometryAccelerationStructure optixGas;
    cudau::Buffer optixGasMem;
    uint32_t numEmitterPrimitives;
    AABB aabb;
};

struct Mesh {
    struct Group {
        const GeometryGroup* geomGroup;
        Matrix4x4 transform;
    };
    std::vector<Group> groups;
};

struct Instance {
    const GeometryGroup* geomGroup;

    cudau::TypedBuffer<uint32_t> geomInstSlots;
    cudau::Array lightGeomInstDist;
    CUtexObject lightGeomInstDistTex;
    uint32_t instSlot;
    optixu::Instance optixInst;
};

struct InstanceController {
    Instance* inst;
    Matrix4x4 defaultTransform;

    float curScale;
    Quaternion curOrientation;
    float3 curPosition;
    Matrix4x4 prevMatM2W;
    Matrix4x4 matM2W;
    Matrix3x3 nMatM2W;

    float beginScale;
    Quaternion beginOrientation;
    float3 beginPosition;
    float endScale;
    Quaternion endOrientation;
    float3 endPosition;
    float time;
    float frequency;

    InstanceController(
        Instance* _inst, const Matrix4x4 &_defaultTranform,
        float _beginScale, const Quaternion &_beginOrienatation, const float3 &_beginPosition,
        float _endScale, const Quaternion &_endOrienatation, const float3 &_endPosition,
        float _frequency, float initTime) :
        inst(_inst), defaultTransform(_defaultTranform),
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
        prevMatM2W = matM2W;
        updateBody(dt);
        matM2W = Matrix4x4(curOrientation.toMatrix3x3() * scale3x3(curScale), curPosition) * defaultTransform;
        nMatM2W = transpose(inverse(matM2W.getUpperLeftMatrix()));

        Matrix4x4 tMatM2W = transpose(matM2W);
        inst->optixInst.setTransform(reinterpret_cast<const float*>(&tMatM2W));

        shared::InstanceData &instData = instDataBuffer[inst->instSlot];
        instData.prevTransform = prevMatM2W;
        instData.transform = matM2W;
        instData.normalMatrix = nMatM2W;
    }
};

// TODO: シーン読み込み周りをもっと綺麗にする。
struct Scene {
    static constexpr cudau::BufferType bufferType = cudau::BufferType::Device;

    static constexpr uint32_t maxNumMaterials = 1024;
    static constexpr uint32_t maxNumGeometryInstances = 65536;
    static constexpr uint32_t maxNumInstances = 16384;

    optixu::Scene optixScene;
    uint32_t numRayTypes;
    optixu::Material optixDefaultMaterial;

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

    cudau::Array lightInstDistArray;
    CUtexObject lightInstDistTex;
    uint2 lightInstDistTexDims;

    optixu::InstanceAccelerationStructure ias;
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> iasInstanceBuffer;

    cudau::Buffer asScratchMem;

    size_t hitGroupSbtSize;

    void initialize(
        CUcontext cuContext, optixu::Context optixContext,
        uint32_t _numRayTypes, optixu::Material material) {
        optixScene = optixContext.createScene();
        numRayTypes = _numRayTypes;

        optixDefaultMaterial = material;

        materialSlotFinder.initialize(maxNumMaterials);
        geomInstSlotFinder.initialize(maxNumGeometryInstances);
        instSlotFinder.initialize(maxNumInstances);

        materialDataBuffer.initialize(cuContext, Scene::bufferType, maxNumMaterials);
        geomInstDataBuffer.initialize(cuContext, Scene::bufferType, maxNumGeometryInstances);
        instDataBuffer[0].initialize(cuContext, Scene::bufferType, maxNumInstances);
        instDataBuffer[1].initialize(cuContext, Scene::bufferType, maxNumInstances);

        ias = optixScene.createInstanceAccelerationStructure();

        cudau::TextureSampler sampler;
        sampler.setXyFilterMode(cudau::TextureFilterMode::Point);
        sampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
        sampler.setReadMode(cudau::TextureReadMode::ElementType);

        lightInstDistTexDims = shared::computeProbabilityTextureDimentions(maxNumInstances);
        uint32_t numMipLevels = nextPowOf2Exponent(lightInstDistTexDims.x) + 1;
        lightInstDistArray.initialize2D(
            cuContext, cudau::ArrayElementType::Float32, 1,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            lightInstDistTexDims.x, lightInstDistTexDims.y, numMipLevels);
        lightInstDistTex = sampler.createTextureObject(lightInstDistArray);
    }

    void finalize() {
        if (lightInstDistTex)
            CUDADRV_CHECK(cuTexObjectDestroy(lightInstDistTex));
        lightInstDistArray.finalize();

        asScratchMem.finalize();
        iasInstanceBuffer.finalize();
        iasMem.finalize();
        for (int i = geomGroups.size() - 1; i >= 0; --i) {
            GeometryGroup* geomGroup = geomGroups[i];
            geomGroup->optixGasMem.finalize();
        }
        ias.destroy();

        for (int i = insts.size() - 1; i >= 0; --i) {
            Instance* inst = insts[i];
            inst->optixInst.destroy();
            if (inst->lightGeomInstDistTex)
                CUDADRV_CHECK(cuTexObjectDestroy(inst->lightGeomInstDistTex));
            inst->lightGeomInstDist.finalize();
        }
        for (int i = geomGroups.size() - 1; i >= 0; --i) {
            GeometryGroup* geomGroup = geomGroups[i];
            geomGroup->optixGas.destroy();
        }
        for (int i = geomInsts.size() - 1; i >= 0; --i) {
            GeometryInstance* geomInst = geomInsts[i];
            geomInst->optixGeomInst.destroy();
            if (geomInst->emitterPrimDistTex)
                CUDADRV_CHECK(cuTexObjectDestroy(geomInst->emitterPrimDistTex));
            geomInst->emitterPrimDist.finalize();
            geomInst->triangleBuffer.finalize();
            geomInst->vertexBuffer.finalize();
        }
        for (int i = materials.size() - 1; i >= 0; --i) {
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

    void setupASes(CUcontext cuContext) {
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
                false, false, false);
            geomGroup->optixGas.prepareForBuild(&asSizes);
            geomGroup->optixGasMem.initialize(cuContext, Scene::bufferType, asSizes.outputSizeInBytes, 1);
            asScratchSize = std::max(asSizes.tempSizeInBytes, asScratchSize);
        }

        ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false, false);
        ias.prepareForBuild(&asSizes);
        iasMem.initialize(cuContext, Scene::bufferType, asSizes.outputSizeInBytes, 1);
        asScratchSize = std::max(asSizes.tempSizeInBytes, asScratchSize);
        iasInstanceBuffer.initialize(cuContext, Scene::bufferType, ias.getNumChildren());

        asScratchMem.initialize(cuContext, Scene::bufferType, asScratchSize, 1);

        for (int i = 0; i < geomGroups.size(); ++i) {
            const GeometryGroup* geomGroup = geomGroups[i];
            geomGroup->optixGas.rebuild(0, geomGroup->optixGasMem, asScratchMem);
        }

        optixScene.generateShaderBindingTableLayout(&hitGroupSbtSize);

        {
            for (int bufIdx = 0; bufIdx < 2; ++bufIdx) {
                cudau::TypedBuffer<shared::InstanceData> &curInstDataBuffer = instDataBuffer[bufIdx];
                shared::InstanceData* instDataBufferOnHost = curInstDataBuffer.map();
                for (int i = 0; i < instControllers.size(); ++i) {
                    InstanceController* controller = instControllers[i];
                    Instance* inst = controller->inst;
                    shared::InstanceData &instData = instDataBufferOnHost[inst->instSlot];
                    controller->update(instDataBufferOnHost, 0.0f);
                    // TODO: まとめて送る。
                    CUDADRV_CHECK(cuMemcpyHtoDAsync(curInstDataBuffer.getCUdeviceptrAt(inst->instSlot),
                                                    &instData, sizeof(instData), 0));
                }
                curInstDataBuffer.unmap();
            }
        }

        ias.rebuild(0, iasInstanceBuffer, iasMem, asScratchMem);
    }
};

void finalizeTextureCaches();

void createTriangleMeshes(
    const std::string &meshName,
    const std::filesystem::path &filePath,
    MaterialConvention matConv,
    const Matrix4x4 &preTransform,
    CUcontext cuContext, Scene* scene);

void createRectangleLight(
    const std::string &meshName,
    float width, float depth,
    const float3 &reflectance,
    const std::filesystem::path &emittancePath,
    const float3 &immEmittance,
    const Matrix4x4 &transform,
    CUcontext cuContext, Scene* scene);

void createSphereLight(
    const std::string &meshName,
    float radius,
    const float3 &reflectance,
    const std::filesystem::path &emittancePath,
    const float3 &immEmittance,
    const float3 &position,
    CUcontext cuContext, Scene* scene);

Instance* createInstance(
    CUcontext cuContext, Scene* scene,
    const GeometryGroup* geomGroup,
    const Matrix4x4 &transform);

void loadEnvironmentalTexture(
    const std::filesystem::path &filePath,
    CUcontext cuContext,
    cudau::Array* envLightArray, CUtexObject* envLightTexture,
    RegularConstantContinuousDistribution2D* envLightImportanceMap);



void saveImage(const std::filesystem::path &filepath, uint32_t width, uint32_t height, const uint32_t* data);
void saveImageHDR(const std::filesystem::path &filepath, uint32_t width, uint32_t height,
                  float brightnessScale,
                  const float4* data);

void saveImage(const std::filesystem::path &filepath, uint32_t width, uint32_t height, const float4* data,
               float brightnessScale,
               bool applyToneMap, bool apply_sRGB_gammaCorrection);

void saveImage(const std::filesystem::path &filepath,
               uint32_t width, cudau::TypedBuffer<float4> &buffer,
               float brightnessScale,
               bool applyToneMap, bool apply_sRGB_gammaCorrection);

void saveImage(const std::filesystem::path &filepath,
               cudau::Array &array,
               float brightnessScale,
               bool applyToneMap, bool apply_sRGB_gammaCorrection);

template <uint32_t log2BlockWidth>
void saveImage(const std::filesystem::path &filepath,
               optixu::HostBlockBuffer2D<float4, log2BlockWidth> &buffer,
               float brightnessScale,
               bool applyToneMap, bool apply_sRGB_gammaCorrection) {
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
    saveImage(filepath, width, height, data,
              brightnessScale,
              applyToneMap, apply_sRGB_gammaCorrection);
    delete[] data;
}
