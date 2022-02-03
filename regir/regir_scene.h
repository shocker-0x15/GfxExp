#pragma once

#include "regir_shared.h"
#include "../common/common_host.h"

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include "../common/dds_loader.h"
#include "../../ext/stb_image.h"
#include "../../ext/tinyexr.h"



struct GPUEnvironment {
    static constexpr cudau::BufferType bufferType = cudau::BufferType::Device;
    static constexpr uint32_t maxNumMaterials = 1024;
    static constexpr uint32_t maxNumGeometryInstances = 65536;
    static constexpr uint32_t maxNumInstances = 16384;

    CUcontext cuContext;
    optixu::Context optixContext;

    CUmodule cellBuilderModule;
    cudau::Kernel kernelBuildCellReservoirs;
    cudau::Kernel kernelBuildCellReservoirsAndTemporalReuse;
    cudau::Kernel kernelUpdateLastAccessFrameIndices;
    CUdeviceptr plpPtr;

    optixu::Pipeline pipeline;
    optixu::Module mainModule;
    optixu::ProgramGroup emptyMissProgram;
    optixu::ProgramGroup setupGBuffersRayGenProgram;
    optixu::ProgramGroup setupGBuffersHitProgramGroup;
    optixu::ProgramGroup setupGBuffersMissProgram;

    optixu::ProgramGroup pathTraceBaselineRayGenProgram;
    optixu::ProgramGroup pathTraceBaselineMissProgram;
    optixu::ProgramGroup pathTraceBaselineHitProgramGroup;
    optixu::ProgramGroup pathTraceRegirRayGenProgram;
    optixu::ProgramGroup pathTraceRegirHitProgramGroup;
    optixu::ProgramGroup visibilityHitProgramGroup;
    std::vector<optixu::ProgramGroup> callablePrograms;

    cudau::Buffer shaderBindingTable;

    optixu::Material defaultMaterial;

    optixu::Scene scene;

    SlotFinder materialSlotFinder;
    SlotFinder geomInstSlotFinder;
    SlotFinder instSlotFinder;
    cudau::TypedBuffer<shared::MaterialData> materialDataBuffer;
    cudau::TypedBuffer<shared::GeometryInstanceData> geomInstDataBuffer;
    cudau::TypedBuffer<shared::InstanceData> instDataBuffer[2];

    void initialize() {
        int32_t cuDeviceCount;
        CUDADRV_CHECK(cuInit(0));
        CUDADRV_CHECK(cuDeviceGetCount(&cuDeviceCount));
        CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
        CUDADRV_CHECK(cuCtxSetCurrent(cuContext));

        optixContext = optixu::Context::create(cuContext/*, 4, DEBUG_SELECT(true, false)*/);

        CUDADRV_CHECK(cuModuleLoad(&cellBuilderModule, (getExecutableDirectory() / "regir/ptxes/build_cell_reservoirs.ptx").string().c_str()));
        kernelBuildCellReservoirs =
            cudau::Kernel(cellBuilderModule, "buildCellReservoirs", cudau::dim3(32), 0);
        kernelBuildCellReservoirsAndTemporalReuse =
            cudau::Kernel(cellBuilderModule, "buildCellReservoirsAndTemporalReuse", cudau::dim3(32), 0);
        kernelUpdateLastAccessFrameIndices =
            cudau::Kernel(cellBuilderModule, "updateLastAccessFrameIndices", cudau::dim3(32), 0);

        size_t plpSize;
        CUDADRV_CHECK(cuModuleGetGlobal(&plpPtr, &plpSize, cellBuilderModule, "plp"));
        Assert(sizeof(shared::PipelineLaunchParameters) == plpSize, "Unexpected plp size.");

        pipeline = optixContext.createPipeline();

        // JP: このサンプルでは2段階のAS(1段階のインスタンシング)を使用する。
        // EN: This sample uses two-level AS (single-level instancing).
        pipeline.setPipelineOptions(
            std::max({
                optixu::calcSumDwords<PrimaryRayPayloadSignature>(),
                optixu::calcSumDwords<VisibilityRayPayloadSignature>(),
                optixu::calcSumDwords<PathTraceRayPayloadSignature>()
                     }),
            optixu::calcSumDwords<float2>(),
            "plp", sizeof(shared::PipelineLaunchParameters),
            false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
            OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
            DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE),
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

        const std::string ptx = readTxtFile(getExecutableDirectory() / "regir/ptxes/optix_kernels.ptx");
        mainModule = pipeline.createModuleFromPTXString(
            ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
            DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        optixu::Module emptyModule;

        emptyMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

        setupGBuffersRayGenProgram = pipeline.createRayGenProgram(
            mainModule, RT_RG_NAME_STR("setupGBuffers"));
        setupGBuffersHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
            mainModule, RT_CH_NAME_STR("setupGBuffers"),
            emptyModule, nullptr);
        setupGBuffersMissProgram = pipeline.createMissProgram(
            mainModule, RT_MS_NAME_STR("setupGBuffers"));

        pathTraceBaselineRayGenProgram = pipeline.createRayGenProgram(
            mainModule, RT_RG_NAME_STR("pathTraceBaseline"));
        pathTraceBaselineMissProgram = pipeline.createMissProgram(
            mainModule, RT_MS_NAME_STR("pathTraceBaseline"));
        pathTraceBaselineHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
            mainModule, RT_CH_NAME_STR("pathTraceBaseline"),
            emptyModule, nullptr);

        pathTraceRegirRayGenProgram = pipeline.createRayGenProgram(
            mainModule, RT_RG_NAME_STR("pathTraceRegir"));
        pathTraceRegirHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
            mainModule, RT_CH_NAME_STR("pathTraceRegir"),
            emptyModule, nullptr);

        visibilityHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
            emptyModule, nullptr,
            mainModule, RT_AH_NAME_STR("visibility"));

        //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");

        // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
        //pipeline.setExceptionProgram(exceptionProgram);
        pipeline.setNumMissRayTypes(shared::NumRayTypes);
        pipeline.setMissProgram(shared::RayType_Primary, setupGBuffersMissProgram);
        pipeline.setMissProgram(shared::RayType_PathTraceBaseline, pathTraceBaselineMissProgram);
        pipeline.setMissProgram(shared::RayType_PathTraceReGIR, emptyMissProgram);
        pipeline.setMissProgram(shared::RayType_Visibility, emptyMissProgram);

        pipeline.setNumCallablePrograms(NumCallablePrograms);
        callablePrograms.resize(NumCallablePrograms);
        for (int i = 0; i < NumCallablePrograms; ++i) {
            optixu::ProgramGroup program = pipeline.createCallableProgramGroup(
                mainModule, callableProgramEntryPoints[i],
                emptyModule, nullptr);
            callablePrograms[i] = program;
            pipeline.setCallableProgram(i, program);
        }

        pipeline.link(2, DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));



        size_t sbtSize;
        pipeline.generateShaderBindingTableLayout(&sbtSize);
        shaderBindingTable.initialize(cuContext, GPUEnvironment::bufferType, sbtSize, 1);
        shaderBindingTable.setMappedMemoryPersistent(true);
        pipeline.setShaderBindingTable(shaderBindingTable, shaderBindingTable.getMappedPointer());



        defaultMaterial = optixContext.createMaterial();
        defaultMaterial.setHitGroup(shared::RayType_Primary, setupGBuffersHitProgramGroup);
        defaultMaterial.setHitGroup(shared::RayType_PathTraceBaseline, pathTraceBaselineHitProgramGroup);
        defaultMaterial.setHitGroup(shared::RayType_PathTraceReGIR, pathTraceRegirHitProgramGroup);
        defaultMaterial.setHitGroup(shared::RayType_Visibility, visibilityHitProgramGroup);



        scene = optixContext.createScene();



        materialSlotFinder.initialize(maxNumMaterials);
        geomInstSlotFinder.initialize(maxNumGeometryInstances);
        instSlotFinder.initialize(maxNumInstances);

        materialDataBuffer.initialize(cuContext, bufferType, maxNumMaterials);
        geomInstDataBuffer.initialize(cuContext, bufferType, maxNumGeometryInstances);
        instDataBuffer[0].initialize(cuContext, bufferType, maxNumInstances);
        instDataBuffer[1].initialize(cuContext, bufferType, maxNumInstances);
    }

    void finalize() {
        instDataBuffer[1].finalize();
        instDataBuffer[0].finalize();
        geomInstDataBuffer.finalize();
        materialDataBuffer.finalize();

        instSlotFinder.finalize();
        geomInstSlotFinder.finalize();
        materialSlotFinder.finalize();

        scene.destroy();

        defaultMaterial.destroy();

        shaderBindingTable.finalize();

        for (int i = 0; i < NumCallablePrograms; ++i)
            callablePrograms[i].destroy();
        visibilityHitProgramGroup.destroy();
        pathTraceRegirHitProgramGroup.destroy();
        pathTraceRegirRayGenProgram.destroy();
        pathTraceBaselineHitProgramGroup.destroy();
        pathTraceBaselineMissProgram.destroy();
        pathTraceBaselineRayGenProgram.destroy();
        setupGBuffersMissProgram.destroy();
        setupGBuffersHitProgramGroup.destroy();
        setupGBuffersRayGenProgram.destroy();
        emptyMissProgram.destroy();
        mainModule.destroy();

        pipeline.destroy();

        CUDADRV_CHECK(cuModuleUnload(cellBuilderModule));

        optixContext.destroy();

        CUDADRV_CHECK(cuCtxDestroy(cuContext));
    }
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
    DiscreteDistribution1D emitterPrimDist;
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

struct Instance {
    const GeometryGroup* geomGroup;

    cudau::TypedBuffer<uint32_t> geomInstSlots;
    DiscreteDistribution1D lightGeomInstDist;
    uint32_t instSlot;
    optixu::Instance optixInst;
};

struct Mesh {
    struct Group {
        const GeometryGroup* geomGroup;
        Matrix4x4 transform;
    };
    std::vector<Group> groups;
};

struct FlattenedNode {
    Matrix4x4 transform;
    std::vector<uint32_t> meshIndices;
};

static void computeFlattenedNodes(const aiScene* scene, const Matrix4x4 &parentXfm, const aiNode* curNode,
                                  std::vector<FlattenedNode> &flattenedNodes) {
    aiMatrix4x4 curAiXfm = curNode->mTransformation;
    Matrix4x4 curXfm = Matrix4x4(float4(curAiXfm.a1, curAiXfm.a2, curAiXfm.a3, curAiXfm.a4),
                                 float4(curAiXfm.b1, curAiXfm.b2, curAiXfm.b3, curAiXfm.b4),
                                 float4(curAiXfm.c1, curAiXfm.c2, curAiXfm.c3, curAiXfm.c4),
                                 float4(curAiXfm.d1, curAiXfm.d2, curAiXfm.d3, curAiXfm.d4));
    FlattenedNode flattenedNode;
    flattenedNode.transform = parentXfm * transpose(curXfm);
    flattenedNode.meshIndices.resize(curNode->mNumMeshes);
    if (curNode->mNumMeshes > 0) {
        std::copy_n(curNode->mMeshes, curNode->mNumMeshes, flattenedNode.meshIndices.data());
        flattenedNodes.push_back(flattenedNode);
    }

    for (int cIdx = 0; cIdx < curNode->mNumChildren; ++cIdx)
        computeFlattenedNodes(scene, flattenedNode.transform, curNode->mChildren[cIdx], flattenedNodes);
}

static void translate(dds::Format ddsFormat, cudau::ArrayElementType* cudaType, bool* needsDegamma) {
    *needsDegamma = false;
    switch (ddsFormat) {
    case dds::Format::BC1_UNorm:
        *cudaType = cudau::ArrayElementType::BC1_UNorm;
        break;
    case dds::Format::BC1_UNorm_sRGB:
        *cudaType = cudau::ArrayElementType::BC1_UNorm;
        *needsDegamma = true;
        break;
    case dds::Format::BC2_UNorm:
        *cudaType = cudau::ArrayElementType::BC2_UNorm;
        break;
    case dds::Format::BC2_UNorm_sRGB:
        *cudaType = cudau::ArrayElementType::BC2_UNorm;
        *needsDegamma = true;
        break;
    case dds::Format::BC3_UNorm:
        *cudaType = cudau::ArrayElementType::BC3_UNorm;
        break;
    case dds::Format::BC3_UNorm_sRGB:
        *cudaType = cudau::ArrayElementType::BC3_UNorm;
        *needsDegamma = true;
        break;
    case dds::Format::BC4_UNorm:
        *cudaType = cudau::ArrayElementType::BC4_UNorm;
        break;
    case dds::Format::BC4_SNorm:
        *cudaType = cudau::ArrayElementType::BC4_SNorm;
        break;
    case dds::Format::BC5_UNorm:
        *cudaType = cudau::ArrayElementType::BC5_UNorm;
        break;
    case dds::Format::BC5_SNorm:
        *cudaType = cudau::ArrayElementType::BC5_SNorm;
        break;
    case dds::Format::BC6H_UF16:
        *cudaType = cudau::ArrayElementType::BC6H_UF16;
        break;
    case dds::Format::BC6H_SF16:
        *cudaType = cudau::ArrayElementType::BC6H_SF16;
        break;
    case dds::Format::BC7_UNorm:
        *cudaType = cudau::ArrayElementType::BC7_UNorm;
        break;
    case dds::Format::BC7_UNorm_sRGB:
        *cudaType = cudau::ArrayElementType::BC7_UNorm;
        *needsDegamma = true;
        break;
    default:
        break;
    }
};

enum class BumpMapTextureType {
    NormalMap = 0,
    NormalMap_BC,
    NormalMap_BC_2ch,
    HeightMap,
    HeightMap_BC,
};

struct TextureCacheKey {
    std::filesystem::path filePath;
    CUcontext cuContext;

    bool operator<(const TextureCacheKey &rKey) const {
        if (filePath < rKey.filePath)
            return true;
        else if (filePath > rKey.filePath)
            return false;
        if (cuContext < rKey.cuContext)
            return true;
        else if (cuContext > rKey.cuContext)
            return false;
        return false;
    }
};

struct Fx1ImmTextureCacheKey {
    float immValue;
    CUcontext cuContext;

    bool operator<(const Fx1ImmTextureCacheKey &rKey) const {
        if (immValue < rKey.immValue)
            return true;
        else if (immValue > rKey.immValue)
            return false;
        if (cuContext < rKey.cuContext)
            return true;
        else if (cuContext > rKey.cuContext)
            return false;
        return false;
    }
};

struct Fx3ImmTextureCacheKey {
    float3 immValue;
    CUcontext cuContext;

    bool operator<(const Fx3ImmTextureCacheKey &rKey) const {
        if (immValue.z < rKey.immValue.z)
            return true;
        else if (immValue.z > rKey.immValue.z)
            return false;
        if (immValue.y < rKey.immValue.y)
            return true;
        else if (immValue.y > rKey.immValue.y)
            return false;
        if (immValue.x < rKey.immValue.x)
            return true;
        else if (immValue.x > rKey.immValue.x)
            return false;
        if (cuContext < rKey.cuContext)
            return true;
        else if (cuContext > rKey.cuContext)
            return false;
        return false;
    }
};

struct Fx4ImmTextureCacheKey {
    float4 immValue;
    CUcontext cuContext;

    bool operator<(const Fx4ImmTextureCacheKey &rKey) const {
        if (immValue.w < rKey.immValue.w)
            return true;
        else if (immValue.w > rKey.immValue.w)
            return false;
        if (immValue.z < rKey.immValue.z)
            return true;
        else if (immValue.z > rKey.immValue.z)
            return false;
        if (immValue.y < rKey.immValue.y)
            return true;
        else if (immValue.y > rKey.immValue.y)
            return false;
        if (immValue.x < rKey.immValue.x)
            return true;
        else if (immValue.x > rKey.immValue.x)
            return false;
        if (cuContext < rKey.cuContext)
            return true;
        else if (cuContext > rKey.cuContext)
            return false;
        return false;
    }
};

struct TextureCacheValue {
    cudau::Array texture;
    bool needsDegamma;
    bool isHDR;
    BumpMapTextureType bumpMapType;
};

static std::map<TextureCacheKey, TextureCacheValue> s_textureCache;
static std::map<Fx1ImmTextureCacheKey, TextureCacheValue> s_Fx1ImmTextureCache;
static std::map<Fx3ImmTextureCacheKey, TextureCacheValue> s_Fx3ImmTextureCache;
static std::map<Fx4ImmTextureCacheKey, TextureCacheValue> s_Fx4ImmTextureCache;

static void finalizeTextureCaches() {
    for (auto &it : s_textureCache)
        it.second.texture.finalize();
    for (auto &it : s_Fx1ImmTextureCache)
        it.second.texture.finalize();
    for (auto &it : s_Fx3ImmTextureCache)
        it.second.texture.finalize();
    for (auto &it : s_Fx4ImmTextureCache)
        it.second.texture.finalize();
}

static void createFx1ImmTexture(
    CUcontext cuContext,
    float immValue,
    bool isNormalized,
    const cudau::Array** texture) {
    Fx1ImmTextureCacheKey cacheKey;
    cacheKey.immValue = immValue;
    cacheKey.cuContext = cuContext;
    if (s_Fx1ImmTextureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_Fx1ImmTextureCache.at(cacheKey);
        *texture = &value.texture;
        return;
    }

    TextureCacheValue cacheValue;
    cacheValue.isHDR = !isNormalized;
    if (isNormalized) {
        uint8_t data = std::min<uint32_t>(255 * immValue, 255);
        cacheValue.texture.initialize2D(
            cuContext, cudau::ArrayElementType::UInt8, 1,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write<uint8_t>(reinterpret_cast<uint8_t*>(&data), 1);
    }
    else {
        cacheValue.texture.initialize2D(
            cuContext, cudau::ArrayElementType::Float32, 1,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write(&immValue, 1);
    }

    s_Fx1ImmTextureCache[cacheKey] = std::move(cacheValue);

    *texture = &s_Fx1ImmTextureCache.at(cacheKey).texture;
}

static void createFx3ImmTexture(
    CUcontext cuContext,
    const float3 &immValue,
    bool isNormalized,
    const cudau::Array** texture) {
    Fx3ImmTextureCacheKey cacheKey;
    cacheKey.immValue = immValue;
    cacheKey.cuContext = cuContext;
    if (s_Fx3ImmTextureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_Fx3ImmTextureCache.at(cacheKey);
        *texture = &value.texture;
        return;
    }

    TextureCacheValue cacheValue;
    cacheValue.isHDR = !isNormalized;
    if (isNormalized) {
        uint32_t data = ((std::min<uint32_t>(255 * immValue.x, 255) << 0) |
                         (std::min<uint32_t>(255 * immValue.y, 255) << 8) |
                         (std::min<uint32_t>(255 * immValue.z, 255) << 16) |
                         255 << 24);
        cacheValue.texture.initialize2D(
            cuContext, cudau::ArrayElementType::UInt8, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write<uint8_t>(reinterpret_cast<uint8_t*>(&data), 4);
    }
    else {
        float data[4] = {
            immValue.x, immValue.y, immValue.z, 1.0f
        };
        cacheValue.texture.initialize2D(
            cuContext, cudau::ArrayElementType::Float32, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write(data, 4);
    }

    s_Fx3ImmTextureCache[cacheKey] = std::move(cacheValue);

    *texture = &s_Fx3ImmTextureCache.at(cacheKey).texture;
}

static void createFx4ImmTexture(
    CUcontext cuContext,
    const float4 &immValue,
    bool isNormalized,
    const cudau::Array** texture) {
    Fx4ImmTextureCacheKey cacheKey;
    cacheKey.immValue = immValue;
    cacheKey.cuContext = cuContext;
    if (s_Fx4ImmTextureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_Fx4ImmTextureCache.at(cacheKey);
        *texture = &value.texture;
        return;
    }

    TextureCacheValue cacheValue;
    cacheValue.isHDR = !isNormalized;
    if (isNormalized) {
        uint32_t data = ((std::min<uint32_t>(255 * immValue.x, 255) << 0) |
                         (std::min<uint32_t>(255 * immValue.y, 255) << 8) |
                         (std::min<uint32_t>(255 * immValue.z, 255) << 16) |
                         (std::min<uint32_t>(255 * immValue.w, 255) << 24));
        cacheValue.texture.initialize2D(
            cuContext, cudau::ArrayElementType::UInt8, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write<uint8_t>(reinterpret_cast<uint8_t*>(&data), 4);
    }
    else {
        float data[4] = {
            immValue.x, immValue.y, immValue.z, immValue.w
        };
        cacheValue.texture.initialize2D(
            cuContext, cudau::ArrayElementType::Float32, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write(data, 4);
    }

    s_Fx4ImmTextureCache[cacheKey] = std::move(cacheValue);

    *texture = &s_Fx4ImmTextureCache.at(cacheKey).texture;
}

static bool loadTexture(
    const std::filesystem::path &filePath, const float4 &fallbackValue,
    CUcontext cuContext,
    const cudau::Array** texture,
    bool* needsDegamma,
    bool* isHDR = nullptr) {
    TextureCacheKey cacheKey;
    cacheKey.filePath = filePath;
    cacheKey.cuContext = cuContext;
    if (s_textureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_textureCache.at(cacheKey);
        *texture = &value.texture;
        *needsDegamma = value.needsDegamma;
        if (isHDR)
            *isHDR = value.isHDR;
        return true;
    }

    bool success = true;
    TextureCacheValue cacheValue;
    if (filePath.extension() == ".dds" ||
        filePath.extension() == ".DDS") {
        int32_t width, height, mipCount;
        dds::Format ddsFormat;
        size_t* sizes;
        uint8_t** imageData = dds::load(filePath.string().c_str(),
                                        &width, &height, &mipCount, &sizes, &ddsFormat);
        if (imageData) {
            cudau::ArrayElementType elemType;
            translate(ddsFormat, &elemType, &cacheValue.needsDegamma);
            cacheValue.isHDR =
                elemType == cudau::ArrayElementType::BC6H_SF16 ||
                elemType == cudau::ArrayElementType::BC6H_UF16;
            cacheValue.texture.initialize2D(
                cuContext, elemType, 1,
                cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                width, height, 1);
            cacheValue.texture.write<uint8_t>(imageData[0], sizes[0]);
            dds::free(imageData, mipCount, sizes);
        }
        else {
            success = false;
        }
    }
    else {
        int32_t width, height, n;
        uint8_t* linearImageData = stbi_load(filePath.string().c_str(),
                                             &width, &height, &n, 4);
        if (linearImageData) {
            cacheValue.texture.initialize2D(
                cuContext, cudau::ArrayElementType::UInt8, 4,
                cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                width, height, 1);
            cacheValue.texture.write<uint8_t>(linearImageData, width * height * 4);
            stbi_image_free(linearImageData);
            cacheValue.needsDegamma = true;
        }
        else {
            success = false;
        }
    }

    if (success) {
        s_textureCache[cacheKey] = std::move(cacheValue);

        *texture = &s_textureCache.at(cacheKey).texture;
        *needsDegamma = s_textureCache.at(cacheKey).needsDegamma;
        if (isHDR)
            *isHDR = s_textureCache.at(cacheKey).isHDR;
    }
    else {
        createFx4ImmTexture(cuContext, fallbackValue, true, texture);
        cacheValue.needsDegamma = true;
        cacheValue.isHDR = false;
    }

    return success;
}

static bool loadNormalTexture(
    const std::filesystem::path &filePath,
    CUcontext cuContext,
    const cudau::Array** texture,
    BumpMapTextureType* bumpMapType) {
    TextureCacheKey cacheKey;
    cacheKey.filePath = filePath;
    cacheKey.cuContext = cuContext;
    if (s_textureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_textureCache.at(cacheKey);
        *texture = &value.texture;
        *bumpMapType = value.bumpMapType;
        return true;
    }

    bool success = true;
    TextureCacheValue cacheValue;
    if (filePath.extension() == ".dds" ||
        filePath.extension() == ".DDS") {
        int32_t width, height, mipCount;
        dds::Format ddsFormat;
        size_t* sizes;
        uint8_t** imageData = dds::load(filePath.string().c_str(),
                                        &width, &height, &mipCount, &sizes, &ddsFormat);
        if (imageData) {
            cudau::ArrayElementType elemType;
            translate(ddsFormat, &elemType, &cacheValue.needsDegamma);
            if (elemType == cudau::ArrayElementType::BC1_UNorm ||
                elemType == cudau::ArrayElementType::BC2_UNorm ||
                elemType == cudau::ArrayElementType::BC3_UNorm ||
                elemType == cudau::ArrayElementType::BC7_UNorm)
                cacheValue.bumpMapType = BumpMapTextureType::NormalMap_BC;
            else if (elemType == cudau::ArrayElementType::BC4_SNorm ||
                     elemType == cudau::ArrayElementType::BC4_UNorm)
                cacheValue.bumpMapType = BumpMapTextureType::HeightMap_BC;
            else if (elemType == cudau::ArrayElementType::BC5_UNorm)
                cacheValue.bumpMapType = BumpMapTextureType::NormalMap_BC_2ch;
            else
                Assert_NotImplemented();
            auto textureGather = cacheValue.bumpMapType == BumpMapTextureType::HeightMap_BC ?
                cudau::ArrayTextureGather::Enable :
                cudau::ArrayTextureGather::Disable;
            cacheValue.texture.initialize2D(
                cuContext, elemType, 1,
                cudau::ArraySurface::Disable,
                textureGather,
                width, height, 1);
            cacheValue.texture.write<uint8_t>(imageData[0], sizes[0]);
            dds::free(imageData, mipCount, sizes);
        }
        else {
            success = false;
        }
    }
    else {
        int32_t width, height, n;
        uint8_t* linearImageData = stbi_load(filePath.string().c_str(),
                                             &width, &height, &n, 4);
        std::string filename = filePath.filename().string();
        if (n > 1 &&
            filename != "spnza_bricks_a_bump.png") // Dedicated fix for crytek sponza model.
            cacheValue.bumpMapType = BumpMapTextureType::NormalMap;
        else
            cacheValue.bumpMapType = BumpMapTextureType::HeightMap;
        if (linearImageData) {
            auto textureGather = cacheValue.bumpMapType == BumpMapTextureType::HeightMap ?
                cudau::ArrayTextureGather::Enable :
                cudau::ArrayTextureGather::Disable;
            cacheValue.texture.initialize2D(
                cuContext, cudau::ArrayElementType::UInt8, 4,
                cudau::ArraySurface::Disable, textureGather,
                width, height, 1);
            cacheValue.texture.write<uint8_t>(linearImageData, width * height * 4);
            stbi_image_free(linearImageData);
        }
        else {
            success = false;
        }
    }

    if (success) {
        s_textureCache[cacheKey] = std::move(cacheValue);
        *texture = &s_textureCache.at(cacheKey).texture;
        *bumpMapType = s_textureCache.at(cacheKey).bumpMapType;
    }
    else {
        createFx3ImmTexture(cuContext, float3(0.5f, 0.5f, 1.0f), true, texture);
        *bumpMapType = BumpMapTextureType::NormalMap;
    }

    return success;
}

static void createNormalTexture(
    GPUEnvironment &gpuEnv,
    const std::filesystem::path &normalPath,
    Material* mat, BumpMapTextureType* bumpMapType) {
    if (normalPath.empty()) {
        createFx3ImmTexture(gpuEnv.cuContext, float3(0.5f, 0.5f, 1.0f), true, &mat->normal);
        *bumpMapType = BumpMapTextureType::NormalMap;
    }
    else {
        hpprintf("  Reading: %s ... ", normalPath.string().c_str());
        if (loadNormalTexture(normalPath, gpuEnv.cuContext, &mat->normal, bumpMapType))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
}

static void createEmittanceTexture(
    GPUEnvironment &gpuEnv,
    const std::filesystem::path &emittancePath, const float3 &immEmittance,
    Material* mat,
    bool* needsDegamma, bool* isHDR) {
    *needsDegamma = false;
    *isHDR = false;
    if (emittancePath.empty()) {
        mat->texEmittance = 0;
        if (immEmittance != float3(0.0f, 0.0f, 0.0f))
            createFx3ImmTexture(gpuEnv.cuContext, immEmittance, false, &mat->emittance);
    }
    else {
        hpprintf("  Reading: %s ... ", emittancePath.string().c_str());
        if (loadTexture(emittancePath, float4(immEmittance, 1.0f), gpuEnv.cuContext,
                        &mat->emittance, needsDegamma, isHDR))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
}

static Material* createLambertMaterial(
    GPUEnvironment &gpuEnv,
    const std::filesystem::path &reflectancePath, const float3 &immReflectance,
    const std::filesystem::path &normalPath,
    const std::filesystem::path &emittancePath, const float3 &immEmittance) {
    shared::MaterialData* matDataOnHost = gpuEnv.materialDataBuffer.getMappedPointer();

    cudau::TextureSampler sampler_sRGB;
    sampler_sRGB.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_sRGB.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_sRGB.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

    cudau::TextureSampler sampler_float;
    sampler_float.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_float.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_float.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_float.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_float.setReadMode(cudau::TextureReadMode::ElementType);

    cudau::TextureSampler sampler_normFloat;
    sampler_normFloat.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_normFloat.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_normFloat.setReadMode(cudau::TextureReadMode::NormalizedFloat);

    Material* mat = new Material();
    bool needsDegamma;

    mat->body = Material::Lambert();
    auto &body = std::get<Material::Lambert>(mat->body);
    if (!reflectancePath.empty()) {
        hpprintf("  Reading: %s ... ", reflectancePath.string().c_str());
        if (loadTexture(reflectancePath, float4(immReflectance, 1.0f), gpuEnv.cuContext,
                        &body.reflectance, &needsDegamma))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
    if (!body.reflectance) {
        createFx3ImmTexture(gpuEnv.cuContext, immReflectance, true, &body.reflectance);
        needsDegamma = true;
    }
    if (needsDegamma)
        body.texReflectance = sampler_sRGB.createTextureObject(*body.reflectance);
    else
        body.texReflectance = sampler_normFloat.createTextureObject(*body.reflectance);

    BumpMapTextureType bumpMapType;
    createNormalTexture(gpuEnv, normalPath, mat, &bumpMapType);
    mat->texNormal = sampler_normFloat.createTextureObject(*mat->normal);
    CallableProgram dcReadModifiedNormal;
    if (bumpMapType == BumpMapTextureType::NormalMap ||
        bumpMapType == BumpMapTextureType::NormalMap_BC)
        dcReadModifiedNormal = CallableProgram_ReadModifiedNormalFromNormalMap;
    else if (bumpMapType == BumpMapTextureType::NormalMap_BC)
        dcReadModifiedNormal = CallableProgram_ReadModifiedNormalFromNormalMap2ch;
    else
        dcReadModifiedNormal = CallableProgram_ReadModifiedNormalFromHeightMap;

    bool isHDR;
    createEmittanceTexture(gpuEnv, emittancePath, immEmittance,
                           mat, &needsDegamma, &isHDR);
    if (mat->emittance) {
        if (needsDegamma)
            mat->texEmittance = sampler_sRGB.createTextureObject(*mat->emittance);
        else if (isHDR)
            mat->texEmittance = sampler_float.createTextureObject(*mat->emittance);
        else
            mat->texEmittance = sampler_normFloat.createTextureObject(*mat->emittance);
    }

    mat->materialSlot = gpuEnv.materialSlotFinder.getFirstAvailableSlot();
    gpuEnv.materialSlotFinder.setInUse(mat->materialSlot);

    shared::MaterialData matData = {};
    matData.asLambert.reflectance = body.texReflectance;
    matData.normal = mat->texNormal;
    matData.emittance = mat->texEmittance;
    matData.normalWidth = mat->normal->getWidth();
    matData.normalHeight = mat->normal->getHeight();
    matData.readModifiedNormal = shared::ReadModifiedNormal(dcReadModifiedNormal);
    matData.setupBSDFBody = shared::SetupBSDFBody(CallableProgram_SetupLambertBRDF);
    matData.bsdfSampleThroughput = shared::BSDFSampleThroughput(CallableProgram_LambertBRDF_sampleThroughput);
    matData.bsdfEvaluate = shared::BSDFEvaluate(CallableProgram_LambertBRDF_evaluate);
    matData.bsdfEvaluatePDF = shared::BSDFEvaluatePDF(CallableProgram_LambertBRDF_evaluatePDF);
    matData.bsdfEvaluateDHReflectanceEstimate = shared::BSDFEvaluateDHReflectanceEstimate(CallableProgram_LambertBRDF_evaluateDHReflectanceEstimate);
    matDataOnHost[mat->materialSlot] = matData;

    return mat;
}

static Material* createDiffuseAndSpecularMaterial(
    GPUEnvironment &gpuEnv,
    const std::filesystem::path &diffuseColorPath, const float3 &immDiffuseColor,
    const std::filesystem::path &specularColorPath, const float3 &immSpecularColor,
    float immSmoothness,
    const std::filesystem::path &normalPath,
    const std::filesystem::path &emittancePath, const float3 &immEmittance) {
    shared::MaterialData* matDataOnHost = gpuEnv.materialDataBuffer.getMappedPointer();

    cudau::TextureSampler sampler_sRGB;
    sampler_sRGB.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_sRGB.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_sRGB.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

    cudau::TextureSampler sampler_float;
    sampler_float.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_float.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_float.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_float.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_float.setReadMode(cudau::TextureReadMode::ElementType);

    cudau::TextureSampler sampler_normFloat;
    sampler_normFloat.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_normFloat.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_normFloat.setReadMode(cudau::TextureReadMode::NormalizedFloat);

    Material* mat = new Material();
    bool needsDegamma = false;

    mat->body = Material::DiffuseAndSpecular();
    auto &body = std::get<Material::DiffuseAndSpecular>(mat->body);

    if (!diffuseColorPath.empty()) {
        hpprintf("  Reading: %s ... ", diffuseColorPath.string().c_str());
        if (loadTexture(diffuseColorPath, float4(immDiffuseColor, 1.0f), gpuEnv.cuContext,
                        &body.diffuse, &needsDegamma))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
    if (!body.diffuse) {
        createFx3ImmTexture(gpuEnv.cuContext, immDiffuseColor, true, &body.diffuse);
        needsDegamma = true;
    }
    if (needsDegamma)
        body.texDiffuse = sampler_sRGB.createTextureObject(*body.diffuse);
    else
        body.texDiffuse = sampler_normFloat.createTextureObject(*body.diffuse);

    if (!specularColorPath.empty()) {
        hpprintf("  Reading: %s ... ", specularColorPath.string().c_str());
        if (loadTexture(specularColorPath, float4(immSpecularColor, 1.0f), gpuEnv.cuContext,
                        &body.specular, &needsDegamma))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
    if (!body.specular) {
        createFx3ImmTexture(gpuEnv.cuContext, immSpecularColor, true, &body.specular);
        needsDegamma = true;
    }
    if (needsDegamma)
        body.texSpecular = sampler_sRGB.createTextureObject(*body.specular);
    else
        body.texSpecular = sampler_normFloat.createTextureObject(*body.specular);

    createFx1ImmTexture(gpuEnv.cuContext, immSmoothness, true, &body.smoothness);
    body.texSmoothness = sampler_normFloat.createTextureObject(*body.smoothness);

    BumpMapTextureType bumpMapType;
    createNormalTexture(gpuEnv, normalPath, mat, &bumpMapType);
    mat->texNormal = sampler_normFloat.createTextureObject(*mat->normal);
    CallableProgram dcReadModifiedNormal;
    if (bumpMapType == BumpMapTextureType::NormalMap ||
        bumpMapType == BumpMapTextureType::NormalMap_BC)
        dcReadModifiedNormal = CallableProgram_ReadModifiedNormalFromNormalMap;
    else if (bumpMapType == BumpMapTextureType::NormalMap_BC)
        dcReadModifiedNormal = CallableProgram_ReadModifiedNormalFromNormalMap2ch;
    else
        dcReadModifiedNormal = CallableProgram_ReadModifiedNormalFromHeightMap;

    bool isHDR;
    createEmittanceTexture(gpuEnv, emittancePath, immEmittance,
                           mat, &needsDegamma, &isHDR);
    if (mat->emittance) {
        if (needsDegamma)
            mat->texEmittance = sampler_sRGB.createTextureObject(*mat->emittance);
        else if (isHDR)
            mat->texEmittance = sampler_float.createTextureObject(*mat->emittance);
        else
            mat->texEmittance = sampler_normFloat.createTextureObject(*mat->emittance);
    }

    mat->materialSlot = gpuEnv.materialSlotFinder.getFirstAvailableSlot();
    gpuEnv.materialSlotFinder.setInUse(mat->materialSlot);

    shared::MaterialData matData = {};
    matData.asDiffuseAndSpecular.diffuse = body.texDiffuse;
    matData.asDiffuseAndSpecular.specular = body.texSpecular;
    matData.asDiffuseAndSpecular.smoothness = body.texSmoothness;
    matData.normal = mat->texNormal;
    matData.emittance = mat->texEmittance;
    matData.normalWidth = mat->normal->getWidth();
    matData.normalHeight = mat->normal->getHeight();
    matData.readModifiedNormal = shared::ReadModifiedNormal(dcReadModifiedNormal);
    matData.setupBSDFBody = shared::SetupBSDFBody(CallableProgram_SetupDiffuseAndSpecularBRDF);
    matData.bsdfSampleThroughput = shared::BSDFSampleThroughput(CallableProgram_DiffuseAndSpecularBRDF_sampleThroughput);
    matData.bsdfEvaluate = shared::BSDFEvaluate(CallableProgram_DiffuseAndSpecularBRDF_evaluate);
    matData.bsdfEvaluatePDF = shared::BSDFEvaluatePDF(CallableProgram_DiffuseAndSpecularBRDF_evaluatePDF);
    matData.bsdfEvaluateDHReflectanceEstimate = shared::BSDFEvaluateDHReflectanceEstimate(CallableProgram_DiffuseAndSpecularBRDF_evaluateDHReflectanceEstimate);
    matDataOnHost[mat->materialSlot] = matData;

    return mat;
}

static Material* createSimplePBRMaterial(
    GPUEnvironment &gpuEnv,
    const std::filesystem::path &baseColor_opacityPath, const float4 &immBaseColor_opacity,
    const std::filesystem::path &occlusion_roughness_metallicPath,
    const float3 &immOcclusion_roughness_metallic,
    const std::filesystem::path &normalPath,
    const std::filesystem::path &emittancePath, const float3 &immEmittance) {
    shared::MaterialData* matDataOnHost = gpuEnv.materialDataBuffer.getMappedPointer();

    cudau::TextureSampler sampler_sRGB;
    sampler_sRGB.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_sRGB.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_sRGB.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

    cudau::TextureSampler sampler_float;
    sampler_float.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_float.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_float.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_float.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_float.setReadMode(cudau::TextureReadMode::ElementType);

    cudau::TextureSampler sampler_normFloat;
    sampler_normFloat.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_normFloat.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_normFloat.setReadMode(cudau::TextureReadMode::NormalizedFloat);

    Material* mat = new Material();
    bool needsDegamma = false;

    mat->body = Material::SimplePBR();
    auto &body = std::get<Material::SimplePBR>(mat->body);

    if (!baseColor_opacityPath.empty()) {
        hpprintf("  Reading: %s ... ", baseColor_opacityPath.string().c_str());
        if (loadTexture(baseColor_opacityPath, immBaseColor_opacity, gpuEnv.cuContext,
                        &body.baseColor_opacity, &needsDegamma))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
    if (!body.baseColor_opacity) {
        createFx4ImmTexture(gpuEnv.cuContext, immBaseColor_opacity, true,
                            &body.baseColor_opacity);
        needsDegamma = true;
    }
    if (needsDegamma)
        body.texBaseColor_opacity = sampler_sRGB.createTextureObject(*body.baseColor_opacity);
    else
        body.texBaseColor_opacity = sampler_normFloat.createTextureObject(*body.baseColor_opacity);

    if (!occlusion_roughness_metallicPath.empty()) {
        hpprintf("  Reading: %s ... ", occlusion_roughness_metallicPath.string().c_str());
        if (loadTexture(occlusion_roughness_metallicPath, float4(immOcclusion_roughness_metallic, 0.0f),
                        gpuEnv.cuContext,
                        &body.occlusion_roughness_metallic, &needsDegamma))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
    if (!body.occlusion_roughness_metallic) {
        createFx3ImmTexture(gpuEnv.cuContext, immOcclusion_roughness_metallic, true,
                            &body.occlusion_roughness_metallic);
    }
    body.texOcclusion_roughness_metallic =
        sampler_normFloat.createTextureObject(*body.occlusion_roughness_metallic);

    BumpMapTextureType bumpMapType;
    createNormalTexture(gpuEnv, normalPath, mat, &bumpMapType);
    mat->texNormal = sampler_normFloat.createTextureObject(*mat->normal);
    CallableProgram dcReadModifiedNormal;
    if (bumpMapType == BumpMapTextureType::NormalMap ||
        bumpMapType == BumpMapTextureType::NormalMap_BC)
        dcReadModifiedNormal = CallableProgram_ReadModifiedNormalFromNormalMap;
    else if (bumpMapType == BumpMapTextureType::NormalMap_BC)
        dcReadModifiedNormal = CallableProgram_ReadModifiedNormalFromNormalMap2ch;
    else
        dcReadModifiedNormal = CallableProgram_ReadModifiedNormalFromHeightMap;

    bool isHDR;
    createEmittanceTexture(gpuEnv, emittancePath, immEmittance,
                           mat, &needsDegamma, &isHDR);
    if (mat->emittance) {
        if (needsDegamma)
            mat->texEmittance = sampler_sRGB.createTextureObject(*mat->emittance);
        else if (isHDR)
            mat->texEmittance = sampler_float.createTextureObject(*mat->emittance);
        else
            mat->texEmittance = sampler_normFloat.createTextureObject(*mat->emittance);
    }

    mat->materialSlot = gpuEnv.materialSlotFinder.getFirstAvailableSlot();
    gpuEnv.materialSlotFinder.setInUse(mat->materialSlot);

    shared::MaterialData matData = {};
    matData.asSimplePBR.baseColor_opacity = body.texBaseColor_opacity;
    matData.asSimplePBR.occlusion_roughness_metallic = body.texOcclusion_roughness_metallic;
    matData.normal = mat->texNormal;
    matData.emittance = mat->texEmittance;
    matData.normalWidth = mat->normal->getWidth();
    matData.normalHeight = mat->normal->getHeight();
    matData.readModifiedNormal = shared::ReadModifiedNormal(dcReadModifiedNormal);
    matData.setupBSDFBody = shared::SetupBSDFBody(CallableProgram_SetupSimplePBR_BRDF);
    matData.bsdfSampleThroughput = shared::BSDFSampleThroughput(CallableProgram_DiffuseAndSpecularBRDF_sampleThroughput);
    matData.bsdfEvaluate = shared::BSDFEvaluate(CallableProgram_DiffuseAndSpecularBRDF_evaluate);
    matData.bsdfEvaluatePDF = shared::BSDFEvaluatePDF(CallableProgram_DiffuseAndSpecularBRDF_evaluatePDF);
    matData.bsdfEvaluateDHReflectanceEstimate = shared::BSDFEvaluateDHReflectanceEstimate(CallableProgram_DiffuseAndSpecularBRDF_evaluateDHReflectanceEstimate);
    matDataOnHost[mat->materialSlot] = matData;

    return mat;
}

static GeometryInstance* createGeometryInstance(
    GPUEnvironment &gpuEnv,
    const std::vector<shared::Vertex> &vertices,
    const std::vector<shared::Triangle> &triangles,
    const Material* mat) {
    shared::GeometryInstanceData* geomInstDataOnHost = gpuEnv.geomInstDataBuffer.getMappedPointer();

    GeometryInstance* geomInst = new GeometryInstance();

    std::vector<float> emitterImportances(triangles.size(), 0.0f);
    // JP: 面積に比例して発光プリミティブをサンプリングできるようインポータンスを計算する。
    // EN: Calculate importance values to make it possible to sample an emitter primitive based on its area.
    for (int triIdx = 0; triIdx < emitterImportances.size(); ++triIdx) {
        const shared::Triangle &tri = triangles[triIdx];
        const shared::Vertex (&vs)[3] = {
            vertices[tri.index0],
            vertices[tri.index1],
            vertices[tri.index2],
        };
        if (mat->texEmittance) {
            float area = 0.5f * length(cross(vs[2].position - vs[0].position,
                                             vs[1].position - vs[0].position));
            Assert(area >= 0.0f, "Area must be positive.");
            emitterImportances[triIdx] = area;
        }
        geomInst->aabb
            .unify(vertices[0].position)
            .unify(vertices[1].position)
            .unify(vertices[2].position);
    }

    geomInst->mat = mat;
    geomInst->vertexBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, vertices);
    geomInst->triangleBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, triangles);
    geomInst->emitterPrimDist.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                         emitterImportances.data(), emitterImportances.size());
    geomInst->geomInstSlot = gpuEnv.geomInstSlotFinder.getFirstAvailableSlot();
    gpuEnv.geomInstSlotFinder.setInUse(geomInst->geomInstSlot);

    shared::GeometryInstanceData geomInstData = {};
    geomInstData.vertexBuffer = geomInst->vertexBuffer.getDevicePointer();
    geomInstData.triangleBuffer = geomInst->triangleBuffer.getDevicePointer();
    geomInst->emitterPrimDist.getDeviceType(&geomInstData.emitterPrimDist);
    geomInstData.materialSlot = mat->materialSlot;
    geomInstData.geomInstSlot = geomInst->geomInstSlot;
    geomInstDataOnHost[geomInst->geomInstSlot] = geomInstData;

    geomInst->optixGeomInst = gpuEnv.scene.createGeometryInstance();
    geomInst->optixGeomInst.setVertexBuffer(geomInst->vertexBuffer);
    geomInst->optixGeomInst.setTriangleBuffer(geomInst->triangleBuffer);
    geomInst->optixGeomInst.setNumMaterials(1, optixu::BufferView());
    geomInst->optixGeomInst.setMaterial(0, 0, gpuEnv.defaultMaterial);
    geomInst->optixGeomInst.setUserData(geomInstData);

    return geomInst;
}

static GeometryGroup* createGeometryGroup(
    GPUEnvironment &gpuEnv,
    const std::set<const GeometryInstance*> &geomInsts) {
    GeometryGroup* geomGroup = new GeometryGroup();
    geomGroup->geomInsts = geomInsts;
    geomGroup->numEmitterPrimitives = 0;

    geomGroup->optixGas = gpuEnv.scene.createGeometryAccelerationStructure();
    for (auto it = geomInsts.cbegin(); it != geomInsts.cend(); ++it) {
        const GeometryInstance* geomInst = *it;
        geomGroup->optixGas.addChild(geomInst->optixGeomInst);
        if (geomInst->mat->texEmittance)
            geomGroup->numEmitterPrimitives += geomInst->triangleBuffer.numElements();
        geomGroup->aabb.unify(geomInst->aabb);
    }
    geomGroup->optixGas.setNumMaterialSets(1);
    geomGroup->optixGas.setNumRayTypes(0, shared::NumRayTypes);

    return geomGroup;
}

static Instance* createInstance(
    GPUEnvironment &gpuEnv,
    const GeometryGroup* geomGroup,
    const Matrix4x4 &transform) {
    shared::InstanceData* instDataOnHost = gpuEnv.instDataBuffer[0].getMappedPointer();

    float3 scale;
    transform.decompose(&scale, nullptr, nullptr);
    float uniformScale = scale.x;

    // JP: 各ジオメトリインスタンスの光源サンプリングに関わるインポータンスは
    //     プリミティブのインポータンスの合計値とする。
    // EN: Use the sum of importance values of primitives as each geometry instances's importance
    //     for sampling a light source
    std::vector<uint32_t> geomInstSlots;
    std::vector<float> lightImportances;
    float sumLightImportances = 0.0f;
    for (auto it = geomGroup->geomInsts.cbegin(); it != geomGroup->geomInsts.cend(); ++it) {
        const GeometryInstance* geomInst = *it;
        geomInstSlots.push_back(geomInst->geomInstSlot);
        float lightImportance = uniformScale * geomInst->emitterPrimDist.getIntengral();
        lightImportances.push_back(lightImportance);
        sumLightImportances += lightImportance;
    }

    if (sumLightImportances > 0.0f &&
        (std::fabs(scale.y - uniformScale) / uniformScale >= 0.001f ||
         std::fabs(scale.z - uniformScale) / uniformScale >= 0.001f ||
         uniformScale <= 0.0f)) {
        hpprintf("Non-uniform scaling (%g, %g, %g) is not recommended for a light source instance.\n",
                 scale.x, scale.y, scale.z);
    }

    Instance* inst = new Instance();
    inst->geomGroup = geomGroup;
    inst->geomInstSlots.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, geomInstSlots);
    inst->lightGeomInstDist.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                       lightImportances.data(), lightImportances.size());
    inst->instSlot = gpuEnv.instSlotFinder.getFirstAvailableSlot();
    gpuEnv.instSlotFinder.setInUse(inst->instSlot);

    shared::InstanceData instData = {};
    instData.transform = transform;
    instData.prevTransform = transform;
    instData.normalMatrix = transpose(inverse(transform.getUpperLeftMatrix()));
    instData.geomInstSlots = inst->geomInstSlots.getDevicePointer();
    instData.numGeomInsts = inst->geomInstSlots.numElements();
    inst->lightGeomInstDist.getDeviceType(&instData.lightGeomInstDist);
    instDataOnHost[inst->instSlot] = instData;

    inst->optixInst = gpuEnv.scene.createInstance();
    inst->optixInst.setID(inst->instSlot);
    inst->optixInst.setChild(geomGroup->optixGas);
    float xfm[12] = {
        transform.m00, transform.m01, transform.m02, transform.m03,
        transform.m10, transform.m11, transform.m12, transform.m13,
        transform.m20, transform.m21, transform.m22, transform.m23,
    };
    inst->optixInst.setTransform(xfm);

    return inst;
}

constexpr bool useLambertMaterial = false;

static void createTriangleMeshes(
    const std::filesystem::path &filePath,
    MaterialConvention matConv,
    const Matrix4x4 &preTransform,
    GPUEnvironment &gpuEnv,
    std::vector<Material*> &materials,
    std::vector<GeometryInstance*> &geomInsts,
    std::vector<GeometryGroup*> &geomGroups,
    Mesh* mesh) {
    hpprintf("Reading: %s ... ", filePath.string().c_str());
    fflush(stdout);
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filePath.string(),
                                             aiProcess_Triangulate |
                                             aiProcess_GenNormals |
                                             aiProcess_CalcTangentSpace |
                                             aiProcess_FlipUVs);
    if (!scene) {
        hpprintf("Failed to load %s.\n", filePath.string().c_str());
        return;
    }
    hpprintf("done.\n");

    std::filesystem::path dirPath = filePath;
    dirPath.remove_filename();

    materials.clear();
    shared::MaterialData* matDataOnHost = gpuEnv.materialDataBuffer.getMappedPointer();
    for (int matIdx = 0; matIdx < scene->mNumMaterials; ++matIdx) {
        std::filesystem::path emittancePath;
        float3 immEmittance = float3(0.0f);

        const aiMaterial* aiMat = scene->mMaterials[matIdx];
        aiString strValue;
        float color[3];

        std::string matName;
        if (aiMat->Get(AI_MATKEY_NAME, strValue) == aiReturn_SUCCESS)
            matName = strValue.C_Str();
        hpprintf("%s:\n", matName.c_str());

        std::filesystem::path reflectancePath;
        float3 immReflectance;
        std::filesystem::path diffuseColorPath;
        float3 immDiffuseColor;
        std::filesystem::path specularColorPath;
        float3 immSpecularColor;
        float immSmoothness;
        if constexpr (useLambertMaterial) {
            if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
                reflectancePath = dirPath / strValue.C_Str();
            }
            else {
                if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) != aiReturn_SUCCESS) {
                    color[0] = 1.0f;
                    color[1] = 0.0f;
                    color[2] = 1.0f;
                }
                immReflectance = float3(color[0], color[1], color[2]);
            }
            (void)diffuseColorPath;
            (void)immDiffuseColor;
            (void)specularColorPath;
            (void)immSpecularColor;
            (void)immSmoothness;
        }
        else {
            if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
                diffuseColorPath = dirPath / strValue.C_Str();
            }
            else {
                if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) != aiReturn_SUCCESS) {
                    color[0] = 0.0f;
                    color[1] = 0.0f;
                    color[2] = 0.0f;
                }
                immDiffuseColor = float3(color[0], color[1], color[2]);
            }

            if (aiMat->Get(AI_MATKEY_TEXTURE_SPECULAR(0), strValue) == aiReturn_SUCCESS) {
                specularColorPath = dirPath / strValue.C_Str();
            }
            else {
                if (aiMat->Get(AI_MATKEY_COLOR_SPECULAR, color, nullptr) != aiReturn_SUCCESS) {
                    color[0] = 0.0f;
                    color[1] = 0.0f;
                    color[2] = 0.0f;
                }
                immSpecularColor = float3(color[0], color[1], color[2]);
            }

            // JP: 極端に鋭いスペキュラーにするとNEEで寄与が一切サンプルできなくなってしまう。
            // EN: Exteremely sharp specular makes it impossible to sample a contribution with NEE.
            if (aiMat->Get(AI_MATKEY_SHININESS, &immSmoothness, nullptr) != aiReturn_SUCCESS)
                immSmoothness = 0.0f;
            immSmoothness = std::sqrt(immSmoothness);
            immSmoothness = immSmoothness / 11.0f/*30.0f*/;

            (void)reflectancePath;
            (void)immReflectance;
        }

        std::filesystem::path normalPath;
        if (aiMat->Get(AI_MATKEY_TEXTURE_HEIGHT(0), strValue) == aiReturn_SUCCESS)
            normalPath = dirPath / strValue.C_Str();
        else if (aiMat->Get(AI_MATKEY_TEXTURE_NORMALS(0), strValue) == aiReturn_SUCCESS)
            normalPath = dirPath / strValue.C_Str();

        if (matName == "Pavement_Cobblestone_Big_BLENDSHADER") {
            immSmoothness = 0.2f;
        }
        else if (matName == "Pavement_Cobblestone_Small_BLENDSHADER") {
            immSmoothness = 0.2f;
        }
        else if (matName == "Pavement_Brick_BLENDSHADER") {
            immSmoothness = 0.2f;
        }
        else if (matName == "Pavement_Cobblestone_Wet_BLENDSHADER") {
            immSmoothness = 0.2f;
        }

        if (aiMat->Get(AI_MATKEY_TEXTURE_EMISSIVE(0), strValue) == aiReturn_SUCCESS)
            emittancePath = dirPath / strValue.C_Str();
        else if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, color, nullptr) == aiReturn_SUCCESS)
            immEmittance = float3(color[0], color[1], color[2]);

        Material* mat;
        if (matConv == MaterialConvention::Traditional) {
            if constexpr (useLambertMaterial) {
                mat = createLambertMaterial(
                    gpuEnv,
                    reflectancePath, immReflectance,
                    normalPath,
                    emittancePath, immEmittance);
            }
            else {
                mat = createDiffuseAndSpecularMaterial(
                    gpuEnv,
                    diffuseColorPath, immDiffuseColor,
                    specularColorPath, immSpecularColor,
                    immSmoothness,
                    normalPath,
                    emittancePath, immEmittance);
            }
        }
        else {
            // JP: diffuseテクスチャーとしてベースカラー + 不透明度
            //     specularテクスチャーとしてオクルージョン、ラフネス、メタリック
            //     が格納されていると仮定している。
            // EN: We assume diffuse texture as base color + opacity,
            //     specular texture as occlusion, roughness, metallic.
            mat = createSimplePBRMaterial(
                gpuEnv,
                diffuseColorPath, float4(immDiffuseColor, 1.0f),
                specularColorPath, immSpecularColor,
                normalPath,
                emittancePath, immEmittance);
        }

        materials.push_back(mat);
    }

    geomInsts.clear();
    Matrix3x3 preNormalTransform = transpose(inverse(preTransform.getUpperLeftMatrix()));
    for (int meshIdx = 0; meshIdx < scene->mNumMeshes; ++meshIdx) {
        const aiMesh* aiMesh = scene->mMeshes[meshIdx];

        std::vector<shared::Vertex> vertices(aiMesh->mNumVertices);
        for (int vIdx = 0; vIdx < vertices.size(); ++vIdx) {
            const aiVector3D &aip = aiMesh->mVertices[vIdx];
            const aiVector3D &ain = aiMesh->mNormals[vIdx];
            aiVector3D aitc0dir;
            if (aiMesh->mTangents)
                aitc0dir = aiMesh->mTangents[vIdx];
            if (!aiMesh->mTangents || !std::isfinite(aitc0dir.x)) {
                const auto makeCoordinateSystem = []
                (const float3 &normal, float3* tangent, float3* bitangent) {
                    float sign = normal.z >= 0 ? 1 : -1;
                    const float a = -1 / (sign + normal.z);
                    const float b = normal.x * normal.y * a;
                    *tangent = make_float3(1 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
                    *bitangent = make_float3(b, sign + normal.y * normal.y * a, -normal.y);
                };
                float3 tangent, bitangent;
                makeCoordinateSystem(float3(ain.x, ain.y, ain.z), &tangent, &bitangent);
                aitc0dir = aiVector3D(tangent.x, tangent.y, tangent.z);
            }
            const aiVector3D ait = aiMesh->mTextureCoords[0] ?
                aiMesh->mTextureCoords[0][vIdx] :
                aiVector3D(0.0f, 0.0f, 0.0f);

            shared::Vertex v;
            v.position = preTransform * float3(aip.x, aip.y, aip.z);
            v.normal = normalize(preNormalTransform * float3(ain.x, ain.y, ain.z));
            v.texCoord0Dir = normalize(preTransform * float3(aitc0dir.x, aitc0dir.y, aitc0dir.z));
            v.texCoord = float2(ait.x, ait.y);
            vertices[vIdx] = v;
        }

        std::vector<shared::Triangle> triangles(aiMesh->mNumFaces);
        for (int fIdx = 0; fIdx < triangles.size(); ++fIdx) {
            const aiFace &aif = aiMesh->mFaces[fIdx];
            Assert(aif.mNumIndices == 3, "Number of face vertices must be 3 here.");
            shared::Triangle tri;
            tri.index0 = aif.mIndices[0];
            tri.index1 = aif.mIndices[1];
            tri.index2 = aif.mIndices[2];
            triangles[fIdx] = tri;
        }

        geomInsts.push_back(createGeometryInstance(gpuEnv, vertices, triangles, materials[aiMesh->mMaterialIndex]));
    }

    std::vector<FlattenedNode> flattenedNodes;
    computeFlattenedNodes(scene, Matrix4x4(), scene->mRootNode, flattenedNodes);
    //for (int i = 0; i < flattenedNodes.size(); ++i) {
    //    const Matrix4x4 &mat = flattenedNodes[i].transform;
    //    hpprintf("%8.5f, %8.5f, %8.5f, %8.5f\n", mat.m00, mat.m01, mat.m02, mat.m03);
    //    hpprintf("%8.5f, %8.5f, %8.5f, %8.5f\n", mat.m10, mat.m11, mat.m12, mat.m13);
    //    hpprintf("%8.5f, %8.5f, %8.5f, %8.5f\n", mat.m20, mat.m21, mat.m22, mat.m23);
    //    hpprintf("%8.5f, %8.5f, %8.5f, %8.5f\n", mat.m30, mat.m31, mat.m32, mat.m33);
    //    hpprintf("\n");
    //}

    geomGroups.clear();
    mesh->groups.clear();
    shared::InstanceData* instDataOnHost = gpuEnv.instDataBuffer[0].getMappedPointer();
    std::map<std::set<const GeometryInstance*>, GeometryGroup*> geomGroupMap;
    for (int nodeIdx = 0; nodeIdx < flattenedNodes.size(); ++nodeIdx) {
        const FlattenedNode &node = flattenedNodes[nodeIdx];
        if (node.meshIndices.size() == 0)
            continue;

        std::set<const GeometryInstance*> srcGeomInsts;
        for (int i = 0; i < node.meshIndices.size(); ++i)
            srcGeomInsts.insert(geomInsts[node.meshIndices[i]]);
        GeometryGroup* geomGroup;
        if (geomGroupMap.count(srcGeomInsts) > 0) {
            geomGroup = geomGroupMap.at(srcGeomInsts);
        }
        else {
            geomGroup = createGeometryGroup(gpuEnv, srcGeomInsts);
            geomGroups.push_back(geomGroup);
        }

        Mesh::Group g = {};
        g.geomGroup = geomGroup;
        g.transform = node.transform;
        mesh->groups.push_back(g);
    }
}

static void createRectangleLight(
    float width, float depth,
    const float3 &reflectance,
    const std::filesystem::path &emittancePath,
    const float3 &immEmittance,
    const Matrix4x4 &transform,
    GPUEnvironment &gpuEnv,
    Material** material,
    GeometryInstance** geomInst,
    GeometryGroup** geomGroup,
    Mesh* mesh) {
    if constexpr (useLambertMaterial)
        *material = createLambertMaterial(gpuEnv, "", reflectance, "", emittancePath, immEmittance);
    else
        *material = createDiffuseAndSpecularMaterial(
            gpuEnv, "", reflectance, "", float3(0.0f), 0.3f,
            "",
            emittancePath, immEmittance);

    std::vector<shared::Vertex> vertices = {
        shared::Vertex{float3(-0.5f * width, 0.0f, -0.5f * depth), float3(0, -1, 0), float3(1, 0, 0), float2(0.0f, 1.0f)},
        shared::Vertex{float3(0.5f * width, 0.0f, -0.5f * depth), float3(0, -1, 0), float3(1, 0, 0), float2(1.0f, 1.0f)},
        shared::Vertex{float3(0.5f * width, 0.0f, 0.5f * depth), float3(0, -1, 0), float3(1, 0, 0), float2(1.0f, 0.0f)},
        shared::Vertex{float3(-0.5f * width, 0.0f, 0.5f * depth), float3(0, -1, 0), float3(1, 0, 0), float2(0.0f, 0.0f)},
    };
    std::vector<shared::Triangle> triangles = {
        shared::Triangle{0, 1, 2},
        shared::Triangle{0, 2, 3},
    };
    *geomInst = createGeometryInstance(gpuEnv, vertices, triangles, *material);

    std::set<const GeometryInstance*> srcGeomInsts = { *geomInst };
    *geomGroup = createGeometryGroup(gpuEnv, srcGeomInsts);

    Mesh::Group g = {};
    g.geomGroup = *geomGroup;
    g.transform = transform;
    mesh->groups.clear();
    mesh->groups.push_back(g);
}

static void createSphereLight(
    float radius,
    const float3 &reflectance,
    const std::filesystem::path &emittancePath,
    const float3 &immEmittance,
    const float3 &position,
    GPUEnvironment &gpuEnv,
    Material** material,
    GeometryInstance** geomInst,
    GeometryGroup** geomGroup,
    Mesh* mesh) {
    if constexpr (useLambertMaterial)
        *material = createLambertMaterial(gpuEnv, "", reflectance, "", emittancePath, immEmittance);
    else
        *material = createDiffuseAndSpecularMaterial(
            gpuEnv, "", reflectance, "", float3(0.0f), 0.3f,
            "",
            emittancePath, immEmittance);

    constexpr uint32_t numZenithSegments = 8;
    constexpr uint32_t numAzimuthSegments = 16;
    constexpr uint32_t numVertices = 2 + (numZenithSegments - 1) * numAzimuthSegments;
    constexpr uint32_t numTriangles = (2 + 2 * (numZenithSegments - 2)) * numAzimuthSegments;
    constexpr float zenithDelta = M_PI / numZenithSegments;
    constexpr float azimushDelta = 2 * M_PI / numAzimuthSegments;
    std::vector<shared::Vertex> vertices(numVertices);
    std::vector<shared::Triangle> triangles(numTriangles);
    uint32_t vIdx = 0;
    uint32_t triIdx = 0;
    vertices[vIdx++] = shared::Vertex{ float3(0, radius, 0), float3(0, 1, 0), float3(1, 0, 0), float2(0, 0) };
    {
        float zenith = zenithDelta;
        float2 texCoord = float2(0, zenith / M_PI);
        for (int aIdx = 0; aIdx < numAzimuthSegments; ++aIdx) {
            float azimuth = aIdx * azimushDelta;
            float3 n = float3(std::cos(azimuth) * std::sin(zenith),
                              std::cos(zenith),
                              std::sin(azimuth) * std::sin(zenith));
            float3 tc0Dir = float3(-std::sin(azimuth), 0, std::cos(azimuth));
            uint32_t lrIdx = 1 + aIdx;
            uint32_t llIdx = 1 + (aIdx + 1) % numAzimuthSegments;
            uint32_t uIdx = 0;
            texCoord.x = azimuth / (2 * M_PI);
            vertices[vIdx++] = shared::Vertex{ radius * n, n, tc0Dir, texCoord };
            triangles[triIdx++] = shared::Triangle{ llIdx, lrIdx, uIdx };
        }
    }
    for (int zIdx = 1; zIdx < numZenithSegments - 1; ++zIdx) {
        float zenith = (zIdx + 1) * zenithDelta;
        float2 texCoord = float2(0, zenith / M_PI);
        uint32_t baseVIdx = vIdx;
        for (int aIdx = 0; aIdx < numAzimuthSegments; ++aIdx) {
            float azimuth = aIdx * azimushDelta;
            float3 n = float3(std::cos(azimuth) * std::sin(zenith),
                              std::cos(zenith),
                              std::sin(azimuth) * std::sin(zenith));
            float3 tc0Dir = float3(-std::sin(azimuth), 0, std::cos(azimuth));
            texCoord.x = azimuth / (2 * M_PI);
            vertices[vIdx++] = shared::Vertex{ radius * n, n, tc0Dir, texCoord };
            uint32_t lrIdx = baseVIdx + aIdx;
            uint32_t llIdx = baseVIdx + (aIdx + 1) % numAzimuthSegments;
            uint32_t ulIdx = baseVIdx - numAzimuthSegments + (aIdx + 1) % numAzimuthSegments;
            uint32_t urIdx = baseVIdx - numAzimuthSegments + aIdx;
            triangles[triIdx++] = shared::Triangle{ llIdx, lrIdx, urIdx };
            triangles[triIdx++] = shared::Triangle{ llIdx, urIdx, ulIdx };
        }
    }
    vertices[vIdx++] = shared::Vertex{ float3(0, -radius, 0), float3(0, -1, 0), float3(1, 0, 0), float2(0, 1) };
    {
        for (int aIdx = 0; aIdx < numAzimuthSegments; ++aIdx) {
            uint32_t lIdx = numVertices - 1;
            uint32_t ulIdx = numVertices - 1 - numAzimuthSegments + (aIdx + 1) % numAzimuthSegments;
            uint32_t urIdx = numVertices - 1 - numAzimuthSegments + aIdx;
            triangles[triIdx++] = shared::Triangle{ lIdx, urIdx, ulIdx };
        }
    }
    *geomInst = createGeometryInstance(gpuEnv, vertices, triangles, *material);

    std::set<const GeometryInstance*> srcGeomInsts = { *geomInst };
    *geomGroup = createGeometryGroup(gpuEnv, srcGeomInsts);

    Mesh::Group g = {};
    g.geomGroup = *geomGroup;
    g.transform = Matrix4x4();
    mesh->groups.clear();
    mesh->groups.push_back(g);
}

static void loadEnvironmentalTexture(
    const std::filesystem::path &filePath,
    GPUEnvironment &gpuEnv,
    cudau::Array* envLightArray, CUtexObject* envLightTexture,
    RegularConstantContinuousDistribution2D* envLightImportanceMap) {
    cudau::TextureSampler sampler_float;
    sampler_float.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_float.setWrapMode(0, cudau::TextureWrapMode::Clamp);
    sampler_float.setWrapMode(1, cudau::TextureWrapMode::Clamp);
    sampler_float.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_float.setReadMode(cudau::TextureReadMode::ElementType);

    int32_t width, height;
    float* textureData;
    const char* errMsg = nullptr;
    int ret = LoadEXR(&textureData, &width, &height, filePath.string().c_str(), &errMsg);
    if (ret == TINYEXR_SUCCESS) {
        float* importanceData = new float[width * height];
        for (int y = 0; y < height; ++y) {
            float theta = M_PI * (y + 0.5f) / height;
            float sinTheta = std::sin(theta);
            for (int x = 0; x < width; ++x) {
                uint32_t idx = 4 * (y * width + x);
                textureData[idx + 0] = std::max(textureData[idx + 0], 0.0f);
                textureData[idx + 1] = std::max(textureData[idx + 1], 0.0f);
                textureData[idx + 2] = std::max(textureData[idx + 2], 0.0f);
                float3 value(textureData[idx + 0],
                             textureData[idx + 1],
                             textureData[idx + 2]);
                importanceData[y * width + x] = sRGB_calcLuminance(value) * sinTheta;
            }
        }

        envLightArray->initialize2D(
            gpuEnv.cuContext, cudau::ArrayElementType::Float32, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            width, height, 1);
        envLightArray->write(textureData, width * height * 4);

        free(textureData);

        envLightImportanceMap->initialize(
            gpuEnv.cuContext, GPUEnvironment::bufferType, importanceData, width, height);
        delete[] importanceData;

        *envLightTexture = sampler_float.createTextureObject(*envLightArray);
    }
    else {
        hpprintf("Failed to read %s\n", filePath.string().c_str());
        hpprintf("%s\n", errMsg);
        FreeEXRErrorMessage(errMsg);
    }
}
