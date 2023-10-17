/*
* 
* コマンドラインオプション例 / Command line option example:
You can load a 3D model for example by downloading from the internet.
(e.g. https://casual-effects.com/data/)

(1) -cam-pos 0 3 0 -cam-yaw 90
    -name sponza -obj crytek_sponza/sponza.obj 0.01 trad
    -begin-pos 0 0 0.36125 -inst sponza
    -name rectlight -emittance 600 600 600 -rectangle 2 2 -begin-pos 0 15 0 -inst rectlight

JP: このプログラムはTFDM (Tessellation-Free Displacement Mapping) [1]の実装例です。
    ディスプレイスメントマッピングによって3Dのサーフェスに詳細なジオメトリを付加することができますが、
    事前にメッシュからBVHなどのAcceleration Structureを作っておく必要があるレイトレーシングでは事前のポリゴン分割、
    事前テッセレーションが必要で、膨大なメモリ使用量が問題になります。
    また適切な事前分割の仕方を考えるのも簡単ではありません。
    TFDMではハイトマップごとに各MIPレベル・各テクセル範囲中の最小値・最大値を階層的に記録したMinmaxミップマップを
    予め計算し、ベースメッシュ形状と切り離した暗黙的なBVH(の一部)として用いることで事前テッセレーションを行うことなく、
    省メモリに緻密なジオメトリに対するレイトレーシングを可能とします。
    トラバーサル中にはベース三角形ごとにIntersectionシェーダーが起動されます。
    三角形の各頂点における位置、法線、テクスチャー座標とMinmaxミップマップの値から、
    アフィン演算を用いてその場でAABBを計算、レイとAABBとの交叉判定をMIPレベルを下りながら階層的に行うことで
    BVHトラバーサルを実行、目的のMIPレベルに到達した段階で最終的な形状との交叉判定を行います。
    テクセル単位で交叉判定を行うためポリゴンの適切な事前分割について考える必要もありません。

EN: This program is an example implementation of TFDM (Tessellation-Free Displacement Mapping) [1].
    Displacement mapping can add fine details to 3D surfaces.
    However, ray tracing requires building an acceleration structure like BVH beforehand from the mesh
    with prior polygon subdivision, that is pre-tessellation, and this means significant amount of memory consumption.
    Also, it is not easy to plan appropriate subdivision.
    In TFDM, a minmax mipmap is pre-computed for each heightmap, where minimum and maximum values
    for each mip-level, texel domain is recorded hierarchically, and use it as (a part of) an implicit BVH
    decoupled from base mesh shape to enable ray tracing against detailed geometry without pre-tessellation,
    which means low memory consumption.
    Intersection shader is invoked during traversal for each base triangle.
    Affine arithmetic is used to compute an AABB on the fly from the position, normal, texture coordinates of
    the triangle vertices and the values from the minmax mipmap, then the shader performs
    ray vs AABB intersection tests hierarchically while descending MIP levels, finally performs
    intersection test against the actual geometry when it reachs a target mip level.
    There is no need to plan appropriate polygon subdivision since the intersection tests are performed texel-wise.

[1] Tessellation-Free Displacement Mapping for Ray Tracing
    https://research.adobe.com/publication/tessellation-free-displacement-mapping-for-ray-tracing/

*/

#include "tfdm_shared.h"
#include "../common/common_host.h"

// Include glfw3.h after our OpenGL definitions
#include "../utils/gl_util.h"
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"



enum class GBufferEntryPoint {
    setupGBuffers = 0,
};
enum class PathTracingEntryPoint {
    pathTrace,
};

struct GPUEnvironment {
    CUcontext cuContext;
    optixu::Context optixContext;

    CUmodule tfdmModule;
    cudau::Kernel kernelGenerateFirstMinMaxMipMap_Box;
    cudau::Kernel kernelGenerateFirstMinMaxMipMap_TwoTriangle;
    cudau::Kernel kernelGenerateFirstMinMaxMipMap_Bilinear;
    cudau::Kernel kernelGenerateFirstMinMaxMipMap_BSpline;
    cudau::Kernel kernelGenerateMinMaxMipMap_Box;
    cudau::Kernel kernelGenerateMinMaxMipMap_TwoTriangle;
    cudau::Kernel kernelGenerateMinMaxMipMap_Bilinear;
    cudau::Kernel kernelGenerateMinMaxMipMap_BSpline;
    cudau::Kernel kernelComputeAABBs;

    template <typename EntryPointType>
    struct Pipeline {
        optixu::Pipeline optixPipeline;
        optixu::Module optixModule;
        std::unordered_map<EntryPointType, optixu::Program> entryPoints;
        std::unordered_map<std::string, optixu::Program> programs;
        std::unordered_map<std::string, optixu::HitProgramGroup> hitPrograms;
        std::vector<optixu::CallableProgramGroup> callablePrograms;
        cudau::Buffer sbt;
        cudau::Buffer hitGroupSbt;

        void setEntryPoint(EntryPointType et) {
            optixPipeline.setRayGenerationProgram(entryPoints.at(et));
        }
    };

    Pipeline<GBufferEntryPoint> gBuffer;
    Pipeline<PathTracingEntryPoint> pathTracing;

    optixu::Material optixDefaultMaterial;
    optixu::Material optixTFDMMaterial;

    void initialize() {
        CUDADRV_CHECK(cuInit(0));
        CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
        CUDADRV_CHECK(cuCtxSetCurrent(cuContext));

        CUDADRV_CHECK(cuModuleLoad(
            &tfdmModule,
            (getExecutableDirectory() / "tfdm/ptxes/tfdm.ptx").string().c_str()));
        kernelGenerateFirstMinMaxMipMap_Box =
            cudau::Kernel(tfdmModule, "generateFirstMinMaxMipMap_Box", cudau::dim3(8, 8), 0);
        kernelGenerateFirstMinMaxMipMap_TwoTriangle =
            cudau::Kernel(tfdmModule, "generateFirstMinMaxMipMap_TwoTriangle", cudau::dim3(8, 8), 0);
        kernelGenerateFirstMinMaxMipMap_Bilinear =
            cudau::Kernel(tfdmModule, "generateFirstMinMaxMipMap_Bilinear", cudau::dim3(8, 8), 0);
        kernelGenerateFirstMinMaxMipMap_BSpline =
            cudau::Kernel(tfdmModule, "generateFirstMinMaxMipMap_BSpline", cudau::dim3(8, 8), 0);
        kernelGenerateMinMaxMipMap_Box =
            cudau::Kernel(tfdmModule, "generateMinMaxMipMap_Box", cudau::dim3(8, 8), 0);
        kernelGenerateMinMaxMipMap_TwoTriangle =
            cudau::Kernel(tfdmModule, "generateMinMaxMipMap_TwoTriangle", cudau::dim3(8, 8), 0);
        kernelGenerateMinMaxMipMap_Bilinear =
            cudau::Kernel(tfdmModule, "generateMinMaxMipMap_Bilinear", cudau::dim3(8, 8), 0);
        kernelGenerateMinMaxMipMap_BSpline =
            cudau::Kernel(tfdmModule, "generateMinMaxMipMap_BSpline", cudau::dim3(8, 8), 0);
        kernelComputeAABBs =
            cudau::Kernel(tfdmModule, "computeAABBs", cudau::dim3(32), 0);

        optixContext = optixu::Context::create(
            cuContext/*, 4, DEBUG_SELECT(optixu::EnableValidation::Yes, optixu::EnableValidation::No)*/);

        optixDefaultMaterial = optixContext.createMaterial();
        optixTFDMMaterial = optixContext.createMaterial();
        optixu::Module emptyModule;

        {
            Pipeline<GBufferEntryPoint> &pipeline = gBuffer;
            optixu::Pipeline &p = pipeline.optixPipeline;
            optixu::Module &m = pipeline.optixModule;
            p = optixContext.createPipeline();

            p.setPipelineOptions(
                std::max({
                    shared::PrimaryRayPayloadSignature::numDwords
                         }),
                std::max({
                    static_cast<uint32_t>(optixu::calcSumDwords<float2>()),
                    shared::AABBAttributeSignature::numDwords,
                    shared::DisplacedSurfaceAttributeSignature::numDwords
                         }),
                "plp", sizeof(shared::PipelineLaunchParameters),
                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM);

            m = p.createModuleFromPTXString(
                readTxtFile(getExecutableDirectory() / "tfdm/ptxes/optix_gbuffer_kernels.ptx"),
                OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
                DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            pipeline.entryPoints[GBufferEntryPoint::setupGBuffers] = p.createRayGenProgram(
                m, RT_RG_NAME_STR("setupGBuffers"));

            pipeline.programs["miss"] = p.createMissProgram(
                m, RT_MS_NAME_STR("setupGBuffers"));

            pipeline.hitPrograms["triangle"] = p.createHitProgramGroupForTriangleIS(
                m, RT_CH_NAME_STR("setupGBuffers"),
                emptyModule, nullptr);
            pipeline.hitPrograms["aabb"] = p.createHitProgramGroupForCustomIS(
                m, RT_CH_NAME_STR("setupGBuffers"),
                emptyModule, nullptr,
                m, RT_IS_NAME_STR("aabb"));
            pipeline.hitPrograms["displacedSurface"] = p.createHitProgramGroupForCustomIS(
                m, RT_CH_NAME_STR("setupGBuffers"),
                emptyModule, nullptr,
                m, RT_IS_NAME_STR("displacedSurface"));

            pipeline.hitPrograms["emptyHitGroup"] = p.createEmptyHitProgramGroup();

            pipeline.setEntryPoint(GBufferEntryPoint::setupGBuffers);
            p.setNumMissRayTypes(shared::GBufferRayType::NumTypes);
            p.setMissProgram(shared::GBufferRayType::Primary, pipeline.programs.at("miss"));

            p.setNumCallablePrograms(NumCallablePrograms);
            pipeline.callablePrograms.resize(NumCallablePrograms);
            for (int i = 0; i < NumCallablePrograms; ++i) {
                optixu::CallableProgramGroup program = p.createCallableProgramGroup(
                    m, callableProgramEntryPoints[i],
                    emptyModule, nullptr);
                pipeline.callablePrograms[i] = program;
                p.setCallableProgram(i, program);
            }

            p.link(1);

            uint32_t maxDcStackSize = 0;
            for (int i = 0; i < NumCallablePrograms; ++i) {
                optixu::CallableProgramGroup program = pipeline.callablePrograms[i];
                maxDcStackSize = std::max(maxDcStackSize, program.getDCStackSize());
            }
            uint32_t maxCcStackSize =
                pipeline.entryPoints.at(GBufferEntryPoint::setupGBuffers).getStackSize() +
                std::max(
                    {
                        pipeline.hitPrograms.at("triangle").getCHStackSize(),
                        pipeline.hitPrograms.at("aabb").getCHStackSize(),
                        pipeline.hitPrograms.at("displacedSurface").getCHStackSize(),
                        pipeline.programs.at("miss").getStackSize()
                    });

            p.setStackSize(0, maxDcStackSize, maxCcStackSize, 2);

            optixDefaultMaterial.setHitGroup(shared::GBufferRayType::Primary, pipeline.hitPrograms.at("triangle"));
            for (uint32_t rayType = shared::GBufferRayType::NumTypes; rayType < shared::maxNumRayTypes; ++rayType)
                optixDefaultMaterial.setHitGroup(rayType, pipeline.hitPrograms.at("emptyHitGroup"));

#if USE_DISPLACED_SURFACES
            optixTFDMMaterial.setHitGroup(
                shared::GBufferRayType::Primary, pipeline.hitPrograms.at("displacedSurface"));
#else
            optixTFDMMaterial.setHitGroup(
                shared::GBufferRayType::Primary, pipeline.hitPrograms.at("aabb"));
#endif
            for (uint32_t rayType = shared::GBufferRayType::NumTypes; rayType < shared::maxNumRayTypes; ++rayType)
                optixTFDMMaterial.setHitGroup(rayType, pipeline.hitPrograms.at("emptyHitGroup"));

            size_t sbtSize;
            p.generateShaderBindingTableLayout(&sbtSize);
            pipeline.sbt.initialize(cuContext, Scene::bufferType, sbtSize, 1);
            pipeline.sbt.setMappedMemoryPersistent(true);
            p.setShaderBindingTable(pipeline.sbt, pipeline.sbt.getMappedPointer());
        }

        {
            Pipeline<PathTracingEntryPoint> &pipeline = pathTracing;
            optixu::Pipeline &p = pipeline.optixPipeline;
            optixu::Module &m = pipeline.optixModule;
            p = optixContext.createPipeline();

            p.setPipelineOptions(
                std::max({
                    shared::PathTraceRayPayloadSignature::numDwords,
                    shared::VisibilityRayPayloadSignature::numDwords
                         }),
                std::max({
                    static_cast<uint32_t>(optixu::calcSumDwords<float2>()),
                    shared::AABBAttributeSignature::numDwords,
                    shared::DisplacedSurfaceAttributeSignature::numDwords
                         }),
                "plp", sizeof(shared::PipelineLaunchParameters),
                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM);

            m = p.createModuleFromPTXString(
                readTxtFile(getExecutableDirectory() / "tfdm/ptxes/optix_pathtracing_kernels.ptx"),
                OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
                DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            pipeline.entryPoints[PathTracingEntryPoint::pathTrace] =
                p.createRayGenProgram(m, RT_RG_NAME_STR("pathTrace"));

            pipeline.programs[RT_MS_NAME_STR("pathTrace")] = p.createMissProgram(
                m, RT_MS_NAME_STR("pathTrace"));

            pipeline.hitPrograms[RT_CH_NAME_STR("pathTraceTriangle")] = p.createHitProgramGroupForTriangleIS(
                m, RT_CH_NAME_STR("pathTrace"),
                emptyModule, nullptr);
            pipeline.hitPrograms[RT_CH_NAME_STR("pathTraceAabb")] = p.createHitProgramGroupForCustomIS(
                m, RT_CH_NAME_STR("pathTrace"),
                emptyModule, nullptr,
                m, RT_IS_NAME_STR("aabb"));
            pipeline.hitPrograms[RT_CH_NAME_STR("pathTraceDisplacedSurface")] = p.createHitProgramGroupForCustomIS(
                m, RT_CH_NAME_STR("pathTrace"),
                emptyModule, nullptr,
                m, RT_IS_NAME_STR("displacedSurface"));

            pipeline.hitPrograms[RT_AH_NAME_STR("visibilityTriangle")] = p.createHitProgramGroupForTriangleIS(
                emptyModule, nullptr,
                m, RT_AH_NAME_STR("visibility"));
            pipeline.hitPrograms[RT_AH_NAME_STR("visibilityAabb")] = p.createHitProgramGroupForCustomIS(
                emptyModule, nullptr,
                m, RT_AH_NAME_STR("visibility"),
                m, RT_IS_NAME_STR("aabb"));
            pipeline.hitPrograms[RT_AH_NAME_STR("visibilityDisplacedSurface")] = p.createHitProgramGroupForCustomIS(
                emptyModule, nullptr,
                m, RT_AH_NAME_STR("visibility"),
                m, RT_IS_NAME_STR("displacedSurface"));

            pipeline.programs["emptyMiss"] = p.createMissProgram(emptyModule, nullptr);

            p.setNumMissRayTypes(shared::PathTracingRayType::NumTypes);
            p.setMissProgram(
                shared::PathTracingRayType::Baseline, pipeline.programs.at(RT_MS_NAME_STR("pathTrace")));
            p.setMissProgram(shared::PathTracingRayType::Visibility, pipeline.programs.at("emptyMiss"));

            p.setNumCallablePrograms(NumCallablePrograms);
            pipeline.callablePrograms.resize(NumCallablePrograms);
            for (int i = 0; i < NumCallablePrograms; ++i) {
                optixu::CallableProgramGroup program = p.createCallableProgramGroup(
                    m, callableProgramEntryPoints[i],
                    emptyModule, nullptr);
                pipeline.callablePrograms[i] = program;
                p.setCallableProgram(i, program);
            }

            p.link(2);

            uint32_t maxDcStackSize = 0;
            for (int i = 0; i < NumCallablePrograms; ++i) {
                optixu::CallableProgramGroup program = pipeline.callablePrograms[i];
                maxDcStackSize = std::max(maxDcStackSize, program.getDCStackSize());
            }
            uint32_t maxCcStackSize =
                pipeline.entryPoints.at(PathTracingEntryPoint::pathTrace).getStackSize() +
                std::max(
                    {
                        std::max({
                            pipeline.hitPrograms.at(RT_CH_NAME_STR("pathTraceTriangle")).getCHStackSize(),
                            pipeline.hitPrograms.at(RT_CH_NAME_STR("pathTraceAabb")).getCHStackSize(),
                            pipeline.hitPrograms.at(RT_CH_NAME_STR("pathTraceDisplacedSurface")).getCHStackSize()
                                 }) +
                        std::max({
                            pipeline.hitPrograms.at(RT_AH_NAME_STR("visibilityTriangle")).getAHStackSize(),
                            pipeline.hitPrograms.at(RT_AH_NAME_STR("visibilityAabb")).getAHStackSize(),
                            pipeline.hitPrograms.at(RT_AH_NAME_STR("visibilityDisplacedSurface")).getAHStackSize(),
                                 }),
                        pipeline.programs.at(RT_MS_NAME_STR("pathTrace")).getStackSize()
                    });

            p.setStackSize(0, maxDcStackSize, maxCcStackSize, 2);

            optixDefaultMaterial.setHitGroup(
                shared::PathTracingRayType::Baseline,
                pipeline.hitPrograms.at(RT_CH_NAME_STR("pathTraceTriangle")));
            optixDefaultMaterial.setHitGroup(
                shared::PathTracingRayType::Visibility,
                pipeline.hitPrograms.at(RT_AH_NAME_STR("visibilityTriangle")));

#if USE_DISPLACED_SURFACES
            optixTFDMMaterial.setHitGroup(
                shared::PathTracingRayType::Baseline,
                pipeline.hitPrograms.at(RT_CH_NAME_STR("pathTraceDisplacedSurface")));
            optixTFDMMaterial.setHitGroup(
                shared::PathTracingRayType::Visibility,
                pipeline.hitPrograms.at(RT_AH_NAME_STR("visibilityDisplacedSurface")));
#else
            optixTFDMMaterial.setHitGroup(
                shared::PathTracingRayType::Baseline,
                pipeline.hitPrograms.at(RT_CH_NAME_STR("pathTraceAabb")));
            optixTFDMMaterial.setHitGroup(
                shared::PathTracingRayType::Visibility,
                pipeline.hitPrograms.at(RT_AH_NAME_STR("visibilityAabb")));
#endif

            size_t sbtSize;
            p.generateShaderBindingTableLayout(&sbtSize);
            pipeline.sbt.initialize(cuContext, Scene::bufferType, sbtSize, 1);
            pipeline.sbt.setMappedMemoryPersistent(true);
            p.setShaderBindingTable(pipeline.sbt, pipeline.sbt.getMappedPointer());
        }
    }

    void finalize() {
        {
            Pipeline<PathTracingEntryPoint> &pipeline = pathTracing;
            pipeline.hitGroupSbt.finalize();
            pipeline.sbt.finalize();
            for (int i = 0; i < NumCallablePrograms; ++i)
                pipeline.callablePrograms[i].destroy();
            for (auto &pair : pipeline.programs)
                pair.second.destroy();
            for (auto &pair : pipeline.entryPoints)
                pair.second.destroy();
            pipeline.optixModule.destroy();
            pipeline.optixPipeline.destroy();
        }

        {
            Pipeline<GBufferEntryPoint> &pipeline = gBuffer;
            pipeline.hitGroupSbt.finalize();
            pipeline.sbt.finalize();
            for (int i = 0; i < NumCallablePrograms; ++i)
                pipeline.callablePrograms[i].destroy();
            for (auto &pair : pipeline.programs)
                pair.second.destroy();
            for (auto &pair : pipeline.entryPoints)
                pair.second.destroy();
            pipeline.optixModule.destroy();
            pipeline.optixPipeline.destroy();
        }

        optixTFDMMaterial.destroy();
        optixDefaultMaterial.destroy();

        optixContext.destroy();

        CUDADRV_CHECK(cuModuleUnload(tfdmModule));

        CUDADRV_CHECK(cuCtxDestroy(cuContext));
    }
};



struct KeyState {
    uint64_t timesLastChanged[5];
    bool statesLastChanged[5];
    uint32_t lastIndex;

    KeyState() : lastIndex(0) {
        for (int i = 0; i < 5; ++i) {
            timesLastChanged[i] = 0;
            statesLastChanged[i] = false;
        }
    }

    void recordStateChange(bool state, uint64_t time) {
        bool lastState = statesLastChanged[lastIndex];
        if (state == lastState)
            return;

        lastIndex = (lastIndex + 1) % 5;
        statesLastChanged[lastIndex] = !lastState;
        timesLastChanged[lastIndex] = time;
    }

    bool getState(int32_t goBack = 0) const {
        Assert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return statesLastChanged[(lastIndex + goBack + 5) % 5];
    }

    uint64_t getTime(int32_t goBack = 0) const {
        Assert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return timesLastChanged[(lastIndex + goBack + 5) % 5];
    }
};

static KeyState g_keyForward;
static KeyState g_keyBackward;
static KeyState g_keyLeftward;
static KeyState g_keyRightward;
static KeyState g_keyUpward;
static KeyState g_keyDownward;
static KeyState g_keyTiltLeft;
static KeyState g_keyTiltRight;
static KeyState g_keyFasterPosMovSpeed;
static KeyState g_keySlowerPosMovSpeed;
static KeyState g_keyDebugPrint;
static KeyState g_buttonRotate;
static double g_mouseX;
static double g_mouseY;

static float g_initBrightness = 0.0f;
static float g_cameraPositionalMovingSpeed;
static float g_cameraDirectionalMovingSpeed;
static float g_cameraTiltSpeed;
static Quaternion g_cameraOrientation;
static Quaternion g_tempCameraOrientation;
static Point3D g_cameraPosition;
static std::filesystem::path g_envLightTexturePath;

static bool g_takeScreenShot = false;

struct MeshGeometryInfo {
    std::filesystem::path path;
    float preScale;
    MaterialConvention matConv;
};

struct RectangleGeometryInfo {
    float dimX;
    float dimZ;
    RGB emittance;
    std::filesystem::path emitterTexPath;
};

struct MeshInstanceInfo {
    std::string name;
    Point3D beginPosition;
    Point3D endPosition;
    float beginScale;
    float endScale;
    Quaternion beginOrientation;
    Quaternion endOrientation;
    float frequency;
    float initTime;
};

using MeshInfo = std::variant<MeshGeometryInfo, RectangleGeometryInfo>;
static std::map<std::string, MeshInfo> g_meshInfos;
static std::vector<MeshInstanceInfo> g_meshInstInfos;

static void parseCommandline(int32_t argc, const char* argv[]) {
    std::string name;

    Quaternion camOrientation = Quaternion();

    Point3D beginPosition(0.0f, 0.0f, 0.0f);
    Point3D endPosition(NAN, NAN, NAN);
    Quaternion beginOrientation = Quaternion();
    Quaternion endOrientation = Quaternion(NAN, NAN, NAN, NAN);
    float beginScale = 1.0f;
    float endScale = NAN;
    float frequency = 5.0f;
    float initTime = 0.0f;
    RGB emittance(0.0f, 0.0f, 0.0f);
    std::filesystem::path rectEmitterTexPath;

    for (int i = 0; i < argc; ++i) {
        const char* arg = argv[i];

        const auto computeOrientation = [&argc, &argv, &i](const char* arg, Quaternion* ori) {
            if (!allFinite(*ori))
                *ori = Quaternion();
            if (strncmp(arg, "-roll", 6) == 0) {
                if (i + 1 >= argc) {
                    hpprintf("Invalid option.\n");
                    exit(EXIT_FAILURE);
                }
                *ori = qRotateZ(static_cast<float>(atof(argv[i + 1])) * pi_v<float> / 180) * *ori;
                i += 1;
            }
            else if (strncmp(arg, "-pitch", 7) == 0) {
                if (i + 1 >= argc) {
                    hpprintf("Invalid option.\n");
                    exit(EXIT_FAILURE);
                }
                *ori = qRotateX(static_cast<float>(atof(argv[i + 1])) * pi_v<float> / 180) * *ori;
                i += 1;
            }
            else if (strncmp(arg, "-yaw", 5) == 0) {
                if (i + 1 >= argc) {
                    hpprintf("Invalid option.\n");
                    exit(EXIT_FAILURE);
                }
                *ori = qRotateY(static_cast<float>(atof(argv[i + 1])) * pi_v<float> / 180) * *ori;
                i += 1;
            }
        };

        if (strncmp(arg, "-", 1) != 0)
            continue;

        if (strncmp(arg, "-screenshot", 12) == 0) {
            g_takeScreenShot = true;
        }
        else if (strncmp(arg, "-cam-pos", 9) == 0) {
            if (i + 3 >= argc) {
                hpprintf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            g_cameraPosition = Point3D(atof(argv[i + 1]), atof(argv[i + 2]), atof(argv[i + 3]));
            i += 3;
        }
        else if (strncmp(arg, "-cam-roll", 10) == 0 ||
                 strncmp(arg, "-cam-pitch", 11) == 0 ||
                 strncmp(arg, "-cam-yaw", 9) == 0) {
            computeOrientation(arg + 4, &camOrientation);
        }
        else if (strncmp(arg, "-brightness", 12) == 0) {
            if (i + 1 >= argc) {
                hpprintf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            g_initBrightness = std::fmin(std::fmax(std::atof(argv[i + 1]), -5.0f), 5.0f);
            i += 1;
        }
        else if (strncmp(arg, "-env-texture", 13) == 0) {
            if (i + 1 >= argc) {
                hpprintf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            g_envLightTexturePath = argv[i + 1];
            i += 1;
        }
        else if (strncmp(arg, "-name", 6) == 0) {
            if (i + 1 >= argc) {
                hpprintf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            name = argv[i + 1];
            i += 1;
        }
        else if (0 == strncmp(arg, "-emittance", 11)) {
            if (i + 3 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            emittance = RGB(atof(argv[i + 1]), atof(argv[i + 2]), atof(argv[i + 3]));
            if (!emittance.allFinite()) {
                printf("Invalid value.\n");
                exit(EXIT_FAILURE);
            }
            i += 3;
        }
        else if (0 == strncmp(arg, "-rect-emitter-tex", 18)) {
            if (i + 1 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            rectEmitterTexPath = argv[i + 1];
            i += 1;
        }
        else if (0 == strncmp(arg, "-obj", 5)) {
            if (i + 3 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }

            MeshInfo info = MeshGeometryInfo();
            auto &mesh = std::get<MeshGeometryInfo>(info);
            mesh.path = std::filesystem::path(argv[i + 1]);
            mesh.preScale = atof(argv[i + 2]);
            std::string matConv = argv[i + 3];
            if (matConv == "trad") {
                mesh.matConv = MaterialConvention::Traditional;
            }
            else if (matConv == "simple_pbr") {
                mesh.matConv = MaterialConvention::SimplePBR;
            }
            else {
                printf("Invalid material convention.\n");
                exit(EXIT_FAILURE);
            }

            g_meshInfos[name] = info;

            i += 3;
        }
        else if (0 == strncmp(arg, "-rectangle", 11)) {
            if (i + 2 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }

            MeshInfo info = RectangleGeometryInfo();
            auto &rect = std::get<RectangleGeometryInfo>(info);
            rect.dimX = atof(argv[i + 1]);
            rect.dimZ = atof(argv[i + 2]);
            rect.emittance = emittance;
            rect.emitterTexPath = rectEmitterTexPath;
            g_meshInfos[name] = info;

            emittance = RGB(0.0f, 0.0f, 0.0f);
            rectEmitterTexPath = "";

            i += 2;
        }
        else if (0 == strncmp(arg, "-begin-pos", 11)) {
            if (i + 3 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            beginPosition = Point3D(atof(argv[i + 1]), atof(argv[i + 2]), atof(argv[i + 3]));
            if (!beginPosition.allFinite()) {
                printf("Invalid value.\n");
                exit(EXIT_FAILURE);
            }
            i += 3;
        }
        else if (strncmp(arg, "-begin-roll", 10) == 0 ||
                 strncmp(arg, "-begin-pitch", 11) == 0 ||
                 strncmp(arg, "-begin-yaw", 9) == 0) {
            computeOrientation(arg + 6, &beginOrientation);
        }
        else if (0 == strncmp(arg, "-begin-scale", 13)) {
            if (i + 1 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            beginScale = atof(argv[i + 1]);
            if (!isfinite(beginScale)) {
                printf("Invalid value.\n");
                exit(EXIT_FAILURE);
            }
            i += 1;
        }
        else if (0 == strncmp(arg, "-end-pos", 9)) {
            if (i + 3 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            endPosition = Point3D(atof(argv[i + 1]), atof(argv[i + 2]), atof(argv[i + 3]));
            if (!endPosition.allFinite()) {
                printf("Invalid value.\n");
                exit(EXIT_FAILURE);
            }
            i += 3;
        }
        else if (strncmp(arg, "-end-roll", 10) == 0 ||
                 strncmp(arg, "-end-pitch", 11) == 0 ||
                 strncmp(arg, "-end-yaw", 9) == 0) {
            computeOrientation(arg + 4, &endOrientation);
        }
        else if (0 == strncmp(arg, "-end-scale", 11)) {
            if (i + 1 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            endScale = atof(argv[i + 1]);
            if (!isfinite(endScale)) {
                printf("Invalid value.\n");
                exit(EXIT_FAILURE);
            }
            i += 1;
        }
        else if (0 == strncmp(arg, "-freq", 6)) {
            if (i + 1 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            frequency = atof(argv[i + 1]);
            if (!isfinite(frequency)) {
                printf("Invalid value.\n");
                exit(EXIT_FAILURE);
            }
            i += 1;
        }
        else if (0 == strncmp(arg, "-time", 6)) {
            if (i + 1 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            initTime = atof(argv[i + 1]);
            if (!isfinite(initTime)) {
                printf("Invalid value.\n");
                exit(EXIT_FAILURE);
            }
            i += 1;
        }
        else if (0 == strncmp(arg, "-inst", 6)) {
            if (i + 1 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }

            MeshInstanceInfo info;
            info.name = argv[i + 1];
            info.beginPosition = beginPosition;
            info.beginOrientation = beginOrientation;
            info.beginScale = beginScale;
            info.endPosition = endPosition.allFinite() ? endPosition : beginPosition;
            info.endOrientation = endOrientation.allFinite() ? endOrientation : beginOrientation;
            info.endScale = std::isfinite(endScale) ? endScale : beginScale;
            info.frequency = frequency;
            info.initTime = initTime;
            g_meshInstInfos.push_back(info);

            beginPosition = Point3D(0.0f, 0.0f, 0.0f);
            endPosition = Point3D(NAN, NAN, NAN);
            beginOrientation = Quaternion();
            endOrientation = Quaternion(NAN, NAN, NAN, NAN);
            beginScale = 1.0f;
            endScale = NAN;
            frequency = 5.0f;
            initTime = 0.0f;

            i += 1;
        }
        else {
            printf("Unknown option.\n");
            exit(EXIT_FAILURE);
        }
    }

    g_cameraOrientation = camOrientation;
}



static void glfw_error_callback(int32_t error, const char* description) {
    hpprintf("Error %d: %s\n", error, description);
}



namespace ImGui {
    template <typename EnumType>
    bool RadioButtonE(const char* label, EnumType* v, EnumType v_button) {
        return RadioButton(label, reinterpret_cast<int*>(v), static_cast<int>(v_button));
    }

    bool InputLog2Int(const char* label, int* v, int max_v, int num_digits = 3) {
        float buttonSize = GetFrameHeight();
        float itemInnerSpacingX = GetStyle().ItemInnerSpacing.x;

        BeginGroup();
        PushID(label);

        ImGui::AlignTextToFramePadding();
        SetNextItemWidth(std::max(1.0f, CalcItemWidth() - (buttonSize + itemInnerSpacingX) * 2));
        Text("%s: %*u", label, num_digits, 1 << *v);
        bool changed = false;
        SameLine(0, itemInnerSpacingX);
        if (Button("-", ImVec2(buttonSize, buttonSize))) {
            *v = std::max(*v - 1, 0);
            changed = true;
        }
        SameLine(0, itemInnerSpacingX);
        if (Button("+", ImVec2(buttonSize, buttonSize))) {
            *v = std::min(*v + 1, max_v);
            changed = true;
        }

        PopID();
        EndGroup();

        return changed;
    }
}

int32_t main(int32_t argc, const char* argv[]) try {
    const std::filesystem::path exeDir = getExecutableDirectory();

    parseCommandline(argc, argv);

    // ----------------------------------------------------------------
    // JP: OpenGL, GLFWの初期化。
    // EN: Initialize OpenGL and GLFW.

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        hpprintf("Failed to initialize GLFW.\n");
        return -1;
    }

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    constexpr bool enableGLDebugCallback = DEBUG_SELECT(true, false);

    // JP: OpenGL 4.6 Core Profileのコンテキストを作成する。
    // EN: Create an OpenGL 4.6 core profile context.
    const uint32_t OpenGLMajorVersion = 4;
    const uint32_t OpenGLMinorVersion = 6;
    const char* glsl_version = "#version 460";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OpenGLMajorVersion);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OpenGLMinorVersion);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if constexpr (enableGLDebugCallback)
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    int32_t renderTargetSizeX = 1920;
    int32_t renderTargetSizeY = 1080;

    // JP: ウインドウの初期化。
    // EN: Initialize a window.
    float contentScaleX, contentScaleY;
    glfwGetMonitorContentScale(monitor, &contentScaleX, &contentScaleY);
    float UIScaling = contentScaleX;
    GLFWwindow* window = glfwCreateWindow(
        static_cast<int32_t>(renderTargetSizeX * UIScaling),
        static_cast<int32_t>(renderTargetSizeY * UIScaling),
        "TFDM", NULL, NULL);
    glfwSetWindowUserPointer(window, nullptr);
    if (!window) {
        hpprintf("Failed to create a GLFW window.\n");
        glfwTerminate();
        return -1;
    }

    int32_t curFBWidth;
    int32_t curFBHeight;
    glfwGetFramebufferSize(window, &curFBWidth, &curFBHeight);

    glfwMakeContextCurrent(window);

    glfwSwapInterval(1); // Enable vsync



    // JP: gl3wInit()は何らかのOpenGLコンテキストが作られた後に呼ぶ必要がある。
    // EN: gl3wInit() must be called after some OpenGL context has been created.
    int32_t gl3wRet = gl3wInit();
    if (!gl3wIsSupported(OpenGLMajorVersion, OpenGLMinorVersion)) {
        hpprintf("gl3w doesn't support OpenGL %u.%u\n", OpenGLMajorVersion, OpenGLMinorVersion);
        glfwTerminate();
        return -1;
    }

    if constexpr (enableGLDebugCallback) {
        glu::enableDebugCallback(true);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, false);
    }

    // END: Initialize OpenGL and GLFW.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: 入力コールバックの設定。
    // EN: Set up input callbacks.

    glfwSetMouseButtonCallback(
        window,
        [](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
            uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);

            switch (button) {
            case GLFW_MOUSE_BUTTON_MIDDLE: {
                devPrintf("Mouse Middle\n");
                g_buttonRotate.recordStateChange(action == GLFW_PRESS, frameIndex);
                break;
            }
            default:
                break;
            }
        });
    glfwSetCursorPosCallback(
        window,
        [](GLFWwindow* window, double x, double y) {
            g_mouseX = x;
            g_mouseY = y;
        });
    glfwSetKeyCallback(
        window,
        [](GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
            uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);

            switch (key) {
            case GLFW_KEY_W: {
                g_keyForward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_S: {
                g_keyBackward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_A: {
                g_keyLeftward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_D: {
                g_keyRightward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_R: {
                g_keyUpward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_F: {
                g_keyDownward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_Q: {
                g_keyTiltLeft.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_E: {
                g_keyTiltRight.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_T: {
                g_keyFasterPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_G: {
                g_keySlowerPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_P: {
                g_keyDebugPrint.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            default:
                break;
            }
        });

    // END: Set up input callbacks.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: ImGuiの初期化。
    // EN: Initialize ImGui.

    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Setup style
    // JP: ガンマ補正が有効なレンダーターゲットで、同じUIの見た目を得るためにデガンマされたスタイルも用意する。
    // EN: Prepare a degamma-ed style to have the identical UI appearance on gamma-corrected render target.
    ImGuiStyle guiStyle, guiStyleWithGamma;
    ImGui::StyleColorsDark(&guiStyle);
    guiStyleWithGamma = guiStyle;
    const auto degamma = [](const ImVec4 &color) {
        return ImVec4(sRGB_degamma_s(color.x),
                      sRGB_degamma_s(color.y),
                      sRGB_degamma_s(color.z),
                      color.w);
    };
    for (int i = 0; i < ImGuiCol_COUNT; ++i) {
        guiStyleWithGamma.Colors[i] = degamma(guiStyleWithGamma.Colors[i]);
    }
    ImGui::GetStyle() = guiStyleWithGamma;

    // END: Initialize ImGui.
    // ----------------------------------------------------------------



    GPUEnvironment gpuEnv;
    gpuEnv.initialize();

    Scene scene;
    scene.initialize(
        getExecutableDirectory() / "tfdm/ptxes",
        gpuEnv.cuContext, gpuEnv.optixContext, shared::maxNumRayTypes);
    cudau::TypedBuffer<shared::TFDMData> tfdmDataBuffer(
        gpuEnv.cuContext, Scene::bufferType, Scene::maxNumGeometryInstances);

    StreamChain<2> streamChain;
    streamChain.initialize(gpuEnv.cuContext);
    CUstream stream = streamChain.waitAvailableAndGetCurrentStream();

    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    scene.map();
    tfdmDataBuffer.map();

    for (auto it = g_meshInfos.cbegin(); it != g_meshInfos.cend(); ++it) {
        const MeshInfo &info = it->second;

        if (std::holds_alternative<MeshGeometryInfo>(info)) {
            const auto &meshInfo = std::get<MeshGeometryInfo>(info);

            createTriangleMeshes(
                it->first,
                meshInfo.path, meshInfo.matConv,
                scale4x4(meshInfo.preScale),
                gpuEnv.cuContext, &scene, gpuEnv.optixDefaultMaterial);
        }
        else if (std::holds_alternative<RectangleGeometryInfo>(info)) {
            const auto &rectInfo = std::get<RectangleGeometryInfo>(info);

            createRectangleLight(
                it->first,
                rectInfo.dimX, rectInfo.dimZ,
                RGB(0.01f),
                rectInfo.emitterTexPath, rectInfo.emittance, Matrix4x4(),
                gpuEnv.cuContext, &scene, gpuEnv.optixDefaultMaterial);
        }
    }

    for (int i = 0; i < g_meshInstInfos.size(); ++i) {
        const MeshInstanceInfo &info = g_meshInstInfos[i];
        const Mesh* mesh = scene.meshes.at(info.name);
        for (int j = 0; j < mesh->groupInsts.size(); ++j) {
            const Mesh::GeometryGroupInstance &groupInst = mesh->groupInsts[j];

            Matrix4x4 instXfm =
                Matrix4x4(info.beginScale * info.beginOrientation.toMatrix3x3(), info.beginPosition);
            Instance* inst = createInstance(gpuEnv.cuContext, &scene, groupInst, instXfm);
            scene.insts.push_back(inst);

            scene.initialSceneAabb.unify(instXfm * groupInst.transform * groupInst.geomGroup->aabb);

            if (any(info.beginPosition != info.endPosition) ||
                info.beginOrientation != info.endOrientation ||
                info.beginScale != info.endScale) {
                auto controller = new InstanceController(
                    inst,
                    info.beginScale, info.beginOrientation, info.beginPosition,
                    info.endScale, info.endOrientation, info.endPosition,
                    info.frequency, info.initTime);
                scene.instControllers.push_back(controller);
            }
        }
    }

    {
        createLambertMaterial(
            gpuEnv.cuContext, &scene, "", RGB(0.8f, 0.8f, 0.8f), "", "", RGB(0.0f));
        Material* material = scene.materials.back();

        std::vector<shared::Vertex> vertices = {
            shared::Vertex{Point3D(-1.0f, 0.0f, -1.0f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), Point2D(0.0f, 0.0f)},
            shared::Vertex{Point3D(1.0f, 0.0f, -1.0f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), Point2D(1.0f, 0.0f)},
            shared::Vertex{Point3D(1.0f, 0.0f, 1.0f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), Point2D(1.0f, 1.0f)},
            shared::Vertex{Point3D(-1.0f, 0.0f, 1.0f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), Point2D(0.0f, 1.0f)},
        };
        std::vector<shared::Triangle> triangles = {
            shared::Triangle{0, 1, 2},
            shared::Triangle{0, 2, 3},
        };
        GeometryInstance* geomInst = createGeometryInstance(
            gpuEnv.cuContext, &scene, vertices, triangles, material, gpuEnv.optixDefaultMaterial, false);
        scene.geomInsts.push_back(geomInst);

        std::set<const GeometryInstance*> srcGeomInsts = { geomInst };
        GeometryGroup* geomGroup = createGeometryGroup(&scene, srcGeomInsts);
        scene.geomGroups.push_back(geomGroup);

        auto mesh = new Mesh();
        {
            Mesh::GeometryGroupInstance groupInst = {};
            groupInst.geomGroup = geomGroup;
            groupInst.transform = Matrix4x4();
            mesh->groupInsts.clear();
            mesh->groupInsts.push_back(groupInst);
            scene.meshes["floor"] = mesh;
        }

        Matrix4x4 instXfm = translate4x4<float>(0, -1.0f, 0);
        Instance* inst = createInstance(gpuEnv.cuContext, &scene, mesh->groupInsts[0], instXfm);
        scene.insts.push_back(inst);

        scene.initialSceneAabb.unify(
            instXfm * mesh->groupInsts[0].transform * mesh->groupInsts[0].geomGroup->aabb);
    }

    float heightOffset = 0.0f;
    float heightScale = 0.2f;
    float heightBias = 0.0f;

#define SHOW_BASE_MESH 0
    Material* tfdmMeshMaterial;
    GeometryInstance* tfdmMeshGeomInst;
    GeometryGroup* tfdmGeomGroup;
    {
        createLambertMaterial(
            gpuEnv.cuContext, &scene, "", RGB(0.8f, 0.8f, 0.8f), "", "", RGB(0.0f));
        tfdmMeshMaterial = scene.materials.back();

        constexpr uint32_t numEdges = 1;
        std::vector<shared::Vertex> vertices(pow2(numEdges + 1));
        std::vector<shared::Triangle> triangles(2 * pow2(numEdges));
        for (int iy = 0; iy < numEdges + 1; ++iy) {
            float py = static_cast<float>(iy) / numEdges;
            float y = -0.5f + 1.0f * py;
            for (int ix = 0; ix < numEdges + 1; ++ix) {
                float px = static_cast<float>(ix) / numEdges;
                float x = -0.5f + 1.0f * px;
                vertices[iy * (numEdges + 1) + ix] = shared::Vertex{
                    Point3D(x, 0, y),
                    normalize(Normal3D(/*-0.5f + px*/0, 1, /*-0.5f + py*/0)),
                    Vector3D(1, 0, 0), Point2D(px, py)
                };
                if (iy < numEdges && ix < numEdges) {
                    uint32_t baseIdx = iy * (numEdges + 1) + ix;
                    triangles[2 * (iy * numEdges + ix) + 0] = shared::Triangle{
                        baseIdx, baseIdx + (numEdges + 1) + 1, baseIdx + 1
                    };
                    triangles[2 * (iy * numEdges + ix) + 1] = shared::Triangle{
                        baseIdx, baseIdx + (numEdges + 1), baseIdx + (numEdges + 1) + 1
                    };
                }
            }
        }
#if SHOW_BASE_MESH
        tfdmMeshGeomInst = createGeometryInstance(
            gpuEnv.cuContext, &scene, vertices, triangles, tfdmMeshMaterial, gpuEnv.optixDefaultMaterial, false);
#else
        tfdmMeshGeomInst = createTFDMGeometryInstance(
            gpuEnv.cuContext, &scene, vertices, triangles, tfdmMeshMaterial, gpuEnv.optixTFDMMaterial);
#endif
        scene.geomInsts.push_back(tfdmMeshGeomInst);
        {
            shared::MaterialData &matDataOnHost =
                scene.materialDataBuffer.getMappedPointer()[tfdmMeshMaterial->materialSlot];

            const std::filesystem::path heightMapPath =
                R"(..\data\TCom_Rock_Cliff3_2x2_1K_height.dds)";
            bool needsDegamma;
            hpprintf("Reading: %s ... ", heightMapPath.string().c_str());
            if (loadTexture<float, true>(
                heightMapPath, 0.0f, gpuEnv.cuContext, &tfdmMeshMaterial->heightMap, &needsDegamma))
                hpprintf("done.\n");
            else
                hpprintf("failed.\n");

            cudau::TextureSampler sampler = {};
            sampler.setXyFilterMode(cudau::TextureFilterMode::Linear);
            sampler.setWrapMode(0, cudau::TextureWrapMode::Repeat);
            sampler.setWrapMode(1, cudau::TextureWrapMode::Repeat);
            sampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
            sampler.setReadMode(cudau::TextureReadMode::NormalizedFloat);
            tfdmMeshMaterial->heightMapTex = sampler.createTextureObject(*tfdmMeshMaterial->heightMap);
            matDataOnHost.heightMapSize = int2(
                tfdmMeshMaterial->heightMap->getWidth(), tfdmMeshMaterial->heightMap->getHeight());
            matDataOnHost.heightMap = tfdmMeshMaterial->heightMapTex;

            if (matDataOnHost.heightMapSize.x != matDataOnHost.heightMapSize.y)
                throw std::runtime_error("Non-square height map is not supported.");
            if (popcnt(matDataOnHost.heightMapSize.x) != 1)
                throw std::runtime_error("Non-power-of-two height map is not supported.");
            const uint32_t numMinMaxMipMapLevels = nextPowOf2Exponent(matDataOnHost.heightMapSize.x) + 1;
            tfdmMeshMaterial->minMaxMipMap.initialize2D(
                gpuEnv.cuContext,
                cudau::ArrayElementType::Float32, 2,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                matDataOnHost.heightMapSize.x, matDataOnHost.heightMapSize.y,
                numMinMaxMipMapLevels);
            std::vector<optixu::NativeBlockBuffer2D<float2>> surfObjs(numMinMaxMipMapLevels);
            for (int mipLevel = 0; mipLevel < numMinMaxMipMapLevels; ++mipLevel)
                surfObjs[mipLevel] = tfdmMeshMaterial->minMaxMipMap.getSurfaceObject(mipLevel);
            tfdmMeshMaterial->minMaxMipMapSurfs.initialize(gpuEnv.cuContext, cudau::BufferType::Device, surfObjs);
            matDataOnHost.minMaxMipMap = tfdmMeshMaterial->minMaxMipMapSurfs.getDevicePointer();

            std::vector<shared::DisplacedTriangleAuxInfo> dispTriAuxInfos(triangles.size());
            for (int triIdx = 0; triIdx < triangles.size(); ++triIdx) {
                const shared::Triangle &tri = triangles[triIdx];
                shared::DisplacedTriangleAuxInfo &dispTriAuxInfo = dispTriAuxInfos[triIdx];

                using Vector2Dd = Vector2D_T<double>;
                using Vector3Dd = Vector3D_T<double, false>;
                using Point3Dd = Point3D_T<double>;
                using Matrix3x3d = Matrix3x3_T<double>;
                using Matrix4x4d = Matrix4x4_T<double>;

                const shared::Vertex (&vs)[] = {
                    vertices[tri.index0],
                    vertices[tri.index1],
                    vertices[tri.index2]
                };

                Vector3Dd tc0Dir;
                Vector3Dd tc1Dir;
                {
                    Vector3Dd dp01 = vs[1].position - vs[0].position;
                    Vector3Dd dp02 = vs[2].position - vs[0].position;
                    Vector2Dd dt01 = vs[1].texCoord - vs[0].texCoord;
                    Vector2Dd dt02 = vs[2].texCoord - vs[0].texCoord;

                    double recDet = 1.0f / (dt01.x * dt02.y - dt01.y * dt02.x);
                    tc0Dir.x = recDet * (dt02.y * dp01.x - dt01.y * dp02.x);
                    tc0Dir.y = recDet * (dt02.y * dp01.y - dt01.y * dp02.y);
                    tc0Dir.z = recDet * (dt02.y * dp01.z - dt01.y * dp02.z);
                    tc1Dir.x = recDet * (-dt02.x * dp01.x + dt01.x * dp02.x);
                    tc1Dir.y = recDet * (-dt02.x * dp01.y + dt01.x * dp02.y);
                    tc1Dir.z = recDet * (-dt02.x * dp01.z + dt01.x * dp02.z);
                }

                Vector3Dd geomNormal = normalize(
                    cross(vs[1].position - vs[0].position, vs[2].position - vs[0].position));

                const Point3D tcs3D[] = {
                    Point3D(vs[0].texCoord, 1.0f),
                    Point3D(vs[1].texCoord, 1.0f),
                    Point3D(vs[2].texCoord, 1.0f),
                };

                Matrix4x4d matObjToTc(invert(Matrix3x3d(tc0Dir, tc1Dir, geomNormal)));
                matObjToTc = translate4x4(Point3Dd(vs[0].texCoord, 0.0f) - matObjToTc * vs[0].position) * matObjToTc;

                const Matrix3x3d matTcToBc = invert(Matrix3x3d(tcs3D[0], tcs3D[1], tcs3D[2]));
                const Matrix3x3d matBcToNInObj(vs[0].normal, vs[1].normal, vs[2].normal);
                const Matrix3x3d matTcToNInObj = matBcToNInObj * matTcToBc;

                dispTriAuxInfo.matObjToTc = static_cast<Matrix4x4>(matObjToTc);
                dispTriAuxInfo.matTcToBc = static_cast<Matrix3x3>(matTcToBc);
                dispTriAuxInfo.matTcToNInObj = static_cast<Matrix3x3>(matTcToNInObj);
                dispTriAuxInfo.matTcToObj = Matrix3x3(
                    static_cast<Vector3D>(tc0Dir),
                    static_cast<Vector3D>(tc1Dir),
                    static_cast<Vector3D>(geomNormal));
            }

            shared::GeometryInstanceData &geomInstData =
                scene.geomInstDataBuffer.getMappedPointer()[tfdmMeshGeomInst->geomInstSlot];
            tfdmMeshGeomInst->dispTriAuxInfoBuffer.initialize(
                gpuEnv.cuContext, cudau::BufferType::Device, dispTriAuxInfos);
            tfdmMeshGeomInst->aabbBuffer.initialize(
                gpuEnv.cuContext, cudau::BufferType::Device, triangles.size());

            shared::TFDMData &tfdmData = tfdmDataBuffer.getMappedPointer()[tfdmMeshGeomInst->geomInstSlot];
            tfdmData.dispTriAuxInfoBuffer =
                tfdmMeshGeomInst->dispTriAuxInfoBuffer.getROBuffer<shared::enableBufferOobCheck>();
            tfdmData.aabbBuffer =
                tfdmMeshGeomInst->aabbBuffer.getROBuffer<shared::enableBufferOobCheck>();
            tfdmData.hOffset = heightOffset;
            tfdmData.hScale = heightScale;
            tfdmData.hBias = heightBias;

#if !SHOW_BASE_MESH
            tfdmMeshGeomInst->optixGeomInst.setCustomPrimitiveAABBBuffer(tfdmMeshGeomInst->aabbBuffer);
#endif
        }

        std::set<const GeometryInstance*> srcGeomInsts = { tfdmMeshGeomInst };
        tfdmGeomGroup = createGeometryGroup(&scene, srcGeomInsts);
        scene.geomGroups.push_back(tfdmGeomGroup);

        auto mesh = new Mesh();
        {
            Mesh::GeometryGroupInstance groupInst = {};
            groupInst.geomGroup = tfdmGeomGroup;
            groupInst.transform = Matrix4x4();
            mesh->groupInsts.clear();
            mesh->groupInsts.push_back(groupInst);
            scene.meshes["obj"] = mesh;
        }

        Matrix4x4 instXfm = /*translate4x4(0, -0.5f, 0) * */rotateX4x4(0.25f * pi_v<float>);
        Instance* inst = createInstance(gpuEnv.cuContext, &scene, mesh->groupInsts[0], instXfm);
        scene.insts.push_back(inst);

        scene.initialSceneAabb.unify(
            instXfm * mesh->groupInsts[0].transform * mesh->groupInsts[0].geomGroup->aabb);
    }

    Vector3D sceneDim = scene.initialSceneAabb.maxP - scene.initialSceneAabb.minP;
    g_cameraPositionalMovingSpeed = 0.003f * std::max({ sceneDim.x, sceneDim.y, sceneDim.z });
    g_cameraDirectionalMovingSpeed = 0.0015f;
    g_cameraTiltSpeed = 0.025f;

    tfdmDataBuffer.unmap();
    scene.unmap();

    scene.setupASs(gpuEnv.cuContext);

    uint32_t totalNumEmitterPrimitives = 0;
    for (int i = 0; i < scene.insts.size(); ++i) {
        const Instance* inst = scene.insts[i];
        totalNumEmitterPrimitives += inst->geomGroupInst.geomGroup->numEmitterPrimitives;
    }
    hpprintf("%u emitter primitives\n", totalNumEmitterPrimitives);

    // JP: 環境光テクスチャーを読み込んで、サンプルするためのCDFを計算する。
    // EN: Read a environmental texture, then compute a CDF to sample it.
    cudau::Array envLightArray;
    CUtexObject envLightTexture = 0;
    RegularConstantContinuousDistribution2D envLightImportanceMap;
    if (!g_envLightTexturePath.empty())
        loadEnvironmentalTexture(g_envLightTexturePath, gpuEnv.cuContext,
                                 &envLightArray, &envLightTexture, &envLightImportanceMap);

    scene.setupLightGeomDistributions();

    // END: Setup a scene.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: スクリーン関連のバッファーを初期化。
    // EN: Initialize screen-related buffers.

    cudau::Array gBuffer0[2];
    cudau::Array gBuffer1[2];
    cudau::Array gBuffer2[2];
    
    cudau::Array beautyAccumBuffer;
    cudau::Array albedoAccumBuffer;
    cudau::Array normalAccumBuffer;

    cudau::TypedBuffer<float4> linearBeautyBuffer;
    cudau::TypedBuffer<float4> linearAlbedoBuffer;
    cudau::TypedBuffer<float4> linearNormalBuffer;
    cudau::TypedBuffer<float2> linearFlowBuffer;
    cudau::TypedBuffer<float4> linearDenoisedBeautyBuffer;

    cudau::Array rngBuffer;

    const auto initializeScreenRelatedBuffers = [&]() {
        for (int i = 0; i < 2; ++i) {
            gBuffer0[i].initialize2D(
                gpuEnv.cuContext, cudau::ArrayElementType::UInt32, (sizeof(shared::GBuffer0) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                renderTargetSizeX, renderTargetSizeY, 1);
            gBuffer1[i].initialize2D(
                gpuEnv.cuContext, cudau::ArrayElementType::UInt32, (sizeof(shared::GBuffer1) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                renderTargetSizeX, renderTargetSizeY, 1);
            gBuffer2[i].initialize2D(
                gpuEnv.cuContext, cudau::ArrayElementType::UInt32, (sizeof(shared::GBuffer2) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                renderTargetSizeX, renderTargetSizeY, 1);
        }

        beautyAccumBuffer.initialize2D(gpuEnv.cuContext, cudau::ArrayElementType::Float32, 4,
                                       cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                                       renderTargetSizeX, renderTargetSizeY, 1);
        albedoAccumBuffer.initialize2D(gpuEnv.cuContext, cudau::ArrayElementType::Float32, 4,
                                       cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                                       renderTargetSizeX, renderTargetSizeY, 1);
        normalAccumBuffer.initialize2D(gpuEnv.cuContext, cudau::ArrayElementType::Float32, 4,
                                       cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                                       renderTargetSizeX, renderTargetSizeY, 1);

        linearBeautyBuffer.initialize(gpuEnv.cuContext, Scene::bufferType,
                                      renderTargetSizeX * renderTargetSizeY);
        linearAlbedoBuffer.initialize(gpuEnv.cuContext, Scene::bufferType,
                                      renderTargetSizeX * renderTargetSizeY);
        linearNormalBuffer.initialize(gpuEnv.cuContext, Scene::bufferType,
                                      renderTargetSizeX * renderTargetSizeY);
        linearFlowBuffer.initialize(gpuEnv.cuContext, Scene::bufferType,
                                    renderTargetSizeX * renderTargetSizeY);
        linearDenoisedBeautyBuffer.initialize(gpuEnv.cuContext, Scene::bufferType,
                                              renderTargetSizeX * renderTargetSizeY);

        rngBuffer.initialize2D(
            gpuEnv.cuContext, cudau::ArrayElementType::UInt32, (sizeof(shared::PCG32RNG) + 3) / 4,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            renderTargetSizeX, renderTargetSizeY, 1);
        {
            auto rngs = rngBuffer.map<shared::PCG32RNG>();
            std::mt19937_64 rngSeed(591842031321323413);
            for (int y = 0; y < renderTargetSizeY; ++y) {
                for (int x = 0; x < renderTargetSizeX; ++x) {
                    shared::PCG32RNG &rng = rngs[y * renderTargetSizeX + x];
                    rng.setState(rngSeed());
                }
            }
            rngBuffer.unmap();
        }
    };

    const auto finalizeScreenRelatedBuffers = [&]() {
        rngBuffer.finalize();

        linearDenoisedBeautyBuffer.finalize();
        linearFlowBuffer.finalize();
        linearNormalBuffer.finalize();
        linearAlbedoBuffer.finalize();
        linearBeautyBuffer.finalize();

        normalAccumBuffer.finalize();
        albedoAccumBuffer.finalize();
        beautyAccumBuffer.finalize();

        for (int i = 1; i >= 0; --i) {
            gBuffer2[i].finalize();
            gBuffer1[i].finalize();
            gBuffer0[i].finalize();
        }
    };

    const auto resizeScreenRelatedBuffers = [&](uint32_t width, uint32_t height) {
        for (int i = 0; i < 2; ++i) {
            gBuffer0[i].resize(width, height);
            gBuffer1[i].resize(width, height);
            gBuffer2[i].resize(width, height);
        }

        beautyAccumBuffer.resize(width, height);
        albedoAccumBuffer.resize(width, height);
        normalAccumBuffer.resize(width, height);

        linearBeautyBuffer.resize(width * height);
        linearAlbedoBuffer.resize(width * height);
        linearNormalBuffer.resize(width * height);
        linearFlowBuffer.resize(width * height);
        linearDenoisedBeautyBuffer.resize(width * height);

        rngBuffer.resize(width, height);
        {
            auto rngs = rngBuffer.map<shared::PCG32RNG>();
            std::mt19937_64 rngSeed(591842031321323413);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    shared::PCG32RNG &rng = rngs[y * renderTargetSizeX + x];
                    rng.setState(rngSeed());
                }
            }
            rngBuffer.unmap();
        }
    };

    initializeScreenRelatedBuffers();

    // END: Initialize screen-related buffers.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: デノイザーのセットアップ。
    //     Temporalデノイザーを使用する。
    // EN: Setup a denoiser.
    //     Use the temporal denoiser.

    constexpr bool useTiledDenoising = false; // Change this to true to use tiled denoising.
    constexpr uint32_t tileWidth = useTiledDenoising ? 256 : 0;
    constexpr uint32_t tileHeight = useTiledDenoising ? 256 : 0;
    optixu::Denoiser denoiser = gpuEnv.optixContext.createDenoiser(
        OPTIX_DENOISER_MODEL_KIND_TEMPORAL,
        optixu::GuideAlbedo::Yes, optixu::GuideNormal::Yes, OPTIX_DENOISER_ALPHA_MODE_COPY);
    optixu::DenoiserSizes denoiserSizes;
    uint32_t numTasks;
    denoiser.prepare(
        renderTargetSizeX, renderTargetSizeY, tileWidth, tileHeight,
        &denoiserSizes, &numTasks);
    hpprintf("Denoiser State Buffer: %llu bytes\n", denoiserSizes.stateSize);
    hpprintf("Denoiser Scratch Buffer: %llu bytes\n", denoiserSizes.scratchSize);
    hpprintf("Compute Intensity Scratch Buffer: %llu bytes\n",
             denoiserSizes.scratchSizeForComputeNormalizer);
    cudau::Buffer denoiserStateBuffer;
    cudau::Buffer denoiserScratchBuffer;
    denoiserStateBuffer.initialize(
        gpuEnv.cuContext, Scene::bufferType, denoiserSizes.stateSize, 1);
    denoiserScratchBuffer.initialize(
        gpuEnv.cuContext, Scene::bufferType,
        std::max(denoiserSizes.scratchSize, denoiserSizes.scratchSizeForComputeNormalizer), 1);

    std::vector<optixu::DenoisingTask> denoisingTasks(numTasks);
    denoiser.getTasks(denoisingTasks.data());

    denoiser.setupState(stream, denoiserStateBuffer, denoiserScratchBuffer);

    // JP: デノイザーは入出力にリニアなバッファーを必要とするため結果をコピーする必要がある。
    // EN: Denoiser requires linear buffers as input/output, so we need to copy the results.
    CUmodule moduleCopyBuffers;
    CUDADRV_CHECK(cuModuleLoad(
        &moduleCopyBuffers,
        (getExecutableDirectory() / "tfdm/ptxes/copy_buffers.ptx").string().c_str()));
    cudau::Kernel kernelCopyToLinearBuffers(
        moduleCopyBuffers, "copyToLinearBuffers", cudau::dim3(8, 8), 0);
    cudau::Kernel kernelVisualizeToOutputBuffer(
        moduleCopyBuffers, "visualizeToOutputBuffer", cudau::dim3(8, 8), 0);

    CUdeviceptr plpPureCUDAOnDevice;
    size_t plpSize;
    CUDADRV_CHECK(cuModuleGetGlobal(&plpPureCUDAOnDevice, &plpSize, moduleCopyBuffers, "plp"));

    CUdeviceptr hdrNormalizer;
    CUDADRV_CHECK(cuMemAlloc(&hdrNormalizer, sizeof(float)));

    // END: Setup a denoiser.
    // ----------------------------------------------------------------



    // JP: OpenGL用バッファーオブジェクトからCUDAバッファーを生成する。
    // EN: Create a CUDA buffer from an OpenGL buffer instObject0.
    glu::Texture2D outputTexture;
    cudau::Array outputArray;
    cudau::InteropSurfaceObjectHolder<2> outputBufferSurfaceHolder;
    outputTexture.initialize(GL_RGBA32F, renderTargetSizeX, renderTargetSizeY, 1);
    outputArray.initializeFromGLTexture2D(gpuEnv.cuContext, outputTexture.getHandle(),
                                          cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
    outputBufferSurfaceHolder.initialize({ &outputArray });

    glu::Sampler outputSampler;
    outputSampler.initialize(glu::Sampler::MinFilter::Nearest, glu::Sampler::MagFilter::Nearest,
                             glu::Sampler::WrapMode::Repeat, glu::Sampler::WrapMode::Repeat);



    // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
    // EN: Empty VAO for full screen qud (or triangle).
    glu::VertexArray vertexArrayForFullScreen;
    vertexArrayForFullScreen.initialize();

    // JP: OptiXの結果をフレームバッファーにコピーするシェーダー。
    // EN: Shader to copy OptiX result to a frame buffer.
    glu::GraphicsProgram drawOptiXResultShader;
    drawOptiXResultShader.initializeVSPS(
        readTxtFile(exeDir / "tfdm/shaders/drawOptiXResult.vert"),
        readTxtFile(exeDir / "tfdm/shaders/drawOptiXResult.frag"));



    shared::StaticPipelineLaunchParameters staticPlp = {};
    {
        staticPlp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
        staticPlp.rngBuffer = rngBuffer.getSurfaceObject(0);

        staticPlp.GBuffer0[0] = gBuffer0[0].getSurfaceObject(0);
        staticPlp.GBuffer0[1] = gBuffer0[1].getSurfaceObject(0);
        staticPlp.GBuffer1[0] = gBuffer1[0].getSurfaceObject(0);
        staticPlp.GBuffer1[1] = gBuffer1[1].getSurfaceObject(0);
        staticPlp.GBuffer2[0] = gBuffer2[0].getSurfaceObject(0);
        staticPlp.GBuffer2[1] = gBuffer2[1].getSurfaceObject(0);

        staticPlp.materialDataBuffer =
            scene.materialDataBuffer.getROBuffer<shared::enableBufferOobCheck>();
        staticPlp.geometryInstanceDataBuffer =
            scene.geomInstDataBuffer.getROBuffer<shared::enableBufferOobCheck>();
        staticPlp.tfdmDataBuffer = tfdmDataBuffer.getROBuffer<shared::enableBufferOobCheck>();
        envLightImportanceMap.getDeviceType(&staticPlp.envLightImportanceMap);
        staticPlp.envLightTexture = envLightTexture;

        staticPlp.beautyAccumBuffer = beautyAccumBuffer.getSurfaceObject(0);
        staticPlp.albedoAccumBuffer = albedoAccumBuffer.getSurfaceObject(0);
        staticPlp.normalAccumBuffer = normalAccumBuffer.getSurfaceObject(0);
    }
    CUdeviceptr staticPlpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&staticPlpOnDevice, sizeof(staticPlp)));
    CUDADRV_CHECK(cuMemcpyHtoD(staticPlpOnDevice, &staticPlp, sizeof(staticPlp)));

    shared::PerFramePipelineLaunchParameters perFramePlp = {};
    perFramePlp.camera.fovY = 50 * pi_v<float> / 180;
    perFramePlp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    perFramePlp.camera.position = g_cameraPosition;
    perFramePlp.camera.orientation = g_cameraOrientation.toMatrix3x3();
    perFramePlp.prevCamera = perFramePlp.camera;
    perFramePlp.envLightPowerCoeff = 0;
    perFramePlp.envLightRotation = 0;

    CUdeviceptr perFramePlpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&perFramePlpOnDevice, sizeof(perFramePlp)));
    CUDADRV_CHECK(cuMemcpyHtoD(perFramePlpOnDevice, &perFramePlp, sizeof(perFramePlp)));
    
    shared::PipelineLaunchParameters plp;
    plp.s = reinterpret_cast<shared::StaticPipelineLaunchParameters*>(staticPlpOnDevice);
    plp.f = reinterpret_cast<shared::PerFramePipelineLaunchParameters*>(perFramePlpOnDevice);

    gpuEnv.gBuffer.hitGroupSbt.initialize(
        gpuEnv.cuContext, Scene::bufferType, scene.hitGroupSbtSize, 1);
    gpuEnv.gBuffer.hitGroupSbt.setMappedMemoryPersistent(true);
    gpuEnv.gBuffer.optixPipeline.setScene(scene.optixScene);
    gpuEnv.gBuffer.optixPipeline.setHitGroupShaderBindingTable(
        gpuEnv.gBuffer.hitGroupSbt, gpuEnv.gBuffer.hitGroupSbt.getMappedPointer());

    gpuEnv.pathTracing.hitGroupSbt.initialize(
        gpuEnv.cuContext, Scene::bufferType, scene.hitGroupSbtSize, 1);
    gpuEnv.pathTracing.hitGroupSbt.setMappedMemoryPersistent(true);
    gpuEnv.pathTracing.optixPipeline.setScene(scene.optixScene);
    gpuEnv.pathTracing.optixPipeline.setHitGroupShaderBindingTable(
        gpuEnv.pathTracing.hitGroupSbt, gpuEnv.pathTracing.hitGroupSbt.getMappedPointer());

    shared::PickInfo initPickInfo = {};
    initPickInfo.hit = false;
    initPickInfo.instSlot = 0xFFFFFFFF;
    initPickInfo.geomInstSlot = 0xFFFFFFFF;
    initPickInfo.matSlot = 0xFFFFFFFF;
    initPickInfo.primIndex = 0xFFFFFFFF;
    initPickInfo.positionInWorld = Point3D(0.0f);
    initPickInfo.albedo = RGB(0.0f);
    initPickInfo.emittance = RGB(0.0f);
    initPickInfo.normalInWorld = Normal3D(0.0f);
    cudau::TypedBuffer<shared::PickInfo> pickInfos[2];
    pickInfos[0].initialize(gpuEnv.cuContext, Scene::bufferType, 1, initPickInfo);
    pickInfos[1].initialize(gpuEnv.cuContext, Scene::bufferType, 1, initPickInfo);

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    struct GPUTimer {
        cudau::Timer frame;
        cudau::Timer update;
        cudau::Timer computePDFTexture;
        cudau::Timer setupGBuffers;
        cudau::Timer pathTrace;
        cudau::Timer denoise;

        void initialize(CUcontext context) {
            frame.initialize(context);
            update.initialize(context);
            computePDFTexture.initialize(context);
            setupGBuffers.initialize(context);
            pathTrace.initialize(context);
            denoise.initialize(context);
        }
        void finalize() {
            denoise.finalize();
            pathTrace.finalize();
            setupGBuffers.finalize();
            computePDFTexture.finalize();
            update.finalize();
            frame.finalize();
        }
    };

    GPUTimer gpuTimers[2];
    gpuTimers[0].initialize(gpuEnv.cuContext);
    gpuTimers[1].initialize(gpuEnv.cuContext);
    uint64_t frameIndex = 0;
    glfwSetWindowUserPointer(window, &frameIndex);
    int32_t requestedSize[2];
    uint32_t numAccumFrames = 0;
    while (true) {
        uint32_t bufferIndex = frameIndex % 2;

        GPUTimer &curGPUTimer = gpuTimers[bufferIndex];

        perFramePlp.prevCamera = perFramePlp.camera;

        if (glfwWindowShouldClose(window))
            break;
        glfwPollEvents();

        CUstream curCuStream = streamChain.waitAvailableAndGetCurrentStream();

        bool resized = false;
        int32_t newFBWidth;
        int32_t newFBHeight;
        glfwGetFramebufferSize(window, &newFBWidth, &newFBHeight);
        if (newFBWidth != curFBWidth || newFBHeight != curFBHeight) {
            curFBWidth = newFBWidth;
            curFBHeight = newFBHeight;

            renderTargetSizeX = curFBWidth / UIScaling;
            renderTargetSizeY = curFBHeight / UIScaling;
            requestedSize[0] = renderTargetSizeX;
            requestedSize[1] = renderTargetSizeY;

            glFinish();
            streamChain.waitAllWorkDone();

            resizeScreenRelatedBuffers(renderTargetSizeX, renderTargetSizeY);

            {
                optixu::DenoiserSizes denoiserSizes;
                uint32_t numTasks;
                denoiser.prepare(
                    renderTargetSizeX, renderTargetSizeY, tileWidth, tileHeight,
                    &denoiserSizes, &numTasks);
                hpprintf("Denoiser State Buffer: %llu bytes\n", denoiserSizes.stateSize);
                hpprintf("Denoiser Scratch Buffer: %llu bytes\n", denoiserSizes.scratchSize);
                hpprintf("Compute Intensity Scratch Buffer: %llu bytes\n",
                         denoiserSizes.scratchSizeForComputeNormalizer);
                denoiserStateBuffer.resize(denoiserSizes.stateSize, 1);
                denoiserScratchBuffer.resize(std::max(
                    denoiserSizes.scratchSize, denoiserSizes.scratchSizeForComputeNormalizer), 1);

                denoisingTasks.resize(numTasks);
                denoiser.getTasks(denoisingTasks.data());

                denoiser.setupState(curCuStream, denoiserStateBuffer, denoiserScratchBuffer);
            }

            outputTexture.finalize();
            outputArray.finalize();
            outputTexture.initialize(GL_RGBA32F, renderTargetSizeX, renderTargetSizeY, 1);
            outputArray.initializeFromGLTexture2D(
                gpuEnv.cuContext, outputTexture.getHandle(),
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);

            // EN: update the pipeline parameters.
            staticPlp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
            staticPlp.rngBuffer = rngBuffer.getSurfaceObject(0);
            staticPlp.GBuffer0[0] = gBuffer0[0].getSurfaceObject(0);
            staticPlp.GBuffer0[1] = gBuffer0[1].getSurfaceObject(0);
            staticPlp.GBuffer1[0] = gBuffer1[0].getSurfaceObject(0);
            staticPlp.GBuffer1[1] = gBuffer1[1].getSurfaceObject(0);
            staticPlp.GBuffer2[0] = gBuffer2[0].getSurfaceObject(0);
            staticPlp.GBuffer2[1] = gBuffer2[1].getSurfaceObject(0);
            staticPlp.beautyAccumBuffer = beautyAccumBuffer.getSurfaceObject(0);
            staticPlp.albedoAccumBuffer = albedoAccumBuffer.getSurfaceObject(0);
            staticPlp.normalAccumBuffer = normalAccumBuffer.getSurfaceObject(0);
            perFramePlp.camera.aspect = (float)renderTargetSizeX / renderTargetSizeY;

            CUDADRV_CHECK(cuMemcpyHtoD(staticPlpOnDevice, &staticPlp, sizeof(staticPlp)));

            resized = true;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();



        bool operatingCamera;
        bool cameraIsActuallyMoving;
        static bool operatedCameraOnPrevFrame = false;
        {
            const auto decideDirection = [](const KeyState &a, const KeyState &b) {
                int32_t dir = 0;
                if (a.getState() == true) {
                    if (b.getState() == true)
                        dir = 0;
                    else
                        dir = 1;
                }
                else {
                    if (b.getState() == true)
                        dir = -1;
                    else
                        dir = 0;
                }
                return dir;
            };

            int32_t trackZ = decideDirection(g_keyForward, g_keyBackward);
            int32_t trackX = decideDirection(g_keyLeftward, g_keyRightward);
            int32_t trackY = decideDirection(g_keyUpward, g_keyDownward);
            int32_t tiltZ = decideDirection(g_keyTiltRight, g_keyTiltLeft);
            int32_t adjustPosMoveSpeed = decideDirection(g_keyFasterPosMovSpeed, g_keySlowerPosMovSpeed);

            g_cameraPositionalMovingSpeed *= 1.0f + 0.02f * adjustPosMoveSpeed;
            g_cameraPositionalMovingSpeed = std::clamp(g_cameraPositionalMovingSpeed, 1e-6f, 1e+6f);

            static double deltaX = 0, deltaY = 0;
            static double lastX, lastY;
            static double g_prevMouseX = g_mouseX, g_prevMouseY = g_mouseY;
            if (g_buttonRotate.getState() == true) {
                if (g_buttonRotate.getTime() == frameIndex) {
                    lastX = g_mouseX;
                    lastY = g_mouseY;
                }
                else {
                    deltaX = g_mouseX - lastX;
                    deltaY = g_mouseY - lastY;
                }
            }

            float deltaAngle = std::sqrt(deltaX * deltaX + deltaY * deltaY);
            Vector3D axis(deltaY, -deltaX, 0);
            axis /= deltaAngle;
            if (deltaAngle == 0.0f)
                axis = Vector3D(1, 0, 0);

            g_cameraOrientation = g_cameraOrientation * qRotateZ(g_cameraTiltSpeed * tiltZ);
            g_tempCameraOrientation =
                g_cameraOrientation *
                qRotate(g_cameraDirectionalMovingSpeed * deltaAngle, axis);
            g_cameraPosition +=
                g_tempCameraOrientation.toMatrix3x3() *
                (g_cameraPositionalMovingSpeed * Vector3D(trackX, trackY, trackZ));
            if (g_buttonRotate.getState() == false && g_buttonRotate.getTime() == frameIndex) {
                g_cameraOrientation = g_tempCameraOrientation;
                deltaX = 0;
                deltaY = 0;
            }

            operatingCamera = (g_keyForward.getState() || g_keyBackward.getState() ||
                               g_keyLeftward.getState() || g_keyRightward.getState() ||
                               g_keyUpward.getState() || g_keyDownward.getState() ||
                               g_keyTiltLeft.getState() || g_keyTiltRight.getState() ||
                               g_buttonRotate.getState());
            cameraIsActuallyMoving = (trackZ != 0 || trackX != 0 || trackY != 0 ||
                                      tiltZ != 0 || (g_mouseX != g_prevMouseX) || (g_mouseY != g_prevMouseY))
                && operatingCamera;

            g_prevMouseX = g_mouseX;
            g_prevMouseY = g_mouseY;

            perFramePlp.camera.position = g_cameraPosition;
            perFramePlp.camera.orientation = g_tempCameraOrientation.toMatrix3x3();
        }



        bool resetAccumulation = false;
        
        // Camera Window
        static shared::BufferToDisplay bufferTypeToDisplay = shared::BufferToDisplay::NoisyBeauty;
        static bool applyToneMapAndGammaCorrection = true;
        static float brightness = g_initBrightness;
        static bool enableEnvLight = true;
        static float log10EnvLightPowerCoeff = 0.0f;
        static float envLightRotation = 0.0f;
        {
            ImGui::Begin("Camera / Env", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Text("W/A/S/D/R/F: Move, Q/E: Tilt");
            ImGui::Text("Mouse Middle Drag: Rotate");

            ImGui::InputFloat3("Position", reinterpret_cast<float*>(&g_cameraPosition));
            static float rollPitchYaw[3];
            g_tempCameraOrientation.toEulerAngles(&rollPitchYaw[0], &rollPitchYaw[1], &rollPitchYaw[2]);
            rollPitchYaw[0] *= 180 / pi_v<float>;
            rollPitchYaw[1] *= 180 / pi_v<float>;
            rollPitchYaw[2] *= 180 / pi_v<float>;
            if (ImGui::InputFloat3("Roll/Pitch/Yaw", rollPitchYaw))
                g_cameraOrientation = qFromEulerAngles(rollPitchYaw[0] * pi_v<float> / 180,
                                                       rollPitchYaw[1] * pi_v<float> / 180,
                                                       rollPitchYaw[2] * pi_v<float> / 180);
            ImGui::Text("Pos. Speed (T/G): %g", g_cameraPositionalMovingSpeed);
            ImGui::SliderFloat("Brightness", &brightness, -5.0f, 5.0f);

            ImGui::AlignTextToFramePadding();
            ImGui::Text("Screen Shot:");
            ImGui::SameLine();
            bool saveSS_LDR = ImGui::Button("SDR");
            ImGui::SameLine();
            bool saveSS_HDR = ImGui::Button("HDR");
            ImGui::SameLine();
            if (ImGui::Button("Both"))
                saveSS_LDR = saveSS_HDR = true;
            if (saveSS_LDR || saveSS_HDR) {
                streamChain.waitAllWorkDone();
                auto rawImage = new float4[renderTargetSizeX * renderTargetSizeY];
                glGetTextureSubImage(
                    outputTexture.getHandle(), 0,
                    0, 0, 0, renderTargetSizeX, renderTargetSizeY, 1,
                    GL_RGBA, GL_FLOAT, sizeof(float4) * renderTargetSizeX * renderTargetSizeY, rawImage);

                if (saveSS_LDR) {
                    SDRImageSaverConfig config;
                    config.brightnessScale = std::pow(10.0f, brightness);
                    config.applyToneMap = applyToneMapAndGammaCorrection;
                    config.apply_sRGB_gammaCorrection = applyToneMapAndGammaCorrection;
                    saveImage("output.png", renderTargetSizeX, renderTargetSizeY, rawImage,
                              config);
                }
                if (saveSS_HDR)
                    saveImageHDR("output.exr", renderTargetSizeX, renderTargetSizeY,
                                 std::pow(10.0f, brightness), rawImage);
                delete[] rawImage;
            }

            if (!g_envLightTexturePath.empty()) {
                ImGui::Separator();

                resetAccumulation |= ImGui::Checkbox("Enable Env Light", &enableEnvLight);
                resetAccumulation |= ImGui::SliderFloat("Env Power", &log10EnvLightPowerCoeff, -5.0f, 5.0f);
                resetAccumulation |= ImGui::SliderAngle("Env Rotation", &envLightRotation);
            }

            ImGui::End();
        }

        static bool useTemporalDenosier = true;
        static float motionVectorScale = -1.0f;
        static bool animate = /*true*/false;
        static bool enableAccumulation = /*true*/false;
        static int32_t log2MaxNumAccums = 16;
        static bool enableJittering = false;
        static bool enableBumpMapping = false;
        bool lastFrameWasAnimated = false;
        static int32_t maxPathLength = 5;
        static int32_t targetMipLevel = 0;
        bool heightParamChanged = false;
        static auto localIntersectionType = shared::LocalIntersectionType::TwoTriangle;
        bool localIntersectionTypeChanged = false;
        static bool debugSwitches[] = {
            false, false, false, false, false, false, false, false
        };
        {
            ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            if (ImGui::Button(animate ? "Stop" : "Play")) {
                if (animate)
                    lastFrameWasAnimated = true;
                animate = !animate;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset Accum"))
                resetAccumulation = true;
            ImGui::Checkbox("Enable Accumulation", &enableAccumulation);
            ImGui::InputLog2Int("#MaxNumAccum", &log2MaxNumAccums, 16, 5);
            resetAccumulation |= ImGui::Checkbox("Enable Jittering", &enableJittering);
            resetAccumulation |= ImGui::Checkbox("Enable Bump Mapping", &enableBumpMapping);

            ImGui::Separator();
            ImGui::Text("Cursor Info: %.1lf, %.1lf", g_mouseX, g_mouseY);
            shared::PickInfo pickInfoOnHost;
            pickInfos[bufferIndex].read(&pickInfoOnHost, 1, curCuStream);
            ImGui::Text("Hit: %s", pickInfoOnHost.hit ? "True" : "False");
            ImGui::Text("Instance: %u", pickInfoOnHost.instSlot);
            ImGui::Text("Geometry Instance: %u", pickInfoOnHost.geomInstSlot);
            ImGui::Text("Primitive Index: %u", pickInfoOnHost.primIndex);
            ImGui::Text("Material: %u", pickInfoOnHost.matSlot);
            ImGui::Text("Position: %.3f, %.3f, %.3f",
                        pickInfoOnHost.positionInWorld.x,
                        pickInfoOnHost.positionInWorld.y,
                        pickInfoOnHost.positionInWorld.z);
            ImGui::Text("Normal: %.3f, %.3f, %.3f",
                        pickInfoOnHost.normalInWorld.x,
                        pickInfoOnHost.normalInWorld.y,
                        pickInfoOnHost.normalInWorld.z);
            ImGui::Text("Albedo: %.3f, %.3f, %.3f",
                        pickInfoOnHost.albedo.r,
                        pickInfoOnHost.albedo.g,
                        pickInfoOnHost.albedo.b);
            ImGui::Text("Emittance: %.3f, %.3f, %.3f",
                        pickInfoOnHost.emittance.r,
                        pickInfoOnHost.emittance.g,
                        pickInfoOnHost.emittance.b);

            ImGui::Separator();

            if (ImGui::BeginTabBar("MyTabBar")) {
                if (ImGui::BeginTabItem("Renderer")) {
                    resetAccumulation |= ImGui::SliderInt("Max Path Length", &maxPathLength, 2, 15);

                    ImGui::Separator();

                    resetAccumulation |= ImGui::SliderInt("Target Mip Level", &targetMipLevel, 0, 15);

                    ImGui::Separator();
                    ImGui::Text("Displacement Parameters");

                    heightParamChanged |= ImGui::SliderFloat("Offset", &heightOffset, -1.0f, 1.0f);
                    heightParamChanged |= ImGui::SliderFloat("Scale", &heightScale, -1.0f, 1.0f);
                    heightParamChanged |= ImGui::SliderFloat("Bias", &heightBias, 0.0f, 1.0f);
                    resetAccumulation |= heightParamChanged;

                    ImGui::Separator();
                    ImGui::Text("Local Intersection Type");

                    shared::LocalIntersectionType oldIsectType = localIntersectionType;
                    ImGui::RadioButtonE(
                        "Box", &localIntersectionType, shared::LocalIntersectionType::Box);
                    ImGui::RadioButtonE(
                        "Two-Triangle", &localIntersectionType, shared::LocalIntersectionType::TwoTriangle);
                    ImGui::BeginDisabled();
                    ImGui::RadioButtonE(
                        "Bilinear", &localIntersectionType, shared::LocalIntersectionType::Bilinear);
                    ImGui::RadioButtonE(
                        "B-Spline", &localIntersectionType, shared::LocalIntersectionType::BSpline);
                    ImGui::EndDisabled();
                    localIntersectionTypeChanged = localIntersectionType != oldIsectType || frameIndex == 0;
                    resetAccumulation |= localIntersectionTypeChanged;

                    ImGui::Separator();

                    ImGui::PushID("Debug Switches");
                    for (int i = lengthof(debugSwitches) - 1; i >= 0; --i) {
                        ImGui::PushID(i);
                        resetAccumulation |= ImGui::Checkbox("", &debugSwitches[i]);
                        ImGui::PopID();
                        if (i > 0)
                            ImGui::SameLine();
                    }
                    ImGui::PopID();

                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Visualize")) {
                    ImGui::Text("Buffer to Display");
                    ImGui::RadioButtonE(
                        "Noisy Beauty", &bufferTypeToDisplay, shared::BufferToDisplay::NoisyBeauty);
                    ImGui::RadioButtonE("Albedo", &bufferTypeToDisplay, shared::BufferToDisplay::Albedo);
                    ImGui::RadioButtonE("Normal", &bufferTypeToDisplay, shared::BufferToDisplay::Normal);
                    ImGui::RadioButtonE("TexCoord", &bufferTypeToDisplay, shared::BufferToDisplay::TexCoord);
                    ImGui::RadioButtonE("Motion Vector", &bufferTypeToDisplay, shared::BufferToDisplay::Flow);
                    ImGui::RadioButtonE(
                        "Denoised Beauty", &bufferTypeToDisplay, shared::BufferToDisplay::DenoisedBeauty);

                    if (ImGui::Checkbox("Temporal Denoiser", &useTemporalDenosier)) {
                        streamChain.waitAllWorkDone();
                        denoiser.destroy();

                        OptixDenoiserModelKind modelKind = useTemporalDenosier ?
                            OPTIX_DENOISER_MODEL_KIND_TEMPORAL :
                            OPTIX_DENOISER_MODEL_KIND_HDR;
                        denoiser = gpuEnv.optixContext.createDenoiser(
                            modelKind, optixu::GuideAlbedo::Yes, optixu::GuideNormal::Yes,
                            OPTIX_DENOISER_ALPHA_MODE_COPY);

                        optixu::DenoiserSizes denoiserSizes;
                        uint32_t numTasks;
                        denoiser.prepare(
                            renderTargetSizeX, renderTargetSizeY, tileWidth, tileHeight,
                            &denoiserSizes, &numTasks);
                        hpprintf("Denoiser State Buffer: %llu bytes\n", denoiserSizes.stateSize);
                        hpprintf("Denoiser Scratch Buffer: %llu bytes\n", denoiserSizes.scratchSize);
                        hpprintf("Compute Intensity Scratch Buffer: %llu bytes\n",
                                 denoiserSizes.scratchSizeForComputeNormalizer);
                        denoiserStateBuffer.resize(denoiserSizes.stateSize, 1);
                        denoiserScratchBuffer.resize(std::max(
                            denoiserSizes.scratchSize, denoiserSizes.scratchSizeForComputeNormalizer), 1);

                        denoisingTasks.resize(numTasks);
                        denoiser.getTasks(denoisingTasks.data());

                        denoiser.setupState(curCuStream, denoiserStateBuffer, denoiserScratchBuffer);
                    }

                    ImGui::SliderFloat("Motion Vector Scale", &motionVectorScale, -2.0f, 2.0f);

                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }

            ImGui::Separator();

            ImGui::End();
        }

        // Stats Window
        {
            ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

#if !defined(USE_HARD_CODED_BSDF_FUNCTIONS)
            ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + 300);
            ImGui::TextColored(
                ImVec4(1.0f, 0.0f, 0.0f, 1.0f),
                "BSDF callables are enabled.\n"
                "USE_HARD_CODED_BSDF_FUNCTIONS is recommended for better performance.");
            ImGui::PopTextWrapPos();
#endif

            static MovingAverageTime cudaFrameTime;
            static MovingAverageTime updateTime;
            static MovingAverageTime computePDFTextureTime;
            static MovingAverageTime setupGBuffersTime;
            static MovingAverageTime pathTraceTime;
            static MovingAverageTime denoiseTime;

            cudaFrameTime.append(curGPUTimer.frame.report());
            updateTime.append(curGPUTimer.update.report());
            computePDFTextureTime.append(curGPUTimer.computePDFTexture.report());
            setupGBuffersTime.append(curGPUTimer.setupGBuffers.report());
            pathTraceTime.append(curGPUTimer.pathTrace.report());
            denoiseTime.append(curGPUTimer.denoise.report());

            //ImGui::SetNextItemWidth(100.0f);
            ImGui::Text("CUDA/OptiX GPU %.3f [ms]:", cudaFrameTime.getAverage());
            ImGui::Text("  Update: %.3f [ms]", updateTime.getAverage());
            ImGui::Text("  Compute PDF Texture: %.3f [ms]", computePDFTextureTime.getAverage());
            ImGui::Text("  Setup G-Buffers: %.3f [ms]", setupGBuffersTime.getAverage());
            ImGui::Text("  Path Trace: %.3f [ms]", pathTraceTime.getAverage());
            if (bufferTypeToDisplay == shared::BufferToDisplay::DenoisedBeauty)
                ImGui::Text("  Denoise: %.3f [ms]", denoiseTime.getAverage());

            ImGui::Text("%u [spp]", std::min(numAccumFrames + 1, (1u << log2MaxNumAccums)));

            ImGui::End();
        }

        applyToneMapAndGammaCorrection =
            bufferTypeToDisplay == shared::BufferToDisplay::NoisyBeauty ||
            bufferTypeToDisplay == shared::BufferToDisplay::DenoisedBeauty;



        curGPUTimer.frame.start(curCuStream);

        // JP: 各インスタンスのトランスフォームを更新する。
        // EN: Update the transform of each instance.
        if (animate || lastFrameWasAnimated) {
            cudau::TypedBuffer<shared::InstanceData> &curInstDataBuffer = scene.instDataBuffer[bufferIndex];
            shared::InstanceData* instDataBufferOnHost = curInstDataBuffer.map();
            for (int i = 0; i < scene.instControllers.size(); ++i) {
                InstanceController* controller = scene.instControllers[i];
                Instance* inst = controller->inst;
                shared::InstanceData &instData = instDataBufferOnHost[inst->instSlot];
                controller->update(instDataBufferOnHost, animate ? 1.0f / 60.0f : 0.0f);
                // TODO: まとめて送る。
                CUDADRV_CHECK(cuMemcpyHtoDAsync(curInstDataBuffer.getCUdeviceptrAt(inst->instSlot),
                                                &instData, sizeof(instData), curCuStream));
            }
            curInstDataBuffer.unmap();
        }

        // JP: Minmax mipmapを計算する。
        // EN: Compute the minmax mipmap.
        if (localIntersectionTypeChanged) {
            const Material* mat = tfdmMeshMaterial;

            const shared::MaterialData* const matData =
                scene.materialDataBuffer.getDevicePointerAt(mat->materialSlot);

            cudau::Kernel generateFirstMinMaxMipMap;
            cudau::Kernel generateMinMaxMipMap;
            if (localIntersectionType == shared::LocalIntersectionType::Box) {
                generateFirstMinMaxMipMap = gpuEnv.kernelGenerateFirstMinMaxMipMap_Box;
                generateMinMaxMipMap = gpuEnv.kernelGenerateMinMaxMipMap_Box;
            }
            else if (localIntersectionType == shared::LocalIntersectionType::TwoTriangle) {
                generateFirstMinMaxMipMap = gpuEnv.kernelGenerateFirstMinMaxMipMap_TwoTriangle;
                generateMinMaxMipMap = gpuEnv.kernelGenerateMinMaxMipMap_TwoTriangle;
            }
            else if (localIntersectionType == shared::LocalIntersectionType::Bilinear) {
                generateFirstMinMaxMipMap = gpuEnv.kernelGenerateFirstMinMaxMipMap_Bilinear;
                generateMinMaxMipMap = gpuEnv.kernelGenerateMinMaxMipMap_Bilinear;
            }
            else if (localIntersectionType == shared::LocalIntersectionType::BSpline) {
                generateFirstMinMaxMipMap = gpuEnv.kernelGenerateFirstMinMaxMipMap_BSpline;
                generateMinMaxMipMap = gpuEnv.kernelGenerateMinMaxMipMap_BSpline;
            }
            else {
                Assert_ShouldNotBeCalled();
            }

            int2 dstImageSize(mat->heightMap->getWidth(), mat->heightMap->getHeight());
            generateFirstMinMaxMipMap.launchWithThreadDim(
                curCuStream, cudau::dim3(dstImageSize.x, dstImageSize.y),
                matData, localIntersectionType);
            //{
            //    CUDADRV_CHECK(cuStreamSynchronize(curCuStream));
            //    std::vector<float2> minMaxValues(dstImageSize.x * dstImageSize.y);
            //    mat->minMaxMipMap.read(minMaxValues, 0);
            //    printf("");
            //}
            dstImageSize /= 2;
            const uint32_t numMinMaxMipMapLevels = nextPowOf2Exponent(mat->heightMap->getWidth()) + 1;
            for (int srcLevel = 0; srcLevel < numMinMaxMipMapLevels - 1; ++srcLevel) {
                generateMinMaxMipMap.launchWithThreadDim(
                    curCuStream, cudau::dim3(dstImageSize.x, dstImageSize.y),
                    matData, srcLevel);

                /*CUDADRV_CHECK(cuStreamSynchronize(curCuStream));
                std::vector<float2> minMaxValues(dstImageSize.x * dstImageSize.y);
                mat->minMaxMipMap.read(minMaxValues, srcLevel + 1);

                bool saveImg = false;
                if (saveImg) {
                    std::vector<float4> minImg(dstImageSize.x * dstImageSize.y);
                    std::vector<float4> maxImg(dstImageSize.x * dstImageSize.y);
                    for (int y = 0; y < dstImageSize.y; ++y) {
                        for (int x = 0; x < dstImageSize.x; ++x) {
                            float2 value = minMaxValues[y * dstImageSize.x + x];
                            minImg[y * dstImageSize.x + x].x = value.x;
                            minImg[y * dstImageSize.x + x].y = value.x;
                            minImg[y * dstImageSize.x + x].z = value.x;
                            minImg[y * dstImageSize.x + x].w = 1.0f;
                            maxImg[y * dstImageSize.x + x].x = value.y;
                            maxImg[y * dstImageSize.x + x].y = value.y;
                            maxImg[y * dstImageSize.x + x].z = value.y;
                            maxImg[y * dstImageSize.x + x].w = 1.0f;
                        }
                    }
                    SDRImageSaverConfig imgConfig = {};
                    saveImage("min_img.png", dstImageSize.x, dstImageSize.y, minImg.data(), imgConfig);
                    saveImage("max_img.png", dstImageSize.x, dstImageSize.y, maxImg.data(), imgConfig);
                }*/

                dstImageSize /= 2;
            }
        }

        // JP: ディスプレイスメントを適用した各プリミティブのAABBを計算する。
        // EN: Compute the AABB of each displacement-enabled primitive.
        if (localIntersectionTypeChanged || heightParamChanged/* || geomChanged*/) {
            const GeometryInstance* geomInst = tfdmMeshGeomInst;
            const Material* mat = tfdmMeshMaterial;

            const shared::GeometryInstanceData* const geomInstData =
                scene.geomInstDataBuffer.getDevicePointerAt(geomInst->geomInstSlot);
            const shared::TFDMData* const tfdmData =
                tfdmDataBuffer.getDevicePointerAt(geomInst->geomInstSlot);
            const shared::MaterialData* const matData =
                scene.materialDataBuffer.getDevicePointerAt(mat->materialSlot);

            CUDADRV_CHECK(cuMemcpyHtoDAsync(
                reinterpret_cast<CUdeviceptr>(&tfdmData->hOffset), &heightOffset, sizeof(heightOffset),
                curCuStream));
            CUDADRV_CHECK(cuMemcpyHtoDAsync(
                reinterpret_cast<CUdeviceptr>(&tfdmData->hScale), &heightScale, sizeof(heightScale),
                curCuStream));
            CUDADRV_CHECK(cuMemcpyHtoDAsync(
                reinterpret_cast<CUdeviceptr>(&tfdmData->hBias), &heightBias, sizeof(heightBias),
                curCuStream));

            gpuEnv.kernelComputeAABBs.launchWithThreadDim(
                curCuStream, cudau::dim3(geomInst->aabbBuffer.numElements()),
                geomInstData, tfdmData, matData);

            //CUDADRV_CHECK(cuStreamSynchronize(curCuStream));
            //std::vector<AABB> aabbs = geomInst->aabbBuffer;

            tfdmGeomGroup->needsRebuild = true;
        }

        // JP: ASesのリビルドを行う。
        // EN: Rebuild the ASes.
        curGPUTimer.update.start(curCuStream);
        if (animate || frameIndex == 0 ||
            localIntersectionTypeChanged || heightParamChanged)
            perFramePlp.travHandle = scene.updateASs(curCuStream);
        curGPUTimer.update.stop(curCuStream);

        // JP: 光源となるインスタンスのProbability Textureを計算する。
        // EN: Compute the probability texture for light instances.
        curGPUTimer.computePDFTexture.start(curCuStream);
        {
            CUdeviceptr probTexAddr =
                staticPlpOnDevice + offsetof(shared::StaticPipelineLaunchParameters, lightInstDist);
            scene.setupLightInstDistribution(curCuStream, probTexAddr, bufferIndex);
        }
        curGPUTimer.computePDFTexture.stop(curCuStream);

        bool newSequence = resized || frameIndex == 0 || resetAccumulation;
        bool firstAccumFrame =
            animate || !enableAccumulation || cameraIsActuallyMoving || newSequence;
        if (firstAccumFrame)
            numAccumFrames = 0;
        else
            numAccumFrames = std::min(numAccumFrames + 1, (1u << log2MaxNumAccums));
        if (newSequence)
            hpprintf("New sequence started.\n");

        perFramePlp.numAccumFrames = numAccumFrames;
        perFramePlp.frameIndex = frameIndex;
        perFramePlp.instanceDataBuffer =
            scene.instDataBuffer[bufferIndex].getROBuffer<shared::enableBufferOobCheck>();
        perFramePlp.envLightPowerCoeff = std::pow(10.0f, log10EnvLightPowerCoeff);
        perFramePlp.envLightRotation = envLightRotation;
        perFramePlp.mousePosition = int2(static_cast<int32_t>(g_mouseX),
                                         static_cast<int32_t>(g_mouseY));
        perFramePlp.pickInfo = pickInfos[bufferIndex].getDevicePointer();

        perFramePlp.maxPathLength = maxPathLength;
        perFramePlp.bufferIndex = bufferIndex;
        perFramePlp.resetFlowBuffer = newSequence;
        perFramePlp.enableJittering = enableJittering;
        perFramePlp.enableEnvLight = enableEnvLight;
        perFramePlp.enableBumpMapping = enableBumpMapping;
        perFramePlp.localIntersectionType = static_cast<uint32_t>(localIntersectionType);
        perFramePlp.targetMipLevel = targetMipLevel;
        perFramePlp.enableDebugPrint = g_keyDebugPrint.getState();
        for (int i = 0; i < lengthof(debugSwitches); ++i)
            perFramePlp.setDebugSwitch(i, debugSwitches[i]);

        CUDADRV_CHECK(cuMemcpyHtoDAsync(perFramePlpOnDevice, &perFramePlp, sizeof(perFramePlp), curCuStream));

        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curCuStream));
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpPureCUDAOnDevice, &plp, sizeof(plp), curCuStream));

        // JP: Gバッファーのセットアップ。
        //     ここではレイトレースを使ってGバッファーを生成しているがもちろんラスタライザーで生成可能。
        // EN: Setup the G-buffers.
        //     Generate the G-buffers using ray trace here, but of course this can be done using rasterizer.
        curGPUTimer.setupGBuffers.start(curCuStream);
        gpuEnv.gBuffer.optixPipeline.launch(
            curCuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
        curGPUTimer.setupGBuffers.stop(curCuStream);

        // JP: パストレーシングによるシェーディングを実行。
        // EN: Perform shading by path tracing.
        curGPUTimer.pathTrace.start(curCuStream);
        gpuEnv.pathTracing.setEntryPoint(PathTracingEntryPoint::pathTrace);
        gpuEnv.pathTracing.optixPipeline.launch(
            curCuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
        curGPUTimer.pathTrace.stop(curCuStream);

        // JP: 結果をリニアバッファーにコピーする。(法線の正規化も行う。)
        // EN: Copy the results to the linear buffers (and normalize normals).
        kernelCopyToLinearBuffers.launchWithThreadDim(
            curCuStream, cudau::dim3(renderTargetSizeX, renderTargetSizeY),
            beautyAccumBuffer.getSurfaceObject(0),
            albedoAccumBuffer.getSurfaceObject(0),
            normalAccumBuffer.getSurfaceObject(0),
            gBuffer2[bufferIndex].getSurfaceObject(0),
            linearBeautyBuffer,
            linearAlbedoBuffer,
            linearNormalBuffer,
            linearFlowBuffer,
            uint2(renderTargetSizeX, renderTargetSizeY));

        curGPUTimer.denoise.start(curCuStream);
        if (bufferTypeToDisplay == shared::BufferToDisplay::DenoisedBeauty) {
            denoiser.computeNormalizer(
                curCuStream,
                linearBeautyBuffer, OPTIX_PIXEL_FORMAT_FLOAT4,
                denoiserScratchBuffer, hdrNormalizer);
            //float hdrNormalizerOnHost;
            //CUDADRV_CHECK(cuMemcpyDtoH(&hdrNormalizerOnHost, hdrNormalizer, sizeof(hdrNormalizerOnHost)));
            //printf("%g\n", hdrNormalizerOnHost);

            optixu::DenoiserInputBuffers inputBuffers = {};
            inputBuffers.noisyBeauty = linearBeautyBuffer;
            inputBuffers.albedo = linearAlbedoBuffer;
            inputBuffers.normal = linearNormalBuffer;
            inputBuffers.flow = linearFlowBuffer;
            inputBuffers.previousDenoisedBeauty = newSequence ?
                linearBeautyBuffer : linearDenoisedBeautyBuffer;
            inputBuffers.beautyFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
            inputBuffers.albedoFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
            inputBuffers.normalFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
            inputBuffers.flowFormat = OPTIX_PIXEL_FORMAT_FLOAT2;

            for (int i = 0; i < denoisingTasks.size(); ++i)
                denoiser.invoke(
                    curCuStream, denoisingTasks[i], inputBuffers,
                    optixu::IsFirstFrame(newSequence), hdrNormalizer, 0.0f,
                    linearDenoisedBeautyBuffer, nullptr,
                    optixu::BufferView());
        }
        curGPUTimer.denoise.stop(curCuStream);

        outputBufferSurfaceHolder.beginCUDAAccess(curCuStream);

        // JP: デノイズ結果や中間バッファーの可視化。
        // EN: Visualize the denoised result or intermediate buffers.
        void* bufferToDisplay = nullptr;
        switch (bufferTypeToDisplay) {
        case shared::BufferToDisplay::NoisyBeauty:
            bufferToDisplay = linearBeautyBuffer.getDevicePointer();
            break;
        case shared::BufferToDisplay::Albedo:
            bufferToDisplay = linearAlbedoBuffer.getDevicePointer();
            break;
        case shared::BufferToDisplay::Normal:
            bufferToDisplay = linearNormalBuffer.getDevicePointer();
            break;
        case shared::BufferToDisplay::TexCoord:
            break;
        case shared::BufferToDisplay::Flow:
            bufferToDisplay = linearFlowBuffer.getDevicePointer();
            break;
        case shared::BufferToDisplay::DenoisedBeauty:
            bufferToDisplay = linearDenoisedBeautyBuffer.getDevicePointer();
            break;
        default:
            Assert_ShouldNotBeCalled();
            break;
        }
        kernelVisualizeToOutputBuffer(
            curCuStream, kernelVisualizeToOutputBuffer.calcGridDim(renderTargetSizeX, renderTargetSizeY),
            staticPlp.GBuffer0[bufferIndex],
            staticPlp.GBuffer1[bufferIndex],
            bufferToDisplay,
            bufferTypeToDisplay,
            0.5f, std::pow(10.0f, motionVectorScale),
            outputBufferSurfaceHolder.getNext(),
            uint2(renderTargetSizeX, renderTargetSizeY));

        outputBufferSurfaceHolder.endCUDAAccess(curCuStream, true);

        curGPUTimer.frame.stop(curCuStream);

        streamChain.swap();



        // ----------------------------------------------------------------
        // JP: OptiXによる描画結果を表示用レンダーターゲットにコピーする。
        // EN: Copy the OptiX rendering results to the display render target.

        if (applyToneMapAndGammaCorrection) {
            glEnable(GL_FRAMEBUFFER_SRGB);
            ImGui::GetStyle() = guiStyleWithGamma;
        }
        else {
            glDisable(GL_FRAMEBUFFER_SRGB);
            ImGui::GetStyle() = guiStyle;
        }

        glViewport(0, 0, curFBWidth, curFBHeight);

        glUseProgram(drawOptiXResultShader.getHandle());

        glUniform2ui(0, curFBWidth, curFBHeight);
        int32_t flags =
            (applyToneMapAndGammaCorrection ? 1 : 0);
        glUniform1i(2, flags);
        glUniform1f(3, std::pow(10.0f, brightness));

        glBindTextureUnit(0, outputTexture.getHandle());
        glBindSampler(0, outputSampler.getHandle());

        glBindVertexArray(vertexArrayForFullScreen.getHandle());
        glDrawArrays(GL_TRIANGLES, 0, 3);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glDisable(GL_FRAMEBUFFER_SRGB);

        // END: Copy the OptiX rendering results to the display render target.
        // ----------------------------------------------------------------

        glfwSwapBuffers(window);

        ++frameIndex;
    }

    streamChain.waitAllWorkDone();
    gpuTimers[1].finalize();
    gpuTimers[0].finalize();



    CUDADRV_CHECK(cuMemFree(plpOnDevice));

    pickInfos[1].finalize();
    pickInfos[0].finalize();

    CUDADRV_CHECK(cuMemFree(perFramePlpOnDevice));
    CUDADRV_CHECK(cuMemFree(staticPlpOnDevice));

    drawOptiXResultShader.finalize();
    vertexArrayForFullScreen.finalize();

    outputSampler.finalize();
    outputBufferSurfaceHolder.finalize();
    outputArray.finalize();
    outputTexture.finalize();


    
    CUDADRV_CHECK(cuMemFree(hdrNormalizer));
    CUDADRV_CHECK(cuModuleUnload(moduleCopyBuffers));    
    denoiserScratchBuffer.finalize();
    denoiserStateBuffer.finalize();
    denoiser.destroy();
    
    finalizeScreenRelatedBuffers();



    envLightImportanceMap.finalize(gpuEnv.cuContext);
    if (envLightTexture)
        cuTexObjectDestroy(envLightTexture);
    envLightArray.finalize();

    finalizeTextureCaches();

    streamChain.finalize();

    tfdmDataBuffer.finalize();
    scene.finalize();
    
    gpuEnv.finalize();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);

    glfwTerminate();

    return 0;
}
catch (const std::exception &ex) {
    hpprintf("Error: %s\n", ex.what());
    return -1;
}
