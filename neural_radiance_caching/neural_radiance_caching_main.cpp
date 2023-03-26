/*

コマンドラインオプション例 / Command line option example:
You can load a 3D model for example by downloading from the internet.
(e.g. https://casual-effects.com/data/)

(1) -cam-pos 7.534, -0.067, -1.612 -cam-yaw -61
    -name scene -obj ZeroDay_v1/MEASURE_SEVEN/MEASURE_SEVEN_COLORED_LIGHTS.fbx 1.0 simple_pbr -brightness 2.0
    -inst scene

    * Zero-Day from Open Research Content Archive (ORCA)
      https://developer.nvidia.com/orca/beeple-zero-day

(2) -cam-pos -9.5 5 0 -cam-yaw 90
    -name sponza -obj crytek_sponza/sponza.obj 0.01 trad
    -name rectlight -emittance 600 600 600 -rectangle 1 1 -begin-pos 0 15 0 -inst rectlight
    -name rectlight0 -emittance 600 0 0 -rectangle 1 1
    -name rectlight1 -emittance 0 600 0 -rectangle 1 1
    -name rectlight2 -emittance 0 0 600 -rectangle 1 1
    -name rectlight3 -emittance 100 100 100 -rectangle 1 1
    -begin-pos 0 0 0.36125 -inst sponza
    -begin-pos -5 13.1 0 -end-pos 5 13.1 0 -freq 5 -time 0.0 -inst rectlight0
    -begin-pos -5 13 0 -end-pos 5 13 0 -freq 10 -time 2.5 -inst rectlight1
    -begin-pos -5 12.9 0 -end-pos 5 12.9 0 -freq 15 -time 7.5 -inst rectlight2
    -begin-pos -5 7 -4.8 -begin-pitch -30 -end-pos 5 7 -4.8 -end-pitch -30 -freq 5 -inst rectlight3
    -begin-pos 5 7 4.8 -begin-pitch 30 -end-pos -5 7 4.8 -end-pitch 30 -freq 5 -inst rectlight3

JP: このプログラムはNeural Radiance Caching (NRC) [1]の実装例です。
    NRCは位置や出射方向、物体表面のパラメターを入力、輝度を出力とするニューラルネットワークです。
    レンダリング時にはパストレーシングによって経路を構築しますが、
    ある経路長より先から得られる寄与をキャッシュからのクエリーによって置き換えることで、
    少しのバイアスと引き換えに低い分散の推定値を得ることができます。
    さらに経路長が短くなることでシーンによっては1フレームの時間も短くなり得ます。
    NRCは比較的小さなネットワークであり、トレーニングはレンダリングの最中に行うオンラインラーニングとすることで、
    「適応による汎化」を実現、推論の実行時間もリアルタイムレンダリングに適した短いものとなります。
    ニューラルネットワーク部分にはtiny-cuda-nn [2]というライブラリーを使用しています。
    ※TuringアーキテクチャーのGPUでも動くと思いますが、現状RTX 3080 (Ampere)でしか動作確認していません。
      CMakeを使わずにこのサンプルをビルドするには外部ライブラリを先に手動でビルドしておく必要があります。
      tiny-cuda-nnのビルドの調整やnetwork_interface.cuのTCNN_MIN_GPU_ARCHの変更などが必要だと思います。
    ※デフォルトではBRDFにOptiXのCallable ProgramやCUDAの関数ポインターを使用した汎用的な実装になっており、
      性能上のオーバーヘッドが著しいため、純粋な性能を見る上では common_shared.h の USE_HARD_CODED_BSDF_FUNCTIONS
      を有効化したほうがよいかもしれません。

EN: This program is an example implementation of Neural Radiance Caching (NRC) [1].
    NRC is a neural network where the inputs are a position and an outgoing direction, surface parameters,
    and the output is radiance. It constructs paths based on path tracing when rendering,
    but replaces contributions given from beyond a certain path length by a query to the cache.
    This achieves low variance estimates at the cost of a little bias.
    Additionally, one frame time can even be reduced depending on the scene thanks to path shortening.
    NRC is a relatively small network, and training is online learning during rendering.
    This achieve "generalization via adaptation", and short inference time appropriate to real-time rendering.
    This program uses tiny-cuda-nn [2] for the neural network part.
    * I have tested only with RTX 3080 (Ampere) while I think the program would work with
      Turing architecture GPU as well.
      It requires to manually build external libaries before building the program when not using CMake.
      Some adjustments for building tiny-cuda-nn and changing TCNN_MIN_GPU_ARCH in network_interface.cu
      are required.
    * The program is generic implementation with OptiX's callable program and CUDA's function pointer,
      and has significant performance overhead, therefore it may be recommended to enable USE_HARD_CODED_BSDF_FUNCTIONS
      in common_shared.h to see pure performance.

[1] Real-time Neural Radiance Caching for Path Tracing
    https://research.nvidia.com/publication/2021-06_Real-time-Neural-Radiance
[2] Tiny CUDA Neural Networks
    https://github.com/NVlabs/tiny-cuda-nn

*/

#include "neural_radiance_caching_shared.h"
#include "../common/common_host.h"
#include "network_interface.h"

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
    Baseline,
    NRC,
    visualizePrediction,
};

struct GPUEnvironment {
    CUcontext cuContext;
    optixu::Context optixContext;

    CUmodule cudaModule;
    cudau::Kernel kernelPreprocessNRC;
    cudau::Kernel kernelAccumulateInferredRadianceValues;
    cudau::Kernel kernelPropagateRadianceValues;
    cudau::Kernel kernelShuffleTrainingData;
    CUdeviceptr plpPtr;

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

    void initialize() {
        int32_t cuDeviceCount;
        CUDADRV_CHECK(cuInit(0));
        CUDADRV_CHECK(cuDeviceGetCount(&cuDeviceCount));
        CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
        CUDADRV_CHECK(cuCtxSetCurrent(cuContext));

        optixContext = optixu::Context::create(
            cuContext/*, 4, DEBUG_SELECT(optixu::EnableValidation::Yes, optixu::EnableValidation::No)*/);

        CUDADRV_CHECK(cuModuleLoad(
            &cudaModule,
            (getExecutableDirectory() / "neural_radiance_caching/ptxes/nrc_setup_kernels.ptx").string().c_str()));
        kernelPreprocessNRC =
            cudau::Kernel(cudaModule, "preprocessNRC", cudau::dim3(32), 0);
        kernelAccumulateInferredRadianceValues =
            cudau::Kernel(cudaModule, "accumulateInferredRadianceValues", cudau::dim3(32), 0);
        kernelPropagateRadianceValues =
            cudau::Kernel(cudaModule, "propagateRadianceValues", cudau::dim3(32), 0);
        kernelShuffleTrainingData =
            cudau::Kernel(cudaModule, "shuffleTrainingData", cudau::dim3(32), 0);

        size_t plpSize;
        CUDADRV_CHECK(cuModuleGetGlobal(&plpPtr, &plpSize, cudaModule, "plp"));
        Assert(sizeof(shared::PipelineLaunchParameters) == plpSize, "Unexpected plp size.");

        optixDefaultMaterial = optixContext.createMaterial();
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
                optixu::calcSumDwords<float2>(),
                "plp", sizeof(shared::PipelineLaunchParameters),
                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE),
                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

            m = p.createModuleFromPTXString(
                readTxtFile(getExecutableDirectory() / "neural_radiance_caching/ptxes/optix_gbuffer_kernels.ptx"),
                OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
                DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            pipeline.entryPoints[GBufferEntryPoint::setupGBuffers] = p.createRayGenProgram(
                m, RT_RG_NAME_STR("setupGBuffers"));

            pipeline.hitPrograms["hitgroup"] = p.createHitProgramGroupForTriangleIS(
                m, RT_CH_NAME_STR("setupGBuffers"),
                emptyModule, nullptr);
            pipeline.programs["miss"] = p.createMissProgram(
                m, RT_MS_NAME_STR("setupGBuffers"));

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

            p.link(1, DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            uint32_t maxDcStackSize = 0;
            for (int i = 0; i < NumCallablePrograms; ++i) {
                optixu::CallableProgramGroup program = pipeline.callablePrograms[i];
                maxDcStackSize = std::max(maxDcStackSize, program.getDCStackSize());
            }
            uint32_t maxCcStackSize =
                pipeline.entryPoints.at(GBufferEntryPoint::setupGBuffers).getStackSize() +
                std::max(
                    {
                        pipeline.hitPrograms.at("hitgroup").getCHStackSize(),
                        pipeline.programs.at("miss").getStackSize()
                    });

            p.setStackSize(0, maxDcStackSize, maxCcStackSize, 2);

            optixDefaultMaterial.setHitGroup(shared::GBufferRayType::Primary, pipeline.hitPrograms.at("hitgroup"));
            for (uint32_t rayType = shared::GBufferRayType::NumTypes; rayType < shared::maxNumRayTypes; ++rayType)
                optixDefaultMaterial.setHitGroup(rayType, pipeline.hitPrograms.at("emptyHitGroup"));

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
                    shared::PathTraceRayPayloadSignature<false>::numDwords,
                    shared::PathTraceRayPayloadSignature<true>::numDwords,
                    shared::VisibilityRayPayloadSignature::numDwords
                         }),
                optixu::calcSumDwords<float2>(),
                "plp", sizeof(shared::PipelineLaunchParameters),
                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE),
                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

            m = p.createModuleFromPTXString(
                readTxtFile(getExecutableDirectory() / "neural_radiance_caching/ptxes/optix_pathtracing_kernels.ptx"),
                OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
                DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            pipeline.entryPoints[PathTracingEntryPoint::Baseline] =
                p.createRayGenProgram(m, RT_RG_NAME_STR("pathTraceBaseline"));
            pipeline.entryPoints[PathTracingEntryPoint::NRC] =
                p.createRayGenProgram(m, RT_RG_NAME_STR("pathTraceNRC"));
            pipeline.entryPoints[PathTracingEntryPoint::visualizePrediction] =
                p.createRayGenProgram(m, RT_RG_NAME_STR("visualizePrediction"));

            pipeline.programs[RT_MS_NAME_STR("pathTraceBaseline")] = p.createMissProgram(
                m, RT_MS_NAME_STR("pathTraceBaseline"));
            pipeline.hitPrograms[RT_CH_NAME_STR("pathTraceBaseline")] = p.createHitProgramGroupForTriangleIS(
                m, RT_CH_NAME_STR("pathTraceBaseline"),
                emptyModule, nullptr);

            pipeline.programs[RT_MS_NAME_STR("pathTraceNRC")] = p.createMissProgram(
                m, RT_MS_NAME_STR("pathTraceNRC"));
            pipeline.hitPrograms[RT_CH_NAME_STR("pathTraceNRC")] = p.createHitProgramGroupForTriangleIS(
                m, RT_CH_NAME_STR("pathTraceNRC"),
                emptyModule, nullptr);

            pipeline.hitPrograms[RT_AH_NAME_STR("visibility")] = p.createHitProgramGroupForTriangleIS(
                emptyModule, nullptr,
                m, RT_AH_NAME_STR("visibility"));

            pipeline.programs["emptyMiss"] = p.createMissProgram(emptyModule, nullptr);

            p.setNumMissRayTypes(shared::PathTracingRayType::NumTypes);
            p.setMissProgram(
                shared::PathTracingRayType::Baseline, pipeline.programs.at(RT_MS_NAME_STR("pathTraceBaseline")));
            p.setMissProgram(
                shared::PathTracingRayType::NRC, pipeline.programs.at(RT_MS_NAME_STR("pathTraceNRC")));
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

            p.link(2, DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            uint32_t maxDcStackSize = 0;
            for (int i = 0; i < NumCallablePrograms; ++i) {
                optixu::CallableProgramGroup program = pipeline.callablePrograms[i];
                maxDcStackSize = std::max(maxDcStackSize, program.getDCStackSize());
            }
            uint32_t maxCcStackSize = std::max({
                pipeline.entryPoints.at(PathTracingEntryPoint::Baseline).getStackSize() +
                std::max(
                    {
                        pipeline.hitPrograms.at(RT_CH_NAME_STR("pathTraceBaseline")).getCHStackSize() +
                        pipeline.hitPrograms.at(RT_AH_NAME_STR("visibility")).getAHStackSize(),
                        pipeline.programs.at(RT_MS_NAME_STR("pathTraceBaseline")).getStackSize()
                    }),
                pipeline.entryPoints.at(PathTracingEntryPoint::NRC).getStackSize() +
                std::max(
                    {
                        pipeline.hitPrograms.at(RT_CH_NAME_STR("pathTraceNRC")).getCHStackSize() +
                        pipeline.hitPrograms.at(RT_AH_NAME_STR("visibility")).getAHStackSize(),
                        pipeline.programs.at(RT_MS_NAME_STR("pathTraceNRC")).getStackSize()
                    }),
                pipeline.entryPoints.at(PathTracingEntryPoint::visualizePrediction).getStackSize()
            });

            p.setStackSize(0, maxDcStackSize, maxCcStackSize, 2);

            optixDefaultMaterial.setHitGroup(
                shared::PathTracingRayType::Baseline, pipeline.hitPrograms.at(RT_CH_NAME_STR("pathTraceBaseline")));
            optixDefaultMaterial.setHitGroup(
                shared::PathTracingRayType::NRC, pipeline.hitPrograms.at(RT_CH_NAME_STR("pathTraceNRC")));
            optixDefaultMaterial.setHitGroup(
                shared::PathTracingRayType::Visibility, pipeline.hitPrograms.at(RT_AH_NAME_STR("visibility")));

            size_t sbtSize;
            p.generateShaderBindingTableLayout(&sbtSize);
            pipeline.sbt.initialize(cuContext, Scene::bufferType, sbtSize, 1);
            pipeline.sbt.setMappedMemoryPersistent(true);
            p.setShaderBindingTable(pipeline.sbt, pipeline.sbt.getMappedPointer());
        }

        std::vector<void*> callablePointers(NumCallablePrograms);
        for (int i = 0; i < NumCallablePrograms; ++i) {
            CUdeviceptr symbolPtr;
            size_t symbolSize;
            CUDADRV_CHECK(cuModuleGetGlobal(&symbolPtr, &symbolSize, cudaModule,
                                            callableProgramPointerNames[i]));
            void* funcPtrOnDevice;
            Assert(symbolSize == sizeof(funcPtrOnDevice), "Unexpected symbol size");
            CUDADRV_CHECK(cuMemcpyDtoH(&funcPtrOnDevice, symbolPtr, sizeof(funcPtrOnDevice)));
            callablePointers[i] = funcPtrOnDevice;
        }

        CUdeviceptr callableToPointerMapPtr;
        size_t callableToPointerMapSize;
        CUDADRV_CHECK(cuModuleGetGlobal(
            &callableToPointerMapPtr, &callableToPointerMapSize, cudaModule,
            "c_callableToPointerMap"));
        CUDADRV_CHECK(cuMemcpyHtoD(callableToPointerMapPtr, callablePointers.data(),
                                   callableToPointerMapSize));
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

        optixDefaultMaterial.destroy();

        CUDADRV_CHECK(cuModuleUnload(cudaModule));

        optixContext.destroy();

        //CUDADRV_CHECK(cuCtxDestroy(cuContext));
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

static PositionEncoding g_positionEncoding = PositionEncoding::HashGrid;
static uint32_t g_numHiddenLayers = 2;
static float g_learningRate = 1e-2f;

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
                *ori = qRotateZ(atof(argv[i + 1]) * pi_v<float> / 180) * *ori;
                i += 1;
            }
            else if (strncmp(arg, "-pitch", 7) == 0) {
                if (i + 1 >= argc) {
                    hpprintf("Invalid option.\n");
                    exit(EXIT_FAILURE);
                }
                *ori = qRotateX(atof(argv[i + 1]) * pi_v<float> / 180) * *ori;
                i += 1;
            }
            else if (strncmp(arg, "-yaw", 5) == 0) {
                if (i + 1 >= argc) {
                    hpprintf("Invalid option.\n");
                    exit(EXIT_FAILURE);
                }
                *ori = qRotateY(atof(argv[i + 1]) * pi_v<float> / 180) * *ori;
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
        else if (strncmp(arg, "-position-encoding", 19) == 0) {
            if (i + 1 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            const char* enc = argv[i + 1];
            if (strncmp(enc, "tri-wave", 8) == 0) {
                g_positionEncoding = PositionEncoding::TriangleWave;
            }
            else if (strncmp(enc, "hash-grid", 10) == 0) {
                g_positionEncoding = PositionEncoding::HashGrid;
            }
            else {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            i += 1;
        }
        else if (0 == strncmp(arg, "-num-hidden-layers", 19)) {
            if (i + 1 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            g_numHiddenLayers = atoi(argv[i + 1]);
            i += 1;
        }
        else if (0 == strncmp(arg, "-learning-rate", 15)) {
            if (i + 1 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            g_learningRate = atof(argv[i + 1]);
            if (!isfinite(g_learningRate)) {
                printf("Invalid value.\n");
                exit(EXIT_FAILURE);
            }
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
    GLFWwindow* window = glfwCreateWindow(static_cast<int32_t>(renderTargetSizeX * UIScaling),
                                          static_cast<int32_t>(renderTargetSizeY * UIScaling),
                                          "Neural Radiance Caching", NULL, NULL);
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
    ImGuiIO& io = ImGui::GetIO(); (void)io;
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
        getExecutableDirectory() / "neural_radiance_caching/ptxes",
        gpuEnv.cuContext, gpuEnv.optixContext, shared::maxNumRayTypes, gpuEnv.optixDefaultMaterial);

    CUstream cuStream;
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    scene.map();

    for (auto it = g_meshInfos.cbegin(); it != g_meshInfos.cend(); ++it) {
        const MeshInfo &info = it->second;

        if (std::holds_alternative<MeshGeometryInfo>(info)) {
            const auto &meshInfo = std::get<MeshGeometryInfo>(info);

            createTriangleMeshes(
                it->first,
                meshInfo.path, meshInfo.matConv,
                scale4x4(meshInfo.preScale),
                gpuEnv.cuContext, &scene);
        }
        else if (std::holds_alternative<RectangleGeometryInfo>(info)) {
            const auto &rectInfo = std::get<RectangleGeometryInfo>(info);

            createRectangleLight(
                it->first,
                rectInfo.dimX, rectInfo.dimZ,
                RGB(0.01f),
                rectInfo.emitterTexPath, rectInfo.emittance, Matrix4x4(),
                gpuEnv.cuContext, &scene);
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

    Vector3D sceneDim = scene.initialSceneAabb.maxP - scene.initialSceneAabb.minP;
    g_cameraPositionalMovingSpeed = 0.003f * std::max({ sceneDim.x, sceneDim.y, sceneDim.z });
    g_cameraDirectionalMovingSpeed = 0.0015f;
    g_cameraTiltSpeed = 0.025f;

    scene.unmap();

    scene.setupASes(gpuEnv.cuContext);
    CUDADRV_CHECK(cuStreamSynchronize(0));

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

    CUdeviceptr sceneAABBOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&sceneAABBOnDevice, sizeof(AABB)));
    CUDADRV_CHECK(cuMemcpyHtoD(sceneAABBOnDevice, &scene.initialSceneAabb, sizeof(scene.initialSceneAabb)));

    // END: Setup a scene.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: NRCの訓練データなどに関わるバッファーを初期化する。
    // EN: Initialize NRC training-related buffers.

    const uint32_t maxNumTrainingSuffixes = renderTargetSizeX * renderTargetSizeY / 16;
    
    cudau::TypedBuffer<uint32_t> numTrainingData[2];
    cudau::TypedBuffer<uint2> tileSize[2];
    cudau::TypedBuffer<RGBAsOrderedInt> targetMinMax[2];
    cudau::TypedBuffer<RGB> targetAvg[2];
    for (int i = 0; i < 2; ++i) {
        numTrainingData[i].initialize(gpuEnv.cuContext, Scene::bufferType, 1, 0);
        tileSize[i].initialize(gpuEnv.cuContext, Scene::bufferType, 1, uint2(8, 8));
        targetMinMax[i].initialize(gpuEnv.cuContext, Scene::bufferType, 2);
        targetAvg[i].initialize(gpuEnv.cuContext, Scene::bufferType, 1);
    }

    uintptr_t offsetToSelectUnbiasedTileOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&offsetToSelectUnbiasedTileOnDevice, sizeof(uint32_t)));

    uintptr_t offsetToSelectTrainingPathOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&offsetToSelectTrainingPathOnDevice, sizeof(uint32_t)));

    cudau::TypedBuffer<shared::RadianceQuery> trainRadianceQueryBuffer[2];
    cudau::TypedBuffer<RGB> trainTargetBuffer[2];
    cudau::TypedBuffer<shared::TrainingVertexInfo> trainVertexInfoBuffer;
    cudau::TypedBuffer<shared::TrainingSuffixTerminalInfo> trainSuffixTerminalInfoBuffer;
    cudau::TypedBuffer<shared::LinearCongruentialGenerator> dataShufflerBuffer;
    for (int i = 0; i < 2; ++i) {
        trainRadianceQueryBuffer[i].initialize(
            gpuEnv.cuContext, Scene::bufferType, shared::trainBufferSize);
        trainTargetBuffer[i].initialize(
            gpuEnv.cuContext, Scene::bufferType, shared::trainBufferSize);
    }
    trainVertexInfoBuffer.initialize(
        gpuEnv.cuContext, Scene::bufferType, shared::trainBufferSize);
    trainSuffixTerminalInfoBuffer.initialize(
        gpuEnv.cuContext, Scene::bufferType, maxNumTrainingSuffixes);
    dataShufflerBuffer.initialize(
        gpuEnv.cuContext, Scene::bufferType, shared::trainBufferSize);
    {
        shared::LinearCongruentialGenerator lcg;
        lcg.setState(471313181);
        shared::LinearCongruentialGenerator* dataShufflers = dataShufflerBuffer.map();
        for (int i = 0; i < shared::numTrainingDataPerFrame; ++i) {
            lcg.next();
            dataShufflers[i] = lcg;
        }
        dataShufflerBuffer.unmap();
    }

    NeuralRadianceCache neuralRadianceCache;
    neuralRadianceCache.initialize(g_positionEncoding, g_numHiddenLayers, g_learningRate);

    // END: Initialize NRC training-related buffers.
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

    cudau::TypedBuffer<shared::RadianceQuery> inferenceRadianceQueryBuffer;
    cudau::TypedBuffer<shared::TerminalInfo> inferenceTerminalInfoBuffer;
    cudau::TypedBuffer<RGB> inferredRadianceBuffer;
    cudau::TypedBuffer<RGB> perFrameContributionBuffer;

    const auto initializeScreenRelatedBuffers = [&]() {
        uint32_t numPixels = renderTargetSizeX * renderTargetSizeY;

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

        linearBeautyBuffer.initialize(gpuEnv.cuContext, Scene::bufferType, numPixels);
        linearAlbedoBuffer.initialize(gpuEnv.cuContext, Scene::bufferType, numPixels);
        linearNormalBuffer.initialize(gpuEnv.cuContext, Scene::bufferType, numPixels);
        linearFlowBuffer.initialize(gpuEnv.cuContext, Scene::bufferType, numPixels);
        linearDenoisedBeautyBuffer.initialize(gpuEnv.cuContext, Scene::bufferType, numPixels);

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



        uint32_t inferenceBatchSize =
            ((numPixels + maxNumTrainingSuffixes) + 255) / 256 * 256;
        inferenceRadianceQueryBuffer.initialize(
            gpuEnv.cuContext, Scene::bufferType, inferenceBatchSize);
        inferenceTerminalInfoBuffer.initialize(gpuEnv.cuContext, Scene::bufferType, numPixels);
        inferredRadianceBuffer.initialize(
            gpuEnv.cuContext, Scene::bufferType, inferenceBatchSize);
        perFrameContributionBuffer.initialize(gpuEnv.cuContext, Scene::bufferType, numPixels);
    };

    const auto finalizeScreenRelatedBuffers = [&]() {
        perFrameContributionBuffer.finalize();
        inferredRadianceBuffer.finalize();
        inferenceTerminalInfoBuffer.finalize();
        inferenceRadianceQueryBuffer.finalize();

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
        uint32_t numPixels = width * height;

        for (int i = 0; i < 2; ++i) {
            gBuffer0[i].resize(width, height);
            gBuffer1[i].resize(width, height);
            gBuffer2[i].resize(width, height);
        }

        beautyAccumBuffer.resize(width, height);
        albedoAccumBuffer.resize(width, height);
        normalAccumBuffer.resize(width, height);

        linearBeautyBuffer.resize(numPixels);
        linearAlbedoBuffer.resize(numPixels);
        linearNormalBuffer.resize(numPixels);
        linearFlowBuffer.resize(numPixels);
        linearDenoisedBeautyBuffer.resize(numPixels);

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



        uint32_t inferenceBatchSize =
            ((numPixels + maxNumTrainingSuffixes) + 255) / 256 * 256;
        inferenceRadianceQueryBuffer.resize(inferenceBatchSize);
        inferenceTerminalInfoBuffer.resize(numPixels);
        inferredRadianceBuffer.resize(inferenceBatchSize);
        perFrameContributionBuffer.resize(numPixels);
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
        optixu::GuideAlbedo::Yes, optixu::GuideNormal::Yes);
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

    denoiser.setupState(cuStream, denoiserStateBuffer, denoiserScratchBuffer);

    // JP: デノイザーは入出力にリニアなバッファーを必要とするため結果をコピーする必要がある。
    // EN: Denoiser requires linear buffers as input/output, so we need to copy the results.
    CUmodule moduleCopyBuffers;
    CUDADRV_CHECK(cuModuleLoad(
        &moduleCopyBuffers,
        (getExecutableDirectory() / "neural_radiance_caching/ptxes/copy_buffers.ptx").string().c_str()));
    cudau::Kernel kernelCopyToLinearBuffers(
        moduleCopyBuffers, "copyToLinearBuffers", cudau::dim3(8, 8), 0);
    cudau::Kernel kernelVisualizeToOutputBuffer(
        moduleCopyBuffers, "visualizeToOutputBuffer", cudau::dim3(8, 8), 0);

    uintptr_t plpPtrOnDeviceForCopyBuffers;
    {
        size_t plpSizeForCopyBuffers;
        CUDADRV_CHECK(cuModuleGetGlobal(
            &plpPtrOnDeviceForCopyBuffers, &plpSizeForCopyBuffers, moduleCopyBuffers, "plp"));
        Assert(sizeof(shared::PipelineLaunchParameters) == plpSizeForCopyBuffers, "Unexpected plp size.");
    }

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
        readTxtFile(exeDir / "neural_radiance_caching/shaders/drawOptiXResult.vert"),
        readTxtFile(exeDir / "neural_radiance_caching/shaders/drawOptiXResult.frag"));



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

        staticPlp.materialDataBuffer = shared::ROBuffer(
            scene.materialDataBuffer.getDevicePointer(), scene.materialDataBuffer.numElements());
        staticPlp.geometryInstanceDataBuffer = shared::ROBuffer(
            scene.geomInstDataBuffer.getDevicePointer(), scene.geomInstDataBuffer.numElements());
        envLightImportanceMap.getDeviceType(&staticPlp.envLightImportanceMap);
        staticPlp.envLightTexture = envLightTexture;

        staticPlp.sceneAABB = reinterpret_cast<AABB*>(sceneAABBOnDevice);

        staticPlp.maxNumTrainingSuffixes = maxNumTrainingSuffixes;
        for (int i = 0; i < 2; ++i) {
            staticPlp.numTrainingData[i] = numTrainingData[i].getDevicePointer();
            staticPlp.tileSize[i] = tileSize[i].getDevicePointer();
            staticPlp.targetMinMax[i] = targetMinMax[i].getDevicePointer();
            staticPlp.targetAvg[i] = targetAvg[i].getDevicePointer();
        }
        staticPlp.offsetToSelectUnbiasedTile = reinterpret_cast<uint32_t*>(offsetToSelectUnbiasedTileOnDevice);
        staticPlp.offsetToSelectTrainingPath = reinterpret_cast<uint32_t*>(offsetToSelectTrainingPathOnDevice);
        staticPlp.inferenceRadianceQueryBuffer = shared::RWBuffer(
            inferenceRadianceQueryBuffer.getDevicePointer(), inferenceRadianceQueryBuffer.numElements());
        staticPlp.inferenceTerminalInfoBuffer = shared::RWBuffer(
            inferenceTerminalInfoBuffer.getDevicePointer(), inferenceTerminalInfoBuffer.numElements());
        staticPlp.inferredRadianceBuffer = shared::RWBuffer(
            inferredRadianceBuffer.getDevicePointer(), inferredRadianceBuffer.numElements());
        staticPlp.perFrameContributionBuffer = shared::RWBuffer(
            perFrameContributionBuffer.getDevicePointer(), perFrameContributionBuffer.numElements());
        for (int i = 0; i < 2; ++i) {
            staticPlp.trainRadianceQueryBuffer[i] = shared::RWBuffer(
                trainRadianceQueryBuffer[i].getDevicePointer(), trainRadianceQueryBuffer[i].numElements());
            staticPlp.trainTargetBuffer[i] = shared::RWBuffer(
                trainTargetBuffer[i].getDevicePointer(), trainTargetBuffer[i].numElements());
        }
        staticPlp.trainVertexInfoBuffer = shared::RWBuffer(
            trainVertexInfoBuffer.getDevicePointer(), trainVertexInfoBuffer.numElements());
        staticPlp.trainSuffixTerminalInfoBuffer = shared::RWBuffer(
            trainSuffixTerminalInfoBuffer.getDevicePointer(), trainSuffixTerminalInfoBuffer.numElements());
        staticPlp.dataShufflerBuffer = shared::RWBuffer(
            dataShufflerBuffer.getDevicePointer(), dataShufflerBuffer.numElements());

        staticPlp.beautyAccumBuffer = beautyAccumBuffer.getSurfaceObject(0);
        staticPlp.albedoAccumBuffer = albedoAccumBuffer.getSurfaceObject(0);
        staticPlp.normalAccumBuffer = normalAccumBuffer.getSurfaceObject(0);
    }
    CUdeviceptr staticPlpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&staticPlpOnDevice, sizeof(staticPlp)));
    CUDADRV_CHECK(cuMemcpyHtoD(staticPlpOnDevice, &staticPlp, sizeof(staticPlp)));

    shared::PerFramePipelineLaunchParameters perFramePlp = {};
    perFramePlp.travHandle = scene.ias.getHandle();
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
        cudau::Timer preprocessNRC;
        cudau::Timer pathTrace;
        cudau::Timer infer;
        cudau::Timer accumulateInferredRadiances;
        cudau::Timer propagateRadiances;
        cudau::Timer shuffleTrainingData;
        cudau::Timer train;
        cudau::Timer visualizeCache;
        cudau::Timer denoise;

        void initialize(CUcontext context) {
            frame.initialize(context);
            update.initialize(context);
            computePDFTexture.initialize(context);
            setupGBuffers.initialize(context);
            preprocessNRC.initialize(context);
            pathTrace.initialize(context);
            infer.initialize(context);
            accumulateInferredRadiances.initialize(context);
            propagateRadiances.initialize(context);
            shuffleTrainingData.initialize(context);
            train.initialize(context);
            visualizeCache.initialize(context);
            denoise.initialize(context);
        }
        void finalize() {
            denoise.finalize();
            visualizeCache.finalize();
            train.finalize();
            shuffleTrainingData.finalize();
            propagateRadiances.finalize();
            accumulateInferredRadiances.finalize();
            infer.finalize();
            pathTrace.finalize();
            preprocessNRC.finalize();
            setupGBuffers.finalize();
            computePDFTexture.finalize();
            update.finalize();
            frame.finalize();
        }
    };

    std::mt19937 perFrameRng(72139121);

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
            CUDADRV_CHECK(cuStreamSynchronize(cuStream));

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

                denoiser.setupState(cuStream, denoiserStateBuffer, denoiserScratchBuffer);
            }

            outputTexture.finalize();
            outputArray.finalize();
            outputTexture.initialize(GL_RGBA32F, renderTargetSizeX, renderTargetSizeY, 1);
            outputArray.initializeFromGLTexture2D(gpuEnv.cuContext, outputTexture.getHandle(),
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
            staticPlp.inferenceRadianceQueryBuffer = shared::RWBuffer(
                inferenceRadianceQueryBuffer.getDevicePointer(), inferenceRadianceQueryBuffer.numElements());
            staticPlp.inferenceTerminalInfoBuffer = shared::RWBuffer(
                inferenceTerminalInfoBuffer.getDevicePointer(), inferenceTerminalInfoBuffer.numElements());
            staticPlp.inferredRadianceBuffer = shared::RWBuffer(
                inferredRadianceBuffer.getDevicePointer(), inferredRadianceBuffer.numElements());
            staticPlp.perFrameContributionBuffer = shared::RWBuffer(
                perFrameContributionBuffer.getDevicePointer(), perFrameContributionBuffer.numElements());
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
            const auto decideDirection = [](const KeyState& a, const KeyState& b) {
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
                CUDADRV_CHECK(cuStreamSynchronize(cuStream));
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
        static bool infBounces = false;
        static int32_t maxPathLength = 5;
        static bool useNRC = true;
        static float log10RadianceScale = brightness;
        static bool visualizeTrainingPath = false;
        static bool train = true;
        bool stepTrain = false;
        static bool showLossValue = false;
        static float lossValue = 0.0f;
        static bool prevTrainDone = false;
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
            pickInfos[bufferIndex].read(&pickInfoOnHost, 1, cuStream);
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

            uint32_t numTrainingDataOnHost = 0;
            uint2 tileSizeOnHost = uint2(0, 0);
            RGBAsOrderedInt targetMinMaxAsOrderedIntOnHost[2];
            RGB targetAvgOnHost;
            if (useNRC) {
                numTrainingData[bufferIndex].read(&numTrainingDataOnHost, 1, cuStream);
                tileSize[bufferIndex].read(&tileSizeOnHost, 1, cuStream);
                targetMinMax[bufferIndex].read(targetMinMaxAsOrderedIntOnHost, 2, cuStream);
                targetAvg[bufferIndex].read(&targetAvgOnHost, 1, cuStream);
            }
            RGB targetMinMaxOnHost[2] = {
                static_cast<RGB>(targetMinMaxAsOrderedIntOnHost[0]),
                static_cast<RGB>(targetMinMaxAsOrderedIntOnHost[1])
            };
            ImGui::Text("#Training Data: %6u", numTrainingDataOnHost);
            ImGui::Text("Tile Size: %2u x %2u", tileSizeOnHost.x, tileSizeOnHost.y);
            ImGui::Text("Target Range:");
            ImGui::Text("  Min: %10.4f, %10.4f, %10.4f",
                        targetMinMaxOnHost[0].r, targetMinMaxOnHost[0].g, targetMinMaxOnHost[0].b);
            ImGui::Text("  Max: %10.4f, %10.4f, %10.4f",
                        targetMinMaxOnHost[1].r, targetMinMaxOnHost[1].g, targetMinMaxOnHost[1].b);
            ImGui::Text("  Avg: %10.4f, %10.4f, %10.4f",
                        targetAvgOnHost.r, targetAvgOnHost.g, targetAvgOnHost.b);
            bool prevShowLossValue = showLossValue;
            ImGui::Checkbox("Show Loss", &showLossValue);
            if (showLossValue && prevShowLossValue) {
                static float log10Losses[100] = {};
                static uint32_t startOffset = 0;
                if (prevTrainDone) {
                    log10Losses[startOffset] = std::log10(lossValue);
                    startOffset = (startOffset + 1) % IM_ARRAYSIZE(log10Losses);
                }

                char overlayText[100];
                sprintf_s(overlayText, "%.6f", lossValue);

                ImGui::PlotLines(
                    "Loss", log10Losses, IM_ARRAYSIZE(log10Losses), startOffset,
                    overlayText, -3.0f, 3.0f, ImVec2(0, 80.0f));
            }

            ImGui::Separator();

            if (ImGui::BeginTabBar("MyTabBar")) {
                if (ImGui::BeginTabItem("Renderer")) {
                    resetAccumulation |= ImGui::Checkbox("Infinite-Bounce", &infBounces);
                    if (!infBounces)
                        resetAccumulation |= ImGui::SliderInt("Max Path Length", &maxPathLength, 2, 15);

                    bool tempUseNRC = useNRC;
                    if (ImGui::RadioButton("Baseline Path Tracing", !useNRC)) {
                        useNRC = false;
                        if (bufferTypeToDisplay == shared::BufferToDisplay::RenderingPathLength ||
                            bufferTypeToDisplay == shared::BufferToDisplay::DirectlyVisualizedPrediction)
                            bufferTypeToDisplay = shared::BufferToDisplay::NoisyBeauty;
                    }
                    if (ImGui::RadioButton("Path Tracing + NRC", useNRC))
                        useNRC = true;
                    resetAccumulation |= useNRC != tempUseNRC;

                    if (useNRC) {
                        if (ImGui::Button(train ? "Stop Training" : "Start Training"))
                            train = !train;
                        ImGui::SameLine();
                        if (ImGui::Button("Step")) {
                            train = false;
                            stepTrain = true;
                        }
                        ImGui::SameLine();
                        bool resetNN = ImGui::Button("Reset");

                        ImGui::Text("Radiance Scale (Log10): %.2e", std::pow(10.0f, log10RadianceScale));
                        resetAccumulation |= ImGui::SliderFloat(
                            "##RadianceScale", &log10RadianceScale, -5, 5, "%.3f", ImGuiSliderFlags_AlwaysClamp);

                        PositionEncoding prevEncoding = g_positionEncoding;
                        ImGui::Text("Position Encoding");
                        ImGui::RadioButtonE("Triangle Wave", &g_positionEncoding, PositionEncoding::TriangleWave);
                        ImGui::RadioButtonE("Hash Grid", &g_positionEncoding, PositionEncoding::HashGrid);

                        ImGui::Text("MLP Num Hidden Layers");
                        uint32_t prevNumHiddenLayers = g_numHiddenLayers;
                        static bool use5HiddenLayers = g_numHiddenLayers == 5;
                        if (ImGui::RadioButton("5", use5HiddenLayers))
                            use5HiddenLayers = true;
                        ImGui::SameLine();
                        if (ImGui::RadioButton("2", !use5HiddenLayers))
                            use5HiddenLayers = false;
                        g_numHiddenLayers = use5HiddenLayers ? 5 : 2;

                        float prevLearningRate = g_learningRate;
                        {
                            static int32_t log10LearningRate =
                                static_cast<int32_t>(std::round(std::log10(g_learningRate)));
                            ImGui::Text("Init Learning Rate: %.0e", g_learningRate);
                            ImGui::SliderInt("##InitLearningRate", &log10LearningRate, -5, -1,
                                             "", ImGuiSliderFlags_AlwaysClamp);
                            g_learningRate = std::pow(10, log10LearningRate);
                        }

                        resetNN |=
                            g_positionEncoding != prevEncoding ||
                            g_numHiddenLayers != prevNumHiddenLayers ||
                            g_learningRate != prevLearningRate;
                        if (resetNN) {
                            neuralRadianceCache.finalize();
                            neuralRadianceCache.initialize(g_positionEncoding, g_numHiddenLayers, g_learningRate);
                            resetAccumulation = true;
                        }
                    }

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
                    ImGui::Checkbox("Visualize Training Paths", &visualizeTrainingPath);
                    ImGui::Text("Buffer to Display");
                    ImGui::RadioButtonE(
                        "Noisy Beauty", &bufferTypeToDisplay, shared::BufferToDisplay::NoisyBeauty);
                    ImGui::RadioButtonE("Albedo", &bufferTypeToDisplay, shared::BufferToDisplay::Albedo);
                    ImGui::RadioButtonE("Normal", &bufferTypeToDisplay, shared::BufferToDisplay::Normal);
                    ImGui::RadioButtonE("Motion Vector", &bufferTypeToDisplay, shared::BufferToDisplay::Flow);
                    if (useNRC) {
                        ImGui::RadioButtonE("Rendering Path Length", &bufferTypeToDisplay, shared::BufferToDisplay::RenderingPathLength);
                        ImGui::RadioButtonE("Directly Visualized Prediction", &bufferTypeToDisplay, shared::BufferToDisplay::DirectlyVisualizedPrediction);
                    }
                    ImGui::RadioButtonE(
                        "Denoised Beauty", &bufferTypeToDisplay, shared::BufferToDisplay::DenoisedBeauty);

                    if (ImGui::Checkbox("Temporal Denoiser", &useTemporalDenosier)) {
                        CUDADRV_CHECK(cuStreamSynchronize(cuStream));
                        denoiser.destroy();

                        OptixDenoiserModelKind modelKind = useTemporalDenosier ?
                            OPTIX_DENOISER_MODEL_KIND_TEMPORAL :
                            OPTIX_DENOISER_MODEL_KIND_HDR;
                        denoiser = gpuEnv.optixContext.createDenoiser(
                            modelKind,
                            optixu::GuideAlbedo::Yes, optixu::GuideNormal::Yes);

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

                        denoiser.setupState(cuStream, denoiserStateBuffer, denoiserScratchBuffer);
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
            static MovingAverageTime preprocessNRCTime;
            static MovingAverageTime pathTraceTime;
            static MovingAverageTime inferTime;
            static MovingAverageTime accumulateInferredRadiancesTime;
            static MovingAverageTime propagateRadiancesTime;
            static MovingAverageTime shuffleTrainingDataTime;
            static MovingAverageTime trainTime;
            static MovingAverageTime visualizeCacheTime;
            static MovingAverageTime denoiseTime;

            cudaFrameTime.append(curGPUTimer.frame.report());
            updateTime.append(curGPUTimer.update.report());
            computePDFTextureTime.append(curGPUTimer.computePDFTexture.report());
            setupGBuffersTime.append(curGPUTimer.setupGBuffers.report());
            preprocessNRCTime.append(curGPUTimer.preprocessNRC.report());
            pathTraceTime.append(curGPUTimer.pathTrace.report());
            inferTime.append(curGPUTimer.infer.report());
            accumulateInferredRadiancesTime.append(curGPUTimer.accumulateInferredRadiances.report());
            propagateRadiancesTime.append(curGPUTimer.propagateRadiances.report());
            shuffleTrainingDataTime.append(curGPUTimer.shuffleTrainingData.report());
            trainTime.append(curGPUTimer.train.report());
            visualizeCacheTime.append(curGPUTimer.visualizeCache.report());
            denoiseTime.append(curGPUTimer.denoise.report());

            //ImGui::SetNextItemWidth(100.0f);
            ImGui::Text("CUDA/OptiX GPU %.3f [ms]:", cudaFrameTime.getAverage());
            ImGui::Text("  update: %.3f [ms]", updateTime.getAverage());
            ImGui::Text("  compute PDF Texture: %.3f [ms]", computePDFTextureTime.getAverage());
            ImGui::Text("  setup G-buffers: %.3f [ms]", setupGBuffersTime.getAverage());
            ImGui::Text("  pre-process NRC: %.3f [ms]", preprocessNRCTime.getAverage());
            ImGui::Text("  pathTrace: %.3f [ms]", pathTraceTime.getAverage());
            ImGui::Text("  inference: %.3f [ms]", inferTime.getAverage());
            ImGui::Text("  accum radiance: %.3f [ms]", accumulateInferredRadiancesTime.getAverage());
            ImGui::Text("  prop radiance: %.3f [ms]", propagateRadiancesTime.getAverage());
            ImGui::Text("  shuffle train data: %.3f [ms]", shuffleTrainingDataTime.getAverage());
            ImGui::Text("  training: %.3f [ms]", trainTime.getAverage());
            if (bufferTypeToDisplay == shared::BufferToDisplay::DirectlyVisualizedPrediction)
                ImGui::Text("  visualize cache: %.3f [ms]", visualizeCacheTime.getAverage());
            if (bufferTypeToDisplay == shared::BufferToDisplay::DenoisedBeauty)
                ImGui::Text("  denoise: %.3f [ms]", denoiseTime.getAverage());

            ImGui::Text("%u [spp]", std::min(numAccumFrames + 1, (1u << log2MaxNumAccums)));

            ImGui::End();
        }

        applyToneMapAndGammaCorrection =
            bufferTypeToDisplay == shared::BufferToDisplay::NoisyBeauty ||
            bufferTypeToDisplay == shared::BufferToDisplay::DirectlyVisualizedPrediction ||
            bufferTypeToDisplay == shared::BufferToDisplay::DenoisedBeauty;



        curGPUTimer.frame.start(cuStream);

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
                                                &instData, sizeof(instData), cuStream));
            }
            curInstDataBuffer.unmap();
        }

        // JP: IASのリビルドを行う。
        //     アップデートの代用としてのリビルドでは、インスタンスの追加・削除や
        //     ASビルド設定の変更を行っていないのでmarkDirty()やprepareForBuild()は必要無い。
        // EN: Rebuild the IAS.
        //     Rebuild as the alternative for update doesn't involves
        //     add/remove of instances and changes of AS build settings
        //     so neither of markDirty() nor prepareForBuild() is required.
        curGPUTimer.update.start(cuStream);
        if (animate)
            perFramePlp.travHandle = scene.ias.rebuild(
                cuStream, scene.iasInstanceBuffer, scene.iasMem, scene.asScratchMem);
        curGPUTimer.update.stop(cuStream);

        // JP: 光源となるインスタンスのProbability Textureを計算する。
        // EN: Compute the probability texture for light instances.
        curGPUTimer.computePDFTexture.start(cuStream);
        {
            CUdeviceptr probTexAddr =
                staticPlpOnDevice + offsetof(shared::StaticPipelineLaunchParameters, lightInstDist);
            scene.setupLightInstDistribution(cuStream, probTexAddr, bufferIndex);
        }
        curGPUTimer.computePDFTexture.stop(cuStream);

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
        perFramePlp.instanceDataBuffer = shared::ROBuffer(
            scene.instDataBuffer[bufferIndex].getDevicePointer(),
            scene.instDataBuffer[bufferIndex].numElements());
        perFramePlp.radianceScale = std::pow(10.0f, log10RadianceScale);
        perFramePlp.envLightPowerCoeff = std::pow(10.0f, log10EnvLightPowerCoeff);
        perFramePlp.envLightRotation = envLightRotation;
        perFramePlp.mousePosition = int2(static_cast<int32_t>(g_mouseX),
                                         static_cast<int32_t>(g_mouseY));
        perFramePlp.pickInfo = pickInfos[bufferIndex].getDevicePointer();

        perFramePlp.maxPathLength = infBounces ? 0 : maxPathLength;
        perFramePlp.bufferIndex = bufferIndex;
        perFramePlp.resetFlowBuffer = newSequence;
        perFramePlp.enableJittering = enableJittering;
        perFramePlp.enableEnvLight = enableEnvLight;
        perFramePlp.enableBumpMapping = enableBumpMapping;
        for (int i = 0; i < lengthof(debugSwitches); ++i)
            perFramePlp.setDebugSwitch(i, debugSwitches[i]);

        CUDADRV_CHECK(cuMemcpyHtoDAsync(perFramePlpOnDevice, &perFramePlp, sizeof(perFramePlp), cuStream));

        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
        CUDADRV_CHECK(cuMemcpyHtoDAsync(gpuEnv.plpPtr, &plp, sizeof(plp), cuStream));
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpPtrOnDeviceForCopyBuffers, &plp, sizeof(plp), cuStream));

        // JP: Gバッファーのセットアップ。
        //     ここではレイトレースを使ってGバッファーを生成しているがもちろんラスタライザーで生成可能。
        // EN: Setup the G-buffers.
        //     Generate the G-buffers using ray trace here, but of course this can be done using rasterizer.
        curGPUTimer.setupGBuffers.start(cuStream);
        gpuEnv.gBuffer.optixPipeline.launch(
            cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
        curGPUTimer.setupGBuffers.stop(cuStream);

        // JP: タイルサイズのアップデートやTraining Suffixの終端情報初期化などを行う。
        // EN: Perform update of the tile size and initialization of training suffixes and so on.
        if (useNRC) {
            curGPUTimer.preprocessNRC.start(cuStream);
            gpuEnv.kernelPreprocessNRC.launchWithThreadDim(
                cuStream, cudau::dim3(maxNumTrainingSuffixes),
                perFrameRng(), perFrameRng(), newSequence);
            curGPUTimer.preprocessNRC.stop(cuStream);
        }

        // JP: パストレースを行い、Rendering Pathと訓練データの生成を行う。
        // EN: Path trace to generate rendering paths and training data.
        curGPUTimer.pathTrace.start(cuStream);
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
        PathTracingEntryPoint entryPoint = useNRC ?
            PathTracingEntryPoint::NRC : PathTracingEntryPoint::Baseline;
        gpuEnv.pathTracing.setEntryPoint(entryPoint);
        gpuEnv.pathTracing.optixPipeline.launch(
            cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
        curGPUTimer.pathTrace.stop(cuStream);

        if (useNRC) {
            // JP: CUDAではdispatchIndirectのような動的なディスパッチサイズの指定が
            //     サポートされていないので仕方なくGPUと同期して訓練データ数などを取得する。
            //     実際のリアルタイムレンダリング実装の場合は動的なディスパッチサイズ指定を行う必要がある。
            // EN: CUDA does not support dynamic dispatch size like dispatchIndirect,
            //     so it has no choice but to synchronize with the GPU to obtain the number of training data and so on.
            //     Practical real-time rendering implementation requires dynamic dispatch size specification.
            uint32_t numTrainingDataOnHost;
            uint2 tileSizeOnHost;
            numTrainingData[bufferIndex].read(&numTrainingDataOnHost, 1, cuStream);
            tileSize[bufferIndex].read(&tileSizeOnHost, 1, cuStream);
            uint2 numTiles = (uint2(renderTargetSizeX, renderTargetSizeY) + tileSizeOnHost - 1) / tileSizeOnHost;
            uint32_t numInferenceQueries = renderTargetSizeX * renderTargetSizeY + numTiles.x * numTiles.y;
            numInferenceQueries = (numInferenceQueries + 127) / 128 * 128;
            //printf("numTrainingData: %u, TileSize: %u x %u\n",
            //       numTrainingData, tileSize.x, tileSize.y);

            // JP: Rendering PathとTraining Suffixの終端の輝度を推定する。
            // EN: Predict radiance values at the terminals of rendering paths and training suffixes.
            curGPUTimer.infer.start(cuStream);
            neuralRadianceCache.infer(
                cuStream,
                reinterpret_cast<float*>(inferenceRadianceQueryBuffer.getDevicePointer()),
                numInferenceQueries,
                reinterpret_cast<float*>(inferredRadianceBuffer.getDevicePointer()));
            curGPUTimer.infer.stop(cuStream);

            // JP: 各ピクセルに推定した輝度を加算して現在のフレームを完成させる。
            // EN: Accumulate the predicted radiance values to the pixels to complete the current frame.
            curGPUTimer.accumulateInferredRadiances.start(cuStream);
            gpuEnv.kernelAccumulateInferredRadianceValues(
                cuStream,
                gpuEnv.kernelAccumulateInferredRadianceValues.calcGridDim(renderTargetSizeX * renderTargetSizeY));
            curGPUTimer.accumulateInferredRadiances.stop(cuStream);

            prevTrainDone = false;
            if (train || stepTrain) {
                prevTrainDone = true;

                // JP: Training Suffixの終端から輝度を伝播させてTraining Vertexのデータを完成させる。
                // EN: Propagate the radiance values from the terminals of training suffixes to
                //     complete training vertex data.
                curGPUTimer.propagateRadiances.start(cuStream);
                gpuEnv.kernelPropagateRadianceValues.launchWithThreadDim(
                    cuStream, cudau::dim3(maxNumTrainingSuffixes));
                curGPUTimer.propagateRadiances.stop(cuStream);

                // JP: 訓練データの空間的な相関を取り除くためにデータをシャッフルする。
                // EN: Shuffle the training data to get rid of spatial correlations of the training data.
                curGPUTimer.shuffleTrainingData.start(cuStream);
                gpuEnv.kernelShuffleTrainingData.launchWithThreadDim(
                    cuStream, cudau::dim3(shared::numTrainingDataPerFrame));
                curGPUTimer.shuffleTrainingData.stop(cuStream);

                // JP: トレーニングの実行。
                // EN: Perform training.
                curGPUTimer.train.start(cuStream);
                {
                    constexpr uint32_t batchSize = shared::numTrainingDataPerFrame / 4;
                    static_assert((batchSize & 0xFF) == 0, "Batch size has to be a multiple of 256.");
                    //const uint32_t targetBatchSize =
                    //    (std::min(numTrainingData, shared::numTrainingDataPerFrame) / 4 + 255) / 256 * 256;
                    uint32_t dataStartIndex = 0;
                    for (int step = 0; step < 4; ++step) {
                        //uint32_t batchSize = std::min(numTrainingData - dataStartIndex, targetBatchSize);
                        //batchSize = batchSize / 256 * 256;
                        neuralRadianceCache.train(
                            cuStream,
                            reinterpret_cast<float*>(trainRadianceQueryBuffer[1].getDevicePointerAt(dataStartIndex)),
                            reinterpret_cast<float*>(trainTargetBuffer[1].getDevicePointerAt(dataStartIndex)),
                            batchSize,
                            (showLossValue && step == 3) ? &lossValue : nullptr);
                        dataStartIndex += batchSize;
                    }
                }
                curGPUTimer.train.stop(cuStream);
            }
        }

        // JP: ニューラルネットワークの推定値を直接可視化する。
        // EN: Directly visualize the predictions of the neural network.
        if (bufferTypeToDisplay == shared::BufferToDisplay::DirectlyVisualizedPrediction) {
            curGPUTimer.visualizeCache.start(cuStream);

            gpuEnv.pathTracing.setEntryPoint(PathTracingEntryPoint::visualizePrediction);
            gpuEnv.pathTracing.optixPipeline.launch(
                cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);

            neuralRadianceCache.infer(
                cuStream,
                reinterpret_cast<float*>(inferenceRadianceQueryBuffer.getDevicePointer()),
                renderTargetSizeX * renderTargetSizeY,
                reinterpret_cast<float*>(inferredRadianceBuffer.getDevicePointer()));

            curGPUTimer.visualizeCache.stop(cuStream);
        }

        // JP: 結果をリニアバッファーにコピーする。(法線の正規化も行う。)
        // EN: Copy the results to the linear buffers (and normalize normals).
        kernelCopyToLinearBuffers.launchWithThreadDim(
            cuStream, cudau::dim3(renderTargetSizeX, renderTargetSizeY),
            beautyAccumBuffer.getSurfaceObject(0),
            albedoAccumBuffer.getSurfaceObject(0),
            normalAccumBuffer.getSurfaceObject(0),
            gBuffer2[bufferIndex].getSurfaceObject(0),
            linearBeautyBuffer,
            linearAlbedoBuffer,
            linearNormalBuffer,
            linearFlowBuffer,
            uint2(renderTargetSizeX, renderTargetSizeY));

        curGPUTimer.denoise.start(cuStream);
        if (bufferTypeToDisplay == shared::BufferToDisplay::DenoisedBeauty) {
            denoiser.computeNormalizer(
                cuStream,
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
                    cuStream, denoisingTasks[i], inputBuffers,
                    optixu::IsFirstFrame(newSequence), OPTIX_DENOISER_ALPHA_MODE_COPY, hdrNormalizer, 0.0f,
                    linearDenoisedBeautyBuffer, nullptr,
                    optixu::BufferView());
        }
        curGPUTimer.denoise.stop(cuStream);

        outputBufferSurfaceHolder.beginCUDAAccess(cuStream);

        // JP: デノイズ結果や中間バッファーの可視化。
        // EN: Visualize the denosed result or intermediate buffers.
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
        case shared::BufferToDisplay::Flow:
            bufferToDisplay = linearFlowBuffer.getDevicePointer();
            break;
        case shared::BufferToDisplay::RenderingPathLength:
        case shared::BufferToDisplay::DirectlyVisualizedPrediction:
            break;
        case shared::BufferToDisplay::DenoisedBeauty:
            bufferToDisplay = linearDenoisedBeautyBuffer.getDevicePointer();
            break;
        default:
            Assert_ShouldNotBeCalled();
            break;
        }
        kernelVisualizeToOutputBuffer.launchWithThreadDim(
            cuStream, cudau::dim3(renderTargetSizeX, renderTargetSizeY),
            visualizeTrainingPath,
            bufferToDisplay, bufferTypeToDisplay,
            0.5f, std::pow(10.0f, motionVectorScale),
            outputBufferSurfaceHolder.getNext());

        outputBufferSurfaceHolder.endCUDAAccess(cuStream, true);

        curGPUTimer.frame.stop(cuStream);



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

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));
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

    neuralRadianceCache.finalize();
    dataShufflerBuffer.finalize();
    trainSuffixTerminalInfoBuffer.finalize();
    trainVertexInfoBuffer.finalize();
    for (int i = 1; i >= 0; --i) {
        trainTargetBuffer[i].finalize();
        trainRadianceQueryBuffer[i].finalize();
    }
    CUDADRV_CHECK(cuMemFree(offsetToSelectTrainingPathOnDevice));
    CUDADRV_CHECK(cuMemFree(offsetToSelectUnbiasedTileOnDevice));
    for (int i = 1; i >= 0; --i) {
        targetAvg[i].finalize();
        targetMinMax[i].finalize();
        tileSize[i].finalize();
        numTrainingData[i].finalize();
    }



    CUDADRV_CHECK(cuMemFree(sceneAABBOnDevice));

    envLightImportanceMap.finalize(gpuEnv.cuContext);
    if (envLightTexture)
        cuTexObjectDestroy(envLightTexture);
    envLightArray.finalize();

    finalizeTextureCaches();

    CUDADRV_CHECK(cuStreamDestroy(cuStream));

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
