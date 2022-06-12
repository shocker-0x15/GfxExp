/*

コマンドラインオプション例 / Command line option example:
You can load a 3D model for example by downloading from the internet.
(e.g. https://casual-effects.com/data/)

(1) -cam-pos 0 3 0 -cam-yaw 90
    -name sponza -obj crytek_sponza/sponza.obj 0.01 trad
    -begin-pos 0 0 0.36125 -inst sponza
    -name rectlight -emittance 600 600 600 -rectangle 2 2 -begin-pos 0 15 0 -inst rectlight

JP: このプログラムはSVGF (Spatiotemporal Variance-Guided Filtering) [1]の実装例です。
    SVGFはパストレーシングなどによって得られたライティング結果を、物体表面のパラメターを参照しつつ画像空間で
    フィルタリングします。各ピクセルのライティングの分散を時間的・空間的にトラッキングし、
    分散が小さな箇所では小さなフィルター半径、分散が大きな箇所では大きなフィルター半径とすることで
    画像のぼけを抑えつつレンダリング画像における視覚的なノイズを低減します。
    フィルターにはà-trous Filterを用いることで大きなフィルタリング半径を比較的小さなコストで実現します。
    またSVGFとは基本的には直交する概念ですが、ライトトランスポートにおけるサンプルのTemporal Accumulation、
    最終レンダリングのTemporal Anti-Aliasingも併用することで画像の安定性を向上させています。
    ※デフォルトではBRDFにOptiXのCallable ProgramやCUDAの関数ポインターを使用した汎用的な実装になっており、
      性能上のオーバーヘッドが著しいため、純粋な性能を見る上では restir_shared.h の USE_HARD_CODED_BSDF_FUNCTIONS
      を有効化したほうがよいかもしれません。

EN: This program is an example implementation of SVGF (Spatiotemporal Variance-Guided Filtering) [1].
    SVGF filters the lighting result in screen-space obtained by methods like path tracing with references
    to surface parameters. It tracks the variance of the lighting for each pixel in spatially and temporally,
    then uses smaller filter radii at lower variance parts and larger filter radii at higher variance parts
    to get rid of perceptual noises from the rendered image while avoiding excessive blurs in the image.
    It uses an à-trous filter so that large filter radii can be used with relatively low costs.
    Additionally, temporal accumulation for light transport samples and temporal anti-alising
    for the final rendered image are used along with SVGF to improve image stability while these are basically
    orthogonally concept to the SVGF.
    * The program is generic implementation with OptiX's callable program and CUDA's function pointer,
      and has significant performance overhead, therefore it may be recommended to enable USE_HARD_CODED_BSDF_FUNCTIONS
      in restir_shared.h to see pure performance.

[1] Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination
    https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced

*/

#include "svgf_shared.h"
#include "../common/common_host.h"

// Include glfw3.h after our OpenGL definitions
#include "../utils/gl_util.h"
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"



enum class PathTracingEntryPoint {
    pathTraceWithoutTemporalAccumulation,
    pathTraceWithTemporalAccumulation,
};

struct GPUEnvironment {
    CUcontext cuContext;
    optixu::Context optixContext;

    CUmodule svgfModule;
    cudau::Kernel kernelEstimateVariance;
    cudau::Kernel kernelApplyATrousFilter_box3x3;
    cudau::Kernel kernelFeedbackNoisyLighting;
    cudau::Kernel kernelApplyAlbedoModulationAndTemporalAntiAliasing;
    CUdeviceptr plpPtrForSvgfModule;

    CUmodule debugVisualizeModule;
    cudau::Kernel kernelDebugVisualize;
    CUdeviceptr plpPtrForDebugVisualizeModule;

    template <typename EntryPointType>
    struct Pipeline {
        optixu::Pipeline optixPipeline;
        optixu::Module optixModule;
        std::unordered_map<EntryPointType, optixu::ProgramGroup> entryPoints;
        std::unordered_map<std::string, optixu::ProgramGroup> programs;
        std::vector<optixu::ProgramGroup> callablePrograms;
        cudau::Buffer sbt;
        cudau::Buffer hitGroupSbt;

        void setEntryPoint(EntryPointType et) {
            optixPipeline.setRayGenerationProgram(entryPoints.at(et));
        }
    };

    Pipeline<PathTracingEntryPoint> pathTracing;

    optixu::Material optixDefaultMaterial;

    void initialize() {
        int32_t cuDeviceCount;
        CUDADRV_CHECK(cuInit(0));
        CUDADRV_CHECK(cuDeviceGetCount(&cuDeviceCount));
        CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
        CUDADRV_CHECK(cuCtxSetCurrent(cuContext));

        CUDADRV_CHECK(cuModuleLoad(
            &svgfModule,
            (getExecutableDirectory() / "svgf/ptxes/svgf.ptx").string().c_str()));
        kernelEstimateVariance =
            cudau::Kernel(svgfModule, "estimateVariance", cudau::dim3(8, 8), 0);
        kernelApplyATrousFilter_box3x3 =
            cudau::Kernel(svgfModule, "applyATrousFilter_box3x3", cudau::dim3(8, 8), 0);
        kernelFeedbackNoisyLighting =
            cudau::Kernel(svgfModule, "feedbackNoisyLighting", cudau::dim3(8, 8), 0);
        kernelApplyAlbedoModulationAndTemporalAntiAliasing =
            cudau::Kernel(svgfModule, "applyAlbedoModulationAndTemporalAntiAliasing", cudau::dim3(8, 8), 0);

        size_t plpSize;

        CUDADRV_CHECK(cuModuleGetGlobal(&plpPtrForSvgfModule, &plpSize, svgfModule, "plp"));
        Assert(sizeof(shared::PipelineLaunchParameters) == plpSize, "Unexpected plp size.");

        CUDADRV_CHECK(cuModuleLoad(
            &debugVisualizeModule,
            (getExecutableDirectory() / "svgf/ptxes/visualize.ptx").string().c_str()));
        kernelDebugVisualize =
            cudau::Kernel(debugVisualizeModule, "debugVisualize", cudau::dim3(8, 8), 0);

        CUDADRV_CHECK(cuModuleGetGlobal(&plpPtrForDebugVisualizeModule, &plpSize, debugVisualizeModule, "plp"));
        Assert(sizeof(shared::PipelineLaunchParameters) == plpSize, "Unexpected plp size.");

        optixContext = optixu::Context::create(cuContext/*, 4, DEBUG_SELECT(true, false)*/);

        optixDefaultMaterial = optixContext.createMaterial();
        optixu::Module emptyModule;

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
                optixu::calcSumDwords<float2>(),
                "plp", sizeof(shared::PipelineLaunchParameters),
                false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE),
                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

            m = p.createModuleFromPTXString(
                readTxtFile(getExecutableDirectory() / "svgf/ptxes/optix_pathtracing_kernels.ptx"),
                OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
                DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            pipeline.entryPoints[PathTracingEntryPoint::pathTraceWithoutTemporalAccumulation] =
                p.createRayGenProgram(m, RT_RG_NAME_STR("pathTraceWithoutTemporalAccumulation"));
            pipeline.entryPoints[PathTracingEntryPoint::pathTraceWithTemporalAccumulation] =
                p.createRayGenProgram(m, RT_RG_NAME_STR("pathTraceWithTemporalAccumulation"));

            pipeline.programs[RT_MS_NAME_STR("pathTrace")] = p.createMissProgram(
                m, RT_MS_NAME_STR("pathTrace"));
            pipeline.programs[RT_CH_NAME_STR("pathTrace")] = p.createHitProgramGroupForTriangleIS(
                m, RT_CH_NAME_STR("pathTrace"),
                emptyModule, nullptr);

            pipeline.programs[RT_AH_NAME_STR("visibility")] = p.createHitProgramGroupForTriangleIS(
                emptyModule, nullptr,
                m, RT_AH_NAME_STR("visibility"));

            pipeline.programs["emptyMiss"] = p.createMissProgram(emptyModule, nullptr);

            p.setNumMissRayTypes(shared::PathTracingRayType::NumTypes);
            p.setMissProgram(
                shared::PathTracingRayType::Baseline, pipeline.programs.at(RT_MS_NAME_STR("pathTrace")));
            p.setMissProgram(shared::PathTracingRayType::Visibility, pipeline.programs.at("emptyMiss"));

            p.setNumCallablePrograms(NumCallablePrograms);
            pipeline.callablePrograms.resize(NumCallablePrograms);
            for (int i = 0; i < NumCallablePrograms; ++i) {
                optixu::ProgramGroup program = p.createCallableProgramGroup(
                    m, callableProgramEntryPoints[i],
                    emptyModule, nullptr);
                pipeline.callablePrograms[i] = program;
                p.setCallableProgram(i, program);
            }

            p.link(2, DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            optixDefaultMaterial.setHitGroup(
                shared::PathTracingRayType::Baseline, pipeline.programs.at(RT_CH_NAME_STR("pathTrace")));
            optixDefaultMaterial.setHitGroup(
                shared::PathTracingRayType::Visibility, pipeline.programs.at(RT_AH_NAME_STR("visibility")));

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

        optixDefaultMaterial.destroy();

        optixContext.destroy();

        CUDADRV_CHECK(cuModuleUnload(debugVisualizeModule));
        CUDADRV_CHECK(cuModuleUnload(svgfModule));

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
static KeyState g_buttonRotate;
static double g_mouseX;
static double g_mouseY;

static float g_initBrightness = 0.0f;
static float g_cameraPositionalMovingSpeed;
static float g_cameraDirectionalMovingSpeed;
static float g_cameraTiltSpeed;
static Quaternion g_cameraOrientation;
static Quaternion g_tempCameraOrientation;
static float3 g_cameraPosition;
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
    float3 emittance;
    std::filesystem::path emitterTexPath;
};

struct MeshInstanceInfo {
    std::string name;
    float3 beginPosition;
    float3 endPosition;
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

    float3 beginPosition = float3(0.0f, 0.0f, 0.0f);
    float3 endPosition = float3(NAN, NAN, NAN);
    Quaternion beginOrientation = Quaternion();
    Quaternion endOrientation = Quaternion(NAN, NAN, NAN, NAN);
    float beginScale = 1.0f;
    float endScale = NAN;
    float frequency = 5.0f;
    float initTime = 0.0f;
    float3 emittance = float3(0.0f, 0.0f, 0.0f);
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
            g_cameraPosition = float3(atof(argv[i + 1]), atof(argv[i + 2]), atof(argv[i + 3]));
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
            emittance = float3(atof(argv[i + 1]), atof(argv[i + 2]), atof(argv[i + 3]));
            if (!allFinite(emittance)) {
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

            emittance = float3(0.0f, 0.0f, 0.0f);
            rectEmitterTexPath = "";

            i += 2;
        }
        else if (0 == strncmp(arg, "-begin-pos", 11)) {
            if (i + 3 >= argc) {
                printf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            beginPosition = float3(atof(argv[i + 1]), atof(argv[i + 2]), atof(argv[i + 3]));
            if (!allFinite(beginPosition)) {
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
            endPosition = float3(atof(argv[i + 1]), atof(argv[i + 2]), atof(argv[i + 3]));
            if (!allFinite(endPosition)) {
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
            info.endPosition = allFinite(endPosition) ? endPosition : beginPosition;
            info.endOrientation = endOrientation.allFinite() ? endOrientation : beginOrientation;
            info.endScale = std::isfinite(endScale) ? endScale : beginScale;
            info.frequency = frequency;
            info.initTime = initTime;
            g_meshInstInfos.push_back(info);

            beginPosition = float3(0.0f, 0.0f, 0.0f);
            endPosition = float3(NAN, NAN, NAN);
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
    GLFWwindow* window = glfwCreateWindow(static_cast<int32_t>(renderTargetSizeX * UIScaling),
                                          static_cast<int32_t>(renderTargetSizeY * UIScaling),
                                          "SVGF", NULL, NULL);
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
        getExecutableDirectory() / "svgf/ptxes",
        gpuEnv.cuContext, gpuEnv.optixContext, shared::maxNumRayTypes, gpuEnv.optixDefaultMaterial);

    CUstream cuStream;
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    scene.map();

    constexpr bool allocateGfxResource = true;

    for (auto it = g_meshInfos.cbegin(); it != g_meshInfos.cend(); ++it) {
        const MeshInfo &info = it->second;

        if (std::holds_alternative<MeshGeometryInfo>(info)) {
            const auto &meshInfo = std::get<MeshGeometryInfo>(info);

            createTriangleMeshes(it->first,
                                 meshInfo.path, meshInfo.matConv,
                                 scale4x4(meshInfo.preScale),
                                 gpuEnv.cuContext, &scene, allocateGfxResource);
        }
        else if (std::holds_alternative<RectangleGeometryInfo>(info)) {
            const auto &rectInfo = std::get<RectangleGeometryInfo>(info);

            createRectangleLight(it->first,
                                 rectInfo.dimX, rectInfo.dimZ,
                                 float3(0.01f),
                                 rectInfo.emitterTexPath, rectInfo.emittance, Matrix4x4(),
                                 gpuEnv.cuContext, &scene, allocateGfxResource);
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

            if (info.beginPosition != info.endPosition ||
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

    float3 sceneDim = scene.initialSceneAabb.maxP - scene.initialSceneAabb.minP;
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

    // END: Setup a scene.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: 解像度依存のバッファーを初期化。
    // EN: Initialize resolution dependent buffers.

    struct TemporalSet {
        glu::FrameBuffer gfxGBuffers;
        // Workaround for CUDA/OpenGL interop.
        // CUDA seems not to allow to create Array from depth texture.
        glu::FrameBuffer gfxDepthBufferCopy;
        cudau::Array cuArrayGBuffer0;
        cudau::Array cuArrayGBuffer1;
        cudau::Array cuArrayGBuffer2;
        cudau::Array cuArrayDepthBuffer;
        cudau::InteropSurfaceObjectHolder<2> gBuffer0InteropHandler;
        cudau::InteropSurfaceObjectHolder<2> gBuffer1InteropHandler;
        cudau::InteropSurfaceObjectHolder<2> gBuffer2InteropHandler;
        cudau::InteropSurfaceObjectHolder<2> depthBufferInteropHandler;

        cudau::Array cuArray_momentPair_sampleInfo_buffer;

        glu::Texture2D gfxFinalLightingBuffer;
        cudau::Array cuArrayFinalLightingBuffer;
        cudau::InteropSurfaceObjectHolder<2> finalLightingBufferInteropHandler;
    };

    GLenum gBufferFormats[] = {
        GL_RGBA32F,
        GL_RGBA32F,
        GL_RGBA32F,
    };
    GLenum depthFormat = GL_DEPTH_COMPONENT32;
    GLenum depthAlternativeFormat = GL_R32F;
    cudau::TextureSampler gBufferSampler;
    TemporalSet temporalSets[2];
    cudau::Array albedoBuffer;
    cudau::Array lighting_variance_buffers[2];
    cudau::Array prevNoisyLightingBuffer;
    cudau::Array rngBuffer;
    glu::Texture2D gfxDebugVisualizeBuffer;
    cudau::Array cuArrayDebugVisualizeBuffer;
    cudau::InteropSurfaceObjectHolder<2> debugVisualizeBufferInteropHandler;

    gBufferSampler.setXyFilterMode(cudau::TextureFilterMode::Point);
    gBufferSampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    gBufferSampler.setWrapMode(0, cudau::TextureWrapMode::Clamp);
    gBufferSampler.setWrapMode(1, cudau::TextureWrapMode::Clamp);

    const auto initializeResolutionDependentBuffers = [&]() {
        for (int i = 0; i < 2; ++i) {
            TemporalSet &temporalSet = temporalSets[i];

            temporalSet.gfxGBuffers.initialize(
                renderTargetSizeX, renderTargetSizeY, 1,
                gBufferFormats, 0, lengthof(gBufferFormats),
                &depthFormat, false);
            temporalSet.gfxDepthBufferCopy.initialize(
                renderTargetSizeX, renderTargetSizeY, 1,
                &depthAlternativeFormat, 0, 1,
                nullptr, false);

            temporalSet.cuArrayGBuffer0.initializeFromGLTexture2D(
                gpuEnv.cuContext, temporalSet.gfxGBuffers.getRenderTargetTexture(0, 0).getHandle(),
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
            temporalSet.cuArrayGBuffer1.initializeFromGLTexture2D(
                gpuEnv.cuContext, temporalSet.gfxGBuffers.getRenderTargetTexture(1, 0).getHandle(),
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
            temporalSet.cuArrayGBuffer2.initializeFromGLTexture2D(
                gpuEnv.cuContext, temporalSet.gfxGBuffers.getRenderTargetTexture(2, 0).getHandle(),
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
            temporalSet.cuArrayDepthBuffer.initializeFromGLTexture2D(
                gpuEnv.cuContext, temporalSet.gfxDepthBufferCopy.getRenderTargetTexture(0, 0).getHandle(),
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);

            temporalSet.gBuffer0InteropHandler.initialize({ &temporalSet.cuArrayGBuffer0 });
            temporalSet.gBuffer1InteropHandler.initialize({ &temporalSet.cuArrayGBuffer1 });
            temporalSet.gBuffer2InteropHandler.initialize({ &temporalSet.cuArrayGBuffer2 });
            temporalSet.depthBufferInteropHandler.initialize({ &temporalSet.cuArrayDepthBuffer });

            temporalSet.cuArray_momentPair_sampleInfo_buffer.initialize2D(
                gpuEnv.cuContext, cudau::ArrayElementType::UInt32,
                (sizeof(shared::MomentPair_SampleInfo) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                renderTargetSizeX, renderTargetSizeY, 1);

            temporalSet.gfxFinalLightingBuffer.initialize(
                GL_RGBA32F, renderTargetSizeX, renderTargetSizeY, 1);
            temporalSet.cuArrayFinalLightingBuffer.initializeFromGLTexture2D(
                gpuEnv.cuContext, temporalSet.gfxFinalLightingBuffer.getHandle(),
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
            temporalSet.finalLightingBufferInteropHandler.initialize({ &temporalSet.cuArrayFinalLightingBuffer });
        }

        albedoBuffer.initialize2D(
            gpuEnv.cuContext, cudau::ArrayElementType::UInt32, (sizeof(shared::Albedo) + 3) / 4,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            renderTargetSizeX, renderTargetSizeY, 1);
        for (int i = 0; i < 2; ++i) {
            lighting_variance_buffers[i].initialize2D(
                gpuEnv.cuContext, cudau::ArrayElementType::UInt32, (sizeof(shared::Lighting_Variance) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                renderTargetSizeX, renderTargetSizeY, 1);
        }
        prevNoisyLightingBuffer.initialize2D(
            gpuEnv.cuContext, cudau::ArrayElementType::UInt32, (sizeof(shared::Lighting_Variance) + 3) / 4,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            renderTargetSizeX, renderTargetSizeY, 1);

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

        gfxDebugVisualizeBuffer.initialize(
            GL_RGBA32F, renderTargetSizeX, renderTargetSizeY, 1);
        cuArrayDebugVisualizeBuffer.initializeFromGLTexture2D(
            gpuEnv.cuContext, gfxDebugVisualizeBuffer.getHandle(),
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
        debugVisualizeBufferInteropHandler.initialize({ &cuArrayDebugVisualizeBuffer });
    };

    const auto finalizeResolutionDependentBuffers = [&]() {
        debugVisualizeBufferInteropHandler.finalize();
        cuArrayDebugVisualizeBuffer.finalize();
        gfxDebugVisualizeBuffer.finalize();

        rngBuffer.finalize();

        prevNoisyLightingBuffer.finalize();
        for (int i = 1; i >= 0; --i)
            lighting_variance_buffers[i].finalize();
        albedoBuffer.finalize();

        for (int i = 1; i >= 0; --i) {
            TemporalSet &temporalSet = temporalSets[i];

            temporalSet.finalLightingBufferInteropHandler.finalize();
            temporalSet.cuArrayFinalLightingBuffer.finalize();

            temporalSet.cuArray_momentPair_sampleInfo_buffer.finalize();

            temporalSet.depthBufferInteropHandler.finalize();
            temporalSet.gBuffer2InteropHandler.finalize();
            temporalSet.gBuffer1InteropHandler.finalize();
            temporalSet.gBuffer0InteropHandler.finalize();

            temporalSet.cuArrayDepthBuffer.finalize();
            temporalSet.cuArrayGBuffer2.finalize();
            temporalSet.cuArrayGBuffer1.finalize();
            temporalSet.cuArrayGBuffer0.finalize();

            temporalSet.gfxDepthBufferCopy.finalize();
            temporalSet.gfxGBuffers.finalize();
        }
    };

    const auto resizeResolutionDependentBuffers = [&](uint32_t width, uint32_t height) {
        for (int i = 0; i < 2; ++i) {
            TemporalSet &temporalSet = temporalSets[i];

            temporalSet.gfxGBuffers.finalize();
            temporalSet.gfxGBuffers.initialize(
                width, height, 1,
                gBufferFormats, 0, lengthof(gBufferFormats),
                &depthFormat, false);
            temporalSet.gfxDepthBufferCopy.finalize();
            temporalSet.gfxDepthBufferCopy.initialize(
                width, height, 1,
                &depthAlternativeFormat, 0, 1,
                nullptr, false);

            temporalSet.cuArrayGBuffer0.finalize();
            temporalSet.cuArrayGBuffer0.initializeFromGLTexture2D(
                gpuEnv.cuContext, temporalSet.gfxGBuffers.getRenderTargetTexture(0, 0).getHandle(),
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
            temporalSet.cuArrayGBuffer1.finalize();
            temporalSet.cuArrayGBuffer1.initializeFromGLTexture2D(
                gpuEnv.cuContext, temporalSet.gfxGBuffers.getRenderTargetTexture(1, 0).getHandle(),
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
            temporalSet.cuArrayGBuffer2.finalize();
            temporalSet.cuArrayGBuffer2.initializeFromGLTexture2D(
                gpuEnv.cuContext, temporalSet.gfxGBuffers.getRenderTargetTexture(2, 0).getHandle(),
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
            temporalSet.cuArrayDepthBuffer.finalize();
            temporalSet.cuArrayDepthBuffer.initializeFromGLTexture2D(
                gpuEnv.cuContext, temporalSet.gfxDepthBufferCopy.getRenderTargetTexture(0, 0).getHandle(),
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);

            temporalSet.cuArray_momentPair_sampleInfo_buffer.finalize();
            temporalSet.cuArray_momentPair_sampleInfo_buffer.initialize2D(
                gpuEnv.cuContext, cudau::ArrayElementType::UInt32,
                (sizeof(shared::MomentPair_SampleInfo) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                renderTargetSizeX, renderTargetSizeY, 1);

            temporalSet.gfxFinalLightingBuffer.finalize();
            temporalSet.gfxFinalLightingBuffer.initialize(
                GL_RGBA32F, renderTargetSizeX, renderTargetSizeY, 1);
            temporalSet.cuArrayFinalLightingBuffer.finalize();
            temporalSet.cuArrayFinalLightingBuffer.initializeFromGLTexture2D(
                gpuEnv.cuContext, temporalSet.gfxFinalLightingBuffer.getHandle(),
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
            temporalSet.finalLightingBufferInteropHandler.initialize({ &temporalSet.cuArrayFinalLightingBuffer });
        }

        albedoBuffer.resize(width, height);
        for (int i = 0; i < 2; ++i)
            lighting_variance_buffers[i].resize(width, height);
        prevNoisyLightingBuffer.resize(width, height);

        rngBuffer.resize(width, height);
        {
            auto rngs = rngBuffer.map<shared::PCG32RNG>();
            std::mt19937_64 rngSeed(591842031321323413);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    shared::PCG32RNG &rng = rngs[y * width + x];
                    rng.setState(rngSeed());
                }
            }
            rngBuffer.unmap();
        }

        gfxDebugVisualizeBuffer.finalize();
        gfxDebugVisualizeBuffer.initialize(
            GL_RGBA32F, renderTargetSizeX, renderTargetSizeY, 1);
        cuArrayDebugVisualizeBuffer.finalize();
        cuArrayDebugVisualizeBuffer.initializeFromGLTexture2D(
            gpuEnv.cuContext, gfxDebugVisualizeBuffer.getHandle(),
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
        debugVisualizeBufferInteropHandler.initialize({ &cuArrayDebugVisualizeBuffer });
    };

    initializeResolutionDependentBuffers();

    // END: Initialize resolution dependent buffers.
    // ----------------------------------------------------------------



    const char* glslHead = R"(#version 460)";

    glu::GraphicsProgram gBufferShader;
    gBufferShader.initializeVSPS(
        glslHead,
        exeDir / "svgf/shaders/draw_g_buffers.vert",
        exeDir / "svgf/shaders/draw_g_buffers.frag");

    glu::GraphicsProgram depthCopyShader;
    depthCopyShader.initializeVSPS(
        glslHead,
        exeDir / "svgf/shaders/full_screen.vert",
        exeDir / "svgf/shaders/depth_copy.frag");

    glu::GraphicsProgram drawResultShader;
    drawResultShader.initializeVSPS(
        glslHead,
        exeDir / "svgf/shaders/full_screen.vert",
        exeDir / "svgf/shaders/draw_result.frag");

    glu::Sampler outputSampler;
    outputSampler.initialize(glu::Sampler::MinFilter::Nearest, glu::Sampler::MagFilter::Nearest,
                             glu::Sampler::WrapMode::Repeat, glu::Sampler::WrapMode::Repeat);

    // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
    // EN: Empty VAO for full screen qud (or triangle).
    glu::VertexArray vertexArrayForFullScreen;
    vertexArrayForFullScreen.initialize();



    shared::StaticPipelineLaunchParameters staticPlp = {};
    {
        staticPlp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
        staticPlp.rngBuffer = rngBuffer.getSurfaceObject(0);

        for (int i = 0; i < 2; ++i) {
            const TemporalSet &srcTemporalSet = temporalSets[i];
            shared::StaticPipelineLaunchParameters::TemporalSet &dstTemporalSet = staticPlp.temporalSets[i];
            
            dstTemporalSet.momentPair_sampleInfo_buffer =
                srcTemporalSet.cuArray_momentPair_sampleInfo_buffer.getSurfaceObject(0);
        }
        staticPlp.albedoBuffer = albedoBuffer.getSurfaceObject(0);
        staticPlp.lighting_variance_buffers[0] = lighting_variance_buffers[0].getSurfaceObject(0);
        staticPlp.lighting_variance_buffers[1] = lighting_variance_buffers[1].getSurfaceObject(0);
        staticPlp.prevNoisyLightingBuffer = prevNoisyLightingBuffer.getSurfaceObject(0);

        staticPlp.materialDataBuffer = scene.materialDataBuffer.getDevicePointer();
        staticPlp.geometryInstanceDataBuffer = scene.geomInstDataBuffer.getDevicePointer();
        envLightImportanceMap.getDeviceType(&staticPlp.envLightImportanceMap);
        staticPlp.envLightTexture = envLightTexture;
    }
    CUdeviceptr staticPlpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&staticPlpOnDevice, sizeof(staticPlp)));
    CUDADRV_CHECK(cuMemcpyHtoD(staticPlpOnDevice, &staticPlp, sizeof(staticPlp)));

    shared::PerFramePipelineLaunchParameters perFramePlp = {};
    perFramePlp.travHandle = scene.ias.getHandle();
    {
        shared::PerspectiveCamera &camera = perFramePlp.temporalSets[0].camera;
        camera.fovY = 50 * pi_v<float> / 180;
        camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
        camera.position = g_cameraPosition;
        camera.orientation = g_cameraOrientation.toMatrix3x3();
        perFramePlp.temporalSets[1].camera = camera;
    }   
    perFramePlp.envLightPowerCoeff = 0;
    perFramePlp.envLightRotation = 0;

    CUdeviceptr perFramePlpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&perFramePlpOnDevice, sizeof(perFramePlp)));
    CUDADRV_CHECK(cuMemcpyHtoD(perFramePlpOnDevice, &perFramePlp, sizeof(perFramePlp)));
    
    shared::PipelineLaunchParameters plp;
    plp.s = reinterpret_cast<shared::StaticPipelineLaunchParameters*>(staticPlpOnDevice);
    plp.f = reinterpret_cast<shared::PerFramePipelineLaunchParameters*>(perFramePlpOnDevice);

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
    initPickInfo.positionInWorld = make_float3(0.0f);
    initPickInfo.albedo = make_float3(0.0f);
    initPickInfo.emittance = make_float3(0.0f);
    initPickInfo.normalInWorld = make_float3(0.0f);
    cudau::TypedBuffer<shared::PickInfo> pickInfos[2];
    pickInfos[0].initialize(gpuEnv.cuContext, Scene::bufferType, 1, initPickInfo);
    pickInfos[1].initialize(gpuEnv.cuContext, Scene::bufferType, 1, initPickInfo);

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    const auto computeHaltonSequence = []
    (uint32_t base, uint32_t idx) {
        const float recBase = 1.0f / base;
        float ret = 0.0f;
        float scale = 1.0f;
        while (idx) {
            scale *= recBase;
            ret += (idx % base) * scale;
            idx /= base;
        }
        return ret;
    };
    float2 subPixelOffsets[256];
    for (int i = 0; i < lengthof(subPixelOffsets); ++i)
        subPixelOffsets[i] = float2(computeHaltonSequence(2, i),
                                    computeHaltonSequence(3, i));



    struct GPUTimer {
        cudau::Timer frame;
        cudau::Timer update;
        cudau::Timer computePDFTexture;
        cudau::Timer setupGBuffers;
        cudau::Timer pathTrace;
        cudau::Timer denoise;
        cudau::Timer estimateVariance;
        cudau::Timer aTrousFilter;
        cudau::Timer temporalAA;

        void initialize(CUcontext context) {
            frame.initialize(context);
            update.initialize(context);
            computePDFTexture.initialize(context);
            setupGBuffers.initialize(context);
            pathTrace.initialize(context);
            denoise.initialize(context);
            estimateVariance.initialize(context);
            aTrousFilter.initialize(context);
            temporalAA.initialize(context);
        }
        void finalize() {
            estimateVariance.finalize();
            aTrousFilter.finalize();
            temporalAA.finalize();
            denoise.finalize();
            pathTrace.finalize();
            setupGBuffers.finalize();
            computePDFTexture.finalize();
            update.finalize();
            frame.finalize();
        }
    };

    Matrix4x4 matV2Cs[2];
    Matrix4x4 matW2Vs[2];
    constexpr float cameraNear = 0.1f;
    constexpr float cameraFar = 1000.0f;
    matV2Cs[0] = camera(
        perFramePlp.temporalSets[0].camera.aspect,
        perFramePlp.temporalSets[0].camera.fovY,
        cameraNear, cameraFar);
    matW2Vs[0] = inverse(Matrix4x4((g_cameraOrientation * qRotateY(M_PI)).toMatrix3x3(), g_cameraPosition));
    matV2Cs[1] = matV2Cs[0];
    matW2Vs[1] = matW2Vs[0];

    GPUTimer gpuTimers[2];
    gpuTimers[0].initialize(gpuEnv.cuContext);
    gpuTimers[1].initialize(gpuEnv.cuContext);
    uint64_t frameIndex = 0;
    glfwSetWindowUserPointer(window, &frameIndex);
    int32_t requestedSize[2];
    uint32_t numAccumFrames = 0;
    while (true) {
        const uint32_t curBufIdx = frameIndex % 2;
        const uint32_t prevBufIdx = (frameIndex + 1) % 2;

        TemporalSet &curTemporalSet = temporalSets[curBufIdx];
        TemporalSet &prevTemporalSet = temporalSets[prevBufIdx];
        shared::PerFramePipelineLaunchParameters::TemporalSet &curPerFrameTemporalSet =
            perFramePlp.temporalSets[curBufIdx];
        shared::PerFramePipelineLaunchParameters::TemporalSet &prevPerFrameTemporalSet =
            perFramePlp.temporalSets[prevBufIdx];
        Matrix4x4 &curMatV2C = matV2Cs[curBufIdx];
        Matrix4x4 &prevMatV2C = matV2Cs[prevBufIdx];
        Matrix4x4 &curMatW2V = matW2Vs[curBufIdx];
        Matrix4x4 &prevMatW2V = matW2Vs[prevBufIdx];

        cudau::TypedBuffer<shared::PickInfo> &curPickInfo = pickInfos[curBufIdx];

        GPUTimer &curGPUTimer = gpuTimers[curBufIdx];

        cudau::TypedBuffer<shared::InstanceData> &curInstDataBuffer = scene.instDataBuffer[curBufIdx];

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

            resizeResolutionDependentBuffers(renderTargetSizeX, renderTargetSizeY);

            // EN: update the pipeline parameters.
            staticPlp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
            staticPlp.rngBuffer = rngBuffer.getSurfaceObject(0);
            for (int i = 0; i < 2; ++i) {
                const TemporalSet &srcTemporalSet = temporalSets[i];
                shared::StaticPipelineLaunchParameters::TemporalSet &dstTemporalSet = staticPlp.temporalSets[i];

                dstTemporalSet.momentPair_sampleInfo_buffer =
                    srcTemporalSet.cuArray_momentPair_sampleInfo_buffer.getSurfaceObject(0);
            }
            staticPlp.albedoBuffer = albedoBuffer.getSurfaceObject(0);
            staticPlp.lighting_variance_buffers[0] = lighting_variance_buffers[0].getSurfaceObject(0);
            staticPlp.lighting_variance_buffers[1] = lighting_variance_buffers[1].getSurfaceObject(0);
            staticPlp.prevNoisyLightingBuffer = prevNoisyLightingBuffer.getSurfaceObject(0);
            curPerFrameTemporalSet.camera.aspect = (float)renderTargetSizeX / renderTargetSizeY;
            curMatV2C = camera(
                curPerFrameTemporalSet.camera.aspect,
                curPerFrameTemporalSet.camera.fovY,
                cameraNear, cameraFar);
            prevMatV2C = curMatV2C;

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
            float3 axis = float3(deltaY, -deltaX, 0);
            axis /= deltaAngle;
            if (deltaAngle == 0.0f)
                axis = float3(1, 0, 0);

            g_cameraOrientation = g_cameraOrientation * qRotateZ(g_cameraTiltSpeed * tiltZ);
            g_tempCameraOrientation =
                g_cameraOrientation *
                qRotate(g_cameraDirectionalMovingSpeed * deltaAngle, axis);
            g_cameraPosition +=
                g_tempCameraOrientation.toMatrix3x3() *
                (g_cameraPositionalMovingSpeed * float3(trackX, trackY, trackZ));
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

            prevPerFrameTemporalSet.camera = curPerFrameTemporalSet.camera;
            curPerFrameTemporalSet.camera.position = g_cameraPosition;
            curPerFrameTemporalSet.camera.orientation = g_tempCameraOrientation.toMatrix3x3();
        }



        bool resetAccumulation = false;
        
        // Camera Window
        static shared::BufferToDisplay bufferTypeToDisplay = shared::BufferToDisplay::FinalRendering;
        static bool applyToneMapAndGammaCorrection = true;
        static float brightness = g_initBrightness;
        static bool enableEnvLight = true;
        static float log10EnvLightPowerCoeff = 0.0f;
        static float envLightRotation = 0.0f;
        {
            ImGui::Begin("Camera / Env", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Text("W/A/S/D/R/F: Move, Q/E: Tilt");
            ImGui::Text("Mouse Middle Drag: Rotate");

            ImGui::InputFloat3("Position", reinterpret_cast<float*>(&curPerFrameTemporalSet.camera.position));
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
                glFinish();
                CUDADRV_CHECK(cuStreamSynchronize(cuStream));
                auto rawImage = new float4[renderTargetSizeX * renderTargetSizeY];
                glu::Texture2D &texToDisplay = bufferTypeToDisplay == shared::BufferToDisplay::FinalRendering ?
                    curTemporalSet.gfxFinalLightingBuffer : gfxDebugVisualizeBuffer;
                glGetTextureSubImage(
                    texToDisplay.getHandle(), 0,
                    0, 0, 0, renderTargetSizeX, renderTargetSizeY, 1,
                    GL_RGBA, GL_FLOAT, sizeof(float4) * renderTargetSizeX * renderTargetSizeY, rawImage);

                if (saveSS_LDR) {
                    SDRImageSaverConfig config;
                    config.brightnessScale = std::pow(10.0f, brightness);
                    config.applyToneMap = applyToneMapAndGammaCorrection;
                    config.apply_sRGB_gammaCorrection = applyToneMapAndGammaCorrection;
                    config.flipY = true;
                    saveImage("output.png", renderTargetSizeX, renderTargetSizeY, rawImage, config);
                }
                if (saveSS_HDR)
                    saveImageHDR("output.exr", renderTargetSizeX, renderTargetSizeY,
                                 std::pow(10.0f, brightness), rawImage, true);
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

        static float motionVectorScale = -1.0f;
        static bool animate = /*true*/false;
        static bool enableAccumulation = /*true*/false;
        static int32_t log2MaxNumAccums = 16;
        bool lastFrameWasAnimated = false;
        static int32_t maxPathLength = 5;
        static bool enableTemporalAccumulation = true;
        static bool enableSVGF = true;
        static bool feedback1stFilteredResult = true;
        static bool specularMollification = true;
        static bool enableTemporalAA = true;
        static bool modulateAlbedo = true;
        static float log2TaaHistoryLength = std::log2(16);
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

            ImGui::Separator();
            ImGui::Text("Cursor Info: %.1lf, %.1lf", g_mouseX, g_mouseY);
            shared::PickInfo pickInfoOnHost;
            curPickInfo.read(&pickInfoOnHost, 1, cuStream);
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
                        pickInfoOnHost.albedo.x,
                        pickInfoOnHost.albedo.y,
                        pickInfoOnHost.albedo.z);
            ImGui::Text("Emittance: %.3f, %.3f, %.3f",
                        pickInfoOnHost.emittance.x,
                        pickInfoOnHost.emittance.y,
                        pickInfoOnHost.emittance.z);

            ImGui::Separator();

            if (ImGui::BeginTabBar("MyTabBar")) {
                if (ImGui::BeginTabItem("Renderer")) {
                    resetAccumulation |= ImGui::SliderInt("Max Path Length", &maxPathLength, 2, 15);

                    resetAccumulation |= ImGui::Checkbox("Temporal Accumulation", &enableTemporalAccumulation);
                    resetAccumulation |= ImGui::Checkbox("SVGF", &enableSVGF);
                    if (enableSVGF) {
                        ImGui::Checkbox("Feedback 1st filtered result", &feedback1stFilteredResult);
                        ImGui::Checkbox("Specular Mollification", &specularMollification);
                    }

                    resetAccumulation |= ImGui::Checkbox("Temporal AA", &enableTemporalAA);
                    ImGui::PushID("TAA History Length");
                    ImGui::SliderFloat("", &log2TaaHistoryLength, 0, 8, "");
                    ImGui::SameLine();
                    uint32_t taaHistoryLength =
                        static_cast<uint32_t>(std::round(std::pow(2, log2TaaHistoryLength)));
                    ImGui::Text("TAA History Length (%2u)", taaHistoryLength);
                    ImGui::PopID();

                    resetAccumulation |= ImGui::Checkbox("Modulate Albedo", &modulateAlbedo);

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
                    ImGui::RadioButtonE(
                        "Variance", &bufferTypeToDisplay, shared::BufferToDisplay::Variance);
                    ImGui::RadioButtonE(
                        "Filtered Variance", &bufferTypeToDisplay, shared::BufferToDisplay::FilteredVariance);
                    ImGui::RadioButtonE(
                        "Albedo", &bufferTypeToDisplay, shared::BufferToDisplay::Albedo);
                    ImGui::RadioButtonE(
                        "Normal", &bufferTypeToDisplay, shared::BufferToDisplay::Normal);
                    ImGui::RadioButtonE(
                        "Motion Vector", &bufferTypeToDisplay, shared::BufferToDisplay::MotionVector);
                    ImGui::RadioButtonE(
                        "Sample Count", &bufferTypeToDisplay, shared::BufferToDisplay::SampleCount);
                    ImGui::RadioButtonE(
                        "Final Rendering", &bufferTypeToDisplay, shared::BufferToDisplay::FinalRendering);
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
            static MovingAverageTime estimateVarianceTime;
            static MovingAverageTime aTrousFilterTime;
            static MovingAverageTime temporalAATime;

            cudaFrameTime.append(curGPUTimer.frame.report());
            updateTime.append(curGPUTimer.update.report());
            computePDFTextureTime.append(curGPUTimer.computePDFTexture.report());
            setupGBuffersTime.append(curGPUTimer.setupGBuffers.report());
            pathTraceTime.append(curGPUTimer.pathTrace.report());
            denoiseTime.append(curGPUTimer.denoise.report());
            estimateVarianceTime.append(curGPUTimer.estimateVariance.report());
            aTrousFilterTime.append(curGPUTimer.aTrousFilter.report());
            temporalAATime.append(curGPUTimer.temporalAA.report());

            //ImGui::SetNextItemWidth(100.0f);
            ImGui::Text("CUDA/OptiX GPU %.3f [ms]:", cudaFrameTime.getAverage());
            ImGui::Text("  Update: %.3f [ms]", updateTime.getAverage());
            ImGui::Text("  Compute PDF Texture: %.3f [ms]", computePDFTextureTime.getAverage());
            ImGui::Text("  Setup G-Buffers: %.3f [ms]", setupGBuffersTime.getAverage());
            ImGui::Text("  Path Trace: %.3f [ms]", pathTraceTime.getAverage());
            ImGui::Text("  Denoise: %.3f [ms]", denoiseTime.getAverage());
            ImGui::Text("    Estimate Variance: %.3f [ms]", estimateVarianceTime.getAverage());
            ImGui::Text("    A-Trous Filter: %.3f [ms]", aTrousFilterTime.getAverage());
            ImGui::Text("  Temporal AA: %.3f [ms]", temporalAATime.getAverage());

            ImGui::Text("%u [spp]", std::min(numAccumFrames + 1, (1u << log2MaxNumAccums)));

            ImGui::End();
        }

        applyToneMapAndGammaCorrection =
            bufferTypeToDisplay == shared::BufferToDisplay::NoisyBeauty ||
            bufferTypeToDisplay == shared::BufferToDisplay::FinalRendering ||
            bufferTypeToDisplay == shared::BufferToDisplay::Variance ||
            bufferTypeToDisplay == shared::BufferToDisplay::FilteredVariance;



        curGPUTimer.frame.start(cuStream);

        // JP: 各インスタンスのトランスフォームを更新する。
        // EN: Update the transform of each instance.
        if (animate || lastFrameWasAnimated) {
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
            scene.setupLightInstDistribtion(cuStream, probTexAddr, curBufIdx);
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
        perFramePlp.instanceDataBuffer = curInstDataBuffer.getDevicePointer();
        perFramePlp.envLightPowerCoeff = std::pow(10.0f, log10EnvLightPowerCoeff);
        perFramePlp.envLightRotation = envLightRotation;
        perFramePlp.mousePosition = int2(static_cast<int32_t>(g_mouseX),
                                         static_cast<int32_t>(g_mouseY));
        perFramePlp.pickInfo = curPickInfo.getDevicePointer();

        perFramePlp.taaHistoryLength = static_cast<uint32_t>(std::round(std::pow(2, log2TaaHistoryLength)));
        perFramePlp.maxPathLength = maxPathLength;
        perFramePlp.bufferIndex = curBufIdx;
        perFramePlp.enableEnvLight = enableEnvLight;
        perFramePlp.enableBumpMapping = true;
        perFramePlp.isFirstFrame = newSequence;
        perFramePlp.enableTemporalAccumulation = enableTemporalAccumulation;
        perFramePlp.enableSVGF = enableSVGF;
        perFramePlp.feedback1stFilteredResult = feedback1stFilteredResult;
        perFramePlp.mollifySpecular = specularMollification;
        perFramePlp.enableTemporalAA = enableTemporalAA;
        perFramePlp.modulateAlbedo = modulateAlbedo;
        for (int i = 0; i < lengthof(debugSwitches); ++i)
            perFramePlp.setDebugSwitch(i, debugSwitches[i]);

        curMatW2V = inverse(Matrix4x4((g_tempCameraOrientation * qRotateY(M_PI)).toMatrix3x3(), g_cameraPosition));

        float2 curSubPixelOffset = { 0.5f, 0.5f };
        if (enableTemporalAA)
            curSubPixelOffset = subPixelOffsets[frameIndex % perFramePlp.taaHistoryLength];
        curPerFrameTemporalSet.camera.subPixelOffset = curSubPixelOffset;

        // JP: Gバッファーのセットアップ。
        // EN: Setup the G-buffers.
        {
            const auto addOffset = []
            (const Matrix4x4 &mat, float offsetX, float offsetY,
             uint32_t windowSizeX, uint32_t windowSizeY) {
                 Matrix4x4 ret = mat;
                 ret[2].x = (2 * offsetX - 1) / windowSizeX;
                 ret[2].y = (2 * offsetY - 1) / windowSizeY;
                 return ret;
            };

            Matrix4x4 prevMatW2C = prevMatV2C * prevMatW2V;
            Matrix4x4 curMatW2C = curMatV2C * curMatW2V;
            Matrix4x4 curMatW2CWithOffset = addOffset(
                curMatV2C, curSubPixelOffset.x, curSubPixelOffset.y,
                renderTargetSizeX, renderTargetSizeY) * curMatW2V;

            glBindFramebuffer(GL_FRAMEBUFFER, curTemporalSet.gfxGBuffers.getHandle(0));
            curTemporalSet.gfxGBuffers.setDrawBuffers();
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);

            glViewport(0, 0, curFBWidth, curFBHeight);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClearDepth(1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glUseProgram(gBufferShader.getHandle());

            glUniform2ui(0, curFBWidth, curFBHeight);
            glUniform3fv(1, 1, &curPerFrameTemporalSet.camera.position.x);
            glUniformMatrix4fv(2, 1, false, reinterpret_cast<float*>(&prevMatW2C));
            glUniformMatrix4fv(3, 1, false, reinterpret_cast<float*>(&curMatW2C));
            glUniformMatrix4fv(4, 1, false, reinterpret_cast<float*>(&curMatW2CWithOffset));

            scene.draw();

            glDisable(GL_DEPTH_TEST);
            curTemporalSet.gfxGBuffers.resetDrawBuffers();
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        // JP: デプスバッファーのコピー。
        // EN: Copy the depth buffer.
        {
            glBindFramebuffer(GL_FRAMEBUFFER, curTemporalSet.gfxDepthBufferCopy.getHandle(0));
            curTemporalSet.gfxDepthBufferCopy.setDrawBuffers();
            glDisable(GL_DEPTH_TEST);

            glViewport(0, 0, curFBWidth, curFBHeight);

            glUseProgram(depthCopyShader.getHandle());

            glBindTextureUnit(0, curTemporalSet.gfxGBuffers.getDepthRenderTargetTexture(0).getHandle());

            glBindVertexArray(vertexArrayForFullScreen.getHandle());
            glDrawArrays(GL_TRIANGLES, 0, 3);

            curTemporalSet.gfxDepthBufferCopy.resetDrawBuffers();
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        prevTemporalSet.gBuffer0InteropHandler.beginCUDAAccess(cuStream);
        prevTemporalSet.gBuffer1InteropHandler.beginCUDAAccess(cuStream);
        prevTemporalSet.gBuffer2InteropHandler.beginCUDAAccess(cuStream);
        prevTemporalSet.depthBufferInteropHandler.beginCUDAAccess(cuStream);
        prevTemporalSet.finalLightingBufferInteropHandler.beginCUDAAccess(cuStream);

        curTemporalSet.gBuffer0InteropHandler.beginCUDAAccess(cuStream);
        curTemporalSet.gBuffer1InteropHandler.beginCUDAAccess(cuStream);
        curTemporalSet.gBuffer2InteropHandler.beginCUDAAccess(cuStream);
        curTemporalSet.depthBufferInteropHandler.beginCUDAAccess(cuStream);
        curTemporalSet.finalLightingBufferInteropHandler.beginCUDAAccess(cuStream);

        debugVisualizeBufferInteropHandler.beginCUDAAccess(cuStream);

        prevPerFrameTemporalSet.GBuffer0 = prevTemporalSet.gBuffer0InteropHandler.getNext();
        prevPerFrameTemporalSet.GBuffer1 = prevTemporalSet.gBuffer1InteropHandler.getNext();
        prevPerFrameTemporalSet.GBuffer2 = prevTemporalSet.gBuffer2InteropHandler.getNext();
        prevPerFrameTemporalSet.depthBuffer = prevTemporalSet.depthBufferInteropHandler.getNext();
        prevPerFrameTemporalSet.finalLightingBuffer = prevTemporalSet.finalLightingBufferInteropHandler.getNext();

        curPerFrameTemporalSet.GBuffer0 = curTemporalSet.gBuffer0InteropHandler.getNext();
        curPerFrameTemporalSet.GBuffer1 = curTemporalSet.gBuffer1InteropHandler.getNext();
        curPerFrameTemporalSet.GBuffer2 = curTemporalSet.gBuffer2InteropHandler.getNext();
        curPerFrameTemporalSet.depthBuffer = curTemporalSet.depthBufferInteropHandler.getNext();
        curPerFrameTemporalSet.finalLightingBuffer = curTemporalSet.finalLightingBufferInteropHandler.getNext();

        perFramePlp.debugVisualizeBuffer = debugVisualizeBufferInteropHandler.getNext();

        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream)); // for OptiX
        // for pure CUDA
        CUDADRV_CHECK(cuMemcpyHtoDAsync(gpuEnv.plpPtrForSvgfModule, &plp, sizeof(plp), cuStream));
        CUDADRV_CHECK(cuMemcpyHtoDAsync(gpuEnv.plpPtrForDebugVisualizeModule, &plp, sizeof(plp), cuStream));

        CUDADRV_CHECK(cuMemcpyHtoDAsync(perFramePlpOnDevice, &perFramePlp, sizeof(perFramePlp), cuStream));

        // JP: パストレーシングによるシェーディングを実行。
        // EN: Perform shading by path tracing.
        curGPUTimer.pathTrace.start(cuStream);
        PathTracingEntryPoint ptEntryPoint = enableTemporalAccumulation ?
            PathTracingEntryPoint::pathTraceWithTemporalAccumulation :
            PathTracingEntryPoint::pathTraceWithoutTemporalAccumulation;
        gpuEnv.pathTracing.setEntryPoint(ptEntryPoint);
        gpuEnv.pathTracing.optixPipeline.launch(
            cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
        curGPUTimer.pathTrace.stop(cuStream);

        constexpr uint32_t numFilteringStages = 5;
        if (enableSVGF) {
            curGPUTimer.denoise.start(cuStream);

            // JP: ピクセルごとの輝度の分散を求める。
            // EN: Compute the variance of the luminance for each pixel.
            curGPUTimer.estimateVariance.start(cuStream);
            gpuEnv.kernelEstimateVariance(
                cuStream, gpuEnv.kernelEstimateVariance.calcGridDim(renderTargetSizeX, renderTargetSizeY));
            curGPUTimer.estimateVariance.stop(cuStream);

            if (bufferTypeToDisplay == shared::BufferToDisplay::NoisyBeauty ||
                bufferTypeToDisplay == shared::BufferToDisplay::Variance) {
                gpuEnv.kernelDebugVisualize(
                    cuStream, gpuEnv.kernelDebugVisualize.calcGridDim(renderTargetSizeX, renderTargetSizeY),
                    bufferTypeToDisplay,
                    0.5f, std::pow(10.0f, motionVectorScale),
                    numFilteringStages);
                CUDADRV_CHECK(cuStreamSynchronize(cuStream));
            }

            // JP: A-Trousフィルターをライティングと分散に複数回適用する。
            // EN: Apply the a-trous filter to lighting and its variance multiple times.
            curGPUTimer.aTrousFilter.start(cuStream);
            for (uint32_t filterStageIndex = 0; filterStageIndex < numFilteringStages; ++filterStageIndex) {
                gpuEnv.kernelApplyATrousFilter_box3x3(
                    cuStream, gpuEnv.kernelApplyATrousFilter_box3x3.calcGridDim(renderTargetSizeX, renderTargetSizeY),
                    filterStageIndex);
            }
            curGPUTimer.aTrousFilter.stop(cuStream);

            curGPUTimer.denoise.stop(cuStream);
        }
        else {
            if (enableTemporalAccumulation) {
                gpuEnv.kernelFeedbackNoisyLighting(
                    cuStream, gpuEnv.kernelFeedbackNoisyLighting.calcGridDim(renderTargetSizeX, renderTargetSizeY));
            }
        }

        curGPUTimer.temporalAA.start(cuStream);
        gpuEnv.kernelApplyAlbedoModulationAndTemporalAntiAliasing(
            cuStream, gpuEnv.kernelApplyAlbedoModulationAndTemporalAntiAliasing.calcGridDim(renderTargetSizeX, renderTargetSizeY),
            numFilteringStages);
        curGPUTimer.temporalAA.stop(cuStream);

        if (bufferTypeToDisplay == shared::BufferToDisplay::Albedo ||
            bufferTypeToDisplay == shared::BufferToDisplay::FilteredVariance ||
            bufferTypeToDisplay == shared::BufferToDisplay::Normal ||
            bufferTypeToDisplay == shared::BufferToDisplay::MotionVector ||
            bufferTypeToDisplay == shared::BufferToDisplay::SampleCount) {
            gpuEnv.kernelDebugVisualize(
                cuStream, gpuEnv.kernelDebugVisualize.calcGridDim(renderTargetSizeX, renderTargetSizeY),
                bufferTypeToDisplay,
                0.5f, std::pow(10.0f, motionVectorScale),
                numFilteringStages);
            CUDADRV_CHECK(cuStreamSynchronize(cuStream));
        }

        debugVisualizeBufferInteropHandler.endCUDAAccess(cuStream, true);

        curTemporalSet.finalLightingBufferInteropHandler.endCUDAAccess(cuStream, true);
        curTemporalSet.depthBufferInteropHandler.endCUDAAccess(cuStream, true);
        curTemporalSet.gBuffer2InteropHandler.endCUDAAccess(cuStream, true);
        curTemporalSet.gBuffer1InteropHandler.endCUDAAccess(cuStream, true);
        curTemporalSet.gBuffer0InteropHandler.endCUDAAccess(cuStream, true);

        prevTemporalSet.finalLightingBufferInteropHandler.endCUDAAccess(cuStream, true);
        prevTemporalSet.depthBufferInteropHandler.endCUDAAccess(cuStream, true);
        prevTemporalSet.gBuffer2InteropHandler.endCUDAAccess(cuStream, true);
        prevTemporalSet.gBuffer1InteropHandler.endCUDAAccess(cuStream, true);
        prevTemporalSet.gBuffer0InteropHandler.endCUDAAccess(cuStream, true);

        curGPUTimer.frame.stop(cuStream);

        // ----------------------------------------------------------------
        // JP: 最終レンダーターゲットに結果を描画する。
        // EN: Draw the result to the final render target.

        if (applyToneMapAndGammaCorrection) {
            glEnable(GL_FRAMEBUFFER_SRGB);
            ImGui::GetStyle() = guiStyleWithGamma;
        }
        else {
            glDisable(GL_FRAMEBUFFER_SRGB);
            ImGui::GetStyle() = guiStyle;
        }

        glViewport(0, 0, curFBWidth, curFBHeight);

        glUseProgram(drawResultShader.getHandle());

        glu::Texture2D &texToDisplay = bufferTypeToDisplay == shared::BufferToDisplay::FinalRendering ?
            curTemporalSet.gfxFinalLightingBuffer : gfxDebugVisualizeBuffer;
        glBindTextureUnit(0, texToDisplay.getHandle());
        glBindSampler(0, outputSampler.getHandle());
        glUniform1f(1, std::pow(10.0f, brightness));
        uint32_t flags =
            (applyToneMapAndGammaCorrection ? 1 : 0);
        glUniform1ui(2, flags);

        glBindVertexArray(vertexArrayForFullScreen.getHandle());
        glDrawArrays(GL_TRIANGLES, 0, 3);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glDisable(GL_FRAMEBUFFER_SRGB);

        // END: Draw the result to the final render target.
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

    vertexArrayForFullScreen.finalize();
    outputSampler.finalize();
    drawResultShader.finalize();
    depthCopyShader.finalize();
    gBufferShader.finalize();


    
    finalizeResolutionDependentBuffers();



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
