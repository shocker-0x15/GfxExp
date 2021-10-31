/*

コマンドラインオプション例 / Command line option example:
You can load a 3D model for example by downloading from the internet.
(e.g. https://casual-effects.com/data/)

(1) -cam-pos -0.753442 0.140257 -0.056083 -cam-yaw 75
    -name exterior -obj Amazon_Bistro/Exterior/exterior.obj 0.001 trad -brightness 2.0
    -name rectlight -emittance 5 5 5 -rectangle 0.1 0.1
    -inst exterior
    -begin-pos 0.362 0.329 -2.0 -begin-pitch -90 -begin-yaw 150
    -end-pos -0.719 0.329 -0.442 -end-pitch -90 -end-yaw 30 -inst rectlight

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

JP: このプログラムはReSTIR (Reservoir-based Spatio-Temporal Importance Resampling) [1]の実装例です。
    ReSTIRでは、Resampled Importance Sampling (RIS), Weighted Reservoir Sampling (WRS)、
    そして複数のReservoirを結合する際の特性を利用することで大量の発光プリミティブからの効率的なサンプリングが可能となります。

EN: This program is an example implementation of ReSTIR (Reservoir-based Spatio-Temporal Importance Resampling) [1].
    ReSTIR enables efficient sampling from a massive amount of emitter primitives by
    Resampled Importance Sampling (RIS), Weighted Reservoir Sampling (WRS) and exploiting the property of
    combining multiple reservoirs.

[1] Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting
    https://research.nvidia.com/publication/2020-07_Spatiotemporal-reservoir-resampling

*/

#include "restir_scene.h"

// Include glfw3.h after our OpenGL definitions
#include "../utils/gl_util.h"
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"



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
                *ori = qRotateZ(atof(argv[i + 1]) * M_PI / 180) * *ori;
                i += 1;
            }
            else if (strncmp(arg, "-pitch", 7) == 0) {
                if (i + 1 >= argc) {
                    hpprintf("Invalid option.\n");
                    exit(EXIT_FAILURE);
                }
                *ori = qRotateX(atof(argv[i + 1]) * M_PI / 180) * *ori;
                i += 1;
            }
            else if (strncmp(arg, "-yaw", 5) == 0) {
                if (i + 1 >= argc) {
                    hpprintf("Invalid option.\n");
                    exit(EXIT_FAILURE);
                }
                *ori = qRotateY(atof(argv[i + 1]) * M_PI / 180) * *ori;
                i += 1;
            }
        };

        if (strncmp(arg, "-", 1) != 0)
            continue;

        if (strncmp(arg, "-screenshot", 12) == 0) {
            g_takeScreenShot = true;
            i += 1;
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
                                          "ReSTIR", NULL, NULL);
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



    // ----------------------------------------------------------------
    // JP: 入力コールバックの設定。
    // EN: Set up input callbacks.

    glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
        uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

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
    glfwSetCursorPosCallback(window, [](GLFWwindow* window, double x, double y) {
        g_mouseX = x;
        g_mouseY = y;
                             });
    glfwSetKeyCallback(window, [](GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
        uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);
        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);

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



    GPUEnvironment gpuEnv;
    gpuEnv.initialize();

    CUstream cuStream;
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    gpuEnv.materialDataBuffer.map();
    gpuEnv.geomInstDataBuffer.map();
    gpuEnv.instDataBuffer[0].map();

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
            float t = 0.5f - 0.5f * std::cos(2 * M_PI * time / frequency);
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

    std::vector<Material*> materials;
    std::vector<GeometryInstance*> geomInsts;
    std::vector<GeometryGroup*> geomGroups;

    std::map<std::string, Mesh*> meshes;
    for (auto it = g_meshInfos.cbegin(); it != g_meshInfos.cend(); ++it) {
        const MeshInfo &info = it->second;

        std::vector<Material*> tempMaterials;
        std::vector<GeometryInstance*> tempGeomInsts;
        std::vector<GeometryGroup*> tempGeomGroups;
        Mesh* mesh = new Mesh();

        if (std::holds_alternative<MeshGeometryInfo>(info)) {
            const auto &meshInfo = std::get<MeshGeometryInfo>(info);

            createTriangleMeshes(meshInfo.path, meshInfo.matConv,
                                 scale4x4(meshInfo.preScale),
                                 gpuEnv,
                                 tempMaterials,
                                 tempGeomInsts,
                                 tempGeomGroups,
                                 mesh);
        }
        else if (std::holds_alternative<RectangleGeometryInfo>(info)) {
            const auto &rectInfo = std::get<RectangleGeometryInfo>(info);

            tempMaterials.push_back(new Material());
            tempGeomInsts.push_back(new GeometryInstance());
            tempGeomGroups.push_back(new GeometryGroup());
            createRectangleLight(rectInfo.dimX, rectInfo.dimZ,
                                 float3(0.01f),
                                 rectInfo.emitterTexPath, rectInfo.emittance, Matrix4x4(),
                                 gpuEnv, &tempMaterials[0], &tempGeomInsts[0], &tempGeomGroups[0],
                                 mesh);
        }

        materials.insert(materials.end(), tempMaterials.begin(), tempMaterials.end());
        geomInsts.insert(geomInsts.end(), tempGeomInsts.begin(), tempGeomInsts.end());
        geomGroups.insert(geomGroups.end(), tempGeomGroups.begin(), tempGeomGroups.end());
        meshes[it->first] = mesh;
    }

    std::vector<Instance*> insts;
    std::vector<InstanceController*> instControllers;
    AABB initialSceneAabb;
    for (int i = 0; i < g_meshInstInfos.size(); ++i) {
        const MeshInstanceInfo &info = g_meshInstInfos[i];
        const Mesh* mesh = meshes.at(info.name);
        for (int j = 0; j < mesh->groups.size(); ++j) {
            const Mesh::Group &group = mesh->groups[j];

            Matrix4x4 instXfm = Matrix4x4(info.beginScale * info.beginOrientation.toMatrix3x3(), info.beginPosition) * group.transform;
            Instance* inst = createInstance(gpuEnv, group.geomGroup, instXfm);
            insts.push_back(inst);

            initialSceneAabb.unify(instXfm * group.geomGroup->aabb);

            if (info.beginPosition != info.endPosition ||
                info.beginOrientation != info.endOrientation ||
                info.beginScale != info.endScale) {
                // TODO: group.transformを追加のtransformにする？
                auto controller = new InstanceController(
                    inst, group.transform,
                    info.beginScale, info.beginOrientation, info.beginPosition,
                    info.endScale, info.endOrientation, info.endPosition,
                    info.frequency, info.initTime);
                instControllers.push_back(controller);
            }
        }
    }

    float3 sceneDim = initialSceneAabb.maxP - initialSceneAabb.minP;
    g_cameraPositionalMovingSpeed = 0.003f * std::max({ sceneDim.x, sceneDim.y, sceneDim.z });
    g_cameraDirectionalMovingSpeed = 0.0015f;
    g_cameraTiltSpeed = 0.025f;

    gpuEnv.instDataBuffer[0].unmap();
    gpuEnv.geomInstDataBuffer.unmap();
    gpuEnv.materialDataBuffer.unmap();

    CUDADRV_CHECK(cuMemcpyDtoD(gpuEnv.instDataBuffer[1].getCUdeviceptr(),
                               gpuEnv.instDataBuffer[0].getCUdeviceptr(),
                               gpuEnv.instDataBuffer[1].sizeInBytes()));



    optixu::InstanceAccelerationStructure ias = gpuEnv.scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> iasInstanceBuffer;
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
        geomGroup->optixGasMem.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, asSizes.outputSizeInBytes, 1);
        asScratchSize = std::max(asSizes.tempSizeInBytes, asScratchSize);
    }

    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false, false);
    ias.prepareForBuild(&asSizes);
    iasMem.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, asSizes.outputSizeInBytes, 1);
    asScratchSize = std::max(asSizes.tempSizeInBytes, asScratchSize);
    iasInstanceBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, ias.getNumChildren());

    cudau::Buffer asScratchMem;
    asScratchMem.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, asScratchSize, 1);

    for (int i = 0; i < geomGroups.size(); ++i) {
        const GeometryGroup* geomGroup = geomGroups[i];
        geomGroup->optixGas.rebuild(cuStream, geomGroup->optixGasMem, asScratchMem);
    }

    cudau::Buffer hitGroupSBT;
    size_t hitGroupSbtSize;
    gpuEnv.scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    hitGroupSBT.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, hitGroupSbtSize, 1);
    hitGroupSBT.setMappedMemoryPersistent(true);

    {
        for (int bufIdx = 0; bufIdx < 2; ++bufIdx) {
            cudau::TypedBuffer<shared::InstanceData> &curInstDataBuffer = gpuEnv.instDataBuffer[bufIdx];
            shared::InstanceData* instDataBufferOnHost = curInstDataBuffer.map();
            for (int i = 0; i < instControllers.size(); ++i) {
                InstanceController* controller = instControllers[i];
                Instance* inst = controller->inst;
                shared::InstanceData &instData = instDataBufferOnHost[inst->instSlot];
                controller->update(instDataBufferOnHost, 0.0f);
                // TODO: まとめて送る。
                CUDADRV_CHECK(cuMemcpyHtoDAsync(curInstDataBuffer.getCUdeviceptrAt(inst->instSlot),
                                                &instData, sizeof(instData), cuStream));
            }
            curInstDataBuffer.unmap();
        }
    }

    OptixTraversableHandle travHandle = ias.rebuild(cuStream, iasInstanceBuffer, iasMem, asScratchMem);

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    // JP: 各インスタンスの重要度から光源インスタンスをサンプルするための分布を計算する。
    // EN: Compute a distribution to sample a light source instance from the importance value of each instance.
    uint32_t totalNumEmitterPrimitives = 0;
    std::vector<float> lightImportances(insts.size());
    for (int i = 0; i < insts.size(); ++i) {
        const Instance* inst = insts[i];
        lightImportances[i] = inst->lightGeomInstDist.getIntengral();
        totalNumEmitterPrimitives += inst->geomGroup->numEmitterPrimitives;
    }
    DiscreteDistribution1D lightInstDist;
    lightInstDist.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                             lightImportances.data(), lightImportances.size());
    Assert(lightInstDist.getIntengral() > 0, "No lights!");
    hpprintf("%u emitter primitives\n", totalNumEmitterPrimitives);

    // JP: 環境光テクスチャーを読み込んで、サンプルするためのCDFを計算する。
    // EN: Read a environmental texture, then compute a CDF to sample it.
    cudau::Array envLightArray;
    CUtexObject envLightTexture = 0;
    RegularConstantContinuousDistribution2D envLightImportanceMap;
    if (!g_envLightTexturePath.empty())
        loadEnvironmentalTexture(g_envLightTexturePath, gpuEnv,
                                 &envLightArray, &envLightTexture, &envLightImportanceMap);

    // END: Setup a scene.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: 
    
    cudau::TypedBuffer<shared::PCG32RNG> lightPreSamplingRngs;
    cudau::TypedBuffer<shared::PreSampledLight> preSampledLights;

    constexpr uint32_t numPreSampledLights = shared::numLightSubsets * shared::lightSubsetSize;
    lightPreSamplingRngs.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, numPreSampledLights);
    {
        shared::PCG32RNG* rngs = lightPreSamplingRngs.map();
        std::mt19937_64 rngSeed(894213312210);
        for (int i = 0; i < numPreSampledLights; ++i)
            rngs[i].setState(rngSeed());
        lightPreSamplingRngs.unmap();
    }
    preSampledLights.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, numPreSampledLights);

    // END: 
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: スクリーン関連のバッファーを初期化。
    // EN: Initialize screen-related buffers.

    cudau::Array gBuffer0[2];
    cudau::Array gBuffer1[2];
    cudau::Array gBuffer2[2];

    optixu::HostBlockBuffer2D<shared::Reservoir<shared::LightSample>, 1> reservoirBuffer[2];
    cudau::Array reservoirInfoBuffer[2];
    cudau::Array sampleVisibilityBuffer[2];
    
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

            reservoirBuffer[i].initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                          renderTargetSizeX, renderTargetSizeY);
            reservoirInfoBuffer[i].initialize2D(
                gpuEnv.cuContext, cudau::ArrayElementType::UInt32, (sizeof(shared::ReservoirInfo) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                renderTargetSizeX, renderTargetSizeY, 1);

            sampleVisibilityBuffer[i].initialize2D(
                gpuEnv.cuContext, cudau::ArrayElementType::UInt32, (sizeof(shared::SampleVisibility) + 3) / 4,
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

        linearBeautyBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                      renderTargetSizeX * renderTargetSizeY);
        linearAlbedoBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                      renderTargetSizeX * renderTargetSizeY);
        linearNormalBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                      renderTargetSizeX * renderTargetSizeY);
        linearFlowBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                    renderTargetSizeX * renderTargetSizeY);
        linearDenoisedBeautyBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
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
            sampleVisibilityBuffer[i].finalize();

            reservoirInfoBuffer[i].finalize();
            reservoirBuffer[i].finalize();

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

            reservoirBuffer[i].resize(renderTargetSizeX, renderTargetSizeY);
            reservoirInfoBuffer[i].resize(renderTargetSizeX, renderTargetSizeY);

            sampleVisibilityBuffer[i].resize(renderTargetSizeX, renderTargetSizeY);
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
    optixu::Denoiser denoiser = gpuEnv.optixContext.createDenoiser(OPTIX_DENOISER_MODEL_KIND_TEMPORAL, true, true);
    size_t stateSize;
    size_t scratchSize;
    size_t scratchSizeForComputeIntensity;
    uint32_t numTasks;
    denoiser.prepare(renderTargetSizeX, renderTargetSizeY, tileWidth, tileHeight,
                     &stateSize, &scratchSize, &scratchSizeForComputeIntensity,
                     &numTasks);
    hpprintf("Denoiser State Buffer: %llu bytes\n", stateSize);
    hpprintf("Denoiser Scratch Buffer: %llu bytes\n", scratchSize);
    hpprintf("Compute Intensity Scratch Buffer: %llu bytes\n", scratchSizeForComputeIntensity);
    cudau::Buffer denoiserStateBuffer;
    cudau::Buffer denoiserScratchBuffer;
    denoiserStateBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, stateSize, 1);
    denoiserScratchBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                     std::max(scratchSize, scratchSizeForComputeIntensity), 1);

    std::vector<optixu::DenoisingTask> denoisingTasks(numTasks);
    denoiser.getTasks(denoisingTasks.data());

    denoiser.setupState(cuStream, denoiserStateBuffer, denoiserScratchBuffer);

    // JP: デノイザーは入出力にリニアなバッファーを必要とするため結果をコピーする必要がある。
    // EN: Denoiser requires linear buffers as input/output, so we need to copy the results.
    CUmodule moduleCopyBuffers;
    CUDADRV_CHECK(cuModuleLoad(&moduleCopyBuffers, (getExecutableDirectory() / "restir/ptxes/copy_buffers.ptx").string().c_str()));
    cudau::Kernel kernelCopyToLinearBuffers(moduleCopyBuffers, "copyToLinearBuffers", cudau::dim3(8, 8), 0);
    cudau::Kernel kernelVisualizeToOutputBuffer(moduleCopyBuffers, "visualizeToOutputBuffer", cudau::dim3(8, 8), 0);

    CUdeviceptr hdrIntensity;
    CUDADRV_CHECK(cuMemAlloc(&hdrIntensity, sizeof(float)));

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
    outputBufferSurfaceHolder.initialize(&outputArray);

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
    drawOptiXResultShader.initializeVSPS(readTxtFile(exeDir / "restir/shaders/drawOptiXResult.vert"),
                                         readTxtFile(exeDir / "restir/shaders/drawOptiXResult.frag"));



    // JP: Spatial Reuseで使用する近傍ピクセルへの方向をLow-discrepancy数列から作成しておく。
    // EN: Generate directions to neighboring pixels used in spatial reuse from a low-discrepancy sequence.
    const auto computeHaltonSequence = [](uint32_t base, uint32_t idx) {
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
    const auto concentricSampleDisk = [](float u0, float u1, float* dx, float* dy) {
        float r, theta;
        float sx = 2 * u0 - 1;
        float sy = 2 * u1 - 1;

        if (sx == 0 && sy == 0) {
            *dx = 0;
            *dy = 0;
            return;
        }
        if (sx >= -sy) { // region 1 or 2
            if (sx > sy) { // region 1
                r = sx;
                theta = sy / sx;
            }
            else { // region 2
                r = sy;
                theta = 2 - sx / sy;
            }
        }
        else { // region 3 or 4
            if (sx > sy) {/// region 4
                r = -sy;
                theta = 6 + sx / sy;
            }
            else {// region 3
                r = -sx;
                theta = 4 + sy / sx;
            }
        }
        theta *= M_PI / 4;
        *dx = r * cos(theta);
        *dy = r * sin(theta);
    };
    std::vector<float2> spatialNeighborDeltasOnHost(1024);
    for (int i = 0; i < spatialNeighborDeltasOnHost.size(); ++i) {
        float2 delta;
        concentricSampleDisk(computeHaltonSequence(2, i), computeHaltonSequence(3, i), &delta.x, &delta.y);
        spatialNeighborDeltasOnHost[i] = delta;
        //printf("%g, %g\n", delta.x, delta.y);
    }
    cudau::TypedBuffer<float2> spatialNeighborDeltas(gpuEnv.cuContext, GPUEnvironment::bufferType, spatialNeighborDeltasOnHost);



    shared::StaticPipelineLaunchParameters staticPlp = {};
    staticPlp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
    staticPlp.numTiles = int2((renderTargetSizeX + shared::tileSizeX - 1) / shared::tileSizeX,
                              (renderTargetSizeY + shared::tileSizeY - 1) / shared::tileSizeY);
    staticPlp.lightPreSamplingRngs = lightPreSamplingRngs.getDevicePointer();
    staticPlp.preSampledLights = preSampledLights.getDevicePointer();
    staticPlp.rngBuffer = rngBuffer.getSurfaceObject(0);
    staticPlp.GBuffer0[0] = gBuffer0[0].getSurfaceObject(0);
    staticPlp.GBuffer0[1] = gBuffer0[1].getSurfaceObject(0);
    staticPlp.GBuffer1[0] = gBuffer1[0].getSurfaceObject(0);
    staticPlp.GBuffer1[1] = gBuffer1[1].getSurfaceObject(0);
    staticPlp.GBuffer2[0] = gBuffer2[0].getSurfaceObject(0);
    staticPlp.GBuffer2[1] = gBuffer2[1].getSurfaceObject(0);
    staticPlp.reservoirBuffer[0] = reservoirBuffer[0].getBlockBuffer2D();
    staticPlp.reservoirBuffer[1] = reservoirBuffer[1].getBlockBuffer2D();
    staticPlp.reservoirInfoBuffer[0] = reservoirInfoBuffer[0].getSurfaceObject(0);
    staticPlp.reservoirInfoBuffer[1] = reservoirInfoBuffer[1].getSurfaceObject(0);
    staticPlp.sampleVisibilityBuffer[0] = sampleVisibilityBuffer[0].getSurfaceObject(0);
    staticPlp.sampleVisibilityBuffer[1] = sampleVisibilityBuffer[1].getSurfaceObject(0);
    staticPlp.spatialNeighborDeltas = spatialNeighborDeltas.getDevicePointer();
    staticPlp.materialDataBuffer = gpuEnv.materialDataBuffer.getDevicePointer();
    staticPlp.geometryInstanceDataBuffer = gpuEnv.geomInstDataBuffer.getDevicePointer();
    lightInstDist.getDeviceType(&staticPlp.lightInstDist);
    envLightImportanceMap.getDeviceType(&staticPlp.envLightImportanceMap);
    staticPlp.envLightTexture = envLightTexture;
    staticPlp.beautyAccumBuffer = beautyAccumBuffer.getSurfaceObject(0);
    staticPlp.albedoAccumBuffer = albedoAccumBuffer.getSurfaceObject(0);
    staticPlp.normalAccumBuffer = normalAccumBuffer.getSurfaceObject(0);

    CUdeviceptr staticPlpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&staticPlpOnDevice, sizeof(staticPlp)));
    CUDADRV_CHECK(cuMemcpyHtoD(staticPlpOnDevice, &staticPlp, sizeof(staticPlp)));

    shared::PerFramePipelineLaunchParameters perFramePlp = {};
    perFramePlp.travHandle = travHandle;
    perFramePlp.camera.fovY = 50 * M_PI / 180;
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
    plp.currentReservoirIndex = 0;

    gpuEnv.pipeline.setScene(gpuEnv.scene);
    gpuEnv.pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

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
    pickInfos[0].initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, 1, initPickInfo);
    pickInfos[1].initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, 1, initPickInfo);

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    struct GPUTimer {
        cudau::Timer frame;
        cudau::Timer update;
        cudau::Timer setupGBuffers;
        cudau::Timer performInitialAndTemporalRIS;
        cudau::Timer performSpatialRIS;
        cudau::Timer shading;
        cudau::Timer performPreSamplingLights;
        cudau::Timer performPerPixelRIS;
        cudau::Timer traceShadowRays;
        cudau::Timer shadeAndResample;
        cudau::Timer denoise;

        void initialize(CUcontext context) {
            frame.initialize(context);
            update.initialize(context);
            setupGBuffers.initialize(context);

            performInitialAndTemporalRIS.initialize(context);
            performSpatialRIS.initialize(context);
            shading.initialize(context);

            performPerPixelRIS.initialize(context);
            performPreSamplingLights.initialize(context);
            traceShadowRays.initialize(context);
            shadeAndResample.initialize(context);

            denoise.initialize(context);
        }
        void finalize() {
            denoise.finalize();

            shadeAndResample.finalize();
            traceShadowRays.finalize();
            performPreSamplingLights.finalize();

            shading.finalize();
            performSpatialRIS.finalize();
            performInitialAndTemporalRIS.finalize();
            performPerPixelRIS.finalize();

            setupGBuffers.finalize();
            update.finalize();
            frame.finalize();
        }
    };

    uint32_t lastSpatialNeighborBaseIndex = 0;
    uint32_t lastReservoirIndex = 1;
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
                size_t stateSize;
                size_t scratchSize;
                size_t scratchSizeForComputeIntensity;
                uint32_t numTasks;
                denoiser.prepare(renderTargetSizeX, renderTargetSizeY, tileWidth, tileHeight,
                                 &stateSize, &scratchSize, &scratchSizeForComputeIntensity,
                                 &numTasks);
                hpprintf("Denoiser State Buffer: %llu bytes\n", stateSize);
                hpprintf("Denoiser Scratch Buffer: %llu bytes\n", scratchSize);
                hpprintf("Compute Intensity Scratch Buffer: %llu bytes\n", scratchSizeForComputeIntensity);
                denoiserStateBuffer.resize(stateSize, 1);
                denoiserScratchBuffer.resize(std::max(scratchSize, scratchSizeForComputeIntensity), 1);

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
            staticPlp.numTiles = int2((renderTargetSizeX + shared::tileSizeX - 1) / shared::tileSizeX,
                                      (renderTargetSizeY + shared::tileSizeY - 1) / shared::tileSizeY);
            staticPlp.rngBuffer = rngBuffer.getSurfaceObject(0);
            staticPlp.GBuffer0[0] = gBuffer0[0].getSurfaceObject(0);
            staticPlp.GBuffer0[1] = gBuffer0[1].getSurfaceObject(0);
            staticPlp.GBuffer1[0] = gBuffer1[0].getSurfaceObject(0);
            staticPlp.GBuffer1[1] = gBuffer1[1].getSurfaceObject(0);
            staticPlp.GBuffer2[0] = gBuffer2[0].getSurfaceObject(0);
            staticPlp.GBuffer2[1] = gBuffer2[1].getSurfaceObject(0);
            staticPlp.reservoirBuffer[0] = reservoirBuffer[0].getBlockBuffer2D();
            staticPlp.reservoirBuffer[1] = reservoirBuffer[1].getBlockBuffer2D();
            staticPlp.reservoirInfoBuffer[0] = reservoirInfoBuffer[0].getSurfaceObject(0);
            staticPlp.reservoirInfoBuffer[1] = reservoirInfoBuffer[1].getSurfaceObject(0);
            staticPlp.sampleVisibilityBuffer[0] = sampleVisibilityBuffer[0].getSurfaceObject(0);
            staticPlp.sampleVisibilityBuffer[1] = sampleVisibilityBuffer[1].getSurfaceObject(0);
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
            float3 axis = float3(deltaY, -deltaX, 0);
            axis /= deltaAngle;
            if (deltaAngle == 0.0f)
                axis = float3(1, 0, 0);

            g_cameraOrientation = g_cameraOrientation * qRotateZ(g_cameraTiltSpeed * tiltZ);
            g_tempCameraOrientation = g_cameraOrientation * qRotate(g_cameraDirectionalMovingSpeed * deltaAngle, axis);
            g_cameraPosition += g_tempCameraOrientation.toMatrix3x3() * (g_cameraPositionalMovingSpeed * float3(trackX, trackY, trackZ));
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
            cameraIsActuallyMoving =
                (trackZ != 0 || trackX != 0 || trackY != 0 ||
                 tiltZ != 0 || (g_mouseX != g_prevMouseX) || (g_mouseY != g_prevMouseY))
                && operatingCamera;

            g_prevMouseX = g_mouseX;
            g_prevMouseY = g_mouseY;

            perFramePlp.camera.position = g_cameraPosition;
            perFramePlp.camera.orientation = g_tempCameraOrientation.toMatrix3x3();
        }



        bool resetAccumulation = false;
        
        // Camera Window
        static float brightness = g_initBrightness;
        static bool enableEnvLight = true;
        static float log10EnvLightPowerCoeff = 0.0f;
        static float envLightRotation = 0.0f;
        {
            ImGui::Begin("Camera / Env", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Text("W/A/S/D/R/F: Move, Q/E: Tilt");
            ImGui::Text("Mouse Middle Drag: Rotate");

            ImGui::InputFloat3("Position", reinterpret_cast<float*>(&perFramePlp.camera.position));
            static float rollPitchYaw[3];
            g_tempCameraOrientation.toEulerAngles(&rollPitchYaw[0], &rollPitchYaw[1], &rollPitchYaw[2]);
            rollPitchYaw[0] *= 180 / M_PI;
            rollPitchYaw[1] *= 180 / M_PI;
            rollPitchYaw[2] *= 180 / M_PI;
            if (ImGui::InputFloat3("Roll/Pitch/Yaw", rollPitchYaw, 3))
                g_cameraOrientation = qFromEulerAngles(rollPitchYaw[0] * M_PI / 180,
                                                       rollPitchYaw[1] * M_PI / 180,
                                                       rollPitchYaw[2] * M_PI / 180);
            ImGui::Text("Pos. Speed (T/G): %g", g_cameraPositionalMovingSpeed);
            ImGui::SliderFloat("Brightness", &brightness, -5.0f, 5.0f);

            if (ImGui::Button("Screenshot")) {
                CUDADRV_CHECK(cuStreamSynchronize(cuStream));
                auto rawImage = new float4[renderTargetSizeX * renderTargetSizeY];
                glGetTextureSubImage(
                    outputTexture.getHandle(), 0,
                    0, 0, 0, renderTargetSizeX, renderTargetSizeY, 1,
                    GL_RGBA, GL_FLOAT, sizeof(float4) * renderTargetSizeX * renderTargetSizeY, rawImage);
                saveImage("output.png", renderTargetSizeX, renderTargetSizeY, rawImage,
                          false, true);
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

        struct ReSTIRConfigs {
            int32_t log2NumCandidateSamples;
            bool enableTemporalReuse = true;
            bool enableSpatialReuse = true;
            int32_t numSpatialReusePasses;
            int32_t numSpatialNeighbors;
            float spatialNeighborRadius = 20.0f;
            float radiusThresholdForSpatialVisReuse = 10.0f;
            bool useLowDiscrepancySpatialNeighbors = true;
            bool reuseVisibility = true;
            bool reuseVisibilityForTemporal = true;
            bool reuseVisibilityForSpatiotemporal = false;

            ReSTIRConfigs(uint32_t _log2NumCandidateSamples,
                          uint32_t _numSpatialReusePasses, uint32_t _numSpatialNeighbors) :
                log2NumCandidateSamples(_log2NumCandidateSamples),
                numSpatialReusePasses(_numSpatialReusePasses),
                numSpatialNeighbors(_numSpatialNeighbors) {}
        };
        enum class Renderer {
            OriginalReSTIRBiased = 0,
            OriginalReSTIRUnbiased,
            RearchitectedReSTIRBiased,
            RearchitectedReSTIRUnbiased,
        };

        static ReSTIRConfigs orgRestirBiasedConfigs(5, 2, 5);
        static ReSTIRConfigs orgRestirUnbiasedConfigs(5, 1, 3);
        static ReSTIRConfigs rearchRestirBiasedConfigs(5, 1, 1);
        static ReSTIRConfigs rearchRestirUnbiasedConfigs(5, 1, 1);
        //static bool tempInit = false;
        //if (!tempInit) {
        //    rearchRestirBiasedConfigs.enableTemporalReuse = false;
        //    rearchRestirBiasedConfigs.enableSpatialReuse = false;
        //    tempInit = true;
        //}
        static Renderer curRenderer = Renderer::RearchitectedReSTIRBiased;
        //static Renderer curRenderer = Renderer::OriginalReSTIRBiased;
        static ReSTIRConfigs* curRendererConfigs = &orgRestirBiasedConfigs;
        static uint32_t numFramesForCurRenderer = 0;
        static float spatialVisibilityReuseRatio = 50.0f;
        static bool useTemporalDenosier = true;
        static shared::BufferToDisplay bufferTypeToDisplay = shared::BufferToDisplay::NoisyBeauty;
        static float motionVectorScale = -1.0f;
        static bool animate = /*true*/false;
        static bool enableAccumulation = /*true*/false;
        static bool enableJittering = false;
        static bool enableBumpMapping = false;
        bool lastFrameWasAnimated = false;
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
            resetAccumulation |= ImGui::Button("Reset Accum");
            ImGui::Checkbox("Enable Accumulation", &enableAccumulation);
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
                    Renderer prevRenderer = curRenderer;
                    ImGui::RadioButtonE("Original ReSTIR (Biased)", &curRenderer,
                                        Renderer::OriginalReSTIRBiased);
                    ImGui::RadioButtonE("Original ReSTIR (Unbiased)", &curRenderer,
                                        Renderer::OriginalReSTIRUnbiased);
                    ImGui::RadioButtonE("Rearchitected ReSTIR (Biased)", &curRenderer,
                                        Renderer::RearchitectedReSTIRBiased);
                    ImGui::RadioButtonE("Rearchitected ReSTIR (Unbiased)", &curRenderer,
                                        Renderer::RearchitectedReSTIRUnbiased);
                    if (curRenderer != prevRenderer) {
                        numFramesForCurRenderer = 0;
                        resetAccumulation = true;
                    }

                    if (curRenderer == Renderer::OriginalReSTIRBiased)
                        curRendererConfigs = &orgRestirBiasedConfigs;
                    else if (curRenderer == Renderer::OriginalReSTIRUnbiased)
                        curRendererConfigs = &orgRestirUnbiasedConfigs;
                    else if (curRenderer == Renderer::RearchitectedReSTIRBiased)
                        curRendererConfigs = &rearchRestirBiasedConfigs;
                    else if (curRenderer == Renderer::RearchitectedReSTIRUnbiased)
                        curRendererConfigs = &rearchRestirUnbiasedConfigs;

                    ImGui::InputLog2Int("#Candidates", &curRendererConfigs->log2NumCandidateSamples, 8);
                    //ImGui::InputLog2Int("#Samples", &curRendererConfigs->log2NumSamples, 3);

                    resetAccumulation |= ImGui::Checkbox("Temporal Reuse", &curRendererConfigs->enableTemporalReuse);
                    resetAccumulation |= ImGui::Checkbox("Spatial Reuse", &curRendererConfigs->enableSpatialReuse);
                    resetAccumulation |= ImGui::SliderFloat(
                        "Radius", &curRendererConfigs->spatialNeighborRadius, 3.0f, 30.0f);
                    if (curRenderer == Renderer::OriginalReSTIRBiased ||
                        curRenderer == Renderer::OriginalReSTIRUnbiased) {
                        resetAccumulation |= ImGui::SliderInt(
                            "#Reuse Passes",
                            &curRendererConfigs->numSpatialReusePasses, 1, 5);
                        // TODO: Allow the rearchicted version use this parameter?
                        resetAccumulation |= ImGui::SliderInt(
                            "#Neighbors",
                            &curRendererConfigs->numSpatialNeighbors, 1, 10);
                    }
                    resetAccumulation |= ImGui::Checkbox("Low Discrepancy",
                                                         &curRendererConfigs->useLowDiscrepancySpatialNeighbors);
                    resetAccumulation |= ImGui::Checkbox("Reuse Visibility",
                                                         &curRendererConfigs->reuseVisibility);
                    if (curRenderer == Renderer::RearchitectedReSTIRBiased) {
                        resetAccumulation |= ImGui::Checkbox("Reuse Temporal Visibility",
                                                             &curRendererConfigs->reuseVisibilityForTemporal);
                        resetAccumulation |= ImGui::Checkbox("Reuse Spatial Visibility",
                                                             &curRendererConfigs->reuseVisibilityForSpatiotemporal);
                        resetAccumulation |= ImGui::InputFloat("Reuse Ratio (%)", &spatialVisibilityReuseRatio,
                                                               25.0f, 25.0f, 1);
                        spatialVisibilityReuseRatio = std::min(std::max(spatialVisibilityReuseRatio, 0.0f), 100.0f);
                        float reusableRadius = curRendererConfigs->spatialNeighborRadius *
                            std::sqrt(spatialVisibilityReuseRatio / 100.0f);
                        curRendererConfigs->radiusThresholdForSpatialVisReuse = reusableRadius;
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
                    ImGui::Text("Buffer to Display");
                    ImGui::RadioButtonE("Noisy Beauty", &bufferTypeToDisplay, shared::BufferToDisplay::NoisyBeauty);
                    ImGui::RadioButtonE("Albedo", &bufferTypeToDisplay, shared::BufferToDisplay::Albedo);
                    ImGui::RadioButtonE("Normal", &bufferTypeToDisplay, shared::BufferToDisplay::Normal);
                    ImGui::RadioButtonE("Motion Vector", &bufferTypeToDisplay, shared::BufferToDisplay::Flow);
                    ImGui::RadioButtonE("Denoised Beauty", &bufferTypeToDisplay, shared::BufferToDisplay::DenoisedBeauty);

                    if (ImGui::Checkbox("Temporal Denoiser", &useTemporalDenosier)) {
                        CUDADRV_CHECK(cuStreamSynchronize(cuStream));
                        denoiser.destroy();

                        OptixDenoiserModelKind modelKind = useTemporalDenosier ?
                            OPTIX_DENOISER_MODEL_KIND_TEMPORAL :
                            OPTIX_DENOISER_MODEL_KIND_HDR;
                        denoiser = gpuEnv.optixContext.createDenoiser(modelKind, true, true);

                        size_t stateSize;
                        size_t scratchSize;
                        size_t scratchSizeForComputeIntensity;
                        uint32_t numTasks;
                        denoiser.prepare(renderTargetSizeX, renderTargetSizeY, tileWidth, tileHeight,
                                         &stateSize, &scratchSize, &scratchSizeForComputeIntensity,
                                         &numTasks);
                        hpprintf("Denoiser State Buffer: %llu bytes\n", stateSize);
                        hpprintf("Denoiser Scratch Buffer: %llu bytes\n", scratchSize);
                        hpprintf("Compute Intensity Scratch Buffer: %llu bytes\n", scratchSizeForComputeIntensity);
                        denoiserStateBuffer.resize(stateSize, 1);
                        denoiserScratchBuffer.resize(std::max(scratchSize, scratchSizeForComputeIntensity), 1);

                        denoisingTasks.resize(numTasks);
                        denoiser.getTasks(denoisingTasks.data());

                        denoiser.setupState(cuStream, denoiserStateBuffer, denoiserScratchBuffer);
                    }

                    ImGui::SliderFloat("Motion Vector Scale", &motionVectorScale, -2.0f, 2.0f);

                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }

            ImGui::End();
        }

        // Stats Window
        {
            ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            static MovingAverageTime cudaFrameTime;
            static MovingAverageTime updateTime;
            static MovingAverageTime setupGBuffersTime;

            static MovingAverageTime performInitialAndTemporalRISTime;
            static MovingAverageTime performSpatialRISTime;
            static MovingAverageTime shadingTime;

            static MovingAverageTime performPreSamplingLightsTime;
            static MovingAverageTime performPerPixelRISTime;
            static MovingAverageTime traceShadowRaysTime;
            static MovingAverageTime shadeAndResampleTime;

            static MovingAverageTime denoiseTime;

            if (frameIndex >= 2) {
                cudaFrameTime.append(curGPUTimer.frame.report());
                updateTime.append(curGPUTimer.update.report());
                setupGBuffersTime.append(curGPUTimer.setupGBuffers.report());
                denoiseTime.append(curGPUTimer.denoise.report());
            }

            if (numFramesForCurRenderer >= 2) {
                if (curRenderer == Renderer::OriginalReSTIRBiased ||
                    curRenderer == Renderer::OriginalReSTIRUnbiased) {
                    performInitialAndTemporalRISTime.append(curGPUTimer.performInitialAndTemporalRIS.report());
                    performSpatialRISTime.append(curGPUTimer.performSpatialRIS.report());
                    shadingTime.append(curGPUTimer.shading.report());
                }
                else {
                    performPreSamplingLightsTime.append(curGPUTimer.performPreSamplingLights.report());
                    performPerPixelRISTime.append(curGPUTimer.performPerPixelRIS.report());
                    traceShadowRaysTime.append(curGPUTimer.traceShadowRays.report());
                    shadeAndResampleTime.append(curGPUTimer.shadeAndResample.report());
                }
            }

            //ImGui::SetNextItemWidth(100.0f);
            ImGui::Text("CUDA/OptiX GPU %.3f [ms]:", cudaFrameTime.getAverage());
            ImGui::Text("  Update: %.3f [ms]", updateTime.getAverage());
            ImGui::Text("  Setup G-Buffers: %.3f [ms]", setupGBuffersTime.getAverage());
            if (curRenderer == Renderer::OriginalReSTIRBiased ||
                curRenderer == Renderer::OriginalReSTIRUnbiased) {
                ImGui::Text("  Initial RIS + Temporal RIS: %.3f [ms]", performInitialAndTemporalRISTime.getAverage());
                ImGui::Text("  Spatial RIS: %.3f [ms]", performSpatialRISTime.getAverage());
                ImGui::Text("  Shading: %.3f [ms]", shadingTime.getAverage());
            }
            else {
                ImGui::Text("  Light Pre-sampling: %.3f [ms]", performPreSamplingLightsTime.getAverage());
                ImGui::Text("  Per-Pixel RIS: %.3f [ms]", performPerPixelRISTime.getAverage());
                ImGui::Text("  Trace Shadow Rays: %.3f [ms]", traceShadowRaysTime.getAverage());
                ImGui::Text("  Shade and Resample: %.3f [ms]", shadeAndResampleTime.getAverage());
            }
            if (bufferTypeToDisplay == shared::BufferToDisplay::DenoisedBeauty)
                ImGui::Text("  Denoise: %.3f [ms]", denoiseTime.getAverage());

            ImGui::Text("%u [spp]", numAccumFrames + 1);

            ImGui::End();
        }



        curGPUTimer.frame.start(cuStream);

        // JP: 各インスタンスのトランスフォームを更新する。
        // EN: Update the transform of each instance.
        if (animate || lastFrameWasAnimated) {
            cudau::TypedBuffer<shared::InstanceData> &curInstDataBuffer = gpuEnv.instDataBuffer[bufferIndex];
            shared::InstanceData* instDataBufferOnHost = curInstDataBuffer.map();
            for (int i = 0; i < instControllers.size(); ++i) {
                InstanceController* controller = instControllers[i];
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
            perFramePlp.travHandle = ias.rebuild(cuStream, iasInstanceBuffer, iasMem, asScratchMem);
        curGPUTimer.update.stop(cuStream);

        bool newSequence = resized || frameIndex == 0 || resetAccumulation;
        bool firstAccumFrame =
            animate || !enableAccumulation || cameraIsActuallyMoving || newSequence;
        if (firstAccumFrame)
            numAccumFrames = 0;
        else
            ++numAccumFrames;
        if (newSequence)
            hpprintf("New sequence started.\n");

        perFramePlp.numAccumFrames = numAccumFrames;
        perFramePlp.frameIndex = frameIndex;
        perFramePlp.instanceDataBuffer = gpuEnv.instDataBuffer[bufferIndex].getDevicePointer();
        perFramePlp.envLightPowerCoeff = std::pow(10.0f, log10EnvLightPowerCoeff);
        perFramePlp.envLightRotation = envLightRotation;
        perFramePlp.spatialNeighborRadius = curRendererConfigs->spatialNeighborRadius;
        perFramePlp.radiusThresholdForSpatialVisReuse = curRendererConfigs->radiusThresholdForSpatialVisReuse;
        perFramePlp.mousePosition = int2(static_cast<int32_t>(g_mouseX),
                                         static_cast<int32_t>(g_mouseY));
        perFramePlp.pickInfo = pickInfos[bufferIndex].getDevicePointer();

        perFramePlp.log2NumCandidateSamples = curRendererConfigs->log2NumCandidateSamples;
        perFramePlp.numSpatialNeighbors = curRendererConfigs->numSpatialNeighbors;
        perFramePlp.useLowDiscrepancyNeighbors = curRendererConfigs->useLowDiscrepancySpatialNeighbors;
        perFramePlp.reuseVisibility = curRendererConfigs->reuseVisibility;
        perFramePlp.reuseVisibilityForTemporal = curRendererConfigs->reuseVisibilityForTemporal;
        perFramePlp.reuseVisibilityForSpatiotemporal = curRendererConfigs->reuseVisibilityForSpatiotemporal;
        perFramePlp.enableTemporalReuse = curRendererConfigs->enableTemporalReuse;
        perFramePlp.enableSpatialReuse = curRendererConfigs->enableSpatialReuse;
        perFramePlp.useUnbiasedEstimator =
            curRenderer == Renderer::OriginalReSTIRUnbiased ||
            curRenderer == Renderer::RearchitectedReSTIRUnbiased;
        perFramePlp.bufferIndex = bufferIndex;
        perFramePlp.resetFlowBuffer = newSequence;
        perFramePlp.enableJittering = enableJittering;
        perFramePlp.enableEnvLight = enableEnvLight;
        perFramePlp.enableBumpMapping = enableBumpMapping;
        for (int i = 0; i < lengthof(debugSwitches); ++i)
            perFramePlp.setDebugSwitch(i, debugSwitches[i]);

        CUDADRV_CHECK(cuMemcpyHtoDAsync(perFramePlpOnDevice, &perFramePlp, sizeof(perFramePlp), cuStream));

        uint32_t currentReservoirIndex = (lastReservoirIndex + 1) % 2;
        //hpprintf("%u\n", currentReservoirIndex);

        plp.currentReservoirIndex = currentReservoirIndex;
        plp.spatialNeighborBaseIndex = lastSpatialNeighborBaseIndex;
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
        CUDADRV_CHECK(cuMemcpyHtoDAsync(gpuEnv.plpPtr, &plp, sizeof(plp), cuStream));

        // JP: Gバッファーのセットアップ。
        //     ここではレイトレースを使ってGバッファーを生成しているがもちろんラスタライザーで生成可能。
        // EN: Setup the G-buffers.
        //     Generate the G-buffers using ray trace here, but of course this can be done using rasterizer.
        curGPUTimer.setupGBuffers.start(cuStream);
        gpuEnv.pipeline.setRayGenerationProgram(gpuEnv.setupGBuffersRayGenProgram);
        gpuEnv.pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
        curGPUTimer.setupGBuffers.stop(cuStream);

        if (curRenderer == Renderer::OriginalReSTIRBiased ||
            curRenderer == Renderer::OriginalReSTIRUnbiased) {
            // JP: 各ピクセルで独立したStreaming RISを実行。
            //     そして前フレームの(時間的)隣接ピクセルとの間でReservoirの結合を行う。
            // EN: Perform independent streaming RIS on each pixel.
            //     Then combine reservoirs between the current pixel and
            //     (temporally) neighboring pixel from the previous frame.
            curGPUTimer.performInitialAndTemporalRIS.start(cuStream);
            if (curRendererConfigs->enableTemporalReuse && !newSequence) {
                if (curRenderer == Renderer::OriginalReSTIRUnbiased)
                    gpuEnv.pipeline.setRayGenerationProgram(
                        gpuEnv.performInitialAndTemporalRISUnbiasedRayGenProgram);
                else
                    gpuEnv.pipeline.setRayGenerationProgram(
                        gpuEnv.performInitialAndTemporalRISBiasedRayGenProgram);
            }
            else {
                gpuEnv.pipeline.setRayGenerationProgram(gpuEnv.performInitialRISRayGenProgram);
            }
            gpuEnv.pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
            curGPUTimer.performInitialAndTemporalRIS.stop(cuStream);

            // JP: 各ピクセルにおいて(空間的)隣接ピクセルとの間でReservoirの結合を行う。
            // EN: For each pixel, combine reservoirs between the current pixel and
            //     (Spatially) neighboring pixels.
            curGPUTimer.performSpatialRIS.start(cuStream);
            if (curRendererConfigs->enableSpatialReuse) {
                int32_t numSpatialReusePasses;
                if (curRenderer == Renderer::OriginalReSTIRUnbiased)
                    gpuEnv.pipeline.setRayGenerationProgram(gpuEnv.performSpatialRISUnbiasedRayGenProgram);
                else
                    gpuEnv.pipeline.setRayGenerationProgram(gpuEnv.performSpatialRISBiasedRayGenProgram);
                numSpatialReusePasses = curRendererConfigs->numSpatialReusePasses;

                for (int i = 0; i < numSpatialReusePasses; ++i) {
                    uint32_t baseIndex = lastSpatialNeighborBaseIndex + curRendererConfigs->numSpatialNeighbors * i;
                    plp.spatialNeighborBaseIndex = baseIndex;
                    CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
                    gpuEnv.pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
                    currentReservoirIndex = (currentReservoirIndex + 1) % 2;
                    plp.currentReservoirIndex = currentReservoirIndex;
                }
                lastSpatialNeighborBaseIndex += curRendererConfigs->numSpatialNeighbors * numSpatialReusePasses;
            }
            curGPUTimer.performSpatialRIS.stop(cuStream);

            // JP: 生き残ったサンプルを使ってシェーディングを実行。
            // EN: Perform shading using the survived samples.
            curGPUTimer.shading.start(cuStream);
            CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
            gpuEnv.pipeline.setRayGenerationProgram(gpuEnv.shadingRayGenProgram);
            gpuEnv.pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
            curGPUTimer.shading.stop(cuStream);
        }
        else {
            constexpr uint32_t numPreSampledLights = shared::numLightSubsets * shared::lightSubsetSize;

            // JP:
            // EN:
            curGPUTimer.performPreSamplingLights.start(cuStream);
            gpuEnv.kernelPerformLightPreSampling(
                cuStream, gpuEnv.kernelPerformLightPreSampling.calcGridDim(numPreSampledLights));
            curGPUTimer.performPreSamplingLights.stop(cuStream);

            // JP:
            // EN:
            curGPUTimer.performPerPixelRIS.start(cuStream);
            gpuEnv.kernelPerformPerPixelRIS(
                cuStream, gpuEnv.kernelPerformPerPixelRIS.calcGridDim(renderTargetSizeX, renderTargetSizeY));
            curGPUTimer.performPerPixelRIS.stop(cuStream);

            // JP:
            // EN:
            curGPUTimer.traceShadowRays.start(cuStream);
            optixu::ProgramGroup traceShadowRaysRayGenProgram;
            if (!newSequence) {
                if (curRendererConfigs->enableTemporalReuse && curRendererConfigs->enableSpatialReuse)
                    traceShadowRaysRayGenProgram = gpuEnv.traceShadowRaysWithSpatioTemporalReuseRayGenProgram;
                else if (curRendererConfigs->enableTemporalReuse)
                    traceShadowRaysRayGenProgram = gpuEnv.traceShadowRaysWithTemporalReuseRayGenProgram;
                else if (curRendererConfigs->enableSpatialReuse)
                    traceShadowRaysRayGenProgram = gpuEnv.traceShadowRaysWithSpatialReuseRayGenProgram;
                else
                    traceShadowRaysRayGenProgram = gpuEnv.traceShadowRaysRayGenProgram;
            }
            else {
                traceShadowRaysRayGenProgram = gpuEnv.traceShadowRaysRayGenProgram;
            }
            gpuEnv.pipeline.setRayGenerationProgram(traceShadowRaysRayGenProgram);
            gpuEnv.pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
            curGPUTimer.traceShadowRays.stop(cuStream);

            // JP:
            // EN:
            curGPUTimer.shadeAndResample.start(cuStream);
            optixu::ProgramGroup shadeRayGenProgram;
            if (!newSequence) {
                if (curRenderer == Renderer::RearchitectedReSTIRBiased) {
                    if (curRendererConfigs->enableTemporalReuse && curRendererConfigs->enableSpatialReuse)
                        shadeRayGenProgram = gpuEnv.shadeAndResampleWithSpatiotemporalReuseBiasedRayGenProgram;
                    else if (curRendererConfigs->enableTemporalReuse)
                        shadeRayGenProgram = gpuEnv.shadeAndResampleWithTemporalReuseBiasedRayGenProgram;
                    else if (curRendererConfigs->enableSpatialReuse)
                        shadeRayGenProgram = gpuEnv.shadeAndResampleWithSpatialReuseBiasedRayGenProgram;
                    else
                        shadeRayGenProgram = gpuEnv.shadeAndResampleRayGenProgram;
                }
                else {
                    if (curRendererConfigs->enableTemporalReuse && curRendererConfigs->enableSpatialReuse)
                        shadeRayGenProgram = gpuEnv.shadeAndResampleWithSpatiotemporalReuseUnbiasedRayGenProgram;
                    else if (curRendererConfigs->enableTemporalReuse)
                        shadeRayGenProgram = gpuEnv.shadeAndResampleWithTemporalReuseUnbiasedRayGenProgram;
                    else if (curRendererConfigs->enableSpatialReuse)
                        shadeRayGenProgram = gpuEnv.shadeAndResampleWithSpatialReuseUnbiasedRayGenProgram;
                    else
                        shadeRayGenProgram = gpuEnv.shadeAndResampleRayGenProgram;
                }
            }
            else {
                shadeRayGenProgram = gpuEnv.shadeAndResampleRayGenProgram;
            }
            gpuEnv.pipeline.setRayGenerationProgram(shadeRayGenProgram);
            gpuEnv.pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
            curGPUTimer.shadeAndResample.stop(cuStream);

            ++lastSpatialNeighborBaseIndex;
        }

        lastReservoirIndex = currentReservoirIndex;

        // JP: 結果をリニアバッファーにコピーする。(法線の正規化も行う。)
        // EN: Copy the results to the linear buffers (and normalize normals).
        cudau::dim3 dimCopyBuffers = kernelCopyToLinearBuffers.calcGridDim(renderTargetSizeX, renderTargetSizeY);
        kernelCopyToLinearBuffers(cuStream, dimCopyBuffers,
                                  beautyAccumBuffer.getSurfaceObject(0),
                                  albedoAccumBuffer.getSurfaceObject(0),
                                  normalAccumBuffer.getSurfaceObject(0),
                                  gBuffer2[bufferIndex].getSurfaceObject(0),
                                  linearBeautyBuffer.getDevicePointer(),
                                  linearAlbedoBuffer.getDevicePointer(),
                                  linearNormalBuffer.getDevicePointer(),
                                  linearFlowBuffer.getDevicePointer(),
                                  uint2(renderTargetSizeX, renderTargetSizeY));

        curGPUTimer.denoise.start(cuStream);
        if (bufferTypeToDisplay == shared::BufferToDisplay::DenoisedBeauty) {
            denoiser.computeIntensity(cuStream,
                                      linearBeautyBuffer, OPTIX_PIXEL_FORMAT_FLOAT4,
                                      denoiserScratchBuffer, hdrIntensity);
            //float hdrIntensityOnHost;
            //CUDADRV_CHECK(cuMemcpyDtoH(&hdrIntensityOnHost, hdrIntensity, sizeof(hdrIntensityOnHost)));
            //printf("%g\n", hdrIntensityOnHost);
            for (int i = 0; i < denoisingTasks.size(); ++i)
                denoiser.invoke(cuStream,
                                false, hdrIntensity, 0.0f,
                                linearBeautyBuffer, OPTIX_PIXEL_FORMAT_FLOAT4,
                                linearAlbedoBuffer, OPTIX_PIXEL_FORMAT_FLOAT4,
                                linearNormalBuffer, OPTIX_PIXEL_FORMAT_FLOAT4,
                                linearFlowBuffer, OPTIX_PIXEL_FORMAT_FLOAT2,
                                newSequence ? linearBeautyBuffer : linearDenoisedBeautyBuffer,
                                linearDenoisedBeautyBuffer,
                                denoisingTasks[i]);
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
        case shared::BufferToDisplay::DenoisedBeauty:
            bufferToDisplay = linearDenoisedBeautyBuffer.getDevicePointer();
            break;
        default:
            Assert_ShouldNotBeCalled();
            break;
        }
        kernelVisualizeToOutputBuffer(
            cuStream, kernelVisualizeToOutputBuffer.calcGridDim(renderTargetSizeX, renderTargetSizeY),
            bufferToDisplay,
            bufferTypeToDisplay,
            std::pow(10.0f, brightness),
            0.5f, std::pow(10.0f, motionVectorScale),
            outputBufferSurfaceHolder.getNext(),
            uint2(renderTargetSizeX, renderTargetSizeY));

        outputBufferSurfaceHolder.endCUDAAccess(cuStream);

        curGPUTimer.frame.stop(cuStream);



        // ----------------------------------------------------------------
        // JP: OptiXによる描画結果を表示用レンダーターゲットにコピーする。
        // EN: Copy the OptiX rendering results to the display render target.

        if (bufferTypeToDisplay == shared::BufferToDisplay::NoisyBeauty ||
            bufferTypeToDisplay == shared::BufferToDisplay::DenoisedBeauty) {
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
        ++numFramesForCurRenderer;
    }

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));
    gpuTimers[1].finalize();
    gpuTimers[0].finalize();



    CUDADRV_CHECK(cuMemFree(plpOnDevice));

    pickInfos[1].finalize();
    pickInfos[0].finalize();

    CUDADRV_CHECK(cuMemFree(perFramePlpOnDevice));
    CUDADRV_CHECK(cuMemFree(staticPlpOnDevice));

    spatialNeighborDeltas.finalize();

    drawOptiXResultShader.finalize();
    vertexArrayForFullScreen.finalize();

    outputSampler.finalize();
    outputBufferSurfaceHolder.finalize();
    outputArray.finalize();
    outputTexture.finalize();


    
    CUDADRV_CHECK(cuMemFree(hdrIntensity));
    CUDADRV_CHECK(cuModuleUnload(moduleCopyBuffers));    
    denoiserScratchBuffer.finalize();
    denoiserStateBuffer.finalize();
    denoiser.destroy();
    
    finalizeScreenRelatedBuffers();

    preSampledLights.finalize();
    lightPreSamplingRngs.finalize();



    envLightImportanceMap.finalize(gpuEnv.cuContext);
    if (envLightTexture)
        cuTexObjectDestroy(envLightTexture);
    envLightArray.finalize();
    lightInstDist.finalize();
    hitGroupSBT.finalize();
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
        inst->lightGeomInstDist.finalize();
    }
    for (int i = geomGroups.size() - 1; i >= 0; --i) {
        GeometryGroup* geomGroup = geomGroups[i];
        geomGroup->optixGas.destroy();
    }
    for (int i = geomInsts.size() - 1; i >= 0; --i) {
        GeometryInstance* geomInst = geomInsts[i];
        geomInst->optixGeomInst.destroy();
        geomInst->emitterPrimDist.finalize();
        geomInst->triangleBuffer.finalize();
        geomInst->vertexBuffer.finalize();
    }
    for (int i = materials.size() - 1; i >= 0; --i) {
        Material* material = materials[i];
    }

    CUDADRV_CHECK(cuStreamDestroy(cuStream));
    
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
