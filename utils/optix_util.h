/*

   Copyright 2024 Shin Watanabe

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

#pragma once

/*

Note:
JP:
- 現状ではあらゆるAPIに破壊的変更が入る可能性がある。
- (少なくともホスト側コンパイラーがMSVC 16.8.2の場合は)"-std=c++17"をptxのコンパイル時に設定する必要あり。
- Visual StudioにおけるCUDAのプロパティ"Use Fast Math"はptxコンパイルに対して機能していない？
EN:
- It is likely for now that any API will have breaking changes.
- Setting "-std=c++17" is required for ptx compilation (at least for the case the host compiler is MSVC 16.8.2).
- In Visual Studio, does the CUDA property "Use Fast Math" not work for ptx compilation??

変更履歴 / Update History:
- JP: - OptiX 8.1.0をサポート。
  EN: - Supported OptiX 8.1.0.

- JP: - optixReportIntersection()に返り値があることを忘れていたのを修正。
  EN: - fixed forgetting that optixReportIntersection has a return value.

- JP: - 各プログラム(グループ)にsetActive()を追加。
  EN: - Added setActive() to programs (groups).

- !!BREAKING
  JP: - OptiX 8.0.0をサポート。
        Context::createDenoiser(), Denoiser::invoke()のパラメターを変更。
  EN: - Supported OptiX 8.0.0.
        Changed the parameters of Context::createDenoiser(), Denoiser::invoke().

- JP: - Displacement Micro-Mapをサポート。
  EN: - Supported displacement micro-map.

- !!BREAKING
  JP: - OptiX 7.7.0をサポート。
      - Pipeline::link()のパラメターを変更。
      - Displaced Micro-Meshは未対応。
  EN: - Supported OptiX 7.7.0.
      - Changed the parameters of Pipeline::link().
      - Does not support displaced micro-mesh yet.

- !!BREAKING
  JP: - ProgramGroupをProgram, HitProgramGroup, CallableProgramGroupに分割した。
  EN: - Separated ProgramGroup into Program, HitProgramGroup, CallableProgramGroup.

- !!BREAKING
  JP: - Opacity Micro-Mapをサポート。
      - インデックスサイズ指定用のenum classを定義。
  EN: - Supported opacity micro-map.
      - Defined an enum class to specify index sizes.

- !!BREAKING
  JP: - OptiX 7.6.0をサポート。
      - ホスト側APIのbool引数それぞれの個別の型を定義。
      - Pipeline::setPipelineOptions()の引数の順序を変更。
  EN: - Supported OptiX 7.6.0.
      - Defined a dedicated type for each bool parameter of the host-side API.
      - Changed the order of parameters of Pipeline::setPipelineOptions().

- !!BREAKING
  JP: - AnnotatedPayloadSignatureテンプレート型を定義。ペイロードアノテーションはこの型経由で行う。
        PayloadSignature::createPayloadType()を削除。
  EN: - Defined AnnotatedPayloadSignature template type. Use this type for payload annotation.
        Removed PayloadSignature::createPayloadType().

- !!BREAKING
  JP: - Upscaling Denoiserをサポート。
      - Denoiserクラスのインターフェースをいくらか変更。
      - reportIntersection(), throwException()を削除。
        代わりに対応するシグネチャー型経由で値の取得・設定を行う。
      - PayloadType::create()を削除、対応するシグネチャー型のstaticメンバー関数として実装。
  EN: - Supported upscaling denoisers.
      - Changed some interfaces of Denoiser class.
      - Removed reportIntersection(), throwException().
        Set or get values via corresponding signature types instead.
      - Removed PayloadType::create(), implemented as a static member function of
        the corresponding signature type instead.

- !!BREAKING
  JP: - OptiX 7.5.0をサポート。
        Upscaling Denoiserは未対応。
      - trace()関数をシグネチャー型のメンバー関数に変更。
  EN: - Supported OptiX 7.5.0.
        Does not support upscaling denoiser yet.
      - Changed the trace() fuction to a member function of the signature type.

- !!BREAKING??
  JP: - RT_DEVICE_FUNCTIONからinline属性を削除。RT_INLINEを新設。
        __CUDACC__と__CUDA_ARCH__の使い分けを明確に。
  EN: - Removed inline qualifier from RT_DEVICE_FUNCTION and added RT_INLINE.
        Disambiguate usage of __CUDACC__ and __CUDA_ARCH__.

- !!BREAKING
  JP: - getPayloads()/setPayloads(), getAttributes(), getExceptionDetails()を削除。
        ペイロードなどの値はシグネチャー型経由で取得・設定を行う。
  EN: - Removed getPayloads()/setPayloads(), getAttributes(), getExceptionDetails().
        Set or get values like payload via signature types.

- JP: - createModuleFromPTXString(), createMissProgram(),
        createHitProgramGroupFor***IS(), createCallableProgramGroup()
        がペイロードアノテーションを追加で受け取れるように変更。
        詳細については新たなサンプル"payload_annotation"を参照。
  EN: - Changed createModuleFromPTXString(), createMissProgram(),
        createHitProgramGroupFor***IS(), createCallableProgramGroup()
        to be able to additionally take payload annotations.
        See a new sample "payload_annotation" for the details.

- !!BREAKING
  JP: - OptiX 7.4.0をサポート。
      - SBTレコード中のユーザーデータの並び順が逆になった。
        !! ユーザーのOptiXカーネルを少し修正する必要があります。
      - 三角形プリミティブ、カーブプリミティブ用のヒットグループ作成をそれぞれ
        createHitProgramGroupForTriangleIS()とcreateHitProgramGroupForCurveIS()で行うように変更。
  EN: - Supported OptiX 7.4.0.
      - The order of user data in a SBT record has been reversed.
        !! User's OptiX kernels need to be modified a bit.
      - Changed hit group creation of triangle and curve primitives to use
        createHitProgramGroupForTriangleIS() and createHitProgramGroupForCurveIS() respectively.

- !!BREAKING
  JP: - OptiX 7.3.0をサポート。
      - InstanceAccelerationStructure::setConfiguration()が一つ多くの引数を受け取るようになった。
      - Denoiserの入出力レイヤーをinvoke(), computeIntensity()に直接渡すように変更。setLayers()を削除。
  EN: - Supported OptiX 7.3.0.
      - InstanceAccelerationStructure::setConfiguration() takes one more additional argument.
      - Changed Denoiser's invoke(), computeIntensisty() to directly take input/output layers.
        Removed setLayers().

- !!BREAKING
  JP: - カーブプリミティブをサポート。
      - ヒットグループを生成する関数を変更。三角形とカーブに関してはcreateHitProgramGroupForBuiltinIS()を、
        カスタムプリミティブに関してはcreateHitProgramGroupForCustomIS()を使用してください。
  EN: - Added support for curve primitives.
      - Changed the function to create a hit group. Use createHitProgramGroupForBuiltinIS() for triangles and
        curves and createHitProgramGroupForCustomIS() for custom primitives.

- !!BREAKING
  JP: - GeometryInstanceとGASをSceneから生成する関数の引数の型をenumに変更。
  EN: - Changed the type of argument of the functions to create a GeometryInstance or a GAS from a Scene to enum.

- !!BREAKING
  JP: - GAS/IASのremoveChild()を削除。代わりにremoveChildAt()を定義。
        GAS/IAS::findChildIndex()を使用すれば目的の子のインデックスを特定できる。
      - また、GAS/IAS::clearChildren()を定義。
  EN: - Removed GAS/IAS's removeChild(), instead defined removeChildAt().
        Use GAS/IAS::findChildIndex() to identify the index of the target child.
      - Also, defined GAS/IAS::clearChildren().

- JP: - GASの子ごとのユーザーデータを設定するAPIを追加。
  EN: - Added APIs to set per-GAS child user data.

- JP: - 各種パラメターを取得するためのAPIを追加。
  EN: - Added APIs to get parameters.

- JP: - マテリアルのユーザーデータのサイズやアラインメントを、シェーダーバインディングテーブルレイアウト生成後に
        変更した場合にレイアウトを手動で無効化するためのScene::markShaderBindingTableLayoutDirty()を追加。
      - 併せてScene::shaderBindingTableLayoutIsReady()も追加。
  EN: - Added Scene::markShaderBindingTableLayoutDirty() to manually invalidate the layout of shader binding table
        for the case changing the size and/or alignment of a material's user data after generating the layout.
      - Added Scene::shaderBindingTableLayoutIsReady() as well.

- !!BREAKING
  JP: - InstanceAccelerationStructure::prepareForBuild()が引数でインスタンス数を返さないように変更。
        InstanceAccelerationStructure::getNumChildren()を代わりに使用してください。
  EN: - Changed InstanceAccelerationStructure::prepareForBuild() not to return the number of instances as an argument.
        Use InstanceAccelerationStructure::getNumChildren() instead.

----------------------------------------------------------------
TODO:
- 深いトラバーサルグラフにおけるインスタンスのSBTオフセットの累積サポート。
- NVRTC環境のテスト。
- フローベクターの信頼性についてテスト。
- AOV Denoiserのサンプル作成。
- Linux環境でのテスト。
- モジュールの並列コンパイル。
- ASのRelocationサポート。
- OMMのRelocationサポート。
- Multi GPUs?
- ユニットテスト。
- Instance Pointersサポート。
- removeUncompacted再考。(compaction終了待ちとしてとらえる？)
- 途中で各オブジェクトのパラメターを変更した際の処理。
  パイプラインのセットアップ順などが現状は暗黙的に固定されている。これを自由な順番で変えられるようにする。
- Assertとexceptionの整理。

検討事項 (Items under author's consideration, ignore this :) ):
- Denoiserの事前設定は画像サイズにも依存するので、各バッファーはinvoke時ではなく事前に渡しておくべき？
- Priv構造体がOptiXの構造体を直接持っていない場合が多々あるのがもったいない？
  => OptixBuildInputは巨大なパディングを含んでいるので好ましくない。
  => IASが直接持っているOptixBuildInputを除去orポインター化？
     OptixBuildInputInstanceArrayを持つようにするとrebuildなどの各処理で毎回OptixBuildInputのクリアが必要。
     ポインター化はメモリの無駄遣いを本質的には解決しない。
- optixuのenumかOptiXのenum、使い分ける基準について考える。
  => OptiX側のenumが余計なものを含んでいる場合はoptixu側でenumを定義したほうがミスが少ない。
  GeometryTypeはOptiX側のでも良い気もするが、OPTIX_PRIMITIVE_TYPE_とOPTIX_PRIMITIVE_TYPE_FLAGS_でミスりそう。
- MaterialのヒットグループのISとGeometryInstanceの一致確認。
  => ついでにプログラムタイプごとの型つくる？
- GeometryInstanceのGASをdirtyにする処理のうち、いくつかは内部的にSBTレイアウトの無効化をスキップできるはず。
- HitGroup以外のProgramGroupにユーザーデータを持たせる。
- Material::setHitGroup()はレイタイプの数値が同じでもヒットグループのパイプラインが違っていれば別個に登録できるが、
  これがAPI上からは読み取りづらい。冗長だが敢えてパイプラインの識別情報も引数として受け取るべき？
- Scene::generateShaderBindingTableLayout()はPipelineに依存すべき？
  => その場合はこの関数自体setSceneを使った後に呼ばれるPipelineの関数となるべき？
     SBT自体内容はレコードのヘッダーによって必ずパイプラインに依存するので
     レイアウトがパイプラインに依存するのは問題ない？
  現状の問題点：
  - パイプラインごとにマテリアルに設定されているレイタイプ数が異なる場合に、
    最大のレイタイプ数をGASに設定すると、SBTレコードを書き込む際にマテリアルがあるレイタイプに対して
    設定されていないと言われてしまう。(とりあえず空のHitGroupを作れるようにして対処してある。)
    => GASのレイタイプ数設定をパイプラインに依存させる？ => Sceneとパイプラインは切り離したい。
    => 多少Sceneがパイプラインに依存するとしてもパイプラインごとに別のレイタイプ数設定のほうがきれいそう。
    => しかしIASが持つSBTオフセットが絶対的な値なので、IASをパイプライン間で共通化させようと思うと結局無理。
  - GASがレイタイプ数設定を持っているのが不自然？ => パイプラインがレイタイプ数を持つようにして
    SBTレイアウト計算もPipelineに依存させる？
----------------------------------------------------------------
- GAS/IASに関してユーザーが気にするところはAS云々ではなくグループ化なので
  名前を変えるべき？GeometryGroup/InstanceGroupのような感じ。
  しかしビルドやアップデートを明示的にしているため結局ASであるということをユーザーが意識する必要がある。
- ユーザーがあるSBTレコード中の各データのストライドを意識せずともそれぞれのオフセットを取得する関数。
  => オフセット値を読み取った後にデータを読み取るというindirectionになるため、そもそもあまり好ましくない気も。
- InstanceのsetChildはTraversal Graph Depthに影響しないので名前を変えるべき？setTraversable()?
  => GASのsetChildもDepthに影響しないことを考えるとこのままで良いかも。

*/



#define OPTIXU_STRINGIFY(x) #x
#define OPTIXU_TO_STRING(x) OPTIXU_STRINGIFY(x)



// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#   define OPTIXU_Platform_Windows
#   if defined(_MSC_VER)
#       define OPTIXU_Platform_Windows_MSVC
#       if defined(__INTELLISENSE__)
#           define OPTIXU_Platform_CodeCompletion
#       endif // if defined(__INTELLISENSE__)
#   endif // if defined(_MSC_VER)
#elif defined(__APPLE__)
#   define OPTIXU_Platform_macOS
#endif // if defined(_WIN32) || defined(_WIN64)



#if defined(__CUDACC_RTC__)
// JP: cstdintやcfloatに対応する定義はユーザーに任せられている。
// EN: Defining things corresponding to cstdint and cfloat is left to the user.
#else // if defined(__CUDACC_RTC__)
#include <cstdint>
#include <cfloat>
#include <string>
#include <vector>
#include <initializer_list>
#   if __cplusplus >= 202002L
#       include <concepts>
#   endif // if __cplusplus >= 202002L
#endif // if defined(__CUDACC_RTC__)

#if defined(OPTIXU_Platform_Windows_MSVC)
#   pragma warning(push)
#   pragma warning(disable:4819)
#endif // if defined(OPTIXU_Platform_Windows_MSVC)
// JP: NVRTCを使う場合でも「アプリケーション」ユーザーはOptiX SDKのインストールを必要とする。
// EN: Even NVRTC requires the "application" user to install OptiX SDK.
#include <optix.h>
#if !defined(__CUDA_ARCH__)
#   include <optix_stubs.h>
#endif // if !defined(__CUDA_ARCH__)
#if defined(OPTIXU_Platform_Windows_MSVC)
#   pragma warning(pop)
#endif // if defined(OPTIXU_Platform_Windows_MSVC)



#if !defined(OPTIXU_ENABLE_ASSERT)
#   if defined(_DEBUG)
#       define OPTIXU_ENABLE_ASSERT 1
#   else
#       define OPTIXU_ENABLE_ASSERT 0
#   endif // if defined(_DEBUG)
#endif // if !defined(OPTIXU_ENABLE_ASSERT)

#if !defined(OPTIXU_DISABLE_RUNTIME_ERROR)
#   define OPTIXU_ENABLE_RUNTIME_ERROR 1
#endif // if !defined(OPTIXU_DISABLE_RUNTIME_ERROR)

#if defined(__CUDACC__)
#   define RT_CALLABLE_PROGRAM extern "C" __device__
#   define RT_INLINE __forceinline__
#   define RT_DEVICE_FUNCTION __device__
#   define RT_COMMON_FUNCTION __host__ __device__
#   if !defined(RT_PIPELINE_LAUNCH_PARAMETERS)
#       define RT_PIPELINE_LAUNCH_PARAMETERS extern "C" __constant__
#   endif
#else // if defined(__CUDACC__)
#   define RT_CALLABLE_PROGRAM
#   define RT_INLINE inline
#   define RT_DEVICE_FUNCTION
#   define RT_COMMON_FUNCTION
#   define RT_PIPELINE_LAUNCH_PARAMETERS
#endif // if defined(__CUDACC__)

#define RT_RG_NAME(name) __raygen__ ## name
#define RT_MS_NAME(name) __miss__ ## name
#define RT_EX_NAME(name) __exception__ ## name
#define RT_CH_NAME(name) __closesthit__ ## name
#define RT_AH_NAME(name) __anyhit__ ## name
#define RT_IS_NAME(name) __intersection__ ## name
#define RT_DC_NAME(name) __direct_callable__ ## name
#define RT_CC_NAME(name) __continuation_callable__ ## name

#define RT_RG_NAME_STR(name) "__raygen__" name
#define RT_MS_NAME_STR(name) "__miss__" name
#define RT_EX_NAME_STR(name) "__exception__" name
#define RT_CH_NAME_STR(name) "__closesthit__" name
#define RT_AH_NAME_STR(name) "__anyhit__" name
#define RT_IS_NAME_STR(name) "__intersection__" name
#define RT_DC_NAME_STR(name) "__direct_callable__" name
#define RT_CC_NAME_STR(name) "__continuation_callable__" name



#define OPTIXU_DEFINE_OPERATORS_FOR_FLAGS(Type) \
    RT_COMMON_FUNCTION RT_INLINE constexpr Type operator~(Type a) { \
        return static_cast<Type>(~static_cast<uint32_t>(a)); \
    } \
    RT_COMMON_FUNCTION RT_INLINE constexpr Type operator|(Type a, Type b) { \
        return static_cast<Type>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b)); \
    } \
    RT_COMMON_FUNCTION RT_INLINE constexpr Type &operator|=(Type &a, Type b) { \
        a = static_cast<Type>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b)); \
        return a; \
    } \
    RT_COMMON_FUNCTION RT_INLINE constexpr Type operator&(Type a, Type b) { \
        return static_cast<Type>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b)); \
    } \
    RT_COMMON_FUNCTION RT_INLINE constexpr Type &operator&=(Type &a, Type b) { \
        a = static_cast<Type>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b)); \
        return a; \
    }

OPTIXU_DEFINE_OPERATORS_FOR_FLAGS(OptixGeometryFlags);
OPTIXU_DEFINE_OPERATORS_FOR_FLAGS(OptixPrimitiveTypeFlags);
OPTIXU_DEFINE_OPERATORS_FOR_FLAGS(OptixInstanceFlags);
OPTIXU_DEFINE_OPERATORS_FOR_FLAGS(OptixMotionFlags);
OPTIXU_DEFINE_OPERATORS_FOR_FLAGS(OptixRayFlags);
OPTIXU_DEFINE_OPERATORS_FOR_FLAGS(OptixTraversableGraphFlags);
OPTIXU_DEFINE_OPERATORS_FOR_FLAGS(OptixExceptionFlags);
OPTIXU_DEFINE_OPERATORS_FOR_FLAGS(OptixPayloadSemantics);
OPTIXU_DEFINE_OPERATORS_FOR_FLAGS(OptixPayloadTypeID);

#undef OPTIXU_DEFINE_OPERATORS_FOR_FLAGS

#if defined(OPTIXU_Platform_CodeCompletion)
struct float3;
#endif // if defined(OPTIXU_Platform_CodeCompletion)



namespace optixu {
    void devPrintf(const char* fmt, ...);

#if 1
#   define optixuPrintf(fmt, ...) \
        do { \
            optixu::devPrintf(fmt, ##__VA_ARGS__); \
            printf(fmt, ##__VA_ARGS__); \
        } while (0)
#else
#   define optixuPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif

#if OPTIXU_ENABLE_ASSERT
#   if defined(__CUDA_ARCH__)
#       define optixuAssert(expr, fmt, ...) \
            do { \
                if (!(expr)) { \
                    ::printf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
                    ::printf(fmt"\n", ##__VA_ARGS__); \
                    assert(0); \
                } \
            } while (0)
#   else
#       define optixuAssert(expr, fmt, ...) \
            do { \
                if (!(expr)) { \
                    optixu::devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
                    optixu::devPrintf(fmt"\n", ##__VA_ARGS__); \
                    abort(); \
                } \
            } while (0)
#   endif
#else // if OPTIXU_ENABLE_ASSERT
#   define optixuAssert(expr, fmt, ...)
#endif // if OPTIXU_ENABLE_ASSERT

#define optixuAssert_ShouldNotBeCalled() optixuAssert(false, "Should not be called!")
#define optixuAssert_NotImplemented() optixuAssert(false, "Not implemented yet!")

    // JP: stdのメタ関数の抽象化定義。
    // EN: Definitions to abstract std meta functions.
#if defined(__CUDACC_RTC__)
    // TODO
#else // if defined(__CUDACC_RTC__)
#   if __cplusplus >= 202002L
    template <class _From, class _To>
    concept convertible_to = std::convertible_to<_From, _To>;
#   endif

    template <class... _Type>
    using tuple = std::tuple<_Type...>;

    template <size_t _Index, class _Tuple>
    using tuple_element_t = std::tuple_element_t<_Index, _Tuple>;

    template <size_t... _Vals>
    using index_sequence = std::index_sequence<_Vals...>;

    template <size_t _Size>
    using make_index_sequence = std::make_index_sequence<_Size>;
#endif // if defined(__CUDACC_RTC__)

    namespace detail {
        template <typename T>
        RT_DEVICE_FUNCTION RT_INLINE constexpr size_t getNumDwords() {
            return (sizeof(T) + 3) / 4;
        }

        template <typename... Types>
        RT_DEVICE_FUNCTION RT_INLINE constexpr size_t calcSumDwords() {
            return (0 + ... + getNumDwords<Types>());
        }
    }



#if !defined(__CUDA_ARCH__)
    struct PayloadType {
        OptixPayloadSemantics semantics[OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT];
        uint32_t numDwords;

        PayloadType() : numDwords(0) {
            for (uint32_t i = 0; i < OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT; ++i)
                semantics[i] = static_cast<OptixPayloadSemantics>(0);
        }

        OptixPayloadType getRawType() const {
            OptixPayloadType ret;
            ret.numPayloadValues = numDwords;
            ret.payloadSemantics = reinterpret_cast<const uint32_t*>(semantics);
            return ret;
        }
    };
#endif // if !defined(__CUDA_ARCH__)



    // ----------------------------------------------------------------
    // JP: ホスト・デバイス共有のクラス定義
    // EN: Definitions of Host-/Device-shared classes

    template <typename FuncType>
    class DirectCallableProgramID;

    template <typename ReturnType, typename... ArgTypes>
    class DirectCallableProgramID<ReturnType(ArgTypes...)> {
        uint32_t m_sbtIndex;

    public:
        RT_COMMON_FUNCTION RT_INLINE DirectCallableProgramID() {}
        RT_COMMON_FUNCTION RT_INLINE explicit DirectCallableProgramID(uint32_t sbtIndex) : m_sbtIndex(sbtIndex) {}
        RT_COMMON_FUNCTION RT_INLINE explicit operator uint32_t() const { return m_sbtIndex; }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        RT_DEVICE_FUNCTION RT_INLINE ReturnType operator()(const ArgTypes &... args) const {
            return optixDirectCall<ReturnType, ArgTypes...>(m_sbtIndex, args...);
        }
#endif
    };

    template <typename FuncType>
    class ContinuationCallableProgramID;

    template <typename ReturnType, typename... ArgTypes>
    class ContinuationCallableProgramID<ReturnType(ArgTypes...)> {
        uint32_t m_sbtIndex;

    public:
        RT_COMMON_FUNCTION RT_INLINE ContinuationCallableProgramID() {}
        RT_COMMON_FUNCTION RT_INLINE explicit ContinuationCallableProgramID(uint32_t sbtIndex) :
            m_sbtIndex(sbtIndex) {}
        RT_COMMON_FUNCTION RT_INLINE explicit operator uint32_t() const { return m_sbtIndex; }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        RT_DEVICE_FUNCTION RT_INLINE ReturnType operator()(const ArgTypes &... args) const {
            return optixContinuationCall<ReturnType, ArgTypes...>(m_sbtIndex, args...);
        }
#endif
    };



#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
#   if __cplusplus >= 202002L
    template <typename T>
    concept Has3D = requires(T v) {
        { v.x } -> convertible_to<float>;
        { v.y } -> convertible_to<float>;
        { v.z } -> convertible_to<float>;
    };
#       define OPTIXU_HAS3D_CONCEPT Has3D
#   else // if __cplusplus >= 202002L
#       define OPTIXU_HAS3D_CONCEPT typename
#   endif // if __cplusplus >= 202002L

    template <OPTIXU_HAS3D_CONCEPT T>
    RT_DEVICE_FUNCTION RT_INLINE float3 toNative(const T &v) {
        return make_float3(
            static_cast<float>(v.x),
            static_cast<float>(v.y),
            static_cast<float>(v.z));
    }
#endif // if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)



    template <typename... PayloadTypes>
    struct PayloadSignature {
        using Types = tuple<PayloadTypes...>;
        template <uint32_t index>
        using TypeAt = tuple_element_t<index, Types>;
        static constexpr uint32_t numParameters = sizeof...(PayloadTypes);
        static constexpr uint32_t numDwords =
            static_cast<uint32_t>(detail::calcSumDwords<PayloadTypes...>());
        static_assert(
            numDwords <= OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT,
            "Maximum number of payloads is "
            OPTIXU_TO_STRING(OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT)
            " in dwords.");
        static constexpr uint32_t _arraySize = numParameters > 0 ? numParameters : 1u;
        static constexpr uint32_t sizesInDwords[_arraySize] = {
            static_cast<uint32_t>(detail::getNumDwords<PayloadTypes>())...
        };

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        template <
            OptixPayloadTypeID payloadTypeID = OPTIX_PAYLOAD_TYPE_DEFAULT,
            OPTIXU_HAS3D_CONCEPT PosType, OPTIXU_HAS3D_CONCEPT DirType>
        RT_DEVICE_FUNCTION RT_INLINE static void trace(
            OptixTraversableHandle handle,
            const PosType &origin, const DirType &direction,
            float tmin, float tmax, float rayTime,
            OptixVisibilityMask visibilityMask, OptixRayFlags rayFlags,
            uint32_t SBToffset, uint32_t SBTstride, uint32_t missSBTIndex,
            PayloadTypes &... payloads);
        template <
            OptixPayloadTypeID payloadTypeID = OPTIX_PAYLOAD_TYPE_DEFAULT,
            OPTIXU_HAS3D_CONCEPT PosType, OPTIXU_HAS3D_CONCEPT DirType>
        RT_DEVICE_FUNCTION RT_INLINE static void traverse(
            OptixTraversableHandle handle,
            const PosType &origin, const DirType &direction,
            float tmin, float tmax, float rayTime,
            OptixVisibilityMask visibilityMask, OptixRayFlags rayFlags,
            uint32_t SBToffset, uint32_t SBTstride, uint32_t missSBTIndex,
            PayloadTypes &... payloads);
        template <OptixPayloadTypeID payloadTypeID = OPTIX_PAYLOAD_TYPE_DEFAULT>
        RT_DEVICE_FUNCTION RT_INLINE static void invoke(PayloadTypes &... payloads);
        RT_DEVICE_FUNCTION RT_INLINE static void get(PayloadTypes*... payloads);
        RT_DEVICE_FUNCTION RT_INLINE static void set(const PayloadTypes*... payloads);
        template <uint32_t index>
        RT_DEVICE_FUNCTION RT_INLINE static void getAt(TypeAt<index>* payload);
        template <uint32_t index>
        RT_DEVICE_FUNCTION RT_INLINE static void setAt(const TypeAt<index> &payload);
#endif
    };

    template <typename T, OptixPayloadSemantics _semantics>
    struct AnnotatedPayload {
        using Type = T;
        static constexpr OptixPayloadSemantics semantics = _semantics;
    };

    template <typename... AnnotatedPayloadTypes>
    struct AnnotatedPayloadSignature :
        public PayloadSignature<typename AnnotatedPayloadTypes::Type...> {
        using BaseSignature = PayloadSignature<typename AnnotatedPayloadTypes::Type...>;

        static constexpr OptixPayloadSemantics semantics[BaseSignature::_arraySize] = {
            AnnotatedPayloadTypes::semantics...
        };

#if !defined(__CUDA_ARCH__)
        static PayloadType getPayloadType() {
            PayloadType ret;
            ret.numDwords = BaseSignature::numDwords;
            uint32_t offset = 0;
            for (uint32_t varIdx = 0; varIdx < BaseSignature::numParameters; ++varIdx) {
                const uint32_t sizeInDwords = BaseSignature::sizesInDwords[varIdx];
                const OptixPayloadSemantics varSem = semantics[varIdx];
                for (uint32_t dwIdx = 0; dwIdx < sizeInDwords; ++dwIdx)
                    ret.semantics[offset + dwIdx] = varSem;
                offset += sizeInDwords;
            }

            return ret;
        }
#endif
    };

    template <typename... AttributeTypes>
    struct AttributeSignature {
        using Types = tuple<AttributeTypes...>;
        template <uint32_t index>
        using TypeAt = tuple_element_t<index, Types>;
        static constexpr uint32_t numParameters = sizeof...(AttributeTypes);
        static constexpr uint32_t numDwords =
            static_cast<uint32_t>(detail::calcSumDwords<AttributeTypes...>());
        static_assert(numDwords <= 8, "Maximum number of attributes is 8 dwords.");
        static constexpr uint32_t sizesInDwords[numParameters] = {
            static_cast<uint32_t>(detail::getNumDwords<AttributeTypes>())...
        };

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        RT_DEVICE_FUNCTION RT_INLINE static bool reportIntersection(
            float hitT, uint32_t hitKind,
            const AttributeTypes &... attributes);
        RT_DEVICE_FUNCTION RT_INLINE static void get(AttributeTypes*... attributes);
        RT_DEVICE_FUNCTION RT_INLINE static void getFromHitObject(AttributeTypes*... attributes);
        template <OPTIXU_HAS3D_CONCEPT PosType, OPTIXU_HAS3D_CONCEPT DirType>
        RT_DEVICE_FUNCTION RT_INLINE static void makeHitObject(
            OptixTraversableHandle handle,
            const PosType &origin, const DirType &direction,
            float tmin, float tmax, float rayTime,
            uint32_t SBToffset, uint32_t SBTstride, uint32_t instIdx,
            const OptixTraversableHandle* transforms, uint32_t numTransforms,
            uint32_t sbtGASIdx, uint32_t primIdx, uint32_t hitKind,
            const AttributeTypes &... attributes);
        template <OPTIXU_HAS3D_CONCEPT PosType, OPTIXU_HAS3D_CONCEPT DirType>
        RT_DEVICE_FUNCTION RT_INLINE static void makeHitObject(
            OptixTraversableHandle handle,
            const PosType &origin, const DirType &direction,
            float tmin, float tmax, float rayTime,
            uint32_t SBToffset, uint32_t SBTstride, uint32_t instIdx,
            uint32_t sbtGASIdx, uint32_t primIdx, uint32_t hitKind,
            const AttributeTypes &... attributes);
        template <OPTIXU_HAS3D_CONCEPT PosType, OPTIXU_HAS3D_CONCEPT DirType>
        RT_DEVICE_FUNCTION RT_INLINE static void makeHitObjectWithRecord(
            OptixTraversableHandle handle,
            const PosType &origin, const DirType &direction,
            float tmin, float tmax, float rayTime,
            uint32_t sbtRecordIndex, uint32_t instIdx,
            const OptixTraversableHandle* transforms, uint32_t numTransforms,
            uint32_t sbtGASIdx, uint32_t primIdx, uint32_t hitKind,
            const AttributeTypes &... attributes);
#endif
    };

    template <typename... ExceptionDetailTypes>
    struct ExceptionDetailSignature {
        using Types = tuple<ExceptionDetailTypes...>;
        template <uint32_t index>
        using TypeAt = tuple_element_t<index, Types>;
        static constexpr uint32_t numParameters = sizeof...(ExceptionDetailTypes);
        static constexpr uint32_t numDwords =
            static_cast<uint32_t>(detail::calcSumDwords<ExceptionDetailTypes...>());
        static_assert(numDwords <= 8, "Maximum number of exception details is 8 dwords.");
        static constexpr uint32_t sizesInDwords[numParameters] = {
            static_cast<uint32_t>(detail::getNumDwords<ExceptionDetailTypes>())...
        };

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        RT_DEVICE_FUNCTION RT_INLINE static void throwException(
            int32_t exceptionCode,
            const ExceptionDetailTypes &... exDetails);
        RT_DEVICE_FUNCTION RT_INLINE static void get(ExceptionDetailTypes*... exDetails);
#endif
    };

    // END: Definitions of Host-/Device-shared classes
    // ----------------------------------------------------------------




    // ----------------------------------------------------------------
    // JP: デバイス関数のラッパー
    // EN: Device-side function wrappers
#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

    namespace detail {
        template <uint32_t index, size_t N>
        RT_DEVICE_FUNCTION RT_INLINE constexpr uint32_t calcOffset(const uint32_t (&sizes)[N]) {
            if constexpr (index == 0)
                return 0;
            else
                return sizes[index - 1] + calcOffset<index - 1>(sizes);
        }

        template <uint32_t start, typename HeadType, typename... TailTypes>
        RT_DEVICE_FUNCTION RT_INLINE void packToUInts(
            uint32_t* v, const HeadType &head, const TailTypes &... tails)
        {
            static_assert(sizeof(HeadType) % sizeof(uint32_t) == 0,
                          "Value type of size not multiple of Dword is not supported.");
            constexpr uint32_t numDwords = sizeof(HeadType) / sizeof(uint32_t);
#pragma unroll
            for (int i = 0; i < numDwords; ++i)
                v[start + i] = *(reinterpret_cast<const uint32_t*>(&head) + i);
            if constexpr (sizeof...(tails) > 0)
                packToUInts<start + numDwords>(v, tails...);
        }

        template <typename Func, typename Type, uint32_t offsetInDst, uint32_t srcSlot>
        RT_DEVICE_FUNCTION RT_INLINE void getValue(
            Type* value)
        {
            if (!value) // hope calls for this function are removed when value is compile-time nullptr.
                return;
            *(reinterpret_cast<uint32_t*>(value) + offsetInDst) = Func::template get<srcSlot>();
            if constexpr (offsetInDst + 1 < getNumDwords<Type>())
                getValue<Func, Type, offsetInDst + 1, srcSlot + 1>(value);
        }

        template <typename Func, uint32_t srcStartSlot, typename HeadType, typename... TailTypes>
        RT_DEVICE_FUNCTION RT_INLINE void getValues(
            HeadType* head, TailTypes*... tails)
        {
            static_assert(sizeof(HeadType) % sizeof(uint32_t) == 0,
                          "Value type of size not multiple of Dword is not supported.");
            getValue<Func, HeadType, 0, srcStartSlot>(head);
            if constexpr (sizeof...(tails) > 0)
                getValues<Func, srcStartSlot + getNumDwords<HeadType>()>(tails...);
        }

        template <typename Func, typename Type, uint32_t offsetInSrc, uint32_t dstSlot>
        RT_DEVICE_FUNCTION RT_INLINE void setValue(
            const Type* value)
        {
            if (!value) // hope calls for this function are removed when value is compile-time nullptr.
                return;
            Func::set<dstSlot>(*(reinterpret_cast<const uint32_t*>(value) + offsetInSrc));
            if constexpr (offsetInSrc + 1 < getNumDwords<Type>())
                setValue<Func, Type, offsetInSrc + 1, dstSlot + 1>(value);
        }

        template <typename Func, uint32_t dstStartSlot, typename HeadType, typename... TailTypes>
        RT_DEVICE_FUNCTION RT_INLINE void setValues(
            const HeadType* head, const TailTypes*... tails)
        {
            static_assert(sizeof(HeadType) % sizeof(uint32_t) == 0,
                          "Value type of size not multiple of Dword is not supported.");
            setValue<Func, HeadType, 0, dstStartSlot>(head);
            if constexpr (sizeof...(tails) > 0)
                setValues<Func, dstStartSlot + getNumDwords<HeadType>()>(tails...);
        }

        template <uint32_t startSlot, typename HeadType, typename... TailTypes>
        RT_DEVICE_FUNCTION RT_INLINE void traceSetPayloads(
            uint32_t** p, HeadType &headPayload, TailTypes &... tailPayloads)
        {
            static_assert(sizeof(HeadType) % sizeof(uint32_t) == 0,
                          "Payload type of size not multiple of Dword is not supported.");
            constexpr uint32_t numDwords = getNumDwords<HeadType>();
#pragma unroll
            for (int i = 0; i < numDwords; ++i)
                p[startSlot + i] = reinterpret_cast<uint32_t*>(&headPayload) + i;
            if constexpr (sizeof...(tailPayloads) > 0)
                traceSetPayloads<startSlot + numDwords>(p, tailPayloads...);
        }

        template <bool withInvoke, OptixPayloadTypeID payloadTypeID, size_t... I>
        RT_DEVICE_FUNCTION RT_INLINE void traverse(
            OptixTraversableHandle handle,
            const float3 &origin, const float3 &direction,
            float tmin, float tmax, float rayTime,
            OptixVisibilityMask visibilityMask, OptixRayFlags rayFlags,
            uint32_t SBToffset, uint32_t SBTstride, uint32_t missSBTIndex,
            uint32_t* const* payloads,
            index_sequence<I...>)
        {
            if constexpr (withInvoke) {
                optixTrace(
                    payloadTypeID,
                    handle,
                    origin, direction,
                    tmin, tmax, rayTime,
                    visibilityMask, rayFlags,
                    SBToffset, SBTstride, missSBTIndex,
                    *payloads[I]...);
            }
            else {
                optixTraverse(
                    payloadTypeID,
                    handle,
                    origin, direction,
                    tmin, tmax, rayTime,
                    visibilityMask, rayFlags,
                    SBToffset, SBTstride, missSBTIndex,
                    *payloads[I]...);
            }
        }

        template <OptixPayloadTypeID payloadTypeID, size_t... I>
        RT_DEVICE_FUNCTION RT_INLINE void invoke(
            uint32_t* const* payloads, index_sequence<I...>)
        {
            optixInvoke(payloadTypeID, *payloads[I]...);
        }

        template <size_t... I>
        RT_DEVICE_FUNCTION RT_INLINE bool reportIntersection(
            float hitT, uint32_t hitKind, const uint32_t* attributes,
            index_sequence<I...>)
        {
            return optixReportIntersection(hitT, hitKind, attributes[I]...);
        }

        template <size_t... I>
        RT_DEVICE_FUNCTION RT_INLINE void makeHitObject(
            OptixTraversableHandle handle,
            const float3 &origin, const float3 &direction,
            float tmin, float tmax, float rayTime,
            uint32_t SBToffset, uint32_t SBTstride, uint32_t instIdx,
            const OptixTraversableHandle* transforms, uint32_t numTransforms,
            uint32_t sbtGASIdx, uint32_t primIdx, uint32_t hitKind,
            const uint32_t* attributes,
            index_sequence<I...>)
        {
            optixMakeHitObject(
                handle,
                origin, direction,
                tmin, tmax, rayTime,
                SBToffset, SBTstride, instIdx,
                transforms, numTransforms,
                sbtGASIdx, primIdx, hitKind,
                attributes[I]...);
        }

        template <size_t... I>
        RT_DEVICE_FUNCTION RT_INLINE void makeHitObject(
            OptixTraversableHandle handle,
            const float3 &origin, const float3 &direction,
            float tmin, float tmax, float rayTime,
            uint32_t SBToffset, uint32_t SBTstride, uint32_t instIdx,
            uint32_t sbtGASIdx, uint32_t primIdx, uint32_t hitKind,
            const uint32_t* attributes,
            index_sequence<I...>)
        {
            optixMakeHitObject(
                handle,
                origin, direction,
                tmin, tmax, rayTime,
                SBToffset, SBTstride, instIdx,
                sbtGASIdx, primIdx, hitKind,
                attributes[I]...);
        }

        template <size_t... I>
        RT_DEVICE_FUNCTION RT_INLINE void makeHitObject(
            OptixTraversableHandle handle,
            const float3 &origin, const float3 &direction,
            float tmin, float tmax, float rayTime,
            uint32_t sbtRecordIndex, uint32_t instIdx,
            const OptixTraversableHandle* transforms, uint32_t numTransforms,
            uint32_t sbtGASIdx, uint32_t primIdx, uint32_t hitKind,
            const uint32_t* attributes,
            index_sequence<I...>)
        {
            optixMakeHitObjectWithRecord(
                handle,
                origin, direction,
                tmin, tmax, rayTime,
                sbtRecordIndex, instIdx,
                transforms, numTransforms,
                sbtGASIdx, primIdx, hitKind,
                attributes[I]...);
        }

        template <size_t... I>
        RT_DEVICE_FUNCTION RT_INLINE void throwException(
            int32_t exceptionCode, const uint32_t* exDetails,
            index_sequence<I...>)
        {
            optixThrowException(exceptionCode, exDetails[I]...);
        }

        struct PayloadFunc {
            template <uint32_t index>
            RT_DEVICE_FUNCTION RT_INLINE static uint32_t get() {
#define OPTIXU_INTRINSIC_GET_PAYLOAD(Index) \
    if constexpr (index == Index) return optixGetPayload_##Index()
                OPTIXU_INTRINSIC_GET_PAYLOAD(0);
                OPTIXU_INTRINSIC_GET_PAYLOAD(1);
                OPTIXU_INTRINSIC_GET_PAYLOAD(2);
                OPTIXU_INTRINSIC_GET_PAYLOAD(3);
                OPTIXU_INTRINSIC_GET_PAYLOAD(4);
                OPTIXU_INTRINSIC_GET_PAYLOAD(5);
                OPTIXU_INTRINSIC_GET_PAYLOAD(6);
                OPTIXU_INTRINSIC_GET_PAYLOAD(7);
                OPTIXU_INTRINSIC_GET_PAYLOAD(8);
                OPTIXU_INTRINSIC_GET_PAYLOAD(9);
                OPTIXU_INTRINSIC_GET_PAYLOAD(10);
                OPTIXU_INTRINSIC_GET_PAYLOAD(11);
                OPTIXU_INTRINSIC_GET_PAYLOAD(12);
                OPTIXU_INTRINSIC_GET_PAYLOAD(13);
                OPTIXU_INTRINSIC_GET_PAYLOAD(14);
                OPTIXU_INTRINSIC_GET_PAYLOAD(15);
                OPTIXU_INTRINSIC_GET_PAYLOAD(16);
                OPTIXU_INTRINSIC_GET_PAYLOAD(17);
                OPTIXU_INTRINSIC_GET_PAYLOAD(18);
                OPTIXU_INTRINSIC_GET_PAYLOAD(19);
                OPTIXU_INTRINSIC_GET_PAYLOAD(20);
                OPTIXU_INTRINSIC_GET_PAYLOAD(21);
                OPTIXU_INTRINSIC_GET_PAYLOAD(22);
                OPTIXU_INTRINSIC_GET_PAYLOAD(23);
                OPTIXU_INTRINSIC_GET_PAYLOAD(24);
                OPTIXU_INTRINSIC_GET_PAYLOAD(25);
                OPTIXU_INTRINSIC_GET_PAYLOAD(26);
                OPTIXU_INTRINSIC_GET_PAYLOAD(27);
                OPTIXU_INTRINSIC_GET_PAYLOAD(28);
                OPTIXU_INTRINSIC_GET_PAYLOAD(29);
                OPTIXU_INTRINSIC_GET_PAYLOAD(30);
                OPTIXU_INTRINSIC_GET_PAYLOAD(31);
#undef OPTIXU_INTRINSIC_GET_PAYLOAD
                return 0;
            }

            template <uint32_t index>
            RT_DEVICE_FUNCTION RT_INLINE static void set(uint32_t p) {
#define OPTIXU_INTRINSIC_SET_PAYLOAD(Index) \
    if constexpr (index == Index) optixSetPayload_ ##Index(p)
                OPTIXU_INTRINSIC_SET_PAYLOAD(0);
                OPTIXU_INTRINSIC_SET_PAYLOAD(1);
                OPTIXU_INTRINSIC_SET_PAYLOAD(2);
                OPTIXU_INTRINSIC_SET_PAYLOAD(3);
                OPTIXU_INTRINSIC_SET_PAYLOAD(4);
                OPTIXU_INTRINSIC_SET_PAYLOAD(5);
                OPTIXU_INTRINSIC_SET_PAYLOAD(6);
                OPTIXU_INTRINSIC_SET_PAYLOAD(7);
                OPTIXU_INTRINSIC_SET_PAYLOAD(8);
                OPTIXU_INTRINSIC_SET_PAYLOAD(9);
                OPTIXU_INTRINSIC_SET_PAYLOAD(10);
                OPTIXU_INTRINSIC_SET_PAYLOAD(11);
                OPTIXU_INTRINSIC_SET_PAYLOAD(12);
                OPTIXU_INTRINSIC_SET_PAYLOAD(13);
                OPTIXU_INTRINSIC_SET_PAYLOAD(14);
                OPTIXU_INTRINSIC_SET_PAYLOAD(15);
                OPTIXU_INTRINSIC_SET_PAYLOAD(16);
                OPTIXU_INTRINSIC_SET_PAYLOAD(17);
                OPTIXU_INTRINSIC_SET_PAYLOAD(18);
                OPTIXU_INTRINSIC_SET_PAYLOAD(19);
                OPTIXU_INTRINSIC_SET_PAYLOAD(20);
                OPTIXU_INTRINSIC_SET_PAYLOAD(21);
                OPTIXU_INTRINSIC_SET_PAYLOAD(22);
                OPTIXU_INTRINSIC_SET_PAYLOAD(23);
                OPTIXU_INTRINSIC_SET_PAYLOAD(24);
                OPTIXU_INTRINSIC_SET_PAYLOAD(25);
                OPTIXU_INTRINSIC_SET_PAYLOAD(26);
                OPTIXU_INTRINSIC_SET_PAYLOAD(27);
                OPTIXU_INTRINSIC_SET_PAYLOAD(28);
                OPTIXU_INTRINSIC_SET_PAYLOAD(29);
                OPTIXU_INTRINSIC_SET_PAYLOAD(30);
                OPTIXU_INTRINSIC_SET_PAYLOAD(31);
#undef OPTIXU_INTRINSIC_SET_PAYLOAD
            }
        };

        struct AttributeFunc {
            template <uint32_t index>
            RT_DEVICE_FUNCTION RT_INLINE static uint32_t get() {
#define OPTIXU_INTRINSIC_GET_ATTRIBUTE(Index) \
    if constexpr (index == Index) return optixGetAttribute_##Index()
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(0);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(1);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(2);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(3);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(4);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(5);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(6);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(7);
#undef OPTIXU_INTRINSIC_GET_ATTRIBUTE
                return 0;
            }
        };

        struct HitObjectAttributeFunc {
            template <uint32_t index>
            RT_DEVICE_FUNCTION RT_INLINE static uint32_t get() {
#define OPTIXU_INTRINSIC_GET_ATTRIBUTE(Index) \
    if constexpr (index == Index) return optixHitObjectGetAttribute_##Index()
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(0);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(1);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(2);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(3);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(4);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(5);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(6);
                OPTIXU_INTRINSIC_GET_ATTRIBUTE(7);
#undef OPTIXU_INTRINSIC_GET_ATTRIBUTE
                return 0;
            }
        };

        struct ExceptionDetailFunc {
            template <uint32_t index>
            RT_DEVICE_FUNCTION RT_INLINE static uint32_t get() {
#define OPTIXU_INTRINSIC_GET_EXCEPTION_DETAIL(Index) \
    if constexpr (index == Index) return optixGetExceptionDetail_##Index()
                OPTIXU_INTRINSIC_GET_EXCEPTION_DETAIL(0);
                OPTIXU_INTRINSIC_GET_EXCEPTION_DETAIL(1);
                OPTIXU_INTRINSIC_GET_EXCEPTION_DETAIL(2);
                OPTIXU_INTRINSIC_GET_EXCEPTION_DETAIL(3);
                OPTIXU_INTRINSIC_GET_EXCEPTION_DETAIL(4);
                OPTIXU_INTRINSIC_GET_EXCEPTION_DETAIL(5);
                OPTIXU_INTRINSIC_GET_EXCEPTION_DETAIL(6);
                OPTIXU_INTRINSIC_GET_EXCEPTION_DETAIL(7);
#undef OPTIXU_INTRINSIC_GET_EXCEPTION_DETAIL
                return 0;
            }
        };
    }

    template <typename... PayloadTypes>
    template <OptixPayloadTypeID payloadTypeID, OPTIXU_HAS3D_CONCEPT PosType, OPTIXU_HAS3D_CONCEPT DirType>
    RT_DEVICE_FUNCTION RT_INLINE void PayloadSignature<PayloadTypes...>::
        trace(
            OptixTraversableHandle handle,
            const PosType &origin, const DirType &direction,
            float tmin, float tmax, float rayTime,
            OptixVisibilityMask visibilityMask, OptixRayFlags rayFlags,
            uint32_t SBToffset, uint32_t SBTstride, uint32_t missSBTIndex,
            PayloadTypes &... payloads)
    {
        uint32_t* p[numDwords > 0 ? numDwords : 1];
        if constexpr (numDwords > 0)
            detail::traceSetPayloads<0>(p, payloads...);
        detail::traverse<true, payloadTypeID>(
            handle,
            toNative(origin), toNative(direction),
            tmin, tmax, rayTime,
            visibilityMask, rayFlags,
            SBToffset, SBTstride, missSBTIndex,
            p, make_index_sequence<numDwords>{});
    }

    template <typename... PayloadTypes>
    template <OptixPayloadTypeID payloadTypeID, OPTIXU_HAS3D_CONCEPT PosType, OPTIXU_HAS3D_CONCEPT DirType>
    RT_DEVICE_FUNCTION RT_INLINE void PayloadSignature<PayloadTypes...>::
        traverse(
            OptixTraversableHandle handle,
            const PosType &origin, const DirType &direction,
            float tmin, float tmax, float rayTime,
            OptixVisibilityMask visibilityMask, OptixRayFlags rayFlags,
            uint32_t SBToffset, uint32_t SBTstride, uint32_t missSBTIndex,
            PayloadTypes &... payloads)
    {
        uint32_t* p[numDwords > 0 ? numDwords : 1];
        if constexpr (numDwords > 0)
            detail::traceSetPayloads<0>(p, payloads...);
        detail::traverse<false, payloadTypeID>(
            handle,
            toNative(origin), toNative(direction),
            tmin, tmax, rayTime,
            visibilityMask, rayFlags,
            SBToffset, SBTstride, missSBTIndex,
            p, make_index_sequence<numDwords>{});
    }

    template <typename... PayloadTypes>
    template <OptixPayloadTypeID payloadTypeID>
    RT_DEVICE_FUNCTION RT_INLINE void PayloadSignature<PayloadTypes...>::
        invoke(PayloadTypes &... payloads)
    {
        uint32_t* p[numDwords > 0 ? numDwords : 1];
        if constexpr (numDwords > 0)
            detail::traceSetPayloads<0>(p, payloads...);
        detail::invoke<payloadTypeID>(p, make_index_sequence<numDwords>{});
    }

    template <typename... PayloadTypes>
    RT_DEVICE_FUNCTION RT_INLINE void PayloadSignature<PayloadTypes...>::
        get(PayloadTypes*... payloads)
    {
        static_assert(numDwords > 0, "Calling this function for this signature has no effect.");
        if constexpr (numDwords > 0)
            detail::getValues<detail::PayloadFunc, 0>(payloads...);
    }

    template <typename... PayloadTypes>
    RT_DEVICE_FUNCTION RT_INLINE void PayloadSignature<PayloadTypes...>::
        set(const PayloadTypes*... payloads)
    {
        static_assert(numDwords > 0, "Calling this function for this signature has no effect.");
        if constexpr (numDwords > 0)
            detail::setValues<detail::PayloadFunc, 0>(payloads...);
    }

    template <typename... PayloadTypes>
    template <uint32_t index>
    RT_DEVICE_FUNCTION RT_INLINE void PayloadSignature<PayloadTypes...>::
        getAt(TypeAt<index>* payload)
    {
        constexpr uint32_t offsetInDwords = detail::calcOffset<index>(sizesInDwords);
        detail::getValue<detail::PayloadFunc, TypeAt<index>, 0, offsetInDwords>(payload);
    }

    template <typename... PayloadTypes>
    template <uint32_t index>
    RT_DEVICE_FUNCTION RT_INLINE void PayloadSignature<PayloadTypes...>::
        setAt(const TypeAt<index> &payload)
    {
        constexpr uint32_t offsetInDwords = detail::calcOffset<index>(sizesInDwords);
        detail::setValue<detail::PayloadFunc, TypeAt<index>, 0, offsetInDwords>(&payload);
    }



    template <typename... AttributeTypes>
    RT_DEVICE_FUNCTION RT_INLINE bool AttributeSignature<AttributeTypes...>::
        reportIntersection(
            float hitT, uint32_t hitKind,
            const AttributeTypes &... attributes)
    {
        uint32_t a[numDwords > 0 ? numDwords : 1];
        if constexpr (numDwords > 0)
            detail::packToUInts<0>(a, attributes...);
        return detail::reportIntersection(hitT, hitKind, a, make_index_sequence<numDwords>{});
    }

    template <typename... AttributeTypes>
    RT_DEVICE_FUNCTION RT_INLINE void AttributeSignature<AttributeTypes...>::
        get(AttributeTypes*... attributes)
    {
        static_assert(numDwords > 0, "Calling this function for this signature has no effect.");
        if constexpr (numDwords > 0)
            detail::getValues<detail::AttributeFunc, 0>(attributes...);
    }

    template <typename... AttributeTypes>
    RT_DEVICE_FUNCTION RT_INLINE void AttributeSignature<AttributeTypes...>::
        getFromHitObject(AttributeTypes*... attributes)
    {
        static_assert(numDwords > 0, "Calling this function for this signature has no effect.");
        if constexpr (numDwords > 0)
            detail::getValues<detail::HitObjectAttributeFunc, 0>(attributes...);
    }

    template <typename... AttributeTypes>
    template <OPTIXU_HAS3D_CONCEPT PosType, OPTIXU_HAS3D_CONCEPT DirType>
    RT_DEVICE_FUNCTION RT_INLINE void AttributeSignature<AttributeTypes...>::
        makeHitObject(
            OptixTraversableHandle handle,
            const PosType &origin, const DirType &direction,
            float tmin, float tmax, float rayTime,
            uint32_t SBToffset, uint32_t SBTstride, uint32_t instIdx,
            const OptixTraversableHandle* transforms, uint32_t numTransforms,
            uint32_t sbtGASIdx, uint32_t primIdx, uint32_t hitKind,
            const AttributeTypes &... attributes)
    {
        uint32_t a[numDwords > 0 ? numDwords : 1];
        if constexpr (numDwords > 0)
            detail::packToUInts<0>(a, attributes...);
        detail::makeHitObject(
            handle,
            origin, direction,
            tmin, tmax, rayTime,
            SBToffset, SBTstride, instIdx,
            transforms, numTransforms,
            sbtGASIdx, primIdx, hitKind,
            a, make_index_sequence<numDwords>{});
    }

    template <typename... AttributeTypes>
    template <OPTIXU_HAS3D_CONCEPT PosType, OPTIXU_HAS3D_CONCEPT DirType>
    RT_DEVICE_FUNCTION RT_INLINE void AttributeSignature<AttributeTypes...>::
        makeHitObject(
            OptixTraversableHandle handle,
            const PosType &origin, const DirType &direction,
            float tmin, float tmax, float rayTime,
            uint32_t SBToffset, uint32_t SBTstride, uint32_t instIdx,
            uint32_t sbtGASIdx, uint32_t primIdx, uint32_t hitKind,
            const AttributeTypes &... attributes)
    {
        uint32_t a[numDwords > 0 ? numDwords : 1];
        if constexpr (numDwords > 0)
            detail::packToUInts<0>(a, attributes...);
        detail::makeHitObject(
            handle,
            origin, direction,
            tmin, tmax, rayTime,
            SBToffset, SBTstride, instIdx,
            sbtGASIdx, primIdx, hitKind,
            a, make_index_sequence<numDwords>{});
    }

    template <typename... AttributeTypes>
    template <OPTIXU_HAS3D_CONCEPT PosType, OPTIXU_HAS3D_CONCEPT DirType>
    RT_DEVICE_FUNCTION RT_INLINE void AttributeSignature<AttributeTypes...>::
        makeHitObjectWithRecord(
            OptixTraversableHandle handle,
            const PosType &origin, const DirType &direction,
            float tmin, float tmax, float rayTime,
            uint32_t sbtRecordIndex, uint32_t instIdx,
            const OptixTraversableHandle* transforms, uint32_t numTransforms,
            uint32_t sbtGASIdx, uint32_t primIdx, uint32_t hitKind,
            const AttributeTypes &... attributes)
    {
        uint32_t a[numDwords > 0 ? numDwords : 1];
        if constexpr (numDwords > 0)
            detail::packToUInts<0>(a, attributes...);
        detail::makeHitObject(
            handle,
            origin, direction,
            tmin, tmax, rayTime,
            sbtRecordIndex, instIdx,
            transforms, numTransforms,
            sbtGASIdx, primIdx, hitKind,
            a, make_index_sequence<numDwords>{});
    }



    template <typename... ExceptionDetailTypes>
    RT_DEVICE_FUNCTION RT_INLINE void ExceptionDetailSignature<ExceptionDetailTypes...>::
        throwException(
            int32_t exceptionCode,
            const ExceptionDetailTypes &... exDetails)
    {
        uint32_t ed[numDwords > 0 ? numDwords : 1];
        if constexpr (numDwords > 0)
            detail::packToUInts<0>(ed, exDetails...);
        detail::throwException(exceptionCode, ed, make_index_sequence<numDwords>{});
    }

    template <typename... ExceptionDetailTypes>
    RT_DEVICE_FUNCTION RT_INLINE void ExceptionDetailSignature<ExceptionDetailTypes...>::
        get(ExceptionDetailTypes*... exDetails)
    {
        static_assert(numDwords > 0, "Calling this function for this signature has no effect.");
        if constexpr (numDwords > 0)
            detail::getValues<detail::ExceptionDetailFunc, 0>(exDetails...);
    }

#undef OPTIXU_HAS3D_CONCEPT

#endif // #if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
    // END: Device-side function wrappers
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: ホスト側API
    // EN: Host-side API
#if !defined(__CUDA_ARCH__)
    /*

    Context --+-- Pipeline --+-- Module
              |              |
              |              +-- Program
              |              |
              |              +-- HitProgramGroup
              |              |
              |              +-- CallableProgramGroup
              |
              +-- Material
              |
              |
              |
              +-- Scene    --+-- IAS
              |              |
              |              +-- Instance
              |              |
              |              +-- Transform
              |              |
              |              +-- GAS
              |              |
              |              +-- GeomInst
              |              |
              |              +-- OMMArray
              |              |
              |              +-- DMMArray
              |
              +-- Denoiser

    JP: 
    EN: 



    Hit Group Shader Binding Table Layout
    | GAS 0 - MS 0 | GAS 0 - MS 1 | ... | GAS 0 - MS * | GAS 1 - MS 0 | GAS 1 - MS 1 | ...
    GAS: Geometry Acceleration Structure
    MS: Material Set

    Per-GAS, Per-Material Set SBT Layout
    | GAS * - MS *                               |
    | GeomInst 0 | GeomInst 1 | ... | GeomInst * |
    GeomInst: Geometry Instance

    Per-Geometry Instance SBT Layout
    | GeomInst *                                 |
    | Material 0 | Material 1 | ... | Material * |

    Per-Material SBT Layout
    | Material *                                 |
    | Ray Type 0 | Ray Type 1 | ... | Ray Type * |
    | SBT Record | SBT Record |     | SBT Record |

    SBT Record
    <-- SBT Record Stride (Globally Common) --------------------------->
    | Header | GAS       | GAS Child | GeomInst  | Material  | Padding |
    |        | User Data | User Data | User Data | User Data |         |
             ^
             |
             optixGetSbtDataPointer()

    JP: CH/AH/ISプログラムにてoptixGetSbtDataPointer()で取得できるポインターの位置に
        GeometryInstanceAccelerationStructureのsetUserData(), setChildUserData(),
        GeometryInstanceのsetUserData(), MaterialのsetUserData()
        で設定したデータが順番に並んでいる(各データの相対的な開始位置は指定したアラインメントに従う)。
        各データの開始位置は前方のデータのサイズによって変わるので、例えば同じGASに属していても
        GASの子ごとのデータサイズが異なればGeometryInstanceのデータの開始位置は異なる可能性があることに注意。
    EN: Data set by each of
        GeometryInstanceAccelerationStructure's setUserData() and setChildUserData(),
        GeometryInstance's setUserData(), Material's setUserData()
        line up in the order (Each relative offset follows the specified alignment)
        at the position pointed by optixGetSbtDataPointer() called in CH/AH/IS programs.
        Note that the start position of each data changes depending on the sizes of forward data.
        Therefore for example, the start positions of GeometryInstance's data are different
        if data sizes of GAS children are different even if those belong to the same GAS.

    */

#define OPTIXU_PREPROCESS_OBJECTS() \
    OPTIXU_PREPROCESS_OBJECT(Material); \
    OPTIXU_PREPROCESS_OBJECT(Scene); \
    OPTIXU_PREPROCESS_OBJECT(OpacityMicroMapArray); \
    OPTIXU_PREPROCESS_OBJECT(DisplacementMicroMapArray); \
    OPTIXU_PREPROCESS_OBJECT(GeometryInstance); \
    OPTIXU_PREPROCESS_OBJECT(GeometryAccelerationStructure); \
    OPTIXU_PREPROCESS_OBJECT(Transform); \
    OPTIXU_PREPROCESS_OBJECT(Instance); \
    OPTIXU_PREPROCESS_OBJECT(InstanceAccelerationStructure); \
    OPTIXU_PREPROCESS_OBJECT(Pipeline); \
    OPTIXU_PREPROCESS_OBJECT(Module); \
    OPTIXU_PREPROCESS_OBJECT(Program); \
    OPTIXU_PREPROCESS_OBJECT(HitProgramGroup); \
    OPTIXU_PREPROCESS_OBJECT(CallableProgramGroup); \
    OPTIXU_PREPROCESS_OBJECT(Denoiser);

    // Forward Declarations
#define OPTIXU_PREPROCESS_OBJECT(Type) class Type
    OPTIXU_PREPROCESS_OBJECTS();
#undef OPTIXU_PREPROCESS_OBJECT

    enum class IndexSize {
        k1Byte = 0,
        k2Bytes,
        k4Bytes,
        None,
    };

    enum class GeometryType {
        Triangles = 0,
        LinearSegments,
        QuadraticBSplines,
        FlatQuadraticBSplines,
        CubicBSplines,
        CatmullRomSplines,
        CubicBezier,
        Spheres,
        CustomPrimitives,
    };

    enum class ASTradeoff {
        Default = 0,
        PreferFastTrace,
        PreferFastBuild,
    };

    enum class TransformType {
        MatrixMotion = 0,
        SRTMotion,
        Static,
        Invalid
    };

    enum class ChildType {
        GAS = 0,
        IAS,
        Transform,
        Invalid
    };

    class BufferView {
        CUdeviceptr m_devicePtr;
        size_t m_numElements;
        uint32_t m_stride;

    public:
        BufferView() :
            m_devicePtr(0),
            m_numElements(0), m_stride(0) {}
        BufferView(CUdeviceptr devicePtr, size_t numElements, uint32_t stride) :
            m_devicePtr(devicePtr),
            m_numElements(numElements), m_stride(stride) {}

        CUdeviceptr getCUdeviceptr() const { return m_devicePtr; }
        size_t numElements() const { return m_numElements; }
        uint32_t stride() const { return m_stride; }
        size_t sizeInBytes() const { return m_numElements * m_stride; }

        bool isValid() const {
            return m_devicePtr != 0;
        }

        bool operator==(const BufferView &b) const {
            return (m_devicePtr == b.m_devicePtr &&
                    m_numElements == b.m_numElements &&
                    m_stride == b.m_stride);
        }
    };

    template <typename... Types>
    constexpr size_t calcSumDwords() {
        return detail::calcSumDwords<Types...>();
    }

    template <typename BaseType>
    class TypedBool {
        bool m_value;

    public:
        struct Bool {
            bool b;
        };

        constexpr TypedBool(const TypedBool &b) : m_value(b.m_value) {}
        constexpr TypedBool(Bool b) : m_value(b.b) {}
        constexpr explicit TypedBool(bool b) : m_value(b) {}

        constexpr operator bool() const {
            return m_value;
        }
        constexpr bool operator==(const TypedBool &r) const {
            return m_value == r.m_value;
        }
        constexpr bool operator!=(const TypedBool &r) const {
            return m_value != r.m_value;
        }

        static constexpr Bool True{ true };
        static constexpr Bool False{ false };
        static constexpr Bool Yes{ true };
        static constexpr Bool No{ false };
    };



#define OPTIXU_DECLARE_TYPED_BOOL(Name) \
    using Name = TypedBool<struct _ ## Name>

    OPTIXU_DECLARE_TYPED_BOOL(EnableValidation);
    OPTIXU_DECLARE_TYPED_BOOL(GuideAlbedo);
    OPTIXU_DECLARE_TYPED_BOOL(GuideNormal);
    OPTIXU_DECLARE_TYPED_BOOL(UseSingleRadius);
    OPTIXU_DECLARE_TYPED_BOOL(AllowUpdate);
    OPTIXU_DECLARE_TYPED_BOOL(AllowCompaction);
    OPTIXU_DECLARE_TYPED_BOOL(AllowRandomVertexAccess);
    OPTIXU_DECLARE_TYPED_BOOL(AllowOpacityMicroMapUpdate);
    OPTIXU_DECLARE_TYPED_BOOL(AllowDisableOpacityMicroMaps);
    OPTIXU_DECLARE_TYPED_BOOL(AllowRandomInstanceAccess);
    OPTIXU_DECLARE_TYPED_BOOL(UseMotionBlur);
    OPTIXU_DECLARE_TYPED_BOOL(UseOpacityMicroMaps);
    OPTIXU_DECLARE_TYPED_BOOL(IsFirstFrame);

#undef OPTIXU_DECLARE_TYPED_BOOL

#define OPTIXU_EN_PRM(Type, Name, Value) Type Name = Type::Value



    class Context {
    public:
        class Priv;
    private:
        Priv* m = nullptr;

    public:
        [[nodiscard]]
        static Context create(
            CUcontext cuContext,
            uint32_t logLevel = 4,
            OPTIXU_EN_PRM(EnableValidation, enableValidation, No));
        void destroy();

        CUcontext getCUcontext() const;

        void setLogCallback(OptixLogCallback callback, void* callbackData, uint32_t logLevel) const;

        [[nodiscard]]
        Pipeline createPipeline() const;
        [[nodiscard]]
        Material createMaterial() const;
        [[nodiscard]]
        Scene createScene() const;
        [[nodiscard]]
        Denoiser createDenoiser(
            OptixDenoiserModelKind modelKind,
            GuideAlbedo guideAlbedo,
            GuideNormal guideNormal,
            OptixDenoiserAlphaMode alphaMode) const;

        uint32_t getRTCoreVersion() const;
        uint32_t getShaderExecutionReorderingFlags() const;

        operator bool() const { return m; }
        bool operator==(const Context &r) const { return m == r.m; }
        bool operator!=(const Context &r) const { return m != r.m; }
        bool operator<(const Context &r) const {
            return m < r.m;
        }

        void setName(const std::string &name) const;
        const char* getName() const;
    };



    template <typename T>
    class Object {
    public:
        class Priv;

    protected:
        Priv* m = nullptr;

    public:
        explicit operator bool() const {
            return m;
        }
        bool operator==(const T &r) const {
            return m == r.m;
        }
        bool operator!=(const T &r) const {
            return m != r.m;
        }
        bool operator<(const T &r) const {
            return m < r.m;
        }

        Context getContext() const;
        void setName(const std::string &name) const;
        const char* getName() const;
    };



    class Material : public Object<Material> {
    public:
        void destroy();

        // JP: 以下のAPIを呼んだ場合はシェーダーバインディングテーブルを更新する必要がある。
        //     パイプラインのmarkHitGroupShaderBindingTableDirty()を呼べばローンチ時にセットアップされる。
        //     シェーダーバインディングテーブルのレイアウト生成後に、再度ユーザーデータのサイズや
        //     アラインメントを変更する場合には、レイアウトをシーンのmarkShaderBindingTableLayoutDirty()を
        //     呼んで無効化する必要もある。
        // EN: Updating a shader binding table is required when calling the following APIs.
        //     Calling pipeline's markHitGroupShaderBindingTableDirty() triggers re-setup of the table at launch.
        //     In the case where user data size and/or alignment changes again after generating the layout of
        //     a shader binding table, the invalidation of the layout is required as well by calling
        //     scene's markShaderBindingTableLayoutDirty().
        void setHitGroup(uint32_t rayType, HitProgramGroup hitGroup) const;
        void setUserData(const void* data, uint32_t size, uint32_t alignment) const;
        template <typename T>
        void setUserData(const T &data) const {
            setUserData(&data, sizeof(T), alignof(T));
        }

        HitProgramGroup getHitGroup(Pipeline pipeline, uint32_t rayType) const;
        void getUserData(void* data, uint32_t* size, uint32_t* alignment) const;
        template <typename T>
        void getUserData(T* data, uint32_t* size = nullptr, uint32_t* alignment = nullptr) const {
            getUserData(reinterpret_cast<void*>(data), size, alignment);
        }
    };



    class Scene : public Object<Scene> {
    public:
        void destroy();

        [[nodiscard]]
        OpacityMicroMapArray createOpacityMicroMapArray() const;
        [[nodiscard]]
        DisplacementMicroMapArray createDisplacementMicroMapArray() const;
        [[nodiscard]]
        GeometryInstance createGeometryInstance(
            OPTIXU_EN_PRM(GeometryType, geomType, Triangles)) const;
        [[nodiscard]]
        GeometryAccelerationStructure createGeometryAccelerationStructure(
            OPTIXU_EN_PRM(GeometryType, geomType, Triangles)) const;
        [[nodiscard]]
        Transform createTransform() const;
        [[nodiscard]]
        Instance createInstance() const;
        [[nodiscard]]
        InstanceAccelerationStructure createInstanceAccelerationStructure() const;

        // JP: シェーダーバインディングテーブルレイアウトをdirty状態にする。
        // EN: Mark the layout of shader binding table dirty.
        void markShaderBindingTableLayoutDirty() const;

        void generateShaderBindingTableLayout(size_t* memorySize) const;

        bool shaderBindingTableLayoutIsReady() const;
    };



    class OpacityMicroMapArray : public Object<OpacityMicroMapArray> {
    public:
        void destroy();

        // JP: 以下のAPIを呼んだ場合はOMM Arrayが自動でdirty状態になる。
        // EN: Calling the following APIs automatically marks the OMM array dirty.
        void setConfiguration(OptixOpacityMicromapFlags config) const;
        void computeMemoryUsage(
            const OptixOpacityMicromapHistogramEntry* microMapHistogramEntries,
            uint32_t numMicroMapHistogramEntries,
            OptixMicromapBufferSizes* memoryRequirement) const;
        void setBuffers(
            const BufferView &rawOmmBuffer, const BufferView &perMicroMapDescBuffer,
            const BufferView &outputBuffer) const;

        // JP: OMM Arrayをdirty状態にする。
        // EN: Mark the OMM array dirty.
        void markDirty() const;

        // JP: OMM Arrayをリビルドした場合、それを参照するGASのリビルド、もしくはアップデートが必要。
        //     アップデートを使う場合は予めGASの設定でOMM Arrayのアップデートを許可する必要がある。
        // EN: If the OMM array is rebuilt, GASs refering the OMM array need to be rebuilt or updated.
        //     Allowing OMM array update is required in the GAS setttings when using updating.
        void rebuild(CUstream stream, const BufferView &scratchBuffer) const;

        bool isReady() const;
        BufferView getOutputBuffer() const;

        OptixOpacityMicromapFlags getConfiguration() const;
    };



    class DisplacementMicroMapArray : public Object<DisplacementMicroMapArray> {
    public:
        void destroy();

        // JP: 以下のAPIを呼んだ場合はDMM Arrayが自動でdirty状態になる。
        // EN: Calling the following APIs automatically marks the DMM array dirty.
        void setConfiguration(OptixDisplacementMicromapFlags config) const;
        void computeMemoryUsage(
            const OptixDisplacementMicromapHistogramEntry* microMapHistogramEntries,
            uint32_t numMicroMapHistogramEntries,
            OptixMicromapBufferSizes* memoryRequirement) const;
        void setBuffers(
            const BufferView &rawDmmBuffer, const BufferView &perMicroMapDescBuffer,
            const BufferView &outputBuffer) const;

        // JP: DMM Arrayをdirty状態にする。
        // EN: Mark the DMM array dirty.
        void markDirty() const;

        // JP: DMM Arrayをリビルドした場合、それを参照するGASのリビルド、もしくはアップデートが必要。
        //     アップデートを使う場合は予めGASの設定でDMM Arrayのアップデートを許可する必要がある。
        // EN: If the DMM array is rebuilt, GASs refering the DMM array need to be rebuilt or updated.
        //     Allowing DMM array update is required in the GAS setttings when using updating.
        void rebuild(CUstream stream, const BufferView &scratchBuffer) const;

        bool isReady() const;
        BufferView getOutputBuffer() const;

        OptixDisplacementMicromapFlags getConfiguration() const;
    };



    class GeometryInstance : public Object<GeometryInstance> {
    public:
        void destroy();

        /*
        JP: 以下のAPIを呼んだ場合は所属するGASのmarkDirty()を呼ぶ必要がある。
            (頂点/Width/AABBバッファーの変更のみの場合は、markDirty()を呼ばずにGASのアップデートだけでも良い。)
        EN: Calling markDirty() of a GAS to which the geometry instance belongs is
            required when calling the following APIs.
            (It is okay to only use update instead of calling markDirty()
            when changing only vertex/width/AABB buffer.)
        */
        void setNumMotionSteps(uint32_t n) const;
        void setVertexFormat(OptixVertexFormat format) const;
        void setVertexBuffer(const BufferView &vertexBuffer, uint32_t motionStep = 0) const;
        void setWidthBuffer(const BufferView &widthBuffer, uint32_t motionStep = 0) const;
        void setNormalBuffer(const BufferView &normalBuffer, uint32_t motionStep = 0) const;
        void setRadiusBuffer(const BufferView &radiusBuffer, uint32_t motionStep = 0) const;
        void setTriangleBuffer(
            const BufferView &triangleBuffer,
            OptixIndicesFormat format = OPTIX_INDICES_FORMAT_UNSIGNED_INT3) const;
        void setOpacityMicroMapArray(
            OpacityMicroMapArray opacityMicroMapArray,
            const OptixOpacityMicromapUsageCount* ommUsageCounts, uint32_t numOmmUsageCounts,
            const BufferView &ommIndexBuffer,
            OPTIXU_EN_PRM(IndexSize, indexSize, k4Bytes), uint32_t indexOffset = 0) const;
        void setDisplacementMicroMapArray(
            const BufferView &vertexDirectionBuffer,
            const BufferView &vertexBiasAndScaleBuffer,
            const BufferView &triangleFlagsBuffer,
            DisplacementMicroMapArray displacementMicroMapArray,
            const OptixDisplacementMicromapUsageCount* dmmUsageCounts, uint32_t numDmmUsageCounts,
            const BufferView &dmmIndexBuffer,
            OPTIXU_EN_PRM(IndexSize, indexSize, k4Bytes), uint32_t indexOffset = 0,
            OptixDisplacementMicromapDirectionFormat vertexDirectionFormat = OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_FLOAT3,
            OptixDisplacementMicromapBiasAndScaleFormat vertexBiasAndScaleFormat = OPTIX_DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_FLOAT2) const;
        void setSegmentIndexBuffer(const BufferView &segmentIndexBuffer) const;
        void setCurveEndcapFlags(OptixCurveEndcapFlags endcapFlags) const;
        void setSingleRadius(UseSingleRadius useSingleRadius) const;
        void setCustomPrimitiveAABBBuffer(
            const BufferView &primitiveAABBBuffer, uint32_t motionStep = 0) const;
        void setPrimitiveIndexOffset(uint32_t offset) const;
        void setNumMaterials(
            uint32_t numMaterials, const BufferView &matIndexBuffer,
            OPTIXU_EN_PRM(IndexSize, indexSize, k4Bytes)) const;
        void setGeometryFlags(uint32_t matIdx, OptixGeometryFlags flags) const;

        /*
        JP: 以下のAPIを呼んだ場合はシェーダーバインディングテーブルを更新する必要がある。
            パイプラインのmarkHitGroupShaderBindingTableDirty()を呼べばローンチ時にセットアップされる。
            シェーダーバインディングテーブルのレイアウト生成後に、再度ユーザーデータのサイズや
            アラインメントを変更する場合レイアウトが自動で無効化される。
        EN: Updating a shader binding table is required when calling the following APIs.
            Calling pipeline's markHitGroupShaderBindingTableDirty() triggers re-setup of the table at launch.
            In the case where user data size and/or alignment changes again after generating the layout of
            a shader binding table, the layout is automatically invalidated.
        */
        void setMaterial(uint32_t matSetIdx, uint32_t matIdx, Material mat) const;
        void setUserData(const void* data, uint32_t size, uint32_t alignment) const;
        template <typename T>
        void setUserData(const T &data) const {
            setUserData(&data, sizeof(T), alignof(T));
        }

        GeometryType getGeometryType() const;
        uint32_t getNumMotionSteps() const;
        OptixVertexFormat getVertexFormat() const;
        BufferView getVertexBuffer(uint32_t motionStep = 0);
        BufferView getWidthBuffer(uint32_t motionStep = 0);
        BufferView getNormalBuffer(uint32_t motionStep = 0);
        BufferView getRadiusBuffer(uint32_t motionStep = 0);
        BufferView getTriangleBuffer(OptixIndicesFormat* format = nullptr) const;
        OpacityMicroMapArray getOpacityMicroMapArray(
            BufferView* ommIndexBuffer = nullptr,
            IndexSize* indexSize = nullptr, uint32_t* indexOffset = nullptr) const;
        DisplacementMicroMapArray getDisplacementMicroMapArray(
            BufferView* vertexDirectionBuffer = nullptr,
            BufferView* vertexBiasAndScaleBuffer = nullptr,
            BufferView* triangleFlagsBuffer = nullptr,
            BufferView* dmmIndexBuffer = nullptr,
            IndexSize* indexSize = nullptr, uint32_t* indexOffset = nullptr,
            OptixDisplacementMicromapDirectionFormat* vertexDirectionFormat = nullptr,
            OptixDisplacementMicromapBiasAndScaleFormat* vertexBiasAndScaleFormat = nullptr) const;
        BufferView getSegmentIndexBuffer() const;
        BufferView getCustomPrimitiveAABBBuffer(uint32_t motionStep = 0) const;
        uint32_t getPrimitiveIndexOffset() const;
        uint32_t getNumMaterials(BufferView* matIndexBuffer = nullptr, IndexSize* indexSize = nullptr) const;
        OptixGeometryFlags getGeometryFlags(uint32_t matIdx) const;
        Material getMaterial(uint32_t matSetIdx, uint32_t matIdx) const;
        void getUserData(void* data, uint32_t* size, uint32_t* alignment) const;
        template <typename T>
        void getUserData(T* data, uint32_t* size = nullptr, uint32_t* alignment = nullptr) const {
            getUserData(reinterpret_cast<void*>(data), size, alignment);
        }
    };



    class GeometryAccelerationStructure : public Object<GeometryAccelerationStructure> {
    public:
        void destroy();

        // JP: 以下のAPIを呼んだ場合はGASが自動でdirty状態になる。
        //     子の数が変更される場合はヒットグループのシェーダーバインディングテーブルレイアウトも無効化される。
        // EN: Calling the following APIs automatically marks the GAS dirty.
        //     Changing the number of children invalidates the shader binding table layout of hit group.
        void setConfiguration(
            ASTradeoff tradeoff,
            OPTIXU_EN_PRM(AllowUpdate, allowUpdate, No),
            OPTIXU_EN_PRM(AllowCompaction, allowCompaction, No),
            OPTIXU_EN_PRM(AllowRandomVertexAccess, allowRandomVertexAccess, No),
            OPTIXU_EN_PRM(AllowOpacityMicroMapUpdate, allowOpacityMicroMapUpdate, No),
            OPTIXU_EN_PRM(AllowDisableOpacityMicroMaps, allowDisableOpacityMicroMaps, No)) const;
        void setMotionOptions(
            uint32_t numKeys, float timeBegin, float timeEnd, OptixMotionFlags flags) const;
        void addChild(
            GeometryInstance geomInst, CUdeviceptr preTransform = 0,
            const void* data = nullptr, uint32_t size = 0, uint32_t alignment = 1) const;
        template <typename T>
        void addChild(GeometryInstance geomInst, CUdeviceptr preTransform, const T &data) const {
            addChild(geomInst, preTransform, &data, sizeof(T), alignof(T));
        }
        void removeChildAt(uint32_t index) const;
        void clearChildren() const;

        // JP: GASをdirty状態にする。
        //     ヒットグループのシェーダーバインディングテーブルレイアウトも無効化される。
        // EN: Mark the GAS dirty.
        //     Invalidate the shader binding table layout of hit group as well.
        void markDirty() const;

        // JP: 以下のAPIを呼んだ場合はヒットグループのシェーダーバインディングテーブルレイアウト
        //     が自動で無効化される。
        // EN: Calling the following APIs automatically invalidates the shader binding table layout of hit group.
        void setNumMaterialSets(uint32_t numMatSets) const;
        void setNumRayTypes(uint32_t matSetIdx, uint32_t numRayTypes) const;

        // JP: リビルド・コンパクトを行った場合はこのGASが(間接的に)所属するTraversable (例: IAS)
        //     のmarkDirty()を呼ぶ必要がある。
        // EN: Calling markDirty() of a traversable (e.g. IAS) to which this GAS (indirectly) belongs
        //     is required when performing rebuild / compact.
        void prepareForBuild(OptixAccelBufferSizes* memoryRequirement) const;
        OptixTraversableHandle rebuild(
            CUstream stream, const BufferView &accelBuffer, const BufferView &scratchBuffer) const;
        // JP: リビルドが完了するのをホスト側で待つ。
        // EN: Wait on the host until rebuild operation finishes.
        void prepareForCompact(size_t* compactedAccelBufferSize) const;
        OptixTraversableHandle compact(CUstream stream, const BufferView &compactedAccelBuffer) const;
        // JP: コンパクトが完了するのをホスト側で待つ。
        // EN: Wait on the host until compact operation finishes.
        void removeUncompacted() const;

        // JP: アップデートを行った場合はこのGASが(間接的に)所属するTraversable (例: IAS)
        //     もアップデートもしくはリビルドする必要がある。
        // EN: Updating or rebuilding a traversable (e.g. IAS) to which this GAS (indirectly) belongs
        //     is required when performing update.
        void update(CUstream stream, const BufferView &scratchBuffer) const;

        /*
        JP: 以下のAPIを呼んだ場合はシェーダーバインディングテーブルを更新する必要がある。
            パイプラインのmarkHitGroupShaderBindingTableDirty()を呼べばローンチ時にセットアップされる。
            シェーダーバインディングテーブルのレイアウト生成後に、再度ユーザーデータのサイズや
            アラインメントを変更する場合レイアウトが自動で無効化される。
        EN: Updating a shader binding table is required when calling the following APIs.
            Calling pipeline's markHitGroupShaderBindingTableDirty() triggers re-setup of the table at launch.
            In the case where user data size and/or alignment changes again after generating the layout of
            a shader binding table, the layout is automatically invalidated.
        */
        void setChildUserData(uint32_t index, const void* data, uint32_t size, uint32_t alignment) const;
        template <typename T>
        void setChildUserData(uint32_t index, const T &data) const {
            setChildUserData(index, &data, sizeof(T), alignof(T));
        }
        void setUserData(const void* data, uint32_t size, uint32_t alignment) const;
        template <typename T>
        void setUserData(const T &data) const {
            setUserData(&data, sizeof(T), alignof(T));
        }

        bool isReady() const;
        OptixTraversableHandle getHandle() const;

        GeometryType getGeometryType() const;
        void getConfiguration(
            ASTradeoff* tradeOff, AllowUpdate* allowUpdate, AllowCompaction* allowCompaction,
            AllowRandomVertexAccess* allowRandomVertexAccess,
            AllowOpacityMicroMapUpdate* allowOpacityMicroMapUpdate,
            AllowDisableOpacityMicroMaps* allowDisableOpacityMicroMaps) const;
        void getMotionOptions(uint32_t* numKeys, float* timeBegin, float* timeEnd, OptixMotionFlags* flags) const;
        uint32_t getNumChildren() const;
        uint32_t findChildIndex(GeometryInstance geomInst, CUdeviceptr preTransform = 0) const;
        GeometryInstance getChild(uint32_t index, CUdeviceptr* preTransform = nullptr) const;
        uint32_t getNumMaterialSets() const;
        uint32_t getNumRayTypes(uint32_t matSetIdx) const;
        void getChildUserData(uint32_t index, void* data, uint32_t* size, uint32_t* alignment) const;
        template <typename T>
        void getChildUserData(
            uint32_t index, T* data, uint32_t* size = nullptr, uint32_t* alignment = nullptr) const {
            getChildUserData(index, reinterpret_cast<void*>(data), size, alignment);
        }
        void getUserData(void* data, uint32_t* size, uint32_t* alignment) const;
        template <typename T>
        void getUserData(T* data, uint32_t* size = nullptr, uint32_t* alignment = nullptr) const {
            getUserData(reinterpret_cast<void*>(data), size, alignment);
        }
    };



    class Transform : public Object<Transform> {
    public:
        void destroy();

        // JP: 以下のAPIを呼んだ場合はTransformが自動でdirty状態になる。
        // EN: Calling the following APIs automatically marks the transform dirty.
        void setConfiguration(TransformType type, uint32_t numKeys, size_t* transformSize) const;
        void setMotionOptions(float timeBegin, float timeEnd, OptixMotionFlags flags) const;
        void setMatrixMotionKey(uint32_t keyIdx, const float matrix[12]) const;
        void setSRTMotionKey(
            uint32_t keyIdx,
            const float scale[3],
            const float orientation[4],
            const float translation[3]) const;
        void setStaticTransform(const float matrix[12]) const;
        void setChild(GeometryAccelerationStructure child) const;
        void setChild(InstanceAccelerationStructure child) const;
        void setChild(Transform child) const;

        // JP: Transformをdirty状態にする。
        // EN: Mark the transform dirty.
        void markDirty() const;

        // JP: (間接的に)所属するTraversable (例: IAS)のmarkDirty()を呼ぶ必要がある。
        // EN: Calling markDirty() of a traversable (e.g. IAS) to which the transform
        //     (indirectly) belongs is required.
        OptixTraversableHandle rebuild(CUstream stream, const BufferView &trDeviceMem) const;

        bool isReady() const;
        OptixTraversableHandle getHandle() const;

        void getConfiguration(TransformType* type, uint32_t* numKeys) const;
        void getMotionOptions(float* timeBegin, float* timeEnd, OptixMotionFlags* flags) const;
        void getMatrixMotionKey(uint32_t keyIdx, float matrix[12]) const;
        void getSRTMotionKey(
            uint32_t keyIdx,
            float scale[3],
            float orientation[4],
            float translation[3]) const;
        void getStaticTransform(float matrix[12]) const;
        ChildType getChildType() const;
        template <typename T>
        T getChild() const;
    };



    class Instance : public Object<Instance> {
    public:
        void destroy();

        // JP: 所属するIASのmarkDirty()を呼ぶ必要がある。
        // EN: Calling markDirty() of a IAS to which the instance belongs is required.
        void setChild(GeometryAccelerationStructure child, uint32_t matSetIdx = 0) const;
        void setChild(InstanceAccelerationStructure child) const;
        void setChild(Transform child, uint32_t matSetIdx = 0) const;

        // JP: 所属するIASをリビルドもしくはアップデートする必要がある。
        // EN: Rebulding or Updating of a IAS to which the instance belongs is required.
        void setID(uint32_t value) const;
        void setVisibilityMask(uint32_t mask) const;
        void setFlags(OptixInstanceFlags flags) const;
        void setTransform(const float transform[12]) const;
        void setMaterialSetIndex(uint32_t matSetIdx) const;

        ChildType getChildType() const;
        template <typename T>
        T getChild() const;
        uint32_t getID() const;
        uint32_t getVisibilityMask() const;
        OptixInstanceFlags getFlags() const;
        void getTransform(float transform[12]) const;
        uint32_t getMaterialSetIndex() const;
    };



    /*
    TODO: インスタンスバッファーもユーザー管理にしたいため、rebuild()が今の形になっているが微妙かもしれない。
          インスタンスバッファーを内部で1つ持つようにすると、
          あるフレームでIASをビルド、次のフレームでインスタンスの追加がありリビルドの必要が生じた場合に
          1フレーム目のGPU処理の終了を待たないと危険という状況になってしまう。
          OptiX的にはASのビルド完了後にはインスタンスバッファーは不要となるが、
          アップデート処理はリビルド時に書かれたインスタンスバッファーの内容を期待しているため、
          基本的にインスタンスバッファーとASのメモリ(コンパクション版にもなり得る)
          は同じ寿命で扱ったほうが良さそう。
    */
    class InstanceAccelerationStructure : public Object<InstanceAccelerationStructure> {
    public:
        void destroy();

        // JP: 以下のAPIを呼んだ場合はIASが自動でdirty状態になる。
        // EN: Calling the following APIs automatically marks the IAS dirty.
        void setConfiguration(
            ASTradeoff tradeoff,
            OPTIXU_EN_PRM(AllowUpdate, allowUpdate, No),
            OPTIXU_EN_PRM(AllowCompaction, allowCompaction, No),
            OPTIXU_EN_PRM(AllowRandomInstanceAccess, allowRandomInstanceAccess, No)) const;
        void setMotionOptions(uint32_t numKeys, float timeBegin, float timeEnd, OptixMotionFlags flags) const;
        void addChild(Instance instance) const;
        void removeChildAt(uint32_t index) const;
        void clearChildren() const;

        // JP: IASをdirty状態にする。
        // EN: Mark the IAS dirty.
        void markDirty() const;

        // JP: リビルド・コンパクトを行った場合はこのIASが(間接的に)所属するTraversable (例: IAS)
        //     のmarkDirty()を呼ぶ必要がある。
        // EN: Calling markDirty() of a traversable (e.g. IAS) to which this IAS (indirectly) belongs
        //     is required when performing rebuild / compact.
        void prepareForBuild(OptixAccelBufferSizes* memoryRequirement) const;
        OptixTraversableHandle rebuild(
            CUstream stream, const BufferView &instanceBuffer,
            const BufferView &accelBuffer, const BufferView &scratchBuffer) const;
        // JP: リビルドが完了するのをホスト側で待つ。
        // EN: Wait on the host until rebuild operation finishes.
        void prepareForCompact(size_t* compactedAccelBufferSize) const;
        OptixTraversableHandle compact(CUstream stream, const BufferView &compactedAccelBuffer) const;
        // JP: コンパクトが完了するのをホスト側で待つ。
        // EN: Wait on the host until compact operation finishes.
        void removeUncompacted() const;

        // JP: アップデートを行った場合はこのIASが(間接的に)所属するTraversable (例: IAS)
        //     もアップデートもしくはリビルドする必要がある。
        // EN: Updating or rebuilding a traversable (e.g. IAS) to which this IAS (indirectly) belongs
        //     is required when performing update.
        void update(CUstream stream, const BufferView &scratchBuffer) const;

        bool isReady() const;
        OptixTraversableHandle getHandle() const;

        void getConfiguration(
            ASTradeoff* tradeOff,
            AllowUpdate* allowUpdate,
            AllowCompaction* allowCompaction,
            AllowRandomInstanceAccess* allowRandomInstanceAccess) const;
        void getMotionOptions(
            uint32_t* numKeys, float* timeBegin, float* timeEnd, OptixMotionFlags* flags) const;
        uint32_t getNumChildren() const;
        uint32_t findChildIndex(Instance instance) const;
        Instance getChild(uint32_t index) const;
    };



    class Pipeline : public Object<Pipeline> {
    public:
        void destroy();

        void setPipelineOptions(
            uint32_t numPayloadValuesInDwords, uint32_t numAttributeValuesInDwords,
            const char* launchParamsVariableName, size_t sizeOfLaunchParams,
            OptixTraversableGraphFlags traversableGraphFlags,
            OptixExceptionFlags exceptionFlags,
            OptixPrimitiveTypeFlags supportedPrimitiveTypeFlags,
            OPTIXU_EN_PRM(UseMotionBlur, useMotionBlur, No),
            OPTIXU_EN_PRM(UseOpacityMicroMaps, useOpacityMicroMaps, No)) const;

        [[nodiscard]]
        Module createModuleFromPTXString(
            const std::string &ptxString, int32_t maxRegisterCount,
            OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel,
            OptixModuleCompileBoundValueEntry* boundValues = nullptr, uint32_t numBoundValues = 0,
            const PayloadType* payloadTypes = nullptr, uint32_t numPayloadTypes = 0) const;
        [[nodiscard]]
        Module createModuleFromOptixIR(
            const std::vector<char> &irBin, int32_t maxRegisterCount,
            OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel,
            OptixModuleCompileBoundValueEntry* boundValues = nullptr, uint32_t numBoundValues = 0,
            const PayloadType* payloadTypes = nullptr, uint32_t numPayloadTypes = 0) const;

        [[nodiscard]]
        Program createRayGenProgram(Module module, const char* entryFunctionName) const;
        [[nodiscard]]
        Program createExceptionProgram(Module module, const char* entryFunctionName) const;
        [[nodiscard]]
        Program createMissProgram(
            Module module, const char* entryFunctionName,
            const PayloadType &payloadType = PayloadType()) const;
        [[nodiscard]]
        HitProgramGroup createHitProgramGroupForTriangleIS(
            Module module_CH, const char* entryFunctionNameCH,
            Module module_AH, const char* entryFunctionNameAH,
            const PayloadType &payloadType = PayloadType()) const;
        [[nodiscard]]
        HitProgramGroup createHitProgramGroupForCurveIS(
            OptixPrimitiveType curveType, OptixCurveEndcapFlags endcapFlags,
            Module module_CH, const char* entryFunctionNameCH,
            Module module_AH, const char* entryFunctionNameAH,
            ASTradeoff tradeoff,
            OPTIXU_EN_PRM(AllowUpdate, allowUpdate, No),
            OPTIXU_EN_PRM(AllowCompaction, allowCompaction, No),
            OPTIXU_EN_PRM(AllowRandomVertexAccess, allowRandomVertexAccess, No),
            const PayloadType &payloadType = PayloadType()) const;
        [[nodiscard]]
        HitProgramGroup createHitProgramGroupForSphereIS(
            Module module_CH, const char* entryFunctionNameCH,
            Module module_AH, const char* entryFunctionNameAH,
            ASTradeoff tradeoff,
            OPTIXU_EN_PRM(AllowUpdate, allowUpdate, No),
            OPTIXU_EN_PRM(AllowCompaction, allowCompaction, No),
            OPTIXU_EN_PRM(AllowRandomVertexAccess, allowRandomVertexAccess, No),
            const PayloadType &payloadType = PayloadType()) const;
        [[nodiscard]]
        HitProgramGroup createHitProgramGroupForCustomIS(
            Module module_CH, const char* entryFunctionNameCH,
            Module module_AH, const char* entryFunctionNameAH,
            Module module_IS, const char* entryFunctionNameIS,
            const PayloadType &payloadType = PayloadType()) const;
        [[nodiscard]]
        HitProgramGroup createEmptyHitProgramGroup() const;
        [[nodiscard]]
        CallableProgramGroup createCallableProgramGroup(
            Module module_DC, const char* entryFunctionNameDC,
            Module module_CC, const char* entryFunctionNameCC,
            const PayloadType &payloadType = PayloadType()) const;

        void link(uint32_t maxTraceDepth) const;

        // JP: 以下のAPIを呼んだ場合は(非ヒットグループの)シェーダーバインディングテーブルレイアウトが
        //     自動で無効化される。
        // EN: Calling the following APIs automatically invalidates
        //     the (non-hit group) shader binding table layout.
        void setNumMissRayTypes(uint32_t numMissRayTypes) const;
        void setNumCallablePrograms(uint32_t numCallablePrograms) const;

        void generateShaderBindingTableLayout(size_t* memorySize) const;

        /*
        JP: 以下のAPIを呼んだ場合は(非ヒットグループの)シェーダーバインディングテーブルが自動でdirty状態になり
            ローンチ時に再セットアップされる。
            ただしローンチ時のセットアップはSBTバッファーの内容変更・転送を伴うので、
            非同期書き換えを行う場合は安全のためにはSBTバッファーをダブルバッファリングする必要がある。
        EN: Calling the following API automatically marks the (non-hit group) shader binding table dirty
            then triggers re-setup of the table at launch.
            However note that the setup in the launch involves the change of the SBT buffer's contents
            and transfer, so double buffered SBT is required for safety
            in the case performing asynchronous update.
        */
        void setRayGenerationProgram(Program program) const;
        void setExceptionProgram(Program program) const;
        void setMissProgram(uint32_t rayType, Program program) const;
        void setCallableProgram(uint32_t index, CallableProgramGroup program) const;
        void setShaderBindingTable(const BufferView &shaderBindingTable, void* hostMem) const;

        /*
        JP: 以下のAPIを呼んだ場合はヒットグループのシェーダーバインディングテーブルが自動でdirty状態になり
            ローンチ時に再セットアップされる。
            ただしローンチ時のセットアップはSBTバッファーの内容変更・転送を伴うので、
            非同期書き換えを行う場合は安全のためにはSBTバッファーをダブルバッファリングする必要がある。
        EN: Calling the following APIs automatically marks the hit group's shader binding table dirty,
            then triggers re-setup of the table at launch.
            However note that the setup in the launch involves the change of the SBT buffer's contents
            and transfer, so double buffered SBT is required for safety
            in the case performing asynchronous update.
        */
        void setScene(const Scene &scene) const;
        void setHitGroupShaderBindingTable(const BufferView &shaderBindingTable, void* hostMem) const;

        // JP: ヒットグループのシェーダーバインディングテーブルをdirty状態にする。
        // EN: Mark the hit group's shader binding table dirty.
        void markHitGroupShaderBindingTableDirty() const;

        void setStackSize(
            uint32_t directCallableStackSizeFromTraversal,
            uint32_t directCallableStackSizeFromState,
            uint32_t continuationStackSize,
            uint32_t maxTraversableGraphDepth) const;

        // JP: セットされたシーンを基にシェーダーバインディングテーブルのセットアップを行い、
        //     Ray Generationシェーダーを起動する。
        // EN: Setup the shader binding table based on the scene set, then launch the ray generation shader.
        void launch(
            CUstream stream, CUdeviceptr plpOnDevice,
            uint32_t dimX, uint32_t dimY, uint32_t dimZ) const;

        Program getRayGenerationProgram() const;
        Program getExceptionProgram() const;
        Program getMissProgram(uint32_t rayType) const;
        CallableProgramGroup getCallableProgram(uint32_t index) const;
        Scene getScene() const;
    };



    // JP: Moduleの寿命はそれを参照するあらゆるProgramGroupの寿命よりも長い必要がある。
    // EN: The lifetime of a module must extend to the lifetime of any ProgramGroup that reference that module.
    class Module : public Object<Module> {
    public:
        void destroy();
    };



    class Program : public Object<Program> {
    public:
        void destroy();

        uint32_t getStackSize() const;

        // JP: これはパイプラインをdirty状態になるためリンクが再度必要。
        // EN: This makes the pipeline dirty and link is required again.
        void setActiveInPipeline(bool b) const;
    };



    class HitProgramGroup : public Object<HitProgramGroup> {
    public:
        void destroy();

        uint32_t getCHStackSize() const;
        uint32_t getAHStackSize() const;
        uint32_t getISStackSize() const;

        // JP: これはパイプラインをdirty状態になるためリンクが再度必要。
        // EN: This makes the pipeline dirty and link is required again.
        void setActiveInPipeline(bool b) const;
    };



    class CallableProgramGroup : public Object<CallableProgramGroup> {
    public:
        void destroy();

        uint32_t getDCStackSize() const;
        uint32_t getCCStackSize() const;

        // JP: これはパイプラインをdirty状態になるためリンクが再度必要。
        // EN: This makes the pipeline dirty and link is required again.
        void setActiveInPipeline(bool b) const;
    };



    class DenoisingTask {
        uint32_t placeHolder[6];

        void getOutputTile(int32_t* offsetX, int32_t* offsetY, int32_t* width, int32_t* height) const;
    };

    struct DenoiserSizes {
        size_t stateSize;
        size_t scratchSize;
        size_t normalizerSize;
        size_t scratchSizeForComputeNormalizer;
        size_t internalGuideLayerPixelSize;
    };

    struct DenoiserInputBuffers {
        BufferView noisyBeauty;
        BufferView albedo;
        BufferView normal;
        BufferView flow;
        BufferView flowTrustworthiness;
        BufferView previousDenoisedBeauty;
        BufferView previousInternalGuideLayer;
        BufferView* noisyAovs;
        BufferView* previousDenoisedAovs;
        OptixPixelFormat beautyFormat;
        OptixPixelFormat albedoFormat;
        OptixPixelFormat normalFormat;
        OptixPixelFormat flowFormat;
        OptixPixelFormat flowTrustworthinessFormat;
        OptixPixelFormat* aovFormats;
        OptixDenoiserAOVType* aovTypes;
        uint32_t numAovs;
    };

    class Denoiser : public Object<Denoiser> {
    public:
        void destroy();

        void prepare(
            uint32_t imageWidth, uint32_t imageHeight, uint32_t tileWidth, uint32_t tileHeight,
            DenoiserSizes* sizes, uint32_t* numTasks) const;
        void getTasks(DenoisingTask* tasks) const;
        void setupState(CUstream stream, const BufferView &stateBuffer, const BufferView &scratchBuffer) const;

        void computeNormalizer(
            CUstream stream,
            const BufferView &noisyBeauty, OptixPixelFormat beautyFormat,
            const BufferView &scratchBuffer, CUdeviceptr normalizer) const;
        void invoke(
            CUstream stream, const DenoisingTask &task,
            const DenoiserInputBuffers &inputBuffers, IsFirstFrame isFirstFrame,
            CUdeviceptr normalizer, float blendFactor,
            const BufferView &denoisedBeauty, const BufferView* denoisedAovs,
            const BufferView &internalGuideLayerForNextFrame) const;
    };

#undef OPTIXU_EN_PRM

#endif // #if !defined(__CUDA_ARCH__)
    // END: Host-side API
    // ----------------------------------------------------------------
} // namespace optixu



// ----------------------------------------------------------------
// Declarations for code completion

// JP: 技術的には以下の宣言は不要。単にIntellisenseなどのコード補完を助ける目的。
// EN: Technically, declarations here are not required, just for helping code completion like Intellisense.
#if defined(OPTIXU_Platform_CodeCompletion)

struct int2;
struct int3;
struct int4;
struct uint2;
struct uint3;
struct uint4;
struct float2;
struct float3;
struct float4;

struct OptixInvalidRayExceptionDetails;
struct OptixParameterMismatchExceptionDetails;

void optixGetCatmullRomVertexData(OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[4]);
void optixGetCubicBezierVertexData(OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[4]);
void optixGetCubicBSplineVertexData(OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[4]);
float optixGetCurveParameter();
int optixGetExceptionCode();
char* optixGetExceptionLineInfo();
unsigned int optixGetGASMotionStepCount(OptixTraversableHandle gas);
float optixGetGASMotionTimeBegin(OptixTraversableHandle gas);
float optixGetGASMotionTimeEnd(OptixTraversableHandle gas);
CUdeviceptr optixGetGASPointerFromHandle(OptixTraversableHandle gas);
OptixTraversableHandle optixGetGASTraversableHandle();
unsigned int optixGetHitKind();
OptixTraversableHandle optixGetInstanceChildFromHandle(OptixTraversableHandle handle);
unsigned int optixGetInstanceId();
unsigned int optixGetInstanceIdFromHandle(OptixTraversableHandle handle);
unsigned int optixGetInstanceIndex();
const float4* optixGetInstanceInverseTransformFromHandle(OptixTraversableHandle handle);
const float4* optixGetInstanceTransformFromHandle(OptixTraversableHandle handle);
OptixTraversableHandle optixGetInstanceTraversableFromIAS(OptixTraversableHandle ias, unsigned int instIdx);
uint3 optixGetLaunchDimensions();
uint3 optixGetLaunchIndex();
void optixGetLinearCurveVertexData(OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[2]);
const OptixMatrixMotionTransform* optixGetMatrixMotionTransformFromHandle(OptixTraversableHandle handle);
void optixGetMicroTriangleBarycentricsData(float2 data[3]);
void optixGetMicroTriangleVertexData(float3 data[3]);
float3 optixGetObjectRayDirection();
float3 optixGetObjectRayOrigin();
void optixGetObjectToWorldTransformMatrix(float m[12]);
unsigned int optixGetPrimitiveIndex();
OptixPrimitiveType optixGetPrimitiveType(unsigned int hitKind);
OptixPrimitiveType optixGetPrimitiveType();
void optixGetQuadraticBSplineVertexData(OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[3]);
unsigned int optixGetRayFlags();
float optixGetRayTime();
float optixGetRayTmax();
float optixGetRayTmin();
unsigned int optixGetRayVisibilityMask();
float3 optixGetRibbonNormal(OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float2 ribbonParameters);
float2 optixGetRibbonParameters();
void optixGetRibbonVertexData(OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[3]);
CUdeviceptr optixGetSbtDataPointer();
unsigned int optixGetSbtGASIndex();
void optixGetSphereData(OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[1]);
const OptixSRTMotionTransform* optixGetSRTMotionTransformFromHandle(OptixTraversableHandle handle);
const OptixStaticTransform* optixGetStaticTransformFromHandle(OptixTraversableHandle handle);
OptixTraversableHandle optixGetTransformListHandle(unsigned int index);
unsigned int optixGetTransformListSize();
OptixTransformType optixGetTransformTypeFromHandle(OptixTraversableHandle handle);
float2 optixGetTriangleBarycentrics();
void optixGetTriangleVertexData(OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float3 data[3]);
float3 optixGetWorldRayDirection();
float3 optixGetWorldRayOrigin();
void optixGetWorldToObjectTransformMatrix(float m[12]);
unsigned int optixHitObjectGetHitKind();
unsigned int optixHitObjectGetInstanceId();
unsigned int optixHitObjectGetInstanceIndex();
unsigned int optixHitObjectGetPrimitiveIndex();
float optixHitObjectGetRayTime();
float optixHitObjectGetRayTmax();
float optixHitObjectGetRayTmin();
CUdeviceptr optixHitObjectGetSbtDataPointer();
unsigned int optixHitObjectGetSbtGASIndex();
unsigned int optixHitObjectGetSbtRecordIndex();
OptixTraversableHandle optixHitObjectGetTransformListHandle(unsigned int index);
unsigned int optixHitObjectGetTransformListSize();
float3 optixHitObjectGetWorldRayDirection();
float3 optixHitObjectGetWorldRayOrigin();
bool optixHitObjectIsHit();
bool optixHitObjectIsMiss();
bool optixHitObjectIsNop();
void optixIgnoreIntersection();
bool optixIsBackFaceHit(unsigned int hitKind);
bool optixIsBackFaceHit();
bool optixIsDisplacedMicromeshTriangleBackFaceHit();
bool optixIsDisplacedMicromeshTriangleFrontFaceHit();
bool optixIsDisplacedMicromeshTriangleHit();
bool optixIsFrontFaceHit(unsigned int hitKind);
bool optixIsFrontFaceHit();
bool optixIsTriangleBackFaceHit();
bool optixIsTriangleFrontFaceHit();
bool optixIsTriangleHit();
void optixMakeMissHitObject(unsigned int missSBTIndex, float3 rayOrigin, float3 rayDirection, float tmin, float tmax, float rayTime);
void optixMakeNopHitObject();
void optixReorder();
void optixReorder(unsigned int coherentHint, unsigned int numCoherentHintBitsFromLSB);
void optixSetPayloadTypes(unsigned int typeMask);
void optixTerminateRay();
uint4 optixTexFootprint2D(unsigned long long tex, unsigned int texInfo, float x, float y, unsigned int* singleMipLevel);
uint4 optixTexFootprint2DGrad(unsigned long long tex, unsigned int texInfo, float x, float y, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y, bool coarse, unsigned int* singleMipLevel);
uint4 optixTexFootprint2DLod(unsigned long long tex, unsigned int texInfo, float x, float y, float level, bool coarse, unsigned int* singleMipLevel);
float3 optixTransformNormalFromObjectToWorldSpace(float3 normal);
float3 optixTransformNormalFromWorldToObjectSpace(float3 normal);
float3 optixTransformPointFromObjectToWorldSpace(float3 point);
float3 optixTransformPointFromWorldToObjectSpace(float3 point);
float3 optixTransformVectorFromObjectToWorldSpace(float3 vec);
float3 optixTransformVectorFromWorldToObjectSpace(float3 vec);
unsigned int optixUndefinedValue();

#endif // if defined(OPTIXU_Platform_CodeCompletion)

// END: Declarations for code completion
// ----------------------------------------------------------------

#undef OPTIXU_TO_STRING
#undef OPTIXU_STRINGIFY
