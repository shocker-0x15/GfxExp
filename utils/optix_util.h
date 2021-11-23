/*

   Copyright 2021 Shin Watanabe

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
- ユニットテスト。
- Linux環境でのテスト。
- CMake整備。
- ASのRelocationサポート。
- AOV Denoiserのサンプル作成。
- Instance Pointersサポート。
- removeUncompacted再考。(compaction終了待ちとしてとらえる？)
- 途中で各オブジェクトのパラメターを変更した際の処理。
  パイプラインのセットアップ順などが現状は暗黙的に固定されている。これを自由な順番で変えられるようにする。
- Multi GPUs?
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
- setPayloads/getPayloadsなどで引数側が必要以上の引数を渡していてもエラーが出ない問題。
  => 言語機能的に難しいか。
- InstanceのsetChildはTraversal Graph Depthに影響しないので名前を変えるべき？setTraversable()?
  => GASのsetChildもDepthに影響しないことを考えるとこのままで良いかも。

*/

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#   define OPTIXU_Platform_Windows
#   if defined(_MSC_VER)
#       define OPTIXU_Platform_Windows_MSVC
#       if defined(__INTELLISENSE__)
#           define OPTIXU_Platform_CodeCompletion
#       endif
#   endif
#elif defined(__APPLE__)
#   define OPTIXU_Platform_macOS
#endif

#if defined(__CUDACC_RTC__)
// Defining things corresponding to cstdint and cfloat is left to the user.
#else
#include <cstdint>
#include <cfloat>
#include <string>
#endif
#include <optix.h>

#if !defined(__CUDA_ARCH__)
#include <optix_stubs.h>
#endif

#ifdef _DEBUG
#   define OPTIXU_ENABLE_ASSERT
#endif
#define OPTIXU_ENABLE_RUNTIME_ERROR

#if defined(__CUDA_ARCH__)
#   define RT_CALLABLE_PROGRAM extern "C" __device__
#   define RT_DEVICE_FUNCTION __device__ __forceinline__
#   if !defined(RT_PIPELINE_LAUNCH_PARAMETERS)
#       define RT_PIPELINE_LAUNCH_PARAMETERS extern "C" __constant__
#   endif
#else
#   define RT_CALLABLE_PROGRAM
#   define RT_DEVICE_FUNCTION
#   define RT_PIPELINE_LAUNCH_PARAMETERS
#endif

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
    RT_DEVICE_FUNCTION inline Type operator~(Type a) { \
        return static_cast<Type>(~static_cast<uint32_t>(a)); \
    } \
    RT_DEVICE_FUNCTION inline Type operator|(Type a, Type b) { \
        return static_cast<Type>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b)); \
    } \
    RT_DEVICE_FUNCTION inline Type &operator|=(Type &a, Type b) { \
        a = static_cast<Type>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b)); \
        return a; \
    } \
    RT_DEVICE_FUNCTION inline Type operator&(Type a, Type b) { \
        return static_cast<Type>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b)); \
    } \
    RT_DEVICE_FUNCTION inline Type &operator&=(Type &a, Type b) { \
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

#undef OPTIXU_DEFINE_OPERATORS_FOR_FLAGS



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

#if defined(OPTIXU_ENABLE_ASSERT)
#   if defined(__CUDA_ARCH__)
#       define optixuAssert(expr, fmt, ...) \
            do { \
                if (!(expr)) { \
                    printf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
                    printf(fmt"\n", ##__VA_ARGS__); \
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
#else
#   define optixuAssert(expr, fmt, ...)
#endif

#define optixuAssert_ShouldNotBeCalled() optixuAssert(false, "Should not be called!")
#define optixuAssert_NotImplemented() optixuAssert(false, "Not implemented yet!")

    namespace detail {
        static constexpr uint32_t maxNumPayloadsInDwords = 32;
#define OPTIXU_STR_MAX_NUM_PAYLOADS "32"

        template <typename T>
        RT_DEVICE_FUNCTION constexpr size_t getNumDwords() {
            return (sizeof(T) + 3) / 4;
        }

        template <typename... Types>
        RT_DEVICE_FUNCTION constexpr size_t calcSumDwords() {
            return (0 + ... + getNumDwords<Types>());
        }
    }



    // ----------------------------------------------------------------
    // JP: ホスト・デバイス共有のクラス定義
    // EN: Definitions of Host-/Device-shared classes

    template <typename FuncType>
    class DirectCallableProgramID;

    template <typename ReturnType, typename... ArgTypes>
    class DirectCallableProgramID<ReturnType(ArgTypes...)> {
        uint32_t m_sbtIndex;

    public:
        RT_DEVICE_FUNCTION DirectCallableProgramID() {}
        RT_DEVICE_FUNCTION explicit DirectCallableProgramID(uint32_t sbtIndex) : m_sbtIndex(sbtIndex) {}
        RT_DEVICE_FUNCTION explicit operator uint32_t() const { return m_sbtIndex; }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        RT_DEVICE_FUNCTION ReturnType operator()(const ArgTypes &... args) const {
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
        RT_DEVICE_FUNCTION ContinuationCallableProgramID() {}
        RT_DEVICE_FUNCTION explicit ContinuationCallableProgramID(uint32_t sbtIndex) : m_sbtIndex(sbtIndex) {}
        RT_DEVICE_FUNCTION explicit operator uint32_t() const { return m_sbtIndex; }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        RT_DEVICE_FUNCTION ReturnType operator()(const ArgTypes &... args) const {
            return optixContinuationCall<ReturnType, ArgTypes...>(m_sbtIndex, args...);
        }
#endif
    };

    // END: Definitions of Host-/Device-shared classes
    // ----------------------------------------------------------------




    // ----------------------------------------------------------------
    // JP: デバイス関数のラッパー
    // EN: Device-side function wrappers
#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

    namespace detail {
        template <uint32_t start, typename HeadType, typename... TailTypes>
        RT_DEVICE_FUNCTION void packToUInts(uint32_t* v, const HeadType &head, const TailTypes &... tails) {
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
        RT_DEVICE_FUNCTION void getValue(Type* value) {
            if (!value)
                return;
            *(reinterpret_cast<uint32_t*>(value) + offsetInDst) = Func::get<srcSlot>();
            if constexpr (offsetInDst + 1 < getNumDwords<Type>())
                getValue<Func, Type, offsetInDst + 1, srcSlot + 1>(value);
        }

        template <typename Func, uint32_t srcStartSlot, typename HeadType, typename... TailTypes>
        RT_DEVICE_FUNCTION void getValues(HeadType* head, TailTypes*... tails) {
            static_assert(sizeof(HeadType) % sizeof(uint32_t) == 0,
                          "Value type of size not multiple of Dword is not supported.");
            getValue<Func, HeadType, 0, srcStartSlot>(head);
            if constexpr (sizeof...(tails) > 0)
                getValues<Func, srcStartSlot + getNumDwords<HeadType>()>(tails...);
        }

        template <typename Func, typename Type, uint32_t offsetInSrc, uint32_t dstSlot>
        RT_DEVICE_FUNCTION void setValue(const Type* value) {
            if (!value)
                return;
            Func::set<dstSlot>(*(reinterpret_cast<const uint32_t*>(value) + offsetInSrc));
            if constexpr (offsetInSrc + 1 < getNumDwords<Type>())
                setValue<Func, Type, offsetInSrc + 1, dstSlot + 1>(value);
        }

        template <typename Func, uint32_t dstStartSlot, typename HeadType, typename... TailTypes>
        RT_DEVICE_FUNCTION void setValues(const HeadType* head, const TailTypes*... tails) {
            static_assert(sizeof(HeadType) % sizeof(uint32_t) == 0,
                          "Value type of size not multiple of Dword is not supported.");
            setValue<Func, HeadType, 0, dstStartSlot>(head);
            if constexpr (sizeof...(tails) > 0)
                setValues<Func, dstStartSlot + getNumDwords<HeadType>()>(tails...);
        }

        template <uint32_t startSlot, typename HeadType, typename... TailTypes>
        RT_DEVICE_FUNCTION void traceSetPayloads(uint32_t** p, HeadType &headPayload, TailTypes &... tailPayloads) {
            static_assert(sizeof(HeadType) % sizeof(uint32_t) == 0,
                          "Payload type of size not multiple of Dword is not supported.");
            constexpr uint32_t numDwords = getNumDwords<HeadType>();
#pragma unroll
            for (int i = 0; i < numDwords; ++i)
                p[startSlot + i] = reinterpret_cast<uint32_t*>(&headPayload) + i;
            if constexpr (sizeof...(tailPayloads) > 0)
                traceSetPayloads<startSlot + numDwords>(p, tailPayloads...);
        }

        template <size_t... I>
        RT_DEVICE_FUNCTION void trace(OptixTraversableHandle handle,
                                      const float3 &origin, const float3 &direction,
                                      float tmin, float tmax, float rayTime,
                                      OptixVisibilityMask visibilityMask, OptixRayFlags rayFlags,
                                      uint32_t SBToffset, uint32_t SBTstride, uint32_t missSBTIndex,
                                      uint32_t** payloads,
                                      std::index_sequence<I...>) {
            optixTrace(handle,
                       origin, direction,
                       tmin, tmax, rayTime,
                       visibilityMask, rayFlags,
                       SBToffset, SBTstride, missSBTIndex,
                       *payloads[I]...);
        }

        template <size_t... I>
        RT_DEVICE_FUNCTION void reportIntersection(float hitT, uint32_t hitKind, const uint32_t* attributes,
                                                   std::index_sequence<I...>) {
            optixReportIntersection(hitT, hitKind, attributes[I]...);
        }

        template <size_t... I>
        RT_DEVICE_FUNCTION void throwException(int32_t exceptionCode, const uint32_t* exDetails,
                                               std::index_sequence<I...>) {
            optixThrowException(exceptionCode, exDetails[I]...);
        }

        struct PayloadFunc {
            template <uint32_t index>
            RT_DEVICE_FUNCTION static uint32_t get() {
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
            RT_DEVICE_FUNCTION static void set(uint32_t p) {
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
            RT_DEVICE_FUNCTION static uint32_t get() {
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

        struct ExceptionDetailFunc {
            template <uint32_t index>
            RT_DEVICE_FUNCTION static uint32_t get() {
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

    // JP: 右辺値参照でペイロードを受け取れば右辺値も受け取れて、かつ値の書き換えも反映できる。
    //     が、optixTraceに仕様をあわせることと、テンプレート引数の整合性チェックを簡単にするためただの参照で受け取る。
    // EN: Taking payloads as rvalue reference makes it possible to take rvalue while reflecting value changes.
    //     However take them as normal reference to ease consistency check of template arguments and for
    //     conforming optixTrace.
    template <typename... PayloadTypes>
    RT_DEVICE_FUNCTION void trace(OptixTraversableHandle handle,
                                  const float3 &origin, const float3 &direction,
                                  float tmin, float tmax, float rayTime,
                                  OptixVisibilityMask visibilityMask, OptixRayFlags rayFlags,
                                  uint32_t SBToffset, uint32_t SBTstride, uint32_t missSBTIndex,
                                  PayloadTypes &... payloads) {
        constexpr size_t numDwords = detail::calcSumDwords<PayloadTypes...>();
        static_assert(numDwords <= detail::maxNumPayloadsInDwords,
                      "Maximum number of payloads is " OPTIXU_STR_MAX_NUM_PAYLOADS " in dwords.");
        if constexpr (numDwords == 0) {
            optixTrace(handle,
                       origin, direction,
                       tmin, tmax, rayTime,
                       visibilityMask, rayFlags,
                       SBToffset, SBTstride, missSBTIndex);
        }
        else {
            uint32_t* p[numDwords];
            detail::traceSetPayloads<0>(p, payloads...);
            detail::trace(handle,
                          origin, direction,
                          tmin, tmax, rayTime,
                          visibilityMask, rayFlags,
                          SBToffset, SBTstride, missSBTIndex,
                          p, std::make_index_sequence<numDwords>{});
        }
    }



    template <typename... PayloadTypes>
    RT_DEVICE_FUNCTION void getPayloads(PayloadTypes*... payloads) {
        constexpr size_t numDwords = detail::calcSumDwords<PayloadTypes...>();
        static_assert(numDwords <= detail::maxNumPayloadsInDwords,
                      "Maximum number of payloads is " OPTIXU_STR_MAX_NUM_PAYLOADS " in dwords.");
        static_assert(numDwords > 0, "Calling this function without payloads has no effect.");
        if constexpr (numDwords > 0)
            detail::getValues<detail::PayloadFunc, 0>(payloads...);
    }

    template <typename... PayloadTypes>
    RT_DEVICE_FUNCTION void setPayloads(const PayloadTypes*... payloads) {
        constexpr size_t numDwords = detail::calcSumDwords<PayloadTypes...>();
        static_assert(numDwords <= detail::maxNumPayloadsInDwords,
                      "Maximum number of payloads is " OPTIXU_STR_MAX_NUM_PAYLOADS " in dwords.");
        static_assert(numDwords > 0, "Calling this function without payloads has no effect.");
        if constexpr (numDwords > 0)
            detail::setValues<detail::PayloadFunc, 0>(payloads...);
    }



    template <typename... AttributeTypes>
    RT_DEVICE_FUNCTION void reportIntersection(float hitT, uint32_t hitKind,
                                               const AttributeTypes &... attributes) {
        constexpr size_t numDwords = detail::calcSumDwords<AttributeTypes...>();
        static_assert(numDwords <= 8, "Maximum number of attributes is 8 dwords.");
        if constexpr (numDwords == 0) {
            optixReportIntersection(hitT, hitKind);
        }
        else {
            uint32_t a[numDwords];
            detail::packToUInts<0>(a, attributes...);
            detail::reportIntersection(hitT, hitKind, a, std::make_index_sequence<numDwords>{});
        }
    }

    template <typename... AttributeTypes>
    RT_DEVICE_FUNCTION void getAttributes(AttributeTypes*... attributes) {
        constexpr size_t numDwords = detail::calcSumDwords<AttributeTypes...>();
        static_assert(numDwords <= 8, "Maximum number of attributes is 8 dwords.");
        static_assert(numDwords > 0, "Calling this function without attributes has no effect.");
        if constexpr (numDwords > 0)
            detail::getValues<detail::AttributeFunc, 0>(attributes...);
    }



    template <typename... ExceptionDetailTypes>
    RT_DEVICE_FUNCTION void throwException(int32_t exceptionCode,
                                           const ExceptionDetailTypes &... exceptionDetails) {
        constexpr size_t numDwords = detail::calcSumDwords<ExceptionDetailTypes...>();
        static_assert(numDwords <= 8, "Maximum number of exception details is 8 dwords.");
        if constexpr (numDwords == 0) {
            optixThrowException(exceptionCode);
        }
        else {
            uint32_t ed[numDwords];
            detail::packToUInts<0>(ed, exceptionDetails...);
            detail::throwException(exceptionCode, ed, std::make_index_sequence<numDwords>{});
        }
    }

    template <typename... ExceptionDetailTypes>
    RT_DEVICE_FUNCTION void getExceptionDetails(ExceptionDetailTypes*... details) {
        constexpr size_t numDwords = detail::calcSumDwords<ExceptionDetailTypes...>();
        static_assert(numDwords <= 8, "Maximum number of exception details is 8 dwords.");
        static_assert(numDwords > 0, "Calling this function without exception details has no effect.");
        if constexpr (numDwords > 0)
            detail::getValues<detail::ExceptionDetailFunc, 0>(details...);
    }

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
              |              +-- ProgramGroup
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
    OPTIXU_PREPROCESS_OBJECT(Context); \
    OPTIXU_PREPROCESS_OBJECT(Material); \
    OPTIXU_PREPROCESS_OBJECT(Scene); \
    OPTIXU_PREPROCESS_OBJECT(GeometryInstance); \
    OPTIXU_PREPROCESS_OBJECT(GeometryAccelerationStructure); \
    OPTIXU_PREPROCESS_OBJECT(Transform); \
    OPTIXU_PREPROCESS_OBJECT(Instance); \
    OPTIXU_PREPROCESS_OBJECT(InstanceAccelerationStructure); \
    OPTIXU_PREPROCESS_OBJECT(Pipeline); \
    OPTIXU_PREPROCESS_OBJECT(Module); \
    OPTIXU_PREPROCESS_OBJECT(ProgramGroup); \
    OPTIXU_PREPROCESS_OBJECT(Denoiser);

    // Forward Declarations
#define OPTIXU_PREPROCESS_OBJECT(Type) class Type
    OPTIXU_PREPROCESS_OBJECTS();
#undef OPTIXU_PREPROCESS_OBJECT

    enum class GeometryType {
        Triangles = 0,
        LinearSegments,
        QuadraticBSplines,
        CubicBSplines,
        CatmullRomSplines,
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



#define OPTIXU_PIMPL() \
public: \
    class Priv; \
private: \
    Priv* m = nullptr

#define OPTIXU_COMMON_FUNCTIONS(SelfType) \
    operator bool() const { return m; } \
    bool operator==(const SelfType &r) const { return m == r.m; } \
    bool operator!=(const SelfType &r) const { return m != r.m; } \
    bool operator<(const SelfType &r) const { \
        static_assert(std::is_same<decltype(r), decltype(*this)>::value, \
                      "This function can be defined only for the self type."); \
        return m < r.m; \
    } \
    Context getContext() const; \
    void setName(const std::string &name) const; \
    const char* getName() const;



    class Context {
        OPTIXU_PIMPL();

    public:
        [[nodiscard]]
        static Context create(CUcontext cuContext, uint32_t logLevel = 4, bool enableValidation = false);
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Context);

        CUcontext getCUcontext() const;

        void setLogCallback(OptixLogCallback callback, void* callbackData, uint32_t logLevel) const;

        [[nodiscard]]
        Pipeline createPipeline() const;
        [[nodiscard]]
        Material createMaterial() const;
        [[nodiscard]]
        Scene createScene() const;
        [[nodiscard]]
        Denoiser createDenoiser(OptixDenoiserModelKind modelKind, bool guideAlbedo, bool guideNormal) const;
    };



    class Material {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Material);

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
        void setHitGroup(uint32_t rayType, ProgramGroup hitGroup) const;
        void setUserData(const void* data, uint32_t size, uint32_t alignment) const;
        template <typename T>
        void setUserData(const T &data) const {
            setUserData(&data, sizeof(T), alignof(T));
        }

        ProgramGroup getHitGroup(Pipeline pipeline, uint32_t rayType) const;
        void getUserData(void* data, uint32_t* size, uint32_t* alignment) const;
        template <typename T>
        void getUserData(T* data, uint32_t* size = nullptr, uint32_t* alignment = nullptr) const {
            getUserData(reinterpret_cast<void*>(data), size, alignment);
        }
    };



    class Scene {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Scene);

        [[nodiscard]]
        GeometryInstance createGeometryInstance(GeometryType geomType = GeometryType::Triangles) const;
        [[nodiscard]]
        GeometryAccelerationStructure createGeometryAccelerationStructure(GeometryType geomType = GeometryType::Triangles) const;
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



    class GeometryInstance {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(GeometryInstance);

        // JP: 以下のAPIを呼んだ場合は所属するGASのmarkDirty()を呼ぶ必要がある。
        //     (頂点/Width/AABBバッファーの変更のみの場合は、markDirty()を呼ばずにGASのアップデートだけでも良い。)
        // EN: Calling markDirty() of a GAS to which the geometry instance belongs is
        //     required when calling the following APIs.
        //     (It is okay to use update instead of calling markDirty() when changing only vertex/width/AABB buffer.)
        void setNumMotionSteps(uint32_t n) const;
        void setVertexFormat(OptixVertexFormat format) const;
        void setVertexBuffer(const BufferView &vertexBuffer, uint32_t motionStep = 0) const;
        void setWidthBuffer(const BufferView &widthBuffer, uint32_t motionStep = 0) const;
        void setTriangleBuffer(const BufferView &triangleBuffer, OptixIndicesFormat format = OPTIX_INDICES_FORMAT_UNSIGNED_INT3) const;
        void setSegmentIndexBuffer(const BufferView &segmentIndexBuffer) const;
        void setCurveEndcapFlags(OptixCurveEndcapFlags endcapFlags) const;
        void setCustomPrimitiveAABBBuffer(const BufferView &primitiveAABBBuffer, uint32_t motionStep = 0) const;
        void setPrimitiveIndexOffset(uint32_t offset) const;
        void setNumMaterials(uint32_t numMaterials, const BufferView &matIndexBuffer, uint32_t indexSize = sizeof(uint32_t)) const;
        void setGeometryFlags(uint32_t matIdx, OptixGeometryFlags flags) const;

        // JP: 以下のAPIを呼んだ場合はシェーダーバインディングテーブルを更新する必要がある。
        //     パイプラインのmarkHitGroupShaderBindingTableDirty()を呼べばローンチ時にセットアップされる。
        //     シェーダーバインディングテーブルのレイアウト生成後に、再度ユーザーデータのサイズや
        //     アラインメントを変更する場合レイアウトが自動で無効化される。
        // EN: Updating a shader binding table is required when calling the following APIs.
        //     Calling pipeline's markHitGroupShaderBindingTableDirty() triggers re-setup of the table at launch.
        //     In the case where user data size and/or alignment changes again after generating the layout of
        //     a shader binding table, the layout is automatically invalidated.
        void setMaterial(uint32_t matSetIdx, uint32_t matIdx, Material mat) const;
        void setUserData(const void* data, uint32_t size, uint32_t alignment) const;
        template <typename T>
        void setUserData(const T &data) const {
            setUserData(&data, sizeof(T), alignof(T));
        }

        uint32_t getNumMotionSteps() const;
        OptixVertexFormat getVertexFormat() const;
        BufferView getVertexBuffer(uint32_t motionStep = 0);
        BufferView getWidthBuffer(uint32_t motionStep = 0);
        BufferView getTriangleBuffer(OptixIndicesFormat* format = nullptr) const;
        BufferView getSegmentIndexBuffer() const;
        BufferView getCustomPrimitiveAABBBuffer(uint32_t motionStep = 0) const;
        uint32_t getPrimitiveIndexOffset() const;
        uint32_t getNumMaterials(BufferView* matIndexBuffer = nullptr, uint32_t* indexSize = nullptr) const;
        OptixGeometryFlags getGeometryFlags(uint32_t matIdx) const;
        Material getMaterial(uint32_t matSetIdx, uint32_t matIdx) const;
        void getUserData(void* data, uint32_t* size, uint32_t* alignment) const;
        template <typename T>
        void getUserData(T* data, uint32_t* size = nullptr, uint32_t* alignment = nullptr) const {
            getUserData(reinterpret_cast<void*>(data), size, alignment);
        }
    };



    class GeometryAccelerationStructure {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(GeometryAccelerationStructure);

        // JP: 以下のAPIを呼んだ場合はGASが自動でdirty状態になる。
        //     子の数が変更される場合はヒットグループのシェーダーバインディングテーブルレイアウトも無効化される。
        // EN: Calling the following APIs automatically marks the GAS dirty.
        //     Changing the number of children invalidates the shader binding table layout of hit group.
        void setConfiguration(ASTradeoff tradeoff, bool allowUpdate, bool allowCompaction, bool allowRandomVertexAccess) const;
        void setMotionOptions(uint32_t numKeys, float timeBegin, float timeEnd, OptixMotionFlags flags) const;
        void addChild(GeometryInstance geomInst, CUdeviceptr preTransform = 0,
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

        // JP: 以下のAPIを呼んだ場合はヒットグループのシェーダーバインディングテーブルレイアウトが自動で無効化される。
        // EN: Calling the following APIs automatically invalidates the shader binding table layout of hit group.
        void setNumMaterialSets(uint32_t numMatSets) const;
        void setNumRayTypes(uint32_t matSetIdx, uint32_t numRayTypes) const;

        // JP: リビルド・コンパクトを行った場合はこのGASが(間接的に)所属するTraversable (例: IAS)
        //     のmarkDirty()を呼ぶ必要がある。
        // EN: Calling markDirty() of a traversable (e.g. IAS) to which this GAS (indirectly) belongs
        //     is required when performing rebuild / compact.
        void prepareForBuild(OptixAccelBufferSizes* memoryRequirement) const;
        OptixTraversableHandle rebuild(CUstream stream, const BufferView &accelBuffer, const BufferView &scratchBuffer) const;
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

        // JP: 以下のAPIを呼んだ場合はシェーダーバインディングテーブルを更新する必要がある。
        //     パイプラインのmarkHitGroupShaderBindingTableDirty()を呼べばローンチ時にセットアップされる。
        //     シェーダーバインディングテーブルのレイアウト生成後に、再度ユーザーデータのサイズや
        //     アラインメントを変更する場合レイアウトが自動で無効化される。
        // EN: Updating a shader binding table is required when calling the following APIs.
        //     Calling pipeline's markHitGroupShaderBindingTableDirty() triggers re-setup of the table at launch.
        //     In the case where user data size and/or alignment changes again after generating the layout of
        //     a shader binding table, the layout is automatically invalidated.
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

        void getConfiguration(ASTradeoff* tradeOff, bool* allowUpdate, bool* allowCompaction, bool* allowRandomVertexAccess) const;
        void getMotionOptions(uint32_t* numKeys, float* timeBegin, float* timeEnd, OptixMotionFlags* flags) const;
        uint32_t getNumChildren() const;
        uint32_t findChildIndex(GeometryInstance geomInst, CUdeviceptr preTransform = 0) const;
        GeometryInstance getChild(uint32_t index, CUdeviceptr* preTransform = nullptr) const;
        uint32_t getNumMaterialSets() const;
        uint32_t getNumRayTypes(uint32_t matSetIdx) const;
        void getChildUserData(uint32_t index, void* data, uint32_t* size, uint32_t* alignment) const;
        template <typename T>
        void getChildUserData(uint32_t index, T* data, uint32_t* size = nullptr, uint32_t* alignment = nullptr) const {
            getChildUserData(index, reinterpret_cast<void*>(data), size, alignment);
        }
        void getUserData(void* data, uint32_t* size, uint32_t* alignment) const;
        template <typename T>
        void getUserData(T* data, uint32_t* size = nullptr, uint32_t* alignment = nullptr) const {
            getUserData(reinterpret_cast<void*>(data), size, alignment);
        }
    };



    class Transform {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Transform);

        // JP: 以下のAPIを呼んだ場合はTransformが自動でdirty状態になる。
        // EN: Calling the following APIs automatically marks the transform dirty.
        void setConfiguration(TransformType type, uint32_t numKeys,
                              size_t* transformSize) const;
        void setMotionOptions(float timeBegin, float timeEnd, OptixMotionFlags flags) const;
        void setMatrixMotionKey(uint32_t keyIdx, const float matrix[12]) const;
        void setSRTMotionKey(uint32_t keyIdx, const float scale[3], const float orientation[4], const float translation[3]) const;
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
        void getSRTMotionKey(uint32_t keyIdx, float scale[3], float orientation[4], float translation[3]) const;
        void getStaticTransform(float matrix[12]) const;
        ChildType getChildType() const;
        template <typename T>
        T getChild() const;
    };



    class Instance {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Instance);

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



    // TODO: インスタンスバッファーもユーザー管理にしたいため、rebuild()が今の形になっているが微妙かもしれない。
    //       インスタンスバッファーを内部で1つ持つようにすると、
    //       あるフレームでIASをビルド、次のフレームでインスタンスの追加がありリビルドの必要が生じた場合に
    //       1フレーム目のGPU処理の終了を待たないと危険という状況になってしまう。
    //       OptiX的にはASのビルド完了後にはインスタンスバッファーは不要となるが、
    //       アップデート処理はリビルド時に書かれたインスタンスバッファーの内容を期待しているため、
    //       基本的にインスタンスバッファーとASのメモリ(コンパクション版にもなり得る)は同じ寿命で扱ったほうが良さそう。
    class InstanceAccelerationStructure {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(InstanceAccelerationStructure);

        // JP: 以下のAPIを呼んだ場合はIASが自動でdirty状態になる。
        // EN: Calling the following APIs automatically marks the IAS dirty.
        void setConfiguration(ASTradeoff tradeoff, bool allowUpdate, bool allowCompaction, bool allowRandomInstanceAccess) const;
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
        OptixTraversableHandle rebuild(CUstream stream, const BufferView &instanceBuffer,
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

        void getConfiguration(ASTradeoff* tradeOff, bool* allowUpdate, bool* allowCompaction) const;
        void getMotionOptions(uint32_t* numKeys, float* timeBegin, float* timeEnd, OptixMotionFlags* flags) const;
        uint32_t getNumChildren() const;
        uint32_t findChildIndex(Instance instance) const;
        Instance getChild(uint32_t index) const;
    };



    class Pipeline {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Pipeline);

        void setPipelineOptions(uint32_t numPayloadValues, uint32_t numAttributeValues,
                                const char* launchParamsVariableName, size_t sizeOfLaunchParams,
                                bool useMotionBlur,
                                OptixTraversableGraphFlags traversableGraphFlags,
                                OptixExceptionFlags exceptionFlags,
                                OptixPrimitiveTypeFlags supportedPrimitiveTypeFlags) const;

        [[nodiscard]]
        Module createModuleFromPTXString(const std::string &ptxString, int32_t maxRegisterCount,
                                         OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel,
                                         OptixModuleCompileBoundValueEntry* boundValues = nullptr, uint32_t numBoundValues = 0) const;

        [[nodiscard]]
        ProgramGroup createRayGenProgram(Module module, const char* entryFunctionName) const;
        [[nodiscard]]
        ProgramGroup createExceptionProgram(Module module, const char* entryFunctionName) const;
        [[nodiscard]]
        ProgramGroup createMissProgram(Module module, const char* entryFunctionName) const;
        [[nodiscard]]
        ProgramGroup createHitProgramGroupForTriangleIS(
            Module module_CH, const char* entryFunctionNameCH,
            Module module_AH, const char* entryFunctionNameAH) const;
        [[nodiscard]]
        ProgramGroup createHitProgramGroupForCurveIS(
            OptixPrimitiveType curveType, OptixCurveEndcapFlags endcapFlags,
            Module module_CH, const char* entryFunctionNameCH,
            Module module_AH, const char* entryFunctionNameAH,
            ASTradeoff tradeoff, bool allowUpdate, bool allowCompaction, bool allowRandomVertexAccess) const;
        [[nodiscard]]
        ProgramGroup createHitProgramGroupForCustomIS(Module module_CH, const char* entryFunctionNameCH,
                                                      Module module_AH, const char* entryFunctionNameAH,
                                                      Module module_IS, const char* entryFunctionNameIS) const;
        [[nodiscard]]
        ProgramGroup createEmptyHitProgramGroup() const;
        [[nodiscard]]
        ProgramGroup createCallableProgramGroup(Module module_DC, const char* entryFunctionNameDC,
                                                Module module_CC, const char* entryFunctionNameCC) const;

        void link(uint32_t maxTraceDepth, OptixCompileDebugLevel debugLevel) const;

        // JP: 以下のAPIを呼んだ場合は(非ヒットグループの)シェーダーバインディングテーブルレイアウトが自動で無効化される。
        // EN: Calling the following APIs automatically invalidates the (non-hit group) shader binding table layout.
        void setNumMissRayTypes(uint32_t numMissRayTypes) const;
        void setNumCallablePrograms(uint32_t numCallablePrograms) const;

        void generateShaderBindingTableLayout(size_t* memorySize) const;

        // JP: 以下のAPIを呼んだ場合は(非ヒットグループの)シェーダーバインディングテーブルが自動でdirty状態になり
        //     ローンチ時に再セットアップされる。
        //     ただしローンチ時のセットアップはSBTバッファーの内容変更・転送を伴うので、
        //     非同期書き換えを行う場合は安全のためにはSBTバッファーをダブルバッファリングする必要がある。
        // EN: Calling the following API automatically marks the (non-hit group) shader binding table dirty
        //     then triggers re-setup of the table at launch.
        //     However note that the setup in the launch involves the change of the SBT buffer's contents
        //     and transfer, so double buffered SBT is required for safety
        //     in the case performing asynchronous update.
        void setRayGenerationProgram(ProgramGroup program) const;
        void setExceptionProgram(ProgramGroup program) const;
        void setMissProgram(uint32_t rayType, ProgramGroup program) const;
        void setCallableProgram(uint32_t index, ProgramGroup program) const;
        void setShaderBindingTable(const BufferView &shaderBindingTable, void* hostMem) const;

        // JP: 以下のAPIを呼んだ場合はヒットグループのシェーダーバインディングテーブルが自動でdirty状態になり
        //     ローンチ時に再セットアップされる。
        //     ただしローンチ時のセットアップはSBTバッファーの内容変更・転送を伴うので、
        //     非同期書き換えを行う場合は安全のためにはSBTバッファーをダブルバッファリングする必要がある。
        // EN: Calling the following APIs automatically marks the hit group's shader binding table dirty,
        //     then triggers re-setup of the table at launch.
        //     However note that the setup in the launch involves the change of the SBT buffer's contents
        //     and transfer, so double buffered SBT is required for safety
        //     in the case performing asynchronous update.
        void setScene(const Scene &scene) const;
        void setHitGroupShaderBindingTable(const BufferView &shaderBindingTable, void* hostMem) const;

        // JP: ヒットグループのシェーダーバインディングテーブルをdirty状態にする。
        // EN: Mark the hit group's shader binding table dirty.
        void markHitGroupShaderBindingTableDirty() const;

        void setStackSize(uint32_t directCallableStackSizeFromTraversal,
                          uint32_t directCallableStackSizeFromState,
                          uint32_t continuationStackSize,
                          uint32_t maxTraversableGraphDepth) const;

        // JP: セットされたシーンを基にシェーダーバインディングテーブルのセットアップを行い、
        //     Ray Generationシェーダーを起動する。
        // EN: Setup the shader binding table based on the scene set, then launch the ray generation shader.
        void launch(CUstream stream, CUdeviceptr plpOnDevice, uint32_t dimX, uint32_t dimY, uint32_t dimZ) const;
    };



    // JP: Moduleの寿命はそれを参照するあらゆるProgramGroupの寿命よりも長い必要がある。
    // EN: The lifetime of a module must extend to the lifetime of any ProgramGroup that reference that module.
    class Module {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Module);
    };



    class ProgramGroup {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(ProgramGroup);

        void getStackSize(OptixStackSizes* sizes) const;
    };



    class DenoisingTask {
        uint32_t placeHolder[6];

        // TODO: ? implement a function to query required window (tile + overlap).
    };

    class Denoiser {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Denoiser);

        void prepare(uint32_t imageWidth, uint32_t imageHeight, uint32_t tileWidth, uint32_t tileHeight,
                     size_t* stateBufferSize, size_t* scratchBufferSize, size_t* scratchBufferSizeForComputeIntensity,
                     uint32_t* numTasks) const;
        void getTasks(DenoisingTask* tasks) const;
        void setupState(CUstream stream, const BufferView &stateBuffer, const BufferView &scratchBuffer) const;

        void computeIntensity(CUstream stream,
                              const BufferView &noisyBeauty, OptixPixelFormat beautyFormat,
                              const BufferView &scratchBuffer, CUdeviceptr outputIntensity) const;
        void computeAverageColor(CUstream stream,
                                 const BufferView &noisyBeauty, OptixPixelFormat beautyFormat,
                                 const BufferView &scratchBuffer, CUdeviceptr outputAverageColor) const;
        void invoke(CUstream stream,
                    bool denoiseAlpha, CUdeviceptr hdrIntensity, float blendFactor,
                    const BufferView &noisyBeauty, OptixPixelFormat beautyFormat,
                    const BufferView &albedo, OptixPixelFormat albedoFormat,
                    const BufferView &normal, OptixPixelFormat normalFormat,
                    const BufferView &flow, OptixPixelFormat flowFormat,
                    const BufferView &previousDenoisedBeauty,
                    const BufferView &denoisedBeauty,
                    const DenoisingTask &task) const;
        // JP: AOVデノイザー用。
        // EN: For AOV denoiser.
        void invoke(CUstream stream,
                    bool denoiseAlpha, CUdeviceptr hdrAverageColor, float blendFactor,
                    const BufferView &noisyBeauty, OptixPixelFormat beautyFormat,
                    const BufferView* noisyAovs, OptixPixelFormat* aovFormats, uint32_t numAovs,
                    const BufferView &albedo, OptixPixelFormat albedoFormat,
                    const BufferView &normal, OptixPixelFormat normalFormat,
                    const BufferView &flow, OptixPixelFormat flowFormat,
                    const BufferView &previousDenoisedBeauty, const BufferView* previousDenoisedAovs,
                    const BufferView &denoisedBeauty, const BufferView* denoisedAovs,
                    const DenoisingTask &task) const;
    };



#undef OPTIXU_COMMON_FUNCTIONS
#undef OPTIXU_PIMPL

#endif // #if !defined(__CUDA_ARCH__)
    // END: Host-side API
    // ----------------------------------------------------------------
} // namespace optixu
