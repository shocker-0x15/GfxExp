/*

   Copyright 2025 Shin Watanabe

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

#include "optix_util.h"

#if defined(OPTIXU_Platform_Windows_MSVC)
#   define _USE_MATH_DEFINES
#   include <Windows.h>
#   undef min
#   undef max
#   undef near
#   undef far
#   undef RGB
#endif // if defined(OPTIXU_Platform_Windows_MSVC)

#if defined(OPTIXU_Platform_Windows_MSVC)
#   pragma warning(push)
#   pragma warning(disable:4819)
#endif // if defined(OPTIXU_Platform_Windows_MSVC)
#include <optix_function_table_definition.h>
#include <cuda.h>
#if defined(OPTIXU_Platform_Windows_MSVC)
#   pragma warning(pop)
#endif // if defined(OPTIXU_Platform_Windows_MSVC)

#include <sstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <variant>

#if __cplusplus <= 199711L
#   if defined(OPTIXU_Platform_Windows_MSVC)
#       pragma message("\"/Zc:__cplusplus\" compiler option to enable the updated __cplusplus definition is recommended.")
#   else
#       pragma message("Enabling the updated __cplusplus definition is recommended.")
#   endif
#endif // if __cplusplus <= 199711L

#if __cplusplus >= 202002L
#include <bit>
#else // if __cplusplus >= 202002L
#include <intrin.h>
#endif // if __cplusplus >= 202002L

#include <stdexcept>

#define CUDADRV_CHECK(call) \
    do { \
        const CUresult error = call; \
        if (error != CUDA_SUCCESS) { \
            std::stringstream ss; \
            const char* errMsg = "failed to get an error message."; \
            cuGetErrorString(error, &errMsg); \
            ss << "CUDA call (" << #call << " ) failed with error: '" \
               << errMsg \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)

#define OPTIX_CHECK(call) \
    do { \
        const OptixResult error = call; \
        if (error != OPTIX_SUCCESS) { \
            std::stringstream ss; \
            ss << "OptiX call (" << #call << ") failed: " \
               << "(" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)

#define OPTIX_CHECK_LOG(call) \
    do { \
        const OptixResult error = call; \
        if (error != OPTIX_SUCCESS) { \
            std::stringstream ss; \
            ss << "OptiX call (" << #call << ") failed: " \
               << "(" __FILE__ << ":" << __LINE__ << ")\n" \
               << "Log: " << log << (logSize > sizeof(log) ? "<TRUNCATED>" : "") \
               << "\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)



namespace optixu {
    template <typename... Types>
    static void _throwRuntimeError(const char* fmt, const Types &... args) {
        char str[2048];
        snprintf(str, sizeof(str), fmt, args...);
        throw std::runtime_error(str);
    }

    static void logCallBack(uint32_t level, const char* tag, const char* message, void* cbdata) {
        optixuPrintf("[%2u][%12s]: %s\n", level, tag, message);
    }

    static constexpr size_t s_maxMaterialUserDataSize = 512;
    static constexpr size_t s_maxGeometryInstanceUserDataSize = 512;
    static constexpr size_t s_maxGASChildUserDataSize = 512;
    static constexpr size_t s_maxGASUserDataSize = 512;



#if __cplusplus < 202002L
    inline uint32_t countr_zero(uint32_t x) {
        return _tzcnt_u32(x);
    }
#else // if __cplusplus < 202002L
    using std::countr_zero;
#endif // if __cplusplus < 202002L



    using _Context = Context::Priv;

    // Alias private classes.
#define OPTIXU_PREPROCESS_OBJECT(Name) using _ ## Name = Name::Priv
    OPTIXU_PREPROCESS_OBJECTS();
#undef OPTIXU_PREPROCESS_OBJECT



#define OPTIXU_OPAQUE_BRIDGE(BaseName) \
    friend class BaseName; \
\
    BaseName getPublicType() { \
        BaseName ret; \
        ret.m = this; \
        return ret; \
    } \
    static BaseName::Priv* extract(BaseName publicType) { \
        return publicType.m; \
    }

    template <typename PublicType>
    static typename PublicType::Priv* extract(const PublicType &obj) {
        return PublicType::Priv::extract(obj);
    }

#if OPTIXU_ENABLE_RUNTIME_ERROR
#   define OPTIXU_DEFINE_THROW_RUNTIME_ERROR(TypeName) \
        template <typename... Types> \
        void throwRuntimeError(bool expr, const char* fmt, const Types &... args) const { \
            if (expr) \
                return; \
\
            std::stringstream ss; \
            ss << TypeName ## " " << getName() << ": " << fmt; \
            optixu::_throwRuntimeError(ss.str().c_str(), args...); \
        }
#else // if OPTIXU_ENABLE_RUNTIME_ERROR
#   define OPTIXU_DEFINE_THROW_RUNTIME_ERROR(TypeName) \
        template <typename... Types> \
        void throwRuntimeError(bool, const char*, const Types &...) const {}
#endif // if OPTIXU_ENABLE_RUNTIME_ERROR



    struct SizeAlign {
        uint32_t size;
        uint32_t alignment;

        constexpr SizeAlign() : size(0), alignment(1) {}
        constexpr SizeAlign(uint32_t s, uint32_t a) : size(s), alignment(a) {}

        constexpr SizeAlign &add(const SizeAlign &sa, uint32_t* offset) {
            const uint32_t mask = sa.alignment - 1;
            alignment = std::max(alignment, sa.alignment);
            size = (size + mask) & ~mask;
            if (offset)
                *offset = size;
            size += sa.size;
            return *this;
        }
        constexpr SizeAlign &operator+=(const SizeAlign &sa) {
            return add(sa, nullptr);
        }
        constexpr SizeAlign &alignUp() {
            const uint32_t mask = alignment - 1;
            size = (size + mask) & ~mask;
            return *this;
        }
    };

    static constexpr inline SizeAlign max(const SizeAlign &sa0, const SizeAlign &sa1) {
        return SizeAlign{ std::max(sa0.size, sa1.size), std::max(sa0.alignment, sa1.alignment) };
    }



    static constexpr inline IndexSize convertToIndexSizeEnum(uint32_t indexSize) {
        return indexSize > 0 ? static_cast<IndexSize>(countr_zero(indexSize)) : IndexSize::None;
    }



    class Context::Priv {
        CUcontext cuContext;
        OptixDeviceContext rawContext;
        uint32_t rtCoreVersion;
        uint32_t shaderExecutionReorderingFlags;
        uint32_t maxInstanceID;
        uint32_t numVisibilityMaskBits;
        std::unordered_map<const void*, std::string> registeredNames;

    public:
        OPTIXU_OPAQUE_BRIDGE(Context);

        Priv(CUcontext _cuContext, uint32_t logLevel, EnableValidation enableValidation) :
            cuContext(_cuContext)
        {
            throwRuntimeError(logLevel <= 4, "Valid range for logLevel is [0, 4].");
            OPTIX_CHECK(optixInit());

            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &logCallBack;
            options.logCallbackData = nullptr;
            options.logCallbackLevel = logLevel;
            options.validationMode = enableValidation ?
                OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL :
                OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
            OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &rawContext));
            OPTIX_CHECK(optixDeviceContextGetProperty(
                rawContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID,
                &maxInstanceID, sizeof(maxInstanceID)));
            OPTIX_CHECK(optixDeviceContextGetProperty(
                rawContext, OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK,
                &numVisibilityMaskBits, sizeof(numVisibilityMaskBits)));
            OPTIX_CHECK(optixDeviceContextGetProperty(
                rawContext, OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
                &rtCoreVersion, sizeof(rtCoreVersion)));
            OPTIX_CHECK(optixDeviceContextGetProperty(
                rawContext, OPTIX_DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING,
                &shaderExecutionReorderingFlags, sizeof(shaderExecutionReorderingFlags)));
        }
        ~Priv() {
            optixDeviceContextDestroy(rawContext);
        }

        uint32_t getMaxInstanceID() const {
            return maxInstanceID;
        }
        uint32_t getNumVisibilityMaskBits() const {
            return numVisibilityMaskBits;
        }

        _Context* getContext() {
            return this;
        }
        OptixDeviceContext getRawContext() const {
            return rawContext;
        }

        void registerName(const void* p, const std::string &name) {
            optixuAssert(p, "Object must not be nullptr.");
            registeredNames[p] = name;
        }
        void unregisterName(const void* p) {
            optixuAssert(p, "Object must not be nullptr.");
            if (registeredNames.count(p) > 0)
                registeredNames.erase(p);
        }
        const char* getRegisteredName(const void* p) const {
            if (registeredNames.count(p) > 0)
                return registeredNames.at(p).c_str();
            return nullptr;
        }
        std::string getName(const void* p) const {
            const char* regName = getRegisteredName(p);
            if (regName) {
                return regName;
            }
            else {
                char ptrStr[32];
                sprintf_s(ptrStr, "%p", p);
                return ptrStr;
            }
        }

        void setName(const std::string &name) {
            registerName(this, name);
        }
        const char* getRegisteredName() const {
            return getRegisteredName(this);
        }
        std::string getName() const {
            const char* regName = getRegisteredName(this);
            if (regName) {
                return regName;
            }
            else {
                char ptrStr[32];
                sprintf_s(ptrStr, "%p", this);
                return ptrStr;
            }
        }

        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("Context");
    };



    class PrivateObject {
    public:
        virtual _Context* getContext() const = 0;
        OptixDeviceContext getRawContext() const {
            return getContext()->getRawContext();
        }

        void setName(const std::string &name) const {
            getContext()->registerName(this, name);
        }
        const char* getRegisteredName() const {
            return getContext()->getRegisteredName(this);
        }
        std::string getName() const {
            return getContext()->getName(this);
        }
    };



    template <>
    class Object<Material>::Priv : public PrivateObject {
        struct Key {
            const _Pipeline* pipeline;
            uint32_t rayType;

            bool operator<(const Key &rKey) const {
                if (pipeline < rKey.pipeline) {
                    return true;
                }
                else if (pipeline == rKey.pipeline) {
                    if (rayType < rKey.rayType)
                        return true;
                }
                return false;
            }

            struct Hash {
                typedef std::size_t result_type;

                std::size_t operator()(const Key &key) const {
                    size_t seed = 0;
                    const auto hash0 = std::hash<const _Pipeline*>()(key.pipeline);
                    const auto hash1 = std::hash<uint32_t>()(key.rayType);
                    seed ^= hash0 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    seed ^= hash1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    return seed;
                }
            };
            bool operator==(const Key &rKey) const {
                return pipeline == rKey.pipeline && rayType == rKey.rayType;
            }
        };

        _Context* context;
        SizeAlign userDataSizeAlign;
        std::vector<uint8_t> userData;

        std::unordered_map<Key, _HitProgramGroup*, Key::Hash> programs;

    public:
        OPTIXU_OPAQUE_BRIDGE(Material);

        Priv(_Context* ctxt) :
            context(ctxt), userData(sizeof(uint32_t))
        {}
        ~Priv() {
            context->unregisterName(this);
        }

        _Context* getContext() const {
            return context;
        }
        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("Material");

        SizeAlign getUserDataSizeAlign() const {
            return userDataSizeAlign;
        }
        void setRecordHeader(
            const _Pipeline* pipeline, uint32_t rayType, uint8_t* record, SizeAlign* curSizeAlign) const;
        void setRecordData(uint8_t* record, SizeAlign* curSizeAlign) const;
    };



    template <>
    class Object<Scene>::Priv : public PrivateObject {
        struct SBTOffsetKey {
            uint32_t gasSerialID;
            uint32_t matSetIndex;

            bool operator<(const SBTOffsetKey &rKey) const {
                if (gasSerialID < rKey.gasSerialID) {
                    return true;
                }
                else if (gasSerialID == rKey.gasSerialID) {
                    if (matSetIndex < rKey.matSetIndex)
                        return true;
                }
                return false;
            }

            struct Hash {
                typedef std::size_t result_type;

                std::size_t operator()(const SBTOffsetKey &key) const {
                    size_t seed = 0;
                    const auto hash0 = std::hash<uint32_t>()(key.gasSerialID);
                    const auto hash1 = std::hash<uint32_t>()(key.matSetIndex);
                    seed ^= hash0 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    seed ^= hash1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    return seed;
                }
            };
            bool operator==(const SBTOffsetKey &rKey) const {
                return gasSerialID == rKey.gasSerialID && matSetIndex == rKey.matSetIndex;
            }
        };

        _Context* context;
        std::unordered_map<uint32_t, _GeometryAccelerationStructure*> geomASs;
        std::unordered_map<SBTOffsetKey, uint32_t, SBTOffsetKey::Hash> sbtOffsets;
        uint32_t nextGeomASSerialID;
        uint32_t singleRecordSize;
        uint32_t numSBTRecords;
        std::unordered_set<_Transform*> transforms;
        std::unordered_set<_InstanceAccelerationStructure*> instASs;
        uint32_t sbtLayoutIsUpToDate : 1;

    public:
        OPTIXU_OPAQUE_BRIDGE(Scene);

        Priv(_Context* ctxt) : context(ctxt),
            nextGeomASSerialID(0),
            singleRecordSize(OPTIX_SBT_RECORD_HEADER_SIZE), numSBTRecords(0),
            sbtLayoutIsUpToDate(false)
        {}
        ~Priv() {
            context->unregisterName(this);
        }

        _Context* getContext() const {
            return context;
        }
        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("Scene");



        void addGAS(_GeometryAccelerationStructure* gas);
        void removeGAS(_GeometryAccelerationStructure* gas);
        void addTransform(_Transform* tr) {
            transforms.insert(tr);
        }
        void removeTransform(_Transform* tr) {
            transforms.erase(tr);
        }
        void addIAS(_InstanceAccelerationStructure* ias) {
            instASs.insert(ias);
        }
        void removeIAS(_InstanceAccelerationStructure* ias) {
            instASs.erase(ias);
        }

        bool sbtLayoutGenerationDone() const {
            return sbtLayoutIsUpToDate;
        }
        void markSBTLayoutDirty();
        uint32_t getSBTOffset(_GeometryAccelerationStructure* gas, uint32_t matSetIdx);

        uint32_t getSingleRecordSize() const {
            return singleRecordSize;
        }
        void setupHitGroupSBT(CUstream stream, const _Pipeline* pipeline, const BufferView &sbt, void* hostMem);

        bool isReady(bool* hasMotionGAS, bool* hasMotionIAS);
    };



    template <>
    class Object<OpacityMicroMapArray>::Priv : public PrivateObject {
        _Scene* scene;
        OptixOpacityMicromapFlags flags;

        BufferView rawOmmBuffer;
        BufferView perMicroMapDescBuffer;
        BufferView outputBuffer;
        std::vector<OptixOpacityMicromapHistogramEntry> microMapHistogramEntries;

        OptixOpacityMicromapArrayBuildInput buildInput;
        OptixMicromapBufferSizes memoryRequirement;

        uint32_t memoryUsageComputed : 1;
        uint32_t buffersSet : 1;
        uint32_t available : 1;

    public:
        OPTIXU_OPAQUE_BRIDGE(OpacityMicroMapArray);

        Priv(_Scene* _scene) :
            scene(_scene),
            memoryUsageComputed(false), buffersSet(false),
            available(false)
        {}
        ~Priv() {
            getContext()->unregisterName(this);
        }

        const _Scene* getScene() const {
            return scene;
        }
        _Context* getContext() const {
            return scene->getContext();
        }
        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("OMM");

        bool isReady() const {
            return available;
        }

        BufferView getBuffer() const {
            throwRuntimeError(outputBuffer.isValid(), "Output buffer has not been set.");
            return outputBuffer;
        }
    };



    template <>
    class Object<GeometryInstance>::Priv : public PrivateObject {
        _Scene* scene;
        SizeAlign userDataSizeAlign;
        std::vector<uint8_t> userData;

        struct TriangleGeometry {
            CUdeviceptr* vertexBufferArray;
            BufferView* vertexBuffers;
            BufferView triangleBuffer;
            BufferView materialIndexBuffer;
            OptixVertexFormat vertexFormat;
            OptixIndicesFormat indexFormat;

            _OpacityMicroMapArray* opacityMicroMapArray;
            BufferView opacityMicroMapIndexBuffer;
            std::vector<OptixOpacityMicromapUsageCount> opacityMicroMapUsageCounts;
            OptixOpacityMicromapArrayIndexingMode opacityMicroMapIndexingMode;
            uint32_t opacityMicroMapIndexOffset;

            uint32_t materialIndexSize : 3;
            uint32_t opacityMicroMapIndexSize : 3;
        };
        struct CurveGeometry {
            CUdeviceptr* vertexBufferArray;
            CUdeviceptr* widthBufferArray;
            CUdeviceptr* normalBufferArray;
            BufferView* vertexBuffers;
            BufferView* widthBuffers;
            BufferView* normalBuffers;
            BufferView segmentIndexBuffer;
            OptixCurveEndcapFlags endcapFlags;
        };
        struct SphereGeometry {
            CUdeviceptr* centerBufferArray;
            CUdeviceptr* radiusBufferArray;
            BufferView* centerBuffers;
            BufferView* radiusBuffers;
            BufferView materialIndexBuffer;
            uint32_t materialIndexSize : 3;
            uint32_t useSingleRadius : 1;
        };
        struct CustomPrimitiveGeometry {
            CUdeviceptr* primitiveAabbBufferArray;
            BufferView* primitiveAabbBuffers;
            BufferView materialIndexBuffer;
            uint32_t materialIndexSize : 3;
        };
        std::variant<
            TriangleGeometry,
            CurveGeometry,
            SphereGeometry,
            CustomPrimitiveGeometry
        > geometry;
        GeometryType geomType;
        uint32_t numMotionSteps;
        uint32_t primitiveIndexOffset;
        std::vector<OptixGeometryFlags> buildInputFlags; // per SBT record

        std::vector<std::vector<_Material*>> materials;

    public:
        OPTIXU_OPAQUE_BRIDGE(GeometryInstance);

        Priv(_Scene* _scene, GeometryType _geomType) :
            scene(_scene),
            userData(),
            geomType(_geomType),
            primitiveIndexOffset(0)
        {
            buildInputFlags.resize(1, OPTIX_GEOMETRY_FLAG_NONE);
            materials.resize(1);
            materials[0].resize(1, nullptr);

            numMotionSteps = 1;
            if (geomType == GeometryType::Triangles) {
                geometry = TriangleGeometry{};
                auto &geom = std::get<TriangleGeometry>(geometry);
                geom.vertexBufferArray = new CUdeviceptr[numMotionSteps];
                geom.vertexBuffers = new BufferView[numMotionSteps];
                geom.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                geom.indexFormat = OPTIX_INDICES_FORMAT_NONE;

                geom.opacityMicroMapArray = nullptr;
                geom.opacityMicroMapIndexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_NONE;
                geom.opacityMicroMapIndexOffset = 0;

                geom.materialIndexSize = 0;
                geom.opacityMicroMapIndexSize = 0;
            }
            else if (geomType == GeometryType::LinearSegments ||
                     geomType == GeometryType::QuadraticBSplines ||
                     geomType == GeometryType::QuadraticBSplineRocaps ||
                     geomType == GeometryType::FlatQuadraticBSplines ||
                     geomType == GeometryType::CubicBSplines ||
                     geomType == GeometryType::CubicBSplineRocaps ||
                     geomType == GeometryType::CatmullRomSplines ||
                     geomType == GeometryType::CatmullRomSplineRocaps ||
                     geomType == GeometryType::CubicBezier ||
                     geomType == GeometryType::CubicBezierRocaps) {
                geometry = CurveGeometry{};
                auto &geom = std::get<CurveGeometry>(geometry);
                geom.vertexBufferArray = new CUdeviceptr[numMotionSteps];
                geom.vertexBuffers = new BufferView[numMotionSteps];
                geom.widthBufferArray = new CUdeviceptr[numMotionSteps];
                geom.widthBuffers = new BufferView[numMotionSteps];
                if (geomType == GeometryType::FlatQuadraticBSplines) {
                    geom.normalBufferArray = new CUdeviceptr[numMotionSteps];
                    geom.normalBuffers = new BufferView[numMotionSteps];
                }
                geom.endcapFlags = OPTIX_CURVE_ENDCAP_DEFAULT;
            }
            else if (geomType == GeometryType::Spheres) {
                geometry = SphereGeometry{};
                auto &geom = std::get<SphereGeometry>(geometry);
                geom.centerBufferArray = new CUdeviceptr[numMotionSteps];
                geom.centerBuffers = new BufferView[numMotionSteps];
                geom.radiusBufferArray = new CUdeviceptr[numMotionSteps];
                geom.radiusBuffers = new BufferView[numMotionSteps];
                geom.useSingleRadius = false;
            }
            else if (geomType == GeometryType::CustomPrimitives) {
                geometry = CustomPrimitiveGeometry{};
                auto &geom = std::get<CustomPrimitiveGeometry>(geometry);
                geom.primitiveAabbBufferArray = new CUdeviceptr[numMotionSteps];
                geom.primitiveAabbBuffers = new BufferView[numMotionSteps];
                geom.materialIndexSize = 0;
            }
            else {
                optixuAssert_ShouldNotBeCalled();
            }
        }
        ~Priv() {
            if (std::holds_alternative<TriangleGeometry>(geometry)) {
                auto &geom = std::get<TriangleGeometry>(geometry);
                delete[] geom.vertexBuffers;
                delete[] geom.vertexBufferArray;
            }
            else if (std::holds_alternative<CurveGeometry>(geometry)) {
                auto &geom = std::get<CurveGeometry>(geometry);
                if (geomType == GeometryType::FlatQuadraticBSplines) {
                    delete[] geom.normalBuffers;
                    delete[] geom.normalBufferArray;
                }
                delete[] geom.widthBuffers;
                delete[] geom.widthBufferArray;
                delete[] geom.vertexBuffers;
                delete[] geom.vertexBufferArray;
            }
            else if (std::holds_alternative<SphereGeometry>(geometry)) {
                auto &geom = std::get<SphereGeometry>(geometry);
                delete[] geom.radiusBuffers;
                delete[] geom.radiusBufferArray;
                delete[] geom.centerBuffers;
                delete[] geom.centerBufferArray;
            }
            else if (std::holds_alternative<CustomPrimitiveGeometry>(geometry)) {
                auto &geom = std::get<CustomPrimitiveGeometry>(geometry);
                delete[] geom.primitiveAabbBuffers;
                delete[] geom.primitiveAabbBufferArray;
            }
            else {
                optixuAssert_ShouldNotBeCalled();
            }
            getContext()->unregisterName(this);
        }

        const _Scene* getScene() const {
            return scene;
        }
        _Context* getContext() const override {
            return scene->getContext();
        }
        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("GeomInst");



        GeometryType getGeometryType() const {
            return geomType;
        }
        uint32_t getNumMotionSteps() const {
            return numMotionSteps;
        }
        void fillBuildInput(OptixBuildInput* input, CUdeviceptr preTransform) const;
        void updateBuildInput(OptixBuildInput* input, CUdeviceptr preTransform) const;

        void calcSBTRequirements(
            uint32_t gasMatSetIdx,
            const SizeAlign &gasUserDataSizeAlign,
            const SizeAlign &gasChildUserDataSizeAlign,
            SizeAlign* maxRecordSizeAlign, uint32_t* numSBTRecords) const;
        uint32_t fillSBTRecords(
            const _Pipeline* pipeline, uint32_t gasMatSetIdx,
            const void* gasUserData, const SizeAlign &gasUserDataSizeAlign,
            const void* gasChildUserData, const SizeAlign &gasChildUserDataSizeAlign,
            uint32_t numRayTypes, uint8_t* records) const;
    };



    template <>
    class Object<GeometryAccelerationStructure>::Priv : public PrivateObject {
        struct Child {
            _GeometryInstance* geomInst;
            CUdeviceptr preTransform;
            SizeAlign userDataSizeAlign;
            std::vector<uint8_t> userData;

            bool operator==(const Child &rChild) const {
                return geomInst == rChild.geomInst && preTransform == rChild.preTransform;
            }
        };

        _Scene* scene;
        uint32_t serialID;
        GeometryType geomType;
        SizeAlign userDataSizeAlign;
        std::vector<uint8_t> userData;

        std::vector<uint32_t> numRayTypesPerMaterialSet;

        std::vector<Child> children;
        std::vector<OptixBuildInput> buildInputs;

        OptixAccelBuildOptions buildOptions;
        OptixAccelBufferSizes memoryRequirement;

        CUevent finishEvent;
        CUdeviceptr compactedSizeOnDevice;
        size_t compactedSize;
        OptixAccelEmitDesc propertyCompactedSize;

        OptixTraversableHandle handle;
        OptixTraversableHandle compactedHandle;
        BufferView accelBuffer;
        BufferView compactedAccelBuffer;
        ASTradeoff tradeoff;

        uint32_t allowUpdate : 1;
        uint32_t allowCompaction : 1;
        uint32_t allowRandomVertexAccess : 1;
        uint32_t allowOpacityMicroMapUpdate : 1;
        uint32_t allowDisableOpacityMicroMaps : 1;
        uint32_t readyToBuild : 1;
        uint32_t available : 1;
        uint32_t readyToCompact : 1;
        uint32_t compactedAvailable : 1;

    public:
        OPTIXU_OPAQUE_BRIDGE(GeometryAccelerationStructure);

        Priv(_Scene* _scene, uint32_t _serialID, GeometryType _geomType) :
            scene(_scene),
            serialID(_serialID),
            geomType(_geomType),
            userData(sizeof(uint32_t)),
            handle(0), compactedHandle(0),
            tradeoff(ASTradeoff::Default),
            allowUpdate(false), allowCompaction(false), allowRandomVertexAccess(false),
            allowOpacityMicroMapUpdate(false), allowDisableOpacityMicroMaps(false),
            readyToBuild(false), available(false),
            readyToCompact(false), compactedAvailable(false)
        {
            scene->addGAS(this);

            numRayTypesPerMaterialSet.resize(1, 0);

            buildOptions = {};

            CUDADRV_CHECK(cuEventCreate(
                &finishEvent, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
            CUDADRV_CHECK(cuMemAlloc(&compactedSizeOnDevice, sizeof(size_t)));

            propertyCompactedSize = OptixAccelEmitDesc{};
            propertyCompactedSize.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            propertyCompactedSize.result = compactedSizeOnDevice;
        }
        ~Priv() {
            cuMemFree(compactedSizeOnDevice);
            cuEventDestroy(finishEvent);

            scene->removeGAS(this);
            getContext()->unregisterName(this);
        }

        const _Scene* getScene() const {
            return scene;
        }
        _Context* getContext() const override {
            return scene->getContext();
        }
        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("GAS");



        uint32_t getSerialID() const {
            return serialID;
        }

        uint32_t getNumMaterialSets() const {
            return static_cast<uint32_t>(numRayTypesPerMaterialSet.size());
        }
        uint32_t getNumRayTypes(uint32_t matSetIdx) const {
            return numRayTypesPerMaterialSet[matSetIdx];
        }

        void calcSBTRequirements(
            uint32_t matSetIdx, SizeAlign* maxRecordSizeAlign, uint32_t* numSBTRecords) const;
        uint32_t fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, uint8_t* records) const;
        bool hasMotion() const {
            return buildOptions.motionOptions.numKeys >= 2;
        }

        void markDirty();
        bool isReady() const {
            return available || compactedAvailable;
        }

        OptixTraversableHandle getHandle() const {
            throwRuntimeError(isReady(), "Traversable handle is not ready.");
            if (compactedAvailable)
                return compactedHandle;
            if (available)
                return handle;
            optixuAssert_ShouldNotBeCalled();
            return 0;
        }
    };



    template <>
    class Object<Transform>::Priv : public PrivateObject {
        _Scene* scene;
        std::variant<
            void*,
            _GeometryAccelerationStructure*,
            _InstanceAccelerationStructure*,
            _Transform*
        > child;
        uint8_t* data;
        size_t dataSize;
        TransformType type;
        OptixMotionOptions options;

        OptixTraversableHandle handle;
        uint32_t available : 1;

    public:
        OPTIXU_OPAQUE_BRIDGE(Transform);

        Priv(_Scene* _scene) :
            scene(_scene),
            data(nullptr), dataSize(0),
            handle(0),
            available(false)
        {
            scene->addTransform(this);

            options.numKeys = 2;
            options.timeBegin = 0.0f;
            options.timeEnd = 0.0f;
            options.flags = OPTIX_MOTION_FLAG_NONE;
        }
        ~Priv() {
            if (data)
                delete data;
            data = nullptr;
            scene->removeTransform(this);
            getContext()->unregisterName(this);
        }

        const _Scene* getScene() const {
            return scene;
        }
        _Context* getContext() const override {
            return scene->getContext();
        }
        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("Xfm");



        _GeometryAccelerationStructure* getDescendantGAS() const;

        void markDirty();
        bool isReady() const {
            return available;
        }

        OptixTraversableHandle getHandle() const {
            throwRuntimeError(isReady(), "IAS %s: Traversable handle is not ready.", getName().c_str());
            return handle;
        }
    };



    template <>
    class Object<Instance>::Priv : public PrivateObject {
        _Scene* scene;
        std::variant<
            void*,
            _GeometryAccelerationStructure*,
            _InstanceAccelerationStructure*,
            _Transform*
        > child;
        uint32_t matSetIndex;
        uint32_t id;
        uint32_t visibilityMask;
        OptixInstanceFlags flags;
        float instTransform[12];

    public:
        OPTIXU_OPAQUE_BRIDGE(Instance);

        Priv(_Scene* _scene) :
            scene(_scene)
        {
            matSetIndex = 0xFFFFFFFF;
            id = 0;
            visibilityMask = 0xFF;
            flags = OPTIX_INSTANCE_FLAG_NONE;
            const float identity[] = {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
            };
            std::copy_n(identity, 12, instTransform);
        }
        ~Priv() {
            getContext()->unregisterName(this);
        }

        const _Scene* getScene() const {
            return scene;
        }
        _Context* getContext() const override {
            return scene->getContext();
        }
        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("Inst");



        void fillInstance(OptixInstance* instance) const;
        void updateInstance(OptixInstance* instance) const;
        bool isMotionAS() const;
        bool isTransform() const;
    };



    template <>
    class Object<InstanceAccelerationStructure>::Priv : public PrivateObject {
        _Scene* scene;

        std::vector<_Instance*> children;
        OptixBuildInput buildInput;
        std::vector<OptixInstance> instances;

        OptixAccelBuildOptions buildOptions;
        OptixAccelBufferSizes memoryRequirement;

        CUevent finishEvent;
        CUdeviceptr compactedSizeOnDevice;
        size_t compactedSize;
        OptixAccelEmitDesc propertyCompactedSize;

        OptixTraversableHandle handle;
        OptixTraversableHandle compactedHandle;
        BufferView instanceBuffer;
        BufferView accelBuffer;
        BufferView compactedAccelBuffer;
        ASTradeoff tradeoff;
        uint32_t allowUpdate : 1;
        uint32_t allowCompaction : 1;
        uint32_t allowRandomInstanceAccess : 1;
        uint32_t readyToBuild : 1;
        uint32_t available : 1;
        uint32_t readyToCompact : 1;
        uint32_t compactedAvailable : 1;

    public:
        OPTIXU_OPAQUE_BRIDGE(InstanceAccelerationStructure);

        Priv(_Scene* _scene) :
            scene(_scene),
            handle(0), compactedHandle(0),
            tradeoff(ASTradeoff::Default),
            allowUpdate(false), allowCompaction(false), allowRandomInstanceAccess(false),
            readyToBuild(false), available(false),
            readyToCompact(false), compactedAvailable(false)
        {
            scene->addIAS(this);

            buildOptions = {};

            CUDADRV_CHECK(cuEventCreate(
                &finishEvent, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
            CUDADRV_CHECK(cuMemAlloc(&compactedSizeOnDevice, sizeof(size_t)));

            std::memset(&propertyCompactedSize, 0, sizeof(propertyCompactedSize));
            propertyCompactedSize.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            propertyCompactedSize.result = compactedSizeOnDevice;
        }
        ~Priv() {
            cuMemFree(compactedSizeOnDevice);
            cuEventDestroy(finishEvent);

            scene->removeIAS(this);
            getContext()->unregisterName(this);
        }

        const _Scene* getScene() const {
            return scene;
        }
        _Context* getContext() const override {
            return scene->getContext();
        }
        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("IAS");



        bool hasMotion() const {
            return buildOptions.motionOptions.numKeys >= 2;
        }



        void markDirty(bool readyToBuild);
        bool isReady() const {
            return available || compactedAvailable;
        }

        OptixTraversableHandle getHandle() const {
            throwRuntimeError(isReady(), "Traversable handle is not ready.");
            if (compactedAvailable)
                return compactedHandle;
            if (available)
                return handle;
            optixuAssert_ShouldNotBeCalled();
            return 0;
        }
    };



    template <>
    class Object<Pipeline>::Priv : public PrivateObject {
        struct KeyForBuiltinISModule {
            OptixPrimitiveType curveType;
            OptixCurveEndcapFlags endcapFlags;
            OptixBuildFlags buildFlags;

            bool operator<(const KeyForBuiltinISModule &rKey) const {
                if (curveType < rKey.curveType)
                    return true;
                else if (curveType > rKey.curveType)
                    return false;
                if (endcapFlags < rKey.endcapFlags)
                    return true;
                else if (endcapFlags > rKey.endcapFlags)
                    return false;
                if (buildFlags < rKey.buildFlags)
                    return true;
                else if (buildFlags > rKey.buildFlags)
                    return false;
                return false;
            }

            struct Hash {
                typedef std::size_t result_type;

                std::size_t operator()(const KeyForBuiltinISModule &key) const {
                    size_t seed = 0;
                    const auto hash0 = std::hash<OptixPrimitiveType>()(key.curveType);
                    const auto hash1 = std::hash<OptixCurveEndcapFlags>()(key.endcapFlags);
                    const auto hash2 = std::hash<OptixBuildFlags>()(key.buildFlags);
                    seed ^= hash0 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    seed ^= hash1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    seed ^= hash2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    return seed;
                }
            };
            bool operator==(const KeyForBuiltinISModule &rKey) const {
                return
                    curveType == rKey.curveType &&
                    endcapFlags == rKey.endcapFlags &&
                    buildFlags == rKey.buildFlags;
            }
        };

        _Context* context;
        OptixPipeline rawPipeline;

        OptixPipelineCompileOptions pipelineCompileOptions;
        size_t sizeOfPipelineLaunchParams;
        std::unordered_set<_Program*> programs;
        std::unordered_set<_HitProgramGroup*> hitProgramGroups;
        std::unordered_set<_CallableProgramGroup*> callableProgramGroups;

        _Scene* scene;
        uint32_t numMissRayTypes;
        uint32_t numCallablePrograms;
        size_t sbtSize;

        std::unordered_map<KeyForBuiltinISModule, _Module*, KeyForBuiltinISModule::Hash> modulesForBuiltinIS;
        _Program* currentRayGenProgram;
        _Program* currentExceptionProgram;
        std::vector<_Program*> currentMissPrograms;
        std::vector<_CallableProgramGroup*> currentCallablePrograms;
        BufferView sbt;
        void* sbtHostMem;
        BufferView hitGroupSbt;
        void* hitGroupSbtHostMem;
        OptixShaderBindingTable sbtParams;

        uint32_t pipelineIsLinked : 1;
        uint32_t sbtLayoutIsUpToDate : 1;
        uint32_t sbtIsUpToDate : 1;
        uint32_t hitGroupSbtIsUpToDate : 1;
        uint32_t hasActiveDirectCallable : 1;
        uint32_t hasActiveContinuationCallable : 1;
        uint32_t stackSizeHasBeenSet : 1;

        Module createModule(
            const char* data, size_t size,
            int32_t maxRegisterCount,
            OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel,
            OptixModuleCompileBoundValueEntry* boundValues, uint32_t numBoundValues,
            const PayloadType* payloadTypes, uint32_t numPayloadTypes);
        OptixModule getModuleForBuiltin(
            OptixPrimitiveType primType, OptixCurveEndcapFlags endcapFlags,
            ASTradeoff tradeoff, bool allowUpdate, bool allowCompaction, bool allowRandomVertexAccess);

        void createProgramGroup(
            const OptixProgramGroupDesc &desc, const OptixProgramGroupOptions &options,
            OptixProgramGroup* group);

        void setupShaderBindingTable(CUstream stream);

    public:
        OPTIXU_OPAQUE_BRIDGE(Pipeline);

        Priv(_Context* ctxt) :
            context(ctxt), rawPipeline(nullptr),
            sizeOfPipelineLaunchParams(0),
            scene(nullptr), numMissRayTypes(0), numCallablePrograms(0),
            currentRayGenProgram(nullptr), currentExceptionProgram(nullptr),
            pipelineIsLinked(false), sbtLayoutIsUpToDate(false),
            sbtIsUpToDate(false), hitGroupSbtIsUpToDate(false),
            hasActiveDirectCallable(false), hasActiveContinuationCallable(false),
            stackSizeHasBeenSet(false)
        {
            sbtParams = {};
        }
        ~Priv();

        _Context* getContext() const override {
            return context;
        }
        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("Pipeline");

        OptixPipeline getRawPipeline() const {
            return rawPipeline;
        }



        bool isLinked() const { return pipelineIsLinked; }
        void markDirty();

        void destroyProgram(_Program* program);
        void destroyHitProgramGroup(_HitProgramGroup* hitGroup);
        void destroyCallableProgramGroup(_CallableProgramGroup* callableGroup);
    };



    template <>
    class Object<Module>::Priv : public PrivateObject {
        const _Pipeline* pipeline;
        OptixModule rawModule;

    public:
        OPTIXU_OPAQUE_BRIDGE(Module);

        Priv(const _Pipeline* pl, OptixModule _rawModule) :
            pipeline(pl), rawModule(_rawModule) {}
        ~Priv() {
            getContext()->unregisterName(this);
        }

        _Context* getContext() const override {
            return pipeline->getContext();
        }
        const _Pipeline* getPipeline() const {
            return pipeline;
        }

        OptixModule getRawModule() const {
            return rawModule;
        }
    };



    template <>
    class Object<Program>::Priv : public PrivateObject {
        _Pipeline* pipeline;
        OptixProgramGroup rawGroup;
        OptixProgramGroupKind kind;
        uint32_t stackSize;
        bool isActiveInPipeline;

    public:
        OPTIXU_OPAQUE_BRIDGE(Program);

        Priv(_Pipeline* pl, OptixProgramGroup _rawGroup, OptixProgramGroupKind _kind) :
            pipeline(pl), rawGroup(_rawGroup), kind(_kind), isActiveInPipeline(true) {}
        ~Priv() {
            getContext()->unregisterName(this);
        }

        _Context* getContext() const override {
            return pipeline->getContext();
        }
        const _Pipeline* getPipeline() const {
            return pipeline;
        }
        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("Program");

        OptixProgramGroup getRawProgramGroup() const {
            return rawGroup;
        }

        bool isActive() const {
            return isActiveInPipeline;
        }
        void setStackSize() {
            OptixStackSizes stackSizes;
            OPTIX_CHECK(optixProgramGroupGetStackSize(rawGroup, &stackSizes, pipeline->getRawPipeline()));
            if (kind == OPTIX_PROGRAM_GROUP_KIND_RAYGEN)
                stackSize = stackSizes.cssRG;
            else if (kind == OPTIX_PROGRAM_GROUP_KIND_MISS)
                stackSize = stackSizes.cssMS;
            else if (kind == OPTIX_PROGRAM_GROUP_KIND_EXCEPTION)
                stackSize = 0;
            else
                optixuAssert_ShouldNotBeCalled();
        }

        void packHeader(uint8_t* record) const {
            OPTIX_CHECK(optixSbtRecordPackHeader(rawGroup, record));
        }
    };



    template <>
    class Object<HitProgramGroup>::Priv : public PrivateObject {
        _Pipeline* pipeline;
        OptixProgramGroup rawGroup;
        uint32_t stackSizeCH;
        uint32_t stackSizeAH;
        uint32_t stackSizeIS;
        bool isActiveInPipeline;

    public:
        OPTIXU_OPAQUE_BRIDGE(HitProgramGroup);

        Priv(_Pipeline* pl, OptixProgramGroup _rawGroup) :
            pipeline(pl), rawGroup(_rawGroup), isActiveInPipeline(true) {}
        ~Priv() {
            getContext()->unregisterName(this);
        }

        _Context* getContext() const override {
            return pipeline->getContext();
        }
        const _Pipeline* getPipeline() const {
            return pipeline;
        }
        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("HitProgramGroup");

        OptixProgramGroup getRawProgramGroup() const {
            return rawGroup;
        }

        bool isActive() const {
            return isActiveInPipeline;
        }
        void setStackSizes() {
            OptixStackSizes stackSizes;
            OPTIX_CHECK(optixProgramGroupGetStackSize(rawGroup, &stackSizes, pipeline->getRawPipeline()));
            stackSizeCH = stackSizes.cssCH;
            stackSizeAH = stackSizes.cssAH;
            stackSizeIS = stackSizes.cssIS;
        }

        void packHeader(uint8_t* record) const {
            OPTIX_CHECK(optixSbtRecordPackHeader(rawGroup, record));
        }
    };



    template <>
    class Object<CallableProgramGroup>::Priv : public PrivateObject {
        _Pipeline* pipeline;
        OptixProgramGroup rawGroup;
        uint32_t stackSizeDC;
        uint32_t stackSizeCC;
        bool isActiveInPipeline;

    public:
        OPTIXU_OPAQUE_BRIDGE(CallableProgramGroup);

        Priv(_Pipeline* pl, OptixProgramGroup _rawGroup) :
            pipeline(pl), rawGroup(_rawGroup), isActiveInPipeline(true) {}
        ~Priv() {
            getContext()->unregisterName(this);
        }

        _Context* getContext() const override {
            return pipeline->getContext();
        }
        const _Pipeline* getPipeline() const {
            return pipeline;
        }
        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("CallableProgramGroup");

        OptixProgramGroup getRawProgramGroup() const {
            return rawGroup;
        }

        bool isActive() const {
            return isActiveInPipeline;
        }
        void setStackSizes() {
            OptixStackSizes stackSizes;
            OPTIX_CHECK(optixProgramGroupGetStackSize(rawGroup, &stackSizes, pipeline->getRawPipeline()));
            stackSizeDC = stackSizes.dssDC;
            stackSizeCC = stackSizes.cssCC;
        }

        void getStackSizes(uint32_t* _stackSizeDC, uint32_t* _stackSizeCC) const {
            *_stackSizeDC = stackSizeDC;
            *_stackSizeCC = stackSizeCC;
        }

        void packHeader(uint8_t* record) const {
            OPTIX_CHECK(optixSbtRecordPackHeader(rawGroup, record));
        }
    };



    static inline uint32_t getPixelSize(OptixPixelFormat format) {
        switch (format) {
        case OPTIX_PIXEL_FORMAT_HALF2:
            return 2 * sizeof(uint16_t);
        case OPTIX_PIXEL_FORMAT_HALF3:
            return 3 * sizeof(uint16_t);
        case OPTIX_PIXEL_FORMAT_HALF4:
            return 4 * sizeof(uint16_t);
        case OPTIX_PIXEL_FORMAT_FLOAT2:
            return 2 * sizeof(float);
        case OPTIX_PIXEL_FORMAT_FLOAT3:
            return 3 * sizeof(float);
        case OPTIX_PIXEL_FORMAT_FLOAT4:
            return 4 * sizeof(float);
        case OPTIX_PIXEL_FORMAT_UCHAR3:
            return 3 * sizeof(uint8_t);
        case OPTIX_PIXEL_FORMAT_UCHAR4:
            return 4 * sizeof(uint8_t);
        default:
            optixuAssert_ShouldNotBeCalled();
            break;
        }
        return 0;
    }

    struct _DenoisingTask {
        int32_t inputOffsetX;
        int32_t inputOffsetY;
        int32_t outputOffsetX;
        int32_t outputOffsetY;
        int32_t outputWidth;
        int32_t outputHeight;

        _DenoisingTask() {}
        _DenoisingTask(const DenoisingTask &v) {
            std::memcpy(this, &v, sizeof(v));
        }
        operator DenoisingTask() const {
            DenoisingTask ret;
            std::memcpy(&ret, this, sizeof(ret));
            return ret;
        }
    };
    static_assert(
        sizeof(DenoisingTask) == sizeof(_DenoisingTask) &&
        alignof(DenoisingTask) == alignof(_DenoisingTask),
        "Size/Alignment mismatch: DenoisingTask vs _DenoisingTask");

    template <>
    class Object<Denoiser>::Priv : public PrivateObject {
        _Context* context;
        OptixDenoiser rawDenoiser;
        OptixDenoiserModelKind modelKind;

        uint32_t imageWidth;
        uint32_t imageHeight;
        uint32_t tileWidth;
        uint32_t tileHeight;
        int32_t overlapWidth;
        uint32_t inputWidth;
        uint32_t inputHeight;
        DenoiserSizes sizes;

        BufferView stateBuffer;
        BufferView scratchBuffer;

        uint32_t guideAlbedo : 1;
        uint32_t guideNormal : 1;

        uint32_t useTiling : 1;
        uint32_t imageSizeSet : 1;
        uint32_t tasksAreReady : 1;
        uint32_t stateIsReady : 1;

    public:
        OPTIXU_OPAQUE_BRIDGE(Denoiser);

        Priv(
            _Context* ctxt,
            OptixDenoiserModelKind _modelKind,
            bool _guideAlbedo,
            bool _guideNormal,
            OptixDenoiserAlphaMode alphaMode) :
            context(ctxt),
            imageWidth(0), imageHeight(0), tileWidth(0), tileHeight(0),
            overlapWidth(0), inputWidth(0), inputHeight(0),
            sizes{ 0, 0, 0, 0, 0 },
            modelKind(_modelKind), guideAlbedo(_guideAlbedo), guideNormal(_guideNormal),
            useTiling(false), imageSizeSet(false), stateIsReady(false)
        {
            OptixDenoiserOptions options = {};
            options.guideAlbedo = _guideAlbedo;
            options.guideNormal = _guideNormal;
            options.denoiseAlpha = alphaMode;
            OPTIX_CHECK(optixDenoiserCreate(context->getRawContext(), modelKind, &options, &rawDenoiser));
        }
        ~Priv() {
            optixDenoiserDestroy(rawDenoiser);
            context->unregisterName(this);
        }

        _Context* getContext() const override {
            return context;
        }
        OPTIXU_DEFINE_THROW_RUNTIME_ERROR("Denoiser");
    };
}
