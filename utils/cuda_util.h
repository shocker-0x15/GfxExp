/*

   Copyright 2023 Shin Watanabe

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

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#   define CUDAUPlatform_Windows
#   if defined(__MINGW32__) // Defined for both 32 bit/64 bit MinGW
#       define CUDAUPlatform_Windows_MinGW
#   elif defined(_MSC_VER)
#       define CUDAUPlatform_Windows_MSVC
#       if defined(__INTELLISENSE__)
#           define CUDAU_CODE_COMPLETION
#       endif
#   endif
#elif defined(__linux__)
#   define CUDAUPlatform_Linux
#elif defined(__APPLE__)
#   define CUDAUPlatform_macOS
#elif defined(__OpenBSD__)
#   define CUDAUPlatform_OpenBSD
#endif



#if __cplusplus <= 199711L
#   if defined(CUDAUPlatform_Windows_MSVC)
#       pragma message("\"/Zc:__cplusplus\" compiler option to enable the updated __cplusplus definition is recommended.")
#   else
#       pragma message("Enabling the updated __cplusplus definition is recommended.")
#   endif
#endif



#if defined(__CUDACC_RTC__)
// Defining things corresponding to cstdint and cfloat is left to the user.
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;
#else
#include <cstdint>
#include <cfloat>
#if defined(CUDAUPlatform_Windows)
#   pragma warning(push)
#   pragma warning(disable:4819)
#endif
#include <cuda.h>
#if defined(CUDAUPlatform_Windows)
#   pragma warning(pop)
#endif
#endif

#if !defined(__CUDA_ARCH__)
#   include <cstdio>
#   include <cstdlib>

#   include <algorithm>
#   include <vector>
#   include <sstream>

// JP: CUDA/OpenGL連携機能が不要な場合はコンパイルオプションとして
//     CUDA_UTIL_DONT_USE_GL_INTEROPの定義を行う。
//     GL/gl3w.hは必要に応じて書き換える。
// EN: Define CUDA_UTIL_DONT_USE_GL_INTEROP as a compile option if CUDA/OpenGL interoperability
//     is not required.
//     Modify GL/gl3w.h as needed.
#   if !defined(CUDA_UTIL_DONT_USE_GL_INTEROP)
#       define CUDA_UTIL_USE_GL_INTEROP
#   endif
#   if defined(CUDA_UTIL_USE_GL_INTEROP)
#       include <GL/gl3w.h>
#       include <cudaGL.h>
#   endif

#   define CUDA_UTIL_TEX_DIM_WORKAROUND 1

#   undef min
#   undef max
#   undef near
#   undef far
#   undef RGB
#endif

#if __cplusplus >= 202002L
#   include <concepts>
#endif



#if defined(__CUDACC__)
#   define CUDA_SHARED_MEM __shared__
#   define CUDA_CONSTANT_MEM __constant__
#   define CUDA_DEVICE_MEM __device__
#   define CUDA_DEVICE_KERNEL extern "C" __global__
#   define CUDA_INLINE __forceinline__
#   define CUDA_DEVICE_FUNCTION __device__
#   define CUDA_COMMON_FUNCTION __host__ __device__
#else
#   define CUDA_SHARED_MEM
#   define CUDA_CONSTANT_MEM
#   define CUDA_DEVICE_MEM
#   define CUDA_DEVICE_KERNEL
#   define CUDA_INLINE inline
#   define CUDA_DEVICE_FUNCTION
#   define CUDA_COMMON_FUNCTION
#endif



#ifdef _DEBUG
#   define CUDAU_ENABLE_ASSERT
#endif

#if defined(CUDAU_ENABLE_ASSERT)
#   if defined(__CUDA_ARCH__)
#       define CUDAUAssert(expr, fmt, ...) \
do { \
    if (!(expr)) { \
        printf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
        printf(fmt"\n", ##__VA_ARGS__); \
    } \
} \
while (0)
#   else
#       define CUDAUAssert(expr, fmt, ...) \
do { \
    if (!(expr)) { \
        cudau::devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
        cudau::devPrintf(fmt"\n", ##__VA_ARGS__); \
        abort(); \
    } \
} \
while (0)
#   endif
#else
#   define CUDAUAssert(expr, fmt, ...)
#endif

#define CUDAUAssert_ShouldNotBeCalled() CUDAUAssert(false, "Should not be called!")
#define CUDAUAssert_NotImplemented() CUDAUAssert(false, "Not implemented yet!")

#define CUDADRV_CHECK(call) \
    do { \
        CUresult error = call; \
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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA call (" << #call << " ) failed with error: '" \
               << cudaGetErrorString(error) \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)



namespace cudau {
#if !defined(__CUDA_ARCH__)
    void devPrintf(const char* fmt, ...);
#endif



#if __cplusplus >= 202002L
#   define CUDAU_INTEGRAL_CONCEPT std::integral
#else
#   define CUDAU_INTEGRAL_CONCEPT typename
#endif

    template <typename T, bool oobCheck>
    class RWBufferTemplate;

    template <typename T, bool oobCheck>
    class ROBufferTemplate {
        friend class RWBufferTemplate<T, oobCheck>;
        const T* m_data;

    public:
        CUDA_COMMON_FUNCTION ROBufferTemplate() : m_data(nullptr) {}
        CUDA_COMMON_FUNCTION ROBufferTemplate(const T* data, uint32_t) :
            m_data(data) {}

        template <CUDAU_INTEGRAL_CONCEPT I>
        CUDA_COMMON_FUNCTION const T &operator[](I idx) const {
            return m_data[idx];
        }

        CUDA_COMMON_FUNCTION operator bool() const {
            return m_data;
        }
    };

    template <typename T>
    class ROBufferTemplate<T, true> {
        friend class RWBufferTemplate<T, true>;
        const T* m_data;
        uint32_t m_numElements;

    public:
        CUDA_COMMON_FUNCTION ROBufferTemplate() : m_data(nullptr), m_numElements(0) {}
        CUDA_COMMON_FUNCTION ROBufferTemplate(const T* data, uint32_t numElements) :
            m_data(data), m_numElements(numElements) {}

        CUDA_COMMON_FUNCTION uint32_t getNumElements() const {
            return m_numElements;
        }

        template <CUDAU_INTEGRAL_CONCEPT I>
        CUDA_COMMON_FUNCTION const T &operator[](I idx) const {
            CUDAUAssert(
                idx < m_numElements, "Buffer 0x%p OOB Access: %u >= %u\n",
                m_data, static_cast<uint32_t>(idx), m_numElements);
            return m_data[idx];
        }

        CUDA_COMMON_FUNCTION operator bool() const {
            return m_data;
        }
    };



    template <typename T, bool oobCheck>
    class RWBufferTemplate {
        T* m_data;

    public:
        CUDA_COMMON_FUNCTION RWBufferTemplate() : m_data(nullptr) {}
        CUDA_COMMON_FUNCTION RWBufferTemplate(T* data, uint32_t) :
            m_data(data) {}
        CUDA_COMMON_FUNCTION RWBufferTemplate(const ROBufferTemplate<T, oobCheck> &buf) :
            m_data(const_cast<T*>(buf.m_data)) {}

        template <CUDAU_INTEGRAL_CONCEPT I>
        CUDA_COMMON_FUNCTION T &operator[](I idx) {
            return m_data[idx];
        }
        template <CUDAU_INTEGRAL_CONCEPT I>
        CUDA_COMMON_FUNCTION const T &operator[](I idx) const {
            return m_data[idx];
        }

        CUDA_COMMON_FUNCTION operator bool() const {
            return m_data;
        }
    };

    template <typename T>
    class RWBufferTemplate<T, true> {
        T* m_data;
        uint32_t m_numElements;

    public:
        CUDA_COMMON_FUNCTION RWBufferTemplate() : m_data(nullptr), m_numElements(0) {}
        CUDA_COMMON_FUNCTION RWBufferTemplate(T* data, uint32_t numElements) :
            m_data(data), m_numElements(numElements) {}
        CUDA_COMMON_FUNCTION RWBufferTemplate(const ROBufferTemplate<T, true> &buf) :
            m_data(const_cast<T*>(buf.m_data)), m_numElements(buf.m_numElements) {}

        CUDA_COMMON_FUNCTION uint32_t getNumElements() const {
            return m_numElements;
        }

        template <CUDAU_INTEGRAL_CONCEPT I>
        CUDA_COMMON_FUNCTION T &operator[](I idx) {
            CUDAUAssert(
                idx < m_numElements, "Buffer 0x%p OOB Access: %u >= %u\n",
                m_data, static_cast<uint32_t>(idx), m_numElements);
            return m_data[idx];
        }
        template <CUDAU_INTEGRAL_CONCEPT I>
        CUDA_COMMON_FUNCTION const T &operator[](I idx) const {
            CUDAUAssert(
                idx < m_numElements, "Buffer 0x%p OOB Access: %u >= %u\n",
                m_data, static_cast<uint32_t>(idx), m_numElements);
            return m_data[idx];
        }

        CUDA_COMMON_FUNCTION operator bool() const {
            return m_data;
        }
    };

#undef CUDAU_INTEGRAL_CONCEPT



#if !defined(__CUDA_ARCH__)

    struct dim3 {
        uint32_t x, y, z;
        dim3(uint32_t xx = 1, uint32_t yy = 1, uint32_t zz = 1) : x(xx), y(yy), z(zz) {}
    };



    using ConstVoidPtr = const void*;

    inline void addArgPointer(ConstVoidPtr* argPointer, CUdeviceptr* pointer) {}

    template <typename HeadType, typename... TailTypes>
    void addArgPointer(ConstVoidPtr* argPointer, CUdeviceptr* pointer, HeadType &&head, TailTypes&&... tails);

    template <typename... ArgTypes>
    void callKernel(
        CUstream stream, CUfunction kernel,
        const dim3 &gridDim, const dim3 &blockDim, uint32_t sharedMemSize,
        ArgTypes&&... args) {
        if constexpr (sizeof...(args) > 0) {
            ConstVoidPtr argPointers[sizeof...(args)];
            CUdeviceptr pointers[sizeof...(args)] = {};
            addArgPointer(argPointers, pointers, std::forward<ArgTypes>(args)...);

            CUDADRV_CHECK(cuLaunchKernel(
                kernel,
                gridDim.x, gridDim.y, gridDim.z,
                blockDim.x, blockDim.y, blockDim.z,
                sharedMemSize, stream,
                const_cast<void**>(argPointers), nullptr));
        }
        else {
            CUDADRV_CHECK(cuLaunchKernel(
                kernel,
                gridDim.x, gridDim.y, gridDim.z,
                blockDim.x, blockDim.y, blockDim.z,
                sharedMemSize, stream,
                nullptr, nullptr));
        }
    }



    class Kernel {
        CUfunction m_kernel;
        dim3 m_blockDim;
        uint32_t m_sharedMemSize;

    public:
        Kernel() : m_kernel(nullptr), m_blockDim(1), m_sharedMemSize(0) {}
        Kernel(CUmodule module, const char* name, const dim3 blockDim, uint32_t sharedMemSize) :
            m_blockDim(blockDim), m_sharedMemSize(sharedMemSize) {
            CUDADRV_CHECK(cuModuleGetFunction(&m_kernel, module, name));
        }

        void set(CUmodule module, const char* name, const dim3 blockDim, uint32_t sharedMemSize) {
            m_blockDim = blockDim;
            m_sharedMemSize = sharedMemSize;
            CUDADRV_CHECK(cuModuleGetFunction(&m_kernel, module, name));
        }

        void setBlockDimensions(const dim3 &blockDim) {
            m_blockDim = blockDim;
        }
        void setSharedMemorySize(uint32_t sharedMemSize) {
            m_sharedMemSize = sharedMemSize;
        }

        uint32_t getBlockDimX() const { return m_blockDim.x; }
        uint32_t getBlockDimY() const { return m_blockDim.y; }
        uint32_t getBlockDimZ() const { return m_blockDim.z; }
        dim3 calcGridDim(uint32_t numItemsX) const {
            return dim3((numItemsX + m_blockDim.x - 1) / m_blockDim.x);
        }
        dim3 calcGridDim(uint32_t numItemsX, uint32_t numItemsY) const {
            return dim3((numItemsX + m_blockDim.x - 1) / m_blockDim.x,
                        (numItemsY + m_blockDim.y - 1) / m_blockDim.y);
        }
        dim3 calcGridDim(uint32_t numItemsX, uint32_t numItemsY, uint32_t numItemsZ) const {
            return dim3((numItemsX + m_blockDim.x - 1) / m_blockDim.x,
                        (numItemsY + m_blockDim.y - 1) / m_blockDim.y,
                        (numItemsZ + m_blockDim.z - 1) / m_blockDim.z);
        }

        template <typename... ArgTypes>
        void operator()(CUstream stream, const dim3 &gridDim, ArgTypes&&... args) const {
            callKernel(
                stream, m_kernel,
                gridDim, m_blockDim, m_sharedMemSize,
                std::forward<ArgTypes>(args)...);
        }

        template <typename... ArgTypes>
        void launchWithThreadDim(CUstream stream, const dim3 &threadDim, ArgTypes&&... args) const {
            dim3 gridDim = calcGridDim(threadDim.x, threadDim.y, threadDim.z);
            callKernel(
                stream, m_kernel,
                gridDim, m_blockDim, m_sharedMemSize,
                std::forward<ArgTypes>(args)...);
        }
    };



    class Timer {
        CUcontext m_context;
        CUevent m_startEvent;
        CUevent m_endEvent;
        bool m_startIsValid;
        bool m_endIsValid;

    public:
        void initialize(CUcontext context) {
            m_context = context;
            CUDADRV_CHECK(cuCtxSetCurrent(m_context));
            CUDADRV_CHECK(cuEventCreate(&m_startEvent, CU_EVENT_BLOCKING_SYNC));
            CUDADRV_CHECK(cuEventCreate(&m_endEvent, CU_EVENT_BLOCKING_SYNC));
            m_startIsValid = false;
            m_endIsValid = false;
        }
        void finalize() {
            m_startIsValid = false;
            m_endIsValid = false;
            CUDADRV_CHECK(cuCtxSetCurrent(m_context));
            CUDADRV_CHECK(cuEventDestroy(m_endEvent));
            CUDADRV_CHECK(cuEventDestroy(m_startEvent));
            m_context = nullptr;
        }

        void start(CUstream stream) {
            CUDADRV_CHECK(cuEventRecord(m_startEvent, stream));
            m_startIsValid = true;
        }
        void stop(CUstream stream) {
            CUDADRV_CHECK(cuEventRecord(m_endEvent, stream));
            m_endIsValid = true;
        }

        float report() {
            float ret = 0.0f;
            if (m_startIsValid && m_endIsValid) {
                CUDADRV_CHECK(cuEventSynchronize(m_endEvent));
                CUDADRV_CHECK(cuEventElapsedTime(&ret, m_startEvent, m_endEvent));
                m_startIsValid = false;
                m_endIsValid = false;
            }
            return ret;
        }
    };



    enum class BufferType {
        Device = 0,
        GL_Interop = 1,
        ZeroCopy = 2, // TODO: test
        Managed = 3, // TODO: test
    };

    //        ReadWrite: Do bidirectional transfers when mapping and unmapping.
    //         ReadOnly: Do not issue a host-to-device transfer when unmapping.
    // WriteOnlyDiscard: Do not issue a device-to-host transfer when mapping and
    //                   the previous contents will be undefined.
    enum class BufferMapFlag {
        Unmapped = 0,
        ReadWrite,
        ReadOnly,
        WriteOnlyDiscard
    };

    class Buffer {
        CUcontext m_cuContext;
        BufferType m_type;

        uint32_t m_numElements;
        uint32_t m_stride;

        void* m_hostPointer;
        CUdeviceptr m_devicePointer;
        void* m_mappedPointer;
        BufferMapFlag m_mapFlag;

        uint32_t m_GLBufferID;
        CUgraphicsResource m_cudaGfxResource;

        struct {
            unsigned int m_persistentMappedMemory : 1;
            unsigned int m_initialized : 1;
        };

        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;

        void initialize(
            CUcontext context, BufferType type,
            uint32_t numElements, uint32_t stride, uint32_t glBufferID);

    public:
        Buffer();
        ~Buffer();

        Buffer(CUcontext context, BufferType type,
               uint32_t numElements, uint32_t stride) : Buffer() {
            initialize(context, type, numElements, stride);
        }
        Buffer(CUcontext context, BufferType type,
               const void* data, uint32_t numElements, uint32_t stride) : Buffer() {
            initialize(context, type, data, numElements, stride);
        }

        Buffer(Buffer &&b);
        Buffer &operator=(Buffer &&b);

        template <typename T>
        inline operator T() const;

        void initialize(
            CUcontext context, BufferType type,
            uint32_t numElements, uint32_t stride) {
            initialize(context, type, numElements, stride, 0);
        }
        void initialize(
            CUcontext context, BufferType type,
            const void* data, uint32_t numElements, uint32_t stride, CUstream stream = 0) {
            initialize(context, type, numElements, stride, 0);
            CUDADRV_CHECK(cuMemcpyHtoDAsync(getCUdeviceptr(), data, numElements * stride, stream));
        }
        void initializeFromGLBuffer(CUcontext context, uint32_t stride, uint32_t glBufferID) {
#if defined(CUDA_UTIL_USE_GL_INTEROP)
            GLint size;
            glGetNamedBufferParameteriv(glBufferID, GL_BUFFER_SIZE, &size);
            if (size % stride != 0)
                throw std::runtime_error("Given buffer's size is not a multiple of the given stride.");
            initialize(context, BufferType::GL_Interop, size / stride, stride, glBufferID);
#else
            (void)context;
            (void)stride;
            (void)glBufferID;
            throw std::runtime_error(
                "Disable \"CUDA_UTIL_DONT_USE_GL_INTEROP\" if you use CUDA/OpenGL interoperability.");
#endif
        }
        void finalize();

        void resize(uint32_t numElements, uint32_t stride, CUstream stream = 0);

        CUcontext getCUcontext() const {
            return m_cuContext;
        }
        BufferType getBufferType() const {
            return m_type;
        }

        CUdeviceptr getCUdeviceptr() const {
            return m_devicePointer;
        }
        CUdeviceptr getCUdeviceptrAt(uint32_t idx) const {
            return m_devicePointer + static_cast<uintptr_t>(m_stride) * idx;
        }
        void* getDevicePointer() const {
            return reinterpret_cast<void*>(getCUdeviceptr());
        }
        void* getDevicePointerAt(uint32_t idx) const {
            return reinterpret_cast<void*>(getCUdeviceptrAt(idx));
        }
        size_t sizeInBytes() const {
            return static_cast<size_t>(m_numElements) * m_stride;
        }
        uint32_t stride() const {
            return m_stride;
        }
        uint32_t numElements() const {
            return m_numElements;
        }
        bool isInitialized() const {
            return m_initialized;
        }

        void beginCUDAAccess(CUstream stream);
        void endCUDAAccess(CUstream stream);

        void setMappedMemoryPersistent(bool b);
        void* map(CUstream stream = 0, BufferMapFlag flag = BufferMapFlag::ReadWrite);
        template <typename T>
        T* map(CUstream stream = 0, BufferMapFlag flag = BufferMapFlag::ReadWrite) {
            return reinterpret_cast<T*>(map(stream, flag));
        }
        void unmap(CUstream stream = 0);
        void* getMappedPointer() const {
            if (m_type == BufferType::ZeroCopy ||
                m_type == BufferType::Managed)
                return m_hostPointer;
            if (m_mappedPointer == nullptr)
                throw std::runtime_error("The buffer is not not mapped.");
            return m_mappedPointer;
        }
        template <typename T>
        T* getMappedPointer() const {
            if (m_type == BufferType::ZeroCopy ||
                m_type == BufferType::Managed)
                return reinterpret_cast<T*>(m_hostPointer);
            if (m_mappedPointer == nullptr)
                throw std::runtime_error("The buffer is not not mapped.");
            return reinterpret_cast<T*>(m_mappedPointer);
        }
        template <typename T>
        void write(const T* srcValues, uint32_t numValues, CUstream stream = 0) const {
            const size_t transferSize = sizeof(T) * numValues;
            const size_t bufferSize = static_cast<size_t>(m_stride) * m_numElements;
            if (transferSize > bufferSize)
                throw std::runtime_error("Too large transfer");
            CUDADRV_CHECK(cuMemcpyHtoDAsync(getCUdeviceptr(), srcValues, transferSize, stream));
        }
        template <typename T>
        void write(const std::vector<T> &values, CUstream stream = 0) const {
            write(values.data(), static_cast<uint32_t>(values.size()), stream);
        }
        template <typename T>
        void read(T* dstValues, uint32_t numValues, CUstream stream = 0) const {
            const size_t transferSize = sizeof(T) * numValues;
            const size_t bufferSize = static_cast<size_t>(m_stride) * m_numElements;
            if (transferSize > bufferSize)
                throw std::runtime_error("Too large transfer");
            CUDADRV_CHECK(cuMemcpyDtoHAsync(dstValues, getCUdeviceptr(), transferSize, stream));
        }
        template <typename T>
        void read(std::vector<T> &values, CUstream stream = 0) const {
            read(values.data(), static_cast<uint32_t>(values.size()), stream);
        }
        template <typename T>
        void fill(const T &value, CUstream stream = 0) const {
            uint32_t numValues = (m_stride * m_numElements) / sizeof(T);
            if (m_persistentMappedMemory) {
                T* values = reinterpret_cast<T*>(m_mappedPointer);
                for (uint32_t i = 0; i < numValues; ++i)
                    values[i] = value;
                write(values, numValues, stream);
            }
            else {
                std::vector<T> values(numValues, value);
                write(values, stream);
            }
        }

        Buffer copy(CUstream stream = 0) const;
    };



    template <typename T>
    class TypedBuffer : public Buffer {
    public:
        TypedBuffer() {}
        TypedBuffer(CUcontext context, BufferType type, uint32_t numElements) : TypedBuffer() {
            Buffer::initialize(context, type, numElements, sizeof(T));
        }
        TypedBuffer(CUcontext context, BufferType type, uint32_t numElements, const T &value) : TypedBuffer() {
            std::vector<T> values(numElements, value);
            Buffer::initialize(context, type, static_cast<uint32_t>(values.size()), sizeof(T));
            CUDADRV_CHECK(cuMemcpyHtoD(Buffer::getCUdeviceptr(), values.data(), values.size() * sizeof(T)));
        }
        TypedBuffer(CUcontext context, BufferType type, const T* v, uint32_t numElements) : TypedBuffer() {
            Buffer::initialize(context, type, numElements, sizeof(T));
            CUDADRV_CHECK(cuMemcpyHtoD(Buffer::getCUdeviceptr(), v, numElements * sizeof(T)));
        }
        TypedBuffer(CUcontext context, BufferType type, const std::vector<T> &v) : TypedBuffer() {
            Buffer::initialize(context, type, static_cast<uint32_t>(v.size()), sizeof(T));
            CUDADRV_CHECK(cuMemcpyHtoD(Buffer::getCUdeviceptr(), v.data(), v.size() * sizeof(T)));
        }

        void initialize(CUcontext context, BufferType type, uint32_t numElements) {
            Buffer::initialize(context, type, numElements, sizeof(T));
        }
        void initialize(
            CUcontext context, BufferType type,
            uint32_t numElements, const T &value,
            CUstream stream = 0) {
            std::vector<T> values(numElements, value);
            initialize(context, type, static_cast<uint32_t>(values.size()));
            CUDADRV_CHECK(cuMemcpyHtoDAsync(Buffer::getCUdeviceptr(), values.data(), values.size() * sizeof(T), stream));
        }
        void initialize(
            CUcontext context, BufferType type,
            const T* v, uint32_t numElements,
            CUstream stream = 0) {
            initialize(context, type, numElements);
            CUDADRV_CHECK(cuMemcpyHtoDAsync(Buffer::getCUdeviceptr(), v, numElements * sizeof(T), stream));
        }
        void initialize(
            CUcontext context, BufferType type,
            const std::vector<T> &v,
            CUstream stream = 0) {
            initialize(context, type, static_cast<uint32_t>(v.size()));
            CUDADRV_CHECK(cuMemcpyHtoDAsync(Buffer::getCUdeviceptr(), v.data(), v.size() * sizeof(T), stream));
        }
        void finalize() {
            Buffer::finalize();
        }

        void resize(int32_t numElements, CUstream stream = 0) {
            Buffer::resize(numElements, sizeof(T), stream);
        }
        void resize(int32_t numElements, const T &value, CUstream stream = 0) {
            std::vector<T> values(numElements, value);
            Buffer::resize(numElements, sizeof(T), stream);
            CUDADRV_CHECK(cuMemcpyHtoDAsync(
                Buffer::getCUdeviceptr(), values.data(), values.size() * sizeof(T), stream));
        }

        T* getDevicePointer() const {
            return reinterpret_cast<T*>(getCUdeviceptr());
        }
        T* getDevicePointerAt(uint32_t idx) const {
            return reinterpret_cast<T*>(getCUdeviceptrAt(idx));
        }
        template <bool oobCheck>
        ROBufferTemplate<T, oobCheck> getROBuffer() const {
            return ROBufferTemplate<T, oobCheck>(getDevicePointer(), numElements());
        }
        template <bool oobCheck>
        RWBufferTemplate<T, oobCheck> getRWBuffer() const {
            return RWBufferTemplate<T, oobCheck>(getDevicePointer(), numElements());
        }

        T* map(CUstream stream = 0, BufferMapFlag flag = BufferMapFlag::ReadWrite) {
            return Buffer::map<T>(stream, flag);
        }
        T* getMappedPointer() const {
            return Buffer::getMappedPointer<T>();
        }
        void write(const T* srcValues, uint32_t numValues, CUstream stream = 0) const {
            Buffer::write<T>(srcValues, numValues, stream);
        }
        void write(const std::vector<T> &values, CUstream stream = 0) const {
            Buffer::write<T>(values, stream);
        }
        void read(T* dstValues, uint32_t numValues, CUstream stream = 0) const {
            Buffer::read<T>(dstValues, numValues, stream);
        }
        void read(std::vector<T> &values, CUstream stream = 0) const {
            Buffer::read<T>(values, stream);
        }
        void fill(const T &value, CUstream stream = 0) const {
            Buffer::fill<T>(value, stream);
        }

        // TODO: ? stream
        T operator[](uint32_t idx) {
            const T* values = map();
            T ret = values[idx];
            unmap();
            return ret;
        }

        TypedBuffer<T> copy(CUstream stream = 0) const {
            TypedBuffer<T> ret;
            // safe ?
            *reinterpret_cast<Buffer*>(&ret) = Buffer::copy(stream);
            return ret;
        }

        operator std::vector<T>() const {
            std::vector<T> ret(numElements());
            read(ret);
            return std::move(ret);
        }
    };



    template <typename>
    static constexpr bool is_TypedBuffer_v = false;

    template <typename T>
    static constexpr bool is_TypedBuffer_v<TypedBuffer<T>> = true;

    template <typename HeadType, typename... TailTypes>
    void addArgPointer(ConstVoidPtr* argPointer, CUdeviceptr* pointer, HeadType &&head, TailTypes&&... tails) {
        using RawHeadType = std::remove_const_t<std::remove_reference_t<HeadType>>;
        if constexpr (is_TypedBuffer_v<RawHeadType> || std::is_same_v<RawHeadType, Buffer>) {
            *pointer = head.getCUdeviceptr();
            *argPointer = pointer;
        }
        else {
            *argPointer = &head;
        }
        addArgPointer(argPointer + 1, pointer + 1, std::forward<TailTypes>(tails)...);
    }



    enum class ArrayElementType {
        UInt8,
        Int8,
        UInt16,
        Int16,
        UInt32,
        Int32,
        Float16,
        Float32,
        BC1_UNorm,
        BC2_UNorm,
        BC3_UNorm,
        BC4_UNorm,
        BC4_SNorm,
        BC5_UNorm,
        BC5_SNorm,
        BC6H_UF16,
        BC6H_SF16,
        BC7_UNorm
    };

    enum class ArraySurface {
        Enable = 0,
        Disable,
    };

    enum class ArrayTextureGather {
        Enable = 0,
        Disable,
    };

#   if defined(CUDA_UTIL_USE_GL_INTEROP)
    void getArrayElementFormat(GLenum internalFormat, ArrayElementType* elemType, uint32_t* numChannels);
#   endif

    inline bool isBCFormat(ArrayElementType elemType) {
        return (elemType == cudau::ArrayElementType::BC1_UNorm ||
                elemType == cudau::ArrayElementType::BC2_UNorm ||
                elemType == cudau::ArrayElementType::BC3_UNorm ||
                elemType == cudau::ArrayElementType::BC4_UNorm ||
                elemType == cudau::ArrayElementType::BC4_SNorm ||
                elemType == cudau::ArrayElementType::BC5_UNorm ||
                elemType == cudau::ArrayElementType::BC5_SNorm ||
                elemType == cudau::ArrayElementType::BC6H_UF16 ||
                elemType == cudau::ArrayElementType::BC6H_SF16 ||
                elemType == cudau::ArrayElementType::BC7_UNorm);
    }

    class Array {
        CUcontext m_cuContext;

        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_depth;
        uint32_t m_numMipmapLevels;
        uint32_t m_stride;
        ArrayElementType m_elemType;
        uint32_t m_numChannels;

        union {
            CUarray m_array;
            CUmipmappedArray m_mipmappedArray;
        };
        void** m_mappedPointers;
        CUarray* m_mipmapArrays;
        BufferMapFlag* m_mapFlags;
        CUsurfObject* m_surfObjs;

        uint32_t m_GLTexID;
        CUgraphicsResource m_cudaGfxResource;

        struct {
            unsigned int m_surfaceLoadStore : 1;
            unsigned int m_useTextureGather : 1;
            unsigned int m_cubemap : 1;
            unsigned int m_layered : 1;
            unsigned int m_initialized : 1;
        };

        Array(const Array &) = delete;
        Array &operator=(const Array &) = delete;

        void initialize(
            CUcontext context, ArrayElementType elemType, uint32_t numChannels,
            uint32_t width, uint32_t height, uint32_t depth, uint32_t numMipmapLevels,
            bool writable, bool useTextureGather, bool cubemap, bool layered, uint32_t glTexID);

        template <bool forDevice>
        void computeDimensionsOfLevel(uint32_t mipmapLevel, uint32_t* width, uint32_t* height) const {
            *width = m_width;
            *height = m_height;

            // JP: CUDAはNon-Power-of-twoテクスチャーを使えるが、MIPMAPに関して計算がおかしい気がする。
            //     本当は常に下の手順でいきたい。
            if constexpr (forDevice) {
                if (isBCFormat(m_elemType)) {
                    *width = (*width + 3) / 4;
                    *height = (*height + 3) / 4;
                }
                *width = std::max<uint32_t>(1, *width >> mipmapLevel);
                *height = std::max<uint32_t>(1, *height >> mipmapLevel);
            }
            else {
                *width = std::max<uint32_t>(1, *width >> mipmapLevel);
                *height = std::max<uint32_t>(1, *height >> mipmapLevel);
                if (isBCFormat(m_elemType)) {
                    *width = (*width + 3) / 4;
                    *height = (*height + 3) / 4;
                }
            }
        }

    public:
        Array();
        ~Array();

        Array(Array &&b);
        Array &operator=(Array &&b);

        void initialize1D(
            CUcontext context, ArrayElementType elemType, uint32_t numChannels,
            ArraySurface surfaceLoadStore,
            uint32_t length, uint32_t numMipmapLevels) {
            initialize(
                context, elemType, numChannels, length, 0, 0, numMipmapLevels,
                surfaceLoadStore == ArraySurface::Enable, false, false, false, 0);
        }
        void initialize2D(
            CUcontext context, ArrayElementType elemType, uint32_t numChannels,
            ArraySurface surfaceLoadStore, ArrayTextureGather useTextureGather,
            uint32_t width, uint32_t height, uint32_t numMipmapLevels) {
            initialize(
                context, elemType, numChannels, width, height, 0, numMipmapLevels,
                surfaceLoadStore == ArraySurface::Enable,
                useTextureGather == ArrayTextureGather::Enable,
                false, false, 0);
        }
        void initialize3D(
            CUcontext context, ArrayElementType elemType, uint32_t numChannels,
            ArraySurface surfaceLoadStore,
            uint32_t width, uint32_t height, uint32_t depth, uint32_t numMipmapLevels) {
            initialize(
                context, elemType, numChannels, width, height, 0, numMipmapLevels,
                surfaceLoadStore == ArraySurface::Enable, false, false, false, 0);
        }
        void initializeFromGLTexture2D(
            CUcontext context, uint32_t glTexID,
            ArraySurface surfaceLoadStore, ArrayTextureGather useTextureGather) {
#if defined(CUDA_UTIL_USE_GL_INTEROP)
            GLint width, height;
            GLint numMipmapLevels;
            GLint format;
            glGetTextureLevelParameteriv(glTexID, 0, GL_TEXTURE_WIDTH, &width);
            glGetTextureLevelParameteriv(glTexID, 0, GL_TEXTURE_HEIGHT, &height);
            glGetTextureLevelParameteriv(glTexID, 0, GL_TEXTURE_INTERNAL_FORMAT, &format);
            glGetTextureParameteriv(glTexID, GL_TEXTURE_VIEW_NUM_LEVELS, &numMipmapLevels);
            numMipmapLevels = std::max(numMipmapLevels, 1);
            ArrayElementType elemType;
            uint32_t numChannels;
            getArrayElementFormat((GLenum)format, &elemType, &numChannels);
            initialize(
                context, elemType, numChannels, width, height, 0, numMipmapLevels,
                surfaceLoadStore == ArraySurface::Enable,
                useTextureGather == ArrayTextureGather::Enable,
                false, false, glTexID);
#else
            (void)context;
            (void)glTexID;
            (void)surfaceLoadStore;
            throw std::runtime_error(
                "Disable \"CUDA_UTIL_DONT_USE_GL_INTEROP\" if you use CUDA/OpenGL interoperability.");
#endif
        }
        void finalize();

        void resize(uint32_t length, CUstream stream = 0);
        void resize(uint32_t width, uint32_t height, CUstream stream = 0);
        void resize(uint32_t width, uint32_t height, uint32_t depth, CUstream stream = 0);

        CUarray getCUarray(uint32_t mipmapLevel) const {
            if (m_GLTexID) {
                if (m_mipmapArrays[mipmapLevel] == nullptr)
                    throw std::runtime_error("This mip level of this interop array is not mapped.");
                return m_mipmapArrays[mipmapLevel];
            }
            else {
                if (m_numMipmapLevels > 1)
                    return m_mipmapArrays[mipmapLevel];
                else
                    return m_array;
            }
        }
        CUmipmappedArray getCUmipmappedArray() const {
            return m_mipmappedArray;
        }

        uint32_t getWidth() const {
            return m_width;
        }
        uint32_t getHeight() const {
            return m_height;
        }
        uint32_t getDepth() const {
            return m_depth;
        }
        uint32_t getNumMipmapLevels() const {
            return m_numMipmapLevels;
        }
        bool isBCTexture() const {
            return isBCFormat(m_elemType);
        }
        bool isInitialized() const {
            return m_initialized;
        }

        void beginCUDAAccess(CUstream stream, uint32_t mipmapLevel);
        void endCUDAAccess(CUstream stream, uint32_t mipmapLevel);

        void* map(
            uint32_t mipmapLevel = 0,
            CUstream stream = 0,
            BufferMapFlag flag = BufferMapFlag::ReadWrite);
        template <typename T>
        T* map(
            uint32_t mipmapLevel = 0,
            CUstream stream = 0,
            BufferMapFlag flag = BufferMapFlag::ReadWrite) {
            return reinterpret_cast<T*>(map(mipmapLevel, stream, flag));
        }
        void unmap(uint32_t mipmapLevel = 0, CUstream stream = 0);
        template <typename T>
        void write(
            const T* srcValues, uint32_t numValues, uint32_t mipmapLevel = 0, CUstream stream = 0) const {
            uint32_t depth = std::max<uint32_t>(1, m_depth);

            uint32_t hostBw;
            uint32_t hostBh;
            computeDimensionsOfLevel<false>(mipmapLevel, &hostBw, &hostBh);
            size_t hostSizePerRow = hostBw * m_stride;
            size_t hostSize = depth * hostBh * hostSizePerRow;
            if (sizeof(T) * numValues > hostSize)
                throw std::runtime_error("Too large transfer.");
            uint32_t writeHeight = (sizeof(T) * numValues) / hostSizePerRow;

            uint32_t deviceBw;
            uint32_t deviceBh;
            computeDimensionsOfLevel<CUDA_UTIL_TEX_DIM_WORKAROUND>(mipmapLevel, &deviceBw, &deviceBh);
            size_t deviceSizePerRow = deviceBw * static_cast<size_t>(m_stride);

            CUDA_MEMCPY3D params = {};
            params.WidthInBytes = deviceSizePerRow;
            params.Height = deviceBh;
            params.Depth = depth;

            params.srcMemoryType = CU_MEMORYTYPE_HOST;
            params.srcHost = srcValues;
            params.srcPitch = hostSizePerRow;
            params.srcHeight = writeHeight;
            params.srcXInBytes = 0;
            params.srcY = 0;
            params.srcZ = 0;
            // srcArray, srcDevice, srcLOD are not used in this case.

            params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            params.dstArray = (m_numMipmapLevels > 1 || m_GLTexID != 0) ? m_mipmapArrays[mipmapLevel] : m_array;
            params.dstXInBytes = 0;
            params.dstY = 0;
            params.dstZ = 0;
            // dstDevice, dstHeight, dstHost, dstLOD, dstPitch are not used in this case.

            CUDADRV_CHECK(cuMemcpy3DAsync(&params, stream));
        }
        template <typename T>
        void write(const std::vector<T> &values, uint32_t mipmapLevel = 0, CUstream stream = 0) const {
            write(values.data(), static_cast<uint32_t>(values.size()), mipmapLevel, stream);
        }
        template <typename T>
        void read(T* dstValues, uint32_t numValues, uint32_t mipmapLevel = 0, CUstream stream = 0) const {
            uint32_t depth = std::max<uint32_t>(1, m_depth);

            uint32_t hostBw;
            uint32_t hostBh;
            computeDimensionsOfLevel<false>(mipmapLevel, &hostBw, &hostBh);
            size_t hostSizePerRow = hostBw * m_stride;
            size_t hostSize = depth * hostBh * hostSizePerRow;
            if (sizeof(T) * numValues > hostSize)
                throw std::runtime_error("Too large transfer.");
            uint32_t readHeight = (sizeof(T) * numValues) / hostSizePerRow;

            uint32_t deviceBw;
            uint32_t deviceBh;
            computeDimensionsOfLevel<CUDA_UTIL_TEX_DIM_WORKAROUND>(mipmapLevel, &deviceBw, &deviceBh);
            size_t deviceSizePerRow = deviceBw * static_cast<size_t>(m_stride);

            CUDA_MEMCPY3D params = {};
            params.WidthInBytes = deviceSizePerRow;
            params.Height = deviceBh;
            params.Depth = depth;

            params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
            params.srcArray = (m_numMipmapLevels > 1 || m_GLTexID != 0) ? m_mipmapArrays[mipmapLevel] : m_array;
            params.srcXInBytes = 0;
            params.srcY = 0;
            params.srcZ = 0;
            // srcDevice, srcHeight, srcHost, srcLOD, srcPitch are not used in this case.

            params.dstMemoryType = CU_MEMORYTYPE_HOST;
            params.dstHost = dstValues;
            params.dstPitch = hostSizePerRow;
            params.dstHeight = readHeight;
            params.dstXInBytes = 0;
            params.dstY = 0;
            params.dstZ = 0;
            // dstArray, dstDevice, dstLOD are not used in this case.

            CUDADRV_CHECK(cuMemcpy3DAsync(&params, stream));
        }
        template <typename T>
        void read(std::vector<T> &values, uint32_t mipmapLevel = 0, CUstream stream = 0) const {
            read(values.data(), static_cast<uint32_t>(values.size()), mipmapLevel, stream);
        }
        template <typename T>
        void fill(const T &value, uint32_t mipmapLevel = 0, CUstream stream = 0) const {
            uint32_t hostBw;
            uint32_t hostBh;
            computeDimensionsOfLevel<false>(mipmapLevel, &hostBw, &hostBh);
            uint32_t depth = std::max<uint32_t>(1, m_depth);
            size_t hostSizePerRow = hostBw * m_stride;
            size_t hostSize = depth * hostBh * hostSizePerRow;
            size_t numValues = hostSize / sizeof(T);
            std::vector<T> values(value, numValues);
            write(values, mipmapLevel, stream);
        }

        CUDA_RESOURCE_VIEW_DESC getResourceViewDesc() const;

        CUsurfObject getSurfaceObject(uint32_t mipmapLevel) const {
            return m_surfObjs[mipmapLevel];
        }
        [[nodiscard]]
        CUsurfObject createGLSurfaceObject(uint32_t mipmapLevel) const {
#if defined(CUDA_UTIL_USE_GL_INTEROP)
            if (m_GLTexID == 0)
                throw std::runtime_error("This is not an array created from OpenGL object.");
            if (m_mipmapArrays[mipmapLevel] == nullptr)
                throw std::runtime_error("Use beginCUDAAccess()/endCUDAAccess().");

            CUsurfObject ret;
            CUDA_RESOURCE_DESC resDesc = {};
            resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
            resDesc.res.array.hArray = m_mipmapArrays[mipmapLevel];
            CUDADRV_CHECK(cuSurfObjectCreate(&ret, &resDesc));
            return ret;
#else
            (void)mipmapLevel;
            throw std::runtime_error(
                "Disable \"CUDA_UTIL_DONT_USE_GL_INTEROP\" if you use CUDA/OpenGL interoperability.");
#endif
        }
    };

    // MIP-level 0 only
    template <uint32_t NumBuffers>
    class InteropSurfaceObjectHolder {
        Array* m_arrays[NumBuffers];
        CUsurfObject m_surfObjs[NumBuffers];
        uint32_t m_numArrays;
        uint32_t m_arrayIndex;
        uint32_t m_bufferIndex;

    public:
        template <uint32_t numArrays>
        void initialize(Array* const (&arrays)[numArrays]) {
            for (uint32_t i = 0; i < NumBuffers; ++i)
                m_arrays[i] = arrays[i % numArrays];
            m_numArrays = numArrays;
            m_arrayIndex = 0;
            m_bufferIndex = 0;
            for (uint32_t i = 0; i < NumBuffers; ++i)
                m_surfObjs[i] = 0;
        }
        void finalize() {
            for (uint32_t i = 0; i < NumBuffers; ++i) {
                CUDADRV_CHECK(cuSurfObjectDestroy(m_surfObjs[i]));
                m_surfObjs[i] = 0;
            }
            m_bufferIndex = 0;
            m_arrayIndex = 0;
        }

        void beginCUDAAccess(CUstream stream) {
            m_arrays[m_arrayIndex]->beginCUDAAccess(stream, 0);
        }
        void endCUDAAccess(CUstream stream, bool endFrame) {
            m_arrays[m_arrayIndex]->endCUDAAccess(stream, 0);
            if (endFrame) {
                m_arrayIndex = (m_arrayIndex + 1) % m_numArrays;
                m_bufferIndex = (m_bufferIndex + 1) % NumBuffers;
            }
        }
        CUsurfObject getNext() {
            CUsurfObject &curSurfObj = m_surfObjs[m_bufferIndex];
            if (curSurfObj)
                CUDADRV_CHECK(cuSurfObjectDestroy(curSurfObj));
            curSurfObj = m_arrays[m_arrayIndex]->createGLSurfaceObject(0);
            return curSurfObj;
        }
    };



    enum class TextureWrapMode {
        Repeat = CU_TR_ADDRESS_MODE_WRAP,
        Clamp = CU_TR_ADDRESS_MODE_CLAMP,
        Mirror = CU_TR_ADDRESS_MODE_MIRROR,
        Border = CU_TR_ADDRESS_MODE_BORDER,
    };

    enum class TextureFilterMode {
        Point = CU_TR_FILTER_MODE_POINT,
        Linear = CU_TR_FILTER_MODE_LINEAR,
    };

    enum class TextureIndexingMode {
        NormalizedCoordinates = 0,
        ArrayIndex,
    };

    enum class TextureReadMode {
        ElementType = 0,
        NormalizedFloat,
        NormalizedFloat_sRGB
    };

    class TextureSampler {
        CUDA_TEXTURE_DESC m_texDesc;

    public:
        TextureSampler() {
            m_texDesc = {};
            m_texDesc.flags |= CU_TRSF_NORMALIZED_COORDINATES;
        }
        ~TextureSampler() {
        }

        void setXyFilterMode(TextureFilterMode xy) {
            m_texDesc.filterMode = static_cast<CUfilter_mode>(xy);
        }
        void setMipMapFilterMode(TextureFilterMode mipmap) {
            m_texDesc.mipmapFilterMode = static_cast<CUfilter_mode>(mipmap);
        }
        void setWrapMode(uint32_t dim, TextureWrapMode mode) {
            if (dim >= 3)
                return;
            m_texDesc.addressMode[dim] = static_cast<CUaddress_mode>(mode);
        }
        void setBorderColor(float r, float g, float b, float a) {
            m_texDesc.borderColor[0] = r;
            m_texDesc.borderColor[1] = g;
            m_texDesc.borderColor[2] = b;
            m_texDesc.borderColor[3] = a;
        }
        void setIndexingMode(TextureIndexingMode mode) {
            if (mode == TextureIndexingMode::ArrayIndex)
                m_texDesc.flags &= ~CU_TRSF_NORMALIZED_COORDINATES;
            else
                m_texDesc.flags |= CU_TRSF_NORMALIZED_COORDINATES;
        }
        void setReadMode(TextureReadMode mode) {
            if (mode == TextureReadMode::ElementType)
                m_texDesc.flags |= CU_TRSF_READ_AS_INTEGER;
            else
                m_texDesc.flags &= ~CU_TRSF_READ_AS_INTEGER;

            if (mode == TextureReadMode::NormalizedFloat_sRGB)
                m_texDesc.flags |= CU_TRSF_SRGB;
            else
                m_texDesc.flags &= ~CU_TRSF_SRGB;
        }

        TextureFilterMode getXyFilterMode() const {
            return static_cast<TextureFilterMode>(m_texDesc.filterMode);
        }
        TextureFilterMode getMipMapFilterMode() const {
            return static_cast<TextureFilterMode>(m_texDesc.mipmapFilterMode);
        }
        TextureWrapMode getWrapMode(uint32_t dim) {
            if (dim >= 3)
                return TextureWrapMode::Repeat;
            return static_cast<TextureWrapMode>(m_texDesc.addressMode[dim]);
        }
        void getBorderColor(float rgba[4]) const {
            rgba[0] = m_texDesc.borderColor[0];
            rgba[1] = m_texDesc.borderColor[1];
            rgba[2] = m_texDesc.borderColor[2];
            rgba[3] = m_texDesc.borderColor[3];
        }
        TextureIndexingMode getIndexingMode() const {
            if (m_texDesc.flags & CU_TRSF_NORMALIZED_COORDINATES)
                return TextureIndexingMode::NormalizedCoordinates;
            else
                return TextureIndexingMode::ArrayIndex;
        }
        TextureReadMode getReadMode() const {
            if (m_texDesc.flags & CU_TRSF_READ_AS_INTEGER)
                return TextureReadMode::ElementType;
            if (m_texDesc.flags & CU_TRSF_SRGB)
                return TextureReadMode::NormalizedFloat_sRGB;
            else
                return TextureReadMode::NormalizedFloat;
        }

        [[nodiscard]]
        CUtexObject createTextureObject(const Array &array) {
            CUDA_RESOURCE_DESC resDesc = {};
            CUDA_RESOURCE_VIEW_DESC resViewDesc = {};
            if (array.getNumMipmapLevels() > 1) {
                resDesc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
                resDesc.res.mipmap.hMipmappedArray = array.getCUmipmappedArray();
                m_texDesc.maxMipmapLevelClamp = static_cast<float>(array.getNumMipmapLevels() - 1);
            }
            else {
                resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
                resDesc.res.array.hArray = array.getCUarray(0);
            }
            resViewDesc = array.getResourceViewDesc();

            CUtexObject texObj;
            CUDADRV_CHECK(cuTexObjectCreate(&texObj, &resDesc, &m_texDesc, &resViewDesc));
            return texObj;
        }
    };

    template <uint32_t NumBuffers>
    class InteropTextureObjectHolder {
        Array* m_arrays[NumBuffers];
        TextureSampler* m_texSampler;
        CUtexObject m_texObjs[NumBuffers];
        uint32_t m_numArrays;
        uint32_t m_arrayIndex;
        uint32_t m_bufferIndex;

    public:
        template <uint32_t numArrays>
        void initialize(Array* const (&arrays)[numArrays], TextureSampler* texSampler) {
            for (uint32_t i = 0; i < NumBuffers; ++i)
                m_arrays[i] = arrays[i % numArrays];
            m_arrays = arrays;
            m_texSampler = texSampler;
            m_numArrays = numArrays;
            m_arrayIndex = 0;
            m_bufferIndex = 0;
            for (uint32_t i = 0; i < NumBuffers; ++i)
                m_texObjs[i] = 0;
        }
        void finalize() {
            for (uint32_t i = 0; i < NumBuffers; ++i) {
                CUDADRV_CHECK(cuTexObjectDestroy(m_texObjs[i]));
                m_texObjs[i] = 0;
            }
            m_bufferIndex = 0;
            m_arrayIndex = 0;
            m_texSampler = nullptr;
        }

        void beginCUDAAccess(CUstream stream) {
            m_arrays[m_arrayIndex].beginCUDAAccess(stream, 0);
        }
        void endCUDAAccess(CUstream stream, bool endFrame) {
            m_arrays[m_arrayIndex].endCUDAAccess(stream, 0);
            if (endFrame) {
                m_arrayIndex = (m_arrayIndex + 1) % m_numArrays;
                m_bufferIndex = (m_bufferIndex + 1) % NumBuffers;
            }
        }
        CUtexObject getNext() {
            CUsurfObject &curTexObj = m_texObjs[m_bufferIndex];
            if (curTexObj)
                CUDADRV_CHECK(cuTexObjectDestroy(curTexObj));
            curTexObj = m_texSampler->createTextureObject(m_arrays[m_arrayIndex]);
            return curTexObj;
        }
    };

#endif // #if !defined(__CUDA_ARCH__)
} // namespace cudau
