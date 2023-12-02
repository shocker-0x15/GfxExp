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

#include "cuda_util.h"

#ifdef CUDAUPlatform_Windows_MSVC
#   include <Windows.h>
#   undef near
#   undef far
#   undef min
#   undef max
#endif



namespace cudau {
#ifdef CUDAUPlatform_Windows_MSVC
    void devPrintf(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        char str[1024];
        vsprintf_s(str, fmt, args);
        va_end(args);
        OutputDebugString(str);
    }
#else
    void devPrintf(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        vprintf_s(fmt, args);
        va_end(args);
    }
#endif


//#define USE_PINNED_MAPPED_MEMORY

    static void* allocHostMem(size_t size) {
#if defined(USE_PINNED_MAPPED_MEMORY)
        void* ret;
        CUDADRV_CHECK(cuMemAllocHost(&ret, size));
        return ret;
#else
        return new uint8_t[size];
#endif
    }

    static void releaseHostMem(void* ptr) {
#if defined(USE_PINNED_MAPPED_MEMORY)
        CUDADRV_CHECK(cuMemFreeHost(ptr));
#else
        delete[] ptr;
#endif
    }



    Buffer::Buffer() :
        m_cuContext(nullptr),
        m_hostPointer(nullptr), m_devicePointer(0), m_mappedPointer(nullptr), m_mapFlag(BufferMapFlag::Unmapped),
        m_GLBufferID(0), m_cudaGfxResource(nullptr),
        m_initialized(false), m_persistentMappedMemory(false) {
    }

    Buffer::~Buffer() {
        if (m_initialized)
            finalize();
    }

    Buffer::Buffer(Buffer &&b) {
        m_cuContext = b.m_cuContext;
        m_type = b.m_type;
        m_numElements = b.m_numElements;
        m_stride = b.m_stride;
        m_hostPointer = b.m_hostPointer;
        m_devicePointer = b.m_devicePointer;
        m_mappedPointer = b.m_mappedPointer;
        m_mapFlag = b.m_mapFlag;
        m_GLBufferID = b.m_GLBufferID;
        m_cudaGfxResource = b.m_cudaGfxResource;
        m_initialized = b.m_initialized;
        m_persistentMappedMemory = b.m_persistentMappedMemory;

        b.m_initialized = false;
    }

    Buffer &Buffer::operator=(Buffer &&b) {
        finalize();

        m_cuContext = b.m_cuContext;
        m_type = b.m_type;
        m_numElements = b.m_numElements;
        m_stride = b.m_stride;
        m_hostPointer = b.m_hostPointer;
        m_devicePointer = b.m_devicePointer;
        m_mappedPointer = b.m_mappedPointer;
        m_mapFlag = b.m_mapFlag;
        m_GLBufferID = b.m_GLBufferID;
        m_cudaGfxResource = b.m_cudaGfxResource;
        m_initialized = b.m_initialized;
        m_persistentMappedMemory = b.m_persistentMappedMemory;

        b.m_initialized = false;

        return *this;
    }

    void Buffer::initialize(
        CUcontext context, BufferType type,
        size_t numElements, size_t stride, uint32_t glBufferID) {
        if (m_initialized)
            throw std::runtime_error("Buffer is already initialized.");

        m_cuContext = context;
        m_type = type;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));

        // If using GL Interop, expect that the active device is also the display device.
        if (m_type == BufferType::GL_Interop) {
            CUdevice currentDevice;
            int32_t isDisplayDevice;
            CUDADRV_CHECK(cuCtxGetDevice(&currentDevice));
            CUDADRV_CHECK(cuDeviceGetAttribute(
                &isDisplayDevice, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, currentDevice));
            if (!isDisplayDevice)
                throw std::runtime_error("GL Interop is only available on the display device.");
        }

        m_numElements = numElements;
        m_stride = stride;

        m_hostPointer = nullptr;
        m_devicePointer = 0;
        m_mappedPointer = nullptr;
        m_mapFlag = BufferMapFlag::Unmapped;

        m_GLBufferID = glBufferID;
        m_cudaGfxResource = nullptr;

        size_t size = m_numElements * m_stride;

        if (m_type == BufferType::Device) {
            CUDADRV_CHECK(cuMemAlloc(&m_devicePointer, size));
        }
        else  if (m_type == BufferType::GL_Interop) {
#if defined(CUDA_UTIL_USE_GL_INTEROP)
            CUDADRV_CHECK(cuGraphicsGLRegisterBuffer(
                &m_cudaGfxResource, m_GLBufferID, CU_GRAPHICS_REGISTER_FLAGS_NONE));
#else
            throw std::runtime_error(
                "Disable \"CUDA_UTIL_DONT_USE_GL_INTEROP\" if you use CUDA/OpenGL interoperability.");
#endif
        }
        else if (m_type == BufferType::ZeroCopy) {
            CUDADRV_CHECK(cuMemHostAlloc(
                &m_hostPointer, size, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
            CUDADRV_CHECK(cuMemHostGetDevicePointer(&m_devicePointer, m_hostPointer, 0));
        }
        else { // m_type == BufferType::Managed
            CUDADRV_CHECK(cuMemAllocManaged(&m_devicePointer, size, CU_MEM_ATTACH_GLOBAL));
            m_hostPointer = reinterpret_cast<void*>(m_devicePointer);
        }

        m_persistentMappedMemory = false;

        m_initialized = true;
    }

    void Buffer::finalize() {
        if (!m_initialized)
            return;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));

        if (m_mapFlag != BufferMapFlag::Unmapped)
            unmap();

        if ((m_type == BufferType::Device || m_type == BufferType::GL_Interop) &&
            m_persistentMappedMemory)
            releaseHostMem(m_mappedPointer);
        m_mappedPointer = nullptr;
        m_persistentMappedMemory = false;

        if (m_type == BufferType::Device) {
            CUDADRV_CHECK(cuMemFree(m_devicePointer));
            m_devicePointer = 0;
        }
        else if (m_type == BufferType::GL_Interop) {
            CUDADRV_CHECK(cuGraphicsUnregisterResource(m_cudaGfxResource));
            m_devicePointer = 0;
        }
        else if (m_type == BufferType::ZeroCopy) {
            CUDADRV_CHECK(cuMemFreeHost(m_hostPointer));
            m_devicePointer = 0;
            m_hostPointer = nullptr;
        }
        else { // m_type == BufferType::Managed
            CUDADRV_CHECK(cuMemFree(m_devicePointer));
            m_devicePointer = 0;
            m_hostPointer = nullptr;
        }

        m_stride = 0;
        m_numElements = 0;

        m_cuContext = nullptr;

        m_initialized = false;
    }

    void Buffer::resize(size_t numElements, size_t stride, CUstream stream) {
        if (!m_initialized)
            throw std::runtime_error("Buffer is not initialized.");
        if (m_type == BufferType::GL_Interop)
            throw std::runtime_error("Resize for GL-interop buffer is not supported.");
        if (stride < m_stride)
            throw std::runtime_error("New stride must be >= the current stride.");

        if (numElements == m_numElements && stride == m_stride)
            return;

        Buffer newBuffer;
        newBuffer.initialize(m_cuContext, m_type, numElements, stride, m_GLBufferID);
        newBuffer.setMappedMemoryPersistent(m_persistentMappedMemory);

        size_t numElementsToCopy = std::min(m_numElements, numElements);
        if (stride == m_stride) {
            size_t numBytesToCopy = numElementsToCopy * m_stride;
            CUDADRV_CHECK(cuMemcpyDtoDAsync(newBuffer.m_devicePointer, m_devicePointer, numBytesToCopy, stream));
        }
        else {
            auto src = map<const uint8_t>(stream, BufferMapFlag::ReadOnly);
            auto dst = newBuffer.map<uint8_t>(stream, BufferMapFlag::WriteOnlyDiscard);
            for (size_t i = 0; i < numElementsToCopy; ++i) {
                std::memset(dst, 0, stride);
                std::memcpy(dst, src, m_stride);
            }
            newBuffer.unmap(stream);
            unmap(stream);
        }

        *this = std::move(newBuffer);
    }

    void Buffer::beginCUDAAccess(CUstream stream) {
        if (m_type != BufferType::GL_Interop)
            throw std::runtime_error("This is not an OpenGL-interop buffer.");

        CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));

        size_t bufferSize = 0;
        CUDADRV_CHECK(cuGraphicsMapResources(1, &m_cudaGfxResource, stream));
        CUDADRV_CHECK(cuGraphicsResourceGetMappedPointer(&m_devicePointer, &bufferSize, m_cudaGfxResource));
    }

    void Buffer::endCUDAAccess(CUstream stream) {
        if (m_type != BufferType::GL_Interop)
            throw std::runtime_error("This is not an OpenGL-interop buffer.");

        CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));

        CUDADRV_CHECK(cuGraphicsUnmapResources(1, &m_cudaGfxResource, stream));
    }

    void Buffer::setMappedMemoryPersistent(bool b) {
        if (m_type != BufferType::Device &&
            m_type != BufferType::GL_Interop)
            return;

        m_persistentMappedMemory = b;
        if (m_mapFlag == BufferMapFlag::Unmapped) {
            if (m_persistentMappedMemory) {
                size_t size = m_numElements * m_stride;
                m_mappedPointer = allocHostMem(size);
            }
            else {
                releaseHostMem(m_mappedPointer);
                m_mappedPointer = nullptr;
            }
        }
    }

    void* Buffer::map(CUstream stream, BufferMapFlag flag) {
        if (m_mapFlag != BufferMapFlag::Unmapped)
            throw std::runtime_error("This buffer is already mapped.");

        m_mapFlag = flag;

        if (m_type == BufferType::Device ||
            m_type == BufferType::GL_Interop) {
            CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));

            size_t size = m_numElements * m_stride;
            if (!m_persistentMappedMemory)
                m_mappedPointer = allocHostMem(size);

            if (m_type == BufferType::GL_Interop)
                beginCUDAAccess(stream);

            if (m_mapFlag != BufferMapFlag::WriteOnlyDiscard) {
                CUDADRV_CHECK(cuMemcpyDtoHAsync(m_mappedPointer, m_devicePointer, size, stream));
#if defined(USE_PINNED_MAPPED_MEMORY)
                CUDADRV_CHECK(cuStreamSynchronize(stream));
#endif
            }

            return m_mappedPointer;
        }
        else {
            return m_hostPointer;
        }
    }

    void Buffer::unmap(CUstream stream) {
        if (m_mapFlag == BufferMapFlag::Unmapped)
            throw std::runtime_error("This buffer is not mapped.");

        m_mapFlag = BufferMapFlag::Unmapped;

        if (m_type == BufferType::Device ||
            m_type == BufferType::GL_Interop) {
            CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));

            size_t size = m_numElements * m_stride;

            if (m_mapFlag != BufferMapFlag::ReadOnly)
                CUDADRV_CHECK(cuMemcpyHtoDAsync(m_devicePointer, m_mappedPointer, size, stream));

            if (m_type == BufferType::GL_Interop)
                endCUDAAccess(stream);

            if (!m_persistentMappedMemory) {
                releaseHostMem(m_mappedPointer);
                m_mappedPointer = nullptr;
            }
        }
    }

    Buffer Buffer::copy(CUstream stream) const {
        if (m_GLBufferID != 0)
            throw std::runtime_error("Copying OpenGL buffer is not supported.");

        Buffer ret;
        ret.initialize(m_cuContext, m_type, m_numElements, m_stride, m_GLBufferID);
        ret.setMappedMemoryPersistent(m_persistentMappedMemory);

        size_t size = m_numElements * m_stride;
        if (m_type == BufferType::Device) {
            CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));

            CUDADRV_CHECK(cuMemcpyDtoDAsync(ret.m_devicePointer, m_devicePointer, size, stream));
        }
        else {
            // TODO: Test copy with stream.
            CUDAUAssert_NotImplemented();
            std::memcpy(ret.m_hostPointer, m_hostPointer, size);
        }

        return ret;
    }



    static CUresourceViewFormat getResourceViewFormat(ArrayElementType elemType, uint32_t numChannels) {
#define CUDA_UTIL_EXPR0(arrayEnum, BaseType, BitWidth) \
    case cudau::ArrayElementType::arrayEnum ## BitWidth: \
        if (numChannels == 1) \
            return CU_RES_VIEW_FORMAT_ ## BaseType ## _1X ## BitWidth; \
        else if (numChannels == 2) \
            return CU_RES_VIEW_FORMAT_ ## BaseType ## _2X ## BitWidth; \
        else if (numChannels == 4) \
            return CU_RES_VIEW_FORMAT_ ## BaseType ## _4X ## BitWidth; \
        break    

        switch (elemType) {
            CUDA_UTIL_EXPR0(UInt, UINT, 8);
            CUDA_UTIL_EXPR0(Int, SINT, 8);
            CUDA_UTIL_EXPR0(UInt, UINT, 16);
            CUDA_UTIL_EXPR0(Int, SINT, 16);
            CUDA_UTIL_EXPR0(UInt, UINT, 32);
            CUDA_UTIL_EXPR0(Int, SINT, 32);
            CUDA_UTIL_EXPR0(Float, FLOAT, 16);
            CUDA_UTIL_EXPR0(Float, FLOAT, 32);
        case cudau::ArrayElementType::BC1_UNorm:
            if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC1;
            break;
        case cudau::ArrayElementType::BC2_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC2;
            break;
        case cudau::ArrayElementType::BC3_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC3;
            break;
        case cudau::ArrayElementType::BC4_UNorm:
            if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC4;
            break;
        case cudau::ArrayElementType::BC4_SNorm:
            if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_SIGNED_BC4;
            break;
        case cudau::ArrayElementType::BC5_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC5;
            break;
        case cudau::ArrayElementType::BC5_SNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_SIGNED_BC5;
            break;
        case cudau::ArrayElementType::BC6H_UF16:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC6H;
            break;
        case cudau::ArrayElementType::BC6H_SF16:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_SIGNED_BC6H;
            break;
        case cudau::ArrayElementType::BC7_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC7;
            break;
        default:
            break;
        }
        CUDAUAssert_ShouldNotBeCalled();
        return CU_RES_VIEW_FORMAT_NONE;

#undef CUDA_UTIL_EXPR0
    }



#if defined(CUDA_UTIL_USE_GL_INTEROP)
    void getArrayElementFormat(GLenum internalFormat, ArrayElementType* elemType, uint32_t* numChannels) {
#define CUDA_UTIL_EXPR0(glEnum, arrayEnum, numCh) \
    case glEnum: \
        *elemType = ArrayElementType::arrayEnum; \
        *numChannels = numCh; \
        break
#define CUDA_UTIL_EXPR1(glEnum, numCh) \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 8,    UInt8,   numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 16,   Int16,   numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 16F,  Float16, numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 32F,  Float32, numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 8UI,  UInt8,   numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 16UI, UInt16,  numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 32UI, UInt32,  numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 8I,   Int8,    numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 16I,  Int16,   numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 32I,  Int32,   numCh)

        switch (internalFormat) {
            CUDA_UTIL_EXPR1(R, 1);
            CUDA_UTIL_EXPR1(RG, 2);
            CUDA_UTIL_EXPR1(RGBA, 4);
            CUDA_UTIL_EXPR0(GL_SRGB8_ALPHA8, UInt8, 4);
        default:
            CUDAUAssert_ShouldNotBeCalled();
            break;
        }

#undef CUDA_UTIL_EXPR1
#undef CUDA_UTIL_EXPR0
    }
#endif

    Array::Array() :
        m_cuContext(nullptr),
        m_array(0), m_mappedPointers(nullptr), m_mipmapArrays(nullptr), m_mapFlags(nullptr),
        m_surfObjs(nullptr),
        m_GLTexID(0), m_cudaGfxResource(nullptr),
        m_surfaceLoadStore(false), m_cubemap(false), m_layered(false),
        m_initialized(false) {
    }

    Array::~Array() {
        if (m_initialized)
            finalize();
    }

    Array::Array(Array &&b) {
        m_cuContext = b.m_cuContext;
        m_width = b.m_width;
        m_height = b.m_height;
        m_depth = b.m_depth;
        m_numMipmapLevels = b.m_numMipmapLevels;
        m_stride = b.m_stride;
        m_elemType = b.m_elemType;
        m_numChannels = b.m_numChannels;
        if (m_numMipmapLevels > 1)
            m_mipmappedArray = b.m_mipmappedArray;
        else
            m_array = b.m_array;
        m_mappedPointers = b.m_mappedPointers;
        m_mipmapArrays = b.m_mipmapArrays;
        m_mapFlags = b.m_mapFlags;
        m_surfObjs = b.m_surfObjs;
        m_GLTexID = b.m_GLTexID;
        m_cudaGfxResource = b.m_cudaGfxResource;
        m_surfaceLoadStore = b.m_surfaceLoadStore;
        m_useTextureGather = b.m_useTextureGather;
        m_cubemap = b.m_cubemap;
        m_layered = b.m_layered;
        m_initialized = b.m_initialized;

        b.m_initialized = false;
    }

    Array &Array::operator=(Array &&b) {
        finalize();

        m_cuContext = b.m_cuContext;
        m_width = b.m_width;
        m_height = b.m_height;
        m_depth = b.m_depth;
        m_numMipmapLevels = b.m_numMipmapLevels;
        m_stride = b.m_stride;
        m_elemType = b.m_elemType;
        m_numChannels = b.m_numChannels;
        if (m_numMipmapLevels > 1)
            m_mipmappedArray = b.m_mipmappedArray;
        else
            m_array = b.m_array;
        m_mappedPointers = b.m_mappedPointers;
        m_mipmapArrays = b.m_mipmapArrays;
        m_mapFlags = b.m_mapFlags;
        m_surfObjs = b.m_surfObjs;
        m_GLTexID = b.m_GLTexID;
        m_cudaGfxResource = b.m_cudaGfxResource;
        m_surfaceLoadStore = b.m_surfaceLoadStore;
        m_useTextureGather = b.m_useTextureGather;
        m_cubemap = b.m_cubemap;
        m_layered = b.m_layered;
        m_initialized = b.m_initialized;

        b.m_initialized = false;

        return *this;
    }

    void Array::initialize(
        CUcontext context, ArrayElementType elemType, uint32_t numChannels,
        size_t width, size_t height, size_t depth, uint32_t numMipmapLevels,
        bool surfaceLoadStore, bool useTextureGather, bool cubemap, bool layered, uint32_t glTexID) {
        if (m_initialized)
            throw std::runtime_error("Array is already initialized.");
        if (numChannels != 1 && numChannels != 2 && numChannels != 4)
            throw std::runtime_error("numChannels must be 1, 2, or 4.");
        if (isBCFormat(elemType) && numChannels != 1)
            throw std::runtime_error("numChannels must be 1 for BC format (internally determined).");

        m_cuContext = context;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));

        m_width = width;
        m_height = height;
        m_depth = depth;
        m_numMipmapLevels = std::max(numMipmapLevels, 1u);
        m_elemType = elemType;
        m_numChannels = numChannels;

        m_GLTexID = glTexID;
        m_cudaGfxResource = nullptr;

        m_surfaceLoadStore = surfaceLoadStore;
        m_useTextureGather = useTextureGather;
        m_cubemap = cubemap;
        m_layered = layered;

        CUDA_ARRAY3D_DESCRIPTOR arrayDesc = {};
        if (surfaceLoadStore)
            arrayDesc.Flags |= CUDA_ARRAY3D_SURFACE_LDST;
        if (useTextureGather)
            arrayDesc.Flags |= CUDA_ARRAY3D_TEXTURE_GATHER;
        if (layered)
            arrayDesc.Flags |= CUDA_ARRAY3D_LAYERED;
        if (cubemap)
            arrayDesc.Flags |= CUDA_ARRAY3D_CUBEMAP;
        switch (m_elemType) {
        case cudau::ArrayElementType::UInt8:
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            m_stride = 1;
            break;
        case cudau::ArrayElementType::UInt16:
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
            m_stride = 2;
            break;
        case cudau::ArrayElementType::UInt32:
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
            m_stride = 4;
            break;
        case cudau::ArrayElementType::Int8:
            arrayDesc.Format = CU_AD_FORMAT_SIGNED_INT8;
            m_stride = 1;
            break;
        case cudau::ArrayElementType::Int16:
            arrayDesc.Format = CU_AD_FORMAT_SIGNED_INT16;
            m_stride = 2;
            break;
        case cudau::ArrayElementType::Int32:
            arrayDesc.Format = CU_AD_FORMAT_SIGNED_INT32;
            m_stride = 4;
            break;
        case cudau::ArrayElementType::Float16:
            arrayDesc.Format = CU_AD_FORMAT_HALF;
            m_stride = 2;
            break;
        case cudau::ArrayElementType::Float32:
            arrayDesc.Format = CU_AD_FORMAT_FLOAT;
            m_stride = 4;
            break;
        case cudau::ArrayElementType::BC1_UNorm:
            arrayDesc.Format = CU_AD_FORMAT_BC1_UNORM;
            m_stride = 8;
            m_numChannels = 4;
            break;
        case cudau::ArrayElementType::BC1_UNorm_sRGB:
            arrayDesc.Format = CU_AD_FORMAT_BC1_UNORM_SRGB;
            m_stride = 8;
            m_numChannels = 4;
            break;
        case cudau::ArrayElementType::BC2_UNorm:
            arrayDesc.Format = CU_AD_FORMAT_BC2_UNORM;
            m_stride = 16;
            m_numChannels = 4;
            break;
        case cudau::ArrayElementType::BC2_UNorm_sRGB:
            arrayDesc.Format = CU_AD_FORMAT_BC2_UNORM_SRGB;
            m_stride = 16;
            m_numChannels = 4;
            break;
        case cudau::ArrayElementType::BC3_UNorm:
            arrayDesc.Format = CU_AD_FORMAT_BC3_UNORM;
            m_stride = 16;
            m_numChannels = 4;
            break;
        case cudau::ArrayElementType::BC3_UNorm_sRGB:
            arrayDesc.Format = CU_AD_FORMAT_BC3_UNORM_SRGB;
            m_stride = 16;
            m_numChannels = 4;
            break;
        case cudau::ArrayElementType::BC4_UNorm:
            arrayDesc.Format = CU_AD_FORMAT_BC4_UNORM;
            m_stride = 8;
            m_numChannels = 1;
            break;
        case cudau::ArrayElementType::BC4_SNorm:
            arrayDesc.Format = CU_AD_FORMAT_BC4_SNORM;
            m_stride = 8;
            m_numChannels = 1;
            break;
        case cudau::ArrayElementType::BC5_UNorm:
            arrayDesc.Format = CU_AD_FORMAT_BC5_UNORM;
            m_stride = 16;
            m_numChannels = 2;
            break;
        case cudau::ArrayElementType::BC5_SNorm:
            arrayDesc.Format = CU_AD_FORMAT_BC5_SNORM;
            m_stride = 16;
            m_numChannels = 2;
            break;
        case cudau::ArrayElementType::BC6H_UF16:
            arrayDesc.Format = CU_AD_FORMAT_BC6H_UF16;
            m_stride = 16;
            m_numChannels = 3;
            break;
        case cudau::ArrayElementType::BC6H_SF16:
            arrayDesc.Format = CU_AD_FORMAT_BC6H_SF16;
            m_stride = 16;
            m_numChannels = 3;
            break;
        case cudau::ArrayElementType::BC7_UNorm:
            arrayDesc.Format = CU_AD_FORMAT_BC7_UNORM;
            m_stride = 16;
            m_numChannels = 4;
            break;
        case cudau::ArrayElementType::BC7_UNorm_sRGB:
            arrayDesc.Format = CU_AD_FORMAT_BC7_UNORM_SRGB;
            m_stride = 16;
            m_numChannels = 4;
            break;
        default:
            CUDAUAssert_ShouldNotBeCalled();
            break;
        }
        arrayDesc.Width = m_width;
        arrayDesc.Height = m_height;
        arrayDesc.Depth = m_depth;
        arrayDesc.NumChannels = m_numChannels;
        if (!isBCFormat(elemType))
            m_stride *= m_numChannels;

        if (m_GLTexID != 0) {
#if defined(CUDA_UTIL_USE_GL_INTEROP)
            uint32_t flags = (
                (surfaceLoadStore ?
                 CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST : CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY) |
                (useTextureGather ?
                 CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER : 0));
            CUDADRV_CHECK(cuGraphicsGLRegisterImage(&m_cudaGfxResource, glTexID, GL_TEXTURE_2D, flags));
#else
            throw std::runtime_error(
                "Disable \"CUDA_UTIL_DONT_USE_GL_INTEROP\" if you use CUDA/OpenGL interoperability.");
#endif
        }
        else {
            if (m_numMipmapLevels > 1)
                CUDADRV_CHECK(cuMipmappedArrayCreate(&m_mipmappedArray, &arrayDesc, m_numMipmapLevels));
            else
                CUDADRV_CHECK(cuArray3DCreate(&m_array, &arrayDesc));
        }

        m_mappedPointers = new void*[m_numMipmapLevels];
        m_mipmapArrays = new CUarray[m_numMipmapLevels];
        m_mapFlags = new BufferMapFlag[m_numMipmapLevels];
        for (uint32_t i = 0; i < m_numMipmapLevels; ++i) {
            m_mappedPointers[i] = nullptr;

            if (m_numMipmapLevels > 1 && m_GLTexID == 0)
                CUDADRV_CHECK(cuMipmappedArrayGetLevel(&m_mipmapArrays[i], m_mipmappedArray, i));
            else
                m_mipmapArrays[i] = nullptr;

            m_mapFlags[i] = BufferMapFlag::Unmapped;
        }
        if (surfaceLoadStore && glTexID == 0) {
            m_surfObjs = new CUsurfObject[m_numMipmapLevels];
            if (m_numMipmapLevels > 1) {
                for (uint32_t i = 0; i < m_numMipmapLevels; ++i) {
                    CUarray array;
                    CUDADRV_CHECK(cuMipmappedArrayGetLevel(&array, m_mipmappedArray, i));
                    CUDA_RESOURCE_DESC resDesc = {};
                    resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
                    resDesc.res.array.hArray = array;
                    CUDADRV_CHECK(cuSurfObjectCreate(&m_surfObjs[i], &resDesc));
                }
            }
            else {
                CUDA_RESOURCE_DESC resDesc = {};
                resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
                resDesc.res.array.hArray = m_array;
                CUDADRV_CHECK(cuSurfObjectCreate(&m_surfObjs[0], &resDesc));
            }
        }
        else {
            m_surfObjs = nullptr;
        }

        m_initialized = true;
    }

    void Array::finalize() {
        if (!m_initialized)
            return;

        if (m_surfObjs) {
            for (int i = m_numMipmapLevels - 1; i >= 0; --i)
                CUDADRV_CHECK(cuSurfObjectDestroy(m_surfObjs[i]));
            delete[] m_surfObjs;
            m_surfObjs = nullptr;
        }
        delete[] m_mapFlags;
        m_mapFlags = nullptr;
        delete[] m_mipmapArrays;
        m_mipmapArrays = nullptr;
        delete[] m_mappedPointers;
        m_mappedPointers = nullptr;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));

        if (m_GLTexID != 0) {
            CUDADRV_CHECK(cuGraphicsUnregisterResource(m_cudaGfxResource));
            m_GLTexID = 0;
        }
        else {
            if (m_numMipmapLevels > 1)
                CUDADRV_CHECK(cuMipmappedArrayDestroy(m_mipmappedArray));
            else
                CUDADRV_CHECK(cuArrayDestroy(m_array));
        }

        m_initialized = false;
    }

    void Array::resize(size_t length, CUstream stream) {
        if (m_height > 0 || m_depth > 0)
            throw std::runtime_error("Array dimension cannot be changed.");
        CUDAUAssert_NotImplemented();
    }

    void Array::resize(size_t width, size_t height, CUstream stream) {
        if (m_depth > 0)
            throw std::runtime_error("Array dimension cannot be changed.");
        if (m_numMipmapLevels > 1)
            throw std::runtime_error("resize() is supported only on non-mipmapped array.");

        if (width == m_width && height == m_height)
            return;

        Array newArray;
        newArray.initialize(
            m_cuContext, m_elemType, m_numChannels, width, height, m_depth, m_numMipmapLevels,
            m_surfaceLoadStore, m_useTextureGather, m_cubemap, m_layered, 0);

        size_t copyWidth = std::max<size_t>(std::min(m_width, width), 1u);
        size_t copyHeight = std::max<size_t>(std::min(m_height, height), 1u);
        if (isBCFormat(m_elemType)) {
            copyWidth = (copyWidth + 3) / 4;
            copyHeight = (copyHeight + 3) / 4;
        }

        size_t sizePerRow = copyWidth * m_stride;

        CUDA_MEMCPY3D params = {};
        params.WidthInBytes = sizePerRow;
        params.Height = copyHeight;
        params.Depth = std::max<size_t>(1, m_depth);

        params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        params.srcArray = m_array;
        params.srcXInBytes = 0;
        params.srcY = 0;
        params.srcZ = 0;
        // srcDevice, srcHeight, srcHost, srcLOD, srcPitch are not used in this case.

        params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        params.dstArray = newArray.m_array;
        params.dstXInBytes = 0;
        params.dstY = 0;
        params.dstZ = 0;
        // dstDevice, dstHeight, dstHost, dstLOD, dstPitch are not used in this case.

        CUDADRV_CHECK(cuMemcpy3DAsync(&params, stream));

        *this = std::move(newArray);
    }

    void Array::resize(size_t width, size_t height, size_t depth, CUstream stream) {
        CUDAUAssert_NotImplemented();
    }

    void Array::beginCUDAAccess(CUstream stream, uint32_t mipmapLevel) {
        if (m_GLTexID == 0)
            throw std::runtime_error("This is not an OpenGL-interop object.");

        CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));

        CUDADRV_CHECK(cuGraphicsMapResources(1, &m_cudaGfxResource, stream));
        CUDADRV_CHECK(cuGraphicsSubResourceGetMappedArray(
            &m_mipmapArrays[mipmapLevel], m_cudaGfxResource, 0, mipmapLevel));
    }

    void Array::endCUDAAccess(CUstream stream, uint32_t mipmapLevel) {
        if (m_GLTexID == 0)
            throw std::runtime_error("This is not an OpenGL-interop object.");

        CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));

        m_mipmapArrays[mipmapLevel] = nullptr;
        CUDADRV_CHECK(cuGraphicsUnmapResources(1, &m_cudaGfxResource, stream));
    }

    void* Array::map(uint32_t mipmapLevel, CUstream stream, BufferMapFlag flag) {
        if (m_mapFlags[mipmapLevel] != BufferMapFlag::Unmapped)
            throw std::runtime_error("This mip-map level is already mapped.");
        if (mipmapLevel >= m_numMipmapLevels)
            throw std::runtime_error("Specified mip-map level is out of bounds.");

        CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));

        size_t bw;
        size_t bh;
        computeDimensionsOfLevel(mipmapLevel, &bw, &bh);
        size_t size = std::max<size_t>(1, m_depth) * bh * bw * m_stride;

        m_mappedPointers[mipmapLevel] = allocHostMem(size);
        m_mapFlags[mipmapLevel] = flag;

        if (m_mapFlags[mipmapLevel] != BufferMapFlag::WriteOnlyDiscard) {
            read(reinterpret_cast<uint8_t*>(m_mappedPointers[mipmapLevel]), size, mipmapLevel, stream);
#if defined(USE_PINNED_MAPPED_MEMORY)
            CUDADRV_CHECK(cuStreamSynchronize(stream));
#endif
        }

        return m_mappedPointers[mipmapLevel];
    }

    void Array::unmap(uint32_t mipmapLevel, CUstream stream) {
        if (m_mapFlags[mipmapLevel] == BufferMapFlag::Unmapped)
            throw std::runtime_error("This mip-map level is not mapped.");
        if (mipmapLevel >= m_numMipmapLevels)
            throw std::runtime_error("Specified mip-map level is out of bounds.");

        CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));

        if (m_mapFlags[mipmapLevel] != BufferMapFlag::ReadOnly) {
            size_t bw;
            size_t bh;
            computeDimensionsOfLevel(mipmapLevel, &bw, &bh);
            size_t size = std::max<size_t>(1, m_depth) * bh * bw * m_stride;

            write(reinterpret_cast<uint8_t*>(m_mappedPointers[mipmapLevel]), size, mipmapLevel, stream);
        }

        releaseHostMem(m_mappedPointers[mipmapLevel]);
        m_mappedPointers[mipmapLevel] = nullptr;
        m_mapFlags[mipmapLevel] = BufferMapFlag::Unmapped;
    }

    CUDA_RESOURCE_VIEW_DESC Array::getResourceViewDesc() const {
        CUDA_RESOURCE_VIEW_DESC ret = {};
        size_t width;
        size_t height;
        computeDimensionsOfLevel(0, &width, &height);
        ret.format = getResourceViewFormat(m_elemType, m_numChannels);
        ret.width = width;
        ret.height = height;
        ret.depth = m_depth;
        ret.firstMipmapLevel = 0;
        ret.lastMipmapLevel = m_numMipmapLevels - 1;
        if (m_layered) {
            CUDAUAssert_NotImplemented();
        }
        else {
            ret.firstLayer = 0;
            ret.lastLayer = 0;
        }

        return ret;
    }
}
