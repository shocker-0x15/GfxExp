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
#   define GLUPlatform_Windows
#   if defined(__MINGW32__) // Defined for both 32 bit/64 bit MinGW
#       define GLUPlatform_Windows_MinGW
#   elif defined(_MSC_VER)
#       define GLUPlatform_Windows_MSVC
#       if defined(__INTELLISENSE__)
#           define GLU_CODE_COMPLETION
#       endif
#   endif
#elif defined(__linux__)
#   define GLUPlatform_Linux
#elif defined(__APPLE__)
#   define GLUPlatform_macOS
#elif defined(__OpenBSD__)
#   define GLUPlatform_OpenBSD
#endif



#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include "GL/gl3w.h"



#ifdef _DEBUG
#   define GLU_ENABLE_ASSERT
#endif

#ifdef GLU_ENABLE_ASSERT
#   define GLUAssert(expr, fmt, ...) \
    if (!(expr)) { \
        cudau::devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
        cudau::devPrintf(fmt"\n", ##__VA_ARGS__); \
        abort(); \
    } 0
#else
#   define GLUAssert(expr, fmt, ...)
#endif

#define GLUAssert_ShouldNotBeCalled() GLUAssert(false, "Should not be called!")
#define GLUAssert_NotImplemented() GLUAssert(false, "Not implemented yet!")

#define GL_CHECK(call) \
    do { \
        call; \
        auto error = static_cast<::glu::Error>(glGetError()); \
        if (error != ::glu::Error::NoError) { \
            std::stringstream ss; \
            const char* errMsg = nullptr; \
            ::glu::getErrorString(error, &errMsg); \
            ss << "GL call (" << #call << " ) failed with error: '" \
               << errMsg \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)



namespace glu {
    void devPrintf(const char* fmt, ...);

    enum class Error {
        NoError = GL_NO_ERROR,
        InvalidEnum = GL_INVALID_ENUM,
        InvalidValue = GL_INVALID_VALUE,
        InvalidOperation = GL_INVALID_OPERATION,
        InvalidFramebufferOperation = GL_INVALID_FRAMEBUFFER_OPERATION,
        OutOfMemory = GL_OUT_OF_MEMORY,
        StackUnderflow = GL_STACK_UNDERFLOW,
        StackOverflow = GL_STACK_OVERFLOW,
    };

    static void getErrorString(Error err, const char** errMsg) {
        switch (err) {
        case Error::NoError:
            *errMsg = "NoError: No error has been recorded. The value of this symbolic constant is guaranteed to be 0.";
            break;
        case Error::InvalidEnum:
            *errMsg = "InvalidEnum: An unacceptable value is specified for an enumerated argument. The offending command is ignored and has no other side effect than to set the error flag.";
            break;
        case Error::InvalidValue:
            *errMsg = "InvalidValue: A numeric argument is out of range. The offending command is ignored and has no other side effect than to set the error flag.";
            break;
        case Error::InvalidOperation:
            *errMsg = "InvalidOperation: The specified operation is not allowed in the current state. The offending command is ignored and has no other side effect than to set the error flag.";
            break;
        case Error::InvalidFramebufferOperation:
            *errMsg = "InvalidFramebufferOperation: The framebuffer object is not complete. The offending command is ignored and has no other side effect than to set the error flag.";
            break;
        case Error::OutOfMemory:
            *errMsg = "OutOfMemory: There is not enough memory left to execute the command. The state of the GL is undefined, except for the state of the error flags, after this error is recorded.";
            break;
        case Error::StackUnderflow:
            *errMsg = "StackUnderflow: An attempt has been made to perform an operation that would cause an internal stack to underflow.";
            break;
        case Error::StackOverflow:
            *errMsg = "StackOverflow: An attempt has been made to perform an operation that would cause an internal stack to overflow.";
            break;
        default:
            *errMsg = "Unknown Error.";
            break;
        }
    }

    void enableDebugCallback(bool synchronous);



    class Buffer {
    public:
        struct Target {
            enum Value {
                ArrayBuffer = GL_ARRAY_BUFFER,
                AtomicCounterBuffer = GL_ATOMIC_COUNTER_BUFFER,
                CopyReadBuffer = GL_COPY_READ_BUFFER,
                CopyWriteBuffer = GL_COPY_WRITE_BUFFER,
                DispatchIndirectBuffer = GL_DISPATCH_INDIRECT_BUFFER,
                DrawIndirectBuffer = GL_DRAW_INDIRECT_BUFFER,
                ElementArrayBuffer = GL_ELEMENT_ARRAY_BUFFER,
                PixelPackBuffer = GL_PIXEL_PACK_BUFFER,
                PixelUnpackBuffer = GL_PIXEL_UNPACK_BUFFER,
                ShaderStorageBuffer = GL_SHADER_STORAGE_BUFFER,
                TextureBuffer = GL_TEXTURE_BUFFER,
                TransformFeedbackBuffer = GL_TRANSFORM_FEEDBACK_BUFFER,
                UniformBuffer = GL_UNIFORM_BUFFER,
                Unbound = 0,
            };

            Value value;

            Target() {}
            Target(uint32_t _value) : value(static_cast<Value>(_value)) {}
            operator GLenum() const {
                return static_cast<GLenum>(value);
            }
        };

        struct Usage {
            enum Value {
                StreamDraw = GL_STREAM_DRAW,
                StreamRead = GL_STREAM_READ,
                StreamCopy = GL_STREAM_COPY,
                StaticDraw = GL_STATIC_DRAW,
                StaticRead = GL_STATIC_READ,
                StaticCopy = GL_STATIC_COPY,
                DynamicDraw = GL_DYNAMIC_DRAW,
                DynamicRead = GL_DYNAMIC_READ,
                DynamicCopy = GL_DYNAMIC_COPY,
            };

            Value value;

            Usage() {}
            Usage(uint32_t _value) : value(static_cast<Value>(_value)) {}
            operator GLenum() const {
                return static_cast<GLenum>(value);
            }
        };

    private:
        GLuint m_handle;
        size_t m_stride;
        size_t m_numElements;
        Usage m_usage;

        void* m_mappedPointer;

        struct {
            unsigned int m_initialized : 1;
            unsigned int m_mapped : 1;
        };

        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;

    public:
        Buffer();
        ~Buffer();

        Buffer(Buffer &&b);
        Buffer &operator=(Buffer &&b);

        void initialize(size_t stride, size_t numElements, Usage usage);
        void finalize();
        bool isInitialized() const {
            return m_initialized;
        }

        void resize(size_t stride, size_t numElements);

        size_t sizeInBytes() const {
            return static_cast<size_t>(m_numElements) * m_stride;
        }
        size_t stride() const {
            return m_stride;
        }
        size_t numElements() const {
            return m_numElements;
        }

        void* map();
        template <typename T>
        T* map() {
            return reinterpret_cast<T*>(map());
        }
        void unmap();
        void* getMappedPointer() const {
            if (m_mappedPointer == nullptr)
                throw std::runtime_error("The buffer is not not mapped.");
            return m_mappedPointer;
        }
        template <typename T>
        T* getMappedPointer() const {
            if (m_mappedPointer == nullptr)
                throw std::runtime_error("The buffer is not not mapped.");
            return reinterpret_cast<T*>(m_mappedPointer);
        }
        template <typename T>
        void write(const T* srcValues, size_t numValues) const {
            const size_t transferSize = sizeof(T) * numValues;
            const size_t bufferSize = static_cast<size_t>(m_stride) * m_numElements;
            if (transferSize > bufferSize)
                throw std::runtime_error("Too large transfer");
            glNamedBufferSubData(m_handle, 0, transferSize, srcValues);
        }
        template <typename T>
        void write(const std::vector<T> &values) const {
            write(values.data(), static_cast<uint32_t>(values.size()));
        }
        template <typename T>
        void fill(const T &value) const {
            size_t numValues = (static_cast<size_t>(m_stride) * m_numElements) / sizeof(T);
            std::vector values(numValues, value);
            write(values);
        }

        GLuint getHandle() const {
            return m_handle;
        }
    };



    class Texture2D {
    public:
    private:
        uint32_t m_handle;
        GLenum m_format;
        GLsizei m_width;
        GLsizei m_height;
        uint32_t m_numMipLevels;

        struct {
            unsigned int m_initialized : 1;
        };

        Texture2D(const Texture2D &) = delete;
        Texture2D &operator=(const Texture2D &) = delete;

    public:
        Texture2D();
        ~Texture2D();

        Texture2D(Texture2D &&b);
        Texture2D &operator=(Texture2D &&b);

        void initialize(GLenum format, GLsizei width, GLsizei height, uint32_t numMipLevels);
        void finalize();
        bool isInitialized() const {
            return m_initialized;
        }

        void transferImage(GLenum format, GLenum type, const void* data, uint32_t mipLevel) const;
        void transferCompressedImage(const void* data, GLsizei size, uint32_t mipLevel) const;

        GLuint getHandle() const {
            return m_handle;
        }
    };



    class Sampler {
    public:
        struct MinFilter {
            enum Value {
                Nearest = GL_NEAREST,
                Linear = GL_LINEAR,
                NearestMipMapNearest = GL_NEAREST_MIPMAP_NEAREST,
                LinearMipMapNearest = GL_LINEAR_MIPMAP_NEAREST,
                NearestMipMapLinear = GL_NEAREST_MIPMAP_LINEAR,
                LinearMipMapLinear = GL_LINEAR_MIPMAP_LINEAR,
            };

            Value value;

            MinFilter() {}
            MinFilter(uint32_t _value) : value(static_cast<Value>(_value)) {}
            operator GLenum() const {
                return static_cast<GLenum>(value);
            }
        };

        struct MagFilter {
            enum Value {
                Nearest = GL_NEAREST,
                Linear = GL_LINEAR,
            };

            Value value;

            MagFilter() {}
            MagFilter(uint32_t _value) : value(static_cast<Value>(_value)) {}
            operator GLenum() const {
                return static_cast<GLenum>(value);
            }
        };

        struct WrapMode {
            enum Value {
                Repeat = GL_REPEAT,
                ClampToEdge = GL_CLAMP_TO_EDGE,
                ClampToBorder = GL_CLAMP_TO_BORDER,
                MirroredRepeat = GL_MIRRORED_REPEAT,
            };

            Value value;

            WrapMode() {}
            WrapMode(uint32_t _value) : value(static_cast<Value>(_value)) {}
            operator GLenum() const {
                return static_cast<GLenum>(value);
            }
        };

    private:
        GLuint m_handle;

        struct {
            unsigned int m_initialized : 1;
        };

        Sampler(const Sampler &) = delete;
        Sampler &operator=(const Sampler &) = delete;

    public:
        Sampler();
        ~Sampler();

        Sampler(Sampler &&b);
        Sampler &operator=(Sampler &&b);

        void initialize(MinFilter minFilter, MagFilter magFilter, WrapMode wrapModeS, WrapMode wrapModeT);
        void finalize();

        GLuint getHandle() const {
            return m_handle;
        }
    };



    class FrameBuffer {
    public:
        struct Target {
            enum Value {
                Draw = GL_DRAW_FRAMEBUFFER,
                Read = GL_READ_FRAMEBUFFER,
                ReadDraw = GL_FRAMEBUFFER,
                Unbound = 0,
            };

            Value value;

            Target() {}
            Target(uint32_t _value) : value(static_cast<Value>(_value)) {}
            operator GLenum() const {
                return static_cast<GLenum>(value);
            }
        };

        struct Status {
            enum Value {
                Complete = GL_FRAMEBUFFER_COMPLETE,
                Undefined = GL_FRAMEBUFFER_UNDEFINED,
                IncompleteAttachment = GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT,
                IncompleteMissingAttachment = GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT,
                IncompleteDrawBuffer = GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER,
                IncompleteReadBuffer = GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER,
                Unsupported = GL_FRAMEBUFFER_UNSUPPORTED,
                IncompleteMultisample = GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE,
                IncompleteLayerTargets = GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS,
            };

            Value value;

            Status() {}
            Status(uint32_t _value) : value(static_cast<Value>(_value)) {}
            operator GLenum() const {
                return static_cast<GLenum>(value);
            }
        };

        static void checkStatus(GLuint handle, Target target);

    private:
        GLuint* m_handles;
        Texture2D* m_renderTargetTextures;
        GLenum* m_renderTargetIDs;
        Texture2D* m_depthRenderTargetTextures;
        GLsizei m_width;
        GLsizei m_height;
        uint32_t m_numColorAttachments;
        uint32_t m_multiBufferingFactor;
        uint32_t m_colorIsMultiBuffered;
        bool m_depthIsMultiBuffered;

        struct {
            unsigned int m_initialized : 1;
        };

        FrameBuffer(const FrameBuffer &) = delete;
        FrameBuffer &operator=(const FrameBuffer &) = delete;

    public:
        FrameBuffer();
        ~FrameBuffer();

        FrameBuffer(FrameBuffer &&b);
        FrameBuffer &operator=(FrameBuffer &&b);

        void initialize(
            GLsizei width, GLsizei height, uint32_t multiBufferingFactor,
            const GLenum* internalFormats, uint32_t colorIsMultiBuffered,
            uint32_t numColorAttachments,
            const GLenum* depthInternalFormat, bool depthIsMultiBuffered);
        void finalize();

        void setDrawBuffers() const {
            glDrawBuffers(m_numColorAttachments, m_renderTargetIDs);
        }
        void resetDrawBuffers() const {
            glDrawBuffer(GL_COLOR_ATTACHMENT0);
        }

        GLuint getHandle(uint32_t frameBufferIndex) const {
            return m_handles[frameBufferIndex];
        }

        const Texture2D &getRenderTargetTexture(uint32_t idx, uint32_t frameBufferIndex) const {
            bool multiBuffered = (m_colorIsMultiBuffered >> idx) & 0b1;
            if (!multiBuffered)
                frameBufferIndex = 0;
            return m_renderTargetTextures[m_multiBufferingFactor * idx + frameBufferIndex];
        }
        const Texture2D &getDepthRenderTargetTexture(uint32_t frameBufferIndex) const {
            if (!m_depthIsMultiBuffered)
                frameBufferIndex = 0;
            return m_depthRenderTargetTextures[frameBufferIndex];
        }

        uint32_t getWidth() const {
            return m_width;
        }
        uint32_t getHeight() const {
            return m_height;
        }
    };



    class VertexArray {
    public:
    private:
        GLuint m_handle;

    public:
        VertexArray() : m_handle(0) {}

        void initialize() {
            glCreateVertexArrays(1, &m_handle);
        }

        void finalize() {
            if (m_handle)
                glDeleteVertexArrays(1, &m_handle);
            m_handle = 0;
        }

        GLuint getHandle() const {
            return m_handle;
        }
    };



    class Shader {
    public:
        struct Type {
            enum Value {
                Compute = GL_COMPUTE_SHADER,
                Vertex = GL_VERTEX_SHADER,
                TessControl = GL_TESS_CONTROL_SHADER,
                TessEvaluation = GL_TESS_EVALUATION_SHADER,
                Geometry = GL_GEOMETRY_SHADER,
                Fragment = GL_FRAGMENT_SHADER,
            };

            Value value;

            Type() {}
            Type(uint32_t _value) : value(static_cast<Value>(_value)) {}
            operator GLenum() const {
                return static_cast<GLenum>(value);
            }
        };

        struct PreProcessorDefinition {
            std::string name;
            std::string value;

            PreProcessorDefinition(const std::string &_name) :
                name(_name) {}
            PreProcessorDefinition(const std::string &_name, const std::string &_value) :
                name(_name), value(_value) {}
        };
    private:
        GLuint m_handle;

        struct {
            unsigned int m_initialized : 1;
        };

        Shader(const Shader &) = delete;
        Shader &operator=(const Shader &) = delete;

    public:
        Shader();
        ~Shader();

        Shader(Shader &&b);
        Shader &operator=(Shader &&b);

        void initialize(Type type, const std::string &source);
        void initialize(Type type, const std::filesystem::path &filePath);
        void finalize();

        GLuint getHandle() const {
            return m_handle;
        }
    };



    class GraphicsProgram {
        GLuint m_handle;
        Shader m_vertex;
        Shader m_fragment;

        struct {
            unsigned int m_initialized;
        };

        GraphicsProgram(const GraphicsProgram &) = delete;
        GraphicsProgram &operator=(const GraphicsProgram &) = delete;
    public:
        GraphicsProgram();
        ~GraphicsProgram();

        GraphicsProgram(GraphicsProgram &&b);
        GraphicsProgram &operator=(GraphicsProgram &&b);

        void initializeVSPS(const std::string &vertexSource, const std::string &fragmentSource);
        void initializeVSPS(
            const std::string &glslHead,
            const std::filesystem::path &vertexSourcePath,
            const std::filesystem::path &fragmentSourcePath);
        void finalize();

        GLuint getHandle() const {
            return m_handle;
        }
    };



    class ComputeProgram {
        GLuint m_handle;
        Shader m_compute;

        struct {
            unsigned int m_initialized;
        };

        ComputeProgram(const ComputeProgram &) = delete;
        ComputeProgram &operator=(const ComputeProgram &) = delete;
    public:
        ComputeProgram();
        ~ComputeProgram();

        ComputeProgram(ComputeProgram &&b);
        ComputeProgram &operator=(ComputeProgram &&b);

        void initialize(const std::string &source);
        void initialize(const std::filesystem::path &filePath);
        void finalize();

        GLuint getHandle() const {
            return m_handle;
        }
    };



    class Timer {
        GLuint m_event;
        bool m_startIsValid;
        bool m_endIsValid;

    public:
        void initialize() {
            glCreateQueries(GL_TIME_ELAPSED, 1, &m_event);
            m_startIsValid = false;
            m_endIsValid = false;
        }
        void finalize() {
            m_startIsValid = false;
            m_endIsValid = false;
            glDeleteQueries(1, &m_event);
        }

        void start() {
            glBeginQuery(GL_TIME_ELAPSED, m_event);
            m_startIsValid = true;
        }
        void stop() {
            glEndQuery(GL_TIME_ELAPSED);
            m_endIsValid = true;
        }

        float report() {
            float ret = 0.0f;
            if (m_startIsValid && m_endIsValid) {
                GLint64 elapsedTime;
                glGetQueryObjecti64v(m_event, GL_QUERY_RESULT, &elapsedTime);
                ret = elapsedTime * 1e-6f;
                m_startIsValid = false;
                m_endIsValid = false;
            }
            return ret;
        }
    };
}
