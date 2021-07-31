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

#include "gl_util.h"

#ifdef GLUPlatform_Windows_MSVC
#   include <Windows.h>
#   undef near
#   undef far
#   undef min
#   undef max
#endif

#include <algorithm>

namespace glu {
#ifdef GLUPlatform_Windows_MSVC
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

    template <typename... Types>
    static void throwRuntimeError(bool expr, const char* fmt, const Types &... args) {
        if (expr)
            return;

        char str[2048];
        snprintf(str, sizeof(str), fmt, args...);
        throw std::runtime_error(str);
    }



    static void debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
                              GLsizei length, const GLchar* message,
                              const GLvoid* userParam) {
        char sourceStr[32];
        switch (source) {
        case GL_DEBUG_SOURCE_API:
            snprintf(sourceStr, sizeof(sourceStr), "API");
            break;
        case GL_DEBUG_SOURCE_APPLICATION:
            snprintf(sourceStr, sizeof(sourceStr), "App");
            break;
        case GL_DEBUG_SOURCE_OTHER:
            snprintf(sourceStr, sizeof(sourceStr), "Other");
            break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER:
            snprintf(sourceStr, sizeof(sourceStr), "ShaderCompiler");
            break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:
            snprintf(sourceStr, sizeof(sourceStr), "ThirdParty");
            break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
            snprintf(sourceStr, sizeof(sourceStr), "WindowSystem");
            break;
        default:
            snprintf(sourceStr, sizeof(sourceStr), "Unknown");
            break;
        }

        char typeStr[32];
        switch (type) {
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
            snprintf(typeStr, sizeof(typeStr), "Deprecated");
            break;
        case GL_DEBUG_TYPE_ERROR:
            snprintf(typeStr, sizeof(typeStr), "Error");
            break;
        case GL_DEBUG_TYPE_MARKER:
            snprintf(typeStr, sizeof(typeStr), "Marker");
            break;
        case GL_DEBUG_TYPE_OTHER:
            snprintf(typeStr, sizeof(typeStr), "Other");
            break;
        case GL_DEBUG_TYPE_PERFORMANCE:
            snprintf(typeStr, sizeof(typeStr), "Performance");
            break;
        case GL_DEBUG_TYPE_POP_GROUP:
            snprintf(typeStr, sizeof(typeStr), "PopGroup");
            break;
        case GL_DEBUG_TYPE_PORTABILITY:
            snprintf(typeStr, sizeof(typeStr), "Portability");
            break;
        case GL_DEBUG_TYPE_PUSH_GROUP:
            snprintf(typeStr, sizeof(typeStr), "PushGroup");
            break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
            snprintf(typeStr, sizeof(typeStr), "UndefinedBehavior");
            break;
        default:
            snprintf(typeStr, sizeof(typeStr), "Unknown");
            break;
        }

        char severityStr[32];
        switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH:
            snprintf(severityStr, sizeof(severityStr), "High");
            break;
        case GL_DEBUG_SEVERITY_MEDIUM:
            snprintf(severityStr, sizeof(severityStr), "Medium");
            break;
        case GL_DEBUG_SEVERITY_LOW:
            snprintf(severityStr, sizeof(severityStr), "Low");
            break;
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            snprintf(severityStr, sizeof(severityStr), "Notification");
            break;
        default:
            snprintf(severityStr, sizeof(severityStr), "Unknown");
            break;
        }

        devPrintf("OpenGL [%s][%s][%u][%s]: %s\n",
                  sourceStr, typeStr, id, severityStr, message);
    }

    void enableDebugCallback(bool synchronous) {
        GL_CHECK(glEnable(GL_DEBUG_OUTPUT));
        if (synchronous)
            GL_CHECK(glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS));
        glDebugMessageCallback(debugCallback, nullptr);
    }



    Buffer::Buffer() :
        m_handle(0),
        m_stride(0), m_numElements(0),
        m_usage(Usage::StreamDraw),
        m_mappedPointer(nullptr),
        m_initialized(false), m_mapped(false) {
    }

    Buffer::~Buffer() {
        if (m_initialized)
            finalize();
    }

    Buffer::Buffer(Buffer &&b) {
        m_handle = b.m_handle;
        m_stride = b.m_stride;
        m_numElements = b.m_numElements;
        m_usage = b.m_usage;
        m_mappedPointer = b.m_mappedPointer;
        m_initialized = b.m_initialized;
        m_mapped = b.m_mapped;

        b.m_initialized = false;
    }

    Buffer &Buffer::operator=(Buffer &&b) {
        finalize();

        m_handle = b.m_handle;
        m_stride = b.m_stride;
        m_numElements = b.m_numElements;
        m_usage = b.m_usage;
        m_mappedPointer = b.m_mappedPointer;
        m_initialized = b.m_initialized;
        m_mapped = b.m_mapped;

        b.m_initialized = false;

        return *this;
    }

    void Buffer::initialize(uint32_t stride, uint32_t numElements, Usage usage) {
        throwRuntimeError(!m_initialized, "Buffer is already initialized.");

        m_stride = stride;
        m_numElements = numElements;
        m_usage = usage;

        size_t size = static_cast<size_t>(m_numElements) * m_stride;

        glCreateBuffers(1, &m_handle);
        glNamedBufferData(m_handle, size, nullptr, m_usage);

        m_initialized = true;
    }

    void Buffer::finalize() {
        if (!m_initialized)
            return;

        if (m_mapped)
            unmap();

        m_mappedPointer = nullptr;

        glDeleteBuffers(1, &m_handle);
        m_handle = 0;

        m_stride = 0;
        m_numElements = 0;

        m_initialized = false;
    }

    void Buffer::resize(uint32_t numElements, uint32_t stride) {
        throwRuntimeError(m_initialized, "Buffer is not initialized.");
        throwRuntimeError(stride >= m_stride, "New stride must be >= the current stride.");

        if (numElements == m_numElements && stride == m_stride)
            return;

        Buffer newBuffer;
        newBuffer.initialize(numElements, stride, m_usage);

        uint32_t numElementsToCopy = std::min(m_numElements, numElements);
        if (stride == m_stride) {
            size_t numBytesToCopy = static_cast<size_t>(numElementsToCopy) * m_stride;
            glCopyNamedBufferSubData(m_handle, newBuffer.m_handle,
                                     0, 0, numBytesToCopy);
        }
        else {
            auto src = map<const uint8_t>();
            auto dst = newBuffer.map<uint8_t>();
            for (uint32_t i = 0; i < numElementsToCopy; ++i) {
                std::memset(dst, 0, stride);
                std::memcpy(dst, src, m_stride);
            }
            newBuffer.unmap();
            unmap();
        }

        *this = std::move(newBuffer);
    }

    void* Buffer::map() {
        throwRuntimeError(!m_mapped, "This buffer is already mapped.");

        m_mapped = true;

        size_t size = static_cast<size_t>(m_numElements) * m_stride;

        uint32_t flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT;
        m_mappedPointer = glMapNamedBufferRange(m_handle, 0, size, flags);

        return m_mappedPointer;
    }

    void Buffer::unmap() {
        throwRuntimeError(m_mapped, "This buffer is not mapped.");

        m_mapped = false;

        glUnmapNamedBuffer(m_handle);
        m_mappedPointer = nullptr;
    }



    Texture2D::Texture2D() :
        m_handle(0),
        m_format(0),
        m_width(0), m_height(0), m_numMipLevels(1),
        m_initialized(false) {
    }

    Texture2D::~Texture2D() {
        if (m_initialized)
            finalize();
    }

    Texture2D::Texture2D(Texture2D &&b) {
        m_handle = b.m_handle;
        m_format = b.m_format;
        m_width = b.m_width;
        m_height = b.m_height;
        m_numMipLevels = b.m_numMipLevels;
        m_initialized = b.m_initialized;

        b.m_initialized = false;
    }

    Texture2D &Texture2D::operator=(Texture2D &&b) {
        finalize();

        m_handle = b.m_handle;
        m_format = b.m_format;
        m_width = b.m_width;
        m_height = b.m_height;
        m_numMipLevels = b.m_numMipLevels;
        m_initialized = b.m_initialized;

        b.m_initialized = false;

        return *this;
    }

    void Texture2D::initialize(GLenum format, uint32_t width, uint32_t height, uint32_t numMipLevels) {
        throwRuntimeError(!m_initialized, "Texture2D is already initialized.");

        m_format = format;
        m_width = width;
        m_height = height;
        m_numMipLevels = numMipLevels;

        glCreateTextures(GL_TEXTURE_2D, 1, &m_handle);
        glTextureStorage2D(m_handle, m_numMipLevels, m_format, m_width, m_height);
        glTextureParameteri(m_handle, GL_TEXTURE_BASE_LEVEL, 0);
        glTextureParameteri(m_handle, GL_TEXTURE_MAX_LEVEL, m_numMipLevels - 1);

        m_initialized = true;
    }

    void Texture2D::finalize() {
        if (!m_initialized)
            return;

        glDeleteTextures(1, &m_handle);
        m_handle = 0;

        m_width = 0;
        m_height = 0;
        m_numMipLevels = 0;

        m_initialized = false;
    }

    void Texture2D::transferImage(GLenum format, GLenum type, const void* data, uint32_t mipLevel) const {
        glTextureSubImage2D(m_handle, mipLevel,
                            0, 0, std::max(m_width >> mipLevel, 1u), std::max(m_height >> mipLevel, 1u),
                            format, type, data);
    }

    void Texture2D::transferCompressedImage(const void* data, size_t size, uint32_t mipLevel) const {
        glCompressedTextureSubImage2D(m_handle, mipLevel,
                                      0, 0, std::max(m_width >> mipLevel, 1u), std::max(m_height >> mipLevel, 1u),
                                      m_format, static_cast<GLsizei>(size), data);
    }



    Sampler::Sampler() :
        m_handle(0),
        m_initialized(false) {
    }

    Sampler::~Sampler() {
        if (m_initialized)
            finalize();
    }

    Sampler::Sampler(Sampler &&b) {
        m_handle = b.m_handle;
        m_initialized = b.m_initialized;

        b.m_initialized = false;
    }

    Sampler &Sampler::operator=(Sampler &&b) {
        finalize();

        m_handle = b.m_handle;
        m_initialized = b.m_initialized;

        b.m_initialized = false;

        return *this;
    }
    
    void Sampler::initialize(MinFilter minFilter, MagFilter magFilter, WrapMode wrapModeS, WrapMode wrapModeT) {
        throwRuntimeError(!m_initialized, "Sampler is already initialized.");

        glCreateSamplers(1, &m_handle);
        glSamplerParameteri(m_handle, GL_TEXTURE_MIN_FILTER, minFilter);
        glSamplerParameteri(m_handle, GL_TEXTURE_MAG_FILTER, magFilter);
        glSamplerParameteri(m_handle, GL_TEXTURE_WRAP_S, wrapModeS);
        glSamplerParameteri(m_handle, GL_TEXTURE_WRAP_T, wrapModeT);
    }

    void Sampler::finalize() {
        if (!m_initialized)
            return;

        if (m_handle)
            glDeleteSamplers(1, &m_handle);
        m_handle = 0;

        m_initialized = false;
    }



    // static
    void FrameBuffer::checkStatus(GLuint handle, Target target) {
        Status status;
        status = static_cast<Status>(glCheckNamedFramebufferStatus(handle, target));
        switch (status) {
        case Status::Complete:
            break;
        case Status::Undefined:
            devPrintf("The specified framebuffer is the default read or draw framebuffer, but the default framebuffer does not exist.\n");
            break;
        case Status::IncompleteAttachment:
            devPrintf("Any of the framebuffer attachment points are framebuffer incomplete.\n");
            break;
        case Status::IncompleteMissingAttachment:
            devPrintf("The framebuffer does not have at least one image attached to it.\n");
            break;
        case Status::IncompleteDrawBuffer:
            devPrintf("The value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for any color attachment point(s) named by GL_DRAW_BUFFERi.\n");
            break;
        case Status::IncompleteReadBuffer:
            devPrintf("GL_READ_BUFFER is not GL_NONE and the value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for the color attachment point named by GL_READ_BUFFER.\n");
            break;
        case Status::Unsupported:
            devPrintf("The combination of internal formats of the attached images violates an implementation-dependent set of restrictions.\n");
            break;
        case Status::IncompleteMultisample:
            devPrintf("The value of GL_RENDERBUFFER_SAMPLES is not the same for all attached renderbuffers; if the value of GL_TEXTURE_SAMPLES is the not same for all attached textures; or, if the attached images are a mix of renderbuffers and textures, the value of GL_RENDERBUFFER_SAMPLES does not match the value of GL_TEXTURE_SAMPLES.\n");
            devPrintf("The value of GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not the same for all attached textures; or, if the attached images are a mix of renderbuffers and textures, the value of GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not GL_TRUE for all attached textures.\n");;
            break;
        case Status::IncompleteLayerTargets:
            devPrintf("Any framebuffer attachment is layered, and any populated attachment is not layered, or if all populated color attachments are not from textures of the same target.\n");
            break;
        default:
            break;
        }
    }

    FrameBuffer::FrameBuffer() :
        m_handles(nullptr),
        m_renderTargetTextures(nullptr), m_renderTargetIDs(nullptr),
        m_depthRenderTargetTextures(nullptr),
        m_width(0), m_height(0), m_numColorAttachments(0),
        m_multiBufferingFactor(0),
        m_colorIsMultiBuffered(0b0000'0000'0000'0000'0000'0000'0000'0000),
        m_depthIsMultiBuffered(false),
        m_initialized(false) {
    }

    FrameBuffer::~FrameBuffer() {
        if (m_initialized)
            finalize();
    }

    FrameBuffer::FrameBuffer(FrameBuffer &&b) {
        m_handles = b.m_handles;
        m_renderTargetTextures = b.m_depthRenderTargetTextures;
        m_renderTargetIDs = b.m_renderTargetIDs;
        m_depthRenderTargetTextures = b.m_depthRenderTargetTextures;
        m_width = b.m_width;
        m_height = b.m_height;
        m_numColorAttachments = b.m_numColorAttachments;
        m_multiBufferingFactor = b.m_multiBufferingFactor;
        m_colorIsMultiBuffered = b.m_colorIsMultiBuffered;
        m_depthIsMultiBuffered = b.m_depthIsMultiBuffered;
        m_initialized = b.m_initialized;

        b.m_initialized = false;
    }

    FrameBuffer &FrameBuffer::operator=(FrameBuffer &&b) {
        finalize();

        m_handles = b.m_handles;
        m_renderTargetTextures = b.m_depthRenderTargetTextures;
        m_renderTargetIDs = b.m_renderTargetIDs;
        m_depthRenderTargetTextures = b.m_depthRenderTargetTextures;
        m_width = b.m_width;
        m_height = b.m_height;
        m_numColorAttachments = b.m_numColorAttachments;
        m_multiBufferingFactor = b.m_multiBufferingFactor;
        m_colorIsMultiBuffered = b.m_colorIsMultiBuffered;
        m_depthIsMultiBuffered = b.m_depthIsMultiBuffered;
        m_initialized = b.m_initialized;

        b.m_initialized = false;

        return *this;
    }

    void FrameBuffer::initialize(uint32_t width, uint32_t height, uint32_t multiBufferingFactor,
                                 const GLenum* internalFormats, uint32_t colorIsMultiBuffered,
                                 uint32_t numColorAttachments,
                                 const GLenum* depthInternalFormat, bool depthIsMultiBuffered) {
        throwRuntimeError(numColorAttachments <= 16, "Maximum number of color attachments is 16.");
        throwRuntimeError(!m_initialized, "FrameBuffer is already initialized.");

        m_width = width;
        m_height = height;
        m_multiBufferingFactor = multiBufferingFactor;
        m_numColorAttachments = numColorAttachments;
        m_renderTargetTextures = new Texture2D[m_numColorAttachments * m_multiBufferingFactor];
        m_renderTargetIDs = new GLenum[m_numColorAttachments];
        m_colorIsMultiBuffered = colorIsMultiBuffered;
        m_depthIsMultiBuffered = depthIsMultiBuffered;

        m_handles = new uint32_t[m_multiBufferingFactor];

        glCreateFramebuffers(m_multiBufferingFactor, m_handles);
        if (depthInternalFormat)
            m_depthRenderTargetTextures = new Texture2D[m_multiBufferingFactor];

        // JP: テクスチャー経由でレンダーターゲットを初期化する。
        for (uint32_t i = 0; i < m_numColorAttachments; ++i) {
            m_renderTargetIDs[i] = GL_COLOR_ATTACHMENT0 + i;
            Texture2D &rt = m_renderTargetTextures[m_multiBufferingFactor * i + 0];
            rt.initialize(internalFormats[i], m_width, m_height, 1);
            glNamedFramebufferTexture(m_handles[0], m_renderTargetIDs[i], rt.getHandle(), 0);
        }

        // JP: デプスレンダーターゲットの初期化。
        if (depthInternalFormat) {
            Texture2D &rt = m_depthRenderTargetTextures[0];
            rt.initialize(*depthInternalFormat, m_width, m_height, 1);
            glNamedFramebufferTexture(m_handles[0], GL_DEPTH_ATTACHMENT, rt.getHandle(), 0);
        }

        checkStatus(m_handles[0], Target::ReadDraw);

        for (uint32_t fbIdx = 1; fbIdx < m_multiBufferingFactor; ++fbIdx) {
            // JP: テクスチャー経由でレンダーターゲットを初期化する。
            for (uint32_t i = 0; i < m_numColorAttachments; ++i) {
                bool multiBuffered = (m_colorIsMultiBuffered >> i) & 0b1;
                Texture2D &rt = m_renderTargetTextures[m_multiBufferingFactor * i + (multiBuffered ? fbIdx : 0)];
                if (multiBuffered)
                    rt.initialize(internalFormats[i], m_width, m_height, 1);
                glNamedFramebufferTexture(m_handles[fbIdx], m_renderTargetIDs[i], rt.getHandle(), 0);
            }

            // JP: デプスレンダーターゲットの初期化。
            if (depthInternalFormat) {
                Texture2D &rt = m_depthRenderTargetTextures[m_depthIsMultiBuffered ? fbIdx : 0];
                if (m_depthIsMultiBuffered)
                    rt.initialize(*depthInternalFormat, m_width, m_height, 1);
                glNamedFramebufferTexture(m_handles[fbIdx], GL_DEPTH_ATTACHMENT, rt.getHandle(), 0);
            }

            checkStatus(m_handles[fbIdx], Target::ReadDraw);
        }

        m_initialized = true;
    }

    void FrameBuffer::finalize() {
        if (!m_initialized)
            return;

        for (int fbIdx = m_multiBufferingFactor - 1; fbIdx >= 1; --fbIdx) {
            if (m_depthRenderTargetTextures && m_depthIsMultiBuffered)
                m_depthRenderTargetTextures[fbIdx].finalize();
            for (int i = m_numColorAttachments - 1; i >= 0; --i) {
                bool multiBuffered = (m_colorIsMultiBuffered >> i) & 0b1;
                if (multiBuffered)
                    m_renderTargetTextures[m_multiBufferingFactor * i + fbIdx].finalize();
            }
        }
        if (m_depthRenderTargetTextures)
            m_depthRenderTargetTextures[0].finalize();
        for (int i = m_numColorAttachments - 1; i >= 0; --i)
            m_renderTargetTextures[m_multiBufferingFactor * i + 0].finalize();

        if (m_depthRenderTargetTextures) {
            delete[] m_depthRenderTargetTextures;
            m_depthRenderTargetTextures = nullptr;
        }

        delete[] m_renderTargetIDs;
        delete[] m_renderTargetTextures;
        m_renderTargetIDs = nullptr;
        m_renderTargetTextures = nullptr;

        if (m_handles) {
            glDeleteFramebuffers(m_multiBufferingFactor, m_handles);
            delete[] m_handles;
            m_handles = nullptr;
        }

        m_initialized = false;
    }



    static GLuint compileShader(Shader::Type type, const std::string &source) {
        GLuint handle;
        handle = glCreateShader(type);

        auto glStrSource = reinterpret_cast<const GLchar*>(source.c_str());
        glShaderSource(handle, 1, &glStrSource, NULL);
        glCompileShader(handle);

        GLint logLength;
        glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLength);
        if (logLength > 0) {
            GLchar* log = (GLchar*)malloc(logLength * sizeof(GLchar));
            glGetShaderInfoLog(handle, logLength, &logLength, log);
            devPrintf("Shader Compile Error:\n%s\n", log);
            free(log);
        }

        GLint status;
        glGetShaderiv(handle, GL_COMPILE_STATUS, &status);
        if (status == 0) {
            glDeleteShader(handle);
            return 0;
        }

        return handle;
    }

    static void validateProgram(GLuint handle) {
        GLint status;
        GLint logLength;
        glValidateProgram(handle);
        glGetProgramiv(handle, GL_INFO_LOG_LENGTH, &logLength);
        if (logLength > 0) {
            GLchar* log = (GLchar*)malloc(logLength * sizeof(GLchar));
            glGetProgramInfoLog(handle, logLength, &logLength, log);
            devPrintf("%s", log);
            free(log);
        }
        glGetProgramiv(handle, GL_VALIDATE_STATUS, &status);
        if (status == 0)
            devPrintf("Program Status : GL_FALSE\n");
    }



    Shader::Shader() :
        m_handle(0),
        m_initialized(false) {
    }

    Shader::~Shader() {
        if (m_initialized)
            finalize();
    }

    Shader::Shader(Shader &&b) {
        m_handle = b.m_handle;
        m_initialized = b.m_initialized;

        b.m_initialized = false;
    }

    Shader &Shader::operator=(Shader &&b) {
        finalize();

        m_handle = b.m_handle;
        m_initialized = b.m_initialized;

        b.m_initialized = false;

        return *this;
    }

    void Shader::initialize(Type type, const std::string &source) {
        throwRuntimeError(!m_initialized, "Shader is already initialized.");
        m_handle = compileShader(type, source);
        throwRuntimeError(m_handle, "failed to create a shader.");

        m_initialized = true;
    }

    void Shader::finalize() {
        if (!m_initialized)
            return;

        if (m_handle) {
            glDeleteShader(m_handle);
            m_handle = 0;
        }

        m_initialized = false;
    }



    GraphicsProgram::GraphicsProgram() :
        m_handle(0),
        m_initialized(false) {
    }

    GraphicsProgram::~GraphicsProgram() {
        if (m_initialized)
            finalize();
    }

    GraphicsProgram::GraphicsProgram(GraphicsProgram &&b) {
        m_handle = b.m_handle;
        m_vertex = std::move(b.m_vertex);
        m_fragment = std::move(b.m_fragment);
        m_initialized = b.m_initialized;

        b.m_initialized = false;
    }

    GraphicsProgram &GraphicsProgram::operator=(GraphicsProgram &&b) {
        finalize();

        m_handle = b.m_handle;
        m_vertex = std::move(b.m_vertex);
        m_fragment = std::move(b.m_fragment);
        m_initialized = b.m_initialized;

        b.m_initialized = false;

        return *this;
    }

    void GraphicsProgram::initializeVSPS(const std::string &vertexSource, const std::string &fragmentSource) {
        throwRuntimeError(!m_initialized, "GraphicsProgram is already initialized.");

        m_vertex.initialize(Shader::Type::Vertex, vertexSource);
        m_fragment.initialize(Shader::Type::Fragment, fragmentSource);

        m_handle = glCreateProgram();
        glAttachShader(m_handle, m_vertex.getHandle());
        glAttachShader(m_handle, m_fragment.getHandle());
        glLinkProgram(m_handle);

        GLint logLength;
        glGetProgramiv(m_handle, GL_INFO_LOG_LENGTH, &logLength);
        if (logLength > 0) {
            GLchar* log = (GLchar*)malloc(logLength * sizeof(GLchar));
            glGetProgramInfoLog(m_handle, logLength, &logLength, log);
            devPrintf("%s\n", log);
            free(log);
        }

        GLint status;
        glGetProgramiv(m_handle, GL_LINK_STATUS, &status);
        if (status == 0) {
            finalize();
            return;
        }

        validateProgram(m_handle);

        m_initialized = true;
    }

    void GraphicsProgram::finalize() {
        if (!m_initialized)
            return;

        if (m_handle) {
            glDeleteProgram(m_handle);
            m_handle = 0;
        }

        m_fragment.finalize();
        m_vertex.finalize();

        m_initialized = false;
    }



    ComputeProgram::ComputeProgram() :
        m_handle(0),
        m_initialized(false) {
    }

    ComputeProgram::~ComputeProgram() {
        if (m_initialized)
            finalize();
    }

    ComputeProgram::ComputeProgram(ComputeProgram &&b) {
        m_handle = b.m_handle;
        m_compute = std::move(b.m_compute);
        m_initialized = b.m_initialized;

        b.m_initialized = false;
    }

    ComputeProgram &ComputeProgram::operator=(ComputeProgram &&b) {
        finalize();

        m_handle = b.m_handle;
        m_compute = std::move(b.m_compute);
        m_initialized = b.m_initialized;

        b.m_initialized = false;

        return *this;
    }

    void ComputeProgram::initialize(const std::string &source) {
        throwRuntimeError(!m_initialized, "ComputeProgram is already initialized.");

        m_compute.initialize(Shader::Type::Compute, source);

        m_handle = glCreateProgram();
        glAttachShader(m_handle, m_compute.getHandle());
        glLinkProgram(m_handle);

        GLint logLength;
        glGetProgramiv(m_handle, GL_INFO_LOG_LENGTH, &logLength);
        if (logLength > 0) {
            GLchar* log = (GLchar*)malloc(logLength * sizeof(GLchar));
            glGetProgramInfoLog(m_handle, logLength, &logLength, log);
            devPrintf("%s\n", log);
            free(log);
        }

        GLint status;
        glGetProgramiv(m_handle, GL_LINK_STATUS, &status);
        if (status == 0) {
            finalize();
            return;
        }

        validateProgram(m_handle);

        m_initialized = true;
    }

    void ComputeProgram::finalize() {
        if (!m_initialized)
            return;

        if (m_handle) {
            glDeleteProgram(m_handle);
            m_handle = 0;
        }

        m_compute.finalize();

        m_initialized = false;
    }
}
