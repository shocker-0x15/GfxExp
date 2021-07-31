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

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#    define Platform_Windows
#    if defined(_MSC_VER)
#        define Platform_Windows_MSVC
#    endif
#elif defined(__APPLE__)
#    define Platform_macOS
#endif

#if defined(Platform_Windows_MSVC)
#   define NOMINMAX
#   define _USE_MATH_DEFINES
#   include <Windows.h>
#   undef near
#   undef far
#   undef RGB
#endif

#include "dds_loader.h"

#include <algorithm>
#include <fstream>

#ifdef _DEBUG
#   define ENABLE_ASSERT
#   define DEBUG_SELECT(A, B) A
#else
#   define DEBUG_SELECT(A, B) B
#endif

#ifdef Platform_Windows_MSVC
static void devPrintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char str[4096];
    vsnprintf_s(str, sizeof(str), _TRUNCATE, fmt, args);
    va_end(args);
    OutputDebugString(str);
}
#else
#   define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#endif

#if 1
#   define hpprintf(fmt, ...) do { devPrintf(fmt, ##__VA_ARGS__); printf(fmt, ##__VA_ARGS__); } while (0)
#else
#   define hpprintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif

#ifdef ENABLE_ASSERT
#   define Assert(expr, fmt, ...) if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); abort(); } 0
#else
#   define Assert(expr, fmt, ...)
#endif

#define Assert_ShouldNotBeCalled() Assert(false, "Should not be called!")
#define Assert_NotImplemented() Assert(false, "Not implemented yet!")

template <typename T, size_t size>
constexpr size_t lengthof(const T(&array)[size]) {
    return size;
}

namespace dds {
    struct Header {
        struct Flags {
            enum Value : uint32_t {
                Caps = 1 << 0,
                Height = 1 << 1,
                Width = 1 << 2,
                Pitch = 1 << 3,
                PixelFormat = 1 << 12,
                MipMapCount = 1 << 17,
                LinearSize = 1 << 19,
                Depth = 1 << 23
            } value;

            Flags() : value((Value)0) {}
            Flags(Value v) : value(v) {}

            Flags operator&(Flags v) const {
                return (Value)(value & v.value);
            }
            Flags operator|(Flags v) const {
                return (Value)(value | v.value);
            }
            bool operator==(uint32_t v) const {
                return value == v;
            }
            bool operator!=(uint32_t v) const {
                return value != v;
            }
        };

        struct PFFlags {
            enum Value : uint32_t {
                AlphaPixels = 1 << 0,
                Alpha = 1 << 1,
                FourCC = 1 << 2,
                PaletteIndexed4 = 1 << 3,
                PaletteIndexed8 = 1 << 5,
                RGB = 1 << 6,
                Luminance = 1 << 17,
                BumpDUDV = 1 << 19,
            } value;

            PFFlags() : value((Value)0) {}
            PFFlags(Value v) : value(v) {}

            PFFlags operator&(PFFlags v) const {
                return (Value)(value & v.value);
            }
            PFFlags operator|(PFFlags v) const {
                return (Value)(value | v.value);
            }
            bool operator==(uint32_t v) const {
                return value == v;
            }
            bool operator!=(uint32_t v) const {
                return value != v;
            }
        };

        struct Caps {
            enum Value : uint32_t {
                Alpha = 1 << 1,
                Complex = 1 << 3,
                Texture = 1 << 12,
                MipMap = 1 << 22,
            } value;

            Caps() : value((Value)0) {}
            Caps(Value v) : value(v) {}

            Caps operator&(Caps v) const {
                return (Value)(value & v.value);
            }
            Caps operator|(Caps v) const {
                return (Value)(value | v.value);
            }
            bool operator==(uint32_t v) const {
                return value == v;
            }
            bool operator!=(uint32_t v) const {
                return value != v;
            }
        };

        struct Caps2 {
            enum Value : uint32_t {
                CubeMap = 1 << 9,
                CubeMapPositiveX = 1 << 10,
                CubeMapNegativeX = 1 << 11,
                CubeMapPositiveY = 1 << 12,
                CubeMapNegativeY = 1 << 13,
                CubeMapPositiveZ = 1 << 14,
                CubeMapNegativeZ = 1 << 15,
                Volume = 1 << 22,
            } value;

            Caps2() : value((Value)0) {}
            Caps2(Value v) : value(v) {}

            Caps2 operator&(Caps2 v) const {
                return (Value)(value & v.value);
            }
            Caps2 operator|(Caps2 v) const {
                return (Value)(value | v.value);
            }
            bool operator==(uint32_t v) const {
                return value == v;
            }
            bool operator!=(uint32_t v) const {
                return value != v;
            }
        };

        uint32_t m_magic;
        uint32_t m_size;
        Flags m_flags;
        uint32_t m_height;
        uint32_t m_width;
        uint32_t m_pitchOrLinearSize;
        uint32_t m_depth;
        uint32_t m_mipmapCount;
        uint32_t m_reserved1[11];
        uint32_t m_PFSize;
        PFFlags m_PFFlags;
        uint32_t m_fourCC;
        uint32_t m_RGBBitCount;
        uint32_t m_RBitMask;
        uint32_t m_GBitMask;
        uint32_t m_BBitMask;
        uint32_t m_RGBAlphaBitMask;
        Caps m_caps;
        Caps2 m_caps2;
        uint32_t m_reservedCaps[2];
        uint32_t m_reserved2;
    };
    static_assert(sizeof(Header) == 128, "sizeof(Header) must be 128.");

    struct HeaderDX10 {
        Format m_format;
        uint32_t m_dimension;
        uint32_t m_miscFlag;
        uint32_t m_arraySize;
        uint32_t m_miscFlag2;
    };
    static_assert(sizeof(HeaderDX10) == 20, "sizeof(HeaderDX10) must be 20.");



    uint8_t** load(const char* filepath, int32_t* width, int32_t* height, int32_t* mipCount, size_t** sizes, Format* format) {
        std::ifstream ifs(filepath, std::ios::in | std::ios::binary);
        if (!ifs.is_open()) {
            hpprintf("Not found: %s\n", filepath);
            return nullptr;
        }

        ifs.seekg(0, std::ios::end);
        size_t fileSize = ifs.tellg();

        ifs.clear();
        ifs.seekg(0, std::ios::beg);

        Header header;
        ifs.read((char*)&header, sizeof(Header));
        if (header.m_magic != 0x20534444 || header.m_fourCC != 0x30315844) {
            hpprintf("Non dds (dx10) file: %s", filepath);
            return nullptr;
        }

        HeaderDX10 dx10Header;
        ifs.read((char*)&dx10Header, sizeof(HeaderDX10));

        *width = header.m_width;
        *height = header.m_height;
        *format = (Format)dx10Header.m_format;

        if (*format != Format::BC1_UNorm && *format != Format::BC1_UNorm_sRGB &&
            *format != Format::BC2_UNorm && *format != Format::BC2_UNorm_sRGB &&
            *format != Format::BC3_UNorm && *format != Format::BC3_UNorm_sRGB &&
            *format != Format::BC4_UNorm && *format != Format::BC4_SNorm &&
            *format != Format::BC5_UNorm && *format != Format::BC5_SNorm &&
            *format != Format::BC6H_UF16 && *format != Format::BC6H_SF16 &&
            *format != Format::BC7_UNorm && *format != Format::BC7_UNorm_sRGB) {
            hpprintf("No support for non block compressed formats: %s", filepath);
            return nullptr;
        }

        const size_t dataSize = fileSize - (sizeof(Header) + sizeof(HeaderDX10));

        *mipCount = 1;
        if ((header.m_flags & Header::Flags::MipMapCount) != 0)
            *mipCount = header.m_mipmapCount;

        uint8_t* singleData = new uint8_t[dataSize];
        ifs.read((char*)singleData, dataSize);

        uint8_t** data = new uint8_t*[*mipCount];
        *sizes = new size_t[*mipCount];
        int32_t mipWidth = *width;
        int32_t mipHeight = *height;
        uint32_t blockSize = 16;
        if (*format == Format::BC1_UNorm || *format == Format::BC1_UNorm_sRGB ||
            *format == Format::BC4_UNorm || *format == Format::BC4_SNorm)
            blockSize = 8;
        size_t cumDataSize = 0;
        for (int i = 0; i < *mipCount; ++i) {
            int32_t bw = (mipWidth + 3) / 4;
            int32_t bh = (mipHeight + 3) / 4;
            size_t mipDataSize = bw * bh * blockSize;

            data[i] = singleData + cumDataSize;
            (*sizes)[i] = mipDataSize;
            cumDataSize += mipDataSize;

            mipWidth = std::max<int32_t>(1, mipWidth / 2);
            mipHeight = std::max<int32_t>(1, mipHeight / 2);
        }
        Assert(cumDataSize == dataSize, "Data size mismatch.");

        return data;
    }

    void free(uint8_t** data, int32_t mipCount, size_t* sizes) {
        void* singleData = data[0];
        delete[] sizes;
        delete[] data;
        delete singleData;
    }
}
