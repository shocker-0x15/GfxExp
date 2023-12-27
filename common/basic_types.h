#pragma once

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#    define HP_Platform_Windows
#    if defined(_MSC_VER)
#        define HP_Platform_Windows_MSVC
#    endif
#elif defined(__APPLE__)
#    define HP_Platform_macOS
#endif

#ifdef _DEBUG
#   define ENABLE_ASSERT
#   define DEBUG_SELECT(A, B) A
#else
#   define DEBUG_SELECT(A, B) B
#endif



#if defined(HP_Platform_Windows_MSVC)
#   define NOMINMAX
#   define _USE_MATH_DEFINES
#   include <Windows.h>
#   undef near
#   undef far
#   undef RGB
#endif

// #includes
#if defined(__CUDA_ARCH__)
#else
#   include <cstdio>
#   include <cstdlib>
#   include <immintrin.h>
#endif

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <type_traits>



#include "../utils/optixu_on_cudau.h"



#ifdef HP_Platform_Windows_MSVC
#   if defined(__CUDA_ARCH__)
#       define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#   else
void devPrintf(const char* fmt, ...);
#   endif
#else
#   define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#endif

#if defined(__CUDA_ARCH__)
#   define __Assert(expr, fmt, ...) do { if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); assert(false); } } while (0)
#else
#   define __Assert(expr, fmt, ...) do { if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); abort(); } } while (0)
#endif

#ifdef ENABLE_ASSERT
#   define Assert(expr, fmt, ...) __Assert(expr, fmt, ##__VA_ARGS__)
#else
#   define Assert(expr, fmt, ...)
#endif

#define Assert_Release(expr, fmt, ...) __Assert(expr, fmt, ##__VA_ARGS__)

#define Assert_ShouldNotBeCalled() __Assert(false, "Should not be called!")
#define Assert_NotImplemented() __Assert(false, "Not implemented yet!")

#define V2FMT "%g, %g"
#define V3FMT "%g, %g, %g"
#define V4FMT "%g, %g, %g, %g"
#define v2print(v) (v).x, (v).y
#define v3print(v) (v).x, (v).y, (v).z
#define v4print(v) (v).x, (v).y, (v).z, (v).w
#define rgbprint(v) (v).r, (v).g, (v).b



template <typename T>
concept Number = std::integral<T> || std::floating_point<T>;

template <typename T>
concept Number32bit =
    std::is_same_v<T, int32_t> ||
    std::is_same_v<T, uint32_t> ||
    std::is_same_v<T, float>;



#if !defined(PURE_CUDA) || defined(CUDAU_CODE_COMPLETION)
CUDA_DEVICE_FUNCTION CUDA_INLINE bool isCursorPixel();
CUDA_DEVICE_FUNCTION CUDA_INLINE bool getDebugPrintEnabled();
#endif



template <typename T, size_t size>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr size_t lengthof(const T (&array)[size]) {
    return size;
}



// std-complementary functions for CUDA
namespace stc {
    template <typename T>
    CUDA_COMMON_FUNCTION CUDA_INLINE void swap(T &a, T &b) {
#if defined(__CUDA_ARCH__)
        T temp = a;
        a = b;
        b = temp;
#else
        std::swap(a, b);
#endif
    }

    template <std::floating_point F>
    CUDA_COMMON_FUNCTION CUDA_INLINE bool isinf(const F x) {
#if defined(__CUDA_ARCH__)
        return static_cast<bool>(::isinf(x));
#else
        return std::isinf(x);
#endif
    }

    template <std::floating_point F>
    CUDA_COMMON_FUNCTION CUDA_INLINE bool isnan(const F x) {
#if defined(__CUDA_ARCH__)
        return static_cast<bool>(::isnan(x));
#else
        return std::isnan(x);
#endif
    }

    template <std::floating_point F>
    CUDA_COMMON_FUNCTION CUDA_INLINE bool isfinite(const F x) {
#if defined(__CUDA_ARCH__)
        return static_cast<bool>(::isfinite(x));
#else
        return std::isfinite(x);
#endif
    }

    template <std::floating_point F>
    CUDA_COMMON_FUNCTION CUDA_INLINE void sincos(const F x, F* const s, F* const c) {
#if defined(__CUDA_ARCH__)
        ::sincosf(x, s, c);
#else
        *s = std::sin(x);
        *c = std::cos(x);
#endif
    }
}



template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow2(const T &x) {
    return x * x;
}

template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow3(const T &x) {
    return x * pow2(x);
}

template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow4(const T &x) {
    return pow2(pow2(x));
}

template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow5(const T &x) {
    return x * pow4(x);
}

template <typename T, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T lerp(const T &v0, const T &v1, const F t) {
    return (1 - t) * v0 + t * v1;
}



template <std::integral IntType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr IntType alignUp(
    const IntType value, const uint32_t alignment) {
    return static_cast<IntType>((value + alignment - 1) / alignment * alignment);
}

CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int32_t floorDiv(
    const int32_t value, const uint32_t modulus) {
    return (value < 0 ? (value - static_cast<int32_t>(modulus - 1)) : value) / static_cast<int32_t>(modulus);
}

CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint32_t floorMod(
    const int32_t value, const uint32_t modulus) {
    int32_t r = value % static_cast<int32_t>(modulus);
    return r < 0 ? r + modulus : r;
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t tzcnt(const uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __clz(__brev(x));
#else
    return _tzcnt_u32(x);
#endif
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t lzcnt(const uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __clz(x);
#else
    return _lzcnt_u32(x);
#endif
}

CUDA_COMMON_FUNCTION CUDA_INLINE int32_t popcnt(const uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __popc(x);
#else
    return _mm_popcnt_u32(x);
#endif
}

//     0: 0
//     1: 0
//  2- 3: 1
//  4- 7: 2
//  8-15: 3
// 16-31: 4
// ...
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t prevPowOf2Exponent(const uint32_t x) {
    if (x == 0)
        return 0;
    return 31 - lzcnt(x);
}

//    0: 0
//    1: 0
//    2: 1
// 3- 4: 2
// 5- 8: 3
// 9-16: 4
// ...
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t nextPowOf2Exponent(const uint32_t x) {
    if (x == 0)
        return 0;
    return 32 - lzcnt(x - 1);
}

//     0: 0
//     1: 1
//  2- 3: 2
//  4- 7: 4
//  8-15: 8
// 16-31: 16
// ...
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t prevPowerOf2(const uint32_t x) {
    if (x == 0)
        return 0;
    return 1 << prevPowOf2Exponent(x);
}

//    0: 0
//    1: 1
//    2: 2
// 3- 4: 4
// 5- 8: 8
// 9-16: 16
// ...
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t nextPowerOf2(const uint32_t x) {
    if (x == 0)
        return 0;
    return 1 << nextPowOf2Exponent(x);
}

template <std::integral IntType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr IntType nextMultiplesForPowOf2(
    const IntType x, const uint32_t exponent) {
    const IntType mask = (1 << exponent) - 1;
    return (x + mask) & ~mask;
}

template <std::integral IntType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr IntType nextMultiplierForPowOf2(
    const IntType x, const uint32_t exponent) {
    return nextMultiplesForPowOf2(x, exponent) >> exponent;
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t nthSetBit(uint32_t value, int32_t n) {
    uint32_t idx = 0;
    int32_t count;
    if (n >= popcnt(value))
        return 0xFFFFFFFF;

    for (uint32_t width = 16; width >= 1; width >>= 1) {
        if (value == 0)
            return 0xFFFFFFFF;

        const uint32_t mask = (1 << width) - 1;
        count = popcnt(value & mask);
        if (n >= count) {
            value >>= width;
            n -= count;
            idx += width;
        }
    }

    return idx;
}



// ----------------------------------------------------------------
// JP: CUDAビルトインに対応する型・関数をホスト側で定義しておく。
// EN: Define types and functions on the host corresponding to CUDA built-ins.
#if !defined(__CUDA_ARCH__) && !defined(__CUDACC__)

struct alignas(8) int2 {
    int32_t x, y;
    constexpr int2(const int32_t v = 0) :
        x(v), y(v) {}
    constexpr int2(const int32_t xx, const int32_t yy) :
        x(xx), y(yy) {}
};
inline constexpr int2 make_int2(const int32_t x, const int32_t y) {
    return int2(x, y);
}
struct int3 {
    int32_t x, y, z;
    constexpr int3(const int32_t v = 0) :
        x(v), y(v), z(v) {}
    constexpr int3(const int32_t xx, const int32_t yy, const int32_t zz) :
        x(xx), y(yy), z(zz) {}
};
inline constexpr int3 make_int3(const int32_t x, const int32_t y, const int32_t z) {
    return int3(x, y, z);
}
struct alignas(16) int4 {
    int32_t x, y, z, w;
    constexpr int4(const int32_t v = 0) :
        x(v), y(v), z(v), w(v) {}
    constexpr int4(const int32_t xx, const int32_t yy, const int32_t zz, const int32_t ww) :
        x(xx), y(yy), z(zz), w(ww) {}
};
inline constexpr int4 make_int4(const int32_t x, const int32_t y, const int32_t z, const int32_t w) {
    return int4(x, y, z, w);
}
struct alignas(8) uint2 {
    uint32_t x, y;
    constexpr uint2(const uint32_t v = 0) : x(v), y(v) {}
    constexpr uint2(const uint32_t xx, const uint32_t yy) : x(xx), y(yy) {}
};
inline constexpr uint2 make_uint2(const uint32_t x, const uint32_t y) {
    return uint2(x, y);
}
struct uint3 {
    uint32_t x, y, z;
    constexpr uint3(const uint32_t v = 0) :
        x(v), y(v), z(v) {}
    constexpr uint3(const uint32_t xx, const uint32_t yy, const uint32_t zz) :
        x(xx), y(yy), z(zz) {}
};
inline constexpr uint3 make_uint3(const uint32_t x, const uint32_t y, const uint32_t z) {
    return uint3(x, y, z);
}
struct uint4 {
    uint32_t x, y, z, w;
    constexpr uint4(const uint32_t v = 0) :
        x(v), y(v), z(v), w(v) {}
    constexpr uint4(const uint32_t xx, const uint32_t yy, const uint32_t zz, const uint32_t ww) :
        x(xx), y(yy), z(zz), w(ww) {}
};
inline constexpr uint4 make_uint4(const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t w) {
    return uint4(x, y, z, w);
}
struct alignas(8) float2 {
    float x, y;
    constexpr float2(const float v = 0) :
        x(v), y(v) {}
    constexpr float2(const float xx, const float yy) :
        x(xx), y(yy) {}
};
inline float2 make_float2(const float x, const float y) {
    return float2(x, y);
}
struct float3 {
    float x, y, z;
    constexpr float3(const float v = 0) :
        x(v), y(v), z(v) {}
    constexpr float3(const float xx, const float yy, const float zz) :
        x(xx), y(yy), z(zz) {}
    constexpr float3(const uint3 &v) :
        x(static_cast<float>(v.x)), y(static_cast<float>(v.y)), z(static_cast<float>(v.z)) {}
};
inline constexpr float3 make_float3(const float x, const float y, const float z) {
    return float3(x, y, z);
}
struct alignas(16) float4 {
    float x, y, z, w;
    constexpr float4(const float v = 0) :
        x(v), y(v), z(v), w(v) {}
    constexpr float4(const float xx, const float yy, const float zz, const float ww) :
        x(xx), y(yy), z(zz), w(ww) {}
    constexpr float4(const float3 &xyz, const float ww) :
        x(xyz.x), y(xyz.y), z(xyz.z), w(ww) {}
};
inline constexpr float4 make_float4(const float x, const float y, const float z, const float w) {
    return float4(x, y, z, w);
}

#endif
// END: Define types and functions on the host corresponding to CUDA built-ins.
// ----------------------------------------------------------------



CUDA_COMMON_FUNCTION CUDA_INLINE int2 make_int2(const int32_t v) {
    return make_int2(v, v);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 make_int2(const int3 &v) {
    return make_int2(v.x, v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 make_int2(const uint3 &v) {
    return make_int2(static_cast<int32_t>(v.x), static_cast<int32_t>(v.y));
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const int2 &a, const int2 &b) {
    return a.x == b.x && a.y == b.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const int2 &a, const int2 &b) {
    return a.x != b.x || a.y != b.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const int2 &a, const uint2 &b) {
    return a.x == b.x && a.y == b.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const int2 &a, const uint2 &b) {
    return a.x != b.x || a.y != b.y;
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator+(const int2 &a, const uint2 &b) {
    return make_uint2(a.x + b.x, a.y + b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator+(const int2 &a, const int2 &b) {
    return make_int2(a.x + b.x, a.y + b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator*(const int2 &a, const int2 &b) {
    return make_int2(a.x * b.x, a.y * b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator*(const int2 &a, const uint2 &b) {
    return make_uint2(a.x * b.x, a.y * b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator*(const int32_t a, const int2 &b) {
    return make_int2(a * b.x, a * b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator*(const uint32_t a, const int2 &b) {
    return make_uint2(a * b.x, a * b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator*(const int2 &a, const int32_t b) {
    return make_int2(a.x * b, a.y * b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator*(const int2 &a, const uint32_t b) {
    return make_uint2(a.x * b, a.y * b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator*=(int2 &a, const int2 &b) {
    a.x *= b.x;
    a.y *= b.y;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator*=(int2 &a, const uint2 &b) {
    a.x *= b.x;
    a.y *= b.y;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator*=(int2 &a, const int32_t b) {
    a.x *= b;
    a.y *= b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator*=(int2 &a, const uint32_t b) {
    a.x *= b;
    a.y *= b;
    return a;
}

CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator/(const int2 &a, const int2 &b) {
    return make_int2(a.x / b.x, a.y / b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const int2 &a, const uint2 &b) {
    return make_uint2(a.x / b.x, a.y / b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator/(const int2 &a, const int32_t b) {
    return make_int2(a.x / b, a.y / b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const int2 &a, const uint32_t b) {
    return make_uint2(a.x / b, a.y / b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator/=(int2 &a, const int2 &b) {
    a.x /= b.x;
    a.y /= b.y;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator/=(int2 &a, const uint2 &b) {
    a.x /= b.x;
    a.y /= b.y;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator/=(int2 &a, const int32_t b) {
    a.x /= b;
    a.y /= b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator/=(int2 &a, const uint32_t b) {
    a.x /= b;
    a.y /= b;
    return a;
}

CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator%(const int2 &a, const int2 &b) {
    return make_int2(a.x % b.x, a.y % b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator<<(const int2 &a, const int32_t b) {
    return make_int2(a.x << b, a.y << b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator<<(const int2 &a, const uint32_t b) {
    return make_int2(a.x << b, a.y << b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator<<=(int2 &a, const int32_t b) {
    a = a << b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator<<=(int2 &a, const uint32_t b) {
    a = a << b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator>>(const int2 &a, const int32_t b) {
    return make_int2(a.x >> b, a.y >> b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator>>(const int2 &a, const uint32_t b) {
    return make_int2(a.x >> b, a.y >> b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator>>=(int2 &a, const int32_t b) {
    a = a >> b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator>>=(int2 &a, const uint32_t b) {
    a = a >> b;
    return a;
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint2 make_uint2(const int3 &v) {
    return make_uint2(static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y));
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 make_uint2(const uint3 &v) {
    return make_uint2(v.x, v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const uint2 &a, const uint2 &b) {
    return a.x == b.x && a.y == b.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const uint2 &a, const uint2 &b) {
    return a.x != b.x || a.y != b.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const uint2 &a, const int2 &b) {
    return a.x == b.x && a.y == b.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const uint2 &a, const int2 &b) {
    return a.x != b.x || a.y != b.y;
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator+(const uint2 &a, const uint2 &b) {
    return make_uint2(a.x + b.x, a.y + b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator+(const uint2 &a, const int2 &b) {
    return make_uint2(a.x + b.x, a.y + b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator+=(uint2 &a, const uint32_t b) {
    a.x += b;
    a.y += b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator-(const uint2 &a, const uint32_t b) {
    return make_uint2(a.x - b, a.y - b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator-=(uint2 &a, const uint32_t b) {
    a.x -= b;
    a.y -= b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator*(const uint32_t a, const uint2 &b) {
    return make_uint2(a * b.x, a * b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator*(const uint2 &a, const uint2 &b) {
    return make_uint2(a.x * b.x, a.y * b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator*=(uint2 &a, const uint2 &b) {
    a.x *= b.x;
    a.y *= b.y;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator*=(uint2 &a, const uint32_t b) {
    a.x *= b;
    a.y *= b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const uint2 &a, const uint2 &b) {
    return make_uint2(a.x / b.x, a.y / b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const uint2 &a, const int2 &b) {
    return make_uint2(a.x / b.x, a.y / b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const uint2 &a, const uint32_t b) {
    return make_uint2(a.x / b, a.y / b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator/=(uint2 &a, const uint32_t b) {
    a.x /= b;
    a.y /= b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator%(const uint2 &a, const uint2 &b) {
    return make_uint2(a.x % b.x, a.y % b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator<<(const uint2 &a, const int32_t b) {
    return make_uint2(a.x << b, a.y << b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator<<(const uint2 &a, const uint32_t b) {
    return make_uint2(a.x << b, a.y << b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator<<=(uint2 &a, const int32_t b) {
    a = a << b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator<<=(uint2 &a, const uint32_t b) {
    a = a << b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator>>(const uint2 &a, const int32_t b) {
    return make_uint2(a.x >> b, a.y >> b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator>>(const uint2 &a, const uint32_t b) {
    return make_uint2(a.x >> b, a.y >> b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator>>=(uint2 &a, const int32_t b) {
    a = a >> b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator>>=(uint2 &a, const uint32_t b) {
    a = a >> b;
    return a;
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint2 min(const uint2 &a, const uint2 &b) {
#if !defined(__CUDA_ARCH__)
    using std::min;
#endif
    return make_uint2(min(a.x, b.x), min(a.y, b.y));
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 max(const uint2 &a, const uint2 &b) {
#if !defined(__CUDA_ARCH__)
    using std::max;
#endif
    return make_uint2(max(a.x, b.x), max(a.y, b.y));
}

CUDA_COMMON_FUNCTION CUDA_INLINE float2 min(const float2 &a, const float2 &b) {
    return make_float2(std::fmin(a.x, b.x), std::fmin(a.y, b.y));
}

CUDA_COMMON_FUNCTION CUDA_INLINE float3 make_float3(const float v) {
    return make_float3(v, v, v);
}

CUDA_COMMON_FUNCTION CUDA_INLINE float4 make_float4(const float3 &xyz, const float w) {
    return make_float4(xyz.x, xyz.y, xyz.z, w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 &operator*=(float4 &a, const float4 &b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 getXYZ(const float4 &v) {
    return make_float3(v.x, v.y, v.z);
}



struct Bool2D {
    bool x, y;

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D() :
        x(false), y(false) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Bool2D(const bool v) :
        x(v), y(v) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D(const bool xx, const bool yy) :
        x(xx), y(yy) {}

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool &operator[](const I idx) {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator[](const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&x + idx);
    }
};

CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool all(const Bool2D &v) {
    return v.x && v.y;
}

CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool any(const Bool2D &v) {
    return v.x || v.y;
}



template <std::floating_point F>
struct Vector2D_T {
    F x, y;

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T() :
        x(0.0f), y(0.0f) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Vector2D_T(const F v) :
        x(v), y(v) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T(const F xx, const F yy) :
        x(xx), y(yy) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T(const Vector2D_T<F2> &v) :
        x(v.x), y(v.y) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Vector2D_T(const Vector2D_T<F2> &v) :
        x(static_cast<F>(v.x)), y(static_cast<F>(v.y)) {}

    CUDA_COMMON_FUNCTION CUDA_INLINE explicit /*constexpr*/ operator float2() const {
        return make_float2(x, y);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ float2 toNative() const {
        return make_float2(x, y);
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F &operator[](const I idx) {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F operator[](const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&x + idx);
    }

#define SWZ2(c0, c1)\
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T c0##c1() const {\
        return Vector2D_T(c0, c1);\
    }

    SWZ2(x, y);
    SWZ2(y, x);

#undef SWZ2

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T operator-() const {
        return Vector2D_T(-x, -y);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T &operator+=(const Vector2D_T &r) {
        x += r.x;
        y += r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T &operator-=(const Vector2D_T &r) {
        x -= r.x;
        y -= r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T &operator*=(const F r) {
        x *= r;
        y *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T &operator*=(const Vector2D_T &r) {
        x *= r.x;
        y *= r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T &operator/=(const F r) {
        const F rr = 1 / r;
        x *= rr;
        y *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T &operator/=(const Vector2D_T &r) {
        x /= r.x;
        y /= r.y;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F sqLength() const {
        return x * x + y * y;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F length() const {
        return std::sqrt(sqLength());
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T &normalize() {
        const F l = length();
        return *this /= l;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool allFinite() const {
        return stc::isfinite(x) && stc::isfinite(y);
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D operator==(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Bool2D(a.x == b.x, a.y == b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D operator!=(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Bool2D(a.x != b.x, a.y != b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D operator<(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Bool2D(a.x < b.x, a.y < b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D operator<=(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Bool2D(a.x <= b.x, a.y <= b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D operator>(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Bool2D(a.x > b.x, a.y > b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D operator>=(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Bool2D(a.x >= b.x, a.y >= b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> operator+(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    Vector2D_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> operator-(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    Vector2D_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> operator*(
    const Vector2D_T<F> &a, const N b) {
    Vector2D_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> operator*(
    const N a, const Vector2D_T<F> &b) {
    Vector2D_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> operator*(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    Vector2D_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> operator/(
    const Vector2D_T<F> &a, const N b) {
    Vector2D_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> operator/(
    const N a, const Vector2D_T<F> &b) {
    Vector2D_T<F> ret(static_cast<F>(a));
    ret /= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> operator/(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    Vector2D_T<F> ret = a;
    ret /= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> step(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Vector2D_T<F>(
        b.x >= a.x ? 1.0f : 0.0f,
        b.y >= a.y ? 1.0f : 0.0f);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> sign(
    const Vector2D_T<F> &v) {
    return Vector2D_T<F>(
        v.x > 0.0f ? 1 : v.x < 0.0f ? -1 : 0,
        v.y > 0.0f ? 1 : v.y < 0.0f ? -1 : 0);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> abs(
    const Vector2D_T<F> &v) {
    return Vector2D_T<F>(std::fabs(v.x), std::fabs(v.y));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> min(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Vector2D_T<F>(std::fmin(a.x, b.x), std::fmin(a.y, b.y));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> max(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Vector2D_T<F>(std::fmax(a.x, b.x), std::fmax(a.y, b.y));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F dot(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return a.x * b.x + a.y * b.y;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F cross(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return a.x * b.y - a.y * b.x;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> normalize(
    const Vector2D_T<F> &v) {
    Vector2D_T<F> ret = v;
    ret.normalize();
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> lerp(
    const Vector2D_T<F> &v0, const Vector2D_T<F> &v1, const Vector2D_T<F> &t) {
    return (Vector2D_T<F>(1.0f) - t) * v0 + t * v1;
}



template <std::floating_point F>
struct Point2D_T {
    F x, y;

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T() :
        x(0.0f), y(0.0f) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Point2D_T(const F v) :
        x(v), y(v) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T(const F xx, const F yy) :
        x(xx), y(yy) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Point2D_T(const Vector2D_T<F> &v) :
        x(v.x), y(v.y) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T(const Point2D_T<F2> &v) :
        x(v.x), y(v.y) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
        CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Point2D_T(const Point2D_T<F2> &v) :
        x(static_cast<F>(v.x)), y(static_cast<F>(v.y)) {}

    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr operator Vector2D_T<F>() const {
        return Vector2D_T<F>(x, y);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit /*constexpr*/ operator float2() const {
        return make_float2(x, y);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ float2 toNative() const {
        return make_float2(x, y);
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F &operator[](const I idx) {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F operator[](const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&x + idx);
    }

#define SWZ2(c0, c1)\
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T c0##c1() const {\
        return Point2D_T(c0, c1);\
    }

    SWZ2(x, y);
    SWZ2(y, x);

#undef SWZ2

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T operator-() const {
        return Point2D_T(-x, -y);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T &operator+=(const Point2D_T &r) {
        x += r.x;
        y += r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T &operator+=(const Vector2D_T<F> &r) {
        x += r.x;
        y += r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T &operator-=(const Vector2D_T<F> &r) {
        x -= r.x;
        y -= r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T &operator*=(const F r) {
        x *= r;
        y *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T &operator*=(const Point2D_T &r) {
        x *= r.x;
        y *= r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T &operator/=(const F r) {
        const F rr = 1 / r;
        x *= rr;
        y *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T &operator/=(const Point2D_T &r) {
        x /= r.x;
        y /= r.y;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool allFinite() const {
        return stc::isfinite(x) && stc::isfinite(y);
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D operator==(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Bool2D(a.x == b.x, a.y == b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D operator!=(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Bool2D(a.x != b.x, a.y != b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D operator<(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Bool2D(a.x < b.x, a.y < b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D operator<=(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Bool2D(a.x <= b.x, a.y <= b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D operator>(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Bool2D(a.x > b.x, a.y > b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool2D operator>=(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Bool2D(a.x >= b.x, a.y >= b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> operator+(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    Point2D_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> operator+(
    const Point2D_T<F> &a, const Vector2D_T<F> &b) {
    Point2D_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> operator+(
    const Vector2D_T<F> &a, const Point2D_T<F> &b) {
    Point2D_T<F> ret = b;
    ret += a;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> operator-(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    auto ret = static_cast<Vector2D_T<F>>(a);
    ret -= static_cast<Vector2D_T<F>>(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> operator-(
    const Point2D_T<F> &a, const Vector2D_T<F> &b) {
    Point2D_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> operator-(
    const Vector2D_T<F> &a, const Point2D_T<F> &b) {
    Point2D_T<F> ret = -b;
    ret += a;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> operator*(
    const Point2D_T<F> &a, const N b) {
    Point2D_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> operator*(
    const N a, const Point2D_T<F> &b) {
    Point2D_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> operator*(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    Point2D_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> operator/(
    const Point2D_T<F> &a, const N b) {
    Point2D_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> operator/(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    Point2D_T<F> ret = a;
    ret /= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> step(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Point2D_T<F>(
        b.x >= a.x ? 1.0f : 0.0f,
        b.y >= a.y ? 1.0f : 0.0f);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> sign(
    const Point2D_T<F> &v) {
    return Point2D_T<F>(
        v.x > 0.0f ? 1 : v.x < 0.0f ? -1 : 0,
        v.y > 0.0f ? 1 : v.y < 0.0f ? -1 : 0);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> abs(
    const Point2D_T<F> &v) {
    return Point2D_T<F>(std::fabs(v.x), std::fabs(v.y));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> min(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Point2D_T<F>(std::fmin(a.x, b.x), std::fmin(a.y, b.y));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> max(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Point2D_T<F>(std::fmax(a.x, b.x), std::fmax(a.y, b.y));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F sqDistance(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    Vector2D_T<F> d = b - a;
    return d.x * d.x + d.y * d.y;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F distance(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
#if !defined(__CUDA_ARCH__)
    using std::sqrtf;
#endif
    return sqrtf(sqDistance(a, b));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> lerp(
    const Point2D_T<F> &v0, const Point2D_T<F> &v1, const Point2D_T<F> &t) {
    return Point2D_T<F>(Point2D_T<F>(1.0f) - t) * v0 + t * v1;
}



struct Bool3D {
    bool x, y, z;

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D() :
        x(false), y(false), z(false) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Bool3D(const bool v) :
        x(v), y(v), z(v) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D(const bool xx, const bool yy, const bool zz) :
        x(xx), y(yy), z(zz) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D(const Bool2D &xy, const bool zz) :
        x(xy.x), y(xy.y), z(zz) {}

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool &operator[](const I idx) {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator[](const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&x + idx);
    }
};

CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool all(const Bool3D &v) {
    return v.x && v.y && v.z;
}

CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool any(const Bool3D &v) {
    return v.x || v.y || v.z;
}



template <std::floating_point F, bool isNormal>
struct Vector3D_T {
    F x, y, z;

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T() :
        x(0.0f), y(0.0f), z(0.0f) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Vector3D_T(const F v) :
        x(v), y(v), z(v) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T(const F xx, const F yy, const F zz) :
        x(xx), y(yy), z(zz) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T(const Vector2D_T<F> &xy, const F zz = 0) :
        x(xy.x), y(xy.y), z(zz) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T(const Point2D_T<F> &xy, const F zz = 1) :
        x(xy.x), y(xy.y), z(zz) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Vector3D_T(const Vector3D_T<F, !isNormal> &v) :
        x(v.x), y(v.y), z(v.z) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Vector3D_T(const float3 &v) :
        x(v.x), y(v.y), z(v.z) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T(const Vector3D_T<F2, isNormal> &v) :
        x(v.x), y(v.y), z(v.z) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
        CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Vector3D_T(const Vector3D_T<F2, isNormal> &v) :
        x(static_cast<F>(v.x)), y(static_cast<F>(v.y)), z(static_cast<F>(v.z)) {}

    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr operator Vector3D_T<F, !isNormal>() const {
        return Vector3D_T<F, !isNormal>(x, y, z);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit /*constexpr*/ operator float3() const {
        return make_float3(x, y, z);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ float3 toNative() const {
        return make_float3(x, y, z);
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F &operator[](const I idx) {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F operator[](const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&x + idx);
    }

#define SWZ2(c0, c1)\
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> c0##c1() const {\
        return Vector2D_T<F>(c0, c1);\
    }
#define SWZ3(c0, c1, c2)\
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T c0##c1##c2() const {\
        return Vector3D_T(c0, c1, c2);\
    }

    SWZ2(x, y); SWZ2(x, z);
    SWZ2(y, x); SWZ2(y, z);
    SWZ2(z, x); SWZ2(z, y);

    SWZ3(x, y, z); SWZ3(x, z, y);
    SWZ3(y, x, z); SWZ3(y, z, x);
    SWZ3(z, x, y); SWZ3(z, y, x);

#undef SWZ3
#undef SWZ2

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T operator-() const {
        return Vector3D_T(-x, -y, -z);
    }

    template <bool isNormalB>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T &operator+=(const Vector3D_T<F, isNormalB> &r) {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }
    template <bool isNormalB>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T &operator-=(const Vector3D_T<F, isNormalB> &r) {
        x -= r.x;
        y -= r.y;
        z -= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T &operator*=(const F r) {
        x *= r;
        y *= r;
        z *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T &operator*=(const Vector3D_T &r) {
        x *= r.x;
        y *= r.y;
        z *= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T &operator/=(const F r) {
        F rr = 1 / r;
        x *= rr;
        y *= rr;
        z *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T &operator/=(const Vector3D_T &r) {
        x /= r.x;
        y /= r.y;
        z /= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T &safeDivide(const Vector3D_T &r) {
        x = r.x != 0 ? x / r.x : 0.0f;
        y = r.y != 0 ? y / r.y : 0.0f;
        z = r.z != 0 ? z / r.z : 0.0f;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F sqLength() const {
        return x * x + y * y + z * z;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F length() const {
        return std::sqrt(sqLength());
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T &normalize() {
        const F l = length();
        return *this /= l;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool allZero() const {
        return x == 0 && y == 0 && z == 0;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool allFinite() const {
        return stc::isfinite(x) && stc::isfinite(y) && stc::isfinite(z);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void makeCoordinateSystem(
        Vector3D_T<F, false>* const tangent, Vector3D_T<F, false>* const bitangent) const {
        const F sign = z >= 0 ? 1.0f : -1.0f;
        const F a = -1 / (sign + z);
        const F b = x * y * a;
        *tangent = Vector3D_T<F, false>(1 + sign * x * x * a, sign * b, -sign * x);
        *bitangent = Vector3D_T<F, false>(b, sign + y * y * a, -y);
    }

    // ( 0, 0,  1) <=> phi:      0
    // (-1, 0,  0) <=> phi: 1/2 pi
    // ( 0, 0, -1) <=> phi:   1 pi
    // ( 1, 0,  0) <=> phi: 3/2 pi
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Vector3D_T fromPolarYUp(const F phi, const F theta) {
        F sinPhi, cosPhi;
        F sinTheta, cosTheta;
        stc::sincos(phi, &sinPhi, &cosPhi);
        stc::sincos(theta, &sinTheta, &cosTheta);
        return Vector3D_T(-sinPhi * sinTheta, cosTheta, cosPhi * sinTheta);
    }

    // ( 1,  0, 0) <=> phi:      0
    // ( 0,  1, 0) <=> phi: 1/2 pi
    // (-1,  0, 0) <=> phi:   1 pi
    // ( 0, -1, 0) <=> phi: 3/2 pi
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Vector3D_T fromPolarZUp(const F phi, const F theta) {
        F sinPhi, cosPhi;
        F sinTheta, cosTheta;
        stc::sincos(phi, &sinPhi, &cosPhi);
        stc::sincos(theta, &sinTheta, &cosTheta);
        return Vector3D_T(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    }
};

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator==(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Bool3D(a.x == b.x, a.y == b.y, a.z == b.z);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator!=(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Bool3D(a.x != b.x, a.y != b.y, a.z != b.z);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator<(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Bool3D(a.x < b.x, a.y < b.y, a.z < b.z);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator<=(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Bool3D(a.x <= b.x, a.y <= b.y, a.z <= b.z);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator>(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Bool3D(a.x > b.x, a.y > b.y, a.z > b.z);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator>=(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Bool3D(a.x >= b.x, a.y >= b.y, a.z >= b.z);
}

template <std::floating_point F, bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormalA> operator+(
    const Vector3D_T<F, isNormalA> &a, const Vector3D_T<F, isNormalB> &b) {
    Vector3D_T<F, isNormalA> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F, bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormalA> operator-(
    const Vector3D_T<F, isNormalA> &a, const Vector3D_T<F, isNormalB> &b) {
    Vector3D_T<F, isNormalA> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, bool isNormal, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> operator*(
    const Vector3D_T<F, isNormal> &a, const N b) {
    Vector3D_T<F, isNormal> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> operator*(
    const N a, const Vector3D_T<F, isNormal> &b) {
    Vector3D_T<F, isNormal> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> operator*(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    Vector3D_T<F, isNormal> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, bool isNormal, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> operator/(
    const Vector3D_T<F, isNormal> &a, const N b) {
    Vector3D_T<F, isNormal> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> operator/(
    const N a, const Vector3D_T<F, isNormal> &b) {
    Vector3D_T<F, isNormal> ret(static_cast<F>(a));
    ret /= b;
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> operator/(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    Vector3D_T<F, isNormal> ret = a;
    ret /= b;
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> safeDivide(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    Vector3D_T<F, isNormal> ret = a;
    ret.safeDivide(b);
    return ret;
}

template <std::floating_point F, bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F dot(
    const Vector3D_T<F, isNormalA> &a, const Vector3D_T<F, isNormalB> &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <std::floating_point F, bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, false> cross(
    const Vector3D_T<F, isNormalA> &a, const Vector3D_T<F, isNormalB> &b) {
    return Vector3D_T<F, false>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F length(
    const Vector3D_T<F, isNormal> &v) {
    return v.length();
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> normalize(
    const Vector3D_T<F, isNormal> &v) {
    Vector3D_T<F, isNormal> ret = v;
    ret.normalize();
    return ret;
}

template <std::floating_point F, bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, false> step(
    const Vector3D_T<F, isNormalA> &a, const Vector3D_T<F, isNormalB> &b) {
    return Vector3D_T<F, false>(
        b.x >= a.x ? 1.0f : 0.0f,
        b.y >= a.y ? 1.0f : 0.0f,
        b.z >= a.z ? 1.0f : 0.0f);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> sign(
    const Vector3D_T<F, isNormal> &v) {
    return Vector3D_T<F, isNormal>(
        v.x > 0.0f ? 1 : v.x < 0.0f ? -1 : 0,
        v.y > 0.0f ? 1 : v.y < 0.0f ? -1 : 0,
        v.z > 0.0f ? 1 : v.z < 0.0f ? -1 : 0);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> abs(
    const Vector3D_T<F, isNormal> &v) {
    return Vector3D_T<F, isNormal>(std::fabs(v.x), std::fabs(v.y), std::fabs(v.z));
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> min(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Vector3D_T<F, isNormal>(std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z));
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> max(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Vector3D_T<F, isNormal>(std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z));
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> lerp(
    const Vector3D_T<F, isNormal> &v0, const Vector3D_T<F, isNormal> &v1,
    const Vector3D_T<F, isNormal> &t) {
    return (Vector3D_T<F, isNormal>(1.0f) - t) * v0 + t * v1;
}



template <std::floating_point F>
struct Point3D_T {
    F x, y, z;

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T() :
        x(0.0f), y(0.0f), z(0.0f) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Point3D_T(const F v) :
        x(v), y(v), z(v) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T(const F xx, const F yy, const F zz) :
        x(xx), y(yy), z(zz) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T(const Point2D_T<F> &xy, const F zz) :
        x(xy.x), y(xy.y), z(zz) {}
    template <bool isNormal>
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Point3D_T(const Vector3D_T<F, isNormal> &v) :
        x(v.x), y(v.y), z(v.z) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Point3D_T(const float3 &p) :
        x(p.x), y(p.y), z(p.z) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T(const Point3D_T<F2> &v) :
        x(v.x), y(v.y), z(v.z) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
        CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Point3D_T(const Point3D_T<F2> &v) :
        x(static_cast<F>(v.x)), y(static_cast<F>(v.y)), z(static_cast<F>(v.z)) {}

    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr operator Point2D_T<F>() const {
        return Point2D_T<F>(x, y);
    }
    template <bool isNormal>
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr operator Vector3D_T<F, isNormal>() const {
        return Vector3D_T<F, isNormal>(x, y, z);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit /*constexpr*/ operator float3() const {
        return make_float3(x, y, z);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ float3 toNative() const {
        return make_float3(x, y, z);
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F &operator[](const I idx) {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F operator[](const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&x + idx);
    }

#define SWZ2(c0, c1)\
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> c0##c1() const {\
        return Point2D_T<F>(c0, c1);\
    }
#define SWZ3(c0, c1, c2)\
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T c0##c1##c2() const {\
        return Point3D_T(c0, c1, c2);\
    }

    SWZ2(x, y); SWZ2(x, z);
    SWZ2(y, x); SWZ2(y, z);
    SWZ2(z, x); SWZ2(z, y);

    SWZ3(x, y, z); SWZ3(x, z, y);
    SWZ3(y, x, z); SWZ3(y, z, x);
    SWZ3(z, x, y); SWZ3(z, y, x);

#undef SWZ3
#undef SWZ2

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T operator-() const {
        return Point3D_T(-x, -y, -z);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T &operator+=(const Point3D_T &r) {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }
    template <bool isNormal>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T &operator+=(const Vector3D_T<F, isNormal> &r) {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }
    template <bool isNormal>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T &operator-=(const Vector3D_T<F, isNormal> &r) {
        x -= r.x;
        y -= r.y;
        z -= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T &operator*=(const F r) {
        x *= r;
        y *= r;
        z *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T &operator*=(const Point3D_T &r) {
        x *= r.x;
        y *= r.y;
        z *= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T &operator/=(const F r) {
        const F rr = 1 / r;
        x *= rr;
        y *= rr;
        z *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T &operator/=(const Point3D_T &r) {
        x /= r.x;
        y /= r.y;
        z /= r.z;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool allFinite() const {
        return stc::isfinite(x) && stc::isfinite(y) && stc::isfinite(z);
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator==(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Bool3D(a.x == b.x, a.y == b.y, a.z == b.z);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator!=(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Bool3D(a.x != b.x, a.y != b.y, a.z != b.z);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator<(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Bool3D(a.x < b.x, a.y < b.y, a.z < b.z);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator<=(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Bool3D(a.x <= b.x, a.y <= b.y, a.z <= b.z);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator>(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Bool3D(a.x > b.x, a.y > b.y, a.z > b.z);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator>=(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Bool3D(a.x >= b.x, a.y >= b.y, a.z >= b.z);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> operator+(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    Point3D_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> operator+(
    const Point3D_T<F> &a, const Vector3D_T<F, isNormal> &b) {
    Point3D_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> operator+(
    const Vector3D_T<F, isNormal> &a, const Point3D_T<F> &b) {
    Point3D_T<F> ret = b;
    ret += a;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, false> operator-(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    auto ret = static_cast<Vector3D_T<F, false>>(a);
    ret -= static_cast<Vector3D_T<F, false>>(b);
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> operator-(
    const Point3D_T<F> &a, const Vector3D_T<F, isNormal> &b) {
    Point3D_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> operator-(
    const Vector3D_T<F, isNormal> &a, const Point3D_T<F> &b) {
    Point3D_T<F> ret = -b;
    ret += a;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> operator*(
    const Point3D_T<F> &a, const N b) {
    Point3D_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> operator*(
    const N a, const Point3D_T<F> &b) {
    Point3D_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> operator*(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    Point3D_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> operator/(
    const Point3D_T<F> &a, const N b) {
    Point3D_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> operator/(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    Point3D_T<F> ret = a;
    ret /= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> min(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Point3D_T<F>(std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> max(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Point3D_T<F>(std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F sqDistance(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    Vector3D_T<F, false> d = b - a;
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F distance(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
#if !defined(__CUDA_ARCH__)
    using std::sqrtf;
#endif
    return sqrtf(sqDistance(a, b));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> lerp(
    const Point3D_T<F> &v0, const Point3D_T<F> &v1,
    const Point3D_T<F> &t) {
    return Point3D_T<F>(Point3D_T<F>(1.0f) - t) * v0 + t * v1;
}



struct Bool4D {
    bool x, y, z, w;

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool4D() : x(false), y(false), z(false), w(false) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Bool4D(const bool v) : x(v), y(v), z(v), w(v) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool4D(const bool xx, bool yy, bool zz, bool ww) :
        x(xx), y(yy), z(zz), w(ww) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool4D(const Bool3D &xyz, bool ww) :
        x(xyz.x), y(xyz.y), z(xyz.z), w(ww) {}
};

CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool all(const Bool4D &v) {
    return v.x && v.y && v.z && v.w;
}

CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool any(const Bool4D &v) {
    return v.x || v.y || v.z || v.w;
}



template <std::floating_point F>
struct Vector4D_T {
    F x, y, z, w;

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T() :
        x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Vector4D_T(const F v) :
        x(v), y(v), z(v), w(v) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T(const F xx, const F yy, const F zz, const F ww) :
        x(xx), y(yy), z(zz), w(ww) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T(const Vector3D_T<F, false> &xyz, const F ww = 0) :
        x(xyz.x), y(xyz.y), z(xyz.z), w(ww) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T(const Point3D_T<F> &xyz, const F ww = 1) :
        x(xyz.x), y(xyz.y), z(xyz.z), w(ww) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T(const Vector4D_T<F2> &v) :
        x(v.x), y(v.y), z(v.z), w(v.w) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
        CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Vector4D_T(const Vector4D_T<F2> &v) :
        x(static_cast<F>(v.x)), y(static_cast<F>(v.y)), z(static_cast<F>(v.z)), w(static_cast<F>(v.w)) {}

    template <bool isNormal>
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr operator Vector3D_T<F, isNormal>() const {
        return Vector3D_T<F, isNormal>(x, y, z);
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F &operator[](const I idx) {
        Assert(static_cast<uint32_t>(idx) < 4, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F operator[](const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 4, "idx is out of bound.");
        return *(&x + idx);
    }

#define SWZ2(c0, c1)\
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> c0##c1() const {\
        return Vector2D_T<F>(c0, c1);\
    }
#define SWZ3(c0, c1, c2)\
    template <bool isNormal = false>\
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> c0##c1##c2() const {\
        return Vector3D_T<F, isNormal>(c0, c1, c2);\
    }
#define SWZ4(c0, c1, c2, c3)\
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T c0##c1##c2##c3() const {\
        return Vector4D_T(c0, c1, c2, c3);\
    }

    SWZ2(x, y); SWZ2(x, z); SWZ2(x, w);
    SWZ2(y, x); SWZ2(y, z); SWZ2(y, w);
    SWZ2(z, x); SWZ2(z, y); SWZ2(z, w);
    SWZ2(w, x); SWZ2(w, y); SWZ2(w, z);

    SWZ3(x, y, z); SWZ3(x, y, w); SWZ3(x, z, y); SWZ3(x, z, w); SWZ3(x, w, y); SWZ3(x, w, z);
    SWZ3(y, x, z); SWZ3(y, x, w); SWZ3(y, z, x); SWZ3(y, z, w); SWZ3(y, w, x); SWZ3(y, w, z);
    SWZ3(z, x, y); SWZ3(z, x, w); SWZ3(z, y, x); SWZ3(z, y, w); SWZ3(z, w, x); SWZ3(z, w, y);
    SWZ3(w, x, y); SWZ3(w, x, z); SWZ3(w, y, x); SWZ3(w, y, z); SWZ3(w, z, x); SWZ3(w, z, y);

    SWZ4(x, y, z, w); SWZ4(x, y, w, z); SWZ4(x, z, y, w); SWZ4(x, z, w, y); SWZ4(x, w, y, z); SWZ4(x, w, z, y);
    SWZ4(y, x, z, w); SWZ4(y, x, w, z); SWZ4(y, z, x, w); SWZ4(y, z, w, x); SWZ4(y, w, x, z); SWZ4(y, w, z, x);
    SWZ4(z, x, y, w); SWZ4(z, x, w, y); SWZ4(z, y, x, w); SWZ4(z, y, w, x); SWZ4(z, w, x, y); SWZ4(z, w, y, x);
    SWZ4(w, x, y, z); SWZ4(w, x, z, y); SWZ4(w, y, x, z); SWZ4(w, y, z, x); SWZ4(w, z, x, y); SWZ4(w, z, y, x);

#undef SWZ4
#undef SWZ3
#undef SWZ2


    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T operator-() const {
        return Vector4D_T(-x, -y, -z, -w);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T &operator+=(const Vector4D_T &r) {
        x += r.x;
        y += r.y;
        z += r.z;
        w += r.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T &operator-=(const Vector4D_T &r) {
        x -= r.x;
        y -= r.y;
        z -= r.z;
        w -= r.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T &operator*=(const F r) {
        x *= r;
        y *= r;
        z *= r;
        w *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T &operator*=(const Vector4D_T &r) {
        x *= r.x;
        y *= r.y;
        z *= r.z;
        w *= r.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T &operator/=(const F r) {
        const F rr = 1 / r;
        x *= rr;
        y *= rr;
        z *= rr;
        w *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T &operator/=(const Vector4D_T &r) {
        x /= r.x;
        y /= r.y;
        z /= r.z;
        w /= r.w;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool allFinite() const {
        return stc::isfinite(x) && stc::isfinite(y) && stc::isfinite(z) && stc::isfinite(w);
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool4D operator==(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return Bool4D(a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool4D operator!=(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return Bool4D(a.x != b.x, a.y != b.y, a.z != b.z, a.w != b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool4D operator<(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return Bool4D(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool4D operator<=(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return Bool4D(a.x <= b.x, a.y <= b.y, a.z <= b.z, a.w <= b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool4D operator>(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return Bool4D(a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool4D operator>=(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return Bool4D(a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T<F> operator+(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    Vector4D_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T<F> operator-(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    Vector4D_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T<F> operator*(
    const Vector4D_T<F> &a, const N b) {
    Vector4D_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T<F> operator*(
    const N a, const Vector4D_T<F> &b) {
    Vector4D_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T<F> operator*(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    Vector4D_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T<F> operator/(
    const Vector4D_T<F> &a, const N b) {
    Vector4D_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T<F> operator/(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    Vector4D_T<F> ret = a;
    ret /= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F dot(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}



template <std::floating_point F>
struct Matrix2x2_T {
    union {
        Vector2D_T<F> c0;
        struct {
            F m00, m10;
        };
    };
    union {
        Vector2D_T<F> c1;
        struct {
            F m01, m11;
        };
    };

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T() :
        c0(1, 0), c1(0, 1) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T(
        const Vector2D_T<F> &cc0,
        const Vector2D_T<F> &cc1) :
        c0(cc0), c1(cc1) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T(
        const Point2D_T<F> &cc0,
        const Point2D_T<F> &cc1) :
        c0(static_cast<Vector2D_T<F>>(cc0)),
        c1(static_cast<Vector2D_T<F>>(cc1)) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T(const Matrix2x2_T<F2> &v) :
        c0(v.c0), c1(v.c1) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
        CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Matrix2x2_T(const Matrix2x2_T<F2> &v) :
        c0(static_cast<Vector2D_T<F>>(v.c0)),
        c1(static_cast<Vector2D_T<F>>(v.c1)) {}

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> &operator[](const I idx) {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&c0 + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> operator[](const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&c0 + idx);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T operator-() const {
        return Matrix2x2_T(-c0, -c1);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T &operator+=(const Matrix2x2_T &r) {
        c0 += r.c0;
        c1 += r.c1;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T &operator-=(const Matrix2x2_T &r) {
        c0 -= r.c0;
        c1 -= r.c1;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T &operator*=(const F r) {
        c0 *= r;
        c1 *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T &operator*=(const Matrix2x2_T &r) {
        const Vector2D_T<F> rs[] = { row(0), row(1) };
        m00 = dot(rs[0], r.c0); m01 = dot(rs[0], r.c1);
        m10 = dot(rs[1], r.c0); m11 = dot(rs[1], r.c1);
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T &operator/=(const F r) {
        const F rr = 1 / r;
        c0 *= rr;
        c1 *= rr;
        return *this;
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> row(const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        switch (idx) {
        case 0:
            return Vector2D_T<F>(c0.x, c1.x);
        case 1:
            return Vector2D_T<F>(c0.y, c1.y);
        default:
            return Vector2D_T<F>(NAN);
        }
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T &transpose() {
        const F temp = m10;
        m10 = m01;
        m01 = temp;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T &adjugate() {
        Matrix2x2_T m;
        m.m00 = m11;
        m.m01 = -m01;
        m.m10 = -m10;
        m.m11 = m00;
        *this = m;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T &adjugateWithoutTranspose() {
        Matrix2x2_T m;
        m.m00 = m11;
        m.m01 = -m10;
        m.m10 = -m01;
        m.m11 = m00;
        *this = m;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T &invert() {
        const F det = m00 * m11 - m01 * m10;
        Matrix2x2_T m = *this;
        m.adjugate();
        m /= det;
        *this = m;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool allFinite() const {
        return c0.allFinite() && c1.allFinite();
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> operator+(
    const Matrix2x2_T<F> &a, const Matrix2x2_T<F> &b) {
    Matrix2x2_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> operator-(
    const Matrix2x2_T<F> &a, const Matrix2x2_T<F> &b) {
    Matrix2x2_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> operator*(
    const Matrix2x2_T<F> &a, const N b) {
    Matrix2x2_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> operator*(
    const N a, const Matrix2x2_T<F> &b) {
    Matrix2x2_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> operator*(
    const Matrix2x2_T<F> &a, const Matrix2x2_T<F> &b) {
    Matrix2x2_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> operator*(
    const Matrix2x2_T<F> &a, const Vector2D_T<F> &b) {
    return Vector2D_T<F>(dot(a.row(0), b), dot(a.row(1), b));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> operator*(
    const Matrix2x2_T<F> &a, const Point2D_T<F> &b) {
    const auto vb = static_cast<Vector2D_T<F>>(b);
    return Point2D_T<F>(
        dot(a.row(0), vb),
        dot(a.row(1), vb));
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> operator/(
    const Matrix2x2_T<F> &a, const N b) {
    Matrix2x2_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> transpose(
    const Matrix2x2_T<F> &m) {
    Matrix2x2_T<F> ret = m;
    ret.transpose();
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> adjugate(
    const Matrix2x2_T<F> &m) {
    Matrix2x2_T<F> ret = m;
    ret.adjugate();
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> adjugateWithoutTranspose(
    const Matrix2x2_T<F> &m) {
    Matrix2x2_T<F> ret = m;
    ret.adjugateWithoutTranspose();
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> invert(
    const Matrix2x2_T<F> &m) {
    Matrix2x2_T<F> ret = m;
    ret.invert();
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> scale2D_2x2(
    const Vector2D_T<F> &s) {
    return Matrix2x2_T<F>(
        Vector2D_T<F>(s.x, 0),
        Vector2D_T<F>(0, s.y));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> scale2D_2x2(
    const F sx, const F sy) {
    return scale2D_2x2(Vector2D_T<F>(sx, sy));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> scale2D_2x2(
    const F s) {
    return scale2D_2x2(Vector2D_T<F>(s, s));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix2x2_T<F> rotate2D_2x2(
    const F angle) {
    const F s = std::sin(angle);
    const F c = std::cos(angle);

    Matrix2x2_T<F> ret;
    ret.m00 = c; ret.m01 = -s;
    ret.m10 = s; ret.m11 = c;

    return ret;
}



template <std::floating_point F>
struct Matrix3x2_T {
    union {
        Vector3D_T<F, false> c0;
        struct {
            F m00, m10, m20;
        };
    };
    union {
        Vector3D_T<F, false> c1;
        struct {
            F m01, m11, m21;
        };
    };

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T() :
        c0(0, 0, 0), c1(0, 0, 0) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T(
        const Vector3D_T<F, false> &cc0,
        const Vector3D_T<F, false> &cc1) :
        c0(cc0), c1(cc1) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T(
        const Vector3D_T<F, true> &cc0,
        const Vector3D_T<F, true> &cc1) :
        c0(cc0), c1(cc1) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T(
        const Point3D_T<F> &cc0,
        const Point3D_T<F> &cc1) :
        c0(static_cast<Vector3D_T<F, false>>(cc0)),
        c1(static_cast<Vector3D_T<F, false>>(cc1)) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T(const Matrix3x2_T<F2> &v) :
        c0(v.c0), c1(v.c1) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
        CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Matrix3x2_T(const Matrix3x2_T<F2> &v) :
        c0(static_cast<Vector3D_T<F, false>>(v.c0)),
        c1(static_cast<Vector3D_T<F, false>>(v.c1)) {}

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, false> &operator[](const I idx) {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&c0 + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, false> operator[](const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&c0 + idx);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T operator-() const {
        return Matrix3x2_T(-c0, -c1);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T &operator+=(const Matrix3x2_T &r) {
        c0 += r.c0;
        c1 += r.c1;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T &operator-=(const Matrix3x2_T &r) {
        c0 -= r.c0;
        c1 -= r.c1;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T &operator*=(const F r) {
        c0 *= r;
        c1 *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T &operator/=(const F r) {
        const F rr = 1 / r;
        c0 *= rr;
        c1 *= rr;
        return *this;
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> row(const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        switch (idx) {
        case 0:
            return Vector2D_T<F>(c0.x, c1.x);
        case 1:
            return Vector2D_T<F>(c0.y, c1.y);
        case 2:
            return Vector2D_T<F>(c0.z, c1.z);
        default:
            return Vector2D_T<F>(NAN);
        }
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool allFinite() const {
        return c0.allFinite() && c1.allFinite();
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T<F> operator+(
    const Matrix3x2_T<F> &a, const Matrix3x2_T<F> &b) {
    Matrix3x2_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T<F> operator-(
    const Matrix3x2_T<F> &a, const Matrix3x2_T<F> &b) {
    Matrix3x2_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T<F> operator*(
    const Matrix3x2_T<F> &a, const N b) {
    Matrix3x2_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T<F> operator*(
    const N a, const Matrix3x2_T<F> &b) {
    Matrix3x2_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x2_T<F> operator/(
    const Matrix3x2_T<F> &a, const N b) {
    Matrix3x2_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}



template <std::floating_point F>
struct Matrix3x3_T {
    union {
        Vector3D_T<F, false> c0;
        struct {
            F m00, m10, m20;
        };
    };
    union {
        Vector3D_T<F, false> c1;
        struct {
            F m01, m11, m21;
        };
    };
    union {
        Vector3D_T<F, false> c2;
        struct {
            F m02, m12, m22;
        };
    };

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T() :
        c0(1, 0, 0), c1(0, 1, 0), c2(0, 0, 1) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T(
        const Vector3D_T<F, false> &cc0,
        const Vector3D_T<F, false> &cc1,
        const Vector3D_T<F, false> &cc2) :
        c0(cc0), c1(cc1), c2(cc2) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T(
        const Vector3D_T<F, true> &cc0,
        const Vector3D_T<F, true> &cc1,
        const Vector3D_T<F, true> &cc2) :
        c0(cc0), c1(cc1), c2(cc2) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T(
        const Point3D_T<F> &cc0,
        const Point3D_T<F> &cc1,
        const Point3D_T<F> &cc2) :
        c0(static_cast<Vector3D_T<F, false>>(cc0)),
        c1(static_cast<Vector3D_T<F, false>>(cc1)),
        c2(static_cast<Vector3D_T<F, false>>(cc2)) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T(
        const Matrix2x2_T<F> &mat2x2) :
        c0(Vector3D_T<F, false>(mat2x2.c0)),
        c1(Vector3D_T<F, false>(mat2x2.c1)),
        c2(0.0f, 0.0f, 1.0f)
    {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T(
        const Matrix2x2_T<F> &mat2x2, const Vector2D_T<F> &position) :
        c0(Vector3D_T<F, false>(mat2x2.c0)),
        c1(Vector3D_T<F, false>(mat2x2.c1)),
        c2(Vector3D_T<F, false>(position, 1.0f))
    {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T(
        const Matrix2x2_T<F> &mat2x2, const Point2D_T<F> &position) :
        c0(Vector3D_T<F, false>(mat2x2.c0)),
        c1(Vector3D_T<F, false>(mat2x2.c1)),
        c2(Vector3D_T<F, false>(position))
    {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T(const Matrix3x3_T<F2> &v) :
        c0(v.c0), c1(v.c1), c2(v.c2) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
        CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Matrix3x3_T(const Matrix3x3_T<F2> &v) :
        c0(static_cast<Vector3D_T<F, false>>(v.c0)),
        c1(static_cast<Vector3D_T<F, false>>(v.c1)),
        c2(static_cast<Vector3D_T<F, false>>(v.c2)) {}

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, false> &operator[](const I idx) {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&c0 + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, false> operator[](const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&c0 + idx);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T operator-() const {
        return Matrix3x3_T(-c0, -c1, -c2);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T &operator+=(const Matrix3x3_T &r) {
        c0 += r.c0;
        c1 += r.c1;
        c2 += r.c2;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T &operator-=(const Matrix3x3_T &r) {
        c0 -= r.c0;
        c1 -= r.c1;
        c2 -= r.c2;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T &operator*=(const F r) {
        c0 *= r;
        c1 *= r;
        c2 *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T &operator*=(const Matrix3x3_T &r) {
        const Vector3D_T<F, false> rs[] = { row(0), row(1), row(2) };
        m00 = dot(rs[0], r.c0); m01 = dot(rs[0], r.c1); m02 = dot(rs[0], r.c2);
        m10 = dot(rs[1], r.c0); m11 = dot(rs[1], r.c1); m12 = dot(rs[1], r.c2);
        m20 = dot(rs[2], r.c0); m21 = dot(rs[2], r.c1); m22 = dot(rs[2], r.c2);
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T &operator/=(const F r) {
        const F rr = 1 / r;
        c0 *= rr;
        c1 *= rr;
        c2 *= rr;
        return *this;
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, false> row(const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        switch (idx) {
        case 0:
            return Vector3D_T<F, false>(c0.x, c1.x, c2.x);
        case 1:
            return Vector3D_T<F, false>(c0.y, c1.y, c2.y);
        case 2:
            return Vector3D_T<F, false>(c0.z, c1.z, c2.z);
        default:
            return Vector3D_T<F, false>(NAN);
        }
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T &transpose() {
        F temp;
        temp = m10; m10 = m01; m01 = temp;
        temp = m20; m20 = m02; m02 = temp;
        temp = m21; m21 = m12; m12 = temp;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T &adjugate() {
        Matrix3x3_T m;
        m.m00 = (m11 * m22 - m12 * m21);
        m.m01 = -(m01 * m22 - m02 * m21);
        m.m02 = (m01 * m12 - m02 * m11);
        m.m10 = -(m10 * m22 - m12 * m20);
        m.m11 = (m00 * m22 - m02 * m20);
        m.m12 = -(m00 * m12 - m02 * m10);
        m.m20 = (m10 * m21 - m11 * m20);
        m.m21 = -(m00 * m21 - m01 * m20);
        m.m22 = (m00 * m11 - m01 * m10);
        *this = m;

        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T &adjugateWithoutTranspose() {
        Matrix3x3_T m;
        m.m00 = (m11 * m22 - m12 * m21);
        m.m01 = -(m10 * m22 - m12 * m20);
        m.m02 = (m10 * m21 - m11 * m20);
        m.m10 = -(m01 * m22 - m02 * m21);
        m.m11 = (m00 * m22 - m02 * m20);
        m.m12 = -(m00 * m21 - m01 * m20);
        m.m20 = (m01 * m12 - m02 * m11);
        m.m21 = -(m00 * m12 - m02 * m10);
        m.m22 = (m00 * m11 - m01 * m10);
        *this = m;

        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T &invert() {
        const F det = m00 * m11 * m22 + m01 * m12 * m20 + m02 * m10 * m21
            - m02 * m11 * m20 - m01 * m10 * m22 - m00 * m12 * m21;
        Matrix3x3_T m = *this;
        m.adjugate();
        m /= det;
        *this = m;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool allFinite() const {
        return c0.allFinite() && c1.allFinite() && c2.allFinite();
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> operator+(
    const Matrix3x3_T<F> &a, const Matrix3x3_T<F> &b) {
    Matrix3x3_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> operator-(
    const Matrix3x3_T<F> &a, const Matrix3x3_T<F> &b) {
    Matrix3x3_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> operator*(
    const Matrix3x3_T<F> &a, const N b) {
    Matrix3x3_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> operator*(
    const N a, const Matrix3x3_T<F> &b) {
    Matrix3x3_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> operator*(
    const Matrix3x3_T<F> &a, const Matrix3x3_T<F> &b) {
    Matrix3x3_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> operator/(
    const Matrix3x3_T<F> &a, const N b) {
    Matrix3x3_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector2D_T<F> operator*(
    const Matrix3x3_T<F> &a, const Vector2D_T<F> &b) {
    const Vector3D_T<F, false> r[] = { a.row(0), a.row(1) };
    const Vector3D_T<F, false> v3(b, 0.0f);
    return Vector2D_T<F>(
        dot(r[0], v3),
        dot(r[1], v3));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point2D_T<F> operator*(
    const Matrix3x3_T<F> &a, const Point2D_T<F> &b) {
    const Vector3D_T<F, false> r[] = { a.row(0), a.row(1) };
    const Vector3D_T<F, false> v3(b, 1.0f);
    return Point2D_T<F>(
        dot(r[0], v3),
        dot(r[1], v3));
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, isNormal> operator*(
    const Matrix3x3_T<F> &a, const Vector3D_T<F, isNormal> &b) {
    return Vector3D_T<F, isNormal>(
        dot(a.row(0), b),
        dot(a.row(1), b),
        dot(a.row(2), b));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> operator*(
    const Matrix3x3_T<F> &a, const Point3D_T<F> &b) {
    const auto vb = static_cast<Vector3D_T<F, false>>(b);
    return Point3D_T<F>(
        dot(a.row(0), vb),
        dot(a.row(1), vb),
        dot(a.row(2), vb));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> transpose(
    const Matrix3x3_T<F> &m) {
    Matrix3x3_T<F> ret = m;
    ret.transpose();
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> adjugate(
    const Matrix3x3_T<F> &m) {
    Matrix3x3_T<F> ret = m;
    ret.adjugate();
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> adjugateWithoutTranspose(
    const Matrix3x3_T<F> &m) {
    Matrix3x3_T<F> ret = m;
    ret.adjugateWithoutTranspose();
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> invert(
    const Matrix3x3_T<F> &m) {
    Matrix3x3_T<F> ret = m;
    ret.invert();
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> scale2D_3x3(
    const Vector2D_T<F> &s) {
    return Matrix3x3_T<F>(
        Vector3D_T<F, false>(s.x, 0, 0),
        Vector3D_T<F, false>(0, s.y, 0),
        Vector3D_T<F, false>(0, 0, 1.0f));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> scale2D_3x3(
    const F sx, const F sy) {
    return scale2D_3x3(Vector2D_T<F>(sx, sy));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> scale2D_3x3(
    const F s) {
    return scale2D_3x3(Vector2D_T<F>(s, s));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> rotate2D_3x3(
    const F angle) {
    const F s = std::sin(angle);
    const F c = std::cos(angle);

    Matrix3x3_T<F> ret;
    ret.m00 = c; ret.m01 = -s;
    ret.m10 = s; ret.m11 = c;

    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> translate2D_3x3(
    const Vector2D_T<F> &t) {
    return Matrix3x3_T<F>(
        Vector3D_T<F, false>(1, 0, 0),
        Vector3D_T<F, false>(0, 1, 0),
        Vector3D_T<F, false>(t, 1.0f));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> translate2D_3x3(
    const Point2D_T<F> &t) {
    return translate2D_3x3(static_cast<Vector2D_T<F>>(t));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> translate2D_3x3(
    const F tx, const F ty) {
    return translate2D_3x3(Vector2D_T<F>(tx, ty));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> scale3D_3x3(
    const Vector3D_T<F, false> &s) {
    return Matrix3x3_T<F>(
        Vector3D_T<F, false>(s.x, 0, 0),
        Vector3D_T<F, false>(0, s.y, 0),
        Vector3D_T<F, false>(0, 0, s.z));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> scale3D_3x3(
    const F sx, const F sy, const F sz) {
    return scale3D_3x3(Vector3D_T<F, false>(sx, sy, sz));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> scale3D_3x3(
    const F s) {
    return scale3D_3x3(Vector3D_T<F, false>(s, s, s));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> rotate3D_3x3(
    const F angle, const Vector3D_T<F, false> &axis) {
    const Vector3D_T<F, false> nAxis = normalize(axis);
    const F s = std::sin(angle);
    const F c = std::cos(angle);
    const F oneMinusC = 1 - c;

    Matrix3x3_T<F> ret;
    ret.m00 = nAxis.x * nAxis.x * oneMinusC + c;
    ret.m10 = nAxis.x * nAxis.y * oneMinusC + nAxis.z * s;
    ret.m20 = nAxis.z * nAxis.x * oneMinusC - nAxis.y * s;
    ret.m01 = nAxis.x * nAxis.y * oneMinusC - nAxis.z * s;
    ret.m11 = nAxis.y * nAxis.y * oneMinusC + c;
    ret.m21 = nAxis.y * nAxis.z * oneMinusC + nAxis.x * s;
    ret.m02 = nAxis.z * nAxis.x * oneMinusC + nAxis.y * s;
    ret.m12 = nAxis.y * nAxis.z * oneMinusC - nAxis.x * s;
    ret.m22 = nAxis.z * nAxis.z * oneMinusC + c;

    return ret;
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> rotate3D_3x3(
    const F angle, const F ax, const F ay, const F az) {
    return rotate3D_3x3(angle, Vector3D_T<F, false>(ax, ay, az));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> rotate3DX_3x3(
    const F angle) {
    return rotate3D_3x3(angle, Vector3D_T<F, false>(1, 0, 0));
}
template <std::floating_point F>
const CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> rotate3DY_3x3(
    const F angle) {
    return rotate3D_3x3(angle, Vector3D_T<F, false>(0, 1, 0));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> rotate3DZ_3x3(
    const F angle) {
    return rotate3D_3x3(angle, Vector3D_T<F, false>(0, 0, 1));
}



template <std::floating_point F>
struct Matrix4x4_T {
    union {
        struct { F m00, m10, m20, m30; };
        Vector4D_T<F> c0;
    };
    union {
        struct { F m01, m11, m21, m31; };
        Vector4D_T<F> c1;
    };
    union {
        struct { F m02, m12, m22, m32; };
        Vector4D_T<F> c2;
    };
    union {
        struct { F m03, m13, m23, m33; };
        Vector4D_T<F> c3;
    };

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T() :
        c0(1, 0, 0, 0),
        c1(0, 1, 0, 0),
        c2(0, 0, 1, 0),
        c3(0, 0, 0, 1) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T(const F array[16]) :
        m00(array[0]), m10(array[1]), m20(array[2]), m30(array[3]),
        m01(array[4]), m11(array[5]), m21(array[6]), m31(array[7]),
        m02(array[8]), m12(array[9]), m22(array[10]), m32(array[11]),
        m03(array[12]), m13(array[13]), m23(array[14]), m33(array[15]) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T(
        const Vector4D_T<F> &col0,
        const Vector4D_T<F> &col1,
        const Vector4D_T<F> &col2,
        const Vector4D_T<F> &col3) :
        c0(col0), c1(col1), c2(col2), c3(col3)
    {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T(
        const Matrix3x3_T<F> &mat3x3) :
        c0(Vector4D_T<F>(mat3x3.c0)),
        c1(Vector4D_T<F>(mat3x3.c1)),
        c2(Vector4D_T<F>(mat3x3.c2)),
        c3(0.0f, 0.0f, 0.0f, 1.0f)
    {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T(
        const Matrix3x3_T<F> &mat3x3, const Vector3D_T<F, false> &position) :
        c0(Vector4D_T<F>(mat3x3.c0)),
        c1(Vector4D_T<F>(mat3x3.c1)),
        c2(Vector4D_T<F>(mat3x3.c2)),
        c3(Vector4D_T<F>(position, 1.0f))
    {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T(
        const Matrix3x3_T<F> &mat3x3, const Point3D_T<F> &position) :
        c0(Vector4D_T<F>(mat3x3.c0)),
        c1(Vector4D_T<F>(mat3x3.c1)),
        c2(Vector4D_T<F>(mat3x3.c2)),
        c3(Vector4D_T<F>(position))
    {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T(const Matrix4x4_T<F2> &v) :
        c0(v.c0), c1(v.c1), c2(v.c2), c3(v.c3) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
        CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Matrix4x4_T(const Matrix4x4_T<F2> &v) :
        c0(static_cast<Vector4D_T<F>>(v.c0)),
        c1(static_cast<Vector4D_T<F>>(v.c1)),
        c2(static_cast<Vector4D_T<F>>(v.c2)),
        c3(static_cast<Vector4D_T<F>>(v.c3)) {}

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T<F> &operator[](const I idx) {
        Assert(static_cast<uint32_t>(idx) < 4, "idx is out of bound.");
        return *(&c0 + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T<F> operator[](const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 4, "idx is out of bound.");
        return *(&c0 + idx);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T operator-() const {
        return Matrix4x4_T(-c0, -c1, -c2, -c3);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T &operator+=(const Matrix4x4_T &r) {
        c0 += r.c0;
        c1 += r.c1;
        c2 += r.c2;
        c3 += r.c3;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T &operator-=(const Matrix4x4_T &r) {
        c0 -= r.c0;
        c1 -= r.c1;
        c2 -= r.c2;
        c3 -= r.c3;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T &operator*=(const F r) {
        c0 *= r;
        c1 *= r;
        c2 *= r;
        c3 *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T &operator*=(const Matrix4x4_T &r) {
        const Vector4D_T<F> rs[] = { row(0), row(1), row(2), row(3) };
        c0 = Vector4D_T<F>(dot(rs[0], r.c0), dot(rs[1], r.c0), dot(rs[2], r.c0), dot(rs[3], r.c0));
        c1 = Vector4D_T<F>(dot(rs[0], r.c1), dot(rs[1], r.c1), dot(rs[2], r.c1), dot(rs[3], r.c1));
        c2 = Vector4D_T<F>(dot(rs[0], r.c2), dot(rs[1], r.c2), dot(rs[2], r.c2), dot(rs[3], r.c2));
        c3 = Vector4D_T<F>(dot(rs[0], r.c3), dot(rs[1], r.c3), dot(rs[2], r.c3), dot(rs[3], r.c3));
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T &operator/=(const F r) {
        const F rr = 1 / r;
        c0 *= rr;
        c1 *= rr;
        c2 *= rr;
        c3 *= rr;
        return *this;
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T<F> row(const I idx) const {
        Assert(static_cast<uint32_t>(idx) < 4, "idx is out of bound.");
        switch (idx) {
        case 0:
            return Vector4D_T<F>(m00, m01, m02, m03);
        case 1:
            return Vector4D_T<F>(m10, m11, m12, m13);
        case 2:
            return Vector4D_T<F>(m20, m21, m22, m23);
        case 3:
            return Vector4D_T<F>(m30, m31, m32, m33);
        default:
            return Vector4D_T<F>(0, 0, 0, 0);
        }
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T &transpose() {
        F temp;
        temp = m10; m10 = m01; m01 = temp;
        temp = m20; m20 = m02; m02 = temp;
        temp = m21; m21 = m12; m12 = temp;
        temp = m30; m30 = m03; m03 = temp;
        temp = m31; m31 = m13; m13 = temp;
        temp = m32; m32 = m23; m23 = temp;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T &invert() {
        F inv[] = {
            +((m11 * m22 * m33) - (m31 * m22 * m13) + (m21 * m32 * m13) - (m11 * m32 * m23) + (m31 * m12 * m23) - (m21 * m12 * m33)),
            -((m10 * m22 * m33) - (m30 * m22 * m13) + (m20 * m32 * m13) - (m10 * m32 * m23) + (m30 * m12 * m23) - (m20 * m12 * m33)),
            +((m10 * m21 * m33) - (m30 * m21 * m13) + (m20 * m31 * m13) - (m10 * m31 * m23) + (m30 * m11 * m23) - (m20 * m11 * m33)),
            -((m10 * m21 * m32) - (m30 * m21 * m12) + (m20 * m31 * m12) - (m10 * m31 * m22) + (m30 * m11 * m22) - (m20 * m11 * m32)),

            -((m01 * m22 * m33) - (m31 * m22 * m03) + (m21 * m32 * m03) - (m01 * m32 * m23) + (m31 * m02 * m23) - (m21 * m02 * m33)),
            +((m00 * m22 * m33) - (m30 * m22 * m03) + (m20 * m32 * m03) - (m00 * m32 * m23) + (m30 * m02 * m23) - (m20 * m02 * m33)),
            -((m00 * m21 * m33) - (m30 * m21 * m03) + (m20 * m31 * m03) - (m00 * m31 * m23) + (m30 * m01 * m23) - (m20 * m01 * m33)),
            +((m00 * m21 * m32) - (m30 * m21 * m02) + (m20 * m31 * m02) - (m00 * m31 * m22) + (m30 * m01 * m22) - (m20 * m01 * m32)),

            +((m01 * m12 * m33) - (m31 * m12 * m03) + (m11 * m32 * m03) - (m01 * m32 * m13) + (m31 * m02 * m13) - (m11 * m02 * m33)),
            -((m00 * m12 * m33) - (m30 * m12 * m03) + (m10 * m32 * m03) - (m00 * m32 * m13) + (m30 * m02 * m13) - (m10 * m02 * m33)),
            +((m00 * m11 * m33) - (m30 * m11 * m03) + (m10 * m31 * m03) - (m00 * m31 * m13) + (m30 * m01 * m13) - (m10 * m01 * m33)),
            -((m00 * m11 * m32) - (m30 * m11 * m02) + (m10 * m31 * m02) - (m00 * m31 * m12) + (m30 * m01 * m12) - (m10 * m01 * m32)),

            -((m01 * m12 * m23) - (m21 * m12 * m03) + (m11 * m22 * m03) - (m01 * m22 * m13) + (m21 * m02 * m13) - (m11 * m02 * m23)),
            +((m00 * m12 * m23) - (m20 * m12 * m03) + (m10 * m22 * m03) - (m00 * m22 * m13) + (m20 * m02 * m13) - (m10 * m02 * m23)),
            -((m00 * m11 * m23) - (m20 * m11 * m03) + (m10 * m21 * m03) - (m00 * m21 * m13) + (m20 * m01 * m13) - (m10 * m01 * m23)),
            +((m00 * m11 * m22) - (m20 * m11 * m02) + (m10 * m21 * m02) - (m00 * m21 * m12) + (m20 * m01 * m12) - (m10 * m01 * m22)),
        };

        const F recDet = 1.0f / (m00 * inv[0] + m10 * inv[4] + m20 * inv[8] + m30 * inv[12]);
        for (int i = 0; i < 16; ++i)
            inv[i] *= recDet;
        *this = Matrix4x4_T(inv);

        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> getUpperLeftMatrix() const {
        return Matrix3x3_T<F>(
            Vector3D_T<F, false>(c0),
            Vector3D_T<F, false>(c1),
            Vector3D_T<F, false>(c2));
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void decompose(
        Vector3D_T<F, false>* const retScale,
        Vector3D_T<F, false>* const rotation,
        Vector3D_T<F, false>* const translation) const {
        Matrix4x4_T mat = *this;

        // JP: 移動成分
        // EN: Translation component
        if (translation)
            *translation = Vector3D_T<F, false>(mat.c3);

        const Vector3D_T<F, false> scale(
            length(Vector3D_T<F, false>(mat.c0)),
            length(Vector3D_T<F, false>(mat.c1)),
            length(Vector3D_T<F, false>(mat.c2)));

        // JP: 拡大縮小成分
        // EN: Scale component
        if (retScale)
            *retScale = scale;

        if (!rotation)
            return;

        // JP: 上記成分を排除
        // EN: Remove the above components
        mat.c3 = Vector4D_T<F>(0, 0, 0, 1);
        if (std::fabs(scale.x) > 0)
            mat.c0 /= scale.x;
        if (std::fabs(scale.y) > 0)
            mat.c1 /= scale.y;
        if (std::fabs(scale.z) > 0)
            mat.c2 /= scale.z;

        // JP: 回転成分がXYZの順で作られている、つまりZYXp(pは何らかのベクトル)と仮定すると、
        //     行列は以下の形式をとっていると考えられる。
        //     A, B, GはそれぞれX, Y, Z軸に対する回転角度。cとsはcosとsin。
        //     cG * cB   -sG * cA + cG * sB * sA    sG * sA + cG * sB * cA
        //     sG * cB    cG * cA + sG * sB * sA   -cG * sA + sG * sB * cA
        //       -sB             cB * sA                   cB * cA
        //     したがって、3行1列成分からまずY軸に対する回転Bが求まる。
        //     次に求めたBを使って回転A, Gが求まる。数値精度を考慮すると、cBが0の場合は別の処理が必要。
        //     cBが0の場合はsBは+-1(Bが90度ならば+、-90度ならば-)なので、上の行列は以下のようになる。
        //      0   -sG * cA +- cG * sA    sG * sA +- cG * cA
        //      0    cG * cA +- sG * sA   -cG * sA +- sG * cA
        //     -+1           0                     0
        //     求めたBを使ってさらに求まる成分がないため、Aを0と仮定する。
        // EN: 
        rotation->y = std::asin(-mat.c0.z);
        const F cosBeta = std::cos(rotation->y);

        if (std::fabs(cosBeta) < 0.000001f) {
            rotation->x = 0;
            rotation->z = std::atan2(-mat.c1.x, mat.c1.y);
        }
        else {
            rotation->x = std::atan2(mat.c1.z, mat.c2.z);
            rotation->z = std::atan2(mat.c0.y, mat.c0.x);
        }
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool allFinite() const {
        return c0.allFinite() && c1.allFinite() && c2.allFinite() && c3.allFinite();
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> operator+(
    const Matrix4x4_T<F> &a, const Matrix4x4_T<F> &b) {
    Matrix4x4_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> operator-(
    const Matrix4x4_T<F> &a, const Matrix4x4_T<F> &b) {
    Matrix4x4_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> operator*(
    const Matrix4x4_T<F> &a, const N b) {
    Matrix4x4_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> operator*(
    const N a, const Matrix4x4_T<F> &b) {
    Matrix4x4_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> operator*(
    const Matrix4x4_T<F> &a, const Matrix4x4_T<F> &b) {
    Matrix4x4_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> operator/(
    const Matrix4x4_T<F> &a, const N b) {
    Matrix4x4_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, false> operator*(
    const Matrix4x4_T<F> &a, const Vector3D_T<F, false> &b) {
    const Vector4D_T<F> r[] = { a.row(0), a.row(1), a.row(2) };
    const Vector4D_T<F> v4(b, 0.0f);
    return Vector3D_T<F, false>(
        dot(r[0], v4),
        dot(r[1], v4),
        dot(r[2], v4));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> operator*(
    const Matrix4x4_T<F> &a, const Point3D_T<F> &b) {
    const Vector4D_T<F> r[] = { a.row(0), a.row(1), a.row(2) };
    const Vector4D_T<F> v4(b, 1.0f);
    return Point3D_T<F>(
        dot(r[0], v4),
        dot(r[1], v4),
        dot(r[2], v4));
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4D_T<F> operator*(
    const Matrix4x4_T<F> &a, const Vector4D_T<F> &b) {
    return Vector4D_T<F>(
        dot(a.row(0), b),
        dot(a.row(1), b),
        dot(a.row(2), b),
        dot(a.row(3), b));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> transpose(
    const Matrix4x4_T<F> &mat) {
    Matrix4x4_T<F> ret = mat;
    return ret.transpose();
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> invert(
    const Matrix4x4_T<F> &mat) {
    Matrix4x4_T<F> ret = mat;
    return ret.invert();
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> scale3D_4x4(
    const Vector3D_T<F, false> &s) {
    return Matrix4x4_T<F>(
        Vector4D_T<F>(s.x, 0, 0, 0),
        Vector4D_T<F>(0, s.y, 0, 0),
        Vector4D_T<F>(0, 0, s.z, 0),
        Vector4D_T<F>(0, 0, 0, 1));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> scale3D_4x4(
    const F sx, const F sy, const F sz) {
    return scale3D_4x4(Vector3D_T<F, false>(sx, sy, sz));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> scale3D_4x4(
    const F s) {
    return scale3D_4x4(Vector3D_T<F, false>(s, s, s));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> rotate3D_4x4(
    const F angle, const Vector3D_T<F, false> &axis) {
    const Vector3D_T<F, false> nAxis = normalize(axis);
    const F s = std::sin(angle);
    const F c = std::cos(angle);
    const F oneMinusC = 1 - c;

    Matrix4x4_T<F> matrix;
    matrix.m00 = nAxis.x * nAxis.x * oneMinusC + c;
    matrix.m10 = nAxis.x * nAxis.y * oneMinusC + nAxis.z * s;
    matrix.m20 = nAxis.z * nAxis.x * oneMinusC - nAxis.y * s;
    matrix.m30 = 0.0f;
    matrix.m01 = nAxis.x * nAxis.y * oneMinusC - nAxis.z * s;
    matrix.m11 = nAxis.y * nAxis.y * oneMinusC + c;
    matrix.m21 = nAxis.y * nAxis.z * oneMinusC + nAxis.x * s;
    matrix.m31 = 0.0f;
    matrix.m02 = nAxis.z * nAxis.x * oneMinusC + nAxis.y * s;
    matrix.m12 = nAxis.y * nAxis.z * oneMinusC - nAxis.x * s;
    matrix.m22 = nAxis.z * nAxis.z * oneMinusC + c;
    matrix.m32 = 0.0f;
    matrix.m03 = 0.0f;
    matrix.m13 = 0.0f;
    matrix.m23 = 0.0f;
    matrix.m33 = 1.0f;

    return matrix;
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> rotate3D_4x4(
    const F angle, const F ax, const F ay, const F az) {
    return rotate3D_4x4(angle, Vector3D_T<F, false>(ax, ay, az));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> rotate3DX_4x4(
    const F angle) {
    return rotate3D_4x4(angle, Vector3D_T<F, false>(1, 0, 0));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> rotate3DY_4x4(
    const F angle) {
    return rotate3D_4x4(angle, Vector3D_T<F, false>(0, 1, 0));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> rotate3DZ_4x4(
    const F angle) {
    return rotate3D_4x4(angle, Vector3D_T<F, false>(0, 0, 1));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> translate3D_4x4(
    const Vector3D_T<F, false> &t) {
    return Matrix4x4_T<F>(
        Vector4D_T<F>(1, 0, 0, 0),
        Vector4D_T<F>(0, 1, 0, 0),
        Vector4D_T<F>(0, 0, 1, 0),
        Vector4D_T<F>(t, 1.0f));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> translate3D_4x4(
    const Point3D_T<F> &t) {
    return translate3D_4x4(static_cast<Vector3D_T<F, false>>(t));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> translate3D_4x4(
    const F tx, const F ty, const F tz) {
    return translate3D_4x4(Vector3D_T<F, false>(tx, ty, tz));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4_T<F> camera(
    const F aspect, const F fovY, const F near, const F far) {
    const F f = 1 / std::tan(fovY / 2);
    const F dz = far - near;

    Matrix4x4_T<F> matrix;
    matrix.m00 = f / aspect;
    matrix.m11 = f;
    matrix.m22 = -(near + far) / dz;
    matrix.m32 = -1;
    matrix.m23 = -2 * far * near / dz;
    matrix.m10 = matrix.m20 = matrix.m30 =
        matrix.m01 = matrix.m21 = matrix.m31 =
        matrix.m02 = matrix.m12 =
        matrix.m03 = matrix.m13 = matrix.m33 = 0;

    return matrix;
}



template <std::floating_point F>
struct Quaternion_T {
    union {
        Vector3D_T<F, false> v;
        struct {
            F x;
            F y;
            F z;
        };
    };
    F w;

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T() :
        v(0), w(1) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T(const F xx, const F yy, const F zz, const F ww) :
        v(xx, yy, zz), w(ww) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T(const Vector3D_T<F, false> &vv, const F ww) :
        v(vv), w(ww) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T(const Quaternion_T<F2> &v) :
        x(v.x), y(v.y), z(v.z), w(v.w) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
        CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr Quaternion_T(const Quaternion_T<F2> &v) :
        x(static_cast<F>(v.x)), y(static_cast<F>(v.y)), z(static_cast<F>(v.z)), w(static_cast<F>(v.w)) {}

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T operator-() const {
        return Quaternion_T(-v, -w);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T &operator+=(const Quaternion_T &r) {
        v += r.v;
        w += r.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T &operator-=(const Quaternion_T &r) {
        v -= r.v;
        w -= r.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T &operator*=(const F r) {
        v *= r;
        w *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T &operator*=(const Quaternion_T &r) {
        const Vector3D_T<F, false> vv = v;
        v = cross(v, r.v) + w * r.v + r.w * v;
        w = w * r.w - dot(vv, r.v);
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T &operator/=(const F r) {
        const F rr = 1 / r;
        v *= rr;
        w *= rr;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void toEulerAngles(
        F* const roll, F* const pitch, F* const yaw) const {
        const F xx = x * x;
        const F xy = x * y;
        const F xz = x * z;
        const F xw = x * w;
        const F yy = y * y;
        const F yz = y * z;
        const F yw = y * w;
        const F zz = z * z;
        const F zw = z * w;
        const F ww = w * w;
        *pitch = std::atan2(2 * (xw + yz), ww - xx - yy + zz); // around x
        *yaw = std::asin(std::fmin(std::fmax(2.0f * (yw - xz), -1.0f), 1.0f)); // around y
        *roll = std::atan2(2 * (zw + xy), ww + xx - yy - zz); // around z
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3_T<F> toMatrix3x3() const {
        const F xx = x * x, yy = y * y, zz = z * z;
        const F xy = x * y, yz = y * z, zx = z * x;
        const F xw = x * w, yw = y * w, zw = z * w;
        return Matrix3x3_T<F>(
            Vector3D_T<F, false>(1 - 2 * (yy + zz), 2 * (xy + zw), 2 * (zx - yw)),
            Vector3D_T<F, false>(2 * (xy - zw), 1 - 2 * (xx + zz), 2 * (yz + xw)),
            Vector3D_T<F, false>(2 * (zx + yw), 2 * (yz - xw), 1 - 2 * (xx + yy)));
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool allFinite() const {
        return v.allFinite() && stc::isfinite(w);
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool4D operator==(
    const Quaternion_T<F> &a, const Quaternion_T<F> &b) {
    return Bool4D(a.v == b.v, a.w == b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool4D operator!=(
    const Quaternion_T<F> &a, const Quaternion_T<F> &b) {
    return Bool4D(a.v != b.v, a.w != b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> operator+(
    const Quaternion_T<F> &a, const Quaternion_T<F> &b) {
    Quaternion_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> operator-(
    const Quaternion_T<F> &a, const Quaternion_T<F> &b) {
    Quaternion_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> operator*(
    const Quaternion_T<F> &a, const N b) {
    Quaternion_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> operator*(
    const N a, const Quaternion_T<F> &b) {
    Quaternion_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> operator*(
    const Quaternion_T<F> &a, const Quaternion_T<F> &b) {
    Quaternion_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> operator/(
    const Quaternion_T<F> &a, const N b) {
    Quaternion_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE bool allFinite(
    const Quaternion_T<F> &q) {
    return q.allFinite();
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F dot(
    const Quaternion_T<F> &q0, const Quaternion_T<F> &q1) {
    return dot(q0.v, q1.v) + q0.w * q1.w;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> normalize(
    const Quaternion_T<F> &q) {
    return q / std::sqrt(dot(q, q));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> qRotate(
    const F angle, const Vector3D_T<F, false> &axis) {
    const F ha = angle / 2;
    const F s = std::sin(ha), c = std::cos(ha);
    return Quaternion_T<F>(s * normalize(axis), c);
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> qRotate(
    const F angle, const F ax, const F ay, const F az) {
    return qRotate(angle, Vector3D_T<F, false>(ax, ay, az));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> qRotateX(
    const F angle) {
    return qRotate(angle, Vector3D_T<F, false>(1, 0, 0));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> qRotateY(
    const F angle) {
    return qRotate(angle, Vector3D_T<F, false>(0, 1, 0));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> qRotateZ(
    const F angle) {
    return qRotate(angle, Vector3D_T<F, false>(0, 0, 1));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> qFromEulerAngles(
    const F roll, const F pitch, const F yaw) {
    return qRotateZ(roll) * qRotateY(yaw) * qRotateX(pitch);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Quaternion_T<F> Slerp(
    const F t, const Quaternion_T<F> &q0, const Quaternion_T<F> &q1) {
    const F cosTheta = dot(q0, q1);
    if (cosTheta > 0.9995f)
        return normalize((1 - t) * q0 + t * q1);
    else {
        const F theta = std::acos(std::fmin(std::fmax(cosTheta, -1.0f), 1.0f));
        const F thetap = theta * t;
        const Quaternion_T<F> qPerp = normalize(q1 - q0 * cosTheta);
        const F sinThetaP = std::sin(thetap);
        const F cosThetaP = std::cos(thetap);
        //sincos(thetap, &sinThetaP, &cosThetaP);
        return q0 * cosThetaP + qPerp * sinThetaP;
    }
}



template <std::floating_point F>
struct RGB_T {
    F r, g, b;

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T() :
        r(0.0f), g(0.0f), b(0.0f) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr RGB_T(const F v) :
        r(v), g(v), b(v) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T(const F rr, const F gg, const F bb) :
        r(rr), g(gg), b(bb) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr RGB_T(const float3 &v) :
        r(v.x), g(v.y), b(v.z) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T(const RGB_T<F2> &v) :
        r(v.r), g(v.g), b(v.b) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr RGB_T(const RGB_T<F2> &v) :
        r(static_cast<F>(v.r)), g(static_cast<F>(v.g)), b(static_cast<F>(v.b)) {}

    CUDA_COMMON_FUNCTION CUDA_INLINE explicit /*constexpr*/ operator float3() const {
        return make_float3(r, g, b);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ float3 toNative() const {
        return make_float3(r, g, b);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T operator-() const {
        return RGB_T(-r, -g, -b);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T &operator+=(const RGB_T &o) {
        r += o.r;
        g += o.g;
        b += o.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T &operator-=(const RGB_T &o) {
        r -= o.r;
        g -= o.g;
        b -= o.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T &operator*=(const F o) {
        r *= o;
        g *= o;
        b *= o;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T &operator*=(const RGB_T &o) {
        r *= o.r;
        g *= o.g;
        b *= o.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T &operator/=(const F o) {
        const F ro = 1 / o;
        r *= ro;
        g *= ro;
        b *= ro;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T &operator/=(const RGB_T &o) {
        r /= o.r;
        g /= o.g;
        b /= o.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T &safeDivide(const F o) {
        const F ro = 1 / o;
        const bool c = o != 0;
        r = c ? r * ro : 0.0f;
        g = c ? g * ro : 0.0f;
        b = c ? b * ro : 0.0f;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T &safeDivide(const RGB_T &o) {
        r = o.r != 0 ? r / o.r : 0.0f;
        g = o.g != 0 ? g / o.g : 0.0f;
        b = o.b != 0 ? b / o.b : 0.0f;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool allFinite() const {
        return stc::isfinite(r) && stc::isfinite(g) && stc::isfinite(b);
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator==(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return Bool3D(a.r == b.r, a.g == b.g, a.b == b.b);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator!=(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return Bool3D(a.r != b.r, a.g != b.g, a.b != b.b);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator<(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return Bool3D(a.r < b.r, a.g < b.g, a.b < b.b);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator<=(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return Bool3D(a.r <= b.r, a.g <= b.g, a.b <= b.b);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator>(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return Bool3D(a.r > b.r, a.g > b.g, a.b > b.b);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Bool3D operator>=(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return Bool3D(a.r >= b.r, a.g >= b.g, a.b >= b.b);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> operator+(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    RGB_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> operator-(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    RGB_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> operator*(
    const RGB_T<F> &a, const N b) {
    RGB_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> operator*(
    const N a, const RGB_T<F> &b) {
    RGB_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> operator*(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    RGB_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> operator/(
    const RGB_T<F> &a, const N b) {
    RGB_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> operator/(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    RGB_T<F> ret = a;
    ret /= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> safeDivide(
    const RGB_T<F> &a, const N b) {
    RGB_T<F> ret = a;
    ret.safeDivide(static_cast<F>(b));
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> safeDivide(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    RGB_T<F> ret = a;
    ret.safeDivide(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> abs(
    const RGB_T<F> &v) {
    return RGB_T<F>(std::fabs(v.r), std::fabs(v.g), std::fabs(v.b));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> min(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return RGB_T<F>(std::fmin(a.r, b.r), std::fmin(a.g, b.g), std::fmin(a.b, b.b));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> max(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return RGB_T<F>(std::fmax(a.r, b.r), std::fmax(a.g, b.g), std::fmax(a.b, b.b));
}



template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> HSVtoRGB(const F h, const F s, const F v) {
    if (s == 0)
        return RGB_T<F>(v, v, v);

    const F hm = h - std::floor(h);
    const int32_t hi = static_cast<int32_t>(hm * 6);
    const F f = hm * 6 - hi;
    const F m = v * (1 - s);
    const F n = v * (1 - s * f);
    const F k = v * (1 - s * (1 - f));
    if (hi == 0)
        return RGB_T<F>(v, k, m);
    else if (hi == 1)
        return RGB_T<F>(n, v, m);
    else if (hi == 2)
        return RGB_T<F>(m, v, k);
    else if (hi == 3)
        return RGB_T<F>(m, n, v);
    else if (hi == 4)
        return RGB_T<F>(k, m, v);
    else if (hi == 5)
        return RGB_T<F>(v, m, n);
    return RGB_T<F>(0, 0, 0);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F simpleToneMap_s(const F value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    return 1 - std::exp(-value);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F sRGB_degamma_s(const F value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    if (value <= 0.04045f)
        return value / 12.92f;
    return std::pow((value + 0.055f) / 1.055f, 2.4f);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F sRGB_gamma_s(const F value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    if (value <= 0.0031308f)
        return 12.92f * value;
    return 1.055f * std::pow(value, 1 / 2.4f) - 0.055f;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGB_T<F> sRGB_degamma(const RGB_T<F> &value) {
    return RGB_T<F>(
        sRGB_degamma_s(value.r),
        sRGB_degamma_s(value.g),
        sRGB_degamma_s(value.b));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F sRGB_calcLuminance(const RGB_T<F> &value) {
    return 0.2126729f * value.r + 0.7151522f * value.g + 0.0721750f * value.b;
}



template <std::floating_point F>
struct AABB_T {
    Point3D_T<F> minP;
    Point3D_T<F> maxP;

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr AABB_T() :
        minP(Point3D_T<F>(INFINITY)), maxP(Point3D_T<F>(-INFINITY)) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr AABB_T(
        const Point3D_T<F> &_minP, const Point3D_T<F> &_maxP) :
        minP(_minP), maxP(_maxP) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr AABB_T(const AABB_T<F2> &v) :
        minP(v.minP), maxP(v.maxP) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit constexpr AABB_T(const AABB_T<F2> &v) :
        minP(static_cast<Point3D_T<F>>(v.minP)), maxP(static_cast<Point3D_T<F>>(v.maxP)) {}

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr AABB_T &unify(const Point3D_T<F> &p) {
        minP = min(minP, p);
        maxP = max(maxP, p);
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr AABB_T &unify(const AABB_T &bb) {
        minP = min(minP, bb.minP);
        maxP = max(maxP, bb.maxP);
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr AABB_T &dilate(const F scale) {
        Vector3D_T<F, false> d = maxP - minP;
        minP -= 0.5f * (scale - 1) * d;
        maxP += 0.5f * (scale - 1) * d;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> normalize(const Point3D_T<F> &p) const {
        return static_cast<Point3D_T<F>>(safeDivide(p - minP, maxP - minP));
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool isValid() const {
        Vector3D_T<F, false> d = maxP - minP;
        return all(d >= Vector3D_T<F, false>(0.0f));
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool intersect(
        const Point3D_T<F> &org, const Vector3D_T<F, false> &dir, const F distMin, const F distMax) const {
        if (!isValid())
            return INFINITY;
        const Vector3D_T<F, false> invRayDir = 1.0f / dir;
        const Vector3D_T<F, false> tNear = (minP - org) * invRayDir;
        const Vector3D_T<F, false> tFar = (maxP - org) * invRayDir;
        const Vector3D_T<F, false> near = min(tNear, tFar);
        const Vector3D_T<F, false> far = max(tNear, tFar);
        F t0 = std::fmax(std::fmax(near.x, near.y), near.z);
        F t1 = std::fmin(std::fmin(far.x, far.y), far.z);
        t0 = std::fmax(t0, distMin);
        t1 = std::fmin(t1, distMax);
        return t0 <= t1 && t1 > 0.0f;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr F intersect(
        const Point3D_T<F> &org, const Vector3D_T<F, false> &dir,
        const F distMin, const F distMax,
        F* const u, F* const v, bool* const isFrontHit) const {
        if (!isValid())
            return INFINITY;
        const Vector3D_T<F, false> invRayDir = 1.0f / dir;
        const Vector3D_T<F, false> tNear = (minP - org) * invRayDir;
        const Vector3D_T<F, false> tFar = (maxP - org) * invRayDir;
        const Vector3D_T<F, false> near = min(tNear, tFar);
        const Vector3D_T<F, false> far = max(tNear, tFar);
        F t0 = std::fmax(std::fmax(near.x, near.y), near.z);
        F t1 = std::fmin(std::fmin(far.x, far.y), far.z);
        *isFrontHit = t0 >= 0.0f;
        t0 = std::fmax(t0, distMin);
        t1 = std::fmin(t1, distMax);
        if (!(t0 <= t1 && t1 > 0.0f))
            return INFINITY;

        const F t = *isFrontHit ? t0 : t1;
        Vector3D_T<F, false> n = -sign(dir) * step(near.yzx(), near) * step(near.zxy(), near);
        if (!*isFrontHit)
            n = -n;

        int32_t faceID = static_cast<int32_t>(dot(abs(n), Vector3D_T<F, false>(2, 4, 8)));
        faceID ^= static_cast<int32_t>(any(n > Vector3D_T<F, false>(0.0f)));

        const int32_t faceDim = tzcnt(faceID & ~0b1) - 1;
        const int32_t dim0 = (faceDim + 1) % 3;
        const int32_t dim1 = (faceDim + 2) % 3;
        const Point3D_T<F> p = org + t * dir;
        const F min0 = minP[dim0];
        const F max0 = maxP[dim0];
        const F min1 = minP[dim1];
        const F max1 = maxP[dim1];
        *u = std::fmin(std::fmax((p[dim0] - min0) / (max0 - min0), 0.0f), 1.0f)
            + static_cast<F>(faceID);
        *v = std::fmin(std::fmax((p[dim1] - min1) / (max1 - min1), 0.0f), 1.0f);

        return t;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3D_T<F> restoreHitPoint(
        F u, const F v, Vector3D_T<F, true>* const normal) const {
        const auto faceID = static_cast<uint32_t>(u);
        u = std::fmod(u, 1.0f);

        const int32_t faceDim = tzcnt(faceID & ~0b1) - 1;
        const bool isPosSide = faceID & 0b1;
        *normal = Vector3D_T<F, true>(0.0f);
        (*normal)[faceDim] = isPosSide ? 1 : -1;

        const int32_t dim0 = (faceDim + 1) % 3;
        const int32_t dim1 = (faceDim + 2) % 3;
        Point3D_T<F> p;
        p[faceDim] = isPosSide ? maxP[faceDim] : minP[faceDim];
        p[dim0] = lerp(minP[dim0], maxP[dim0], u);
        p[dim1] = lerp(minP[dim1], maxP[dim1], v);

        return p;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3D_T<F, true> restoreNormal(const F u, const F v) const {
        const auto faceID = static_cast<uint32_t>(u);
        const int32_t faceDim = tzcnt(faceID & ~0b1) - 1;
        const bool isPosSide = faceID & 0b1;
        auto normal = Vector3D_T<F, true>(0.0f);
        normal[faceDim] = isPosSide ? 1 : -1;
        return normal;
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr AABB_T<F> operator*(
    const Matrix4x4_T<F> &mat, const AABB_T<F> &aabb) {
    AABB_T<F> ret;
    ret
        .unify(mat * Point3D_T<F>(aabb.minP.x, aabb.minP.y, aabb.minP.z))
        .unify(mat * Point3D_T<F>(aabb.maxP.x, aabb.minP.y, aabb.minP.z))
        .unify(mat * Point3D_T<F>(aabb.minP.x, aabb.maxP.y, aabb.minP.z))
        .unify(mat * Point3D_T<F>(aabb.maxP.x, aabb.maxP.y, aabb.minP.z))
        .unify(mat * Point3D_T<F>(aabb.minP.x, aabb.minP.y, aabb.maxP.z))
        .unify(mat * Point3D_T<F>(aabb.maxP.x, aabb.minP.y, aabb.maxP.z))
        .unify(mat * Point3D_T<F>(aabb.minP.x, aabb.maxP.y, aabb.maxP.z))
        .unify(mat * Point3D_T<F>(aabb.maxP.x, aabb.maxP.y, aabb.maxP.z));
    return ret;
}



template <typename RealType>
struct CompensatedSum_T {
    RealType result;
    RealType comp;

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr CompensatedSum_T(const RealType &value = RealType(0)) :
        result(value), comp(0.0) {};

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr CompensatedSum_T &operator=(const RealType &value) {
        result = value;
        comp = 0;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr CompensatedSum_T &operator+=(const RealType &value) {
        const RealType cInput = value - comp;
        const RealType sumTemp = result + cInput;
        comp = (sumTemp - result) - cInput;
        result = sumTemp;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr operator RealType() const {
        return result;
    };
};



using Vector2D = Vector2D_T<float>;
using Point2D = Point2D_T<float>;
using Vector3D = Vector3D_T<float, false>;
using Normal3D = Vector3D_T<float, true>;
using Point3D = Point3D_T<float>;
using Vector4D = Vector4D_T<float>;
using Matrix2x2 = Matrix2x2_T<float>;
using Matrix3x2 = Matrix3x2_T<float>;
using Matrix3x3 = Matrix3x3_T<float>;
using Matrix4x4 = Matrix4x4_T<float>;
using Quaternion = Quaternion_T<float>;
using RGB = RGB_T<float>;
using AABB = AABB_T<float>;
using FloatSum = CompensatedSum_T<float>;



CUDA_COMMON_FUNCTION CUDA_INLINE int32_t floatToOrderedInt(const float fVal) {
#if defined(__CUDA_ARCH__)
    int32_t iVal = __float_as_int(fVal);
#else
    const int32_t iVal = *reinterpret_cast<const int32_t*>(&fVal);
#endif
    return (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
}

CUDA_COMMON_FUNCTION CUDA_INLINE float orderedIntToFloat(const int32_t iVal) {
    int32_t orgVal = (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
#if defined(__CUDA_ARCH__)
    return __int_as_float(orgVal);
#else
    return *reinterpret_cast<float*>(&orgVal);
#endif
}

struct RGBAsOrderedInt {
    int32_t r, g, b;

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBAsOrderedInt() :
        r(0), g(0), b(0) {
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ RGBAsOrderedInt(const RGB &v) :
        r(floatToOrderedInt(v.r)), g(floatToOrderedInt(v.g)), b(floatToOrderedInt(v.b)) {
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBAsOrderedInt &operator=(
        const RGBAsOrderedInt &v) {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBAsOrderedInt &operator=(
        const volatile RGBAsOrderedInt &v) {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE volatile constexpr RGBAsOrderedInt &operator=(
        const RGBAsOrderedInt &v) volatile {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE volatile constexpr RGBAsOrderedInt &operator=(
        const volatile RGBAsOrderedInt &v) volatile {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE explicit /*constexpr*/ operator RGB() const {
        return RGB(orderedIntToFloat(r), orderedIntToFloat(g), orderedIntToFloat(b));
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE explicit /*constexpr*/ operator RGB() const volatile {
        return RGB(orderedIntToFloat(r), orderedIntToFloat(g), orderedIntToFloat(b));
    }
};

#if defined(__CUDA_ARCH__) || defined(__INTELLISENSE__)
#   if __CUDA_ARCH__ < 600
#       define atomicOr_block atomicOr
#       define atomicAnd_block atomicAnd
#       define atomicAdd_block atomicAdd
#       define atomicMin_block atomicMin
#       define atomicMax_block atomicMax
#   endif

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMin_RGB(
    RGBAsOrderedInt* const dst, const RGBAsOrderedInt &v) {
    atomicMin(&dst->r, v.r);
    atomicMin(&dst->g, v.g);
    atomicMin(&dst->b, v.b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMax_RGB(
    RGBAsOrderedInt* const dst, const RGBAsOrderedInt &v) {
    atomicMax(&dst->r, v.r);
    atomicMax(&dst->g, v.g);
    atomicMax(&dst->b, v.b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicAdd_RGB(RGB* dst, const RGB &v) {
    atomicAdd(&dst->r, v.r);
    atomicAdd(&dst->g, v.g);
    atomicAdd(&dst->b, v.b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMin_RGB_block(
    RGBAsOrderedInt* const dst, const RGBAsOrderedInt &v) {
    atomicMin_block(&dst->r, v.r);
    atomicMin_block(&dst->g, v.g);
    atomicMin_block(&dst->b, v.b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMax_RGB_block(
    RGBAsOrderedInt* const dst, const RGBAsOrderedInt &v) {
    atomicMax_block(&dst->r, v.r);
    atomicMax_block(&dst->g, v.g);
    atomicMax_block(&dst->b, v.b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicAdd_RGB_block(RGB* const dst, const RGB &v) {
    atomicAdd_block(&dst->r, v.r);
    atomicAdd_block(&dst->g, v.g);
    atomicAdd_block(&dst->b, v.b);
}
#endif