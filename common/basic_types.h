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
#   include <cstdint>
#   include <cmath>

#   include <algorithm>

#   include <immintrin.h>
#endif



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



template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow2(T x) {
    return x * x;
}

template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow3(T x) {
    return x * pow2(x);
}

template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow4(T x) {
    return pow2(pow2(x));
}

template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow5(T x) {
    return x * pow4(x);
}

template <typename T, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T lerp(const T &v0, const T &v1, F t) {
    return (1 - t) * v0 + t * v1;
}



template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE T alignUp(T value, uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t tzcnt(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __clz(__brev(x));
#else
    return _tzcnt_u32(x);
#endif
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t lzcnt(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __clz(x);
#else
    return _lzcnt_u32(x);
#endif
}

CUDA_COMMON_FUNCTION CUDA_INLINE int32_t popcnt(uint32_t x) {
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
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t prevPowOf2Exponent(uint32_t x) {
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
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t nextPowOf2Exponent(uint32_t x) {
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
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t prevPowerOf2(uint32_t x) {
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
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t nextPowerOf2(uint32_t x) {
    if (x == 0)
        return 0;
    return 1 << nextPowOf2Exponent(x);
}

template <typename IntType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr IntType nextMultiplesForPowOf2(IntType x, uint32_t exponent) {
    IntType mask = (1 << exponent) - 1;
    return (x + mask) & ~mask;
}

template <typename IntType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr IntType nextMultiplierForPowOf2(IntType x, uint32_t exponent) {
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

        uint32_t mask = (1 << width) - 1;
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
    constexpr int2(int32_t v = 0) : x(v), y(v) {}
    constexpr int2(int32_t xx, int32_t yy) : x(xx), y(yy) {}
};
inline constexpr int2 make_int2(int32_t x, int32_t y) {
    return int2(x, y);
}
struct int3 {
    int32_t x, y, z;
    constexpr int3(int32_t v = 0) : x(v), y(v), z(v) {}
    constexpr int3(int32_t xx, int32_t yy, int32_t zz) : x(xx), y(yy), z(zz) {}
};
inline constexpr int3 make_int3(int32_t x, int32_t y, int32_t z) {
    return int3(x, y, z);
}
struct alignas(16) int4 {
    int32_t x, y, z, w;
    constexpr int4(int32_t v = 0) : x(v), y(v), z(v), w(v) {}
    constexpr int4(int32_t xx, int32_t yy, int32_t zz, int32_t ww) : x(xx), y(yy), z(zz), w(ww) {}
};
inline constexpr int4 make_int4(int32_t x, int32_t y, int32_t z, int32_t w) {
    return int4(x, y, z, w);
}
struct alignas(8) uint2 {
    uint32_t x, y;
    constexpr uint2(uint32_t v = 0) : x(v), y(v) {}
    constexpr uint2(uint32_t xx, uint32_t yy) : x(xx), y(yy) {}
};
inline constexpr uint2 make_uint2(uint32_t x, uint32_t y) {
    return uint2(x, y);
}
struct uint3 {
    uint32_t x, y, z;
    constexpr uint3(uint32_t v = 0) : x(v), y(v), z(v) {}
    constexpr uint3(uint32_t xx, uint32_t yy, uint32_t zz) : x(xx), y(yy), z(zz) {}
};
inline constexpr uint3 make_uint3(uint32_t x, uint32_t y, uint32_t z) {
    return uint3(x, y, z);
}
struct uint4 {
    uint32_t x, y, z, w;
    constexpr uint4(uint32_t v = 0) : x(v), y(v), z(v), w(v) {}
    constexpr uint4(uint32_t xx, uint32_t yy, uint32_t zz, uint32_t ww) : x(xx), y(yy), z(zz), w(ww) {}
};
inline constexpr uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    return uint4(x, y, z, w);
}
struct alignas(8) float2 {
    float x, y;
    constexpr float2(float v = 0) : x(v), y(v) {}
    constexpr float2(float xx, float yy) : x(xx), y(yy) {}
};
inline float2 make_float2(float x, float y) {
    return float2(x, y);
}
struct float3 {
    float x, y, z;
    constexpr float3(float v = 0) : x(v), y(v), z(v) {}
    constexpr float3(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
    constexpr float3(const uint3 &v) :
        x(static_cast<float>(v.x)), y(static_cast<float>(v.y)), z(static_cast<float>(v.z)) {}
};
inline constexpr float3 make_float3(float x, float y, float z) {
    return float3(x, y, z);
}
struct alignas(16) float4 {
    float x, y, z, w;
    constexpr float4(float v = 0) : x(v), y(v), z(v), w(v) {}
    constexpr float4(float xx, float yy, float zz, float ww) : x(xx), y(yy), z(zz), w(ww) {}
    constexpr float4(const float3 &xyz, float ww) : x(xyz.x), y(xyz.y), z(xyz.z), w(ww) {}
};
inline constexpr float4 make_float4(float x, float y, float z, float w) {
    return float4(x, y, z, w);
}

#endif
// END: Define types and functions on the host corresponding to CUDA built-ins.
// ----------------------------------------------------------------



CUDA_COMMON_FUNCTION CUDA_INLINE int2 make_int2(int32_t v) {
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
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator*(int32_t a, const int2 &b) {
    return make_int2(a * b.x, a * b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator*(uint32_t a, const int2 &b) {
    return make_uint2(a * b.x, a * b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator*(const int2 &a, int32_t b) {
    return make_int2(a.x * b, a.y * b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator*(const int2 &a, uint32_t b) {
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
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator*=(int2 &a, int32_t b) {
    a.x *= b;
    a.y *= b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator*=(int2 &a, uint32_t b) {
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
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator/(const int2 &a, int32_t b) {
    return make_int2(a.x / b, a.y / b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const int2 &a, uint32_t b) {
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
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator/=(int2 &a, int32_t b) {
    a.x /= b;
    a.y /= b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator/=(int2 &a, uint32_t b) {
    a.x /= b;
    a.y /= b;
    return a;
}

CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator%(const int2 &a, const int2 &b) {
    return make_int2(a.x % b.x, a.y % b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator<<(const int2 &a, int32_t b) {
    return make_int2(a.x << b, a.y << b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator<<(const int2 &a, uint32_t b) {
    return make_int2(a.x << b, a.y << b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator<<=(int2 &a, int32_t b) {
    a = a << b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator<<=(int2 &a, uint32_t b) {
    a = a << b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator>>(const int2 &a, int32_t b) {
    return make_int2(a.x >> b, a.y >> b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator>>(const int2 &a, uint32_t b) {
    return make_int2(a.x >> b, a.y >> b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator>>=(int2 &a, int32_t b) {
    a = a >> b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator>>=(int2 &a, uint32_t b) {
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

CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator+=(uint2 &a, uint32_t b) {
    a.x += b;
    a.y += b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator-(const uint2 &a, uint32_t b) {
    return make_uint2(a.x - b, a.y - b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator-=(uint2 &a, uint32_t b) {
    a.x -= b;
    a.y -= b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator*(uint32_t a, const uint2 &b) {
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
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator*=(uint2 &a, uint32_t b) {
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
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const uint2 &a, uint32_t b) {
    return make_uint2(a.x / b, a.y / b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator/=(uint2 &a, uint32_t b) {
    a.x /= b;
    a.y /= b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator%(const uint2 &a, const uint2 &b) {
    return make_uint2(a.x % b.x, a.y % b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator<<(const uint2 &a, int32_t b) {
    return make_uint2(a.x << b, a.y << b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator<<(const uint2 &a, uint32_t b) {
    return make_uint2(a.x << b, a.y << b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator<<=(uint2 &a, int32_t b) {
    a = a << b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator<<=(uint2 &a, uint32_t b) {
    a = a << b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator>>(const uint2 &a, int32_t b) {
    return make_uint2(a.x >> b, a.y >> b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator>>(const uint2 &a, uint32_t b) {
    return make_uint2(a.x >> b, a.y >> b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator>>=(uint2 &a, int32_t b) {
    a = a >> b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator>>=(uint2 &a, uint32_t b) {
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

CUDA_COMMON_FUNCTION CUDA_INLINE float3 make_float3(float v) {
    return make_float3(v, v, v);
}

CUDA_COMMON_FUNCTION CUDA_INLINE float4 make_float4(const float3 &xyz, float w) {
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

    CUDA_COMMON_FUNCTION explicit Bool2D() : x(false), y(false) {}
    CUDA_COMMON_FUNCTION explicit Bool2D(bool v) : x(v), y(v) {}
    CUDA_COMMON_FUNCTION Bool2D(bool xx, bool yy) :
        x(xx), y(yy) {}

    template <std::integral I>
    CUDA_COMMON_FUNCTION bool &operator[](I idx) {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION bool operator[](I idx) const {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&x + idx);
    }
};

CUDA_COMMON_FUNCTION CUDA_INLINE bool all(const Bool2D &v) {
    return v.x && v.y;
}

CUDA_COMMON_FUNCTION CUDA_INLINE bool any(const Bool2D &v) {
    return v.x || v.y;
}



template <std::floating_point F>
struct Vector2D_T {
    F x, y;

    CUDA_COMMON_FUNCTION Vector2D_T() : x(0.0f), y(0.0f) {}
    CUDA_COMMON_FUNCTION explicit Vector2D_T(F v) : x(v), y(v) {}
    CUDA_COMMON_FUNCTION Vector2D_T(F xx, F yy) :
        x(xx), y(yy) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION Vector2D_T(const Vector2D_T<F2> &v) :
        x(v.x), y(v.y) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION explicit Vector2D_T(const Vector2D_T<F2> &v) :
        x(static_cast<F>(v.x)), y(static_cast<F>(v.y)) {}

    template <std::integral I>
    CUDA_COMMON_FUNCTION F &operator[](I idx) {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION F operator[](I idx) const {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&x + idx);
    }

#define SWZ2(c0, c1)\
    CUDA_COMMON_FUNCTION Vector2D_T c0##c1() const {\
        return Vector2D_T(c0, c1);\
    }

    SWZ2(x, y);
    SWZ2(y, x);

#undef SWZ2

    CUDA_COMMON_FUNCTION Vector2D_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector2D_T operator-() const {
        return Vector2D_T(-x, -y);
    }

    CUDA_COMMON_FUNCTION Vector2D_T &operator+=(const Vector2D_T &r) {
        x += r.x;
        y += r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector2D_T &operator-=(const Vector2D_T &r) {
        x -= r.x;
        y -= r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector2D_T &operator*=(F r) {
        x *= r;
        y *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector2D_T &operator*=(const Vector2D_T &r) {
        x *= r.x;
        y *= r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector2D_T &operator/=(F r) {
        F rr = 1 / r;
        x *= rr;
        y *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector2D_T &operator/=(const Vector2D_T &r) {
        x /= r.x;
        y /= r.y;
        return *this;
    }

    CUDA_COMMON_FUNCTION F sqLength() const {
        return x * x + y * y;
    }
    CUDA_COMMON_FUNCTION F length() const {
        return std::sqrt(sqLength());
    }
    CUDA_COMMON_FUNCTION Vector2D_T &normalize() {
        F l = length();
        return *this /= l;
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator==(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Bool2D(a.x == b.x, a.y == b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator!=(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Bool2D(a.x != b.x, a.y != b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator<(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Bool2D(a.x < b.x, a.y < b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator<=(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Bool2D(a.x <= b.x, a.y <= b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator>(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Bool2D(a.x > b.x, a.y > b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator>=(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Bool2D(a.x >= b.x, a.y >= b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> operator+(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    Vector2D_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> operator-(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    Vector2D_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> operator*(
    const Vector2D_T<F> &a, N b) {
    Vector2D_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> operator*(
    N a, const Vector2D_T<F> &b) {
    Vector2D_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> operator*(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    Vector2D_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> operator/(
    const Vector2D_T<F> &a, N b) {
    Vector2D_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> operator/(
    N a, const Vector2D_T<F> &b) {
    Vector2D_T<F> ret(static_cast<F>(a));
    ret /= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> operator/(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    Vector2D_T<F> ret = a;
    ret /= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> step(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Vector2D_T<F>(
        b.x >= a.x ? 1.0f : 0.0f,
        b.y >= a.y ? 1.0f : 0.0f);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> sign(
    const Vector2D_T<F> &v) {
    return Vector2D_T<F>(
        v.x > 0.0f ? 1 : v.x < 0.0f ? -1 : 0,
        v.y > 0.0f ? 1 : v.y < 0.0f ? -1 : 0);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> abs(
    const Vector2D_T<F> &v) {
    return Vector2D_T<F>(std::fabs(v.x), std::fabs(v.y));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> min(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Vector2D_T<F>(std::fmin(a.x, b.x), std::fmin(a.y, b.y));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> max(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return Vector2D_T<F>(std::fmax(a.x, b.x), std::fmax(a.y, b.y));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE F dot(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return a.x * b.x + a.y * b.y;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE F cross(
    const Vector2D_T<F> &a, const Vector2D_T<F> &b) {
    return a.x * b.y - a.y * b.x;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> normalize(
    const Vector2D_T<F> &v) {
    Vector2D_T<F> ret = v;
    ret.normalize();
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D_T<F> lerp(
    const Vector2D_T<F> &v0, const Vector2D_T<F> &v1, const Vector2D_T<F> &t) {
    return (Vector2D_T<F>(1.0f) - t) * v0 + t * v1;
}



template <std::floating_point F>
struct Point2D_T {
    F x, y;

    CUDA_COMMON_FUNCTION Point2D_T() : x(0.0f), y(0.0f) {}
    CUDA_COMMON_FUNCTION explicit Point2D_T(F v) : x(v), y(v) {}
    CUDA_COMMON_FUNCTION Point2D_T(F xx, F yy) :
        x(xx), y(yy) {}
    CUDA_COMMON_FUNCTION explicit Point2D_T(const Vector2D_T<F> &v) :
        x(v.x), y(v.y) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION Point2D_T(const Point2D_T<F2> &v) :
        x(v.x), y(v.y) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION explicit Point2D_T(const Point2D_T<F2> &v) :
        x(static_cast<F>(v.x)), y(static_cast<F>(v.y)) {}

    CUDA_COMMON_FUNCTION explicit operator Vector2D_T<F>() const {
        return Vector2D_T<F>(x, y);
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION F &operator[](I idx) {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION F operator[](I idx) const {
        Assert(static_cast<uint32_t>(idx) < 2, "idx is out of bound.");
        return *(&x + idx);
    }

#define SWZ2(c0, c1)\
    CUDA_COMMON_FUNCTION Point2D_T c0##c1() const {\
        return Point2D_T(c0, c1);\
    }

    SWZ2(x, y);
    SWZ2(y, x);

#undef SWZ2

    CUDA_COMMON_FUNCTION Point2D_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D_T operator-() const {
        return Point2D_T(-x, -y);
    }

    CUDA_COMMON_FUNCTION Point2D_T &operator+=(const Point2D_T &r) {
        x += r.x;
        y += r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D_T &operator+=(const Vector2D_T<F> &r) {
        x += r.x;
        y += r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D_T &operator-=(const Vector2D_T<F> &r) {
        x -= r.x;
        y -= r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D_T &operator*=(F r) {
        x *= r;
        y *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D_T &operator*=(const Point2D_T &r) {
        x *= r.x;
        y *= r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D_T &operator/=(F r) {
        F rr = 1 / r;
        x *= rr;
        y *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D_T &operator/=(const Point2D_T &r) {
        x /= r.x;
        y /= r.y;
        return *this;
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator==(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Bool2D(a.x == b.x, a.y == b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator!=(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Bool2D(a.x != b.x, a.y != b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator<(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Bool2D(a.x < b.x, a.y < b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator<=(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Bool2D(a.x <= b.x, a.y <= b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator>(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Bool2D(a.x > b.x, a.y > b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator>=(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Bool2D(a.x >= b.x, a.y >= b.y);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> operator+(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    Point2D_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> operator+(
    const Point2D_T<F> &a, const Vector2D_T<F> &b) {
    Point2D_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> operator+(
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
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> operator-(
    const Point2D_T<F> &a, const Vector2D_T<F> &b) {
    Point2D_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> operator-(
    const Vector2D_T<F> &a, const Point2D_T<F> &b) {
    Point2D_T<F> ret = -b;
    ret += a;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> operator*(
    const Point2D_T<F> &a, N b) {
    Point2D_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> operator*(
    N a, const Point2D_T<F> &b) {
    Point2D_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> operator*(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    Point2D_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> operator/(
    const Point2D_T<F> &a, N b) {
    Point2D_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> operator/(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    Point2D_T<F> ret = a;
    ret /= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> step(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Point2D_T<F>(
        b.x >= a.x ? 1.0f : 0.0f,
        b.y >= a.y ? 1.0f : 0.0f);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> sign(
    const Point2D_T<F> &v) {
    return Point2D_T<F>(
        v.x > 0.0f ? 1 : v.x < 0.0f ? -1 : 0,
        v.y > 0.0f ? 1 : v.y < 0.0f ? -1 : 0);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> abs(
    const Point2D_T<F> &v) {
    return Point2D_T<F>(std::fabs(v.x), std::fabs(v.y));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> min(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Point2D_T<F>(std::fmin(a.x, b.x), std::fmin(a.y, b.y));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> max(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    return Point2D_T<F>(std::fmax(a.x, b.x), std::fmax(a.y, b.y));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE F sqDistance(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
    Vector2D_T<F> d = b - a;
    return d.x * d.x + d.y * d.y;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE F distance(
    const Point2D_T<F> &a, const Point2D_T<F> &b) {
#if !defined(__CUDA_ARCH__)
    using std::sqrtf;
#endif
    return sqrtf(sqDistance(a, b));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D_T<F> lerp(
    const Point2D_T<F> &v0, const Point2D_T<F> &v1, const Point2D_T<F> &t) {
    return Point2D_T<F>(Point2D_T<F>(1.0f) - t) * v0 + t * v1;
}



struct Bool3D {
    bool x, y, z;

    CUDA_COMMON_FUNCTION Bool3D() : x(false), y(false), z(false) {}
    CUDA_COMMON_FUNCTION explicit Bool3D(bool v) : x(v), y(v), z(v) {}
    CUDA_COMMON_FUNCTION Bool3D(bool xx, bool yy, bool zz) :
        x(xx), y(yy), z(zz) {}

    template <std::integral I>
    CUDA_COMMON_FUNCTION bool &operator[](I idx) {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION bool operator[](I idx) const {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&x + idx);
    }
};

CUDA_COMMON_FUNCTION CUDA_INLINE bool all(const Bool3D &v) {
    return v.x && v.y && v.z;
}

CUDA_COMMON_FUNCTION CUDA_INLINE bool any(const Bool3D &v) {
    return v.x || v.y || v.z;
}



template <std::floating_point F, bool isNormal>
struct Vector3D_T {
    F x, y, z;

    CUDA_COMMON_FUNCTION Vector3D_T() : x(0.0f), y(0.0f), z(0.0f) {}
    CUDA_COMMON_FUNCTION explicit Vector3D_T(F v) : x(v), y(v), z(v) {}
    CUDA_COMMON_FUNCTION Vector3D_T(F xx, F yy, F zz) :
        x(xx), y(yy), z(zz) {}
    CUDA_COMMON_FUNCTION Vector3D_T(const Vector2D_T<F> &xy, F zz) :
        x(xy.x), y(xy.y), z(zz) {}
    CUDA_COMMON_FUNCTION explicit Vector3D_T(const Vector3D_T<F, !isNormal> &v) :
        x(v.x), y(v.y), z(v.z) {}
    CUDA_COMMON_FUNCTION explicit Vector3D_T(const float3 &v) :
        x(v.x), y(v.y), z(v.z) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION Vector3D_T(const Vector3D_T<F2, isNormal> &v) :
        x(v.x), y(v.y), z(v.z) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION explicit Vector3D_T(const Vector3D_T<F2, isNormal> &v) :
        x(static_cast<F>(v.x)), y(static_cast<F>(v.y)), z(static_cast<F>(v.z)) {}

    CUDA_COMMON_FUNCTION explicit operator Vector3D_T<F, !isNormal>() const {
        return Vector3D_T<F, !isNormal>(x, y, z);
    }
    CUDA_COMMON_FUNCTION explicit operator float3() const {
        return make_float3(x, y, z);
    }

    CUDA_COMMON_FUNCTION float3 toNative() const {
        return make_float3(x, y, z);
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION F &operator[](I idx) {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION F operator[](I idx) const {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&x + idx);
    }

#define SWZ2(c0, c1)\
    CUDA_COMMON_FUNCTION Vector2D_T<F> c0##c1() const {\
        return Vector2D_T<F>(c0, c1);\
    }
#define SWZ3(c0, c1, c2)\
    CUDA_COMMON_FUNCTION Vector3D_T c0##c1##c2() const {\
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

    CUDA_COMMON_FUNCTION Vector3D_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector3D_T operator-() const {
        return Vector3D_T(-x, -y, -z);
    }

    template <bool isNormalB>
    CUDA_COMMON_FUNCTION Vector3D_T &operator+=(const Vector3D_T<F, isNormalB> &r) {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }
    template <bool isNormalB>
    CUDA_COMMON_FUNCTION Vector3D_T &operator-=(const Vector3D_T<F, isNormalB> &r) {
        x -= r.x;
        y -= r.y;
        z -= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector3D_T &operator*=(F r) {
        x *= r;
        y *= r;
        z *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector3D_T &operator*=(const Vector3D_T &r) {
        x *= r.x;
        y *= r.y;
        z *= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector3D_T &operator/=(F r) {
        F rr = 1 / r;
        x *= rr;
        y *= rr;
        z *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector3D_T &operator/=(const Vector3D_T &r) {
        x /= r.x;
        y /= r.y;
        z /= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector3D_T &safeDivide(const Vector3D_T &r) {
        x = r.x != 0 ? x / r.x : 0.0f;
        y = r.y != 0 ? y / r.y : 0.0f;
        z = r.z != 0 ? z / r.z : 0.0f;
        return *this;
    }

    CUDA_COMMON_FUNCTION F sqLength() const {
        return x * x + y * y + z * z;
    }
    CUDA_COMMON_FUNCTION F length() const {
        return std::sqrt(sqLength());
    }
    CUDA_COMMON_FUNCTION Vector3D_T &normalize() {
        F l = length();
        return *this /= l;
    }

    CUDA_COMMON_FUNCTION bool allFinite() const {
#if !defined(__CUDA_ARCH__)
        using std::isfinite;
#endif
        return isfinite(x) && isfinite(y) && isfinite(z);
    }
};

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator==(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Bool3D(a.x == b.x, a.y == b.y, a.z == b.z);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator!=(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Bool3D(a.x != b.x, a.y != b.y, a.z != b.z);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator<(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Bool3D(a.x < b.x, a.y < b.y, a.z < b.z);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator<=(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Bool3D(a.x <= b.x, a.y <= b.y, a.z <= b.z);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator>(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Bool3D(a.x > b.x, a.y > b.y, a.z > b.z);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator>=(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Bool3D(a.x >= b.x, a.y >= b.y, a.z >= b.z);
}

template <std::floating_point F, bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormalA> operator+(
    const Vector3D_T<F, isNormalA> &a, const Vector3D_T<F, isNormalB> &b) {
    Vector3D_T<F, isNormalA> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F, bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormalA> operator-(
    const Vector3D_T<F, isNormalA> &a, const Vector3D_T<F, isNormalB> &b) {
    Vector3D_T<F, isNormalA> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, bool isNormal, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> operator*(
    const Vector3D_T<F, isNormal> &a, N b) {
    Vector3D_T<F, isNormal> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> operator*(
    N a, const Vector3D_T<F, isNormal> &b) {
    Vector3D_T<F, isNormal> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> operator*(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    Vector3D_T<F, isNormal> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, bool isNormal, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> operator/(
    const Vector3D_T<F, isNormal> &a, N b) {
    Vector3D_T<F, isNormal> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> operator/(
    N a, const Vector3D_T<F, isNormal> &b) {
    Vector3D_T<F, isNormal> ret(static_cast<F>(a));
    ret /= b;
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> operator/(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    Vector3D_T<F, isNormal> ret = a;
    ret /= b;
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> safeDivide(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    Vector3D_T<F, isNormal> ret = a;
    ret.safeDivide(b);
    return ret;
}

template <std::floating_point F, bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE F dot(
    const Vector3D_T<F, isNormalA> &a, const Vector3D_T<F, isNormalB> &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <std::floating_point F, bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, false> cross(
    const Vector3D_T<F, isNormalA> &a, const Vector3D_T<F, isNormalB> &b) {
    return Vector3D_T<F, false>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE F length(
    const Vector3D_T<F, isNormal> &v) {
    return v.length();
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> normalize(
    const Vector3D_T<F, isNormal> &v) {
    Vector3D_T<F, isNormal> ret = v;
    ret.normalize();
    return ret;
}

template <std::floating_point F, bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, false> step(
    const Vector3D_T<F, isNormalA> &a, const Vector3D_T<F, isNormalB> &b) {
    return Vector3D_T<F, false>(
        b.x >= a.x ? 1.0f : 0.0f,
        b.y >= a.y ? 1.0f : 0.0f,
        b.z >= a.z ? 1.0f : 0.0f);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> sign(
    const Vector3D_T<F, isNormal> &v) {
    return Vector3D_T<F, isNormal>(
        v.x > 0.0f ? 1 : v.x < 0.0f ? -1 : 0,
        v.y > 0.0f ? 1 : v.y < 0.0f ? -1 : 0,
        v.z > 0.0f ? 1 : v.z < 0.0f ? -1 : 0);
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> abs(
    const Vector3D_T<F, isNormal> &v) {
    return Vector3D_T<F, isNormal>(std::fabs(v.x), std::fabs(v.y), std::fabs(v.z));
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> min(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Vector3D_T<F, isNormal>(std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z));
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> max(
    const Vector3D_T<F, isNormal> &a, const Vector3D_T<F, isNormal> &b) {
    return Vector3D_T<F, isNormal>(std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z));
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> lerp(
    const Vector3D_T<F, isNormal> &v0, const Vector3D_T<F, isNormal> &v1,
    const Vector3D_T<F, isNormal> &t) {
    return (Vector3D_T<F, isNormal>(1.0f) - t) * v0 + t * v1;
}



template <std::floating_point F>
struct Point3D_T {
    F x, y, z;

    CUDA_COMMON_FUNCTION Point3D_T() : x(0.0f), y(0.0f), z(0.0f) {}
    CUDA_COMMON_FUNCTION explicit Point3D_T(F v) : x(v), y(v), z(v) {}
    CUDA_COMMON_FUNCTION Point3D_T(F xx, F yy, F zz) :
        x(xx), y(yy), z(zz) {}
    CUDA_COMMON_FUNCTION Point3D_T(const Point2D_T<F> &xy, F zz) :
        x(xy.x), y(xy.y), z(zz) {}
    template <bool isNormal>
    CUDA_COMMON_FUNCTION explicit Point3D_T(const Vector3D_T<F, isNormal> &v) : x(v.x), y(v.y), z(v.z) {}
    CUDA_COMMON_FUNCTION explicit Point3D_T(const float3 &p) : x(p.x), y(p.y), z(p.z) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION Point3D_T(const Point3D_T<F2> &v) :
        x(v.x), y(v.y), z(v.z) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION explicit Point3D_T(const Point3D_T<F2> &v) :
        x(static_cast<F>(v.x)), y(static_cast<F>(v.y)), z(static_cast<F>(v.z)) {}

    template <typename F2 = F, std::enable_if_t<std::is_same_v<F2, double>, int> = 0>
    CUDA_COMMON_FUNCTION Point3D_T(const Point3D_T<float> &v) :
        x(v.x), y(v.y), z(v.z) {}
    template <typename F2 = F, std::enable_if_t<std::is_same_v<F2, float>, int> = 0>
    CUDA_COMMON_FUNCTION explicit Point3D_T(const Point3D_T<double> &v) :
        x(static_cast<F>(v.x)), y(static_cast<F>(v.y)), z(static_cast<F>(v.z)) {}

    CUDA_COMMON_FUNCTION explicit operator Point2D_T<F>() const {
        return Point2D_T<F>(x, y);
    }
    template <bool isNormal>
    CUDA_COMMON_FUNCTION explicit operator Vector3D_T<F, isNormal>() const {
        return Vector3D_T<F, isNormal>(x, y, z);
    }
    CUDA_COMMON_FUNCTION explicit operator float3() const {
        return make_float3(x, y, z);
    }

    CUDA_COMMON_FUNCTION float3 toNative() const {
        return make_float3(x, y, z);
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION F &operator[](I idx) {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION F operator[](I idx) const {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&x + idx);
    }

#define SWZ2(c0, c1)\
    CUDA_COMMON_FUNCTION Point2D_T<F> c0##c1() const {\
        return Point2D_T<F>(c0, c1);\
    }
#define SWZ3(c0, c1, c2)\
    CUDA_COMMON_FUNCTION Point3D_T c0##c1##c2() const {\
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

    CUDA_COMMON_FUNCTION Point3D_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Point3D_T operator-() const {
        return Point3D_T(-x, -y, -z);
    }

    CUDA_COMMON_FUNCTION Point3D_T &operator+=(const Point3D_T &r) {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }
    template <bool isNormal>
    CUDA_COMMON_FUNCTION Point3D_T &operator+=(const Vector3D_T<F, isNormal> &r) {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }
    template <bool isNormal>
    CUDA_COMMON_FUNCTION Point3D_T &operator-=(const Vector3D_T<F, isNormal> &r) {
        x -= r.x;
        y -= r.y;
        z -= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point3D_T &operator*=(F r) {
        x *= r;
        y *= r;
        z *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point3D_T &operator*=(const Point3D_T &r) {
        x *= r.x;
        y *= r.y;
        z *= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point3D_T &operator/=(F r) {
        F rr = 1 / r;
        x *= rr;
        y *= rr;
        z *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point3D_T &operator/=(const Point3D_T &r) {
        x /= r.x;
        y /= r.y;
        z /= r.z;
        return *this;
    }

    CUDA_COMMON_FUNCTION bool allFinite() const {
#if !defined(__CUDA_ARCH__)
        using std::isfinite;
#endif
        return isfinite(x) && isfinite(y) && isfinite(z);
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator==(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Bool3D(a.x == b.x, a.y == b.y, a.z == b.z);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator!=(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Bool3D(a.x != b.x, a.y != b.y, a.z != b.z);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator<(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Bool3D(a.x < b.x, a.y < b.y, a.z < b.z);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator<=(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Bool3D(a.x <= b.x, a.y <= b.y, a.z <= b.z);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator>(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Bool3D(a.x > b.x, a.y > b.y, a.z > b.z);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator>=(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Bool3D(a.x >= b.x, a.y >= b.y, a.z >= b.z);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> operator+(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    Point3D_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> operator+(
    const Point3D_T<F> &a, const Vector3D_T<F, isNormal> &b) {
    Point3D_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> operator+(
    const Vector3D_T<F, isNormal> &a, const Point3D_T<F> &b) {
    Point3D_T<F> ret = b;
    ret += a;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, false> operator-(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    auto ret = static_cast<Vector3D_T<F, false>>(a);
    ret -= static_cast<Vector3D_T<F, false>>(b);
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> operator-(
    const Point3D_T<F> &a, const Vector3D_T<F, isNormal> &b) {
    Point3D_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> operator-(
    const Vector3D_T<F, isNormal> &a, const Point3D_T<F> &b) {
    Point3D_T<F> ret = -b;
    ret += a;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> operator*(
    const Point3D_T<F> &a, N b) {
    Point3D_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> operator*(
    N a, const Point3D_T<F> &b) {
    Point3D_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> operator*(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    Point3D_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> operator/(
    const Point3D_T<F> &a, N b) {
    Point3D_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> operator/(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    Point3D_T<F> ret = a;
    ret /= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> min(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Point3D_T<F>(std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> max(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    return Point3D_T<F>(std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE F sqDistance(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
    Vector3D_T<F, false> d = b - a;
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE F distance(
    const Point3D_T<F> &a, const Point3D_T<F> &b) {
#if !defined(__CUDA_ARCH__)
    using std::sqrtf;
#endif
    return sqrtf(sqDistance(a, b));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> lerp(
    const Point3D_T<F> &v0, const Point3D_T<F> &v1,
    const Point3D_T<F> &t) {
    return Point3D_T<F>(Point3D_T<F>(1.0f) - t) * v0 + t * v1;
}



struct Bool4D {
    bool x, y, z, w;

    CUDA_COMMON_FUNCTION Bool4D() : x(false), y(false), z(false), w(false) {}
    CUDA_COMMON_FUNCTION explicit Bool4D(bool v) : x(v), y(v), z(v), w(v) {}
    CUDA_COMMON_FUNCTION Bool4D(bool xx, bool yy, bool zz, bool ww) :
        x(xx), y(yy), z(zz), w(ww) {}
};

CUDA_COMMON_FUNCTION CUDA_INLINE bool all(const Bool4D &v) {
    return v.x && v.y && v.z && v.w;
}

CUDA_COMMON_FUNCTION CUDA_INLINE bool any(const Bool4D &v) {
    return v.x || v.y || v.z || v.w;
}



template <std::floating_point F>
struct Vector4D_T {
    F x, y, z, w;

    CUDA_COMMON_FUNCTION Vector4D_T() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    CUDA_COMMON_FUNCTION explicit Vector4D_T(F v) : x(v), y(v), z(v), w(v) {}
    CUDA_COMMON_FUNCTION Vector4D_T(F xx, F yy, F zz, F ww) :
        x(xx), y(yy), z(zz), w(ww) {}
    CUDA_COMMON_FUNCTION Vector4D_T(const Vector3D_T<F, false> &v, F ww = 0) :
        x(v.x), y(v.y), z(v.z), w(ww) {}
    CUDA_COMMON_FUNCTION Vector4D_T(const Point3D_T<F> &p, F ww = 1) :
        x(p.x), y(p.y), z(p.z), w(ww) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION Vector4D_T(const Vector4D_T<F2> &v) :
        x(v.x), y(v.y), z(v.z), w(v.w) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION explicit Vector4D_T(const Vector4D_T<F2> &v) :
        x(static_cast<F>(v.x)), y(static_cast<F>(v.y)), z(static_cast<F>(v.z)), w(static_cast<F>(v.w)) {}

    template <bool isNormal>
    CUDA_COMMON_FUNCTION explicit operator Vector3D_T<F, isNormal>() const {
        return Vector3D_T<F, isNormal>(x, y, z);
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION F &operator[](I idx) {
        Assert(static_cast<uint32_t>(idx) < 4, "idx is out of bound.");
        return *(&x + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION F operator[](I idx) const {
        Assert(static_cast<uint32_t>(idx) < 4, "idx is out of bound.");
        return *(&x + idx);
    }

#define SWZ2(c0, c1)\
    CUDA_COMMON_FUNCTION Vector2D_T<F> c0##c1() const {\
        return Vector2D_T<F>(c0, c1);\
    }
#define SWZ3(c0, c1, c2)\
    template <bool isNormal = false>\
    CUDA_COMMON_FUNCTION Vector3D_T<F, isNormal> c0##c1##c2() const {\
        return Vector3D_T<F, isNormal>(c0, c1, c2);\
    }
#define SWZ4(c0, c1, c2, c3)\
    CUDA_COMMON_FUNCTION Vector4D_T c0##c1##c2##c3() const {\
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


    CUDA_COMMON_FUNCTION Vector4D_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector4D_T operator-() const {
        return Vector4D_T(-x, -y, -z, -w);
    }

    CUDA_COMMON_FUNCTION Vector4D_T &operator+=(const Vector4D_T &r) {
        x += r.x;
        y += r.y;
        z += r.z;
        w += r.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector4D_T &operator-=(const Vector4D_T &r) {
        x -= r.x;
        y -= r.y;
        z -= r.z;
        w -= r.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector4D_T &operator*=(F r) {
        x *= r;
        y *= r;
        z *= r;
        w *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector4D_T &operator*=(const Vector4D_T &r) {
        x *= r.x;
        y *= r.y;
        z *= r.z;
        w *= r.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector4D_T &operator/=(F r) {
        F rr = 1 / r;
        x *= rr;
        y *= rr;
        z *= rr;
        w *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector4D_T &operator/=(const Vector4D_T &r) {
        x /= r.x;
        y /= r.y;
        z /= r.z;
        w /= r.w;
        return *this;
    }

    CUDA_COMMON_FUNCTION bool allFinite() const {
#if !defined(__CUDA_ARCH__)
        using std::isfinite;
#endif
        return isfinite(x) && isfinite(y) && isfinite(z) && isfinite(w);
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool4D operator==(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return Bool4D(a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool4D operator!=(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return Bool4D(a.x != b.x, a.y != b.y, a.z != b.z, a.w != b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool4D operator<(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return Bool4D(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool4D operator<=(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return Bool4D(a.x <= b.x, a.y <= b.y, a.z <= b.z, a.w <= b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool4D operator>(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return Bool4D(a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool4D operator>=(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return Bool4D(a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D_T<F> operator+(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    Vector4D_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D_T<F> operator-(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    Vector4D_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D_T<F> operator*(
    const Vector4D_T<F> &a, N b) {
    Vector4D_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D_T<F> operator*(
    N a, const Vector4D_T<F> &b) {
    Vector4D_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D_T<F> operator*(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    Vector4D_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D_T<F> operator/(
    const Vector4D_T<F> &a, N b) {
    Vector4D_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D_T<F> operator/(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    Vector4D_T<F> ret = a;
    ret /= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE F dot(
    const Vector4D_T<F> &a, const Vector4D_T<F> &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
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

    CUDA_COMMON_FUNCTION Matrix3x3_T() :
        c0(1, 0, 0), c1(0, 1, 0), c2(0, 0, 1) {}
    CUDA_COMMON_FUNCTION Matrix3x3_T(
        const Vector3D_T<F, false> &cc0,
        const Vector3D_T<F, false> &cc1,
        const Vector3D_T<F, false> &cc2) :
        c0(cc0), c1(cc1), c2(cc2) {}
    CUDA_COMMON_FUNCTION Matrix3x3_T(
        const Vector3D_T<F, true> &cc0,
        const Vector3D_T<F, true> &cc1,
        const Vector3D_T<F, true> &cc2) :
        c0(cc0), c1(cc1), c2(cc2) {}
    CUDA_COMMON_FUNCTION Matrix3x3_T(
        const Point3D_T<F> &cc0,
        const Point3D_T<F> &cc1,
        const Point3D_T<F> &cc2) :
        c0(static_cast<Vector3D_T<F, false>>(cc0)),
        c1(static_cast<Vector3D_T<F, false>>(cc1)),
        c2(static_cast<Vector3D_T<F, false>>(cc2)) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION Matrix3x3_T(const Matrix3x3_T<F2> &v) :
        c0(v.c0), c1(v.c1), c2(v.c2) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION explicit Matrix3x3_T(const Matrix3x3_T<F2> &v) :
        c0(static_cast<Vector3D_T<F, false>>(v.c0)),
        c1(static_cast<Vector3D_T<F, false>>(v.c1)),
        c2(static_cast<Vector3D_T<F, false>>(v.c2)) {}

    template <std::integral I>
    CUDA_COMMON_FUNCTION Vector3D_T<F, false> &operator[](I idx) {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&c0 + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION Vector3D_T<F, false> operator[](I idx) const {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&c0 + idx);
    }

    CUDA_COMMON_FUNCTION Matrix3x3_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Matrix3x3_T operator-() const {
        return Matrix3x3_T(-c0, -c1, -c2);
    }

    template <Number N>
    CUDA_COMMON_FUNCTION Matrix3x3_T &operator*=(N r) {
        c0 *= r;
        c1 *= r;
        c2 *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION Matrix3x3_T &operator*=(const Matrix3x3_T &r) {
        Vector3D_T<F, false> rs[] = { row(0), row(1), row(2) };
        m00 = dot(rs[0], r.c0); m01 = dot(rs[0], r.c1); m02 = dot(rs[0], r.c2);
        m10 = dot(rs[1], r.c0); m11 = dot(rs[1], r.c1); m12 = dot(rs[1], r.c2);
        m20 = dot(rs[2], r.c0); m21 = dot(rs[2], r.c1); m22 = dot(rs[2], r.c2);
        return *this;
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION Vector3D_T<F, false> row(I index) const {
        switch (index) {
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

    CUDA_COMMON_FUNCTION Matrix3x3_T &invert() {
        F det = 1 /
            (m00 * m11 * m22 + m01 * m12 * m20 + m02 * m10 * m21
             - m02 * m11 * m20 - m01 * m10 * m22 - m00 * m12 * m21);
        Matrix3x3_T m;
        m.m00 = det * (m11 * m22 - m12 * m21);
        m.m01 = -det * (m01 * m22 - m02 * m21);
        m.m02 = det * (m01 * m12 - m02 * m11);
        m.m10 = -det * (m10 * m22 - m12 * m20);
        m.m11 = det * (m00 * m22 - m02 * m20);
        m.m12 = -det * (m00 * m12 - m02 * m10);
        m.m20 = det * (m10 * m21 - m11 * m20);
        m.m21 = -det * (m00 * m21 - m01 * m20);
        m.m22 = det * (m00 * m11 - m01 * m10);
        *this = m;

        return *this;
    }

    CUDA_COMMON_FUNCTION Matrix3x3_T &transpose() {
        F temp;
        temp = m10; m10 = m01; m01 = temp;
        temp = m20; m20 = m02; m02 = temp;
        temp = m21; m21 = m12; m12 = temp;
        return *this;
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3_T<F> operator*(
    const Matrix3x3_T<F> &a, const Matrix3x3_T<F> &b) {
    Matrix3x3_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D_T<F, isNormal> operator*(
    const Matrix3x3_T<F> &a, const Vector3D_T<F, isNormal> &b) {
    return Vector3D_T<F, isNormal>(dot(a.row(0), b), dot(a.row(1), b), dot(a.row(2), b));
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3_T<F> operator*(N a, const Matrix3x3_T<F> &b) {
    Matrix3x3_T<F> ret = b;
    ret *= a;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D_T<F> operator*(
    const Matrix3x3_T<F> &a, const Point3D_T<F> &b) {
    auto vb = static_cast<Vector3D_T<F, false>>(b);
    return Point3D_T<F>(
        dot(a.row(0), vb),
        dot(a.row(1), vb),
        dot(a.row(2), vb));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3_T<F> invert(
    const Matrix3x3_T<F> &m) {
    Matrix3x3_T<F> ret = m;
    ret.invert();
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3_T<F> transpose(
    const Matrix3x3_T<F> &m) {
    Matrix3x3_T<F> ret = m;
    ret.transpose();
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3_T<F> scale3x3(
    const Vector3D_T<F, false> &s) {
    return Matrix3x3_T<F>(
        Vector3D_T<F, false>(s.x, 0, 0),
        Vector3D_T<F, false>(0, s.y, 0),
        Vector3D_T<F, false>(0, 0, s.z));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3_T<F> scale3x3(
    F sx, F sy, F sz) {
    return scale3x3(Vector3D_T<F, false>(sx, sy, sz));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3_T<F> scale3x3(
    F s) {
    return scale3x3(Vector3D_T<F, false>(s, s, s));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3_T<F> rotate3x3(
    F angle, const Vector3D_T<F, false> &axis) {

    Matrix3x3_T<F> ret;
    Vector3D_T<F, false> nAxis = normalize(axis);
    F s = std::sin(angle);
    F c = std::cos(angle);
    F oneMinusC = 1 - c;

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
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3_T<F> rotate3x3(
    F angle, F ax, F ay, F az) {
    return rotate3x3(angle, Vector3D_T<F, false>(ax, ay, az));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3_T<F> rotateX3x3(
    F angle) {
    return rotate3x3(angle, Vector3D_T<F, false>(1, 0, 0));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3_T<F> rotateY3x3(
    F angle) {
    return rotate3x3(angle, Vector3D_T<F, false>(0, 1, 0));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3_T<F> rotateZ3x3(
    F angle) {
    return rotate3x3(angle, Vector3D_T<F, false>(0, 0, 1));
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

    CUDA_COMMON_FUNCTION /*constexpr*/ Matrix4x4_T() :
        c0(1, 0, 0, 0),
        c1(0, 1, 0, 0),
        c2(0, 0, 1, 0),
        c3(0, 0, 0, 1) { }
    CUDA_COMMON_FUNCTION Matrix4x4_T(const F array[9]) :
        m00(array[0]), m10(array[1]), m20(array[2]), m30(array[3]),
        m01(array[4]), m11(array[5]), m21(array[6]), m31(array[7]),
        m02(array[8]), m12(array[9]), m22(array[10]), m32(array[11]),
        m03(array[12]), m13(array[13]), m23(array[14]), m33(array[15]) { }
    CUDA_COMMON_FUNCTION constexpr Matrix4x4_T(
        const Vector4D_T<F> &col0,
        const Vector4D_T<F> &col1,
        const Vector4D_T<F> &col2,
        const Vector4D_T<F> &col3) :
        c0(col0), c1(col1), c2(col2), c3(col3)
    { }
    CUDA_COMMON_FUNCTION Matrix4x4_T(
        const Matrix3x3_T<F> &mat3x3, const Point3D_T<F> &position = Point3D_T<F>(0.0f)) :
        c0(Vector4D_T<F>(mat3x3.c0)),
        c1(Vector4D_T<F>(mat3x3.c1)),
        c2(Vector4D_T<F>(mat3x3.c2)),
        c3(Vector4D_T<F>(position))
    { }

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION Matrix4x4_T(const Matrix4x4_T<F2> &v) :
        c0(v.c0), c1(v.c1), c2(v.c2), c3(v.c3) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION explicit Matrix4x4_T(const Matrix4x4_T<F2> &v) :
        c0(static_cast<Vector4D_T<F>>(v.c0)),
        c1(static_cast<Vector4D_T<F>>(v.c1)),
        c2(static_cast<Vector4D_T<F>>(v.c2)),
        c3(static_cast<Vector4D_T<F>>(v.c3)) {}

    template <std::integral I>
    CUDA_COMMON_FUNCTION Vector4D_T<F> &operator[](I idx) {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&c0 + idx);
    }
    template <std::integral I>
    CUDA_COMMON_FUNCTION Vector4D_T<F> operator[](I idx) const {
        Assert(static_cast<uint32_t>(idx) < 3, "idx is out of bound.");
        return *(&c0 + idx);
    }

    CUDA_COMMON_FUNCTION Matrix4x4_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Matrix4x4_T operator-() const {
        return Matrix4x4_T(-c0, -c1, -c2, -c3);
    }

    CUDA_COMMON_FUNCTION Matrix4x4_T operator+(const Matrix4x4_T &mat) const {
        return Matrix4x4_T(c0 + mat.c0, c1 + mat.c1, c2 + mat.c2, c3 + mat.c3);
    }
    CUDA_COMMON_FUNCTION Matrix4x4_T operator-(const Matrix4x4_T &mat) const {
        return Matrix4x4_T(c0 - mat.c0, c1 - mat.c1, c2 - mat.c2, c3 - mat.c3);
    }
    CUDA_COMMON_FUNCTION Matrix4x4_T operator*(const Matrix4x4_T &mat) const {
        const Vector4D_T<F> r[] = { row(0), row(1), row(2), row(3) };
        return Matrix4x4_T(
            Vector4D_T<F>(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0), dot(r[3], mat.c0)),
            Vector4D_T<F>(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1), dot(r[3], mat.c1)),
            Vector4D_T<F>(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2), dot(r[3], mat.c2)),
            Vector4D_T<F>(dot(r[0], mat.c3), dot(r[1], mat.c3), dot(r[2], mat.c3), dot(r[3], mat.c3)));
    }
    CUDA_COMMON_FUNCTION Vector3D_T<F, false> operator*(const Vector3D_T<F, false> &v) const {
        const Vector4D_T<F> r[] = { row(0), row(1), row(2), row(3) };
        Vector4D_T<F> v4(v, 0.0f);
        return Vector3D_T<F, false>(
            dot(r[0], v4),
            dot(r[1], v4),
            dot(r[2], v4));
    }
    CUDA_COMMON_FUNCTION Point3D_T<F> operator*(const Point3D_T<F> &p) const {
        const Vector4D_T<F> r[] = { row(0), row(1), row(2), row(3) };
        Vector4D_T<F> v4(p, 1.0f);
        return Point3D_T<F>(
            dot(r[0], v4),
            dot(r[1], v4),
            dot(r[2], v4));
    }
    CUDA_COMMON_FUNCTION Vector4D_T<F> operator*(const Vector4D_T<F> &v) const {
        const Vector4D_T<F> r[] = { row(0), row(1), row(2), row(3) };
        return Vector4D_T<F>(
            dot(r[0], v),
            dot(r[1], v),
            dot(r[2], v),
            dot(r[3], v));
    }

    CUDA_COMMON_FUNCTION Matrix4x4_T &operator*=(const Matrix4x4_T &mat) {
        const Vector4D_T<F> r[] = { row(0), row(1), row(2), row(3) };
        c0 = Vector4D_T<F>(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0), dot(r[3], mat.c0));
        c1 = Vector4D_T<F>(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1), dot(r[3], mat.c1));
        c2 = Vector4D_T<F>(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2), dot(r[3], mat.c2));
        c3 = Vector4D_T<F>(dot(r[0], mat.c3), dot(r[1], mat.c3), dot(r[2], mat.c3), dot(r[3], mat.c3));
        return *this;
    }

    CUDA_COMMON_FUNCTION Vector4D_T<F> &operator[](uint32_t c) {
        //Assert(c < 3, "\"c\" is out of range [0, 3].");
        return *(&c0 + c);
    }
    CUDA_COMMON_FUNCTION Vector4D_T<F> row(unsigned int r) const {
        //Assert(r < 3, "\"r\" is out of range [0, 3].");
        switch (r) {
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

    CUDA_COMMON_FUNCTION Matrix4x4_T &invert() {
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

        F recDet = 1.0f / (m00 * inv[0] + m10 * inv[4] + m20 * inv[8] + m30 * inv[12]);
        for (int i = 0; i < 16; ++i)
            inv[i] *= recDet;
        *this = Matrix4x4_T(inv);

        return *this;
    }

    CUDA_COMMON_FUNCTION Matrix4x4_T &transpose() {
        F temp;
        temp = m10; m10 = m01; m01 = temp;
        temp = m20; m20 = m02; m02 = temp;
        temp = m21; m21 = m12; m12 = temp;
        temp = m30; m30 = m03; m03 = temp;
        temp = m31; m31 = m13; m13 = temp;
        temp = m32; m32 = m23; m23 = temp;
        return *this;
    }

    CUDA_COMMON_FUNCTION Matrix3x3_T<F> getUpperLeftMatrix() const {
        return Matrix3x3_T<F>(
            Vector3D_T<F, false>(c0),
            Vector3D_T<F, false>(c1),
            Vector3D_T<F, false>(c2));
    }

    CUDA_COMMON_FUNCTION void decompose(
        Vector3D_T<F, false>* retScale,
        Vector3D_T<F, false>* rotation,
        Vector3D_T<F, false>* translation) const {
        Matrix4x4_T mat = *this;

        // JP: 移動成分
        // EN: Translation component
        if (translation)
            *translation = Vector3D_T<F, false>(mat.c3);

        Vector3D_T<F, false> scale(
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

        // JP: 回転成分がXYZの順で作られている、つまりZYXp(pは何らかのベクトル)と仮定すると、行列は以下の形式をとっていると考えられる。
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
        F cosBeta = std::cos(rotation->y);

        if (std::fabs(cosBeta) < 0.000001f) {
            rotation->x = 0;
            rotation->z = std::atan2(-mat.c1.x, mat.c1.y);
        }
        else {
            rotation->x = std::atan2(mat.c1.z, mat.c2.z);
            rotation->z = std::atan2(mat.c0.y, mat.c0.x);
        }
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4_T<F> transpose(
    const Matrix4x4_T<F> &mat) {
    Matrix4x4_T<F> ret = mat;
    return ret.transpose();
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4_T<F> invert(
    const Matrix4x4_T<F> &mat) {
    Matrix4x4_T<F> ret = mat;
    return ret.invert();
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4_T<F> scale4x4(
    const Vector3D_T<F, false> &s) {
    return Matrix4x4_T<F>(
        Vector4D_T<F>(s.x, 0, 0, 0),
        Vector4D_T<F>(0, s.y, 0, 0),
        Vector4D_T<F>(0, 0, s.z, 0),
        Vector4D_T<F>(0, 0, 0, 1));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4_T<F> scale4x4(
    F sx, F sy, F sz) {
    return scale4x4(Vector3D_T<F, false>(sx, sy, sz));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4_T<F> scale4x4(
    F s) {
    return scale4x4(Vector3D_T<F, false>(s, s, s));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4_T<F> rotate4x4(
    F angle, const Vector3D_T<F, false> &axis) {
    Matrix4x4_T<F> matrix;
    Vector3D_T<F, false> nAxis = normalize(axis);
    F s = std::sin(angle);
    F c = std::cos(angle);
    F oneMinusC = 1 - c;

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
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4_T<F> rotate4x4(
    F angle, F ax, F ay, F az) {
    return rotate4x4(angle, Vector3D_T<F, false>(ax, ay, az));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4_T<F> rotateX4x4(
    F angle) {
    return rotate4x4(angle, Vector3D_T<F, false>(1, 0, 0));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4_T<F> rotateY4x4(
    F angle) {
    return rotate4x4(angle, Vector3D_T<F, false>(0, 1, 0));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4_T<F> rotateZ4x4(
    F angle) {
    return rotate4x4(angle, Vector3D_T<F, false>(0, 0, 1));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4_T<F> translate4x4(
    const Vector3D_T<F, false> &t) {
    return Matrix4x4_T<F>(
        Vector4D_T<F>(1, 0, 0, 0),
        Vector4D_T<F>(0, 1, 0, 0),
        Vector4D_T<F>(0, 0, 1, 0),
        Vector4D_T<F>(t, 1.0f));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4_T<F> translate4x4(
    F tx, F ty, F tz) {
    return translate4x4(Vector3D_T<F, false>(tx, ty, tz));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4_T<F> camera(
    F aspect, F fovY, F near, F far) {
    Matrix4x4_T<F> matrix;
    F f = 1 / std::tan(fovY / 2);
    F dz = far - near;

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

    CUDA_COMMON_FUNCTION Quaternion_T() :
        v(0), w(1) {}
    CUDA_COMMON_FUNCTION Quaternion_T(F xx, F yy, F zz, F ww) :
        v(xx, yy, zz), w(ww) {}
    CUDA_COMMON_FUNCTION Quaternion_T(const Vector3D_T<F, false> &vv, F ww) :
        v(vv), w(ww) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION Quaternion_T(const Quaternion_T<F2> &v) :
        x(v.x), y(v.y), z(v.z), w(v.w) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION explicit Quaternion_T(const Quaternion_T<F2> &v) :
        x(static_cast<F>(v.x)), y(static_cast<F>(v.y)), z(static_cast<F>(v.z)), w(static_cast<F>(v.w)) {}

    CUDA_COMMON_FUNCTION bool operator==(const Quaternion_T &q) const {
        return all(v == q.v) && w == q.w;
    }
    CUDA_COMMON_FUNCTION bool operator!=(const Quaternion_T &q) const {
        return any(v != q.v) || w != q.w;
    }

    CUDA_COMMON_FUNCTION Quaternion_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Quaternion_T operator-() const {
        return Quaternion_T(-v, -w);
    }

    CUDA_COMMON_FUNCTION Quaternion_T operator+(const Quaternion_T &q) const {
        return Quaternion_T(v + q.v, w + q.w);
    }
    CUDA_COMMON_FUNCTION Quaternion_T operator-(const Quaternion_T &q) const {
        return Quaternion_T(v - q.v, w - q.w);
    }
    CUDA_COMMON_FUNCTION Quaternion_T operator*(const Quaternion_T &q) const {
        return Quaternion_T(cross(v, q.v) + w * q.v + q.w * v, w * q.w - dot(v, q.v));
    }
    CUDA_COMMON_FUNCTION Quaternion_T operator*(F s) const {
        return Quaternion_T(v * s, w * s);
    }
    CUDA_COMMON_FUNCTION Quaternion_T operator/(F s) const {
        F r = 1 / s;
        return *this * r;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE friend Quaternion_T operator*(F s, const Quaternion_T &q) {
        return q * s;
    }

    CUDA_COMMON_FUNCTION void toEulerAngles(F* roll, F* pitch, F* yaw) const {
        F xx = x * x;
        F xy = x * y;
        F xz = x * z;
        F xw = x * w;
        F yy = y * y;
        F yz = y * z;
        F yw = y * w;
        F zz = z * z;
        F zw = z * w;
        F ww = w * w;
        *pitch = std::atan2(2 * (xw + yz), ww - xx - yy + zz); // around x
        *yaw = std::asin(std::fmin(std::fmax(2.0f * (yw - xz), -1.0f), 1.0f)); // around y
        *roll = std::atan2(2 * (zw + xy), ww + xx - yy - zz); // around z
    }
    CUDA_COMMON_FUNCTION Matrix3x3_T<F> toMatrix3x3() const {
        F xx = x * x, yy = y * y, zz = z * z;
        F xy = x * y, yz = y * z, zx = z * x;
        F xw = x * w, yw = y * w, zw = z * w;
        return Matrix3x3_T<F>(
            Vector3D_T<F, false>(1 - 2 * (yy + zz), 2 * (xy + zw), 2 * (zx - yw)),
            Vector3D_T<F, false>(2 * (xy - zw), 1 - 2 * (xx + zz), 2 * (yz + xw)),
            Vector3D_T<F, false>(2 * (zx + yw), 2 * (yz - xw), 1 - 2 * (xx + yy)));
    }

    CUDA_COMMON_FUNCTION bool allFinite() const {
#if !defined(__CUDA_ARCH__)
        using std::isfinite;
#endif
        return v.allFinite() && isfinite(w);
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE bool allFinite(
    const Quaternion_T<F> &q) {
    return q.allFinite();
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE F dot(
    const Quaternion_T<F> &q0, const Quaternion_T<F> &q1) {
    return dot(q0.v, q1.v) + q0.w * q1.w;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion_T<F> normalize(
    const Quaternion_T<F> &q) {
    return q / std::sqrt(dot(q, q));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion_T<F> qRotate(
    F angle, const Vector3D_T<F, false> &axis) {
    F ha = angle / 2;
    F s = std::sin(ha), c = std::cos(ha);
    return Quaternion_T<F>(s * normalize(axis), c);
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion_T<F> qRotate(
    F angle, F ax, F ay, F az) {
    return qRotate(angle, Vector3D_T<F, false>(ax, ay, az));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion_T<F> qRotateX(
    F angle) {
    return qRotate(angle, Vector3D_T<F, false>(1, 0, 0));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion_T<F> qRotateY(
    F angle) {
    return qRotate(angle, Vector3D_T<F, false>(0, 1, 0));
}
template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion_T<F> qRotateZ(
    F angle) {
    return qRotate(angle, Vector3D_T<F, false>(0, 0, 1));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion_T<F> qFromEulerAngles(
    F roll, F pitch, F yaw) {
    return qRotateZ(roll) * qRotateY(yaw) * qRotateX(pitch);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion_T<F> Slerp(
    F t, const Quaternion_T<F> &q0, const Quaternion_T<F> &q1) {
    F cosTheta = dot(q0, q1);
    if (cosTheta > 0.9995f)
        return normalize((1 - t) * q0 + t * q1);
    else {
        F theta = std::acos(std::fmin(std::fmax(cosTheta, -1.0f), 1.0f));
        F thetap = theta * t;
        Quaternion_T<F> qPerp = normalize(q1 - q0 * cosTheta);
        F sinThetaP, cosThetaP;
        sinThetaP = std::sin(thetap);
        cosThetaP = std::cos(thetap);
        //sincos(thetap, &sinThetaP, &cosThetaP);
        return q0 * cosThetaP + qPerp * sinThetaP;
    }
}



template <std::floating_point F>
struct RGB_T {
    F r, g, b;

    CUDA_COMMON_FUNCTION RGB_T() : r(0.0f), g(0.0f), b(0.0f) {}
    CUDA_COMMON_FUNCTION explicit RGB_T(F v) : r(v), g(v), b(v) {}
    CUDA_COMMON_FUNCTION RGB_T(F rr, F gg, F bb) :
        r(rr), g(gg), b(bb) {}
    CUDA_COMMON_FUNCTION explicit RGB_T(const float3 &v) :
        r(v.x), g(v.y), b(v.z) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION RGB_T(const RGB_T<F2> &v) :
        r(v.r), g(v.g), b(v.b) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION explicit RGB_T(const RGB_T<F2> &v) :
        r(static_cast<F>(v.r)), g(static_cast<F>(v.g)), b(static_cast<F>(v.b)) {}

    CUDA_COMMON_FUNCTION explicit operator float3() const {
        return make_float3(r, g, b);
    }

    CUDA_COMMON_FUNCTION float3 toNative() const {
        return make_float3(r, g, b);
    }

    CUDA_COMMON_FUNCTION RGB_T operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB_T operator-() const {
        return RGB_T(-r, -g, -b);
    }

    CUDA_COMMON_FUNCTION RGB_T &operator+=(const RGB_T &o) {
        r += o.r;
        g += o.g;
        b += o.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB_T &operator-=(const RGB_T &o) {
        r -= o.r;
        g -= o.g;
        b -= o.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB_T &operator*=(F o) {
        r *= o;
        g *= o;
        b *= o;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB_T &operator*=(const RGB_T &o) {
        r *= o.r;
        g *= o.g;
        b *= o.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB_T &operator/=(F o) {
        F ro = 1 / o;
        r *= ro;
        g *= ro;
        b *= ro;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB_T &operator/=(const RGB_T &o) {
        r /= o.r;
        g /= o.g;
        b /= o.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB_T &safeDivide(const RGB_T &o) {
        r = o.r != 0 ? r / o.r : 0.0f;
        g = o.g != 0 ? g / o.g : 0.0f;
        b = o.b != 0 ? b / o.b : 0.0f;
        return *this;
    }

    CUDA_COMMON_FUNCTION bool allFinite() const {
#if !defined(__CUDA_ARCH__)
        using std::isfinite;
#endif
        return isfinite(r) && isfinite(g) && isfinite(b);
    }
};

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator==(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return Bool3D(a.r == b.r, a.g == b.g, a.b == b.b);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator!=(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return Bool3D(a.r != b.r, a.g != b.g, a.b != b.b);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator<(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return Bool3D(a.r < b.r, a.g < b.g, a.b < b.b);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator<=(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return Bool3D(a.r <= b.r, a.g <= b.g, a.b <= b.b);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator>(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return Bool3D(a.r > b.r, a.g > b.g, a.b > b.b);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator>=(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return Bool3D(a.r >= b.r, a.g >= b.g, a.b >= b.b);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> operator+(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    RGB_T<F> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> operator-(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    RGB_T<F> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> operator*(
    const RGB_T<F> &a, N b) {
    RGB_T<F> ret = a;
    ret *= static_cast<F>(b);
    return ret;
}

template <Number N, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> operator*(N a, const RGB_T<F> &b) {
    RGB_T<F> ret = b;
    ret *= static_cast<F>(a);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> operator*(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    RGB_T<F> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> operator/(
    const RGB_T<F> &a, N b) {
    RGB_T<F> ret = a;
    ret /= static_cast<F>(b);
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> operator/(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    RGB_T<F> ret = a;
    ret /= b;
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> safeDivide(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    RGB_T<F> ret = a;
    ret.safeDivide(b);
    return ret;
}

template <std::floating_point F, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> safeDivide(
    const RGB_T<F> &a, N b) {
    RGB_T<F> ret = a;
    ret.safeDivide(RGB_T<F>(b));
    return ret;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> abs(
    const RGB_T<F> &v) {
    return RGB_T<F>(std::fabs(v.r), std::fabs(v.g), std::fabs(v.b));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> min(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return RGB_T<F>(std::fmin(a.r, b.r), std::fmin(a.g, b.g), std::fmin(a.b, b.b));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> max(
    const RGB_T<F> &a, const RGB_T<F> &b) {
    return RGB_T<F>(std::fmax(a.r, b.r), std::fmax(a.g, b.g), std::fmax(a.b, b.b));
}



template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> HSVtoRGB(F h, F s, F v) {
    if (s == 0)
        return RGB_T<F>(v, v, v);

    h = h - std::floor(h);
    int32_t hi = static_cast<int32_t>(h * 6);
    F f = h * 6 - hi;
    F m = v * (1 - s);
    F n = v * (1 - s * f);
    F k = v * (1 - s * (1 - f));
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
CUDA_COMMON_FUNCTION CUDA_INLINE F simpleToneMap_s(F value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    return 1 - std::exp(-value);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE F sRGB_degamma_s(F value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    if (value <= 0.04045f)
        return value / 12.92f;
    return std::pow((value + 0.055f) / 1.055f, 2.4f);
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE F sRGB_gamma_s(F value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    if (value <= 0.0031308f)
        return 12.92f * value;
    return 1.055f * std::pow(value, 1 / 2.4f) - 0.055f;
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB_T<F> sRGB_degamma(const RGB_T<F> &value) {
    return RGB_T<F>(
        sRGB_degamma_s(value.r),
        sRGB_degamma_s(value.g),
        sRGB_degamma_s(value.b));
}

template <std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE F sRGB_calcLuminance(const RGB_T<F> &value) {
    return 0.2126729f * value.r + 0.7151522f * value.g + 0.0721750f * value.b;
}



template <std::floating_point F>
struct AABB_T {
    Point3D_T<F> minP;
    Point3D_T<F> maxP;

    CUDA_COMMON_FUNCTION AABB_T() :
        minP(Point3D_T<F>(INFINITY)), maxP(Point3D_T<F>(-INFINITY)) {}
    CUDA_COMMON_FUNCTION AABB_T(
        const Point3D_T<F> &_minP, const Point3D_T<F> &_maxP) :
        minP(_minP), maxP(_maxP) {}

    template <typename F2 = F, std::enable_if_t<(sizeof(F2) <= sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION AABB_T(const AABB_T<F2> &v) :
        minP(v.minP), maxP(v.maxP) {}
    template <typename F2 = F, std::enable_if_t<(sizeof(F2) > sizeof(F)), int> = 0>
    CUDA_COMMON_FUNCTION explicit AABB_T(const AABB_T<F2> &v) :
        minP(static_cast<Point3D_T<F>>(v.minP)), maxP(static_cast<Point3D_T<F>>(v.maxP)) {}

    CUDA_COMMON_FUNCTION AABB_T &unify(const Point3D_T<F> &p) {
        minP = min(minP, p);
        maxP = max(maxP, p);
        return *this;
    }
    CUDA_COMMON_FUNCTION AABB_T &unify(const AABB_T &bb) {
        minP = min(minP, bb.minP);
        maxP = max(maxP, bb.maxP);
        return *this;
    }

    CUDA_COMMON_FUNCTION AABB_T &dilate(F scale) {
        Vector3D_T<F, false> d = maxP - minP;
        minP -= 0.5f * (scale - 1) * d;
        maxP += 0.5f * (scale - 1) * d;
        return *this;
    }

    CUDA_COMMON_FUNCTION Point3D_T<F> normalize(const Point3D_T<F> &p) const {
        return static_cast<Point3D_T<F>>(safeDivide(p - minP, maxP - minP));
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE friend AABB_T operator*(
        const Matrix4x4_T<F> &mat, const AABB_T &aabb) {
        AABB_T ret;
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

    CUDA_COMMON_FUNCTION bool isValid() const {
        Vector3D_T<F, false> d = maxP - minP;
        return all(d >= Vector3D_T<F, false>(0.0f));
    }

    CUDA_COMMON_FUNCTION bool intersect(
        const Point3D_T<F> &org, const Vector3D_T<F, false> &dir, F distMin, F distMax) const {
        if (!isValid())
            return INFINITY;
        Vector3D_T<F, false> invRayDir = 1.0f / dir;
        Vector3D_T<F, false> tNear = (minP - org) * invRayDir;
        Vector3D_T<F, false> tFar = (maxP - org) * invRayDir;
        Vector3D_T<F, false> near = min(tNear, tFar);
        Vector3D_T<F, false> far = max(tNear, tFar);
        F t0 = std::fmax(std::fmax(near.x, near.y), near.z);
        F t1 = std::fmin(std::fmin(far.x, far.y), far.z);
        t0 = std::fmax(t0, distMin);
        t1 = std::fmin(t1, distMax);
        return t0 <= t1 && t1 > 0.0f;
    }

    CUDA_COMMON_FUNCTION F intersect(
        const Point3D_T<F> &org, const Vector3D_T<F, false> &dir,
        F distMin, F distMax,
        F* u, F* v, bool* isFrontHit) const {
        if (!isValid())
            return INFINITY;
        Vector3D_T<F, false> invRayDir = 1.0f / dir;
        Vector3D_T<F, false> tNear = (minP - org) * invRayDir;
        Vector3D_T<F, false> tFar = (maxP - org) * invRayDir;
        Vector3D_T<F, false> near = min(tNear, tFar);
        Vector3D_T<F, false> far = max(tNear, tFar);
        F t0 = std::fmax(std::fmax(near.x, near.y), near.z);
        F t1 = std::fmin(std::fmin(far.x, far.y), far.z);
        *isFrontHit = t0 >= 0.0f;
        t0 = std::fmax(t0, distMin);
        t1 = std::fmin(t1, distMax);
        if (!(t0 <= t1 && t1 > 0.0f))
            return INFINITY;

        F t = *isFrontHit ? t0 : t1;
        Vector3D_T<F, false> n = -sign(dir) * step(near.yzx(), near) * step(near.zxy(), near);
        if (!*isFrontHit)
            n = -n;

        int32_t faceID = static_cast<int32_t>(dot(abs(n), Vector3D_T<F, false>(2, 4, 8)));
        faceID ^= static_cast<int32_t>(any(n > Vector3D_T<F, false>(0.0f)));

        int32_t faceDim = tzcnt(faceID & ~0b1) - 1;
        int32_t dim0 = (faceDim + 1) % 3;
        int32_t dim1 = (faceDim + 2) % 3;
        Point3D_T<F> p = org + t * dir;
        F min0 = minP[dim0];
        F max0 = maxP[dim0];
        F min1 = minP[dim1];
        F max1 = maxP[dim1];
        *u = std::fmin(std::fmax((p[dim0] - min0) / (max0 - min0), 0.0f), 1.0f)
            + static_cast<F>(faceID);
        *v = std::fmin(std::fmax((p[dim1] - min1) / (max1 - min1), 0.0f), 1.0f);

        return t;
    }

    CUDA_COMMON_FUNCTION Point3D_T<F> restoreHitPoint(
        F u, F v, Vector3D_T<F, true>* normal) const {
        auto faceID = static_cast<uint32_t>(u);
        u = std::fmod(u, 1.0f);

        int32_t faceDim = tzcnt(faceID & ~0b1) - 1;
        bool isPosSide = faceID & 0b1;
        *normal = Vector3D_T<F, true>(0.0f);
        (*normal)[faceDim] = isPosSide ? 1 : -1;

        int32_t dim0 = (faceDim + 1) % 3;
        int32_t dim1 = (faceDim + 2) % 3;
        Point3D_T<F> p;
        p[faceDim] = isPosSide ? maxP[faceDim] : minP[faceDim];
        p[dim0] = lerp(minP[dim0], maxP[dim0], u);
        p[dim1] = lerp(minP[dim1], maxP[dim1], v);

        return p;
    }

    CUDA_COMMON_FUNCTION Vector3D_T<F, true> restoreNormal(F u, F v) const {
        auto faceID = static_cast<uint32_t>(u);
        int32_t faceDim = tzcnt(faceID & ~0b1) - 1;
        bool isPosSide = faceID & 0b1;
        auto normal = Vector3D_T<F, true>(0.0f);
        normal[faceDim] = isPosSide ? 1 : -1;
        return normal;
    }
};



template <typename RealType>
struct CompensatedSum_T {
    RealType result;
    RealType comp;

    CUDA_COMMON_FUNCTION CompensatedSum_T(const RealType &value = RealType(0)) : result(value), comp(0.0) { };

    CUDA_COMMON_FUNCTION CompensatedSum_T &operator=(const RealType &value) {
        result = value;
        comp = 0;
        return *this;
    }

    CUDA_COMMON_FUNCTION CompensatedSum_T &operator+=(const RealType &value) {
        RealType cInput = value - comp;
        RealType sumTemp = result + cInput;
        comp = (sumTemp - result) - cInput;
        result = sumTemp;
        return *this;
    }

    CUDA_COMMON_FUNCTION operator RealType() const { return result; };
};



using Vector2D = Vector2D_T<float>;
using Point2D = Point2D_T<float>;
using Vector3D = Vector3D_T<float, false>;
using Normal3D = Vector3D_T<float, true>;
using Point3D = Point3D_T<float>;
using Vector4D = Vector4D_T<float>;
using Matrix3x3 = Matrix3x3_T<float>;
using Matrix4x4 = Matrix4x4_T<float>;
using Quaternion = Quaternion_T<float>;
using RGB = RGB_T<float>;
using AABB = AABB_T<float>;
using FloatSum = CompensatedSum_T<float>;



CUDA_COMMON_FUNCTION CUDA_INLINE int32_t floatToOrderedInt(float fVal) {
#if defined(__CUDA_ARCH__)
    int32_t iVal = __float_as_int(fVal);
#else
    int32_t iVal = *reinterpret_cast<int32_t*>(&fVal);
#endif
    return (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
}

CUDA_COMMON_FUNCTION CUDA_INLINE float orderedIntToFloat(int32_t iVal) {
    int32_t orgVal = (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
#if defined(__CUDA_ARCH__)
    return __int_as_float(orgVal);
#else
    return *reinterpret_cast<float*>(&orgVal);
#endif
}

struct RGBAsOrderedInt {
    int32_t r, g, b;

    CUDA_COMMON_FUNCTION RGBAsOrderedInt() : r(0), g(0), b(0) {
    }
    CUDA_COMMON_FUNCTION RGBAsOrderedInt(const RGB &v) :
        r(floatToOrderedInt(v.r)), g(floatToOrderedInt(v.g)), b(floatToOrderedInt(v.b)) {
    }

    CUDA_COMMON_FUNCTION RGBAsOrderedInt &operator=(const RGBAsOrderedInt &v) {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGBAsOrderedInt &operator=(const volatile RGBAsOrderedInt &v) {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION volatile RGBAsOrderedInt &operator=(const RGBAsOrderedInt &v) volatile {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION volatile RGBAsOrderedInt &operator=(const volatile RGBAsOrderedInt &v) volatile {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }

    CUDA_COMMON_FUNCTION explicit operator RGB() const {
        return RGB(orderedIntToFloat(r), orderedIntToFloat(g), orderedIntToFloat(b));
    }
    CUDA_COMMON_FUNCTION explicit operator RGB() const volatile {
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
    RGBAsOrderedInt* dst, const RGBAsOrderedInt &v) {
    atomicMin(&dst->r, v.r);
    atomicMin(&dst->g, v.g);
    atomicMin(&dst->b, v.b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMax_RGB(
    RGBAsOrderedInt* dst, const RGBAsOrderedInt &v) {
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
    RGBAsOrderedInt* dst, const RGBAsOrderedInt &v) {
    atomicMin_block(&dst->r, v.r);
    atomicMin_block(&dst->g, v.g);
    atomicMin_block(&dst->b, v.b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMax_RGB_block(
    RGBAsOrderedInt* dst, const RGBAsOrderedInt &v) {
    atomicMax_block(&dst->r, v.r);
    atomicMax_block(&dst->g, v.g);
    atomicMax_block(&dst->b, v.b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicAdd_RGB_block(RGB* dst, const RGB &v) {
    atomicAdd_block(&dst->r, v.r);
    atomicAdd_block(&dst->g, v.g);
    atomicAdd_block(&dst->b, v.b);
}
#endif