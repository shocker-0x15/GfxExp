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
#if !defined(__CUDA_ARCH__)
#   undef CUDA_DEVICE_FUNCTION
#   define CUDA_DEVICE_FUNCTION inline
#endif



#ifdef HP_Platform_Windows_MSVC
#   if defined(__CUDA_ARCH__)
#       define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#   else
void devPrintf(const char* fmt, ...);
#   endif
#else
#   define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#endif

#ifdef ENABLE_ASSERT
#   if defined(__CUDA_ARCH__)
#       define Assert(expr, fmt, ...) do { if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); assert(false); } } while (0)
#   else
#       define Assert(expr, fmt, ...) do { if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); abort(); } } while (0)
#   endif
#else
#   define Assert(expr, fmt, ...)
#endif

#define Assert_ShouldNotBeCalled() Assert(false, "Should not be called!")
#define Assert_NotImplemented() Assert(false, "Not implemented yet!")



template <typename T, size_t size>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr size_t lengthof(const T(&array)[size]) {
    return size;
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



#if !defined(__CUDA_ARCH__) && !defined(__CUDACC__)
// ----------------------------------------------------------------
// JP: CUDAビルトインに対応する型・関数をホスト側で定義しておく。
// EN: Define types and functions on the host corresponding to CUDA built-ins.

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

// END: Define types and functions on the host corresponding to CUDA built-ins.
// ----------------------------------------------------------------
#endif

CUDA_COMMON_FUNCTION CUDA_INLINE float3 getXYZ(const float4 &v) {
    return make_float3(v.x, v.y, v.z);
}

CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const int2 &v0, const int2 &v1) {
    return v0.x == v1.x && v0.y == v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const int2 &v0, const int2 &v1) {
    return v0.x != v1.x || v0.y != v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const int2 &v0, const uint2 &v1) {
    return v0.x == v1.x && v0.y == v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const int2 &v0, const uint2 &v1) {
    return v0.x != v1.x || v0.y != v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator+(const int2 &v0, const uint2 &v1) {
    return make_uint2(v0.x + v1.x, v0.y + v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator*(const int2 &v0, const int2 &v1) {
    return make_int2(v0.x * v1.x, v0.y * v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator/(const int2 &v0, const int2 &v1) {
    return make_int2(v0.x / v1.x, v0.y / v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const int2 &v0, const uint2 &v1) {
    return make_uint2(v0.x / v1.x, v0.y / v1.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const uint2 &v0, const uint2 &v1) {
    return v0.x == v1.x && v0.y == v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const uint2 &v0, const uint2 &v1) {
    return v0.x != v1.x || v0.y != v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const uint2 &v0, const int2 &v1) {
    return v0.x == v1.x && v0.y == v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const uint2 &v0, const int2 &v1) {
    return v0.x != v1.x || v0.y != v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator+(const uint2 &v0, const uint2 &v1) {
    return make_uint2(v0.x + v1.x, v0.y + v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator-(const uint2 &v, uint32_t s) {
    return make_uint2(v.x - s, v.y - s);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator*(const uint2 &v0, const uint2 &v1) {
    return make_uint2(v0.x * v1.x, v0.y * v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const uint2 &v0, const uint2 &v1) {
    return make_uint2(v0.x / v1.x, v0.y / v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const uint2 &v0, const int2 &v1) {
    return make_uint2(v0.x / v1.x, v0.y / v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator%(const uint2 &v0, const uint2 &v1) {
    return make_uint2(v0.x % v1.x, v0.y % v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator+=(uint2 &v, uint32_t s) {
    v.x += s;
    v.x += s;
    return v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator-=(uint2 &v, uint32_t s) {
    v.x -= s;
    v.x -= s;
    return v;
}

CUDA_COMMON_FUNCTION CUDA_INLINE float2 make_float2(float v) {
    return make_float2(v, v);
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const float2 &v0, const float2 &v1) {
    return v0.x == v1.x && v0.y == v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const float2 &v0, const float2 &v1) {
    return v0.x != v1.x || v0.y != v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator-(const float2 &v) {
    return make_float2(-v.x, -v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator+(const float2 &v0, const float2 &v1) {
    return make_float2(v0.x + v1.x, v0.y + v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator-(const float2 &v0, const float2 &v1) {
    return make_float2(v0.x - v1.x, v0.y - v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator*(const float2 &v0, const float2 &v1) {
    return make_float2(v0.x * v1.x, v0.y * v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator*(float s, const float2 &v) {
    return make_float2(s * v.x, s * v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator*(const float2 &v, float s) {
    return make_float2(s * v.x, s * v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator/(const float2 &v, float s) {
    float r = 1 / s;
    return r * v;
}

CUDA_COMMON_FUNCTION CUDA_INLINE float3 make_float3(float v) {
    return make_float3(v, v, v);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 make_float3(const float4 &v) {
    return make_float3(v.x, v.y, v.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const float3 &v0, const float3 &v1) {
    return v0.x == v1.x && v0.y == v1.y && v0.z == v1.z;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const float3 &v0, const float3 &v1) {
    return v0.x != v1.x || v0.y != v1.y || v0.z != v1.z;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator-(const float3 &v) {
    return make_float3(-v.x, -v.y, -v.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator+(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 &operator+=(float3 &v0, const float3 &v1) {
    v0.x += v1.x;
    v0.y += v1.y;
    v0.z += v1.z;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator-(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 &operator-=(float3 &v0, const float3 &v1) {
    v0.x -= v1.x;
    v0.y -= v1.y;
    v0.z -= v1.z;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator*(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator*(float s, const float3 &v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator*(const float3 &v, float s) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 &operator*=(float3 &v0, const float3 &v1) {
    v0.x *= v1.x;
    v0.y *= v1.y;
    v0.z *= v1.z;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 &operator*=(float3 &v, float s) {
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator/(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator/(const float3 &v, float s) {
    float r = 1 / s;
    return r * v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 &operator/=(float3 &v, float s) {
    float r = 1 / s;
    return v *= r;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool allFinite(const float3 &v) {
#if !defined(__CUDA_ARCH__)
    using std::isfinite;
#endif
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

CUDA_COMMON_FUNCTION CUDA_INLINE float4 make_float4(float v) {
    return make_float4(v, v, v, v);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 make_float4(const float3 &v) {
    return make_float4(v.x, v.y, v.z, 0.0f);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 make_float4(const float3 &v, float w) {
    return make_float4(v.x, v.y, v.z, w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const float4 &v0, const float4 &v1) {
    return v0.x == v1.x && v0.y == v1.y && v0.z == v1.z && v0.w == v1.w;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const float4 &v0, const float4 &v1) {
    return v0.x != v1.x || v0.y != v1.y || v0.z != v1.z || v0.w != v1.w;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator-(const float4 &v) {
    return make_float4(-v.x, -v.y, -v.z, -v.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator+(const float4 &v0, const float4 &v1) {
    return make_float4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 &operator+=(float4 &v0, const float4 &v1) {
    v0.x += v1.x;
    v0.y += v1.y;
    v0.z += v1.z;
    v0.w += v1.w;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator-(const float4 &v0, const float4 &v1) {
    return make_float4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 &operator-=(float4 &v0, const float4 &v1) {
    v0.x -= v1.x;
    v0.y -= v1.y;
    v0.z -= v1.z;
    v0.w -= v1.w;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator*(const float4 &v0, const float4 &v1) {
    return make_float4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator*(float s, const float4 &v) {
    return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator*(const float4 &v, float s) {
    return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 &operator*=(float4 &v0, const float4 &v1) {
    v0.x *= v1.x;
    v0.y *= v1.y;
    v0.z *= v1.z;
    v0.w *= v1.w;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 &operator*=(float4 &v, float s) {
    v.x *= s;
    v.y *= s;
    v.z *= s;
    v.w *= s;
    return v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator/(const float4 &v0, const float4 &v1) {
    return make_float4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator/(const float4 &v, float s) {
    float r = 1 / s;
    return r * v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 &operator/=(float4 &v, float s) {
    float r = 1 / s;
    return v *= r;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool allFinite(const float4 &v) {
#if !defined(__CUDA_ARCH__)
    using std::isfinite;
#endif
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z) && isfinite(v.w);
}

CUDA_COMMON_FUNCTION CUDA_INLINE float3 min(const float3 &v0, const float3 &v1) {
    return make_float3(std::fmin(v0.x, v1.x),
                       std::fmin(v0.y, v1.y),
                       std::fmin(v0.z, v1.z));
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 max(const float3 &v0, const float3 &v1) {
    return make_float3(std::fmax(v0.x, v1.x),
                       std::fmax(v0.y, v1.y),
                       std::fmax(v0.z, v1.z));
}
CUDA_COMMON_FUNCTION CUDA_INLINE float dot(const float3 &v0, const float3 &v1) {
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 cross(const float3 &v0, const float3 &v1) {
    return make_float3(v0.y * v1.z - v0.z * v1.y,
                       v0.z * v1.x - v0.x * v1.z,
                       v0.x * v1.y - v0.y * v1.x);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float squaredDistance(const float3 &p0, const float3 &p1) {
    float3 d = p1 - p0;
    return dot(d, d);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float length(const float3 &v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float sqLength(const float3 &v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 normalize(const float3 &v) {
    return v / length(v);
}

CUDA_COMMON_FUNCTION CUDA_INLINE float4 min(const float4 &v0, const float4 &v1) {
    return make_float4(std::fmin(v0.x, v1.x),
                       std::fmin(v0.y, v1.y),
                       std::fmin(v0.z, v1.z),
                       std::fmin(v0.w, v1.w));
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 max(const float4 &v0, const float4 &v1) {
    return make_float4(std::fmax(v0.x, v1.x),
                       std::fmax(v0.y, v1.y),
                       std::fmax(v0.z, v1.z),
                       std::fmax(v0.w, v1.w));
}
CUDA_COMMON_FUNCTION CUDA_INLINE float dot(const float4 &v0, const float4 &v1) {
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z + v0.w * v1.w;
}



CUDA_COMMON_FUNCTION CUDA_INLINE float3 HSVtoRGB(float h, float s, float v) {
    if (s == 0)
        return make_float3(v, v, v);

    h = h - std::floor(h);
    int32_t hi = static_cast<int32_t>(h * 6);
    float f = h * 6 - hi;
    float m = v * (1 - s);
    float n = v * (1 - s * f);
    float k = v * (1 - s * (1 - f));
    if (hi == 0)
        return make_float3(v, k, m);
    else if (hi == 1)
        return make_float3(n, v, m);
    else if (hi == 2)
        return make_float3(m, v, k);
    else if (hi == 3)
        return make_float3(m, n, v);
    else if (hi == 4)
        return make_float3(k, m, v);
    else if (hi == 5)
        return make_float3(v, m, n);
    return make_float3(0, 0, 0);
}

CUDA_COMMON_FUNCTION CUDA_INLINE float simpleToneMap_s(float value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    return 1 - std::exp(-value);
}

CUDA_COMMON_FUNCTION CUDA_INLINE float sRGB_degamma_s(float value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    if (value <= 0.04045f)
        return value / 12.92f;
    return std::pow((value + 0.055f) / 1.055f, 2.4f);
}

CUDA_COMMON_FUNCTION CUDA_INLINE float sRGB_gamma_s(float value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    if (value <= 0.0031308f)
        return 12.92f * value;
    return 1.055f * std::pow(value, 1 / 2.4f) - 0.055f;
}

CUDA_COMMON_FUNCTION CUDA_INLINE float3 sRGB_degamma(const float3 &value) {
    return make_float3(sRGB_degamma_s(value.x),
                       sRGB_degamma_s(value.y),
                       sRGB_degamma_s(value.z));
}

CUDA_COMMON_FUNCTION CUDA_INLINE float sRGB_calcLuminance(const float3 &value) {
    return 0.2126729f * value.x + 0.7151522f * value.y + 0.0721750f * value.z;
}



template <typename RealType>
struct CompensatedSum {
    RealType result;
    RealType comp;

    CUDA_COMMON_FUNCTION CompensatedSum(const RealType &value = RealType(0)) : result(value), comp(0.0) { };

    CUDA_COMMON_FUNCTION CompensatedSum &operator=(const RealType &value) {
        result = value;
        comp = 0;
        return *this;
    }

    CUDA_COMMON_FUNCTION CompensatedSum &operator+=(const RealType &value) {
        RealType cInput = value - comp;
        RealType sumTemp = result + cInput;
        comp = (sumTemp - result) - cInput;
        result = sumTemp;
        return *this;
    }

    CUDA_COMMON_FUNCTION operator RealType() const { return result; };
};

//using FloatSum = float;
using FloatSum = CompensatedSum<float>;



struct Matrix3x3 {
    union {
        struct { float m00, m10, m20; };
        float3 c0;
    };
    union {
        struct { float m01, m11, m21; };
        float3 c1;
    };
    union {
        struct { float m02, m12, m22; };
        float3 c2;
    };

    CUDA_COMMON_FUNCTION /*constexpr*/ Matrix3x3() :
        c0(make_float3(1, 0, 0)),
        c1(make_float3(0, 1, 0)),
        c2(make_float3(0, 0, 1)) { }
    CUDA_COMMON_FUNCTION Matrix3x3(const float array[9]) :
        m00(array[0]), m10(array[1]), m20(array[2]),
        m01(array[3]), m11(array[4]), m21(array[5]),
        m02(array[6]), m12(array[7]), m22(array[8]) { }
    CUDA_COMMON_FUNCTION constexpr Matrix3x3(const float3 &col0, const float3 &col1, const float3 &col2) :
        c0(col0), c1(col1), c2(col2)
    { }

    CUDA_COMMON_FUNCTION Matrix3x3 operator+() const { return *this; }
    CUDA_COMMON_FUNCTION Matrix3x3 operator-() const { return Matrix3x3(-c0, -c1, -c2); }

    CUDA_COMMON_FUNCTION Matrix3x3 operator+(const Matrix3x3 &mat) const { return Matrix3x3(c0 + mat.c0, c1 + mat.c1, c2 + mat.c2); }
    CUDA_COMMON_FUNCTION Matrix3x3 operator-(const Matrix3x3 &mat) const { return Matrix3x3(c0 - mat.c0, c1 - mat.c1, c2 - mat.c2); }
    CUDA_COMMON_FUNCTION Matrix3x3 operator*(const Matrix3x3 &mat) const {
        const float3 r[] = { row(0), row(1), row(2) };
        return Matrix3x3(make_float3(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0)),
                         make_float3(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1)),
                         make_float3(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2)));
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE friend Matrix3x3 operator*(float s, const Matrix3x3 &mat) {
        return Matrix3x3(s * mat.c0, s * mat.c1, s * mat.c2);
    }
    CUDA_COMMON_FUNCTION float3 operator*(const float3 &v) const {
        const float3 r[] = { row(0), row(1), row(2) };
        return make_float3(dot(r[0], v),
                           dot(r[1], v),
                           dot(r[2], v));
    }

    CUDA_COMMON_FUNCTION Matrix3x3 &operator*=(const Matrix3x3 &mat) {
        const float3 r[] = { row(0), row(1), row(2) };
        c0 = make_float3(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0));
        c1 = make_float3(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1));
        c2 = make_float3(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2));
        return *this;
    }

    CUDA_COMMON_FUNCTION float3 row(unsigned int r) const {
        //Assert(r < 3, "\"r\" is out of range [0, 2].");
        switch (r) {
        case 0:
            return make_float3(m00, m01, m02);
        case 1:
            return make_float3(m10, m11, m12);
        case 2:
            return make_float3(m20, m21, m22);
        default:
            return make_float3(0, 0, 0);
        }
    }

    CUDA_COMMON_FUNCTION Matrix3x3 &inverse() {
        float det = 1.0f / (m00 * m11 * m22 + m01 * m12 * m20 + m02 * m10 * m21 -
                            m02 * m11 * m20 - m01 * m10 * m22 - m00 * m12 * m21);
        Matrix3x3 m;
        m.m00 = det * (m11 * m22 - m12 * m21); m.m01 = -det * (m01 * m22 - m02 * m21); m.m02 = det * (m01 * m12 - m02 * m11);
        m.m10 = -det * (m10 * m22 - m12 * m20); m.m11 = det * (m00 * m22 - m02 * m20); m.m12 = -det * (m00 * m12 - m02 * m10);
        m.m20 = det * (m10 * m21 - m11 * m20); m.m21 = -det * (m00 * m21 - m01 * m20); m.m22 = det * (m00 * m11 - m01 * m10);
        *this = m;

        return *this;
    }

    CUDA_COMMON_FUNCTION Matrix3x3 &transpose() {
        float temp;
        temp = m10; m10 = m01; m01 = temp;
        temp = m20; m20 = m02; m02 = temp;
        temp = m21; m21 = m12; m12 = temp;
        return *this;
    }
};

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 transpose(const Matrix3x3 &mat) {
    Matrix3x3 ret = mat;
    return ret.transpose();
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 inverse(const Matrix3x3 &mat) {
    Matrix3x3 ret = mat;
    return ret.inverse();
}

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 scale3x3(const float3 &s) {
    return Matrix3x3(make_float3(s.x, 0, 0),
                     make_float3(0, s.y, 0),
                     make_float3(0, 0, s.z));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 scale3x3(float sx, float sy, float sz) {
    return scale3x3(make_float3(sx, sy, sz));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 scale3x3(float s) {
    return scale3x3(make_float3(s, s, s));
}

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 rotate3x3(float angle, const float3 &axis) {
    Matrix3x3 matrix;
    float3 nAxis = normalize(axis);
    float s = std::sin(angle);
    float c = std::cos(angle);
    float oneMinusC = 1 - c;

    matrix.m00 = nAxis.x * nAxis.x * oneMinusC + c;
    matrix.m10 = nAxis.x * nAxis.y * oneMinusC + nAxis.z * s;
    matrix.m20 = nAxis.z * nAxis.x * oneMinusC - nAxis.y * s;
    matrix.m01 = nAxis.x * nAxis.y * oneMinusC - nAxis.z * s;
    matrix.m11 = nAxis.y * nAxis.y * oneMinusC + c;
    matrix.m21 = nAxis.y * nAxis.z * oneMinusC + nAxis.x * s;
    matrix.m02 = nAxis.z * nAxis.x * oneMinusC + nAxis.y * s;
    matrix.m12 = nAxis.y * nAxis.z * oneMinusC - nAxis.x * s;
    matrix.m22 = nAxis.z * nAxis.z * oneMinusC + c;

    return matrix;
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 rotate3x3(float angle, float ax, float ay, float az) {
    return rotate3x3(angle, make_float3(ax, ay, az));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 rotateX3x3(float angle) { return rotate3x3(angle, make_float3(1, 0, 0)); }
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 rotateY3x3(float angle) { return rotate3x3(angle, make_float3(0, 1, 0)); }
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 rotateZ3x3(float angle) { return rotate3x3(angle, make_float3(0, 0, 1)); }



struct Matrix4x4 {
    union {
        struct { float m00, m10, m20, m30; };
        float4 c0;
    };
    union {
        struct { float m01, m11, m21, m31; };
        float4 c1;
    };
    union {
        struct { float m02, m12, m22, m32; };
        float4 c2;
    };
    union {
        struct { float m03, m13, m23, m33; };
        float4 c3;
    };

    CUDA_COMMON_FUNCTION /*constexpr*/ Matrix4x4() :
        c0(make_float4(1, 0, 0, 0)),
        c1(make_float4(0, 1, 0, 0)),
        c2(make_float4(0, 0, 1, 0)),
        c3(make_float4(0, 0, 0, 1)) { }
    CUDA_COMMON_FUNCTION Matrix4x4(const float array[9]) :
        m00(array[0]), m10(array[1]), m20(array[2]), m30(array[3]),
        m01(array[4]), m11(array[5]), m21(array[6]), m31(array[7]),
        m02(array[8]), m12(array[9]), m22(array[10]), m32(array[11]),
        m03(array[12]), m13(array[13]), m23(array[14]), m33(array[15]) { }
    CUDA_COMMON_FUNCTION constexpr Matrix4x4(const float4 &col0, const float4 &col1, const float4 &col2, const float4 &col3) :
        c0(col0), c1(col1), c2(col2), c3(col3)
    { }
    CUDA_COMMON_FUNCTION Matrix4x4(const Matrix3x3 &mat3x3, const float3 &position) :
        c0(make_float4(mat3x3.c0)), c1(make_float4(mat3x3.c1)), c2(make_float4(mat3x3.c2)), c3(make_float4(position, 1.0f))
    { }

    CUDA_COMMON_FUNCTION Matrix4x4 operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Matrix4x4 operator-() const {
        return Matrix4x4(-c0, -c1, -c2, -c3);
    }

    CUDA_COMMON_FUNCTION Matrix4x4 operator+(const Matrix4x4 &mat) const {
        return Matrix4x4(c0 + mat.c0, c1 + mat.c1, c2 + mat.c2, c3 + mat.c3);
    }
    CUDA_COMMON_FUNCTION Matrix4x4 operator-(const Matrix4x4 &mat) const {
        return Matrix4x4(c0 - mat.c0, c1 - mat.c1, c2 - mat.c2, c3 - mat.c3);
    }
    CUDA_COMMON_FUNCTION Matrix4x4 operator*(const Matrix4x4 &mat) const {
        const float4 r[] = { row(0), row(1), row(2), row(3) };
        return Matrix4x4(make_float4(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0), dot(r[3], mat.c0)),
                         make_float4(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1), dot(r[3], mat.c1)),
                         make_float4(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2), dot(r[3], mat.c2)),
                         make_float4(dot(r[0], mat.c3), dot(r[1], mat.c3), dot(r[2], mat.c3), dot(r[3], mat.c3)));
    }
    CUDA_COMMON_FUNCTION float3 operator*(const float3 &v) const {
        const float4 r[] = { row(0), row(1), row(2), row(3) };
        float4 v4 = make_float4(v, 1.0f);
        return make_float3(dot(r[0], v4),
                           dot(r[1], v4),
                           dot(r[2], v4));
    }
    CUDA_COMMON_FUNCTION float4 operator*(const float4 &v) const {
        const float4 r[] = { row(0), row(1), row(2), row(3) };
        return make_float4(dot(r[0], v),
                           dot(r[1], v),
                           dot(r[2], v),
                           dot(r[3], v));
    }

    CUDA_COMMON_FUNCTION Matrix4x4 &operator*=(const Matrix4x4 &mat) {
        const float4 r[] = { row(0), row(1), row(2), row(3) };
        c0 = make_float4(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0), dot(r[3], mat.c0));
        c1 = make_float4(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1), dot(r[3], mat.c1));
        c2 = make_float4(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2), dot(r[3], mat.c2));
        c3 = make_float4(dot(r[0], mat.c3), dot(r[1], mat.c3), dot(r[2], mat.c3), dot(r[3], mat.c3));
        return *this;
    }

    CUDA_COMMON_FUNCTION float4 row(unsigned int r) const {
        //Assert(r < 3, "\"r\" is out of range [0, 2].");
        switch (r) {
        case 0:
            return make_float4(m00, m01, m02, m03);
        case 1:
            return make_float4(m10, m11, m12, m13);
        case 2:
            return make_float4(m20, m21, m22, m23);
        case 3:
            return make_float4(m30, m31, m32, m33);
        default:
            return make_float4(0, 0, 0, 0);
        }
    }

    CUDA_COMMON_FUNCTION Matrix4x4 &inverse() {
        float inv[] = {
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

        float recDet = 1.0f / (m00 * inv[0] + m10 * inv[4] + m20 * inv[8] + m30 * inv[12]);
        for (int i = 0; i < 16; ++i)
            inv[i] *= recDet;
        *this = Matrix4x4(inv);

        return *this;
    }

    CUDA_COMMON_FUNCTION Matrix4x4 &transpose() {
        float temp;
        temp = m10; m10 = m01; m01 = temp;
        temp = m20; m20 = m02; m02 = temp;
        temp = m21; m21 = m12; m12 = temp;
        temp = m30; m30 = m03; m03 = temp;
        temp = m31; m31 = m13; m13 = temp;
        temp = m32; m32 = m23; m23 = temp;
        return *this;
    }

    CUDA_COMMON_FUNCTION Matrix3x3 getUpperLeftMatrix() const {
        return Matrix3x3(make_float3(c0), make_float3(c1), make_float3(c2));
    }

    CUDA_COMMON_FUNCTION void decompose(float3* retScale, float3* rotation, float3* translation) const {
        Matrix4x4 mat = *this;

        // JP: 移動成分
        // EN: Translation component
        if (translation)
            *translation = make_float3(mat.c3);

        float3 scale = make_float3(
            length(make_float3(mat.c0)),
            length(make_float3(mat.c1)),
            length(make_float3(mat.c2)));

        // JP: 拡大縮小成分
        // EN: Scale component
        if (retScale)
            *retScale = scale;

        if (!rotation)
            return;

        // JP: 上記成分を排除
        // EN: Remove the above components
        mat.c3 = make_float4(0, 0, 0, 1);
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
        float cosBeta = std::cos(rotation->y);

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

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 transpose(const Matrix4x4 &mat) {
    Matrix4x4 ret = mat;
    return ret.transpose();
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 inverse(const Matrix4x4 &mat) {
    Matrix4x4 ret = mat;
    return ret.inverse();
}

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 scale4x4(const float3 &s) {
    return Matrix4x4(make_float4(s.x, 0, 0, 0),
                     make_float4(0, s.y, 0, 0),
                     make_float4(0, 0, s.z, 0),
                     make_float4(0, 0, 0, 1));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 scale4x4(float sx, float sy, float sz) {
    return scale4x4(make_float3(sx, sy, sz));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 scale4x4(float s) {
    return scale4x4(make_float3(s, s, s));
}

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 rotate4x4(float angle, const float3 &axis) {
    Matrix4x4 matrix;
    float3 nAxis = normalize(axis);
    float s = std::sin(angle);
    float c = std::cos(angle);
    float oneMinusC = 1 - c;

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
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 rotate4x4(float angle, float ax, float ay, float az) {
    return rotate4x4(angle, make_float3(ax, ay, az));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 rotateX4x4(float angle) { return rotate4x4(angle, make_float3(1, 0, 0)); }
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 rotateY4x4(float angle) { return rotate4x4(angle, make_float3(0, 1, 0)); }
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 rotateZ4x4(float angle) { return rotate4x4(angle, make_float3(0, 0, 1)); }

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 translate4x4(const float3 &t) {
    return Matrix4x4(make_float4(1, 0, 0, 0),
                     make_float4(0, 1, 0, 0),
                     make_float4(0, 0, 1, 0),
                     make_float4(t, 1.0f));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 translate4x4(float tx, float ty, float tz) {
    return translate4x4(make_float3(tx, ty, tz));
}



struct Quaternion {
    union {
        float3 v;
        struct {
            float x;
            float y;
            float z;
        };
    };
    float w;

    CUDA_COMMON_FUNCTION constexpr Quaternion() : v(), w(1) {}
    CUDA_COMMON_FUNCTION /*constexpr*/ Quaternion(float xx, float yy, float zz, float ww) : v(make_float3(xx, yy, zz)), w(ww) {}
    CUDA_COMMON_FUNCTION constexpr Quaternion(const float3 &vv, float ww) : v(vv), w(ww) {}

    CUDA_COMMON_FUNCTION bool operator==(const Quaternion &q) const {
        return v == q.v && w == q.w;
    }
    CUDA_COMMON_FUNCTION bool operator!=(const Quaternion &q) const {
        return v != q.v || w != q.w;
    }

    CUDA_COMMON_FUNCTION Quaternion operator+() const { return *this; }
    CUDA_COMMON_FUNCTION Quaternion operator-() const { return Quaternion(-v, -w); }

    CUDA_COMMON_FUNCTION Quaternion operator+(const Quaternion &q) const {
        return Quaternion(v + q.v, w + q.w);
    }
    CUDA_COMMON_FUNCTION Quaternion operator-(const Quaternion &q) const {
        return Quaternion(v - q.v, w - q.w);
    }
    CUDA_COMMON_FUNCTION Quaternion operator*(const Quaternion &q) const {
        return Quaternion(cross(v, q.v) + w * q.v + q.w * v, w * q.w - dot(v, q.v));
    }
    CUDA_COMMON_FUNCTION Quaternion operator*(float s) const { return Quaternion(v * s, w * s); }
    CUDA_COMMON_FUNCTION Quaternion operator/(float s) const { float r = 1 / s; return *this * r; }
    CUDA_COMMON_FUNCTION CUDA_INLINE friend Quaternion operator*(float s, const Quaternion &q) { return q * s; }

    CUDA_COMMON_FUNCTION void toEulerAngles(float* roll, float* pitch, float* yaw) const {
        float xx = x * x;
        float xy = x * y;
        float xz = x * z;
        float xw = x * w;
        float yy = y * y;
        float yz = y * z;
        float yw = y * w;
        float zz = z * z;
        float zw = z * w;
        float ww = w * w;
        *pitch = std::atan2(2 * (xw + yz), ww - xx - yy + zz); // around x
        *yaw = std::asin(std::fmin(std::fmax(2.0f * (yw - xz), -1.0f), 1.0f)); // around y
        *roll = std::atan2(2 * (zw + xy), ww + xx - yy - zz); // around z
    }
    CUDA_COMMON_FUNCTION Matrix3x3 toMatrix3x3() const {
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, zx = z * x;
        float xw = x * w, yw = y * w, zw = z * w;
        return Matrix3x3(make_float3(1 - 2 * (yy + zz), 2 * (xy + zw), 2 * (zx - yw)),
                         make_float3(2 * (xy - zw), 1 - 2 * (xx + zz), 2 * (yz + xw)),
                         make_float3(2 * (zx + yw), 2 * (yz - xw), 1 - 2 * (xx + yy)));
    }

    CUDA_COMMON_FUNCTION bool allFinite() const {
#if defined(__CUDA_ARCH__)
        return ::allFinite(v) && isfinite(w);
#else
        return ::allFinite(v) && std::isfinite(w);
#endif
    }
};

CUDA_COMMON_FUNCTION CUDA_INLINE bool allFinite(const Quaternion &q) {
    return q.allFinite();
}

CUDA_COMMON_FUNCTION CUDA_INLINE float dot(const Quaternion &q0, const Quaternion &q1) {
    return dot(q0.v, q1.v) + q0.w * q1.w;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion normalize(const Quaternion &q) {
    return q / std::sqrt(dot(q, q));
}

CUDA_COMMON_FUNCTION CUDA_INLINE static Quaternion qRotate(float angle, const float3 &axis) {
    float ha = angle / 2;
    float s = std::sin(ha), c = std::cos(ha);
    return Quaternion(s * normalize(axis), c);
}
CUDA_COMMON_FUNCTION CUDA_INLINE static Quaternion qRotate(float angle, float ax, float ay, float az) {
    return qRotate(angle, make_float3(ax, ay, az));
}
CUDA_COMMON_FUNCTION CUDA_INLINE static Quaternion qRotateX(float angle) { return qRotate(angle, make_float3(1, 0, 0)); }
CUDA_COMMON_FUNCTION CUDA_INLINE static Quaternion qRotateY(float angle) { return qRotate(angle, make_float3(0, 1, 0)); }
CUDA_COMMON_FUNCTION CUDA_INLINE static Quaternion qRotateZ(float angle) { return qRotate(angle, make_float3(0, 0, 1)); }

CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion qFromEulerAngles(float roll, float pitch, float yaw) {
    return qRotateZ(roll) * qRotateY(yaw) * qRotateX(pitch);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion Slerp(float t, const Quaternion &q0, const Quaternion &q1) {
    float cosTheta = dot(q0, q1);
    if (cosTheta > 0.9995f)
        return normalize((1 - t) * q0 + t * q1);
    else {
        float theta = std::acos(std::fmin(std::fmax(cosTheta, -1.0f), 1.0f));
        float thetap = theta * t;
        Quaternion qPerp = normalize(q1 - q0 * cosTheta);
        float sinThetaP, cosThetaP;
        sinThetaP = std::sin(thetap);
        cosThetaP = std::cos(thetap);
        //sincos(thetap, &sinThetaP, &cosThetaP);
        return q0 * cosThetaP + qPerp * sinThetaP;
    }
}



struct AABB {
    float3 minP;
    float3 maxP;

    CUDA_COMMON_FUNCTION AABB() : minP(make_float3(INFINITY)), maxP(make_float3(-INFINITY)) {}

    CUDA_COMMON_FUNCTION AABB &unify(const float3 &p) {
        minP = min(minP, p);
        maxP = max(maxP, p);
        return *this;
    }
    CUDA_COMMON_FUNCTION AABB &unify(const AABB &bb) {
        minP = min(minP, bb.minP);
        maxP = max(maxP, bb.maxP);
        return *this;
    }

    CUDA_COMMON_FUNCTION AABB &dilate(float scale) {
        float3 d = maxP - minP;
        minP -= 0.5f * (scale - 1) * d;
        maxP += 0.5f * (scale - 1) * d;
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE friend AABB operator*(const Matrix4x4 &mat, const AABB &aabb) {
        AABB ret;
        ret
            .unify(mat * make_float3(aabb.minP.x, aabb.minP.y, aabb.minP.z))
            .unify(mat * make_float3(aabb.maxP.x, aabb.minP.y, aabb.minP.z))
            .unify(mat * make_float3(aabb.minP.x, aabb.maxP.y, aabb.minP.z))
            .unify(mat * make_float3(aabb.maxP.x, aabb.maxP.y, aabb.minP.z))
            .unify(mat * make_float3(aabb.minP.x, aabb.minP.y, aabb.maxP.z))
            .unify(mat * make_float3(aabb.maxP.x, aabb.minP.y, aabb.maxP.z))
            .unify(mat * make_float3(aabb.minP.x, aabb.maxP.y, aabb.maxP.z))
            .unify(mat * make_float3(aabb.maxP.x, aabb.maxP.y, aabb.maxP.z));
        return ret;
    }
};



// JP: Callable Programや関数ポインターによる動的な関数呼び出しを
//     無くした場合の性能を見たい場合にこのマクロを有効化する。
// EN: Enable this switch when you want to see performance
//     without dynamic function calls by callable programs or function pointers.
//#define USE_HARD_CODED_BSDF_FUNCTIONS
#define HARD_CODED_BSDF DiffuseAndSpecularBRDF
//#define HARD_CODED_BSDF SimplePBR_BRDF

// Use Walker's alias method with initialization by Vose's algorithm
#define USE_WALKER_ALIAS_METHOD

#define PROCESS_DYNAMIC_FUNCTIONS \
    PROCESS_DYNAMIC_FUNCTION(readModifiedNormalFromNormalMap), \
    PROCESS_DYNAMIC_FUNCTION(readModifiedNormalFromNormalMap2ch), \
    PROCESS_DYNAMIC_FUNCTION(readModifiedNormalFromHeightMap), \
    PROCESS_DYNAMIC_FUNCTION(setupLambertBRDF), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_getSurfaceParameters), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_sampleThroughput), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluate), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluatePDF), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluateDHReflectanceEstimate), \
    PROCESS_DYNAMIC_FUNCTION(setupDiffuseAndSpecularBRDF), \
    PROCESS_DYNAMIC_FUNCTION(setupSimplePBR_BRDF), \
    PROCESS_DYNAMIC_FUNCTION(DiffuseAndSpecularBRDF_getSurfaceParameters), \
    PROCESS_DYNAMIC_FUNCTION(DiffuseAndSpecularBRDF_sampleThroughput), \
    PROCESS_DYNAMIC_FUNCTION(DiffuseAndSpecularBRDF_evaluate), \
    PROCESS_DYNAMIC_FUNCTION(DiffuseAndSpecularBRDF_evaluatePDF), \
    PROCESS_DYNAMIC_FUNCTION(DiffuseAndSpecularBRDF_evaluateDHReflectanceEstimate),

enum CallableProgram {
#define PROCESS_DYNAMIC_FUNCTION(Func) CallableProgram_ ## Func
    PROCESS_DYNAMIC_FUNCTIONS
#undef PROCESS_DYNAMIC_FUNCTION
    NumCallablePrograms
};

constexpr const char* callableProgramEntryPoints[] = {
#define PROCESS_DYNAMIC_FUNCTION(Func) RT_DC_NAME_STR(#Func)
    PROCESS_DYNAMIC_FUNCTIONS
#undef PROCESS_DYNAMIC_FUNCTION
};

#define CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR(name) "ptr_" #name
constexpr const char* callableProgramPointerNames[] = {
#define PROCESS_DYNAMIC_FUNCTION(Func) CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR(Func)
    PROCESS_DYNAMIC_FUNCTIONS
#undef PROCESS_DYNAMIC_FUNCTION
};
#undef CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR

#undef PROCESS_DYNAMIC_FUNCTIONS

#if (defined(__CUDA_ARCH__) && defined(PURE_CUDA)) || defined(OPTIXU_Platform_CodeCompletion)
CUDA_CONSTANT_MEM void* c_callableToPointerMap[NumCallablePrograms];
#endif

#if defined(PURE_CUDA)
#   define CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(name) \
        extern "C" CUDA_DEVICE_MEM auto ptr_ ## name = RT_DC_NAME(name)
#else
#   define CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(name)
#endif



namespace shared {
    template <typename FuncType>
    class DynamicFunction;

    template <typename ReturnType, typename... ArgTypes>
    class DynamicFunction<ReturnType(ArgTypes...)> {
        using Signature = ReturnType (*)(ArgTypes...);
        optixu::DirectCallableProgramID<ReturnType(ArgTypes...)> m_callableHandle;

    public:
        CUDA_COMMON_FUNCTION DynamicFunction() {}
        CUDA_COMMON_FUNCTION DynamicFunction(uint32_t sbtIndex) : m_callableHandle(sbtIndex) {}

        CUDA_COMMON_FUNCTION explicit operator uint32_t() const { return static_cast<uint32_t>(m_callableHandle); }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION ReturnType operator()(const ArgTypes &... args) const {
#   if defined(PURE_CUDA)
            void* ptr = c_callableToPointerMap[static_cast<uint32_t>(m_callableHandle)];
            auto func = reinterpret_cast<Signature>(ptr);
            return func(args...);
#   else
            return m_callableHandle(args...);
#   endif
        }
#endif
    };



    class PCG32RNG {
        uint64_t state;

    public:
        CUDA_COMMON_FUNCTION PCG32RNG() {}

        CUDA_COMMON_FUNCTION void setState(uint64_t _state) { state = _state; }

        CUDA_COMMON_FUNCTION uint32_t operator()() {
            uint64_t oldstate = state;
            // Advance internal state
            state = oldstate * 6364136223846793005ULL + 1;
            // Calculate output function (XSH RR), uses old state for max ILP
            uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-static_cast<int32_t>(rot)) & 31));
        }

        CUDA_COMMON_FUNCTION float getFloat0cTo1o() {
            uint32_t fractionBits = ((*this)() >> 9) | 0x3f800000;
            return *(float*)&fractionBits - 1.0f;
        }
    };



    CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t mapPrimarySampleToDiscrete(
        float u01, uint32_t numValues, float* uRemapped = nullptr) {
#if defined(__CUDA_ARCH__)
        uint32_t idx = min(static_cast<uint32_t>(u01 * numValues), numValues - 1);
#else
        uint32_t idx = std::min(static_cast<uint32_t>(u01 * numValues), numValues - 1);
#endif
        if (uRemapped)
            *uRemapped = u01 * numValues - idx;
        return idx;
    }



    template <typename RealType>
    struct AliasTableEntry {
        uint32_t secondIndex;
        RealType probToPickFirst;

        CUDA_COMMON_FUNCTION AliasTableEntry() {}
        CUDA_COMMON_FUNCTION AliasTableEntry(uint32_t _secondIndex, RealType _probToPickFirst) :
            secondIndex(_secondIndex), probToPickFirst(_probToPickFirst) {}
    };

    template <typename RealType>
    struct AliasValueMap {
        RealType scaleForFirst;
        RealType scaleForSecond;
        RealType offsetForSecond;
    };



    template <typename RealType>
    class DiscreteDistribution1DTemplate {
        const RealType* m_PMF;
#if defined(USE_WALKER_ALIAS_METHOD)
        const AliasTableEntry<RealType>* m_aliasTable;
        const AliasValueMap<RealType>* m_valueMaps;
#else
        const RealType* m_CDF;
#endif
        RealType m_integral;
        uint32_t m_numValues;

    public:
#if defined(USE_WALKER_ALIAS_METHOD)
        DiscreteDistribution1DTemplate(
            const RealType* PMF, const AliasTableEntry<RealType>* aliasTable, const AliasValueMap<RealType>* valueMaps,
            RealType integral, uint32_t numValues) :
            m_PMF(PMF), m_aliasTable(aliasTable), m_valueMaps(valueMaps),
            m_integral(integral), m_numValues(numValues) {}
#else
        DiscreteDistribution1DTemplate(
            const RealType* PMF, const RealType* CDF, RealType integral, uint32_t numValues) :
            m_PMF(PMF), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {}
#endif

        CUDA_COMMON_FUNCTION DiscreteDistribution1DTemplate() {}

        CUDA_COMMON_FUNCTION uint32_t sample(RealType u, RealType* prob) const {
            Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
#if defined(USE_WALKER_ALIAS_METHOD)
            uint32_t idx = mapPrimarySampleToDiscrete(u, m_numValues, &u);
            const AliasTableEntry<RealType> &entry = m_aliasTable[idx];
            if (u >= entry.probToPickFirst)
                idx = entry.secondIndex;
#else
            int idx = 0;
            for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
                if (idx + d >= m_numValues)
                    continue;
                if (m_CDF[idx + d] <= u)
                    idx += d;
            }
#endif
            Assert(idx < m_numValues, "Invalid Index!: %d", idx);
            *prob = m_PMF[idx];
            return idx;
        }

        CUDA_COMMON_FUNCTION uint32_t sample(RealType u, RealType* prob, RealType* remapped) const {
            Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
#if defined(USE_WALKER_ALIAS_METHOD)
            uint32_t idx = mapPrimarySampleToDiscrete(u, m_numValues, &u);
            const AliasTableEntry<RealType> &entry = m_aliasTable[idx];
            const AliasValueMap<RealType> &valueMap = m_valueMaps[idx];
            if (u < entry.probToPickFirst) {
                *remapped = valueMap.scaleForFirst * u;
            }
            else {
                idx = entry.secondIndex;
                *remapped = valueMap.scaleForSecond * u + valueMap.offsetForSecond;
            }
#else
            int idx = 0;
            for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
                if (idx + d >= m_numValues)
                    continue;
                if (m_CDF[idx + d] <= u)
                    idx += d;
            }
            Assert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
            *remapped = (u - m_CDF[idx]) / (m_CDF[idx + 1] - m_CDF[idx]);
#endif
            *prob = m_PMF[idx];
            Assert(isfinite(*remapped), "Remapped value is not a finite value %g.",
                   *remapped);
            return idx;
        }

        CUDA_COMMON_FUNCTION RealType evaluatePMF(uint32_t idx) const {
            Assert(idx < m_numValues, "\"idx\" is out of range [0, %u)", m_numValues);
            return m_PMF[idx];
        }

        CUDA_COMMON_FUNCTION RealType integral() const { return m_integral; }

        CUDA_COMMON_FUNCTION uint32_t numValues() const { return m_numValues; }
    };

    using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;



    template <typename RealType>
    class RegularConstantContinuousDistribution1DTemplate {
        const RealType* m_PDF;
#if defined(USE_WALKER_ALIAS_METHOD)
        const AliasTableEntry<RealType>* m_aliasTable;
        const AliasValueMap<RealType>* m_valueMaps;
#else
        const RealType* m_CDF;
#endif
        RealType m_integral;
        uint32_t m_numValues;

    public:
#if defined(USE_WALKER_ALIAS_METHOD)
        RegularConstantContinuousDistribution1DTemplate(
            const RealType* PDF, const AliasTableEntry<RealType>* aliasTable, const AliasValueMap<RealType>* valueMaps,
            RealType integral, uint32_t numValues) :
            m_PDF(PDF), m_aliasTable(aliasTable), m_valueMaps(valueMaps),
            m_integral(integral), m_numValues(numValues) {}
#else
        RegularConstantContinuousDistribution1DTemplate(
            const RealType* PDF, const RealType* CDF, RealType integral, uint32_t numValues) :
            m_PDF(PDF), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {}
#endif

        CUDA_COMMON_FUNCTION RegularConstantContinuousDistribution1DTemplate() {}

        CUDA_COMMON_FUNCTION RealType sample(RealType u, RealType* probDensity) const {
            Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
#if defined(USE_WALKER_ALIAS_METHOD)
            uint32_t idx = mapPrimarySampleToDiscrete(u, m_numValues, &u);
            const AliasTableEntry<RealType> &entry = m_aliasTable[idx];
            const AliasValueMap<RealType> &valueMap = m_valueMaps[idx];
            RealType t;
            if (u < entry.probToPickFirst) {
                t = valueMap.scaleForFirst * u;
            }
            else {
                idx = entry.secondIndex;
                t = valueMap.scaleForSecond * u + valueMap.offsetForSecond;
            }
#else
            int idx = 0;
            for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
                if (idx + d >= m_numValues)
                    continue;
                if (m_CDF[idx + d] <= u)
                    idx += d;
            }
            Assert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
            RealType t = (u - m_CDF[idx]) / (m_CDF[idx + 1] - m_CDF[idx]);
#endif
            *probDensity = m_PDF[idx];
            return (idx + t) / m_numValues;
        }
        CUDA_COMMON_FUNCTION RealType evaluatePDF(RealType smp) const {
            Assert(smp >= 0 && smp < 1.0, "\"smp\": %g is out of range [0, 1).", smp);
            int32_t idx = min(m_numValues - 1, static_cast<uint32_t>(smp * m_numValues));
            return m_PDF[idx];
        }
        CUDA_COMMON_FUNCTION RealType integral() const { return m_integral; }

        CUDA_COMMON_FUNCTION uint32_t numValues() const { return m_numValues; }
    };

    using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float>;



    template <typename RealType>
    class RegularConstantContinuousDistribution2DTemplate {
        const RegularConstantContinuousDistribution1DTemplate<RealType>* m_1DDists;
        RegularConstantContinuousDistribution1DTemplate<RealType> m_top1DDist;

    public:
        RegularConstantContinuousDistribution2DTemplate(
            const RegularConstantContinuousDistribution1DTemplate<RealType>* _1DDists,
            const RegularConstantContinuousDistribution1DTemplate<RealType> &top1DDist) :
            m_1DDists(_1DDists), m_top1DDist(top1DDist) {}

        CUDA_COMMON_FUNCTION RegularConstantContinuousDistribution2DTemplate() {}

        CUDA_COMMON_FUNCTION void sample(RealType u0, RealType u1, RealType* d0, RealType* d1, RealType* probDensity) const {
            RealType topPDF;
            *d1 = m_top1DDist.sample(u1, &topPDF);
            uint32_t idx1D = mapPrimarySampleToDiscrete(*d1, m_top1DDist.numValues());
            *d0 = m_1DDists[idx1D].sample(u0, probDensity);
            *probDensity *= topPDF;
        }
        CUDA_COMMON_FUNCTION RealType evaluatePDF(RealType d0, RealType d1) const {
            uint32_t idx1D = mapPrimarySampleToDiscrete(d1, m_top1DDist.numValues());
            return m_top1DDist.evaluatePDF(d1) * m_1DDists[idx1D].evaluatePDF(d0);
        }
    };

    using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float>;



    // Reference:
    // Long-Period Hash Functions for Procedural Texturing
    // combined permutation table of the hash function of period 739,024 = lcm(11, 13, 16, 17, 19)
#if defined(__CUDA_ARCH__)
    CUDA_CONSTANT_MEM
#endif
    static uint8_t PermutationTable[] = {
        // table 0: 11 numbers
        0, 10, 2, 7, 3, 5, 6, 4, 8, 1, 9,
        // table 1: 13 numbers
        5, 11, 6, 8, 1, 10, 12, 9, 3, 7, 0, 4, 2,
        // table 2: 16 numbers = the range of the hash function required by Perlin noise.
        13, 10, 11, 5, 6, 9, 4, 3, 8, 7, 14, 2, 0, 1, 15, 12,
        // table 3: 17 numbers
        1, 13, 5, 14, 12, 3, 6, 16, 0, 8, 9, 2, 11, 4, 15, 7, 10,
        // table 4: 19 numbers
        10, 6, 5, 8, 15, 0, 17, 7, 14, 18, 13, 16, 2, 9, 12, 1, 11, 4, 3,
        //// table 6: 23 numbers
        //20, 21, 4, 5, 0, 18, 14, 2, 6, 22, 10, 17, 3, 7, 8, 16, 19, 11, 9, 13, 1, 15, 12
    };

    // References
    // Improving Noise
    // This code is based on the web site: adrian's soapbox
    // http://flafla2.github.io/2014/08/09/perlinnoise.html
    class PerlinNoise3D {
        int32_t m_repeat;

        CUDA_COMMON_FUNCTION CUDA_INLINE static uint8_t hash(int32_t x, int32_t y, int32_t z) {
            uint32_t sum = 0;
            sum += PermutationTable[0 + (PermutationTable[0 + (PermutationTable[0 + x % 11] + y) % 11] + z) % 11];
            sum += PermutationTable[11 + (PermutationTable[11 + (PermutationTable[11 + x % 13] + y) % 13] + z) % 13];
            sum += PermutationTable[24 + (PermutationTable[24 + (PermutationTable[24 + x % 16] + y) % 16] + z) % 16];
            sum += PermutationTable[40 + (PermutationTable[40 + (PermutationTable[40 + x % 17] + y) % 17] + z) % 17];
            sum += PermutationTable[57 + (PermutationTable[57 + (PermutationTable[57 + x % 19] + y) % 19] + z) % 19];
            return sum % 16;
        }

        CUDA_COMMON_FUNCTION CUDA_INLINE static float gradient(uint32_t hash, float xu, float yu, float zu) {
            switch (hash & 0xF) {
                // Dot products with 12 vectors defined by the directions from the center of a cube to its edges.
            case 0x0: return  xu + yu; // ( 1,  1,  0)
            case 0x1: return -xu + yu; // (-1,  1,  0)
            case 0x2: return  xu - yu; // ( 1, -1,  0)
            case 0x3: return -xu - yu; // (-1, -1,  0)
            case 0x4: return  xu + zu; // ( 1,  0,  1)
            case 0x5: return -xu + zu; // (-1,  0,  1)
            case 0x6: return  xu - zu; // ( 1,  0, -1)
            case 0x7: return -xu - zu; // (-1,  0, -1)
            case 0x8: return  yu + zu; // ( 0,  1,  1)
            case 0x9: return -yu + zu; // ( 0, -1,  1)
            case 0xA: return  yu - zu; // ( 0,  1, -1)
            case 0xB: return -yu - zu; // ( 0, -1, -1)

                // To avoid the cost of dividing by 12, we pad to 16 gradient directions.
                // These form a regular tetrahedron, so adding them redundantly introduces no visual bias in the texture.
            case 0xC: return  xu + yu; // ( 1,  1,  0)
            case 0xD: return -yu + zu; // ( 0, -1,  1)
            case 0xE: return -xu + yu; // (-1 , 1,  0)
            case 0xF: return -yu - zu; // ( 0, -1, -1)

            default: return 0; // never happens
            }
        }

    public:
        CUDA_COMMON_FUNCTION PerlinNoise3D(int32_t repeat) : m_repeat(repeat) {}

        CUDA_COMMON_FUNCTION float evaluate(const float3 &p, float frequency) const {
            float x = frequency * p.x;
            float y = frequency * p.y;
            float z = frequency * p.z;
            const uint32_t repeat = static_cast<uint32_t>(m_repeat * frequency);

            // If we have any repeat on, change the coordinates to their "local" repetitions.
            if (repeat > 0) {
#if defined(__CUDA_ARCH__)
                x = fmodf(x, repeat);
                y = fmodf(y, repeat);
                z = fmodf(z, repeat);
#else
                x = std::fmod(x, static_cast<float>(repeat));
                y = std::fmod(y, static_cast<float>(repeat));
                z = std::fmod(z, static_cast<float>(repeat));
#endif
                if (x < 0)
                    x += repeat;
                if (y < 0)
                    y += repeat;
                if (z < 0)
                    z += repeat;
            }

            // Calculate the "unit cube" that the point asked will be located in.
            // The left bound is ( |_x_|,|_y_|,|_z_| ) and the right bound is that plus 1.
#if defined(__CUDA_ARCH__)
            int32_t xi = floorf(x);
            int32_t yi = floorf(y);
            int32_t zi = floorf(z);
#else
            int32_t xi = static_cast<int32_t>(std::floor(x));
            int32_t yi = static_cast<int32_t>(std::floor(y));
            int32_t zi = static_cast<int32_t>(std::floor(z));
#endif

            const auto fade = [](float t) {
                // Fade function as defined by Ken Perlin.
                // This eases coordinate values so that they will "ease" towards integral values.
                // This ends up smoothing the final output.
                // 6t^5 - 15t^4 + 10t^3
                return t * t * t * (t * (t * 6 - 15) + 10);
            };

            // Next we calculate the location (from 0.0 to 1.0) in that cube.
            // We also fade the location to smooth the result.
            float xu = x - xi;
            float yu = y - yi;
            float zu = z - zi;
            float u = fade(xu);
            float v = fade(yu);
            float w = fade(zu);

            const auto inc = [this, repeat](int32_t num) {
                ++num;
                if (repeat > 0)
                    num %= repeat;
                return num;
            };

            uint8_t lll, llu, lul, luu, ull, ulu, uul, uuu;
            lll = hash(xi, yi, zi);
            ull = hash(inc(xi), yi, zi);
            lul = hash(xi, inc(yi), zi);
            uul = hash(inc(xi), inc(yi), zi);
            llu = hash(xi, yi, inc(zi));
            ulu = hash(inc(xi), yi, inc(zi));
            luu = hash(xi, inc(yi), inc(zi));
            uuu = hash(inc(xi), inc(yi), inc(zi));

            const auto lerp = [](float v0, float v1, float t) {
                return v0 * (1 - t) + v1 * t;
            };

            // The gradient function calculates the dot product between a pseudorandom gradient vector and 
            // the vector from the input coordinate to the 8 surrounding points in its unit cube.
            // This is all then lerped together as a sort of weighted average based on the faded (u,v,w) values we made earlier.
            float _llValue = lerp(gradient(lll, xu, yu, zu), gradient(ull, xu - 1, yu, zu), u);
            float _ulValue = lerp(gradient(lul, xu, yu - 1, zu), gradient(uul, xu - 1, yu - 1, zu), u);
            float __lValue = lerp(_llValue, _ulValue, v);

            float _luValue = lerp(gradient(llu, xu, yu, zu - 1), gradient(ulu, xu - 1, yu, zu - 1), u);
            float _uuValue = lerp(gradient(luu, xu, yu - 1, zu - 1), gradient(uuu, xu - 1, yu - 1, zu - 1), u);
            float __uValue = lerp(_luValue, _uuValue, v);

            float ret = lerp(__lValue, __uValue, w);
            return ret;
        }
    };

    class MultiOctavePerlinNoise3D {
        PerlinNoise3D m_primaryNoiseGen;
        uint32_t m_numOctaves;
        float m_initialFrequency;
        float m_initialAmplitude;
        float m_frequencyMultiplier;
        float m_persistence;
        float m_supValue;

    public:
        CUDA_COMMON_FUNCTION MultiOctavePerlinNoise3D(uint32_t numOctaves, float initialFrequency, float supValueOrInitialAmplitude, bool supSpecified,
                                                      float frequencyMultiplier, float persistence, uint32_t repeat) :
            m_primaryNoiseGen(repeat),
            m_numOctaves(numOctaves),
            m_initialFrequency(initialFrequency),
            m_frequencyMultiplier(frequencyMultiplier), m_persistence(persistence) {
            if (supSpecified) {
                float amplitude = 1.0f;
                float tempSupValue = 0;
                for (int i = 0; i < static_cast<int32_t>(m_numOctaves); ++i) {
                    tempSupValue += amplitude;
                    amplitude *= m_persistence;
                }
                m_initialAmplitude = supValueOrInitialAmplitude / tempSupValue;
                m_supValue = supValueOrInitialAmplitude;
            }
            else {
                m_initialAmplitude = supValueOrInitialAmplitude;
                float amplitude = m_initialAmplitude;
                m_supValue = 0;
                for (int i = 0; i < static_cast<int32_t>(m_numOctaves); ++i) {
                    m_supValue += amplitude;
                    amplitude *= m_persistence;
                }
            }
        }

        CUDA_COMMON_FUNCTION float evaluate(const float3 &p) const {
            float total = 0;
            float frequency = m_initialFrequency;
            float amplitude = m_initialAmplitude;
            for (int i = 0; i < static_cast<int32_t>(m_numOctaves); ++i) {
                total += m_primaryNoiseGen.evaluate(p, frequency) * amplitude;

                amplitude *= m_persistence;
                frequency *= m_frequencyMultiplier;
            }

            return total;
        }
    };



    using ReadModifiedNormal = DynamicFunction<
        float3(CUtexObject texture, const float2 &texCoord, uint32_t texDim)>;

    using BSDFGetSurfaceParameters = DynamicFunction<
        void(const uint32_t* data, float3* diffuseReflectance, float3* specularReflectance, float* roughness)>;
    using BSDFSampleThroughput = DynamicFunction<
        float3(const uint32_t* data, const float3 &vGiven, float uDir0, float uDir1,
               float3* vSampled, float* dirPDensity)>;
    using BSDFEvaluate = DynamicFunction<
        float3(const uint32_t* data, const float3 &vGiven, const float3 &vSampled)>;
    using BSDFEvaluatePDF = DynamicFunction<
        float(const uint32_t* data, const float3 &vGiven, const float3 &vSampled)>;
    using BSDFEvaluateDHReflectanceEstimate = DynamicFunction<
        float3(const uint32_t* data, const float3 &vGiven)>;



    struct Vertex {
        float3 position;
        float3 normal;
        float3 texCoord0Dir;
        float2 texCoord;
    };

    struct Triangle {
        uint32_t index0, index1, index2;
    };

    struct MaterialData;

    using SetupBSDFBody = DynamicFunction<
        void(const MaterialData &matData, const float2 &texCoord, uint32_t* bodyData)>;

    struct MaterialData {
        union {
            struct {
                CUtexObject reflectance;
            } asLambert;
            struct {
                CUtexObject diffuse;
                CUtexObject specular;
                CUtexObject smoothness;
            } asDiffuseAndSpecular;
            struct {
                CUtexObject baseColor_opacity;
                CUtexObject occlusion_roughness_metallic;
            } asSimplePBR;
        };
        CUtexObject normal;
        CUtexObject emittance;
        union {
            struct {
                unsigned int normalWidth : 16;
                unsigned int normalHeight : 16;
            };
            uint32_t normalDimension;
        };

        ReadModifiedNormal readModifiedNormal;

        SetupBSDFBody setupBSDFBody;
        BSDFGetSurfaceParameters bsdfGetSurfaceParameters;
        BSDFSampleThroughput bsdfSampleThroughput;
        BSDFEvaluate bsdfEvaluate;
        BSDFEvaluatePDF bsdfEvaluatePDF;
        BSDFEvaluateDHReflectanceEstimate bsdfEvaluateDHReflectanceEstimate;
    };

    struct GeometryInstanceData {
        const Vertex* vertexBuffer;
        const Triangle* triangleBuffer;
        DiscreteDistribution1D emitterPrimDist;
        uint32_t materialSlot;
        uint32_t geomInstSlot;
    };

    struct InstanceData {
        Matrix4x4 transform;
        Matrix4x4 prevTransform;
        Matrix3x3 normalMatrix;

        const uint32_t* geomInstSlots;
        uint32_t numGeomInsts;
        DiscreteDistribution1D lightGeomInstDist;
    };
}
