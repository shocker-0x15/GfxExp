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



template <typename T>
concept Number = std::integral<T> || std::floating_point<T>;

template <typename T>
concept Number32bit =
    std::is_same_v<T, int32_t> ||
    std::is_same_v<T, uint32_t> ||
    std::is_same_v<T, float>;

using FloatType = float;



template <typename T, size_t size>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr size_t lengthof(const T (&array)[size]) {
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
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator*(uint32_t a, const int2 &b) {
    return make_int2(a * b.x, a * b.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator*(const int2 &a, uint32_t b) {
    return make_int2(a.x * b, a.y * b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator*=(int2 &a, const int2 &b) {
    a.x *= b.x;
    a.y *= b.y;
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
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator<<(const uint2 &a, uint32_t b) {
    return make_uint2(a.x << b, a.y << b);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator<<=(uint2 &a, uint32_t b) {
    a = a << b;
    return a;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator>>(const uint2 &a, uint32_t b) {
    return make_uint2(a.x >> b, a.y >> b);
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

    CUDA_COMMON_FUNCTION Bool2D(bool v = 0) : x(v), y(v) {}
    CUDA_COMMON_FUNCTION Bool2D(bool xx, bool yy) :
        x(xx), y(yy) {}
};

CUDA_COMMON_FUNCTION CUDA_INLINE bool all(const Bool2D &v) {
    return v.x && v.y;
}

CUDA_COMMON_FUNCTION CUDA_INLINE bool any(const Bool2D &v) {
    return v.x || v.y;
}



struct Vector2D {
    FloatType x, y;

    CUDA_COMMON_FUNCTION Vector2D(FloatType v = 0) : x(v), y(v) {}
    CUDA_COMMON_FUNCTION Vector2D(FloatType xx, FloatType yy) :
        x(xx), y(yy) {}

    CUDA_COMMON_FUNCTION Vector2D operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector2D operator-() const {
        return Vector2D(-x, -y);
    }

    CUDA_COMMON_FUNCTION Vector2D &operator+=(const Vector2D &r) {
        x += r.x;
        y += r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector2D &operator-=(const Vector2D &r) {
        x -= r.x;
        y -= r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector2D &operator*=(FloatType r) {
        x *= r;
        y *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector2D &operator*=(const Vector2D &r) {
        x *= r.x;
        y *= r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector2D &operator/=(FloatType r) {
        FloatType rr = 1 / r;
        x *= rr;
        y *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector2D &operator/=(const Vector2D &r) {
        x /= r.x;
        y /= r.y;
        return *this;
    }

    CUDA_COMMON_FUNCTION FloatType sqLength() const {
        return x * x + y * y;
    }
    CUDA_COMMON_FUNCTION FloatType length() const {
        return std::sqrt(sqLength());
    }
    CUDA_COMMON_FUNCTION Vector2D &normalize() {
        FloatType l = length();
        return *this /= l;
    }
};

CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator==(
    const Vector2D &a, const Vector2D &b) {
    return Bool2D(a.x == b.x, a.y == b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator!=(
    const Vector2D &a, const Vector2D &b) {
    return Bool2D(a.x != b.x, a.y != b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator<(
    const Vector2D &a, const Vector2D &b) {
    return Bool2D(a.x < b.x, a.y < b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator<=(
    const Vector2D &a, const Vector2D &b) {
    return Bool2D(a.x <= b.x, a.y <= b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator>(
    const Vector2D &a, const Vector2D &b) {
    return Bool2D(a.x > b.x, a.y > b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator>=(
    const Vector2D &a, const Vector2D &b) {
    return Bool2D(a.x >= b.x, a.y >= b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D operator+(
    const Vector2D &a, const Vector2D &b) {
    Vector2D ret = a;
    ret += b;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D operator-(
    const Vector2D &a, const Vector2D &b) {
    Vector2D ret = a;
    ret -= b;
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D operator*(
    const Vector2D &a, N b) {
    Vector2D ret = a;
    ret *= static_cast<FloatType>(b);
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D operator*(
    N a, const Vector2D &b) {
    Vector2D ret = b;
    ret *= static_cast<FloatType>(a);
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D operator*(
    const Vector2D &a, const Vector2D &b) {
    Vector2D ret = a;
    ret *= b;
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D operator/(
    const Vector2D &a, N b) {
    Vector2D ret = a;
    ret /= static_cast<FloatType>(b);
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D operator/(
    const Vector2D &a, const Vector2D &b) {
    Vector2D ret = a;
    ret /= b;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D min(
    const Vector2D &a, const Vector2D &b) {
    return Vector2D(std::fmin(a.x, b.x), std::fmin(a.y, b.y));
}

CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D max(
    const Vector2D &a, const Vector2D &b) {
    return Vector2D(std::fmax(a.x, b.x), std::fmax(a.y, b.y));
}

CUDA_COMMON_FUNCTION CUDA_INLINE FloatType dot(
    const Vector2D &a, const Vector2D &b) {
    return a.x * b.x + a.y * b.y;
}

CUDA_COMMON_FUNCTION CUDA_INLINE FloatType cross(
    const Vector2D &a, const Vector2D &b) {
    return a.x * b.y - a.y * b.x;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D normalize(
    const Vector2D &v) {
    Vector2D ret = v;
    ret.normalize();
    return ret;
}



struct Point2D {
    FloatType x, y;

    CUDA_COMMON_FUNCTION Point2D(FloatType v = 0) : x(v), y(v) {}
    CUDA_COMMON_FUNCTION Point2D(FloatType xx, FloatType yy) :
        x(xx), y(yy) {}
    CUDA_COMMON_FUNCTION explicit Point2D(const Vector2D &v) : x(v.x), y(v.y) {}

    CUDA_COMMON_FUNCTION explicit operator Vector2D() const {
        return Vector2D(x, y);
    }

    CUDA_COMMON_FUNCTION Point2D operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D operator-() const {
        return Point2D(-x, -y);
    }

    CUDA_COMMON_FUNCTION Point2D &operator+=(const Point2D &r) {
        x += r.x;
        y += r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D &operator+=(const Vector2D &r) {
        x += r.x;
        y += r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D &operator-=(const Vector2D &r) {
        x -= r.x;
        y -= r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D &operator*=(FloatType r) {
        x *= r;
        y *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D &operator*=(const Point2D &r) {
        x *= r.x;
        y *= r.y;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D &operator/=(FloatType r) {
        FloatType rr = 1 / r;
        x *= rr;
        y *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point2D &operator/=(const Point2D &r) {
        x /= r.x;
        y /= r.y;
        return *this;
    }
};

CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator==(const Point2D &a, const Point2D &b) {
    return Bool2D(a.x == b.x, a.y == b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator!=(const Point2D &a, const Point2D &b) {
    return Bool2D(a.x != b.x, a.y != b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator<(const Point2D &a, const Point2D &b) {
    return Bool2D(a.x < b.x, a.y < b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator<=(const Point2D &a, const Point2D &b) {
    return Bool2D(a.x <= b.x, a.y <= b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator>(const Point2D &a, const Point2D &b) {
    return Bool2D(a.x > b.x, a.y > b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool2D operator>=(const Point2D &a, const Point2D &b) {
    return Bool2D(a.x >= b.x, a.y >= b.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point2D operator+(const Point2D &a, const Point2D &b) {
    Point2D ret = a;
    ret += b;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point2D operator+(const Point2D &a, const Vector2D &b) {
    Point2D ret = a;
    ret += b;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point2D operator+(const Vector2D &a, const Point2D &b) {
    Point2D ret = b;
    ret += a;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Vector2D operator-(const Point2D &a, const Point2D &b) {
    auto ret = static_cast<Vector2D>(a);
    ret -= static_cast<Vector2D>(b);
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point2D operator-(const Point2D &a, const Vector2D &b) {
    Point2D ret = a;
    ret -= b;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point2D operator-(const Vector2D &a, const Point2D &b) {
    Point2D ret = -b;
    ret += a;
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D operator*(const Point2D &a, N b) {
    Point2D ret = a;
    ret *= static_cast<FloatType>(b);
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D operator*(N a, const Point2D &b) {
    Point2D ret = b;
    ret *= static_cast<FloatType>(a);
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point2D operator*(const Point2D &a, const Point2D &b) {
    Point2D ret = a;
    ret *= b;
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Point2D operator/(const Point2D &a, N b) {
    Point2D ret = a;
    ret /= static_cast<FloatType>(b);
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point2D operator/(const Point2D &a, const Point2D &b) {
    Point2D ret = a;
    ret /= b;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point2D min(const Point2D &a, const Point2D &b) {
    return Point2D(std::fmin(a.x, b.x), std::fmin(a.y, b.y));
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point2D max(const Point2D &a, const Point2D &b) {
    return Point2D(std::fmax(a.x, b.x), std::fmax(a.y, b.y));
}

CUDA_COMMON_FUNCTION CUDA_INLINE FloatType sqDistance(const Point2D &a, const Point2D &b) {
    Vector2D d = b - a;
    return d.x * d.x + d.y * d.y;
}

CUDA_COMMON_FUNCTION CUDA_INLINE FloatType distance(const Point2D &a, const Point2D &b) {
#if !defined(__CUDA_ARCH__)
    using std::sqrtf;
#endif
    return sqrtf(sqDistance(a, b));
}



struct Bool3D {
    bool x, y, z;

    CUDA_COMMON_FUNCTION Bool3D(bool v = 0) : x(v), y(v), z(v) {}
    CUDA_COMMON_FUNCTION Bool3D(bool xx, bool yy, bool zz) :
        x(xx), y(yy), z(zz) {}
};

CUDA_COMMON_FUNCTION CUDA_INLINE bool all(const Bool3D &v) {
    return v.x && v.y && v.z;
}

CUDA_COMMON_FUNCTION CUDA_INLINE bool any(const Bool3D &v) {
    return v.x || v.y || v.z;
}



template <bool isNormal>
struct Vector3DTemplate {
    FloatType x, y, z;

    CUDA_COMMON_FUNCTION Vector3DTemplate(FloatType v = 0) : x(v), y(v), z(v) {}
    CUDA_COMMON_FUNCTION Vector3DTemplate(FloatType xx, FloatType yy, FloatType zz) :
        x(xx), y(yy), z(zz) {}
    CUDA_COMMON_FUNCTION explicit Vector3DTemplate(const Vector3DTemplate<!isNormal> &v) :
        x(v.x), y(v.y), z(v.z) {}
    CUDA_COMMON_FUNCTION explicit Vector3DTemplate(const float3 &v) :
        x(v.x), y(v.y), z(v.z) {}

    CUDA_COMMON_FUNCTION explicit operator Vector3DTemplate<!isNormal>() const {
        return Vector3DTemplate<!isNormal>(x, y, z);
    }
    CUDA_COMMON_FUNCTION explicit operator float3() const {
        return make_float3(x, y, z);
    }

    CUDA_COMMON_FUNCTION float3 toNative() const {
        return make_float3(x, y, z);
    }

    CUDA_COMMON_FUNCTION Vector3DTemplate operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector3DTemplate operator-() const {
        return Vector3DTemplate(-x, -y, -z);
    }

    template <bool isNormalB>
    CUDA_COMMON_FUNCTION Vector3DTemplate &operator+=(const Vector3DTemplate<isNormalB> &r) {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }
    template <bool isNormalB>
    CUDA_COMMON_FUNCTION Vector3DTemplate &operator-=(const Vector3DTemplate<isNormalB> &r) {
        x -= r.x;
        y -= r.y;
        z -= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector3DTemplate &operator*=(FloatType r) {
        x *= r;
        y *= r;
        z *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector3DTemplate &operator*=(const Vector3DTemplate &r) {
        x *= r.x;
        y *= r.y;
        z *= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector3DTemplate &operator/=(FloatType r) {
        FloatType rr = 1 / r;
        x *= rr;
        y *= rr;
        z *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector3DTemplate &operator/=(const Vector3DTemplate &r) {
        x /= r.x;
        y /= r.y;
        z /= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector3DTemplate &safeDivide(const Vector3DTemplate &r) {
        x = r.x != 0 ? x / r.x : 0.0f;
        y = r.y != 0 ? y / r.y : 0.0f;
        z = r.z != 0 ? z / r.z : 0.0f;
        return *this;
    }

    CUDA_COMMON_FUNCTION FloatType sqLength() const {
        return x * x + y * y + z * z;
    }
    CUDA_COMMON_FUNCTION FloatType length() const {
        return std::sqrt(sqLength());
    }
    CUDA_COMMON_FUNCTION Vector3DTemplate &normalize() {
        FloatType l = length();
        return *this /= l;
    }

    CUDA_COMMON_FUNCTION bool allFinite() const {
#if !defined(__CUDA_ARCH__)
        using std::isfinite;
#endif
        return isfinite(x) && isfinite(y) && isfinite(z);
    }
};

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator==(
    const Vector3DTemplate<isNormal> &a, const Vector3DTemplate<isNormal> &b) {
    return Bool3D(a.x == b.x, a.y == b.y, a.z == b.z);
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator!=(
    const Vector3DTemplate<isNormal> &a, const Vector3DTemplate<isNormal> &b) {
    return Bool3D(a.x != b.x, a.y != b.y, a.z != b.z);
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator<(
    const Vector3DTemplate<isNormal> &a, const Vector3DTemplate<isNormal> &b) {
    return Bool3D(a.x < b.x, a.y < b.y, a.z < b.z);
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator<=(
    const Vector3DTemplate<isNormal> &a, const Vector3DTemplate<isNormal> &b) {
    return Bool3D(a.x <= b.x, a.y <= b.y, a.z <= b.z);
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator>(
    const Vector3DTemplate<isNormal> &a, const Vector3DTemplate<isNormal> &b) {
    return Bool3D(a.x > b.x, a.y > b.y, a.z > b.z);
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator>=(
    const Vector3DTemplate<isNormal> &a, const Vector3DTemplate<isNormal> &b) {
    return Bool3D(a.x >= b.x, a.y >= b.y, a.z >= b.z);
}

template <bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<isNormalA> operator+(
    const Vector3DTemplate<isNormalA> &a, const Vector3DTemplate<isNormalB> &b) {
    Vector3DTemplate<isNormalA> ret = a;
    ret += b;
    return ret;
}

template <bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<isNormalA> operator-(
    const Vector3DTemplate<isNormalA> &a, const Vector3DTemplate<isNormalB> &b) {
    Vector3DTemplate<isNormalA> ret = a;
    ret -= b;
    return ret;
}

template <bool isNormal, Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<isNormal> operator*(
    const Vector3DTemplate<isNormal> &a, N b) {
    Vector3DTemplate<isNormal> ret = a;
    ret *= static_cast<FloatType>(b);
    return ret;
}

template <Number32bit N, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<isNormal> operator*(
    N a, const Vector3DTemplate<isNormal> &b) {
    Vector3DTemplate<isNormal> ret = b;
    ret *= static_cast<FloatType>(a);
    return ret;
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<isNormal> operator*(
    const Vector3DTemplate<isNormal> &a, const Vector3DTemplate<isNormal> &b) {
    Vector3DTemplate<isNormal> ret = a;
    ret *= b;
    return ret;
}

template <bool isNormal, Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<isNormal> operator/(
    const Vector3DTemplate<isNormal> &a, N b) {
    Vector3DTemplate<isNormal> ret = a;
    ret /= static_cast<FloatType>(b);
    return ret;
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<isNormal> operator/(
    const Vector3DTemplate<isNormal> &a, const Vector3DTemplate<isNormal> &b) {
    Vector3DTemplate<isNormal> ret = a;
    ret /= b;
    return ret;
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<isNormal> safeDivide(
    const Vector3DTemplate<isNormal> &a, const Vector3DTemplate<isNormal> &b) {
    Vector3DTemplate<isNormal> ret = a;
    ret.safeDivide(b);
    return ret;
}

template <bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE FloatType dot(
    const Vector3DTemplate<isNormalA> &a, const Vector3DTemplate<isNormalB> &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <bool isNormalA, bool isNormalB>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<false> cross(
    const Vector3DTemplate<isNormalA> &a, const Vector3DTemplate<isNormalB> &b) {
    return Vector3DTemplate<false>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE FloatType length(
    const Vector3DTemplate<isNormal> &v) {
    return v.length();
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<isNormal> normalize(
    const Vector3DTemplate<isNormal> &v) {
    Vector3DTemplate<isNormal> ret = v;
    ret.normalize();
    return ret;
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<isNormal> abs(
    const Vector3DTemplate<isNormal> &v) {
    return Vector3DTemplate<isNormal>(std::fabs(v.x), std::fabs(v.y), std::fabs(v.z));
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<isNormal> min(
    const Vector3DTemplate<isNormal> &a, const Vector3DTemplate<isNormal> &b) {
    return Vector3DTemplate<isNormal>(std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z));
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<isNormal> max(
    const Vector3DTemplate<isNormal> &a, const Vector3DTemplate<isNormal> &b) {
    return Vector3DTemplate<isNormal>(std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z));
}

using Vector3D = Vector3DTemplate<false>;
using Normal3D = Vector3DTemplate<true>;



struct Point3D {
    FloatType x, y, z;

    CUDA_COMMON_FUNCTION Point3D(FloatType v = 0) : x(v), y(v), z(v) {}
    CUDA_COMMON_FUNCTION Point3D(FloatType xx, FloatType yy, FloatType zz) :
        x(xx), y(yy), z(zz) {}
    template <bool isNormal>
    CUDA_COMMON_FUNCTION explicit Point3D(const Vector3DTemplate<isNormal> &v) : x(v.x), y(v.y), z(v.z) {}
    CUDA_COMMON_FUNCTION explicit Point3D(const float3 &p) : x(p.x), y(p.y), z(p.z) {}

    template <bool isNormal>
    CUDA_COMMON_FUNCTION explicit operator Vector3DTemplate<isNormal>() const {
        return Vector3DTemplate<isNormal>(x, y, z);
    }
    CUDA_COMMON_FUNCTION explicit operator float3() const {
        return make_float3(x, y, z);
    }

    CUDA_COMMON_FUNCTION float3 toNative() const {
        return make_float3(x, y, z);
    }

    CUDA_COMMON_FUNCTION Point3D operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Point3D operator-() const {
        return Point3D(-x, -y, -z);
    }

    CUDA_COMMON_FUNCTION Point3D &operator+=(const Point3D &r) {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }
    template <bool isNormal>
    CUDA_COMMON_FUNCTION Point3D &operator+=(const Vector3DTemplate<isNormal> &r) {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }
    template <bool isNormal>
    CUDA_COMMON_FUNCTION Point3D &operator-=(const Vector3DTemplate<isNormal> &r) {
        x -= r.x;
        y -= r.y;
        z -= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point3D &operator*=(FloatType r) {
        x *= r;
        y *= r;
        z *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point3D &operator*=(const Point3D &r) {
        x *= r.x;
        y *= r.y;
        z *= r.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point3D &operator/=(FloatType r) {
        FloatType rr = 1 / r;
        x *= rr;
        y *= rr;
        z *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point3D &operator/=(const Point3D &r) {
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

CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator==(
    const Point3D &a, const Point3D &b) {
    return Bool3D(a.x == b.x, a.y == b.y, a.z == b.z);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator!=(
    const Point3D &a, const Point3D &b) {
    return Bool3D(a.x != b.x, a.y != b.y, a.z != b.z);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator<(
    const Point3D &a, const Point3D &b) {
    return Bool3D(a.x < b.x, a.y < b.y, a.z < b.z);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator<=(
    const Point3D &a, const Point3D &b) {
    return Bool3D(a.x <= b.x, a.y <= b.y, a.z <= b.z);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator>(
    const Point3D &a, const Point3D &b) {
    return Bool3D(a.x > b.x, a.y > b.y, a.z > b.z);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator>=(
    const Point3D &a, const Point3D &b) {
    return Bool3D(a.x >= b.x, a.y >= b.y, a.z >= b.z);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point3D operator+(
    const Point3D &a, const Point3D &b) {
    Point3D ret = a;
    ret += b;
    return ret;
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D operator+(
    const Point3D &a, const Vector3DTemplate<isNormal> &b) {
    Point3D ret = a;
    ret += b;
    return ret;
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D operator+(
    const Vector3DTemplate<isNormal> &a, const Point3D &b) {
    Point3D ret = b;
    ret += a;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D operator-(
    const Point3D &a, const Point3D &b) {
    auto ret = static_cast<Vector3D>(a);
    ret -= static_cast<Vector3D>(b);
    return ret;
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D operator-(
    const Point3D &a, const Vector3DTemplate<isNormal> &b) {
    Point3D ret = a;
    ret -= b;
    return ret;
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D operator-(
    const Vector3DTemplate<isNormal> &a, const Point3D &b) {
    Point3D ret = -b;
    ret += a;
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D operator*(
    const Point3D &a, N b) {
    Point3D ret = a;
    ret *= static_cast<FloatType>(b);
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D operator*(
    N a, const Point3D &b) {
    Point3D ret = b;
    ret *= static_cast<FloatType>(a);
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point3D operator*(
    const Point3D &a, const Point3D &b) {
    Point3D ret = a;
    ret *= b;
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Point3D operator/(
    const Point3D &a, N b) {
    Point3D ret = a;
    ret /= static_cast<FloatType>(b);
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point3D operator/(
    const Point3D &a, const Point3D &b) {
    Point3D ret = a;
    ret /= b;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point3D min(
    const Point3D &a, const Point3D &b) {
    return Point3D(std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z));
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point3D max(
    const Point3D &a, const Point3D &b) {
    return Point3D(std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z));
}

CUDA_COMMON_FUNCTION CUDA_INLINE FloatType sqDistance(const Point3D &a, const Point3D &b) {
    Vector3D d = b - a;
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

CUDA_COMMON_FUNCTION CUDA_INLINE FloatType distance(const Point3D &a, const Point3D &b) {
#if !defined(__CUDA_ARCH__)
    using std::sqrtf;
#endif
    return sqrtf(sqDistance(a, b));
}



struct Bool4D {
    bool x, y, z, w;

    CUDA_COMMON_FUNCTION Bool4D(bool v = 0) : x(v), y(v), z(v), w(v) {}
    CUDA_COMMON_FUNCTION Bool4D(bool xx, bool yy, bool zz, bool ww) :
        x(xx), y(yy), z(zz), w(ww) {}
};

CUDA_COMMON_FUNCTION CUDA_INLINE bool all(const Bool4D &v) {
    return v.x && v.y && v.z && v.w;
}

CUDA_COMMON_FUNCTION CUDA_INLINE bool any(const Bool4D &v) {
    return v.x || v.y || v.z || v.w;
}



struct Vector4D {
    FloatType x, y, z, w;

    CUDA_COMMON_FUNCTION Vector4D(FloatType v = 0) : x(v), y(v), z(v), w(w) {}
    CUDA_COMMON_FUNCTION Vector4D(FloatType xx, FloatType yy, FloatType zz, FloatType ww) :
        x(xx), y(yy), z(zz), w(ww) {}
    CUDA_COMMON_FUNCTION Vector4D(const Vector3D &v, FloatType ww = 0) :
        x(v.x), y(v.y), z(v.z), w(ww) {}
    CUDA_COMMON_FUNCTION Vector4D(const Point3D &p, FloatType ww = 1) :
        x(p.x), y(p.y), z(p.z), w(ww) {}

    template <bool isNormal>
    CUDA_COMMON_FUNCTION explicit operator Vector3DTemplate<isNormal>() const {
        return Vector3DTemplate<isNormal>(x, y, z);
    }

    CUDA_COMMON_FUNCTION Vector4D operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector4D operator-() const {
        return Vector4D(-x, -y, -z, -w);
    }

    CUDA_COMMON_FUNCTION Vector4D &operator+=(const Vector4D &r) {
        x += r.x;
        y += r.y;
        z += r.z;
        w += r.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector4D &operator-=(const Vector4D &r) {
        x -= r.x;
        y -= r.y;
        z -= r.z;
        w -= r.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector4D &operator*=(FloatType r) {
        x *= r;
        y *= r;
        z *= r;
        w *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector4D &operator*=(const Vector4D &r) {
        x *= r.x;
        y *= r.y;
        z *= r.z;
        w *= r.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector4D &operator/=(FloatType r) {
        FloatType rr = 1 / r;
        x *= rr;
        y *= rr;
        z *= rr;
        w *= rr;
        return *this;
    }
    CUDA_COMMON_FUNCTION Vector4D &operator/=(const Vector4D &r) {
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

CUDA_COMMON_FUNCTION CUDA_INLINE Bool4D operator==(
    const Vector4D &a, const Vector4D &b) {
    return Bool4D(a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool4D operator!=(
    const Vector4D &a, const Vector4D &b) {
    return Bool4D(a.x != b.x, a.y != b.y, a.z != b.z, a.w != b.w);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool4D operator<(
    const Vector4D &a, const Vector4D &b) {
    return Bool4D(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool4D operator<=(
    const Vector4D &a, const Vector4D &b) {
    return Bool4D(a.x <= b.x, a.y <= b.y, a.z <= b.z, a.w <= b.w);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool4D operator>(
    const Vector4D &a, const Vector4D &b) {
    return Bool4D(a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool4D operator>=(
    const Vector4D &a, const Vector4D &b) {
    return Bool4D(a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D operator+(
    const Vector4D &a, const Vector4D &b) {
    Vector4D ret = a;
    ret += b;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D operator-(
    const Vector4D &a, const Vector4D &b) {
    Vector4D ret = a;
    ret -= b;
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D operator*(
    const Vector4D &a, N b) {
    Vector4D ret = a;
    ret *= static_cast<FloatType>(b);
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D operator*(
    N a, const Vector4D &b) {
    Vector4D ret = b;
    ret *= static_cast<FloatType>(a);
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D operator*(
    const Vector4D &a, const Vector4D &b) {
    Vector4D ret = a;
    ret *= b;
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D operator/(
    const Vector4D &a, N b) {
    Vector4D ret = a;
    ret /= static_cast<FloatType>(b);
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Vector4D operator/(
    const Vector4D &a, const Vector4D &b) {
    Vector4D ret = a;
    ret /= b;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE FloatType dot(
    const Vector4D &a, const Vector4D &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}



struct Matrix3x3 {
    union {
        Vector3D c0;
        struct {
            FloatType m00, m10, m20;
        };
    };
    union {
        Vector3D c1;
        struct {
            FloatType m01, m11, m21;
        };
    };
    union {
        Vector3D c2;
        struct {
            FloatType m02, m12, m22;
        };
    };

    CUDA_COMMON_FUNCTION Matrix3x3() :
        c0(1, 0, 0), c1(0, 1, 0), c2(0, 0, 1) {}
    CUDA_COMMON_FUNCTION Matrix3x3(const Vector3D cc0, const Vector3D &cc1, const Vector3D cc2) :
        c0(cc0), c1(cc1), c2(cc2) {}
    CUDA_COMMON_FUNCTION Matrix3x3(const Point3D cc0, const Point3D &cc1, const Point3D cc2) :
        c0(static_cast<Vector3D>(cc0)),
        c1(static_cast<Vector3D>(cc1)),
        c2(static_cast<Vector3D>(cc2)) {}

    CUDA_COMMON_FUNCTION Matrix3x3 operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Matrix3x3 operator-() const {
        return Matrix3x3(-c0, -c1, -c2);
    }

    template <Number32bit N>
    CUDA_COMMON_FUNCTION Matrix3x3 &operator*=(N r) {
        c0 *= r;
        c1 *= r;
        c2 *= r;
        return *this;
    }
    CUDA_COMMON_FUNCTION Matrix3x3 &operator*=(const Matrix3x3 &r) {
        Vector3D rs[] = { row(0), row(1), row(2) };
        m00 = dot(rs[0], r.c0); m01 = dot(rs[0], r.c1); m02 = dot(rs[0], r.c2);
        m10 = dot(rs[1], r.c0); m11 = dot(rs[1], r.c1); m12 = dot(rs[1], r.c2);
        m20 = dot(rs[2], r.c0); m21 = dot(rs[2], r.c1); m22 = dot(rs[2], r.c2);
        return *this;
    }

    template <std::integral I>
    CUDA_COMMON_FUNCTION Vector3D row(I index) const {
        switch (index) {
        case 0:
            return Vector3D(c0.x, c1.x, c2.x);
        case 1:
            return Vector3D(c0.y, c1.y, c2.y);
        case 2:
            return Vector3D(c0.z, c1.z, c2.z);
        default:
            return Vector3D(NAN);
        }
    }

    CUDA_COMMON_FUNCTION Matrix3x3 &invert() {
        FloatType det = 1 /
            (m00 * m11 * m22 + m01 * m12 * m20 + m02 * m10 * m21
             - m02 * m11 * m20 - m01 * m10 * m22 - m00 * m12 * m21);
        Matrix3x3 m;
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

    CUDA_COMMON_FUNCTION Matrix3x3 &transpose() {
        FloatType temp;
        temp = m10; m10 = m01; m01 = temp;
        temp = m20; m20 = m02; m02 = temp;
        temp = m21; m21 = m12; m12 = temp;
        return *this;
    }
};

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 operator*(
    const Matrix3x3 &a, const Matrix3x3 &b) {
    Matrix3x3 ret = a;
    ret *= b;
    return ret;
}

template <bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE Vector3DTemplate<isNormal> operator*(
    const Matrix3x3 &a, const Vector3DTemplate<isNormal> &b) {
    return Vector3DTemplate<isNormal>(dot(a.row(0), b), dot(a.row(1), b), dot(a.row(2), b));
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 operator*(N a, const Matrix3x3 &b) {
    Matrix3x3 ret = b;
    ret *= a;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Point3D operator*(
    const Matrix3x3 &a, const Point3D &b) {
    auto vb = static_cast<Vector3D>(b);
    return Point3D(
        dot(a.row(0), vb),
        dot(a.row(1), vb),
        dot(a.row(2), vb));
}

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 invert(
    const Matrix3x3 &m) {
    Matrix3x3 ret = m;
    ret.invert();
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 transpose(
    const Matrix3x3 &m) {
    Matrix3x3 ret = m;
    ret.transpose();
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 scale3x3(
    const Vector3D &s) {
    return Matrix3x3(Vector3D(s.x, 0, 0),
                     Vector3D(0, s.y, 0),
                     Vector3D(0, 0, s.z));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 scale3x3(
    FloatType sx, FloatType sy, FloatType sz) {
    return scale3x3(Vector3D(sx, sy, sz));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 scale3x3(
    FloatType s) {
    return scale3x3(Vector3D(s, s, s));
}

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 rotate3x3(
    FloatType angle, const Vector3D &axis) {

    Matrix3x3 ret;
    Vector3D nAxis = normalize(axis);
    FloatType s = std::sin(angle);
    FloatType c = std::cos(angle);
    FloatType oneMinusC = 1 - c;

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
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 rotate3x3(
    FloatType angle, FloatType ax, FloatType ay, FloatType az) {
    return rotate3x3(angle, Vector3D(ax, ay, az));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 rotateX3x3(
    FloatType angle) {
    return rotate3x3(angle, Vector3D(1, 0, 0));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 rotateY3x3(
    FloatType angle) {
    return rotate3x3(angle, Vector3D(0, 1, 0));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix3x3 rotateZ3x3(
    FloatType angle) {
    return rotate3x3(angle, Vector3D(0, 0, 1));
}



struct Matrix4x4 {
    union {
        struct { FloatType m00, m10, m20, m30; };
        Vector4D c0;
    };
    union {
        struct { FloatType m01, m11, m21, m31; };
        Vector4D c1;
    };
    union {
        struct { FloatType m02, m12, m22, m32; };
        Vector4D c2;
    };
    union {
        struct { FloatType m03, m13, m23, m33; };
        Vector4D c3;
    };

    CUDA_COMMON_FUNCTION /*constexpr*/ Matrix4x4() :
        c0(1, 0, 0, 0),
        c1(0, 1, 0, 0),
        c2(0, 0, 1, 0),
        c3(0, 0, 0, 1) { }
    CUDA_COMMON_FUNCTION Matrix4x4(const FloatType array[9]) :
        m00(array[0]), m10(array[1]), m20(array[2]), m30(array[3]),
        m01(array[4]), m11(array[5]), m21(array[6]), m31(array[7]),
        m02(array[8]), m12(array[9]), m22(array[10]), m32(array[11]),
        m03(array[12]), m13(array[13]), m23(array[14]), m33(array[15]) { }
    CUDA_COMMON_FUNCTION constexpr Matrix4x4(
        const Vector4D &col0, const Vector4D &col1, const Vector4D &col2, const Vector4D &col3) :
        c0(col0), c1(col1), c2(col2), c3(col3)
    { }
    CUDA_COMMON_FUNCTION Matrix4x4(const Matrix3x3 &mat3x3, const Point3D &position) :
        c0(Vector4D(mat3x3.c0)), c1(Vector4D(mat3x3.c1)), c2(Vector4D(mat3x3.c2)), c3(Vector4D(position))
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
        const Vector4D r[] = { row(0), row(1), row(2), row(3) };
        return Matrix4x4(Vector4D(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0), dot(r[3], mat.c0)),
                         Vector4D(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1), dot(r[3], mat.c1)),
                         Vector4D(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2), dot(r[3], mat.c2)),
                         Vector4D(dot(r[0], mat.c3), dot(r[1], mat.c3), dot(r[2], mat.c3), dot(r[3], mat.c3)));
    }
    CUDA_COMMON_FUNCTION Vector3D operator*(const Vector3D &v) const {
        const Vector4D r[] = { row(0), row(1), row(2), row(3) };
        Vector4D v4(v, 0.0f);
        return Vector3D(dot(r[0], v4),
                        dot(r[1], v4),
                        dot(r[2], v4));
    }
    CUDA_COMMON_FUNCTION Point3D operator*(const Point3D &p) const {
        const Vector4D r[] = { row(0), row(1), row(2), row(3) };
        Vector4D v4(p, 1.0f);
        return Point3D(dot(r[0], v4),
                       dot(r[1], v4),
                       dot(r[2], v4));
    }
    CUDA_COMMON_FUNCTION Vector4D operator*(const Vector4D &v) const {
        const Vector4D r[] = { row(0), row(1), row(2), row(3) };
        return Vector4D(dot(r[0], v),
                        dot(r[1], v),
                        dot(r[2], v),
                        dot(r[3], v));
    }

    CUDA_COMMON_FUNCTION Matrix4x4 &operator*=(const Matrix4x4 &mat) {
        const Vector4D r[] = { row(0), row(1), row(2), row(3) };
        c0 = Vector4D(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0), dot(r[3], mat.c0));
        c1 = Vector4D(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1), dot(r[3], mat.c1));
        c2 = Vector4D(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2), dot(r[3], mat.c2));
        c3 = Vector4D(dot(r[0], mat.c3), dot(r[1], mat.c3), dot(r[2], mat.c3), dot(r[3], mat.c3));
        return *this;
    }

    CUDA_COMMON_FUNCTION Vector4D &operator[](uint32_t c) {
        //Assert(c < 3, "\"c\" is out of range [0, 3].");
        return *(&c0 + c);
    }
    CUDA_COMMON_FUNCTION Vector4D row(unsigned int r) const {
        //Assert(r < 3, "\"r\" is out of range [0, 3].");
        switch (r) {
        case 0:
            return Vector4D(m00, m01, m02, m03);
        case 1:
            return Vector4D(m10, m11, m12, m13);
        case 2:
            return Vector4D(m20, m21, m22, m23);
        case 3:
            return Vector4D(m30, m31, m32, m33);
        default:
            return Vector4D(0, 0, 0, 0);
        }
    }

    CUDA_COMMON_FUNCTION Matrix4x4 &invert() {
        FloatType inv[] = {
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

        FloatType recDet = 1.0f / (m00 * inv[0] + m10 * inv[4] + m20 * inv[8] + m30 * inv[12]);
        for (int i = 0; i < 16; ++i)
            inv[i] *= recDet;
        *this = Matrix4x4(inv);

        return *this;
    }

    CUDA_COMMON_FUNCTION Matrix4x4 &transpose() {
        FloatType temp;
        temp = m10; m10 = m01; m01 = temp;
        temp = m20; m20 = m02; m02 = temp;
        temp = m21; m21 = m12; m12 = temp;
        temp = m30; m30 = m03; m03 = temp;
        temp = m31; m31 = m13; m13 = temp;
        temp = m32; m32 = m23; m23 = temp;
        return *this;
    }

    CUDA_COMMON_FUNCTION Matrix3x3 getUpperLeftMatrix() const {
        return Matrix3x3(Vector3D(c0), Vector3D(c1), Vector3D(c2));
    }

    CUDA_COMMON_FUNCTION void decompose(
        Vector3D* retScale, Vector3D* rotation, Vector3D* translation) const {
        Matrix4x4 mat = *this;

        // JP: 移動成分
        // EN: Translation component
        if (translation)
            *translation = Vector3D(mat.c3);

        Vector3D scale(
            length(Vector3D(mat.c0)),
            length(Vector3D(mat.c1)),
            length(Vector3D(mat.c2)));

        // JP: 拡大縮小成分
        // EN: Scale component
        if (retScale)
            *retScale = scale;

        if (!rotation)
            return;

        // JP: 上記成分を排除
        // EN: Remove the above components
        mat.c3 = Vector4D(0, 0, 0, 1);
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
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 invert(const Matrix4x4 &mat) {
    Matrix4x4 ret = mat;
    return ret.invert();
}

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 scale4x4(
    const Vector3D &s) {
    return Matrix4x4(Vector4D(s.x, 0, 0, 0),
                     Vector4D(0, s.y, 0, 0),
                     Vector4D(0, 0, s.z, 0),
                     Vector4D(0, 0, 0, 1));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 scale4x4(
    FloatType sx, FloatType sy, FloatType sz) {
    return scale4x4(Vector3D(sx, sy, sz));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 scale4x4(
    FloatType s) {
    return scale4x4(Vector3D(s, s, s));
}

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 rotate4x4(
    FloatType angle, const Vector3D &axis) {
    Matrix4x4 matrix;
    Vector3D nAxis = normalize(axis);
    FloatType s = std::sin(angle);
    FloatType c = std::cos(angle);
    FloatType oneMinusC = 1 - c;

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
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 rotate4x4(
    FloatType angle, FloatType ax, FloatType ay, FloatType az) {
    return rotate4x4(angle, Vector3D(ax, ay, az));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 rotateX4x4(
    FloatType angle) {
    return rotate4x4(angle, Vector3D(1, 0, 0));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 rotateY4x4(
    FloatType angle) {
    return rotate4x4(angle, Vector3D(0, 1, 0));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 rotateZ4x4(
    FloatType angle) {
    return rotate4x4(angle, Vector3D(0, 0, 1));
}

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 translate4x4(
    const Vector3D &t) {
    return Matrix4x4(Vector4D(1, 0, 0, 0),
                     Vector4D(0, 1, 0, 0),
                     Vector4D(0, 0, 1, 0),
                     Vector4D(t, 1.0f));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 translate4x4(
    FloatType tx, FloatType ty, FloatType tz) {
    return translate4x4(Vector3D(tx, ty, tz));
}

CUDA_COMMON_FUNCTION CUDA_INLINE Matrix4x4 camera(
    FloatType aspect, FloatType fovY, FloatType near, FloatType far) {
    Matrix4x4 matrix;
    FloatType f = 1 / std::tan(fovY / 2);
    FloatType dz = far - near;

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



struct Quaternion {
    union {
        Vector3D v;
        struct {
            FloatType x;
            FloatType y;
            FloatType z;
        };
    };
    FloatType w;

    CUDA_COMMON_FUNCTION Quaternion() :
        v(0), w(1) {}
    CUDA_COMMON_FUNCTION Quaternion(FloatType xx, FloatType yy, FloatType zz, FloatType ww) :
        v(xx, yy, zz), w(ww) {}
    CUDA_COMMON_FUNCTION Quaternion(const Vector3D &vv, FloatType ww) :
        v(vv), w(ww) {}

    CUDA_COMMON_FUNCTION bool operator==(const Quaternion &q) const {
        return all(v == q.v) && w == q.w;
    }
    CUDA_COMMON_FUNCTION bool operator!=(const Quaternion &q) const {
        return any(v != q.v) || w != q.w;
    }

    CUDA_COMMON_FUNCTION Quaternion operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION Quaternion operator-() const {
        return Quaternion(-v, -w);
    }

    CUDA_COMMON_FUNCTION Quaternion operator+(const Quaternion &q) const {
        return Quaternion(v + q.v, w + q.w);
    }
    CUDA_COMMON_FUNCTION Quaternion operator-(const Quaternion &q) const {
        return Quaternion(v - q.v, w - q.w);
    }
    CUDA_COMMON_FUNCTION Quaternion operator*(const Quaternion &q) const {
        return Quaternion(cross(v, q.v) + w * q.v + q.w * v, w * q.w - dot(v, q.v));
    }
    CUDA_COMMON_FUNCTION Quaternion operator*(FloatType s) const { return Quaternion(v * s, w * s); }
    CUDA_COMMON_FUNCTION Quaternion operator/(FloatType s) const { FloatType r = 1 / s; return *this * r; }
    CUDA_COMMON_FUNCTION CUDA_INLINE friend Quaternion operator*(FloatType s, const Quaternion &q) { return q * s; }

    CUDA_COMMON_FUNCTION void toEulerAngles(FloatType* roll, FloatType* pitch, FloatType* yaw) const {
        FloatType xx = x * x;
        FloatType xy = x * y;
        FloatType xz = x * z;
        FloatType xw = x * w;
        FloatType yy = y * y;
        FloatType yz = y * z;
        FloatType yw = y * w;
        FloatType zz = z * z;
        FloatType zw = z * w;
        FloatType ww = w * w;
        *pitch = std::atan2(2 * (xw + yz), ww - xx - yy + zz); // around x
        *yaw = std::asin(std::fmin(std::fmax(2.0f * (yw - xz), -1.0f), 1.0f)); // around y
        *roll = std::atan2(2 * (zw + xy), ww + xx - yy - zz); // around z
    }
    CUDA_COMMON_FUNCTION Matrix3x3 toMatrix3x3() const {
        FloatType xx = x * x, yy = y * y, zz = z * z;
        FloatType xy = x * y, yz = y * z, zx = z * x;
        FloatType xw = x * w, yw = y * w, zw = z * w;
        return Matrix3x3(Vector3D(1 - 2 * (yy + zz), 2 * (xy + zw), 2 * (zx - yw)),
                         Vector3D(2 * (xy - zw), 1 - 2 * (xx + zz), 2 * (yz + xw)),
                         Vector3D(2 * (zx + yw), 2 * (yz - xw), 1 - 2 * (xx + yy)));
    }

    CUDA_COMMON_FUNCTION bool allFinite() const {
#if !defined(__CUDA_ARCH__)
        using std::isfinite;
#endif
        return v.allFinite() && isfinite(w);
    }
};

CUDA_COMMON_FUNCTION CUDA_INLINE bool allFinite(const Quaternion &q) {
    return q.allFinite();
}

CUDA_COMMON_FUNCTION CUDA_INLINE FloatType dot(const Quaternion &q0, const Quaternion &q1) {
    return dot(q0.v, q1.v) + q0.w * q1.w;
}

CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion normalize(const Quaternion &q) {
    return q / std::sqrt(dot(q, q));
}

CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion qRotate(FloatType angle, const Vector3D &axis) {
    FloatType ha = angle / 2;
    FloatType s = std::sin(ha), c = std::cos(ha);
    return Quaternion(s * normalize(axis), c);
}
CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion qRotate(FloatType angle, FloatType ax, FloatType ay, FloatType az) {
    return qRotate(angle, Vector3D(ax, ay, az));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion qRotateX(FloatType angle) {
    return qRotate(angle, Vector3D(1, 0, 0));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion qRotateY(FloatType angle) {
    return qRotate(angle, Vector3D(0, 1, 0));
}
CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion qRotateZ(FloatType angle) {
    return qRotate(angle, Vector3D(0, 0, 1));
}

CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion qFromEulerAngles(FloatType roll, FloatType pitch, FloatType yaw) {
    return qRotateZ(roll) * qRotateY(yaw) * qRotateX(pitch);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Quaternion Slerp(FloatType t, const Quaternion &q0, const Quaternion &q1) {
    FloatType cosTheta = dot(q0, q1);
    if (cosTheta > 0.9995f)
        return normalize((1 - t) * q0 + t * q1);
    else {
        FloatType theta = std::acos(std::fmin(std::fmax(cosTheta, -1.0f), 1.0f));
        FloatType thetap = theta * t;
        Quaternion qPerp = normalize(q1 - q0 * cosTheta);
        FloatType sinThetaP, cosThetaP;
        sinThetaP = std::sin(thetap);
        cosThetaP = std::cos(thetap);
        //sincos(thetap, &sinThetaP, &cosThetaP);
        return q0 * cosThetaP + qPerp * sinThetaP;
    }
}



struct RGB {
    FloatType r, g, b;

    CUDA_COMMON_FUNCTION RGB(FloatType v = 0) : r(v), g(v), b(v) {}
    CUDA_COMMON_FUNCTION RGB(FloatType rr, FloatType gg, FloatType bb) :
        r(rr), g(gg), b(bb) {}
    CUDA_COMMON_FUNCTION explicit RGB(const float3 &v) :
        r(v.x), g(v.y), b(v.z) {}

    CUDA_COMMON_FUNCTION explicit operator float3() const {
        return make_float3(r, g, b);
    }

    CUDA_COMMON_FUNCTION float3 toNative() const {
        return make_float3(r, g, b);
    }

    CUDA_COMMON_FUNCTION RGB operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB operator-() const {
        return RGB(-r, -g, -b);
    }

    CUDA_COMMON_FUNCTION RGB &operator+=(const RGB &o) {
        r += o.r;
        g += o.g;
        b += o.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB &operator-=(const RGB &o) {
        r -= o.r;
        g -= o.g;
        b -= o.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB &operator*=(FloatType o) {
        r *= o;
        g *= o;
        b *= o;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB &operator*=(const RGB &o) {
        r *= o.r;
        g *= o.g;
        b *= o.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB &operator/=(FloatType o) {
        FloatType ro = 1 / o;
        r *= ro;
        g *= ro;
        b *= ro;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB &operator/=(const RGB &o) {
        r /= o.r;
        g /= o.g;
        b /= o.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGB &safeDivide(const RGB &o) {
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

CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator==(const RGB &a, const RGB &b) {
    return Bool3D(a.r == b.r, a.g == b.g, a.b == b.b);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator!=(const RGB &a, const RGB &b) {
    return Bool3D(a.r != b.r, a.g != b.g, a.b != b.b);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator<(const RGB &a, const RGB &b) {
    return Bool3D(a.r < b.r, a.g < b.g, a.b < b.b);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator<=(const RGB &a, const RGB &b) {
    return Bool3D(a.r <= b.r, a.g <= b.g, a.b <= b.b);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator>(const RGB &a, const RGB &b) {
    return Bool3D(a.r > b.r, a.g > b.g, a.b > b.b);
}

CUDA_COMMON_FUNCTION CUDA_INLINE Bool3D operator>=(const RGB &a, const RGB &b) {
    return Bool3D(a.r >= b.r, a.g >= b.g, a.b >= b.b);
}

CUDA_COMMON_FUNCTION CUDA_INLINE RGB operator+(const RGB &a, const RGB &b) {
    RGB ret = a;
    ret += b;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE RGB operator-(const RGB &a, const RGB &b) {
    RGB ret = a;
    ret -= b;
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB operator*(const RGB &a, N b) {
    RGB ret = a;
    ret *= static_cast<FloatType>(b);
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB operator*(N a, const RGB &b) {
    RGB ret = b;
    ret *= static_cast<FloatType>(a);
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE RGB operator*(const RGB &a, const RGB &b) {
    RGB ret = a;
    ret *= b;
    return ret;
}

template <Number32bit N>
CUDA_COMMON_FUNCTION CUDA_INLINE RGB operator/(const RGB &a, N b) {
    RGB ret = a;
    ret /= static_cast<FloatType>(b);
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE RGB operator/(const RGB &a, const RGB &b) {
    RGB ret = a;
    ret /= b;
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE RGB safeDivide(const RGB &a, const RGB &b) {
    RGB ret = a;
    ret.safeDivide(b);
    return ret;
}

CUDA_COMMON_FUNCTION CUDA_INLINE RGB abs(const RGB &v) {
    return RGB(std::fabs(v.r), std::fabs(v.g), std::fabs(v.b));
}

CUDA_COMMON_FUNCTION CUDA_INLINE RGB min(const RGB &a, const RGB &b) {
    return RGB(std::fmin(a.r, b.r), std::fmin(a.g, b.g), std::fmin(a.b, b.b));
}

CUDA_COMMON_FUNCTION CUDA_INLINE RGB max(const RGB &a, const RGB &b) {
    return RGB(std::fmax(a.r, b.r), std::fmax(a.g, b.g), std::fmax(a.b, b.b));
}



CUDA_COMMON_FUNCTION CUDA_INLINE RGB HSVtoRGB(float h, float s, float v) {
    if (s == 0)
        return RGB(v, v, v);

    h = h - std::floor(h);
    int32_t hi = static_cast<int32_t>(h * 6);
    float f = h * 6 - hi;
    float m = v * (1 - s);
    float n = v * (1 - s * f);
    float k = v * (1 - s * (1 - f));
    if (hi == 0)
        return RGB(v, k, m);
    else if (hi == 1)
        return RGB(n, v, m);
    else if (hi == 2)
        return RGB(m, v, k);
    else if (hi == 3)
        return RGB(m, n, v);
    else if (hi == 4)
        return RGB(k, m, v);
    else if (hi == 5)
        return RGB(v, m, n);
    return RGB(0, 0, 0);
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

CUDA_COMMON_FUNCTION CUDA_INLINE RGB sRGB_degamma(const RGB &value) {
    return RGB(sRGB_degamma_s(value.r),
               sRGB_degamma_s(value.g),
               sRGB_degamma_s(value.b));
}

CUDA_COMMON_FUNCTION CUDA_INLINE float sRGB_calcLuminance(const RGB &value) {
    return 0.2126729f * value.r + 0.7151522f * value.g + 0.0721750f * value.b;
}



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

    CUDA_COMMON_FUNCTION RGBAsOrderedInt& operator=(const RGBAsOrderedInt &v) {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGBAsOrderedInt& operator=(const volatile RGBAsOrderedInt &v) {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION volatile RGBAsOrderedInt& operator=(const RGBAsOrderedInt &v) volatile {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION volatile RGBAsOrderedInt& operator=(const volatile RGBAsOrderedInt &v) volatile {
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



struct AABB {
    Point3D minP;
    Point3D maxP;

    CUDA_COMMON_FUNCTION AABB() : minP(Point3D(INFINITY)), maxP(Point3D(-INFINITY)) {}

    CUDA_COMMON_FUNCTION AABB &unify(const Point3D &p) {
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
        Vector3D d = maxP - minP;
        minP -= 0.5f * (scale - 1) * d;
        maxP += 0.5f * (scale - 1) * d;
        return *this;
    }

    CUDA_COMMON_FUNCTION Point3D normalize(const Point3D &p) const {
        return static_cast<Point3D>(safeDivide(p - minP, maxP - minP));
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE friend AABB operator*(const Matrix4x4 &mat, const AABB &aabb) {
        AABB ret;
        ret
            .unify(mat * Point3D(aabb.minP.x, aabb.minP.y, aabb.minP.z))
            .unify(mat * Point3D(aabb.maxP.x, aabb.minP.y, aabb.minP.z))
            .unify(mat * Point3D(aabb.minP.x, aabb.maxP.y, aabb.minP.z))
            .unify(mat * Point3D(aabb.maxP.x, aabb.maxP.y, aabb.minP.z))
            .unify(mat * Point3D(aabb.minP.x, aabb.minP.y, aabb.maxP.z))
            .unify(mat * Point3D(aabb.maxP.x, aabb.minP.y, aabb.maxP.z))
            .unify(mat * Point3D(aabb.minP.x, aabb.maxP.y, aabb.maxP.z))
            .unify(mat * Point3D(aabb.maxP.x, aabb.maxP.y, aabb.maxP.z));
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
//#define HARD_CODED_BSDF LambertBRDF

#define USE_PROBABILITY_TEXTURE 0

// Use Walker's alias method with initialization by Vose's algorithm
//#define USE_WALKER_ALIAS_METHOD

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



    template <typename T, bool oobCheck>
    class ROBufferTemplate {
        const T* m_data;

    public:
        CUDA_COMMON_FUNCTION ROBufferTemplate() : m_data(nullptr) {}
        CUDA_COMMON_FUNCTION ROBufferTemplate(const T* data, uint32_t) :
            m_data(data) {}

        template <std::integral I>
        CUDA_COMMON_FUNCTION const T &operator[](I idx) const {
            return m_data[idx];
        }
    };

    template <typename T>
    class ROBufferTemplate<T, true> {
        const T* m_data;
        uint32_t m_numElements;

    public:
        CUDA_COMMON_FUNCTION ROBufferTemplate() : m_data(nullptr), m_numElements(0) {}
        CUDA_COMMON_FUNCTION ROBufferTemplate(const T* data, uint32_t numElements) :
            m_data(data), m_numElements(numElements) {}

        CUDA_COMMON_FUNCTION uint32_t getNumElements() const {
            return m_numElements;
        }

        template <std::integral I>
        CUDA_COMMON_FUNCTION const T &operator[](I idx) const {
            Assert(idx < m_numElements, "Buffer 0x%p OOB Access: %u >= %u\n",
                   m_data, static_cast<uint32_t>(idx), m_numElements);
            return m_data[idx];
        }
    };



    template <typename T, bool oobCheck>
    class RWBufferTemplate {
        T* m_data;

    public:
        CUDA_COMMON_FUNCTION RWBufferTemplate() : m_data(nullptr) {}
        CUDA_COMMON_FUNCTION RWBufferTemplate(T* data, uint32_t) :
            m_data(data) {}

        template <std::integral I>
        CUDA_COMMON_FUNCTION T &operator[](I idx) {
            return m_data[idx];
        }
        template <std::integral I>
        CUDA_COMMON_FUNCTION const T &operator[](I idx) const {
            return m_data[idx];
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

        CUDA_COMMON_FUNCTION uint32_t getNumElements() const {
            return m_numElements;
        }

        template <std::integral I>
        CUDA_COMMON_FUNCTION T &operator[](I idx) {
            Assert(idx < m_numElements, "Buffer 0x%p OOB Access: %u >= %u\n",
                   m_data, static_cast<uint32_t>(idx), m_numElements);
            return m_data[idx];
        }
        template <std::integral I>
        CUDA_COMMON_FUNCTION const T &operator[](I idx) const {
            Assert(idx < m_numElements, "Buffer 0x%p OOB Access: %u >= %u\n",
                   m_data, static_cast<uint32_t>(idx), m_numElements);
            return m_data[idx];
        }
    };



    static constexpr bool enableBufferOobCheck = true;
    template <typename T>
    using ROBuffer = ROBufferTemplate<T, enableBufferOobCheck>;
    template <typename T>
    using RWBuffer = RWBufferTemplate<T, enableBufferOobCheck>;



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
        RealType* m_weights;
#if defined(USE_WALKER_ALIAS_METHOD)
        const AliasTableEntry<RealType>* m_aliasTable;
        const AliasValueMap<RealType>* m_valueMaps;
#else
        RealType* m_CDF;
#endif
        RealType m_integral;
        uint32_t m_numValues;

    public:
#if defined(USE_WALKER_ALIAS_METHOD)
        DiscreteDistribution1DTemplate(
            RealType* weights, AliasTableEntry<RealType>* aliasTable, AliasValueMap<RealType>* valueMaps,
            RealType integral, uint32_t numValues) :
            m_weights(weights), m_aliasTable(aliasTable), m_valueMaps(valueMaps),
            m_integral(integral), m_numValues(numValues) {}
#else
        DiscreteDistribution1DTemplate(
            RealType* weights, RealType* CDF, RealType integral, uint32_t numValues) :
            m_weights(weights), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {}
#endif

        CUDA_COMMON_FUNCTION DiscreteDistribution1DTemplate() {}

        CUDA_COMMON_FUNCTION uint32_t sample(RealType u, RealType* prob, RealType* remapped = nullptr) const {
            Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
#if defined(USE_WALKER_ALIAS_METHOD)
            uint32_t idx = mapPrimarySampleToDiscrete(u, m_numValues, &u);
            const AliasTableEntry<RealType> &entry = m_aliasTable[idx];
            const AliasValueMap<RealType> &valueMap = m_valueMaps[idx];
            if (u < entry.probToPickFirst) {
                if (remapped)
                    *remapped = valueMap.scaleForFirst * u;
            }
            else {
                idx = entry.secondIndex;
                if (remapped)
                    *remapped = valueMap.scaleForSecond * u + valueMap.offsetForSecond;
            }
#else
            u *= m_integral;
            int idx = 0;
            for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
                if (idx + d >= m_numValues)
                    continue;
                if (m_CDF[idx + d] <= u)
                    idx += d;
            }
            Assert(idx < m_numValues, "Invalid Index!: %u >= %u, u: %g, integ: %g",
                   idx, m_numValues, u, m_integral);
            if (remapped) {
                RealType lCDF = m_CDF[idx];
                RealType rCDF = m_integral;
                if (idx < m_numValues - 1)
                    rCDF = m_CDF[idx + 1];
                *remapped = (u - lCDF) / (rCDF - lCDF);
                Assert(isfinite(*remapped), "Remapped value is not a finite value %g.",
                       *remapped);
            }
#endif
            *prob = m_weights[idx] / m_integral;
            return idx;
        }

        CUDA_COMMON_FUNCTION RealType evaluatePMF(uint32_t idx) const {
            if (!m_weights || m_integral == 0.0f)
                return 0.0f;
            Assert(idx < m_numValues, "\"idx\" is out of range [0, %u)", m_numValues);
            return m_weights[idx] / m_integral;
        }

        CUDA_COMMON_FUNCTION RealType integral() const { return m_integral; }

        CUDA_COMMON_FUNCTION uint32_t numValues() const { return m_numValues; }

        CUDA_COMMON_FUNCTION uint32_t setNumValues(uint32_t numValues) {
            m_numValues = numValues;
        }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION uint32_t setWeightAt(uint32_t index, RealType value) {
            m_weights[index] = value;
        }

#   if !defined(USE_WALKER_ALIAS_METHOD)
        CUDA_DEVICE_FUNCTION void finalize() {
            uint32_t lastIndex = m_numValues - 1;
            m_integral = m_CDF[lastIndex] + m_weights[lastIndex];
        }
#   endif
#endif
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



    CUDA_COMMON_FUNCTION CUDA_INLINE uint2 computeProbabilityTextureDimentions(uint32_t maxNumElems) {
#if !defined(__CUDA_ARCH__)
        using std::max;
#endif
        uint2 dims = make_uint2(max(nextPowerOf2(maxNumElems), 2u), 1u);
        while ((dims.x != dims.y) && (dims.x != 2 * dims.y)) {
            dims.x /= 2;
            dims.y *= 2;
        }
        return dims;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE uint2 compute2DFrom1D(const uint2 &dims, uint32_t index1D) {
        return make_uint2(index1D % dims.x, index1D / dims.x);
    }

    class ProbabilityTexture {
        CUtexObject m_cuTexObj;
        unsigned int m_maxDimX : 16;
        unsigned int m_maxDimY : 16;
        unsigned int m_dimX : 16;
        unsigned int m_dimY : 16;
        float m_integral;

    public:
        CUDA_COMMON_FUNCTION void setTexObject(CUtexObject texObj, uint2 maxDims) {
            m_cuTexObj = texObj;
            m_maxDimX = maxDims.x;
            m_maxDimY = maxDims.y;
        }

        CUDA_COMMON_FUNCTION void setDimensions(const uint2 &dims) {
            m_dimX = dims.x;
            m_dimY = dims.y;
        }

        CUDA_COMMON_FUNCTION uint2 getDimensions() const {
            return make_uint2(m_dimX, m_dimY);
        }

        CUDA_COMMON_FUNCTION uint32_t calcNumMipLevels() const {
            return nextPowOf2Exponent(m_dimX) + 1;
        }
        CUDA_COMMON_FUNCTION uint32_t calcMaxNumMipLevels() const {
            return nextPowOf2Exponent(m_maxDimX) + 1;
        }

        CUDA_COMMON_FUNCTION uint2 compute2DFrom1D(uint32_t index1D) const {
            return make_uint2(index1D % m_dimX, index1D / m_dimX);
        }
        CUDA_COMMON_FUNCTION uint32_t compute1DFrom2D(const uint2 &index2D) const {
            return index2D.y * m_dimX + index2D.x;
        }

        CUDA_COMMON_FUNCTION float integral() const {
            if (m_cuTexObj == 0 || m_integral == 0.0f)
                return 0.0f;
            return m_integral;
        }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION uint32_t sample(float u, float* prob, float* remapped = nullptr) const {
            Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
            uint2 index2D = make_uint2(0, 0);
            uint32_t numMipLevels = calcNumMipLevels();
            *prob = 1;
            Vector2D recCurActualDims;
            {
                uint2 curActualDims = make_uint2(2, m_maxDimX > m_maxDimY ? 1 : 2);
                curActualDims <<= calcMaxNumMipLevels() - numMipLevels;
                recCurActualDims = Vector2D(1.0f / curActualDims.x, 1.0f / curActualDims.y);
            }
            uint2 curDims = make_uint2(2, m_dimX > m_dimY ? 1 : 2);
            for (uint32_t mipLevel = numMipLevels - 2; mipLevel != UINT32_MAX; --mipLevel) {
                index2D = 2 * index2D;
                Vector2D tc(index2D.x + 0.5f, index2D.y + 0.5f);
                Vector2D ll = tc + Vector2D(0, 1);
                Vector2D lr = tc + Vector2D(1, 1);
                Vector2D ur = tc + Vector2D(1, 0);
                Vector2D ul = tc + Vector2D(0, 0);
                Vector2D nll = ll * recCurActualDims;
                Vector2D nlr = lr * recCurActualDims;
                Vector2D nur = ur * recCurActualDims;
                Vector2D nul = ul * recCurActualDims;
                Vector4D neighbors;
                neighbors.x = ll.y < curDims.y ?
                    tex2DLod<float>(m_cuTexObj, nll.x, nll.y, mipLevel) : 0.0f;
                neighbors.y = (lr.x < curDims.x && lr.y < curDims.y) ?
                    tex2DLod<float>(m_cuTexObj, nlr.x, nlr.y, mipLevel) : 0.0f;
                neighbors.z = ur.x < curDims.x ?
                    tex2DLod<float>(m_cuTexObj, nur.x, nur.y, mipLevel) : 0.0f;
                neighbors.w = tex2DLod<float>(m_cuTexObj, nul.x, nul.y, mipLevel);
                float sumProbs = neighbors.x + neighbors.y + neighbors.z + neighbors.w;
                u *= sumProbs;
                float accProb = 0;
                float stepProb;
                if ((accProb + neighbors.x) > u) {
                    stepProb = neighbors.x;
                    index2D.y += 1;
                }
                else {
                    accProb += neighbors.x;
                    if ((accProb + neighbors.y) > u) {
                        stepProb = neighbors.y;
                        u -= accProb;
                        index2D.x += 1;
                        index2D.y += 1;
                    }
                    else {
                        accProb += neighbors.y;
                        if ((accProb + neighbors.z) > u) {
                            stepProb = neighbors.z;
                            u -= accProb;
                            index2D.x += 1;
                        }
                        else {
                            accProb += neighbors.z;
                            stepProb = neighbors.w;
                            u -= accProb;
                        }
                    }
                }
                *prob *= stepProb / sumProbs;
                u /= stepProb;
                recCurActualDims /= 2.0f;
                curDims *= 2;
            }
            if (remapped)
                *remapped = u;
            return compute1DFrom2D(index2D);
        }

        CUDA_DEVICE_FUNCTION void setIntegral(float v) {
            m_integral = v;
        }
#endif
    };

    using LightDistribution =
#if USE_PROBABILITY_TEXTURE
        ProbabilityTexture;
#else
        DiscreteDistribution1D;
#endif



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

        CUDA_COMMON_FUNCTION CUDA_INLINE static float gradient(
            uint32_t hash, float xu, float yu, float zu) {
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

        CUDA_COMMON_FUNCTION float evaluate(const Point3D &p, float frequency) const {
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
        CUDA_COMMON_FUNCTION MultiOctavePerlinNoise3D(
            uint32_t numOctaves, float initialFrequency, float supValueOrInitialAmplitude, bool supSpecified,
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

        CUDA_COMMON_FUNCTION float evaluate(const Point3D &p) const {
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



    struct TexDimInfo {
        uint32_t dimX : 14;
        uint32_t dimY : 14;
        uint32_t isNonPowerOfTwo : 1;
        uint32_t isBCTexture : 1;
        uint32_t isLeftHanded : 1; // for normal map
    };



    using ReadModifiedNormal = DynamicFunction<
        Normal3D(CUtexObject texture, TexDimInfo dimInfo, Point2D texCoord)>;

    using BSDFGetSurfaceParameters = DynamicFunction<
        void(const uint32_t* data, RGB* diffuseReflectance, RGB* specularReflectance, float* roughness)>;
    using BSDFSampleThroughput = DynamicFunction<
        RGB(const uint32_t* data, const Vector3D &vGiven, float uDir0, float uDir1,
            Vector3D* vSampled, float* dirPDensity)>;
    using BSDFEvaluate = DynamicFunction<
        RGB(const uint32_t* data, const Vector3D &vGiven, const Vector3D &vSampled)>;
    using BSDFEvaluatePDF = DynamicFunction<
        float(const uint32_t* data, const Vector3D &vGiven, const Vector3D &vSampled)>;
    using BSDFEvaluateDHReflectanceEstimate = DynamicFunction<
        RGB(const uint32_t* data, const Vector3D &vGiven)>;



    struct Vertex {
        Point3D position;
        Normal3D normal;
        Vector3D texCoord0Dir;
        Point2D texCoord;
    };

    struct Triangle {
        uint32_t index0, index1, index2;
    };

    struct MaterialData;

    struct BSDFFlags {
        enum Value {
            None = 0,
            Regularize = 1 << 0,
        } value;

        CUDA_DEVICE_FUNCTION constexpr BSDFFlags(Value v = None) : value(v) {}

        CUDA_DEVICE_FUNCTION operator uint32_t() const {
            return static_cast<uint32_t>(value);
        }
    };

    using SetupBSDFBody = DynamicFunction<
        void(const MaterialData &matData, Point2D texCoord, uint32_t* bodyData, BSDFFlags flags)>;

    struct MaterialData {
        union {
            struct {
                CUtexObject reflectance;
                TexDimInfo reflectanceDimInfo;
            } asLambert;
            struct {
                CUtexObject diffuse;
                CUtexObject specular;
                CUtexObject smoothness;
                TexDimInfo diffuseDimInfo;
                TexDimInfo specularDimInfo;
                TexDimInfo smoothnessDimInfo;
            } asDiffuseAndSpecular;
            struct {
                CUtexObject baseColor_opacity;
                CUtexObject occlusion_roughness_metallic;
                TexDimInfo baseColor_opacity_dimInfo;
                TexDimInfo occlusion_roughness_metallic_dimInfo;
            } asSimplePBR;
        };
        CUtexObject normal;
        CUtexObject emittance;
        TexDimInfo normalDimInfo;

        ReadModifiedNormal readModifiedNormal;

        SetupBSDFBody setupBSDFBody;
        BSDFGetSurfaceParameters bsdfGetSurfaceParameters;
        BSDFSampleThroughput bsdfSampleThroughput;
        BSDFEvaluate bsdfEvaluate;
        BSDFEvaluatePDF bsdfEvaluatePDF;
        BSDFEvaluateDHReflectanceEstimate bsdfEvaluateDHReflectanceEstimate;
    };

    struct GeometryInstanceData {
        ROBuffer<Vertex> vertexBuffer;
        ROBuffer<Triangle> triangleBuffer;
        LightDistribution emitterPrimDist;
        uint32_t materialSlot;
        uint32_t geomInstSlot;
    };

    struct InstanceData {
        Matrix4x4 transform;
        Matrix4x4 prevTransform;
        Matrix3x3 normalMatrix;
        float uniformScale;

        ROBuffer<uint32_t> geomInstSlots;
        LightDistribution lightGeomInstDist;
    };
}
