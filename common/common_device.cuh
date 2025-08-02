﻿#pragma once

#include "common_shared.h"
#include "curve_evaluator.h"

static constexpr float RayEpsilon = 1e-4;



// ( 0, 0,  1) <=> phi:      0
// (-1, 0,  0) <=> phi: 1/2 pi
// ( 0, 0, -1) <=> phi:   1 pi
// ( 1, 0,  0) <=> phi: 3/2 pi
CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D fromPolarYUp(float phi, float theta) {
    float sinPhi, cosPhi;
    float sinTheta, cosTheta;
    sincosf(phi, &sinPhi, &cosPhi);
    sincosf(theta, &sinTheta, &cosTheta);
    return Vector3D(-sinPhi * sinTheta, cosTheta, cosPhi * sinTheta);
}
CUDA_DEVICE_FUNCTION CUDA_INLINE void toPolarYUp(const Vector3D &v, float* phi, float* theta) {
    *theta = std::acos(min(max(v.y, -1.0f), 1.0f));
    *phi = std::fmod(std::atan2(-v.x, v.z) + 2 * pi_v<float>,
                     2 * pi_v<float>);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE uint16_t encodeBarycentric(float bc) {
    return static_cast<uint16_t>(min(static_cast<uint32_t>(bc * 65535u), 65535u));
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float decodeBarycentric(uint16_t qbc) {
    return qbc / 65535.0f;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t encodeVector(const Vector3D &v) {
    float phi, theta;
    toPolarYUp(v, &phi, &theta);
    const uint32_t qPhi = min(static_cast<uint32_t>((phi / (2 * pi_v<float>)) * 65535u), 65535u);
    const uint32_t qTheta = min(static_cast<uint32_t>((theta / pi_v<float>) * 65535u), 65535u);
    return (qTheta << 16) | qPhi;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D decodeVector(uint32_t qv) {
    const uint32_t qPhi = qv & 0xFFFF;
    const uint32_t qTheta = qv >> 16;
    const float phi = 2 * pi_v<float> * (qPhi / 65535.0f);
    const float theta = pi_v<float> * (qTheta / 65535.0f);
    return fromPolarYUp(phi, theta);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t encodeNormal(const Normal3D &n) {
    float phi, theta;
    toPolarYUp(static_cast<Vector3D>(n), &phi, &theta);
    const uint32_t qPhi = min(static_cast<uint32_t>((phi / (2 * pi_v<float>)) * 65535u), 65535u);
    const uint32_t qTheta = min(static_cast<uint32_t>((theta / pi_v<float>) * 65535u), 65535u);
    return (qTheta << 16) | qPhi;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Normal3D decodeNormal(uint32_t qn) {
    const uint32_t qPhi = qn & 0xFFFF;
    const uint32_t qTheta = qn >> 16;
    const float phi = 2 * pi_v<float> * (qPhi / 65535.0f);
    const float theta = pi_v<float> * (qTheta / 65535.0f);
    return static_cast<Normal3D>(fromPolarYUp(phi, theta));
}

CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t encodeTexCoords(const Point2D &tc) {
    const uint32_t qtc0 = min(static_cast<uint32_t>((tc.x - std::floor(tc.x)) * 65535u), 65535u);
    const uint32_t qtc1 = min(static_cast<uint32_t>((tc.y - std::floor(tc.y)) * 65535u), 65535u);
    return (qtc1 << 16) | qtc0;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Point2D decodeTexCoords(uint32_t qtc) {
    const uint32_t qtc0 = qtc & 0xFFFF;
    const uint32_t qtc1 = qtc >> 16;
    const float tc0 = qtc0 / 65535.0f;
    const float tc1 = qtc1 / 65535.0f;
    return Point2D(tc0, tc1);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D halfVector(const Vector3D &a, const Vector3D &b) {
    return normalize(a + b);
}

template <bool isNormalA, bool isNormalB>
CUDA_DEVICE_FUNCTION CUDA_INLINE float absDot(
    const Vector3D_T<float, isNormalA> &a, const Vector3D_T<float, isNormalB> &b)
{
    return std::fabs(dot(a, b));
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void makeCoordinateSystem(
    const Normal3D &normal, Vector3D* tangent, Vector3D* bitangent)
{
    float sign = normal.z >= 0 ? 1 : -1;
    const float a = -1 / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    *tangent = Vector3D(1 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    *bitangent = Vector3D(b, sign + normal.y * normal.y * a, -normal.y);
}

// JP: 自己交叉回避のためにレイの原点にオフセットを付加する。
// EN: Add an offset to a ray origin to avoid self-intersection.
CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D offsetRayOriginNaive(
    const Point3D &p, const Normal3D &geometricNormal)
{
    return p + RayEpsilon * geometricNormal;
}

// Reference:
// Chapter 6. A Fast and Robust Method for Avoiding Self-Intersection, Ray Tracing Gems, 2019
CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D offsetRayOrigin(
    const Point3D &p, const Normal3D &geometricNormal)
{
    constexpr float kOrigin = 1.0f / 32.0f;
    constexpr float kFloatScale = 1.0f / 65536.0f;
    constexpr float kIntScale = 256.0f;

    int32_t offsetInInt[] = {
        static_cast<int32_t>(kIntScale * geometricNormal.x),
        static_cast<int32_t>(kIntScale * geometricNormal.y),
        static_cast<int32_t>(kIntScale * geometricNormal.z)
    };

    // JP: 数学的な衝突点の座標と、実際の座標の誤差は原点からの距離に比例する。
    //     intとしてオフセットを加えることでスケール非依存に適切なオフセットを加えることができる。
    // EN: The error of the actual coorinates of the intersection point to the mathematical one is proportional to the distance to the origin.
    //     Applying the offset as int makes applying appropriate scale invariant amount of offset possible.
    Point3D newP1(__int_as_float(__float_as_int(p.x) + (p.x < 0 ? -1 : 1) * offsetInInt[0]),
                  __int_as_float(__float_as_int(p.y) + (p.y < 0 ? -1 : 1) * offsetInInt[1]),
                  __int_as_float(__float_as_int(p.z) + (p.z < 0 ? -1 : 1) * offsetInInt[2]));

    // JP: 原点に近い場所では、原点からの距離に依存せず一定の誤差が残るため別処理が必要。
    // EN: A constant amount of error remains near the origin independent of the distance to the origin so we need handle it separately.
    Point3D newP2 = p + kFloatScale * geometricNormal;

    return Point3D(std::fabs(p.x) < kOrigin ? newP2.x : newP1.x,
                   std::fabs(p.y) < kOrigin ? newP2.y : newP1.y,
                   std::fabs(p.z) < kOrigin ? newP2.z : newP1.z);
}

template <typename T>
CUDA_DEVICE_FUNCTION CUDA_INLINE T sample(
    CUtexObject texture, shared::TexDimInfo dimInfo, const Point2D &texCoord, float mipLevel)
{
    return tex2DLod<T>(texture, texCoord.x, texCoord.y, mipLevel);
}

struct ReferenceFrame {
    Vector3D tangent;
    Vector3D bitangent;
    Normal3D normal;

    CUDA_DEVICE_FUNCTION ReferenceFrame() {}
    CUDA_DEVICE_FUNCTION ReferenceFrame(
        const Vector3D &_tangent, const Vector3D &_bitangent, const Normal3D &_normal) :
        tangent(_tangent), bitangent(_bitangent), normal(_normal) {}
    CUDA_DEVICE_FUNCTION ReferenceFrame(const Normal3D &_normal) : normal(_normal) {
        makeCoordinateSystem(normal, &tangent, &bitangent);
    }
    CUDA_DEVICE_FUNCTION ReferenceFrame(const Normal3D &_normal, const Vector3D &_tangent) :
        tangent(_tangent), normal(_normal) {
        bitangent = cross(normal, tangent);
    }

    CUDA_DEVICE_FUNCTION Vector3D toLocal(const Vector3D &v) const {
        return Vector3D(dot(tangent, v), dot(bitangent, v), dot(normal, v));
    }
    CUDA_DEVICE_FUNCTION Vector3D fromLocal(const Vector3D &v) const {
        return Vector3D(dot(Vector3D(tangent.x, bitangent.x, normal.x), v),
                        dot(Vector3D(tangent.y, bitangent.y, normal.y), v),
                        dot(Vector3D(tangent.z, bitangent.z, normal.z), v));
    }
};

CUDA_DEVICE_FUNCTION CUDA_INLINE void applyBumpMapping(
    const Normal3D &modNormalInTF, ReferenceFrame* frameToModify)
{
    // JP: 法線から回転軸と回転角(、Quaternion)を求めて対応する接平面ベクトルを求める。
    // EN: calculate a rotating axis and an angle (and quaternion) from the normal then calculate corresponding tangential vectors.
    float projLength = std::sqrt(modNormalInTF.x * modNormalInTF.x + modNormalInTF.y * modNormalInTF.y);
    if (projLength < 1e-3f)
        return;
    float tiltAngle = std::atan(projLength / modNormalInTF.z);
    float qSin, qCos;
    sincosf(tiltAngle / 2, &qSin, &qCos);
    float qX = (-modNormalInTF.y / projLength) * qSin;
    float qY = (modNormalInTF.x / projLength) * qSin;
    float qW = qCos;
    Vector3D modTangentInTF(1 - 2 * qY * qY, 2 * qX * qY, -2 * qY * qW);
    Vector3D modBitangentInTF(2 * qX * qY, 1 - 2 * qX * qX, 2 * qX * qW);

    Matrix3x3 matTFtoW(
        frameToModify->tangent,
        frameToModify->bitangent,
        Vector3D(frameToModify->normal));
    ReferenceFrame bumpShadingFrame(
        matTFtoW * modTangentInTF,
        matTFtoW * modBitangentInTF,
        matTFtoW * modNormalInTF);

    *frameToModify = bumpShadingFrame;
}

RT_CALLABLE_PROGRAM Normal3D RT_DC_NAME(readModifiedNormalFromNormalMap)
(CUtexObject texture, shared::TexDimInfo dimInfo, Point2D texCoord, float mipLevel) {
    float4 texValue = sample<float4>(texture, dimInfo, texCoord, mipLevel);
    Normal3D modLocalNormal(getXYZ(texValue));
    modLocalNormal = 2.0f * modLocalNormal - Normal3D(1.0f);
    if (dimInfo.isLeftHanded)
        modLocalNormal.y *= -1; // DirectX convention
    return modLocalNormal;
}
CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(readModifiedNormalFromNormalMap);

RT_CALLABLE_PROGRAM Normal3D RT_DC_NAME(readModifiedNormalFromNormalMap2ch)
(CUtexObject texture, shared::TexDimInfo dimInfo, Point2D texCoord, float mipLevel) {
    float2 texValue = sample<float2>(texture, dimInfo, texCoord, mipLevel);
    Vector2D xy(texValue.x, texValue.y);
    xy = 2.0f * xy - Vector2D(1.0f);
    float z = std::sqrt(1.0f - pow2(xy.x) - pow2(xy.y));
    Normal3D modLocalNormal(xy.x, xy.y, z);
    if (dimInfo.isLeftHanded)
        modLocalNormal.y *= -1; // DirectX convention
    return modLocalNormal;
}
CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(readModifiedNormalFromNormalMap2ch);

RT_CALLABLE_PROGRAM Normal3D RT_DC_NAME(readModifiedNormalFromHeightMap)
(CUtexObject texture, shared::TexDimInfo dimInfo, Point2D texCoord) {
    float4 heightValues = tex2Dgather<float4>(texture, texCoord.x, texCoord.y, 0);
    constexpr float coeff = (5.0f / 1024);
    uint32_t width = dimInfo.dimX;
    uint32_t height = dimInfo.dimY;
    float dhdu = (coeff * width) * (heightValues.y - heightValues.x);
    float dhdv = (coeff * height) * (heightValues.x - heightValues.w);
    Normal3D modLocalNormal = normalize(Normal3D(-dhdu, dhdv, 1));
    return modLocalNormal;
}
CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(readModifiedNormalFromHeightMap);



static constexpr bool useEmbeddedVertexData = true;

template <OptixPrimitiveType curveType>
CUDA_DEVICE_FUNCTION CUDA_INLINE Normal3D calcCurveSurfaceNormal(
    const shared::GeometryInstanceData &geom, uint32_t primIndex, float curveParam, const Point3D &hp)
{
    using namespace shared;

    constexpr uint32_t numControlPoints = curve::getNumControlPoints<curveType>();
    float4 controlPoints[numControlPoints];
    if constexpr (useEmbeddedVertexData) {
        OptixTraversableHandle gasHandle = optixGetGASTraversableHandle();
        uint32_t sbtGasIndex = optixGetSbtGASIndex();
        if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR)
            optixGetLinearCurveVertexData(gasHandle, primIndex, sbtGasIndex, 0.0f, controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE)
            optixGetQuadraticBSplineVertexData(gasHandle, primIndex, sbtGasIndex, 0.0f, controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE)
            optixGetCubicBSplineVertexData(gasHandle, primIndex, sbtGasIndex, 0.0f, controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM)
            optixGetCatmullRomVertexData(gasHandle, primIndex, sbtGasIndex, 0.0f, controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER)
            optixGetCubicBezierVertexData(gasHandle, primIndex, sbtGasIndex, 0.0f, controlPoints);
    }
    else {
        uint32_t baseIndex = geom.segmentIndexBuffer[primIndex];
#pragma unroll
        for (int i = 0; i < numControlPoints; ++i) {
            const CurveVertex &v = geom.curveVertexBuffer[baseIndex + i];
            controlPoints[i] = make_float4(v.position.toNative(), v.width);
        }
    }

    curve::Evaluator<curveType> ce(controlPoints);
    float3 sn = ce.calcNormal(curveParam, hp.toNative());

    return Normal3D(sn);
}



CUDA_DEVICE_FUNCTION CUDA_INLINE void concentricSampleDisk(float u0, float u1, float* dx, float* dy) {
    float r, theta;
    float sx = 2 * u0 - 1;
    float sy = 2 * u1 - 1;

    if (sx == 0 && sy == 0) {
        *dx = 0;
        *dy = 0;
        return;
    }
    if (sx >= -sy) { // region 1 or 2
        if (sx > sy) { // region 1
            r = sx;
            theta = sy / sx;
        }
        else { // region 2
            r = sy;
            theta = 2 - sx / sy;
        }
    }
    else { // region 3 or 4
        if (sx > sy) {/// region 4
            r = -sy;
            theta = 6 + sx / sy;
        }
        else {// region 3
            r = -sx;
            theta = 4 + sy / sx;
        }
    }
    theta *= pi_v<float> / 4;
    *dx = r * cos(theta);
    *dy = r * sin(theta);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D cosineSampleHemisphere(float u0, float u1) {
    float x, y;
    concentricSampleDisk(u0, u1, &x, &y);
    return Vector3D(x, y, std::sqrt(std::fmax(0.0f, 1.0f - x * x - y * y)));
}



template <typename BSDFType>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setupBSDFBody(
    const shared::MaterialData &matData, Point2D texCoord, float mipLevel,
    uint32_t* bodyData, shared::BSDFFlags flags);



class LambertBRDF {
    RGB m_reflectance;

public:
    CUDA_DEVICE_FUNCTION LambertBRDF(const RGB &reflectance) :
        m_reflectance(reflectance) {}

    CUDA_DEVICE_FUNCTION void getSurfaceParameters(
        RGB* diffuseReflectance, RGB* specularReflectance, float* roughness) const {
        *diffuseReflectance = m_reflectance;
        *specularReflectance = RGB(0.0f, 0.0f, 0.0f);
        *roughness = 1.0f;
    }
    CUDA_DEVICE_FUNCTION RGB sampleThroughput(
        const Vector3D &vGiven, float uDir0, float uDir1,
        Vector3D* vSampled, float* dirPDensity) const
    {
        *vSampled = cosineSampleHemisphere(uDir0, uDir1);
        *dirPDensity = vSampled->z / pi_v<float>;
        if (vGiven.z <= 0.0f)
            vSampled->z *= -1;
        return m_reflectance;
    }
    CUDA_DEVICE_FUNCTION RGB evaluate(const Vector3D &vGiven, const Vector3D &vSampled) const {
        if (vGiven.z * vSampled.z > 0)
            return m_reflectance / pi_v<float>;
        else
            return RGB(0.0f, 0.0f, 0.0f);
    }
    CUDA_DEVICE_FUNCTION float evaluatePDF(const Vector3D &vGiven, const Vector3D &vSampled) const {
        if (vGiven.z * vSampled.z > 0)
            return fabs(vSampled.z) / pi_v<float>;
        else
            return 0.0f;
    }

    CUDA_DEVICE_FUNCTION RGB evaluateDHReflectanceEstimate(const Vector3D &vGiven) const {
        return m_reflectance;
    }
};

template <>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setupBSDFBody<LambertBRDF>(
    const shared::MaterialData &matData, Point2D texCoord, float mipLevel,
    uint32_t* bodyData, shared::BSDFFlags /*flags*/)
{
    float4 reflectance = sample<float4>(
        matData.asLambert.reflectance, matData.asLambert.reflectanceDimInfo, texCoord, mipLevel);
    auto &bsdfBody = *reinterpret_cast<LambertBRDF*>(bodyData);
    bsdfBody = LambertBRDF(RGB(getXYZ(reflectance)));
}



// DiffuseAndSpecularBRDFのDirectional-Hemispherical Reflectanceを事前計算して
// テクスチャー化した結果をフィッティングする。
// Diffuse、Specular成分はそれぞれ
// - baseColor * diffusePreInt(cosV, roughness)
// - specularF0 * specularPreIntA(cosV, roughness) + (1 - specularF0) * specularPreIntB(cosV, roughness)
// で表される。
// https://www.shadertoy.com/view/WtjfRD
CUDA_DEVICE_FUNCTION CUDA_INLINE void calcFittedPreIntegratedTerms(
    float cosV, float roughness,
    float* diffusePreInt, float* specularPreIntA, float* specularPreIntB)
{
        {
            float u = cosV;
            float v = roughness;
            float uu = u * u;
            float uv = u * v;
            float vv = v * v;

            *diffusePreInt = min(max(-0.417425f * uu +
                                     -0.958929f * uv +
                                     -0.096977f * vv +
                                     1.050356f * u +
                                     0.534528f * v +
                                     0.407112f * 1.0f,
                                     0.0f), 1.0f);
        }
        {
            float u = std::atan2(roughness, cosV);
            float v = std::sqrt(cosV * cosV + roughness * roughness);
            float uu = u * u;
            float uv = u * v;
            float vv = v * v;

            *specularPreIntA = min(max(0.133105f * uu +
                                       -0.278877f * uv +
                                       -0.417142f * vv +
                                       -0.192809f * u +
                                       0.426076f * v +
                                       0.996565f * 1.0f,
                                       0.0f), 1.0f);
            *specularPreIntB = min(max(0.055070f * uu +
                                       -0.163511f * uv +
                                       1.211598f * vv +
                                       0.089837f * u +
                                       -1.956888f * v +
                                       0.741397f * 1.0f,
                                       0.0f), 1.0f);
        }
}

#define USE_HEIGHT_CORRELATED_SMITH
//#define USE_FITTED_PRE_INTEGRATION_FOR_WEIGHTS
//#define USE_FITTED_PRE_INTEGRATION_FOR_DH_REFLECTANCE

class DiffuseAndSpecularBRDF {
    struct GGXMicrofacetDistribution {
        float alpha_g;

        CUDA_DEVICE_FUNCTION float evaluate(const Normal3D &m) const {
            if (m.z <= 0.0f)
                return 0.0f;
            float temp = pow2(m.x) + pow2(m.y) + pow2(m.z * alpha_g);
            return pow2(alpha_g) / (pi_v<float> * pow2(temp));
        }
        CUDA_DEVICE_FUNCTION float evaluateSmithG1(const Vector3D &v, const Normal3D &m) const {
            if (dot(v, m) * v.z <= 0)
                return 0.0f;
            float temp = pow2(alpha_g) * (pow2(v.x) + pow2(v.y)) / pow2(v.z);
            return 2 / (1 + std::sqrt(1 + temp));
        }
        CUDA_DEVICE_FUNCTION float evaluateHeightCorrelatedSmithG(
            const Vector3D &v1, const Vector3D &v2, const Normal3D &m)
        {
            float alpha_g2_tanTheta2_1 = pow2(alpha_g) * (pow2(v1.x) + pow2(v1.y)) / pow2(v1.z);
            float alpha_g2_tanTheta2_2 = pow2(alpha_g) * (pow2(v2.x) + pow2(v2.y)) / pow2(v2.z);
            float Lambda1 = (-1 + std::sqrt(1 + alpha_g2_tanTheta2_1)) / 2;
            float Lambda2 = (-1 + std::sqrt(1 + alpha_g2_tanTheta2_2)) / 2;
            float chi1 = (dot(v1, m) / v1.z) > 0 ? 1 : 0;
            float chi2 = (dot(v2, m) / v2.z) > 0 ? 1 : 0;
            return chi1 * chi2 / (1 + Lambda1 + Lambda2);
        }
        CUDA_DEVICE_FUNCTION float sample(
            const Vector3D &v, float u0, float u1,
            Normal3D* m, float* mPDensity) const
        {
            // stretch view
            Vector3D sv = normalize(Vector3D(alpha_g * v.x, alpha_g * v.y, v.z));

            // orthonormal basis
            float distIn2D = std::sqrt(sv.x * sv.x + sv.y * sv.y);
            float recDistIn2D = 1.0f / distIn2D;
            Vector3D T1 = (sv.z < 0.9999f) ?
                Vector3D(sv.y * recDistIn2D, -sv.x * recDistIn2D, 0) :
                Vector3D(1, 0, 0);
            Vector3D T2(T1.y * sv.z, -T1.x * sv.z, distIn2D);

            // sample point with polar coordinates (r, phi)
            float a = 1.0f / (1.0f + sv.z);
            float r = std::sqrt(u0);
            float phi = pi_v<float> * ((u1 < a) ? u1 / a : 1 + (u1 - a) / (1.0f - a));
            float sinPhi, cosPhi;
            stc::sincos(phi, &sinPhi, &cosPhi);
            float P1 = r * cosPhi;
            float P2 = r * sinPhi * ((u1 < a) ? 1.0f : sv.z);

            // compute normal
            *m = Normal3D(P1 * T1 + P2 * T2 + std::sqrt(1.0f - P1 * P1 - P2 * P2) * sv);

            // unstretch
            *m = normalize(Normal3D(alpha_g * m->x, alpha_g * m->y, m->z));

            float D = evaluate(*m);
            *mPDensity = evaluateSmithG1(v, *m) * absDot(v, *m) * D / std::fabs(v.z);

            return D;
        }
        CUDA_DEVICE_FUNCTION float evaluatePDF(const Vector3D &v, const Normal3D &m) {
            return evaluateSmithG1(v, m) * absDot(v, m) * evaluate(m) / std::fabs(v.z);
        }
    };

protected:
    RGB m_diffuseColor;
    RGB m_specularF0Color;
    float m_roughness;

public:
    CUDA_DEVICE_FUNCTION DiffuseAndSpecularBRDF() {}
    CUDA_DEVICE_FUNCTION DiffuseAndSpecularBRDF(
        const RGB &diffuseColor, const RGB &specularF0Color, float smoothness)
    {
        m_diffuseColor = diffuseColor;
        m_specularF0Color = specularF0Color;
        m_roughness = 1 - smoothness;
    }

    CUDA_DEVICE_FUNCTION void getSurfaceParameters(
        RGB* diffuseReflectance, RGB* specularReflectance, float* roughness) const
    {
        *diffuseReflectance = m_diffuseColor;
        *specularReflectance = m_specularF0Color;
        *roughness = m_roughness;
    }
    CUDA_DEVICE_FUNCTION RGB sampleThroughput(
        const Vector3D &vGiven, float uDir0, float uDir1,
        Vector3D* vSampled, float* dirPDensity) const
    {
        GGXMicrofacetDistribution ggx;
        ggx.alpha_g = m_roughness * m_roughness;

        bool entering = vGiven.z >= 0.0f;
        Vector3D dirL;
        Vector3D dirV = entering ? vGiven : -vGiven;

        float oneMinusDotVN5 = pow5(1 - dirV.z);

#if defined(USE_FITTED_PRE_INTEGRATION_FOR_WEIGHTS)
        float diffusePreInt;
        float specularPreIntA, specularPreIntB;
        calcFittedPreIntegratedTerms(dirV.z, m_roughness, &diffusePreInt, &specularPreIntA, &specularPreIntB);

        float diffuseWeight = sRGB_calcLuminance(m_diffuseColor * diffusePreInt);
        float specularWeight = sRGB_calcLuminance(m_specularF0Color * specularPreIntA + (RGB(1.0f) - m_specularF0Color) * specularPreIntB);
#else
        float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * vGiven.z * vGiven.z;
        float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
        float iBaseColor = sRGB_calcLuminance(m_diffuseColor) * pow2(expectedDiffuseFresnel) *
            lerp(1.0f, 1.0f / 1.51f, m_roughness);

        float expectedOneMinusDotVH5 = pow5(1 - dirV.z);
        float iSpecularF0 = sRGB_calcLuminance(m_specularF0Color);

        float diffuseWeight = iBaseColor;
        float specularWeight = lerp(iSpecularF0, 1.0f, expectedOneMinusDotVH5);
#endif
        float sumWeights = diffuseWeight + specularWeight;
        if (sumWeights == 0.0f) {
            *dirPDensity = 0.0f;
            return RGB(0.0f);
        }

        float uComponent = uDir1;

        float diffuseDirPDF, specularDirPDF;
        Normal3D m;
        float dotLH;
        float D;
        if (sumWeights * uComponent < diffuseWeight) {
            uDir1 = (sumWeights * uComponent - 0) / diffuseWeight;

            // JP: コサイン分布からサンプルする。
            // EN: sample based on cosine distribution.
            dirL = cosineSampleHemisphere(uDir0, uDir1);
            diffuseDirPDF = dirL.z / pi_v<float>;

            // JP: 同じ方向サンプルをスペキュラー層からサンプルする確率密度を求める。
            // EN: calculate PDF to generate the sampled direction from the specular layer.
            m = Normal3D(halfVector(dirL, dirV));
            dotLH = min(dot(dirL, m), 1.0f);
            float commonPDFTerm = 1.0f / (4 * dotLH);
            specularDirPDF = commonPDFTerm * ggx.evaluatePDF(dirV, m);

            D = ggx.evaluate(m);
        }
        else {
            uDir1 = (sumWeights * uComponent - diffuseWeight) / specularWeight;

            // JP: スペキュラー層のマイクロファセット分布からサンプルする。
            // EN: sample based on the specular microfacet distribution.
            float mPDF;
            D = ggx.sample(dirV, uDir0, uDir1, &m, &mPDF);
            float dotVH = min(dot(dirV, m), 1.0f);
            dotLH = dotVH;
            dirL = Vector3D(2 * dotVH * m - dirV);
            if (dirL.z * dirV.z <= 0) {
                *dirPDensity = 0.0f;
                return RGB(0.0f);
            }
            float commonPDFTerm = 1.0f / (4 * dotLH);
            specularDirPDF = commonPDFTerm * mPDF;

            // JP: 同じ方向サンプルをコサイン分布からサンプルする確率密度を求める。
            // EN: calculate PDF to generate the sampled direction from the cosine distribution.
            diffuseDirPDF = dirL.z / pi_v<float>;
        }

        float oneMinusDotLH5 = pow5(1 - dotLH);

#if defined(USE_HEIGHT_CORRELATED_SMITH)
        float G = ggx.evaluateHeightCorrelatedSmithG(dirL, dirV, m);
#else
        float G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
#endif
        constexpr float F90 = 1.0f;
        RGB F = lerp(m_specularF0Color, RGB(F90), oneMinusDotLH5);

        float microfacetDenom = 4 * dirL.z * dirV.z;
        RGB specularValue = F * ((D * G) / microfacetDenom);
        if (G == 0)
            specularValue = RGB(0.0f);

        float F_D90 = 0.5f * m_roughness + 2 * m_roughness * dotLH * dotLH;
        float oneMinusDotLN5 = pow5(1 - dirL.z);
        float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
        float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);
        RGB diffuseValue = m_diffuseColor *
            (diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, m_roughness) / pi_v<float>);

        RGB ret = diffuseValue + specularValue;

        *vSampled = entering ? dirL : -dirL;

        // PDF based on one-sample model MIS.
        *dirPDensity = (diffuseDirPDF * diffuseWeight + specularDirPDF * specularWeight) / sumWeights;

        ret *= dirL.z / *dirPDensity;

        return ret;
    }
    CUDA_DEVICE_FUNCTION RGB evaluate(const Vector3D &vGiven, const Vector3D &vSampled) const {
        GGXMicrofacetDistribution ggx;
        ggx.alpha_g = m_roughness * m_roughness;

        if (vSampled.z * vGiven.z <= 0)
            return RGB(0.0f);

        bool entering = vGiven.z >= 0.0f;
        Vector3D dirV = entering ? vGiven : -vGiven;
        Vector3D dirL = entering ? vSampled : -vSampled;

        Normal3D m(halfVector(dirL, dirV));
        float dotLH = dot(dirL, m);

        float oneMinusDotLH5 = pow5(1 - dotLH);

        float D = ggx.evaluate(m);
#if defined(USE_HEIGHT_CORRELATED_SMITH)
        float G = ggx.evaluateHeightCorrelatedSmithG(dirL, dirV, m);
#else
        float G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
#endif
        constexpr float F90 = 1.0f;
        RGB F = lerp(m_specularF0Color, RGB(F90), oneMinusDotLH5);

        float microfacetDenom = 4 * dirL.z * dirV.z;
        RGB specularValue = F * ((D * G) / microfacetDenom);
        if (G == 0)
            specularValue = RGB(0.0f);

        float F_D90 = 0.5f * m_roughness + 2 * m_roughness * dotLH * dotLH;
        float oneMinusDotVN5 = pow5(1 - dirV.z);
        float oneMinusDotLN5 = pow5(1 - dirL.z);
        float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
        float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);

        RGB diffuseValue = m_diffuseColor *
            (diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, m_roughness) / pi_v<float>);

        RGB ret = diffuseValue + specularValue;

        return ret;
    }
    CUDA_DEVICE_FUNCTION float evaluatePDF(const Vector3D &vGiven, const Vector3D &vSampled) const {
        GGXMicrofacetDistribution ggx;
        ggx.alpha_g = m_roughness * m_roughness;

        bool entering = vGiven.z >= 0.0f;
        Vector3D dirV = entering ? vGiven : -vGiven;
        Vector3D dirL = entering ? vSampled : -vSampled;

        Normal3D m(halfVector(dirL, dirV));
        float dotLH = dot(dirL, m);
        float commonPDFTerm = 1.0f / (4 * dotLH);

#if defined(USE_FITTED_PRE_INTEGRATION_FOR_WEIGHTS)
        float diffusePreInt;
        float specularPreIntA, specularPreIntB;
        calcFittedPreIntegratedTerms(dirV.z, m_roughness, &diffusePreInt, &specularPreIntA, &specularPreIntB);

        float diffuseWeight = sRGB_calcLuminance(m_diffuseColor * diffusePreInt);
        float specularWeight = sRGB_calcLuminance(m_specularF0Color * specularPreIntA + (RGB(1.0f) - m_specularF0Color) * specularPreIntB);
#else
        float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * vGiven.z * vGiven.z;
        float oneMinusDotVN5 = pow5(1 - dirV.z);
        float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
        float iBaseColor = sRGB_calcLuminance(m_diffuseColor) * pow2(expectedDiffuseFresnel) *
            lerp(1.0f, 1.0f / 1.51f, m_roughness);

        float expectedOneMinusDotVH5 = pow5(1 - dirV.z);
        float iSpecularF0 = sRGB_calcLuminance(m_specularF0Color);

        float diffuseWeight = iBaseColor;
        float specularWeight = lerp(iSpecularF0, 1.0f, expectedOneMinusDotVH5);
#endif

        float sumWeights = diffuseWeight + specularWeight;
        if (sumWeights == 0.0f)
            return 0.0f;

        float diffuseDirPDF = dirL.z / pi_v<float>;
        float specularDirPDF = commonPDFTerm * ggx.evaluatePDF(dirV, m);

        float ret = (diffuseDirPDF * diffuseWeight + specularDirPDF * specularWeight) / sumWeights;

        return ret;
    }

    CUDA_DEVICE_FUNCTION RGB evaluateDHReflectanceEstimate(const Vector3D &vGiven) const {
        bool entering = vGiven.z >= 0.0f;
        Vector3D dirV = entering ? vGiven : -vGiven;

#if defined(USE_FITTED_PRE_INTEGRATION_FOR_DH_REFLECTANCE)
        float diffusePreInt;
        float specularPreIntA, specularPreIntB;
        calcFittedPreIntegratedTerms(dirV.z, m_roughness, &diffusePreInt, &specularPreIntA, &specularPreIntB);

        RGB diffuseDHR = m_diffuseColor * diffusePreInt;
        RGB specularDHR = m_specularF0Color * specularPreIntA + (RGB(1.0f) - m_specularF0Color) * specularPreIntB;
#else
        float expectedCosTheta_d = dirV.z;
        float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * pow2(expectedCosTheta_d);
        float oneMinusDotVN5 = pow5(1 - dirV.z);
        float expectedDiffFGiven = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
        float expectedDiffFSampled = 1.0f; // ad-hoc
        RGB diffuseDHR = m_diffuseColor *
            expectedDiffFGiven * expectedDiffFSampled * lerp(1.0f, 1.0f / 1.51f, m_roughness);

        //float expectedOneMinusDotVH5 = oneMinusDotVN5;
        // (1 - m_roughness) is an ad-hoc adjustment.
        float expectedOneMinusDotVH5 = pow5(1 - dirV.z) * (1 - m_roughness);

        RGB specularDHR = lerp(m_specularF0Color, RGB(1.0f), expectedOneMinusDotVH5);
#endif

        return min(diffuseDHR + specularDHR, RGB(1.0f));
    }
};

class SimplePBR_BRDF : public DiffuseAndSpecularBRDF {
public:
    CUDA_DEVICE_FUNCTION SimplePBR_BRDF(
        const RGB &baseColor, float reflectance, float smoothness, float metallic)
    {
        m_diffuseColor = baseColor * (1 - metallic);
        m_specularF0Color = RGB(0.16f * pow2(reflectance) * (1 - metallic)) + baseColor * metallic;
        m_roughness = 1 - smoothness;
    }
};

template <>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setupBSDFBody<DiffuseAndSpecularBRDF>(
    const shared::MaterialData &matData, Point2D texCoord, float mipLevel,
    uint32_t* bodyData, shared::BSDFFlags flags)
{
    float4 diffuseColor = sample<float4>(
        matData.asDiffuseAndSpecular.diffuse,
        matData.asDiffuseAndSpecular.diffuseDimInfo,
        texCoord, mipLevel);
    float4 specularF0Color = sample<float4>(
        matData.asDiffuseAndSpecular.specular,
        matData.asDiffuseAndSpecular.specularDimInfo,
        texCoord, mipLevel);
    float smoothness = sample<float>(
        matData.asDiffuseAndSpecular.smoothness,
        matData.asDiffuseAndSpecular.smoothnessDimInfo,
        texCoord, mipLevel);
    bool regularize = (flags & shared::BSDFFlags::Regularize) != 0;
    if (regularize)
        smoothness *= 0.5f;
    auto &bsdfBody = *reinterpret_cast<DiffuseAndSpecularBRDF*>(bodyData);
    bsdfBody = DiffuseAndSpecularBRDF(
        RGB(getXYZ(diffuseColor)),
        RGB(getXYZ(specularF0Color)),
        min(smoothness, 0.999f));
}

template <>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setupBSDFBody<SimplePBR_BRDF>(
    const shared::MaterialData &matData, Point2D texCoord, float mipLevel,
    uint32_t* bodyData, shared::BSDFFlags flags)
{
    float4 baseColor_opacity = sample<float4>(
        matData.asSimplePBR.baseColor_opacity,
        matData.asSimplePBR.baseColor_opacity_dimInfo,
        texCoord, mipLevel);
    float4 occlusion_roughness_metallic = sample<float4>(
        matData.asSimplePBR.occlusion_roughness_metallic,
        matData.asSimplePBR.occlusion_roughness_metallic_dimInfo,
        texCoord, mipLevel);
    RGB baseColor(baseColor_opacity.x, baseColor_opacity.y, baseColor_opacity.z);
    float smoothness = min(1.0f - occlusion_roughness_metallic.y, 0.999f);
    float metallic = occlusion_roughness_metallic.z;
    bool regularize = (flags & shared::BSDFFlags::Regularize) != 0;
    if (regularize)
        smoothness *= 0.5f;
    auto &bsdfBody = *reinterpret_cast<SimplePBR_BRDF*>(bodyData);
    bsdfBody = SimplePBR_BRDF(baseColor, 0.5f, smoothness, metallic);
}



#define DEFINE_BSDF_CALLABLES(BSDFType) \
    RT_CALLABLE_PROGRAM void RT_DC_NAME(BSDFType ## _getSurfaceParameters)(\
        const uint32_t* data, RGB* diffuseReflectance, RGB* specularReflectance, float* roughness)\
    {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.getSurfaceParameters(diffuseReflectance, specularReflectance, roughness);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _getSurfaceParameters);\
    RT_CALLABLE_PROGRAM RGB RT_DC_NAME(BSDFType ## _sampleThroughput)(\
        const uint32_t* data, const Vector3D &vGiven, float uDir0, float uDir1,\
        Vector3D* vSampled, float* dirPDensity)\
    {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.sampleThroughput(vGiven, uDir0, uDir1, vSampled, dirPDensity);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _sampleThroughput);\
    RT_CALLABLE_PROGRAM RGB RT_DC_NAME(BSDFType ## _evaluate)(\
        const uint32_t* data, const Vector3D &vGiven, const Vector3D &vSampled)\
    {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluate(vGiven, vSampled);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _evaluate);\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(BSDFType ## _evaluatePDF)(\
        const uint32_t* data, const Vector3D &vGiven, const Vector3D &vSampled)\
    {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluatePDF(vGiven, vSampled);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _evaluatePDF);\
    RT_CALLABLE_PROGRAM RGB RT_DC_NAME(BSDFType ## _evaluateDHReflectanceEstimate)(\
        const uint32_t* data, const Vector3D &vGiven)\
    {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluateDHReflectanceEstimate(vGiven);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _evaluateDHReflectanceEstimate);

DEFINE_BSDF_CALLABLES(LambertBRDF);
DEFINE_BSDF_CALLABLES(DiffuseAndSpecularBRDF);

#undef DEFINE_SETUP_BSDF_CALLABLE

#define DEFINE_SETUP_BSDF_CALLABLE(BSDFType) \
    RT_CALLABLE_PROGRAM void RT_DC_NAME(setup ## BSDFType)(\
        const shared::MaterialData &matData, Point2D texCoord, float mipLevel,\
        uint32_t* bodyData, shared::BSDFFlags flags)\
    {\
        setupBSDFBody<BSDFType>(matData, texCoord, mipLevel, bodyData, flags);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(setup ## BSDFType)

DEFINE_SETUP_BSDF_CALLABLE(LambertBRDF);
DEFINE_SETUP_BSDF_CALLABLE(DiffuseAndSpecularBRDF);
DEFINE_SETUP_BSDF_CALLABLE(SimplePBR_BRDF);

#undef DEFINE_SETUP_BSDF_CALLABLE



struct BSDF {
#if defined(USE_HARD_CODED_BSDF_FUNCTIONS)
    uint32_t m_data[(sizeof(HARD_CODED_BSDF) + 3) / 4];
#else
    static constexpr uint32_t NumDwords = 16;
    shared::BSDFGetSurfaceParameters m_getSurfaceParameters;
    shared::BSDFSampleThroughput m_sampleThroughput;
    shared::BSDFEvaluate m_evaluate;
    shared::BSDFEvaluatePDF m_evaluatePDF;
    shared::BSDFEvaluateDHReflectanceEstimate m_evaluateDHReflectanceEstimate;
    uint32_t m_data[NumDwords];
#endif

    CUDA_DEVICE_FUNCTION void setup(
        const shared::MaterialData &matData, const Point2D &texCoord, float mipLevel,
        shared::BSDFFlags flags = shared::BSDFFlags::None)
    {
#if defined(USE_HARD_CODED_BSDF_FUNCTIONS)
        setupBSDFBody<HARD_CODED_BSDF>(matData, texCoord, mipLevel, m_data, flags);
#else
        m_getSurfaceParameters = matData.bsdfGetSurfaceParameters;
        m_sampleThroughput = matData.bsdfSampleThroughput;
        m_evaluate = matData.bsdfEvaluate;
        m_evaluatePDF = matData.bsdfEvaluatePDF;
        m_evaluateDHReflectanceEstimate = matData.bsdfEvaluateDHReflectanceEstimate;
        matData.setupBSDFBody(matData, texCoord, mipLevel, m_data, flags);
#endif
    }
    CUDA_DEVICE_FUNCTION void getSurfaceParameters(
        RGB* diffuseReflectance, RGB* specularReflectance, float* roughness) const
    {
#if defined(USE_HARD_CODED_BSDF_FUNCTIONS)
        auto &bsdf = *reinterpret_cast<const HARD_CODED_BSDF*>(m_data);
        return bsdf.getSurfaceParameters(diffuseReflectance, specularReflectance, roughness);
#else
        return m_getSurfaceParameters(m_data, diffuseReflectance, specularReflectance, roughness);
#endif
    }
    CUDA_DEVICE_FUNCTION RGB sampleThroughput(
        const Vector3D &vGiven, float uDir0, float uDir1,
        Vector3D* vSampled, float* dirPDensity) const
    {
#if defined(USE_HARD_CODED_BSDF_FUNCTIONS)
        auto &bsdf = *reinterpret_cast<const HARD_CODED_BSDF*>(m_data);
        return bsdf.sampleThroughput(vGiven, uDir0, uDir1, vSampled, dirPDensity);
#else
        return m_sampleThroughput(m_data, vGiven, uDir0, uDir1, vSampled, dirPDensity);
#endif
    }
    CUDA_DEVICE_FUNCTION RGB evaluate(const Vector3D &vGiven, const Vector3D &vSampled) const {
#if defined(USE_HARD_CODED_BSDF_FUNCTIONS)
        auto &bsdf = *reinterpret_cast<const HARD_CODED_BSDF*>(m_data);
        return bsdf.evaluate(vGiven, vSampled);
#else
        return m_evaluate(m_data, vGiven, vSampled);
#endif
    }
    CUDA_DEVICE_FUNCTION float evaluatePDF(const Vector3D &vGiven, const Vector3D &vSampled) const {
#if defined(USE_HARD_CODED_BSDF_FUNCTIONS)
        auto &bsdf = *reinterpret_cast<const HARD_CODED_BSDF*>(m_data);
        return bsdf.evaluatePDF(vGiven, vSampled);
#else
        return m_evaluatePDF(m_data, vGiven, vSampled);
#endif
    }
    CUDA_DEVICE_FUNCTION RGB evaluateDHReflectanceEstimate(const Vector3D &vGiven) const {
#if defined(USE_HARD_CODED_BSDF_FUNCTIONS)
        auto &bsdf = *reinterpret_cast<const HARD_CODED_BSDF*>(m_data);
        return bsdf.evaluateDHReflectanceEstimate(vGiven);
#else
        return m_evaluateDHReflectanceEstimate(m_data, vGiven);
#endif
    }
};



CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D transformPointFromObjectToWorldSpace(const Point3D &p) {
    float3 xfmP = optixTransformPointFromObjectToWorldSpace(make_float3(p.x, p.y, p.z));
    return Point3D(xfmP.x, xfmP.y, xfmP.z);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D transformVectorFromObjectToWorldSpace(const Vector3D &v) {
    float3 xfmV = optixTransformVectorFromObjectToWorldSpace(make_float3(v.x, v.y, v.z));
    return Vector3D(xfmV.x, xfmV.y, xfmV.z);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Normal3D transformNormalFromObjectToWorldSpace(const Normal3D &n) {
    float3 xfmN = optixTransformNormalFromObjectToWorldSpace(make_float3(n.x, n.y, n.z));
    return Normal3D(xfmN.x, xfmN.y, xfmN.z);
}
