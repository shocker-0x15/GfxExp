#pragma once

#include "../nrtdsm_shared.h"

using namespace shared;

#define DEBUG_TRAVERSAL 0

CUDA_DEVICE_FUNCTION CUDA_INLINE bool isDebugPixel() {
    return optixGetLaunchIndex().x == 935 && optixGetLaunchIndex().y == 358;
    //return isCursorPixel();
}



CUDA_DEVICE_FUNCTION CUDA_INLINE bool testRayVsTriangle(
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    float* const hitDist, Normal3D* const hitNormal, float* const bcB, float* const bcC) {
    const Vector3D eAB = pB - pA;
    const Vector3D eCA = pA - pC;
    *hitNormal = static_cast<Normal3D>(cross(eCA, eAB));

    const Vector3D e = (1.0f / dot(*hitNormal, rayDir)) * (pA - rayOrg);
    const Vector3D i = cross(rayDir, e);

    *bcB = dot(i, eCA);
    *bcC = dot(i, eAB);
    *hitDist = dot(*hitNormal, e);

    return
        ((*hitDist < distMax) && (*hitDist > distMin)
         && (*bcB >= 0.0f) && (*bcC >= 0.0f) && (*bcB + *bcC <= 1));
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D restoreTriangleHitPoint(
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const float bcB, const float bcC, Normal3D* const hitNormal) {
    *hitNormal = static_cast<Normal3D>(cross(pB - pA, pC - pA));
    return (1 - (bcB + bcC)) * pA + bcB * pB + bcC * pC;
}

// Reference: Chapter 8. Cool Patches: A Geometric Approach to Ray/Bilinear Patch Intersections
//            Ray Tracing Gems
CUDA_DEVICE_FUNCTION CUDA_INLINE bool testRayVsBilinearPatch(
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const Point3D &pA, const Point3D &pB, const Point3D &pC, const Point3D &pD,
    float* const hitDist, Normal3D* const hitNormal, float* const u, float* const v) {
    const Vector3D eAB = pB - pA;
    const Vector3D eAC = pC - pA;
    const Vector3D eBD = pD - pB;
    const Vector3D eCD = pD - pC;
    const Vector3D pARel = pA - rayOrg;
    const Vector3D pBRel = pB - rayOrg;

    float u1, u2;
    {
        const Vector3D qN = cross(eAB, -eCD);
        const float a = dot(qN, rayDir);
        float b = dot(cross(pBRel, rayDir), eBD);
        const float c = dot(cross(pARel, rayDir), eAC);
        b -= a + c;

        const float det = pow2(b) - 4 * a * c;
        if (det < 0)
            return false;

        if (a == 0) {
            u1 = -c / b;
            u2 = -1;
        }
        else {
            const float sqrtDet = std::sqrt(det);
            const float temp = -0.5f * (b + std::copysign(sqrtDet, b));
            u1 = temp / a;
            u2 = c / temp;
        }
    }

    *hitDist = distMax;

    const auto find_v_t = [&](const float uu) {
        if (uu >= 0 && uu <= 1) {
            const Vector3D pAB = lerp(pARel, pBRel, uu);
            const Vector3D pCD = lerp(eAC, eBD, uu);
            Vector3D n = cross(rayDir, pCD);
            const float recDet = 1.0f / dot(n, n);
            n = cross(n, pAB);
            const float tt = dot(n, pCD) * recDet;
            const float vv = dot(n, rayDir) * recDet;
            if (vv >= 0 && vv <= 1 && tt > distMin && tt < *hitDist) {
                *hitDist = tt;
                *u = uu;
                *v = vv;
                return true;
            }
        }
        return false;
    };

    bool hit = false;
    hit |= find_v_t(u1);
    hit |= find_v_t(u2);
    if (!hit)
        return false;

    const Vector3D dpdu = lerp(eAB, eCD, *v);
    const Vector3D dpdv = lerp(eAC, eBD, *u);
    *hitNormal = static_cast<Normal3D>(cross(dpdu, dpdv));

    return true;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D restoreBilinearPatchHitPoint(
    const Point3D &pA, const Point3D &pB, const Point3D &pC, const Point3D &pD,
    const float u, const float v, Normal3D* const hitNormal) {
    const Vector3D eAB = pB - pA;
    const Vector3D eAC = pC - pA;
    const Vector3D eBD = pD - pB;
    const Vector3D eCD = pD - pC;
    const Vector3D dpdu = lerp(eAB, eCD, v);
    const Vector3D dpdv = lerp(eAC, eBD, u);
    *hitNormal = static_cast<Normal3D>(cross(dpdu, dpdv));
    return (1 - u) * (1 - v) * pA + u * (1 - v) * pB + (1 - u) * v * pC + u * v * pD;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE bool testRayVsPrism(
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const Point3D &pD, const Point3D &pE, const Point3D &pF,
    float* const hitDist, float* const hitParam0, float* const hitParam1,
    bool* isFrontHit) {
    *hitDist = distMax;

    const auto updateHit = [&]
    (const float t, const Normal3D &n, uint32_t faceID, const float u, const float v) {
        *hitDist = t;
        *hitParam0 = faceID + u;
        *hitParam1 = v;
        //*hitNormal = normalize(n);
        *isFrontHit = dot(n, rayDir) < 0;
    };

    float tt;
    Normal3D nn;
    float uu, vv;
    if (testRayVsTriangle(
        rayOrg, rayDir, distMin, *hitDist,
        pC, pB, pA,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 0, uu, vv);
    }
    if (testRayVsTriangle(
        rayOrg, rayDir, distMin, *hitDist,
        pD, pE, pF,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 1, uu, vv);
    }
    if (testRayVsBilinearPatch(
        rayOrg, rayDir, distMin, *hitDist,
        pA, pB, pD, pE,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 2, uu, vv);
    }
    if (testRayVsBilinearPatch(
        rayOrg, rayDir, distMin, *hitDist,
        pB, pC, pE, pF,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 3, uu, vv);
    }
    if (testRayVsBilinearPatch(
        rayOrg, rayDir, distMin, *hitDist,
        pC, pA, pF, pD,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 4, uu, vv);
    }

    return *hitDist < distMax;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE bool testRayVsPrism(
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const Point3D &pD, const Point3D &pE, const Point3D &pF,
    float* const hitDistMin, float* const hitDistMax) {
    *hitDistMin = INFINITY;
    *hitDistMax = -INFINITY;

    float tt;
    Normal3D nn;
    float uu, vv;
    if (testRayVsTriangle(
        rayOrg, rayDir, -INFINITY, INFINITY,
        pC, pB, pA,
        &tt, &nn, &uu, &vv)) {
        *hitDistMin = std::fmin(*hitDistMin, tt);
        *hitDistMax = std::fmax(*hitDistMax, tt);
    }
    if (testRayVsTriangle(
        rayOrg, rayDir, -INFINITY, INFINITY,
        pD, pE, pF,
        &tt, &nn, &uu, &vv)) {
        *hitDistMin = std::fmin(*hitDistMin, tt);
        *hitDistMax = std::fmax(*hitDistMax, tt);
    }
    if (testRayVsBilinearPatch(
        rayOrg, rayDir, -INFINITY, INFINITY,
        pA, pB, pD, pE,
        &tt, &nn, &uu, &vv)) {
        *hitDistMin = std::fmin(*hitDistMin, tt);
        *hitDistMax = std::fmax(*hitDistMax, tt);
    }
    if (testRayVsBilinearPatch(
        rayOrg, rayDir, -INFINITY, INFINITY,
        pB, pC, pE, pF,
        &tt, &nn, &uu, &vv)) {
        *hitDistMin = std::fmin(*hitDistMin, tt);
        *hitDistMax = std::fmax(*hitDistMax, tt);
    }
    if (testRayVsBilinearPatch(
        rayOrg, rayDir, -INFINITY, INFINITY,
        pC, pA, pF, pD,
        &tt, &nn, &uu, &vv)) {
        *hitDistMin = std::fmin(*hitDistMin, tt);
        *hitDistMax = std::fmax(*hitDistMax, tt);
    }

    *hitDistMin = std::fmax(*hitDistMin, distMin);
    *hitDistMax = std::fmin(*hitDistMax, distMax);

    return *hitDistMin <= *hitDistMax && *hitDistMax > 0.0f;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D restorePrismHitPoint(
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const Point3D &pD, const Point3D &pE, const Point3D &pF,
    const float hitParam0, const float hitParam1,
    Normal3D* const hitNormal) {
    const uint32_t faceID = static_cast<uint32_t>(hitParam0);
    const float u = std::fmod(hitParam0, 1.0f);
    const float v = std::fmod(hitParam1, 1.0f);
    if (faceID == 0)
        return restoreTriangleHitPoint(pC, pB, pA, u, v, hitNormal);
    else if (faceID == 1)
        return restoreTriangleHitPoint(pD, pE, pF, u, v, hitNormal);
    else if (faceID == 2)
        return restoreBilinearPatchHitPoint(pA, pB, pD, pE, u, v, hitNormal);
    else if (faceID == 3)
        return restoreBilinearPatchHitPoint(pB, pC, pE, pF, u, v, hitNormal);
    else if (faceID == 4)
        return restoreBilinearPatchHitPoint(pC, pA, pF, pD, u, v, hitNormal);
    return Point3D(NAN);
}



CUDA_DEVICE_KERNEL void RT_IS_NAME(prism)() {
    const auto sbtr = HitGroupSBTRecordData::get();
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];
    const GeometryInstanceDataForNRTDSM &nrtdsmGeomInst = plp.s->geomInstNrtdsmDataBuffer[sbtr.geomInstSlot];

    const uint32_t primIdx = optixGetPrimitiveIndex();
    const Triangle &tri = geomInst.triangleBuffer[primIdx];
    const Vertex &vA = geomInst.vertexBuffer[tri.index0];
    const Vertex &vB = geomInst.vertexBuffer[tri.index1];
    const Vertex &vC = geomInst.vertexBuffer[tri.index2];
    const NRTDSMTriangleAuxInfo &dispTriAuxInfo = nrtdsmGeomInst.dispTriAuxInfoBuffer[primIdx];
    const float minHeight = dispTriAuxInfo.minHeight;
    const float maxHeight = minHeight + dispTriAuxInfo.amplitude;
    const Point3D pA = vA.position + minHeight * vA.normal;
    const Point3D pB = vB.position + minHeight * vB.normal;
    const Point3D pC = vC.position + minHeight * vC.normal;
    const Point3D pD = vA.position + maxHeight * vA.normal;
    const Point3D pE = vB.position + maxHeight * vB.normal;
    const Point3D pF = vC.position + maxHeight * vC.normal;
    float hitDist;
    float hitParam0, hitParam1;
    bool isFrontHit;
    const bool hit = testRayVsPrism(
        Point3D(optixGetObjectRayOrigin()), Vector3D(optixGetObjectRayDirection()),
        optixGetRayTmin(), optixGetRayTmax(),
        pA, pB, pC,
        pD, pE, pF,
        &hitDist, &hitParam0, &hitParam1, &isFrontHit);
    if (!hit)
        return;

    //if (isCursorPixel() && getDebugPrintEnabled()) {
    //    printf("frame %u\n", plp.f->frameIndex);
    //    printf("height: %g, %g\n", minHeight, maxHeight);
    //    printf(
    //        "org: (%g, %g, %g), dir: (%g, %g, %g)\n",
    //        v3print(optixGetObjectRayOrigin()),
    //        v3print(optixGetObjectRayDirection()));
    //    printf("dist: %g - %g\n", optixGetRayTmin(), optixGetRayTmax());
    //    printf(
    //        "(%g, %g, %g), (%g, %g, %g), (%g, %g, %g)\n",
    //        v3print(pA), v3print(pB), v3print(pC));
    //    printf(
    //        "(%g, %g, %g), (%g, %g, %g), (%g, %g, %g)\n",
    //        v3print(pD), v3print(pE), v3print(pF));
    //    printf("params: %g, %g\n", hitParam0, hitParam1);
    //}

    PrismAttributeSignature::reportIntersection(
        hitDist,
        isFrontHit ? CustomHitKind_PrismFrontFace : CustomHitKind_PrismBackFace,
        hitParam0, hitParam1);
}



CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluateQuadraticPolynomial(
    const float a, const float b, const float c, const float x) {
    return (a * x + b) * x + c;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluateCubicPolynomial(
    const float a, const float b, const float c, const float d, const float x) {
    return ((a * x + b) * x + c) * x + d;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t solveQuadraticEquation(
    const float coeffs[3], const float xMin, const float xMax,
    float roots[2]) {
    const float a = coeffs[2];
    const float b = coeffs[1];
    const float c = coeffs[0];
    if (a == 0.0f) {
        if (b == 0.0f)
            return 0;
        roots[0] = -c / b;
        return roots[0] >= xMin && roots[0] <= xMax;
    }
    const float D = pow2(b) - 4 * a * c;
    if (D < 0)
        return 0;
    if (D == 0.0f) {
        roots[0] = -b / (2 * a);
        return roots[0] >= xMin && roots[0] <= xMax;
    }
    const float sqrtD = std::sqrt(D);
    const float temp = -0.5f * (b + std::copysign(sqrtD, b));
    float xx0 = c / temp;
    float xx1 = temp / a;
    if (xx0 > xx1) {
        stc::swap(xx0, xx1);
    }
    uint32_t idx = 0;
    if (xx0 >= xMin && xx0 <= xMax)
        roots[idx++] = xx0;
    if (xx1 >= xMin && xx1 <= xMax)
        roots[idx++] = xx1;
    return idx;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t solveCubicEquationAnalytical(
    const float coeffs[4], const float xMin, const float xMax,
    float roots[3]) {
    uint32_t numRoots = 0;
    const auto testRoot = [&](const float x) {
        if (x >= xMin && x <= xMax)
            roots[numRoots++] = x;
    };

    const auto sortAndTestRoots = [&]
    (const float x0, const float x1, const float x2) {
        float root0;
        float root1;
        float root2;
        if (x0 < x1) {
            if (x0 < x2) {
                root0 = x0;
                root1 = x1 < x2 ? x1 : x2;
                root2 = x1 < x2 ? x2 : x1;
            }
            else {
                root0 = x2;
                root1 = x0;
                root2 = x1;
            }
        }
        else {
            if (x1 < x2) {
                root0 = x1;
                root1 = x0 < x2 ? x0 : x2;
                root2 = x0 < x2 ? x2 : x0;
            }
            else {
                root0 = x2;
                root1 = x1;
                root2 = x0;
            }
        }
        testRoot(root0);
        testRoot(root1);
        testRoot(root2);
    };

    if (coeffs[3] == 0.0f)
        return solveQuadraticEquation(coeffs, xMin, xMax, roots);

    const float recCubicCoeff = 1.0f / coeffs[3];
    const float a = coeffs[2] * recCubicCoeff;
    const float b = coeffs[1] * recCubicCoeff;
    const float c = coeffs[0] * recCubicCoeff;

#if 1 // Reference: Numerical Recipes in C
    const float Q = (pow2(a) - 3 * b) / 9;
    const float R = (2 * pow3(a) - 9 * a * b + 27 * c) / 54;
    const float R2 = pow2(R);
    const float Q3 = pow3(Q);
    const float a_over_3 = a / 3;
    if (R2 >= Q3) {
        const float temp = std::fabs(R) + std::sqrt(R2 - Q3);
        const float A = -std::copysign(std::pow(temp, 1.0f / 3), R);
        const float B = A != 0 ? Q / A : 0;
        testRoot((A + B) - a_over_3);
        return numRoots;
    }
    const float theta = std::acos(stc::clamp(R / std::sqrt(Q3), -1.0f, 1.0f));

    constexpr float _2pi_over_3 = 2 * pi_v<float> / 3;
    const float theta_over_3 = theta / 3;
    const float minus2sqrtQ = -2 * std::sqrt(Q);
    sortAndTestRoots(
        minus2sqrtQ * std::cos(theta_over_3) - a_over_3,
        minus2sqrtQ * std::cos(theta_over_3 + _2pi_over_3) - a_over_3,
        minus2sqrtQ * std::cos(theta_over_3 - _2pi_over_3) - a_over_3);
#else
    const float p = (-pow2(a) + 3 * b) / 9;
    const float q = (2 * pow3(a) - 9 * a * b + 27 * c) / 54;
    const float r2 = pow2(q) + pow3(p);
    const float a_over_3 = a / 3;
    if (r2 > 0) {
        const float r = std::sqrt(r2);
        const float qrA = -q + r;
        const float qrB = -q - r;
        const float x = -a_over_3
            + std::copysign(std::pow(std::fabs(qrA), 1.0f / 3.0f), qrA)
            + std::copysign(std::pow(std::fabs(qrB), 1.0f / 3.0f), qrB);
        testRoot(x);
    }
    else if (r2 * pow4(a) >= -1e-6) {
        const float temp = std::copysign(std::pow(std::fabs(q), 1.0f / 3.0f), q);
        const float xx0 = -a_over_3 - 2 * temp;
        const float xx1 = -a_over_3 + temp;
        testRoot(std::fmin(xx0, xx1));
        testRoot(std::fmax(xx0, xx1));
    }
    else {
        const float r = std::sqrt(-r2);
        const float radius = std::pow(pow2(q) + pow2(r), 1.0f / 6.0f);
        const float arg = std::atan2(r, -q) / 3.0f;
        const float zr = radius * std::cos(arg);
        const float zi = radius * std::sin(arg);
        const float sqrt3 = std::sqrt(3.0f);
        sortAndTestRoots(
            -a_over_3 + 2 * zr,
            -a_over_3 - zr - sqrt3 * zi,
            -a_over_3 - zr + sqrt3 * zi);
    }
#endif

    return numRoots;
}

template <uint32_t degree>
CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluatePolynomial(const float coeffs[degree + 1], const float x) {
    // a_d * x^d + a_{d-1} * x^{d-1} + ... + a_1 * x + a_0
    float ret = coeffs[degree];
    for (int32_t deg = static_cast<int32_t>(degree) - 1; deg >= 0; --deg)
        ret = ret * x + coeffs[deg];
    return ret;
}

template <uint32_t degree>
CUDA_DEVICE_FUNCTION CUDA_INLINE void deflatePolynomial(
    const float coeffs[degree + 1], const float root, float defCoeffs[degree]) {
    defCoeffs[degree - 1] = coeffs[degree];
    for (int32_t deg = static_cast<int32_t>(degree) - 1; deg > 0; --deg)
        defCoeffs[deg - 1] = coeffs[deg] + root * defCoeffs[deg];
}

CUDA_DEVICE_FUNCTION CUDA_INLINE bool testIfDifferentSigns(const float a, const float b) {
    return std::signbit(a) != std::signbit(b);
}

template <uint32_t degree, bool boundError>
CUDA_DEVICE_FUNCTION CUDA_INLINE float findSingleRootClosed(
    const float coeffs[degree + 1], const float derivCoeffs[degree],
    const float xMin, const float xMax, const float yMin, const float yMax,
    const float epsilon) {
    // JP: 初期値は区間の中点から始める。
    // EN: The initial guess is the mid point of the interval.
    float xr = (xMin + xMax) / 2;
    const float epsDia = 2 * epsilon;
    if (xMax - xMin <= epsDia)
        return xr;

    // JP: 収束保証の無いニュートン法を実行する。
    // EN: Perform Newton iterations without convergence guarantree.
    if constexpr (degree <= 3) {
        const float xr0 = xr;
        for (int32_t itr = 0; itr < 16; ++itr) {
            const float yr = evaluatePolynomial<degree>(coeffs, xr);
            const float dyr = evaluatePolynomial<degree - 1>(derivCoeffs, xr);
            const float xn = xr - yr / dyr;
            const float xrn = stc::clamp(xn, xMin, xMax);
            if (std::fabs(xrn - xr) <= epsilon)
                return xrn;
            xr = xrn;
        }
        if (!stc::isfinite(xr))
            xr = xr0;
    }

    // JP: 上記で失敗した場合は収束保証のあるニュートン法と二分法のハイブリッド手法に切り替える。
    // EN: In case the above fails, move to the hybrid solution of Newton method and bisection
    //     with guaranteed convergence.
    float yr = evaluatePolynomial<degree>(coeffs, xr);
    float xLo = xMin;
    float xHi = xMax;

    while (true) {
        const bool isDiffSide = testIfDifferentSigns(yMin, yr);
        if (isDiffSide)
            xHi = xr;
        else
            xLo = xr;

        const float dyr = evaluatePolynomial<degree - 1>(derivCoeffs, xr);
        const float delta = yr / dyr;
        const float xn = xr - delta;
        if (xn > xLo && xn < xHi) {
            const float stepsize = std::fabs(xr - xn);
            xr = xn;
            if (stepsize > epsilon) {
                yr = evaluatePolynomial<degree>(coeffs, xr);
            }
            else {
                if constexpr (boundError) {
                    float xs;
                    if (epsilon == 0) {
                        xs = std::nextafter(isDiffSide ? xHi : xLo, isDiffSide ? xLo : xHi);
                    }
                    else {
                        xs = xn - std::copysign(epsilon, static_cast<float>(isDiffSide - 1));
                        if (xs == xn)
                            xs = std::nextafter(isDiffSide ? xHi : xLo, isDiffSide ? xLo : xHi);
                    }
                    const float ys = evaluatePolynomial<degree>(coeffs, xs);
                    const bool s = testIfDifferentSigns(yMin, ys);
                    if (isDiffSide != s)
                        return xn;
                    xr = xs;
                    yr = ys;
                }
                else {
                    break;
                }
            }
        }
        else {
            xr = (xLo + xHi) / 2;
            if (xr == xLo || xr == xHi || xHi - xLo <= epsDia) {
                if constexpr (boundError) {
                    if (epsilon == 0) {
                        const float xm = isDiffSide ? xLo : xHi;
                        const float ym = evaluatePolynomial<degree>(coeffs, xm);
                        if (std::fabs(ym) < std::fabs(yr))
                            xr = xm;
                    }
                }
                break;
            }
            yr = evaluatePolynomial<degree>(coeffs, xr);
        }
    }
    return xr;
}

template <bool boundError>
CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t solveCubicEquationNumerical(
    const float coeffs[4], const float xMin, const float xMax, float epsilon,
    float roots[3]) {
    Assert(stc::isfinite(xMin) && stc::isfinite(xMax) && xMin < xMax, "Invalid interval.");
    constexpr uint32_t degree = 3;
    const float a = coeffs[3];
    const float b = coeffs[2];
    const float c = coeffs[1];
    const float Dq = pow2(2 * b) - 12 * a * c;

    const float yMin = evaluatePolynomial<degree>(coeffs, xMin);
    const float yMax = evaluatePolynomial<degree>(coeffs, xMax);

    const float derivCoeffs[] = {
        c, 2 * b, 3 * a
    };

    // JP: 極値点が2つある場合は考えうる区間が最大で3つある。
    // EN: If there are two critical points, there are up to three possible intervals.
    if (Dq > 0) {
        float cps[2];
        {
            const float sqrtDq = std::sqrt(Dq);
            const float temp = -b + 0.5f * std::copysign(sqrtDq, b);
            cps[0] = c / temp;
            cps[1] = temp / (3 * a);
            if (cps[0] > cps[1])
                stc::swap(cps[0], cps[1]);
        }

        // JP: 有効範囲が単調増加/減少区間内に収まっていて、
        // EN: If the valid range is confined to a monotonic increasing/decreasing interval,
        if (xMax <= cps[0] ||
            (xMin >= cps[0] && xMax <= cps[1]) ||
            xMin >= cps[1]) {
            if (testIfDifferentSigns(yMin, yMax)) {
                // JP: かつ端点の符号が異なる場合はひとつだけ有効な解が存在する。
                // EN: and if the signs of the end point values differ, there is only one valid root.
                roots[0] = findSingleRootClosed<degree, boundError>(
                    coeffs, derivCoeffs, xMin, xMax, yMin, yMax, epsilon);
                return 1;
            }
            else {
                // JP: かつ端点の符号が同じ場合は有効な解は存在しない。
                // EN: and if the signs of the end point values are the same, there is no valid root.
                return 0;
            }
        }

        uint32_t numRoots = 0;
        // JP: 考えうる区間は最大で3つ。
        // EN: There are up to three possible intervals.
        if (cps[0] > xMin) {
            const float yCp0 = evaluatePolynomial<degree>(coeffs, cps[0]);

            // JP: 左側の区間を考える。
            // EN: Consider the left interval.
            if (testIfDifferentSigns(yMin, yCp0)) {
                roots[0] = findSingleRootClosed<degree, boundError>(
                    coeffs, derivCoeffs, xMin, cps[0], yMin, yCp0, epsilon);
                if constexpr (!boundError) {
                    // JP: 減次を用いて残りの解を求める。
                    // EN: Find the remaining roots with deflation.
                    const float yCp1 = evaluatePolynomial<degree>(coeffs, cps[1]);
                    if (testIfDifferentSigns(yCp0, yMax) || (cps[1] < xMax && testIfDifferentSigns(yCp0, yCp1))) {
                        float defCoeffs[3];
                        deflatePolynomial<degree>(coeffs, roots[0], defCoeffs);
                        return 1 + solveQuadraticEquation(defCoeffs, cps[0], xMax, roots + 1);
                    }
                    else {
                        return 1;
                    }
                }
                else {
                    ++numRoots;
                }
            }

            if (cps[1] < xMax) {
                const float yCp1 = evaluatePolynomial<degree>(coeffs, cps[1]);

                // JP: 真ん中の区間を考える。
                // EN: Consider the middle interval.
                if (testIfDifferentSigns(yCp0, yCp1)) {
                    roots[/*!boundError ? 0 : */numRoots++] = findSingleRootClosed<degree, boundError>(
                        coeffs, derivCoeffs, cps[0], cps[1], yCp0, yCp1, epsilon);
                    if constexpr (!boundError) {
                        // JP: 減次を用いて残りの解を求める。
                        // EN: Find the remaining roots with deflation.
                        if (testIfDifferentSigns(yCp1, yMax)) {
                            float defCoeffs[3];
                            deflatePolynomial<degree>(coeffs, roots[0], defCoeffs);
                            return 1 + solveQuadraticEquation(defCoeffs, cps[1], xMax, roots + 1);
                        }
                        else {
                            return 1;
                        }
                    }
                }

                // JP: 右側の区間を考える。
                // EN: Consider the right interval.
                if (testIfDifferentSigns(yCp1, yMax)) {
                    roots[/*!boundError ? 0 : */numRoots++] = findSingleRootClosed<degree, boundError>(
                        coeffs, derivCoeffs, cps[1], xMax, yCp1, yMax, epsilon);
                    if constexpr (!boundError)
                        return 1;
                }
            }
            else {
                // JP: 真ん中の区間を考える。
                // EN: Consider the middle interval.
                if (testIfDifferentSigns(yCp0, yMax)) {
                    roots[/*!boundError ? 0 : */numRoots++] = findSingleRootClosed<degree, boundError>(
                        coeffs, derivCoeffs, cps[0], xMax, yCp0, yMax, epsilon);
                    if constexpr (!boundError)
                        return 1;
                }
            }
        }
        // JP: 考えうる区間は最大で2つ。
        // EN: There are up to two possible intervals.
        else {
            const float yCp1 = evaluatePolynomial<degree>(coeffs, cps[1]);

            // JP: 真ん中の区間を考える。
            // EN: Consider the middle interval.
            if (testIfDifferentSigns(yMin, yCp1)) {
                roots[/*!boundError ? 0 : */numRoots++] = findSingleRootClosed<degree, boundError>(
                    coeffs, derivCoeffs, xMin, cps[1], yMin, yCp1, epsilon);
                if constexpr (!boundError) {
                    // JP: 減次を用いて残りの解を求める。
                    // EN: Find the remaining roots with deflation.
                    if (testIfDifferentSigns(yCp1, yMax)) {
                        float defCoeffs[3];
                        deflatePolynomial<degree>(coeffs, roots[0], defCoeffs);
                        return 1 + solveQuadraticEquation(defCoeffs, cps[1], xMax, roots + 1);
                    }
                    else {
                        return 1;
                    }
                }
                else {
                    ++numRoots;
                }
            }

            // JP: 右側の区間を考える。
            // EN: Consider the right interval.
            if (testIfDifferentSigns(yCp1, yMax)) {
                roots[/*!boundError ? 0 : */numRoots++] = findSingleRootClosed<degree, boundError>(
                    coeffs, derivCoeffs, cps[1], xMax, yCp1, yMax, epsilon);
                if constexpr (!boundError)
                    return 1;
            }
        }
        return numRoots;
    }
    // JP: 極値点が一つだけ、もしくはない場合は関数は単調増加・減少である。
    // EN: If there is only one or zero critical point, the function is monotonically increasing/decreasing.
    else {
        // JP: したがって端点の符号が異なる場合はひとつだけ有効な解が存在する。
        // EN: Therefore, if the signs of the end point values differ, there is only one valid root.
        if (testIfDifferentSigns(yMin, yMax)) {
            roots[0] = findSingleRootClosed<degree, boundError>(
                coeffs, derivCoeffs, xMin, xMax, yMin, yMax, epsilon);
            return 1;
        }
        return 0;
    }
}



CUDA_DEVICE_FUNCTION CUDA_INLINE void computeCanonicalSpaceRayCoeffs(
    const Point3D &rayOrg, const Vector3D &rayDir, const Vector3D &e0, const Vector3D &e1,
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const Normal3D &nA, const Normal3D &nB, const Normal3D &nC,
    float* const alpha2, float* const alpha1, float* const alpha0,
    float* const beta2, float* const beta1, float* const beta0,
    float* const denom2, float* const denom1, float* const denom0) {
    Vector2D eAB, fAB;
    Vector2D eAC, fAC;
    Vector2D eAO, NA;
    {
        const Vector3D eABInObj = pB - pA;
        const Vector3D eACInObj = pC - pA;
        const Vector3D fABInObj = static_cast<Vector3D>(nB - nA);
        const Vector3D fACInObj = static_cast<Vector3D>(nC - nA);
        const Vector3D eAOInObj = rayOrg - pA;

        eAB = Vector2D(dot(eABInObj, e0), dot(eABInObj, e1));
        eAC = Vector2D(dot(eACInObj, e0), dot(eACInObj, e1));
        fAB = Vector2D(dot(fABInObj, e0), dot(fABInObj, e1));
        fAC = Vector2D(dot(fACInObj, e0), dot(fACInObj, e1));
        eAO = Vector2D(dot(eAOInObj, e0), dot(eAOInObj, e1));
        NA = Vector2D(dot(nA, e0), dot(nA, e1));
    }

    //*alpha2 = -NA.x * fAC.y + NA.y * fAC.x;
    //*alpha1 = eAO.x * fAC.y - eAC.y * NA.x - eAO.y * fAC.x + NA.y * eAC.x;
    //*alpha0 = eAO.x * eAC.y - eAO.y * eAC.x;
    //const float denA2 = fAB.x * fAC.y - fAB.y * fAC.x;
    //const float denA1 = eAB.x * fAC.y + fAB.x * eAC.y - eAB.y * fAC.x - fAB.y * eAC.x;
    //const float denA0 = eAB.x * eAC.y - eAB.y * eAC.x;
    //*beta2 = -NA.x * fAB.y + NA.y * fAB.x;
    //*beta1 = eAO.x * fAB.y - eAB.y * NA.x - eAO.y * fAB.x + NA.y * eAB.x;
    //*beta0 = eAO.x * eAB.y - eAO.y * eAB.x;
    //const float denB2 = fAC.x * fAB.y - fAC.y * fAB.x;
    //const float denB1 = eAC.x * fAB.y + fAC.x * eAB.y - eAC.y * fAB.x - fAC.y * eAB.x;
    //const float denB0 = eAC.x * eAB.y - eAC.y * eAB.x;

    // denA* == -denB* となるので分母はbeta*を反転すれば共通で使える。
    *denom2 = fAB.x * fAC.y - fAB.y * fAC.x;
    *denom1 = eAB.x * fAC.y + fAB.x * eAC.y - eAB.y * fAC.x - fAB.y * eAC.x;
    *denom0 = eAB.x * eAC.y - eAB.y * eAC.x;
    *alpha2 = -NA.x * fAC.y + NA.y * fAC.x;
    *alpha1 = eAO.x * fAC.y - eAC.y * NA.x - eAO.y * fAC.x + NA.y * eAC.x;
    *alpha0 = eAO.x * eAC.y - eAO.y * eAC.x;
    *beta2 = -(-NA.x * fAB.y + NA.y * fAB.x);
    *beta1 = -(eAO.x * fAB.y - eAB.y * NA.x - eAO.y * fAB.x + NA.y * eAB.x);
    *beta0 = -(eAO.x * eAB.y - eAO.y * eAB.x);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void computeTextureSpaceRayCoeffs(
    const Point2D &tcA, const Point2D &tcB, const Point2D &tcC,
    const float alpha2, const float alpha1, const float alpha0,
    const float beta2, const float beta1, const float beta0,
    const float denom2, const float denom1, const float denom0,
    Point2D* const tc2, Point2D* const tc1, Point2D* const tc0) {
    *tc2 = (denom2 - alpha2 - beta2) * tcA + alpha2 * tcB + beta2 * tcC;
    *tc1 = (denom1 - alpha1 - beta1) * tcA + alpha1 * tcB + beta1 * tcC;
    *tc0 = (denom0 - alpha0 - beta0) * tcA + alpha0 * tcB + beta0 * tcC;
}



class MipMapStack {
    // JP: 各8ビットがカウンター(2ビット)と最大3つの要素(それぞれ2ビット)を持つ。
    // EN: every 8 bits represents counter (2-bit) and up to three entries (each with 2-bit).
    uint64_t m_data0;
    uint64_t m_data1;
    uint64_t m_data2;

public:
    union Entry {
        struct {
            uint8_t offsetX : 1;
            uint8_t offsetY : 1;
        };
        uint8_t asUInt8;

        CUDA_DEVICE_FUNCTION CUDA_INLINE Entry() {}
        CUDA_DEVICE_FUNCTION CUDA_INLINE Entry(uint8_t _offsetX, uint8_t _offsetY) :
            asUInt8(0) {
            offsetX = _offsetX;
            offsetY = _offsetY;
        }
    };

    CUDA_DEVICE_FUNCTION CUDA_INLINE MipMapStack() :
        m_data0(0), m_data1(0), m_data2(0) {
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE void push(
        const uint32_t level,
        const Entry entry0, const Entry entry1, const Entry entry2, const int32_t numEntries) {
        Assert(level < 24, "Level must be < 24.");
        Assert(numEntries <= 3, "Num entries to push must be <= 3.");
        if (numEntries > 0) {
            uint8_t newData = 0;
            newData |= entry0.asUInt8 << 2;
            if (numEntries > 1) {
                newData |= entry1.asUInt8 << 4;
                if (numEntries > 2) {
                    newData |= entry2.asUInt8 << 6;
                }
            }
            newData |= numEntries;
            const uint32_t dataIdx = level / 8;
            const uint32_t bitPosInData = (level % 8) * 8;
            (dataIdx == 0 ? m_data0 : dataIdx == 1 ? m_data1 : m_data2) &=
                ~(static_cast<uint64_t>(0xFF) << bitPosInData);
            (dataIdx == 0 ? m_data0 : dataIdx == 1 ? m_data1 : m_data2) |=
                (static_cast<uint64_t>(newData) << bitPosInData);
        }
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE bool tryPop(const uint32_t level, Entry* const entry) {
        Assert(level < 24, "Level must be < 24.");
        const uint32_t dataIdx = level / 8;
        const uint32_t bitPosInData = (level % 8) * 8;
        const uint8_t data = ((dataIdx == 0 ? m_data0 : dataIdx == 1 ? m_data1 : m_data2) >> bitPosInData) & 0xFF;
        const uint32_t counter = data & 0b11;
        if (counter == 0)
            return false;
        entry->asUInt8 = (data >> 2) & 0b11;
        const uint8_t newData = ((data >> 2) & ~0b11) | (counter - 1);
        (dataIdx == 0 ? m_data0 : dataIdx == 1 ? m_data1 : m_data2) &=
            ~(static_cast<uint64_t>(0xFF) << bitPosInData);
        (dataIdx == 0 ? m_data0 : dataIdx == 1 ? m_data1 : m_data2) |=
            (static_cast<uint64_t>(newData) << bitPosInData);
        return true;
    }
};

CUDA_DEVICE_FUNCTION CUDA_INLINE bool testNonlinearRayVsAabb(
    // Prism
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const Normal3D &nA, const Normal3D &nB, const Normal3D &nC,
    // AABB in texture space
    const AABB &aabb,
    // Ray
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const float alpha2, const float alpha1, const float alpha0,
    const float beta2, const float beta1, const float beta0,
    const float denom2, const float denom1, const float denom0,
    const Point2D &tc2, const Point2D &tc1, const Point2D &tc0,
    // results
    float* const hitDistMin, float* const hitDistMax) {
    *hitDistMin = INFINITY;
    *hitDistMax = -INFINITY;

    const auto computeHitDistance = [&]
    (const float h, const float recDenom) {
        const float alpha = evaluateQuadraticPolynomial(alpha2, alpha1, alpha0, h) * recDenom;
        const float beta = evaluateQuadraticPolynomial(beta2, beta1, beta0, h) * recDenom;
        const Point3D SAh = pA + h * nA;
        const Point3D SBh = pB + h * nB;
        const Point3D SCh = pC + h * nC;
        const float dist = dot(
            rayDir,
            (1 - alpha - beta) * SAh + alpha * SBh + beta * SCh - rayOrg);
        *hitDistMin = std::fmin(*hitDistMin, dist);
        *hitDistMax = std::fmax(*hitDistMax, dist);
    };

    const auto testHeightPlane = [&]
    (const float h) {
        if (const float denom = evaluateQuadraticPolynomial(denom2, denom1, denom0, h);
            denom != 0) {
            const float recDenom = 1.0f / denom;
            const float u = evaluateQuadraticPolynomial(tc2.x, tc1.x, tc0.x, h) * recDenom;
            const float v = evaluateQuadraticPolynomial(tc2.y, tc1.y, tc0.y, h) * recDenom;
            if (u >= aabb.minP.x && u <= aabb.maxP.x && v >= aabb.minP.y && v <= aabb.maxP.y)
                computeHitDistance(h, recDenom);
        }
    };

    // min/max height plane
    testHeightPlane(aabb.minP.z);
    testHeightPlane(aabb.maxP.z);

    const auto testUPlane = [&]
    (const float u) {
        const float coeffs[] = {
            tc0.x - u * denom0,
            tc1.x - u * denom1,
            tc2.x - u * denom2,
        };
        float hs[2];
        const uint32_t numRoots = solveQuadraticEquation(coeffs, aabb.minP.z, aabb.maxP.z, hs);
#pragma unroll
        for (uint32_t rIdx = 0; rIdx < 2; ++rIdx) {
            if (rIdx >= numRoots)
                break;
            const float h = hs[rIdx];
            const float recDenom = 1.0f / evaluateQuadraticPolynomial(denom2, denom1, denom0, h);
            const float v = evaluateQuadraticPolynomial(tc2.y, tc1.y, tc0.y, h) * recDenom;
            if (v >= aabb.minP.y && v <= aabb.maxP.y)
                computeHitDistance(h, recDenom);
        }
    };

    // min/max u plane
    testUPlane(aabb.minP.x);
    testUPlane(aabb.maxP.x);

    const auto testVPlane = [&]
    (const float v) {
        const float coeffs[] = {
            tc0.y - v * denom0,
            tc1.y - v * denom1,
            tc2.y - v * denom2,
        };
        float hs[2];
        const uint32_t numRoots = solveQuadraticEquation(coeffs, aabb.minP.z, aabb.maxP.z, hs);
#pragma unroll
        for (uint32_t rIdx = 0; rIdx < 2; ++rIdx) {
            if (rIdx >= numRoots)
                break;
            const float h = hs[rIdx];
            const float recDenom = 1.0f / evaluateQuadraticPolynomial(denom2, denom1, denom0, h);
            const float u = evaluateQuadraticPolynomial(tc2.x, tc1.x, tc0.x, h) * recDenom;
            if (u >= aabb.minP.x && u <= aabb.maxP.x)
                computeHitDistance(h, recDenom);
        }
    };

    // min/max v plane
    testVPlane(aabb.minP.y);
    testVPlane(aabb.maxP.y);

    *hitDistMin = std::fmax(*hitDistMin, distMin);
    *hitDistMax = std::fmin(*hitDistMax, distMax);

    return *hitDistMin <= *hitDistMax && *hitDistMax > 0.0f;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE bool testNonlinearRayVsAabb(
    // Prism
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const Normal3D &nA, const Normal3D &nB, const Normal3D &nC,
    // AABB in texture space
    const AABB &aabb,
    // Ray
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const float alpha2, const float alpha1, const float alpha0,
    const float beta2, const float beta1, const float beta0,
    const float denom2, const float denom1, const float denom0,
    const Point2D &tc2, const Point2D &tc1, const Point2D &tc0,
    // Intermediate intersection results
    const float hs_uLo[2], const float vs_uLo[2],
    const float hs_uHi[2], const float vs_uHi[2],
    const float hs_vLo[2], const float us_vLo[2],
    const float hs_vHi[2], const float us_vHi[2],
    // results
    float* const hitDistMin, float* const hitDistMax) {
    *hitDistMin = INFINITY;
    *hitDistMax = -INFINITY;

    const auto computeHitDistance = [&]
    (const float h, const float recDenom) {
        const float alpha = evaluateQuadraticPolynomial(alpha2, alpha1, alpha0, h) * recDenom;
        const float beta = evaluateQuadraticPolynomial(beta2, beta1, beta0, h) * recDenom;
        const Point3D SAh = pA + h * nA;
        const Point3D SBh = pB + h * nB;
        const Point3D SCh = pC + h * nC;
        const float dist = dot(
            rayDir,
            (1 - alpha - beta) * SAh + alpha * SBh + beta * SCh - rayOrg);
        *hitDistMin = std::fmin(*hitDistMin, dist);
        *hitDistMax = std::fmax(*hitDistMax, dist);
    };

    const auto testHeightPlane = [&]
    (const float h) {
        if (const float denom = evaluateQuadraticPolynomial(denom2, denom1, denom0, h);
            denom != 0) {
            const float recDenom = 1.0f / denom;
            const float u = evaluateQuadraticPolynomial(tc2.x, tc1.x, tc0.x, h) * recDenom;
            const float v = evaluateQuadraticPolynomial(tc2.y, tc1.y, tc0.y, h) * recDenom;
            if (u >= aabb.minP.x && u <= aabb.maxP.x && v >= aabb.minP.y && v <= aabb.maxP.y)
                computeHitDistance(h, recDenom);
        }
    };

    // min/max height plane
    testHeightPlane(aabb.minP.z);
    testHeightPlane(aabb.maxP.z);

    const auto testUPlane = [&]
    (const float vs[2], const float hs[2]) {
        for (uint32_t rIdx = 0; rIdx < 2; ++rIdx) {
            const float v = vs[rIdx];
            const float h = hs[rIdx];
            if (v >= aabb.minP.y && v <= aabb.maxP.y && h >= aabb.minP.z && h <= aabb.maxP.z) {
                const float recDenom = 1.0f / evaluateQuadraticPolynomial(denom2, denom1, denom0, h);
                computeHitDistance(h, recDenom);
            }
        }
    };

    // min/max u plane
    testUPlane(vs_uLo, hs_uLo);
    testUPlane(vs_uHi, hs_uHi);

    const auto testVPlane = [&]
    (const float us[2], const float hs[2]) {
        for (uint32_t rIdx = 0; rIdx < 2; ++rIdx) {
            const float u = us[rIdx];
            const float h = hs[rIdx];
            if (u >= aabb.minP.x && u <= aabb.maxP.x && h >= aabb.minP.z && h <= aabb.maxP.z) {
                const float recDenom = 1.0f / evaluateQuadraticPolynomial(denom2, denom1, denom0, h);
                computeHitDistance(h, recDenom);
            }
        }
    };

    // min/max v plane
    testVPlane(us_vLo, hs_vLo);
    testVPlane(us_vHi, hs_vHi);

    *hitDistMin = std::fmax(*hitDistMin, distMin);
    *hitDistMax = std::fmin(*hitDistMax, distMax);

    return *hitDistMin <= *hitDistMax && *hitDistMax > 0.0f;
}



CUDA_DEVICE_FUNCTION CUDA_INLINE bool testNonlinearRayVsMicroTriangle(
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const Normal3D &nA, const Normal3D &nB, const Normal3D &nC,
    const Point2D &tcA, const Point2D &tcB, const Point2D &tcC,
    const Point3D &mpAInTex, const Point3D &mpBInTex, const Point3D &mpCInTex,
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const Vector3D &e0, const Vector3D &e1,
    const Point2D &tc2, const Point2D &tc1, const Point2D &tc0,
    const float denom2, const float denom1, const float denom0,
    Point3D* const hitPointInCan, /*Point3D* const hitPointInTex,*/
    float* const hitDist, Normal3D* const hitNormalInObj) {
    // JP: テクスチャー空間中のマイクロ三角形を含む平面の方程式の係数を求める。
    const Normal3D nInTex(normalize(cross(mpBInTex - mpAInTex, mpCInTex - mpAInTex)));
    const float KInTex = -dot(nInTex, static_cast<Vector3D>(mpAInTex));

    // JP: 正準空間中のマイクロ三角形を含む平面の方程式の係数を求める。
    const Normal3D nInCan(
        nInTex.x * (tcB.x - tcA.x) + nInTex.y * (tcB.y - tcA.y),
        nInTex.x * (tcC.x - tcA.x) + nInTex.y * (tcC.y - tcA.y),
        nInTex.z);
    const float KInCan = nInTex.x * tcA.x + nInTex.y * tcA.y + KInTex;
    const float minHeight = std::fmin(std::fmin(mpAInTex.z, mpBInTex.z), mpCInTex.z) - 1e-4f;
    const float maxHeight = std::fmax(std::fmax(mpAInTex.z, mpBInTex.z), mpCInTex.z) + 1e-4f;

    // JP: テクスチャー空間中のレイとマイクロ三角形を含む平面の交差判定。
    float hs[3];
    uint32_t numRoots;
    {
        const float coeffs[] = {
            nInTex.x * tc0.x + nInTex.y * tc0.y + KInTex * denom0,
            nInTex.x * tc1.x + nInTex.y * tc1.y + nInTex.z * denom0 + KInTex * denom1,
            nInTex.x * tc2.x + nInTex.y * tc2.y + nInTex.z * denom1 + KInTex * denom2,
            nInTex.z * denom2
        };
        //numRoots = solveCubicEquationAnalytical(coeffs, minHeight, maxHeight, hs);
        numRoots = solveCubicEquationNumerical<false>(coeffs, minHeight, maxHeight, 1e-5f, hs);
    }

    *hitDist = distMax;
    for (int rootIdx = 0; rootIdx < numRoots; ++rootIdx) {
        const float h = hs[rootIdx];

        const Point3D SAh = pA + h * nA;
        const Point3D SBh = pB + h * nB;
        const Point3D SCh = pC + h * nC;

        // JP: 正準空間の他の座標を求める。
        float alpha, beta;
        Matrix3x3 transposedAdjMat;
        {
            const Vector3D eSABInObj = SBh - SAh;
            const Vector3D eSACInObj = SCh - SAh;
            const Vector3D eSAOInObj = rayOrg - SAh;

            const Vector2D eSAB(dot(eSABInObj, e0), dot(eSABInObj, e1));
            const Vector2D eSAC(dot(eSACInObj, e0), dot(eSACInObj, e1));
            const Vector2D eSAO(dot(eSAOInObj, e0), dot(eSAOInObj, e1));

            const float det0 = eSAB.x * eSAC.y - eSAC.x * eSAB.y;
            float curDet = det0;
            Matrix2x2 lhsMat(eSAB, eSAC);
            Vector2D rhsConsts = eSAO;
            const float det1 = eSAB.x * nInCan.y - eSAC.x * nInCan.x;
            if (std::fabs(det1) > std::fabs(curDet)) {
                lhsMat = Matrix2x2(Vector2D(eSAB.x, nInCan.x), Vector2D(eSAC.x, nInCan.y));
                rhsConsts = Vector2D(eSAO.x, -nInCan.z * h - KInCan);
                curDet = det1;
            }
            const float det2 = eSAB.y * nInCan.y - eSAC.y * nInCan.x;
            if (std::fabs(det2) > std::fabs(curDet)) {
                lhsMat = Matrix2x2(Vector2D(eSAB.y, nInCan.x), Vector2D(eSAC.y, nInCan.y));
                rhsConsts = Vector2D(eSAO.y, -nInCan.z * h - KInCan);
                curDet = det2;
            }

            const float recDet = 1.0f / curDet;
            alpha = recDet * (lhsMat[1][1] * rhsConsts.x - lhsMat[1][0] * rhsConsts.y);
            beta = recDet * (-lhsMat[0][1] * rhsConsts.x + lhsMat[0][0] * rhsConsts.y);

            // JP: 論文中の余因子行列は「ij成分がij余因子である行列」を指しているが、
            //     このコードではadjugate()は「ij成分がij余因子である行列の転置行列」を指す。
            transposedAdjMat = adjugateWithoutTranspose(Matrix3x3(
                eSABInObj,
                eSACInObj,
                static_cast<Vector3D>((1 - alpha - beta) * nA + alpha * nB + beta * nC)));
        }
        if (alpha < 0.0f || alpha > 1.0f ||
            beta < 0.0f || beta > 1.0f ||
            (alpha + beta) > 1.0f)
            continue;

        const Point3D hpInCan(alpha, beta, h);
        const Point3D hpInTex((1 - alpha - beta) * tcA + alpha * tcB + beta * tcC, h);

        // JP: 上で求まったα, βはベース三角形における重心座標に過ぎない。
        //     求めた交点がマイクロ三角形内にあるかチェックする必要がある。
        {
            const Vector3D eAB = mpBInTex - mpAInTex;
            const Vector3D eAC = mpCInTex - mpAInTex;
            const Vector3D eAP = hpInTex - mpAInTex;
            const float dotAB_AB = dot(eAB, eAB);
            const float dotAB_AC = dot(eAB, eAC);
            const float dotAC_AC = dot(eAC, eAC);
            const float dotAP_AB = dot(eAP, eAB);
            const float dotAP_AC = dot(eAP, eAC);
            const float recDenom = 1.0f / (dotAB_AB * dotAC_AC - pow2(dotAB_AC));
            const float mBcB = recDenom * (dotAC_AC * dotAP_AB - dotAB_AC * dotAP_AC);
            const float mBcC = recDenom * (dotAB_AB * dotAP_AC - dotAB_AC * dotAP_AB);
            const float mBcA = 1.0f - (mBcB + mBcC);
            if (mBcA <= -1e-5f || mBcB <= -1e-5f || mBcC <= -1e-5f)
                continue;
        }

        const float dist = dot(
            rayDir,
            (1 - alpha - beta) * SAh + alpha * SBh + beta * SCh - rayOrg);
        if (dist > distMin && dist < *hitDist) {
            *hitDist = dist;
            *hitPointInCan = hpInCan;
            //*hitPointInTex = hpInTex;
            *hitNormalInObj = transposedAdjMat * -nInCan;
        }
    }

    return *hitDist < distMax;
}



CUDA_DEVICE_KERNEL void RT_IS_NAME(displacedSurface)() {
    const auto sbtr = HitGroupSBTRecordData::get();
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];
    const GeometryInstanceDataForNRTDSM &nrtdsmGeomInst = plp.s->geomInstNrtdsmDataBuffer[sbtr.geomInstSlot];

    const Point3D rayOrgInObj(optixGetObjectRayOrigin());
    const Vector3D rayDirInObj(optixGetObjectRayDirection());

    const uint32_t primIdx = optixGetPrimitiveIndex();

    const Triangle &tri = geomInst.triangleBuffer[primIdx];
    const Vertex &vA = geomInst.vertexBuffer[tri.index0];
    const Vertex &vB = geomInst.vertexBuffer[tri.index1];
    const Vertex &vC = geomInst.vertexBuffer[tri.index2];

    // JP: まずは直線レイとプリズムの交差判定を行う。
    // EN: Test rectlinear ray vs prism intersection first.
    float prismHitDistEnter, prismHitDistLeave;
    {
        const NRTDSMTriangleAuxInfo &dispTriAuxInfo = nrtdsmGeomInst.dispTriAuxInfoBuffer[primIdx];
        const float minHeight = dispTriAuxInfo.minHeight;
        const float maxHeight = minHeight + dispTriAuxInfo.amplitude;
        const Point3D pA = vA.position + minHeight * vA.normal;
        const Point3D pB = vB.position + minHeight * vB.normal;
        const Point3D pC = vC.position + minHeight * vC.normal;
        const Point3D pD = vA.position + maxHeight * vA.normal;
        const Point3D pE = vB.position + maxHeight * vB.normal;
        const Point3D pF = vC.position + maxHeight * vC.normal;

        const bool hit = testRayVsPrism(
            rayOrgInObj, rayDirInObj,
            optixGetRayTmin(), optixGetRayTmax(),
            pA, pB, pC,
            pD, pE, pF,
            &prismHitDistEnter, &prismHitDistLeave);
        if (!hit)
            return;
    }

    const DisplacementParameters &dispParams = nrtdsmGeomInst.params;

    const float baseHeight = dispParams.hOffset - dispParams.hScale * dispParams.hBias;
    const Point3D pA = vA.position + baseHeight * vA.normal;
    const Point3D pB = vB.position + baseHeight * vB.normal;
    const Point3D pC = vC.position + baseHeight * vC.normal;
    const Normal3D nA = dispParams.hScale * vA.normal;
    const Normal3D nB = dispParams.hScale * vB.normal;
    const Normal3D nC = dispParams.hScale * vC.normal;
    const Matrix3x3 &texXfm = dispParams.textureTransform;
    const Point2D tcA = texXfm * vA.texCoord;
    const Point2D tcB = texXfm * vB.texCoord;
    const Point2D tcC = texXfm * vC.texCoord;

    prismHitDistEnter *= 0.9999f;
    prismHitDistLeave *= 1.0001f;
    float hitDist = prismHitDistLeave;
    float hitBcB, hitBcC;
    Normal3D hitNormal;

    Vector3D e0, e1;
    rayDirInObj.makeCoordinateSystem(&e0, &e1);

    // JP: 正準空間中のレイの係数を求める。
    // EN: Compute the coefficients of the ray in canonical space.
    float alpha2, alpha1, alpha0;
    float beta2, beta1, beta0;
    float denom2, denom1, denom0;
    computeCanonicalSpaceRayCoeffs(
        rayOrgInObj, rayDirInObj, e0, e1,
        pA, pB, pC,
        nA, nB, nC,
        &alpha2, &alpha1, &alpha0,
        &beta2, &beta1, &beta0,
        &denom2, &denom1, &denom0);

    // JP: テクスチャー空間中のレイの係数を求める。
    // EN: Compute the coefficients of the ray in texture space.
    Point2D tc2, tc1, tc0;
    computeTextureSpaceRayCoeffs(
        tcA, tcB, tcC,
        alpha2, alpha1, alpha0,
        beta2, beta1, beta0,
        denom2, denom1, denom0,
        &tc2, &tc1, &tc0);

    const float triAreaInTex = cross(tcB - tcA, tcC - tcA)/* * 0.5f*/;
    const bool tcFlipped = triAreaInTex < 0;
    //const float recTriAreaInTex = 1.0f / triAreaInTex;

    const Vector2D texTriEdgeNormals[] = {
        Vector2D(tcB.y - tcA.y, tcA.x - tcB.x),
        Vector2D(tcC.y - tcB.y, tcB.x - tcC.x),
        Vector2D(tcA.y - tcC.y, tcC.x - tcA.x),
    };
    const Point2D texTriAabbMinP = min(tcA, min(tcB, tcC));
    const Point2D texTriAabbMaxP = max(tcA, max(tcB, tcC));

    const int32_t maxDepth =
        prevPowOf2Exponent(stc::max(nrtdsmGeomInst.heightMapSize.x, nrtdsmGeomInst.heightMapSize.y));
    constexpr int32_t targetMipLevel = 0;

#if OUTPUT_TRAVERSAL_STATS
    uint16_t numAabbTests = 0;
    uint16_t numLeafTests = 0;
#endif

    Texel roots[4];
    uint32_t numRoots;
    findRoots(texTriAabbMinP, texTriAabbMaxP, maxDepth, targetMipLevel, roots, &numRoots);
    MipMapStack stack;
#if DEBUG_TRAVERSAL
    uint32_t numIterations = 0;
    if (isDebugPixel() && getDebugPrintEnabled()) {
        printf(
            "%u-%u: pA: (%g, %g, %g), pB: (%g, %g, %g), pC: (%g, %g, %g)\n",
            plp.f->frameIndex, optixGetPrimitiveIndex(),
            v3print(pA), v3print(pB), v3print(pC));
        printf(
            "%u-%u: nA: (%g, %g, %g), nB: (%g, %g, %g), nC: (%g, %g, %g)\n",
            plp.f->frameIndex, optixGetPrimitiveIndex(),
            v3print(nA), v3print(nB), v3print(nC));
        printf(
            "%u-%u: tcA: (%g, %g), tcB: (%g, %g), tcC: (%g, %g)\n",
            plp.f->frameIndex, optixGetPrimitiveIndex(),
            v2print(tcA), v2print(tcB), v2print(tcC));
        printf(
            "%u-%u: rayOrg: (%g, %g, %g), rayDir: (%g, %g, %g), range: (%g, %g)\n",
            plp.f->frameIndex, optixGetPrimitiveIndex(),
            v3print(rayOrgInObj), v3print(rayDirInObj),
            prismHitDistEnter, prismHitDistLeave);
        printf(
            "%u-%u: alpha: (%g, %g, %g), beta: (%g, %g, %g), denom: (%g, %g, %g)\n",
            plp.f->frameIndex, optixGetPrimitiveIndex(),
            alpha2, alpha1, alpha0,
            beta2, beta1, beta0,
            denom2, denom1, denom0);
        printf(
            "%u-%u: uCoeffs: (%g, %g, %g), vCoeffs: (%g, %g, %g)\n",
            plp.f->frameIndex, optixGetPrimitiveIndex(),
            tc2.x, tc1.x, tc0.x,
            tc2.y, tc1.y, tc0.y);
        printf(
            "%u-%u: TriAABB: (%g, %g) - (%g, %g), %u roots\n",
            plp.f->frameIndex, optixGetPrimitiveIndex(),
            v2print(texTriAabbMinP), v2print(texTriAabbMaxP),
            numRoots);
        printf(
            "%u-%u: maxDepth: %d\n",
            plp.f->frameIndex, optixGetPrimitiveIndex(),
            maxDepth);
    }
#endif
    for (int rootIdx = 0; rootIdx < lengthof(roots); ++rootIdx) {
        if (rootIdx >= numRoots)
            break;
        Texel curTexel = roots[rootIdx];
        const int16_t initialLod = curTexel.lod;

#if DEBUG_TRAVERSAL
        if (isDebugPixel() && getDebugPrintEnabled()) {
            printf(
                "%u-%u, Root %u: [%d - %d, %d]\n",
                plp.f->frameIndex, optixGetPrimitiveIndex(), rootIdx,
                curTexel.lod, curTexel.x, curTexel.y);
        }
#endif

        MipMapStack::Entry curEntry(floorMod(curTexel.x, 2), floorMod(curTexel.y, 2));
        while (curTexel.lod <= initialLod) {
#if DEBUG_TRAVERSAL
            ++numIterations;
#endif
            if (curEntry.asUInt8 == 0xFF) {
                if (!stack.tryPop(curTexel.lod, &curEntry)) {
#if DEBUG_TRAVERSAL
                    if (isDebugPixel() && getDebugPrintEnabled()) {
                        printf(
                            "%u-%u, Root %u: [%d - %d, %d] up\n",
                            plp.f->frameIndex, optixGetPrimitiveIndex(), rootIdx,
                            curTexel.lod, curTexel.x, curTexel.y);
                    }
#endif
                    up(curTexel);
                    continue;
                }
            }
            curTexel.x = floorDiv(curTexel.x, 2) * 2 + curEntry.offsetX;
            curTexel.y = floorDiv(curTexel.y, 2) * 2 + curEntry.offsetY;

#if DEBUG_TRAVERSAL
            if (isDebugPixel() && getDebugPrintEnabled()) {
                printf(
                    "%u-%u, Root %u: Itr: %2u, [%d - %d, %d]\n",
                    plp.f->frameIndex, optixGetPrimitiveIndex(), rootIdx,
                    numIterations - 1,
                    curTexel.lod, curTexel.x, curTexel.y);
            }
#endif

            const float texelScale = std::pow(2.0f, static_cast<float>(curTexel.lod - maxDepth));
            const Point2D texelCenter = Point2D(curTexel.x + 0.5f, curTexel.y + 0.5f) * texelScale;
            const TriangleSquareIntersection2DResult isectResult =
                testTriangleSquareIntersection2D(
                    tcA, tcB, tcC, tcFlipped, texTriEdgeNormals, texTriAabbMinP, texTriAabbMaxP,
                    texelCenter, 0.5f * texelScale);

            // JP: テクセルがベース三角形の外にある場合はテクセルをスキップ。
            // EN: Skip the texel if it is outside of the base triangle.
            if (isectResult == TriangleSquareIntersection2DResult::SquareOutsideTriangle) {
#if DEBUG_TRAVERSAL
                if (isDebugPixel() && getDebugPrintEnabled()) {
                    printf(
                        "%u-%u, Root %u: [%d - %d, %d] OutTri\n",
                        plp.f->frameIndex, optixGetPrimitiveIndex(), rootIdx,
                        curTexel.lod, curTexel.x, curTexel.y);
                }
#endif
                curEntry.asUInt8 = 0xFF;
                continue;
            }

            // JP: 現在のテクセルの4つの子のAABBとレイの交差判定を行う。
            // EN: Test ray vs four AABBs of the current texel's children.
            if (curTexel.lod > targetMipLevel) {
                const float us[3] = {
                    curTexel.x * texelScale,
                    (curTexel.x + 0.5f) * texelScale,
                    (curTexel.x + 1.0f) * texelScale,
                };
                const float vs[3] = {
                    curTexel.y * texelScale,
                    (curTexel.y + 0.5f) * texelScale,
                    (curTexel.y + 1.0f) * texelScale,
                };

                const auto solveQuadraticEquation = [](
                    const float a, const float b, const float c, const float xMin, const float xMax,
                    float roots[2]) {
                    const float coeffs[] = { c, b, a };
                    const uint32_t numRoots = ::solveQuadraticEquation(coeffs, xMin, xMax, roots);
#pragma unroll
                    for (uint32_t i = numRoots; i < 2; ++i)
                        roots[i] = NAN;
                };

                const auto compute_h_v = [&]
                (const float u_plane,
                 float hs[2], float vs[2]) {
                    solveQuadraticEquation(
                        tc2.x - u_plane * denom2,
                        tc1.x - u_plane * denom1,
                        tc0.x - u_plane * denom0, 0.0f, 1.0f,
                        hs);
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        vs[i] = NAN;
                        if (stc::isfinite(hs[i])) {
                            vs[i] = evaluateQuadraticPolynomial(tc2.y, tc1.y, tc0.y, hs[i])
                                / evaluateQuadraticPolynomial(denom2, denom1, denom0, hs[i]);
                        }
                    }
                };

                const auto compute_h_u = [&]
                (const float v_plane,
                 float hs[2], float us[2]) {
                    solveQuadraticEquation(
                        tc2.y - v_plane * denom2,
                        tc1.y - v_plane * denom1,
                        tc0.y - v_plane * denom0, 0.0f, 1.0f,
                        hs);
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        us[i] = NAN;
                        if (stc::isfinite(hs[i])) {
                            us[i] = evaluateQuadraticPolynomial(tc2.x, tc1.x, tc0.x, hs[i])
                                / evaluateQuadraticPolynomial(denom2, denom1, denom0, hs[i]);
                        }
                    }
                };

                // JP: minmaxミップマップから作られるAABBは兄弟と共通の面を持っているため、
                //     u, v軸それぞれに垂直な面との交差判定は6回で足りる。
                // EN: AABBs made from the minmax mipmap shares planes among their siblings,
                //     so six intersection tests is enough for planes parpendicular to the u and v axes.
                float hs_u[3][2], vs_u[3][2];
                float hs_v[3][2], us_v[3][2];
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    compute_h_v(us[i], hs_u[i], vs_u[i]);
                    compute_h_u(vs[i], hs_v[i], us_v[i]);
                }

                down(curTexel);

                // JP: AABBの高さ方向の面の位置はそれぞれ異なる。
                // EN: Each AABB has different planes in the height direction.
                const int2 nextImgSize = make_int2(1 << stc::max(maxDepth - curTexel.lod, 0));
                const auto readMinMax = [&nrtdsmGeomInst, &nextImgSize, &curTexel, &maxDepth]
                (const int32_t x, const int32_t y,
                 float* const hMin, float* const hMax) {
                    const int2 wrapIndex = make_int2(
                        floorDiv(x, nextImgSize.x),
                        floorDiv(y, nextImgSize.y));
                    const uint2 wrappedTexel = curTexel.lod <= maxDepth ?
                        make_uint2(x - wrapIndex.x * nextImgSize.x, y - wrapIndex.y * nextImgSize.y) :
                        make_uint2(0, 0);
                    const float2 minmax =
                        nrtdsmGeomInst.minMaxMipMap[stc::min<int16_t>(curTexel.lod, maxDepth)].read(wrappedTexel);
                    *hMin = minmax.x;
                    *hMax = minmax.y;
                };

                MipMapStack::Entry entries[4];
                float dists[4];
                int32_t numValidEntries = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#if OUTPUT_TRAVERSAL_STATS
                    ++numAabbTests;
#endif
                    const int32_t uOff = i % 2;
                    const int32_t vOff = i / 2;
                    entries[i] = MipMapStack::Entry(uOff, vOff);

                    const int32_t x = curTexel.x + uOff;
                    const int32_t y = curTexel.y + vOff;
                    float hMin, hMax;
                    readMinMax(x, y, &hMin, &hMax);

                    const int32_t iuLo = uOff;
                    const int32_t iuHi = uOff + 1;
                    const int32_t ivLo = vOff;
                    const int32_t ivHi = vOff + 1;
                    const AABB aabb(Point3D(us[iuLo], vs[ivLo], hMin), Point3D(us[iuHi], vs[ivHi], hMax));
                    float distMin, distMax;
                    const bool hit = testNonlinearRayVsAabb(
                        pA, pB, pC, nA, nB, nC,
                        aabb,
                        rayOrgInObj, rayDirInObj, prismHitDistEnter, hitDist,
                        alpha2, alpha1, alpha0, beta2, beta1, beta0, denom2, denom1, denom0,
                        tc2, tc1, tc0,
                        hs_u[iuLo], vs_u[iuLo], hs_u[iuHi], vs_u[iuHi],
                        hs_v[ivLo], us_v[ivLo], hs_v[ivHi], us_v[ivHi],
                        &distMin, &distMax);
                    float dist = INFINITY;
                    if (hit) {
                        dist = 0.5f * (distMin + distMax);
                        ++numValidEntries;
                    }
                    dists[i] = dist;

#if DEBUG_TRAVERSAL
                    if (isDebugPixel() && getDebugPrintEnabled()) {
                        printf(
                            "%u-%u, Root %u: [%d - %d, %d], tMax: %g, AABB (%g, %g, %g) - (%g, %g, %g): %s, %g - %g\n",
                            plp.f->frameIndex, optixGetPrimitiveIndex(), rootIdx,
                            curTexel.lod, x, y,
                            hitDist,
                            v3print(aabb.minP), v3print(aabb.maxP),
                            hit ? "Hit" : "Miss", hit ? distMin : NAN, hit ? distMax : NAN);
                    }
#endif
                }

                const auto sort = []
                (float &distA, MipMapStack::Entry &entryA,
                 float &distB, MipMapStack::Entry &entryB) {
                    if (distA > distB) {
                        const float tempDist = distA;
                        distA = distB;
                        distB = tempDist;
                        const MipMapStack::Entry tempEntry = entryA;
                        entryA = entryB;
                        entryB = tempEntry;
                    }
                };

                // JP: 子ノードをレイのヒット距離の近い順にソート。
                // EN: Sort child nodes in the order of closest hit distance of the ray.
                sort(dists[0], entries[0], dists[1], entries[1]);
                sort(dists[2], entries[2], dists[3], entries[3]);
                sort(dists[0], entries[0], dists[2], entries[2]);
                sort(dists[1], entries[1], dists[3], entries[3]);
                sort(dists[1], entries[1], dists[2], entries[2]);

                curEntry = entries[0];
                stack.push(curTexel.lod, entries[1], entries[2], entries[3], numValidEntries - 1);
                if (numValidEntries == 0)
                    curEntry.asUInt8 = 0xFF;

                continue;
            }

#if OUTPUT_TRAVERSAL_STATS
            ++numLeafTests;
#endif

            const int2 imgSize = make_int2(1 << stc::max(maxDepth - curTexel.lod, 0));
            const auto sample = [&](float px, float py) {
                // No need to explicitly consider texture wrapping since the sampler is responsible for it.
                return tex2DLod<float>(nrtdsmGeomInst.heightMap, px / imgSize.x, py / imgSize.y, curTexel.lod);
            };

            const float cornerHeightTL = sample(curTexel.x - 0.0f, curTexel.y - 0.0f);
            const float cornerHeightTR = sample(curTexel.x + 1.0f, curTexel.y - 0.0f);
            const float cornerHeightBL = sample(curTexel.x - 0.0f, curTexel.y + 1.0f);
            const float cornerHeightBR = sample(curTexel.x + 1.0f, curTexel.y + 1.0f);
            const float uLeft = curTexel.x * texelScale;
            const float vTop = curTexel.y * texelScale;
            const float uRight = (curTexel.x + 1) * texelScale;
            const float vBottom = (curTexel.y + 1) * texelScale;
            const Point3D mpTL(uLeft, vTop, cornerHeightTL);
            const Point3D mpTR(uRight, vTop, cornerHeightTR);
            const Point3D mpBL(uLeft, vBottom, cornerHeightBL);
            const Point3D mpBR(uRight, vBottom, cornerHeightBR);

            // JP: レイと現在のテクセルに対応する2つのマイクロ三角形の交差判定を行う。
            // EN: Test the intersection of the ray vs two micro triangles corresponding to the current texel.
            float tt;
            Normal3D nn;
            Point3D hpInCan;
            if (testNonlinearRayVsMicroTriangle(
                pA, pB, pC,
                nA, nB, nC,
                tcA, tcB, tcC,
                mpTL, mpBL, mpBR,
                rayOrgInObj, rayDirInObj, prismHitDistEnter, hitDist,
                e0, e1,
                tc2, tc1, tc0,
                denom2, denom1, denom0,
                &hpInCan, &tt, &nn)) {
                hitDist = tt;
                hitBcB = hpInCan.x;
                hitBcC = hpInCan.y;
                hitNormal = nn;

#if DEBUG_TRAVERSAL
                if (isDebugPixel() && getDebugPrintEnabled()) {
                    printf(
                        "%u-%u, Root %u: [%d - %d, %d], tMax: %g, uTri0 (%g, %g, %g), (%g, %g, %g), (%g, %g, %g) Hit\n",
                        plp.f->frameIndex, optixGetPrimitiveIndex(), rootIdx,
                        curTexel.lod, curTexel.x, curTexel.y,
                        hitDist,
                        v3print(mpTL), v3print(mpBL), v3print(mpBR));
                }
#endif
            }
#if DEBUG_TRAVERSAL
            else {
                if (isDebugPixel() && getDebugPrintEnabled()) {
                    printf(
                        "%u-%u, Root %u: [%d - %d, %d], tMax: %g, uTri0 (%g, %g, %g), (%g, %g, %g), (%g, %g, %g) Miss\n",
                        plp.f->frameIndex, optixGetPrimitiveIndex(), rootIdx,
                        curTexel.lod, curTexel.x, curTexel.y,
                        hitDist,
                        v3print(mpTL), v3print(mpBL), v3print(mpBR));
                }
            }
#endif
            if (testNonlinearRayVsMicroTriangle(
                pA, pB, pC,
                nA, nB, nC,
                tcA, tcB, tcC,
                mpTL, mpBR, mpTR,
                rayOrgInObj, rayDirInObj, prismHitDistEnter, hitDist,
                e0, e1,
                tc2, tc1, tc0,
                denom2, denom1, denom0,
                &hpInCan, &tt, &nn)) {
                hitDist = tt;
                hitBcB = hpInCan.x;
                hitBcC = hpInCan.y;
                hitNormal = nn;

#if DEBUG_TRAVERSAL
                if (isDebugPixel() && getDebugPrintEnabled()) {
                    printf(
                        "%u-%u, Root %u: [%d - %d, %d], tMax: %g, uTri1 (%g, %g, %g), (%g, %g, %g), (%g, %g, %g) Hit\n",
                        plp.f->frameIndex, optixGetPrimitiveIndex(), rootIdx,
                        curTexel.lod, curTexel.x, curTexel.y,
                        hitDist,
                        v3print(mpTL), v3print(mpBR), v3print(mpTR));
                }
#endif
            }
#if DEBUG_TRAVERSAL
            else {
                if (isDebugPixel() && getDebugPrintEnabled()) {
                    printf(
                        "%u-%u, Root %u: [%d - %d, %d], tMax: %g, uTri1 (%g, %g, %g), (%g, %g, %g), (%g, %g, %g) Miss\n",
                        plp.f->frameIndex, optixGetPrimitiveIndex(), rootIdx,
                        curTexel.lod, curTexel.x, curTexel.y,
                        hitDist,
                        v3print(mpTL), v3print(mpBR), v3print(mpTR));
                }
            }
#endif

            curEntry.asUInt8 = 0xFF;
        }
    }

    if (hitDist == prismHitDistLeave)
        return;

    DisplacedSurfaceAttributes attr = {};
    attr.normalInObj = hitNormal;
#if OUTPUT_TRAVERSAL_STATS
    attr.travStats.numAabbTests = numAabbTests;
    attr.travStats.numLeafTests = numLeafTests;
#endif
    const uint8_t hitKind = dot(rayDirInObj, hitNormal) <= 0 ?
        CustomHitKind_DisplacedSurfaceFrontFace :
        CustomHitKind_DisplacedSurfaceBackFace;
    DisplacedSurfaceAttributeSignature::reportIntersection(hitDist, hitKind, hitBcB, hitBcC, attr);
}
