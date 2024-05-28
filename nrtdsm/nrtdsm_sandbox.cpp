/*

JP: このソースコードは作者が色々試すためだけのもの。
    ここの関数は一切サンプルでは使われない。
EN: This source code is just a sand box, where the author try different things.
    Functions here are not used in the sample at all.

*/

#include "nrtdsm_shared.h"
#include "../common/common_host.h"

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include "../common/bvh_builder.h"

#if ENABLE_VDB

static Vector3D uniformSampleSphere(float u0, float u1) {
    float phi = 2 * pi_v<float> * u1;
    float theta = std::acos(1 - 2 * u0);
    return Vector3D::fromPolarZUp(phi, theta);
}



template <uint32_t degree>
inline float evaluatePolynomial(const float coeffs[degree + 1], const float x) {
    // a_d * x^d + a_{d-1} * x^{d-1} + ... + a_1 * x + a_0
    float ret = coeffs[degree];
    for (int32_t deg = static_cast<int32_t>(degree) - 1; deg >= 0; --deg)
        ret = ret * x + coeffs[deg];
    return ret;
}

static uint32_t solveQuadraticEquation(
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
    if (xx0 > xx1)
        std::swap(xx0, xx1);
    uint32_t idx = 0;
    if (xx0 >= xMin && xx0 <= xMax)
        roots[idx++] = xx0;
    if (xx1 >= xMin && xx1 <= xMax)
        roots[idx++] = xx1;
    return idx;
}

// ----------------------------------------------------------------
// Bracketed Newton Bisection
// Reference: High-Performance Polynomial Root Finding for Graphics

template <uint32_t degree>
inline void deflatePolynomial(const float coeffs[degree + 1], const float root, float defCoeffs[degree]) {
    defCoeffs[degree - 1] = coeffs[degree];
    for (int32_t deg = static_cast<int32_t>(degree) - 1; deg > 0; --deg)
        defCoeffs[deg - 1] = coeffs[deg] + root * defCoeffs[deg];
}

inline bool testIfDifferentSigns(const float a, const float b) {
    return std::signbit(a) != std::signbit(b);
}

template <uint32_t degree, bool boundError>
static float findSingleRootClosed(
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
            const float xrn = std::clamp(xn, xMin, xMax);
            if (std::fabs(xrn - xr) <= epsilon)
                return xrn;
            xr = xrn;
        }
        if (!std::isfinite(xr))
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
static uint32_t solveCubicEquationNumerical(
    const float coeffs[4], const float xMin, const float xMax, float epsilon,
    float roots[3]) {
    Assert(std::isfinite(xMin) && std::isfinite(xMax) && xMin < xMax, "Invalid interval.");
    constexpr uint32_t degree = 3;
    const float a = coeffs[3];
    const float b = coeffs[2];
    const float c = coeffs[1];
    const float d = coeffs[0];
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
                std::swap(cps[0], cps[1]);
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

// END: Bracketed Newton Bisection
// ----------------------------------------------------------------

static uint32_t solveCubicEquationAnalytical(
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
    const float theta = std::acos(std::clamp(R / std::sqrt(Q3), -1.0f, 1.0f));

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

void testSolveCubicEquation() {
    struct TestData {
        float coeffs[4];
    };

    std::mt19937 rng(149013111);
    std::uniform_real_distribution<float> u01;

    constexpr uint32_t numTests = 100000;
    std::vector<TestData> tests(numTests);
    for (uint32_t testIdx = 0; testIdx < numTests; ++testIdx) {
        TestData &test = tests[testIdx];
#if 1
        // Bernstein Polynomials
        const float beta0 = 2 * u01(rng) - 1;
        const float beta1 = 2 * u01(rng) - 1;
        const float beta2 = 2 * u01(rng) - 1;
        const float beta3 = 2 * u01(rng) - 1;
        test.coeffs[0] = beta0;
        test.coeffs[1] = 3 * beta1 - beta0;
        test.coeffs[2] = 3 * beta2 - 6 * beta1 + beta0;
        test.coeffs[3] = beta3 - 3 * beta2 + 3 * beta1 - beta0;
#else
        test.coeffs[0] = 2 * u01(rng) - 1;
        test.coeffs[1] = 2 * u01(rng) - 1;
        test.coeffs[2] = 2 * u01(rng) - 1;
        test.coeffs[3] = 2 * u01(rng) - 1;
#endif
    }

    // JP: この範囲だとNumericalのほうが速い様子。
    //     論文を見ると有効な解が見つからない場合の処理が特に速いことが貢献しているように見えるが、
    //     解が存在する場合もこの範囲だとNumericalがAnalyticalと同等の速さ。
    constexpr float rootMin = 0.0f;
    constexpr float rootMax = 1.0f;

    uint32_t numRootsCounts[4] = { 0, 0, 0, 0 };
    StopWatchHiRes sw;
    sw.start();
    for (uint32_t testIdx = 0; testIdx < numTests; ++testIdx) {
        const TestData &test = tests[testIdx];
        float xs[3];
#if 0
        const uint32_t numRoots = solveCubicEquationAnalytical(test.coeffs, rootMin, rootMax, xs);
#else
        const float epsilon = 3.5e-4f;
        const uint32_t numRoots = solveCubicEquationNumerical<false>(test.coeffs, rootMin, rootMax, epsilon, xs);
#endif
        //for (uint32_t i = numRoots; i < 3; ++i)
        //    xs[i] = NAN;
        //hpprintf(
        //    "%4u: (%9.6f, %9.6f), (%9.6f, %9.6f), (%9.6f, %9.6f)\n", testIdx,
        //    xs[0], evaluatePolynomial<3>(test.coeffs, xs[0]),
        //    xs[1], evaluatePolynomial<3>(test.coeffs, xs[1]),
        //    xs[2], evaluatePolynomial<3>(test.coeffs, xs[2]));
        //hpprintf(
        //    "%4u: (%9.6f, %9.6f)\n", testIdx,
        //    xs[0], evaluatePolynomial<3>(test.coeffs, xs[0]));
        ++numRootsCounts[numRoots];
    }
    const uint32_t mIdx = sw.stop();

    hpprintf("%.3f [ms]\n", sw.getMeasurement(mIdx, StopWatchDurationType::Microseconds) * 1e-3f);
    hpprintf("0 roots: %5u\n", numRootsCounts[0]);
    hpprintf("1 roots: %5u\n", numRootsCounts[1]);
    hpprintf("2 roots: %5u\n", numRootsCounts[2]);
    hpprintf("3 roots: %5u\n", numRootsCounts[3]);
}



// Solve the equation (4)
static void findHeight(
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const Normal3D &nA, const Normal3D &nB, const Normal3D &nC,
    const Point3D &p,
    float hs[3]) {
    const Vector3D eAB = pB - pA;
    const Vector3D eAC = pC - pA;
    const Vector3D fAB = static_cast<Vector3D>(nB - nA);
    const Vector3D fAC = static_cast<Vector3D>(nC - nA);
    const Vector3D eAp = p - pA;

    const Vector3D alpha2 = cross(fAB, fAC);
    const Vector3D alpha1 = cross(eAB, fAC) + cross(fAB, eAC);
    const Vector3D alpha0 = cross(eAB, eAC);

    const float coeffs[] = {
        dot(eAp, alpha0),
        dot(eAp, alpha1) - dot(nA, alpha0),
        dot(eAp, alpha2) - dot(nA, alpha1),
        -dot(nA, alpha2)
    };

    const uint32_t numRoots = solveCubicEquationAnalytical(coeffs, -INFINITY, INFINITY, hs);
    for (int i = numRoots; i < 3; ++i)
        hs[i] = NAN;
}

void testFindHeight() {
    struct TestData {
        Point3D pA;
        Point3D pB;
        Point3D pC;
        Normal3D nA;
        Normal3D nB;
        Normal3D nC;
        Point2D tcA;
        Point2D tcB;
        Point2D tcC;

        Point3D SA(float h) const {
            return pA + h * nA;
        }
        Point3D SB(float h) const {
            return pB + h * nB;
        }
        Point3D SC(float h) const {
            return pC + h * nC;
        }
        Point3D p(const float alpha, const float beta) const {
            return (1 - alpha - beta) * pA + alpha * pB + beta * pC;
        }
        Normal3D n(const float alpha, const float beta) const {
            return (1 - alpha - beta) * nA + alpha * nB + beta * nC;
        }
        Point2D tc(const float alpha, const float beta) const {
            return (1 - alpha - beta) * tcA + alpha * tcB + beta * tcC;
        }
        Point3D S(const float alpha, const float beta, const float h) const {
            const Point3D ret = (1 - alpha - beta) * SA(h) + alpha * SB(h) + beta * SC(h);
            return ret;
        }
    };

    const TestData test = {
        Point3D(-0.5f, -0.4f, 0.1f),
        Point3D(0.4f, 0.1f, 0.4f),
        Point3D(-0.3f, 0.5f, 0.6f),
        normalize(Normal3D(-0.3f, -0.2f, 1.0f)),
        normalize(Normal3D(0.8f, -0.3f, 0.4f)),
        normalize(Normal3D(0.4f, 0.2f, 1.0f)),
        Point2D(0.4f, 0.7f),
        Point2D(0.2f, 0.2f),
        Point2D(0.7f, 0.4f)
    };

    vdb_frame();

    constexpr float axisScale = 1.0f;
    drawAxes(axisScale);

    constexpr bool showNegativeShell = true;

    // World-space Shell
    setColor(RGB(0.25f));
    drawWiredTriangle(test.pA, test.pB, test.pC);
    setColor(RGB(0.0f, 0.5f, 1.0f));
    drawVector(test.pA, test.nA, 1.0f);
    drawVector(test.pB, test.nB, 1.0f);
    drawVector(test.pC, test.nC, 1.0f);
    for (int i = 1; i <= 10; ++i) {
        const float p = static_cast<float>(i) / 10;
        setColor(RGB(p));
        drawWiredTriangle(test.SA(p), test.SB(p), test.SC(p));
    }
    if constexpr (showNegativeShell) {
        setColor(RGB(0.0f, 0.05f, 0.1f));
        drawVector(test.pA, test.nA, -1.0f);
        drawVector(test.pB, test.nB, -1.0f);
        drawVector(test.pC, test.nC, -1.0f);
        for (int i = 1; i <= 10; ++i) {
            const float p = -static_cast<float>(i) / 10;
            setColor(RGB(-p));
            drawWiredTriangle(test.SA(p), test.SB(p), test.SC(p));
        }
    }

    // World-space Ray
    const Point3D rayOrg(0.5f, -0.5f, 1.0f);
    const Vector3D rayDir = normalize(Vector3D(-0.7f, 1.3f, -0.5f));
    constexpr float rayLength = 2.0f;
    setColor(RGB(1.0f));
    drawCross(rayOrg, 0.05f);
    drawVector(rayOrg, rayDir, rayLength);

    // 3.3 Mapping between Two Spaces
    // Eq.4
    setColor(RGB(1.0f, 0.5f, 0));
    for (int i = 0; i <= 10; ++i) {
        const float t = static_cast<float>(i) / 10;
        const Point3D p = rayOrg + t * rayLength * rayDir;
        float hs[3];
        findHeight(
            test.pA, test.pB, test.pC,
            test.nA, test.nB, test.nC,
            p,
            hs);
        for (int j = 0; j < 3; ++j) {
            const float h = hs[j];
            if (!std::isfinite(h))
                continue;

            const Point3D SAh = test.SA(h);
            const Point3D SBh = test.SB(h);
            const Point3D SCh = test.SC(h);
            drawWiredTriangle(SAh, SBh, SCh);

            const Vector3D eAB = SBh - SAh;
            const Vector3D eAC = SCh - SAh;
            const Vector3D eAp = p - SAh;
            const float recDenom = 1.0f / (eAB.x * eAC.y - eAB.y * eAC.x);
            const float bcB = recDenom * (eAp.x * eAC.y - eAp.y * eAC.x);
            const float bcC = -recDenom * (eAp.x * eAB.y - eAp.y * eAB.x);

            drawCross((1 - bcB - bcC) * SAh + bcB * SBh + bcC * SCh, 0.025f);
        }
    }
}



// Compute canonical-space ray coefficients of the equation (7)
static void computeCanonicalSpaceRayCoeffs(
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

void testComputeCanonicalSpaceRayCoeffs() {
    struct TestData {
        Point3D pA;
        Point3D pB;
        Point3D pC;
        Normal3D nA;
        Normal3D nB;
        Normal3D nC;
        Point2D tcA;
        Point2D tcB;
        Point2D tcC;
    };

    const TestData test = {
        Point3D(-0.5f, -0.4f, 0.1f),
        Point3D(0.4f, 0.1f, 0.4f),
        Point3D(-0.3f, 0.5f, 0.6f),
        normalize(Normal3D(-0.3f, -0.2f, 1.0f)),
        normalize(Normal3D(0.8f, -0.3f, 0.4f)),
        normalize(Normal3D(0.4f, 0.2f, 1.0f)),
        Point2D(0.4f, 0.7f),
        Point2D(0.2f, 0.2f),
        Point2D(0.7f, 0.4f)
    };

#if 1
    const Point3D rayOrg(0.5f, -0.5f, 1.0f);
    const Vector3D rayDir = normalize(Vector3D(-0.7f, 1.3f, -0.5f));

    const Point3D pA = test.pA;
    const Point3D pB = test.pB;
    const Point3D pC = test.pC;
    const Normal3D nA = test.nA;
    const Normal3D nB = test.nB;
    const Normal3D nC = test.nC;
    const Point2D tcA = test.tcA;
    const Point2D tcB = test.tcB;
    const Point2D tcC = test.tcC;
#else
    const Point3D rayOrg(0, 1.06066, 1.06066);
    const Vector3D rayDir(-0.0121685, -0.853633, -0.520733);

    const Point3D pA(-0.0714286, 0.00573736, 0.357143);
    const Point3D pB(-0.0714286, 0.0433884, 0.5);
    const Point3D pC(0.0714286, -0.0433884, 0.5);
    const Normal3D nA(0.452981, 0.800184, -0.393082);
    const Normal3D nB(0.492636, 0.870235, 9.56029e-08);
    const Normal3D nC(0.492636, 0.870235, 9.56029e-08);
    const Point2D tcA(0.428571, 0.857143);
    const Point2D tcB(0.428571, 1);
    const Point2D tcC(0.571429, 1);
#endif

    const auto SA = [&]
    (const float h) {
        return pA + h * nA;
    };
    const auto SB = [&]
    (const float h) {
        return pB + h * nB;
    };
    const auto SC = [&]
    (const float h) {
        return pC + h * nC;
    };

    vdb_frame();

    constexpr float axisScale = 1.0f;
    drawAxes(axisScale);

    constexpr bool showNegativeShell = true;

    const auto drawWiredDottedTriangle = []
    (const Point3D &pA, const Point3D pB, const Point3D &pC) {
        drawWiredTriangle(pA, pB, pC);
        setColor(RGB(0, 1, 1));
        drawPoint(pA);
        setColor(RGB(1, 0, 1));
        drawPoint(pB);
        setColor(RGB(1, 1, 0));
        drawPoint(pC);
    };

    // World-space Shell
    setColor(RGB(0.25f));
    drawWiredTriangle(pA, pB, pC);
    setColor(RGB(0.0f, 0.5f, 1.0f));
    drawVector(pA, nA, 1.0f);
    drawVector(pB, nB, 1.0f);
    drawVector(pC, nC, 1.0f);
    for (int i = 1; i <= 10; ++i) {
        const float p = static_cast<float>(i) / 10;
        setColor(RGB(p));
        drawWiredDottedTriangle(SA(p), SB(p), SC(p));
    }
    if constexpr (showNegativeShell) {
        for (int i = 1; i <= 10; ++i) {
            const float p = -static_cast<float>(i) / 10;
            setColor(RGB(-p));
            drawWiredDottedTriangle(SA(p), SB(p), SC(p));
        }
    }

    // World-space Ray
    constexpr float rayLength = 2.0f;
    setColor(RGB(1.0f));
    drawCross(rayOrg, 0.05f);
    drawVector(rayOrg, rayDir, rayLength);

    // JP: 単一間区間の終わりと多重解区間の終わり
    drawCross(rayOrg + 348.0f / 500.0f * rayLength * rayDir, 0.05f);
    drawCross(rayOrg + 375.0f / 500.0f * rayLength * rayDir, 0.05f);

    constexpr Vector3D globalOffsetForCanonical(-1.0f, -2.0f, 0);
    constexpr Vector3D globalOffsetForTexture(1.0f, -2.0f, 0);
    drawAxes(axisScale, globalOffsetForCanonical);
    drawAxes(axisScale, globalOffsetForTexture);

    // Canonical-space and Texture-space Shell
    setColor(RGB(0.25f));
    drawWiredTriangle(
        globalOffsetForCanonical + Point3D(0, 0, 0),
        globalOffsetForCanonical + Point3D(1, 0, 0),
        globalOffsetForCanonical + Point3D(0, 1, 0));
    setColor(RGB(0.25f));
    drawWiredTriangle(
        globalOffsetForTexture + Point3D(tcA, 0.0f),
        globalOffsetForTexture + Point3D(tcB, 0.0f),
        globalOffsetForTexture + Point3D(tcC, 0.0f));
    setColor(RGB(0.0f, 0.5f, 1.0f));
    drawVector(globalOffsetForCanonical + Point3D(0, 0, 0), Normal3D(0, 0, 1), 1.0f);
    drawVector(globalOffsetForCanonical + Point3D(1, 0, 0), Normal3D(0, 0, 1), 1.0f);
    drawVector(globalOffsetForCanonical + Point3D(0, 1, 0), Normal3D(0, 0, 1), 1.0f);
    drawVector(globalOffsetForTexture + Point3D(tcA, 0), Normal3D(0, 0, 1), 1.0f);
    drawVector(globalOffsetForTexture + Point3D(tcB, 0), Normal3D(0, 0, 1), 1.0f);
    drawVector(globalOffsetForTexture + Point3D(tcC, 0), Normal3D(0, 0, 1), 1.0f);
    for (int i = 1; i <= 10; ++i) {
        const float p = static_cast<float>(i) / 10;
        setColor(RGB(p));
        drawWiredDottedTriangle(
            globalOffsetForCanonical + Point3D(0, 0, 0) + p * Normal3D(0, 0, 1),
            globalOffsetForCanonical + Point3D(1, 0, 0) + p * Normal3D(0, 0, 1),
            globalOffsetForCanonical + Point3D(0, 1, 0) + p * Normal3D(0, 0, 1));
        setColor(RGB(p));
        drawWiredDottedTriangle(
            globalOffsetForTexture + Point3D(tcA, 0) + p * Normal3D(0, 0, 1),
            globalOffsetForTexture + Point3D(tcB, 0) + p * Normal3D(0, 0, 1),
            globalOffsetForTexture + Point3D(tcC, 0) + p * Normal3D(0, 0, 1));
    }
    if constexpr (showNegativeShell) {
        for (int i = 1; i <= 10; ++i) {
            const float p = -static_cast<float>(i) / 10;
            setColor(RGB(-p));
            drawWiredDottedTriangle(
                globalOffsetForCanonical + Point3D(0, 0, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForCanonical + Point3D(1, 0, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForCanonical + Point3D(0, 1, 0) + p * Normal3D(0, 0, 1));
            setColor(RGB(-p));
            drawWiredDottedTriangle(
                globalOffsetForTexture + Point3D(tcA, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForTexture + Point3D(tcB, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForTexture + Point3D(tcC, 0) + p * Normal3D(0, 0, 1));
        }
    }

    Vector3D e0, e1;
    rayDir.makeCoordinateSystem(&e0, &e1);

    float alpha2, alpha1, alpha0;
    float beta2, beta1, beta0;
    float denom2, denom1, denom0;
    computeCanonicalSpaceRayCoeffs(
        rayOrg, rayDir, e0, e1,
        pA, pB, pC,
        nA, nB, nC,
        &alpha2, &alpha1, &alpha0,
        &beta2, &beta1, &beta0,
        &denom2, &denom1, &denom0);

    const Point2D tc2 =
        (denom2 - alpha2 - beta2) * tcA
        + alpha2 * tcB
        + beta2 * tcC;
    const Point2D tc1 =
        (denom1 - alpha1 - beta1) * tcA
        + alpha1 * tcB
        + beta1 * tcC;
    const Point2D tc0 =
        (denom0 - alpha0 - beta0) * tcA
        + alpha0 * tcB
        + beta0 * tcC;

    // Canonical-space and Texture-space Ray
    // JP: 非線形レイだからと言ってレイが分岐したり非連続になったりしない。
    //     ただし解として求まる順番は単純ではなくなる。
    //     * レイが非連続に変化する可視化がされた場合は可視化するレイの範囲(rayLengthなど)が足りない。
    std::vector<float> heightValues;
    std::vector<int32_t> indices;
    std::vector<Point3D> canPs;
    std::vector<Point3D> texPs;
    int32_t heightIdx = 0;
    for (int i = 0; i <= 500; ++i) {
        const float t = static_cast<float>(i) / 500;
        float hs[3];
        findHeight(
            pA, pB, pC,
            nA, nB, nC,
            rayOrg + t * rayLength * rayDir,
            hs);
        for (int j = 0; j < 3; ++j) {
            const float h = hs[j];
            if (!std::isfinite(h))
                continue;
            const float h2 = pow2(h);
            const float denom = denom2 * h2 + denom1 * h + denom0;
            const Point3D p(
                (alpha2 * h2 + alpha1 * h + alpha0) / denom,
                (beta2 * h2 + beta1 * h + beta0) / denom,
                h);
            const Point3D tcp((tc2 * h2 + tc1 * h + tc0) / denom, h);

            heightValues.push_back(h);
            indices.push_back(heightIdx++);
            canPs.push_back(p);
            texPs.push_back(tcp);
        }
    }

    std::sort(
        indices.begin(), indices.end(),
        [&](int32_t a, int32_t b) { return heightValues[a] < heightValues[b]; });

    Point3D prevRayPInCan = canPs[*indices.cbegin()];
    Point3D prevRayPInTex = texPs[*indices.cbegin()];
    setColor(RGB(1.0f));
    drawCross(globalOffsetForCanonical + canPs[0], 0.05f);
    drawCross(globalOffsetForTexture + texPs[0], 0.05f);
    for (auto it = ++indices.cbegin(); it != indices.cend(); ++it) {
        const Point3D &p = canPs[*it];
        const Point3D &tcp = texPs[*it];
        drawLine(globalOffsetForCanonical + prevRayPInCan, globalOffsetForCanonical + p);
        drawLine(globalOffsetForTexture + prevRayPInTex, globalOffsetForTexture + tcp);
        prevRayPInCan = p;
        prevRayPInTex = tcp;
    }
}



bool testNonlinearRayVsMicroTriangle(
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const Normal3D &nA, const Normal3D &nB, const Normal3D &nC,
    const Point2D &tcA, const Point2D &tcB, const Point2D &tcC,
    const Point3D &mpAInTex, const Point3D &mpBInTex, const Point3D &mpCInTex,
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const Vector3D &e0, const Vector3D &e1,
    const Point2D &tc2, const Point2D &tc1, const Point2D &tc0,
    const float denom2, const float denom1, const float denom0,
    Point3D* const hitPointInCan, Point3D* const hitPointInTex,
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
        numRoots = solveCubicEquationAnalytical(coeffs, minHeight, maxHeight, hs);
    }

    *hitDist = distMax;
    for (uint32_t rootIdx = 0; rootIdx < numRoots; ++rootIdx) {
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
            //const Vector2D eAB = mpBInTex.xy() - mpAInTex.xy();
            //const Vector2D eBC = mpCInTex.xy() - mpBInTex.xy();
            //const Vector2D eCA = mpAInTex.xy() - mpCInTex.xy();
            //const Vector2D eAP = hpInTex.xy() - mpAInTex.xy();
            //const Vector2D eBP = hpInTex.xy() - mpBInTex.xy();
            //const Vector2D eCP = hpInTex.xy() - mpCInTex.xy();
            //const float cAB = cross(eAB, eAP);
            //const float cBC = cross(eBC, eBP);
            //const float cCA = cross(eCA, eCP);
            //if ((cAB < 0 || cBC < 0 || cCA < 0) && (cAB >= 0 || cBC >= 0 || cCA >= 0))
            //    continue;
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
            *hitPointInTex = hpInTex;
            *hitNormalInObj = transposedAdjMat * nInCan;
        }
    }

    return *hitDist < distMax;
}

void testNonlinearRayVsMicroTriangle() {
    struct TestData {
        Point3D pA;
        Point3D pB;
        Point3D pC;
        Normal3D nA;
        Normal3D nB;
        Normal3D nC;
        Point2D tcA;
        Point2D tcB;
        Point2D tcC;

        Point3D microTris[3]; // Canonical coordinates

        Point3D SA(float h) const {
            return pA + h * nA;
        }
        Point3D SB(float h) const {
            return pB + h * nB;
        }
        Point3D SC(float h) const {
            return pC + h * nC;
        }
        Point3D p(const float alpha, const float beta) const {
            return (1 - alpha - beta) * pA + alpha * pB + beta * pC;
        }
        Normal3D n(const float alpha, const float beta) const {
            return (1 - alpha - beta) * nA + alpha * nB + beta * nC;
        }
        Point2D tc(const float alpha, const float beta) const {
            return (1 - alpha - beta) * tcA + alpha * tcB + beta * tcC;
        }
        Point3D S(const float alpha, const float beta, const float h) const {
            const Point3D ret = (1 - alpha - beta) * SA(h) + alpha * SB(h) + beta * SC(h);
            return ret;
        }
    };

    const TestData test = {
        Point3D(-0.5f, -0.4f, 0.1f),
        Point3D(0.4f, 0.1f, 0.4f),
        Point3D(-0.3f, 0.5f, 0.6f),
        normalize(Normal3D(-0.3f, -0.2f, 1.0f)),
        normalize(Normal3D(0.8f, -0.3f, 0.4f)),
        normalize(Normal3D(0.4f, 0.2f, 1.0f)),
        Point2D(0.4f, 0.7f),
        Point2D(0.2f, 0.2f),
        Point2D(0.7f, 0.4f),
        {
            Point3D(0.2f, 0.1f, 0.4f),
            Point3D(0.6f, 0.3f, 0.1f),
            Point3D(0.2f, 0.8f, 0.8f),
        }
    };

    const float mAlphaA = test.microTris[0].x;
    const float mBetaA = test.microTris[0].y;
    const float mhA = test.microTris[0].z;
    const float mAlphaB = test.microTris[1].x;
    const float mBetaB = test.microTris[1].y;
    const float mhB = test.microTris[1].z;
    const float mAlphaC = test.microTris[2].x;
    const float mBetaC = test.microTris[2].y;
    const float mhC = test.microTris[2].z;

    const Point3D mpAInCan(mAlphaA, mBetaA, mhA);
    const Point3D mpBInCan(mAlphaB, mBetaB, mhB);
    const Point3D mpCInCan(mAlphaC, mBetaC, mhC);
    const Point3D mpAInTex(test.tc(mAlphaA, mBetaA), mhA);
    const Point3D mpBInTex(test.tc(mAlphaB, mBetaB), mhB);
    const Point3D mpCInTex(test.tc(mAlphaC, mBetaC), mhC);

    std::mt19937 rng(51231011);
    std::uniform_real_distribution<float> u01;

    AABB prismAabb;
    prismAabb
        .unify(test.pA).unify(test.pB).unify(test.pC)
        .unify(test.SA(1)).unify(test.SB(1)).unify(test.SC(1));
    const Point3D prismCenter = (prismAabb.minP + prismAabb.maxP) * 0.5f;

    constexpr uint32_t numRays = 500;
    for (int rayIdx = 0; rayIdx < numRays; ++rayIdx) {
        constexpr float rayLength = 1.5f;

        const Point3D rayOrg(
            0.5f * (2 * u01(rng) - 1) + prismCenter.x,
            0.5f * (2 * u01(rng) - 1) + prismCenter.y,
            0.5f * (2 * u01(rng) - 1) + prismCenter.z);
        const Vector3D rayDir = uniformSampleSphere(u01(rng), u01(rng));

        const Point3D pA = test.pA;
        const Point3D pB = test.pB;
        const Point3D pC = test.pC;
        const Normal3D nA = test.nA;
        const Normal3D nB = test.nB;
        const Normal3D nC = test.nC;
        const Point2D tcA = test.tcA;
        const Point2D tcB = test.tcB;
        const Point2D tcC = test.tcC;

        const auto SA = [&](const float h) {
            return pA + h * nA;
        };
        const auto SB = [&](const float h) {
            return pB + h * nB;
        };
        const auto SC = [&](const float h) {
            return pC + h * nC;
        };
        const auto S = [&](const float alpha, const float beta, const float h) {
            const Point3D ret = (1 - alpha - beta) * SA(h) + alpha * SB(h) + beta * SC(h);
            return ret;
        };

        Vector3D e0, e1;
        rayDir.makeCoordinateSystem(&e0, &e1);

        float alpha2, alpha1, alpha0;
        float beta2, beta1, beta0;
        float denom2, denom1, denom0;
        computeCanonicalSpaceRayCoeffs(
            rayOrg, rayDir, e0, e1,
            pA, pB, pC,
            nA, nB, nC,
            &alpha2, &alpha1, &alpha0,
            &beta2, &beta1, &beta0,
            &denom2, &denom1, &denom0);

        const auto computeTcCoeffs = []
        (const Point2D &tcA, const Point2D &tcB, const Point2D &tcC,
         const float denom, const float alpha, const float beta) {
            return (denom - alpha - beta) * tcA + alpha * tcB + beta * tcC;
        };

        const Point2D tc2 = computeTcCoeffs(tcA, tcB, tcC, denom2, alpha2, beta2);
        const Point2D tc1 = computeTcCoeffs(tcA, tcB, tcC, denom1, alpha1, beta1);
        const Point2D tc0 = computeTcCoeffs(tcA, tcB, tcC, denom0, alpha0, beta0);

        Point3D hitPointInCan;
        Point3D hitPointInTex;
        float hitDist;
        Normal3D hitNormalInObj;
        const bool hit = testNonlinearRayVsMicroTriangle(
            pA, pB, pC,
            nA, nB, nC,
            tcA, tcB, tcC,
            mpAInTex, mpBInTex, mpCInTex,
            rayOrg, rayDir, 0, rayLength,
            e0, e1,
            tc2, tc1, tc0,
            denom2, denom1, denom0,
            &hitPointInCan, &hitPointInTex, &hitDist, &hitNormalInObj);

        vdb_frame();

        constexpr float axisScale = 1.0f;
        drawAxes(axisScale);

        constexpr bool showNegativeShell = false;

        const auto drawWiredDottedTriangle = []
        (const Point3D &pA, const Point3D pB, const Point3D &pC) {
            drawWiredTriangle(pA, pB, pC);
            setColor(RGB(0, 1, 1));
            drawPoint(pA);
            setColor(RGB(1, 0, 1));
            drawPoint(pB);
            setColor(RGB(1, 1, 0));
            drawPoint(pC);
        };

        // World-space Shell
        setColor(RGB(0.25f));
        drawWiredTriangle(pA, pB, pC);
        setColor(RGB(0.0f, 0.5f, 1.0f));
        drawVector(pA, nA, 1.0f);
        drawVector(pB, nB, 1.0f);
        drawVector(pC, nC, 1.0f);
        for (int i = 1; i <= 10; ++i) {
            const float p = static_cast<float>(i) / 10;
            setColor(RGB(p));
            drawWiredDottedTriangle(SA(p), SB(p), SC(p));
        }
        if constexpr (showNegativeShell) {
            for (int i = 1; i <= 10; ++i) {
                const float p = -static_cast<float>(i) / 10;
                setColor(RGB(-p));
                drawWiredDottedTriangle(SA(p), SB(p), SC(p));
            }
        }

        // World-space Ray
        setColor(RGB(1.0f));
        drawCross(rayOrg, 0.05f);
        drawVector(rayOrg, rayDir, rayLength);
        if (hit) {
            setColor(RGB(1, 0.5f, 0));
            const Point3D hpA = S(hitPointInCan.x, hitPointInCan.y, hitPointInCan.z);
            const Point3D hpB = rayOrg + hitDist * rayDir;
            drawCross(hpA, 0.05f);
            drawCross(hpB, 0.05f);
            setColor(RGB(0, 1, 1));
            drawVector(hpA, hitNormalInObj, 0.1f);
        }

        // World-space Micro-Triangle
        Point3D prevSAToB;
        Point3D prevSBToC;
        Point3D prevSCToA;
        {
            const float p = 0.0f;

            const float mAlphaAToB = lerp(mAlphaA, mAlphaB, p);
            const float mBetaAToB = lerp(mBetaA, mBetaB, p);
            const float mhAToB = lerp(mhA, mhB, p);
            prevSAToB = S(mAlphaAToB, mBetaAToB, mhAToB);

            const float mAlphaBToC = lerp(mAlphaB, mAlphaC, p);
            const float mBetaBToC = lerp(mBetaB, mBetaC, p);
            const float mhBToC = lerp(mhB, mhC, p);
            prevSBToC = S(mAlphaBToC, mBetaBToC, mhBToC);

            const float mAlphaCToA = lerp(mAlphaC, mAlphaA, p);
            const float mBetaCToA = lerp(mBetaC, mBetaA, p);
            const float mhCToA = lerp(mhC, mhA, p);
            prevSCToA = S(mAlphaCToA, mBetaCToA, mhCToA);

            setColor(RGB(0, 1, 1));
            drawPoint(prevSAToB);
            setColor(RGB(1, 0, 1));
            drawPoint(prevSBToC);
            setColor(RGB(1, 1, 0));
            drawPoint(prevSCToA);
        }
        setColor(RGB(1.0f));
        for (int i = 1; i <= 100; ++i) {
            const float p = static_cast<float>(i) / 100;

            const float mAlphaAToB = lerp(mAlphaA, mAlphaB, p);
            const float mBetaAToB = lerp(mBetaA, mBetaB, p);
            const float mhAToB = lerp(mhA, mhB, p);
            const Point3D SAToB = S(mAlphaAToB, mBetaAToB, mhAToB);
            drawLine(prevSAToB, SAToB);
            prevSAToB = SAToB;

            const float mAlphaBToC = lerp(mAlphaB, mAlphaC, p);
            const float mBetaBToC = lerp(mBetaB, mBetaC, p);
            const float mhBToC = lerp(mhB, mhC, p);
            const Point3D SBToC = S(mAlphaBToC, mBetaBToC, mhBToC);
            drawLine(prevSBToC, SBToC);
            prevSBToC = SBToC;

            const float mAlphaCToA = lerp(mAlphaC, mAlphaA, p);
            const float mBetaCToA = lerp(mBetaC, mBetaA, p);
            const float mhCToA = lerp(mhC, mhA, p);
            const Point3D SCToA = S(mAlphaCToA, mBetaCToA, mhCToA);
            drawLine(prevSCToA, SCToA);
            prevSCToA = SCToA;
        }

        constexpr Vector3D globalOffsetForCanonical(-1.0f, -2.0f, 0);
        constexpr Vector3D globalOffsetForTexture(1.0f, -2.0f, 0);
        drawAxes(axisScale, globalOffsetForCanonical);
        drawAxes(axisScale, globalOffsetForTexture);

        // Canonical-space and Texture-space Shell
        setColor(RGB(0.25f));
        drawWiredTriangle(
            globalOffsetForCanonical + Point3D(0, 0, 0),
            globalOffsetForCanonical + Point3D(1, 0, 0),
            globalOffsetForCanonical + Point3D(0, 1, 0));
        setColor(RGB(0.25f));
        drawWiredTriangle(
            globalOffsetForTexture + Point3D(tcA, 0.0f),
            globalOffsetForTexture + Point3D(tcB, 0.0f),
            globalOffsetForTexture + Point3D(tcC, 0.0f));
        setColor(RGB(0.0f, 0.5f, 1.0f));
        drawVector(globalOffsetForCanonical + Point3D(0, 0, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForCanonical + Point3D(1, 0, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForCanonical + Point3D(0, 1, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForTexture + Point3D(tcA, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForTexture + Point3D(tcB, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForTexture + Point3D(tcC, 0), Normal3D(0, 0, 1), 1.0f);
        for (int i = 1; i <= 10; ++i) {
            const float p = static_cast<float>(i) / 10;
            setColor(RGB(p));
            drawWiredDottedTriangle(
                globalOffsetForCanonical + Point3D(0, 0, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForCanonical + Point3D(1, 0, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForCanonical + Point3D(0, 1, 0) + p * Normal3D(0, 0, 1));
            setColor(RGB(p));
            drawWiredDottedTriangle(
                globalOffsetForTexture + Point3D(tcA, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForTexture + Point3D(tcB, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForTexture + Point3D(tcC, 0) + p * Normal3D(0, 0, 1));
        }
        if constexpr (showNegativeShell) {
            for (int i = 1; i <= 10; ++i) {
                const float p = -static_cast<float>(i) / 10;
                setColor(RGB(-p));
                drawWiredDottedTriangle(
                    globalOffsetForCanonical + Point3D(0, 0, 0) + p * Normal3D(0, 0, 1),
                    globalOffsetForCanonical + Point3D(1, 0, 0) + p * Normal3D(0, 0, 1),
                    globalOffsetForCanonical + Point3D(0, 1, 0) + p * Normal3D(0, 0, 1));
                setColor(RGB(-p));
                drawWiredDottedTriangle(
                    globalOffsetForTexture + Point3D(tcA, 0) + p * Normal3D(0, 0, 1),
                    globalOffsetForTexture + Point3D(tcB, 0) + p * Normal3D(0, 0, 1),
                    globalOffsetForTexture + Point3D(tcC, 0) + p * Normal3D(0, 0, 1));
            }
        }

        // Canonical-space and Texture-space Ray
        std::vector<float> heightValues;
        std::vector<int32_t> indices;
        std::vector<Point3D> canPs;
        std::vector<Point3D> texPs;
        int32_t heightIdx = 0;
        for (int i = 0; i <= 500; ++i) {
            const float t = static_cast<float>(i) / 500;
            float hs[3];
            findHeight(
                pA, pB, pC,
                nA, nB, nC,
                rayOrg + t * rayLength * rayDir,
                hs);
            for (int j = 0; j < 3; ++j) {
                const float h = hs[j];
                if (!std::isfinite(h))
                    continue;
                const float h2 = pow2(h);
                const float denom = denom2 * h2 + denom1 * h + denom0;
                const Point3D p(
                    (alpha2 * h2 + alpha1 * h + alpha0) / denom,
                    (beta2 * h2 + beta1 * h + beta0) / denom,
                    h);
                const Point3D tcp((tc2 * h2 + tc1 * h + tc0) / denom, h);

                heightValues.push_back(h);
                indices.push_back(heightIdx++);
                canPs.push_back(p);
                texPs.push_back(tcp);
            }
        }

        std::sort(
            indices.begin(), indices.end(),
            [&](int32_t a, int32_t b) { return heightValues[a] < heightValues[b]; });

        Point3D prevRayPInCan = canPs[*indices.cbegin()];
        Point3D prevRayPInTex = texPs[*indices.cbegin()];
        setColor(RGB(1.0f));
        drawCross(globalOffsetForCanonical + canPs[0], 0.05f);
        drawCross(globalOffsetForTexture + texPs[0], 0.05f);
        for (auto it = ++indices.cbegin(); it != indices.cend(); ++it) {
            const Point3D &p = canPs[*it];
            const Point3D &tcp = texPs[*it];
            drawLine(globalOffsetForCanonical + prevRayPInCan, globalOffsetForCanonical + p);
            drawLine(globalOffsetForTexture + prevRayPInTex, globalOffsetForTexture + tcp);
            prevRayPInCan = p;
            prevRayPInTex = tcp;
        }

        // Canonical-space and Texture-space Micro-Triangle
        {
            setColor(RGB(1.0f));
            drawWiredDottedTriangle(
                globalOffsetForCanonical + mpAInCan,
                globalOffsetForCanonical + mpBInCan,
                globalOffsetForCanonical + mpCInCan);

            setColor(RGB(1.0f));
            drawWiredDottedTriangle(
                globalOffsetForTexture + mpAInTex,
                globalOffsetForTexture + mpBInTex,
                globalOffsetForTexture + mpCInTex);
        }

        if (hit) {
            setColor(RGB(1, 0.5f, 0));
            drawCross(globalOffsetForCanonical + hitPointInCan, 0.05f);
            drawCross(globalOffsetForTexture + hitPointInTex, 0.05f);
            printf("");
        }

        printf("");
    }
}



static bool testRayVsTriangle(
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    float* const hitDist, Normal3D* const hitNormal, float* const bcB, float* const bcC) {
    const Vector3D eAB = pB - pA;
    const Vector3D eAC = pA - pC;
    *hitNormal = static_cast<Normal3D>(cross(eAC, eAB));

    const Vector3D e = (1.0f / dot(*hitNormal, rayDir)) * (pA - rayOrg);
    const Vector3D i = cross(rayDir, e);

    *bcB = dot(i, eAC);
    *bcC = dot(i, eAB);
    *hitDist = dot(*hitNormal, e);

    return
        ((*hitDist < distMax) & (*hitDist > distMin)
         & (*bcB >= 0.0f) & (*bcC >= 0.0f) & (*bcB + *bcC <= 1));
}

static Point3D restoreTriangleHitPoint(
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const float bcB, const float bcC, Normal3D* const hitNormal) {
    *hitNormal = static_cast<Normal3D>(cross(pB - pA, pC - pA));
    return (1 - (bcB + bcC)) * pA + bcB * pB + bcC * pC;
}

// Reference: Chapter 8. Cool Patches: A Geometric Approach to Ray/Bilinear Patch Intersections
//            Ray Tracing Gems
static bool testRayVsBilinearPatch(
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
            }
        }
    };

    find_v_t(u1);
    find_v_t(u2);
    if (*hitDist == distMax)
        return false;

    const Vector3D dpdu = lerp(eAB, eCD, *v);
    const Vector3D dpdv = lerp(eAC, eBD, *u);
    *hitNormal = static_cast<Normal3D>(cross(dpdu, dpdv));

    return true;
}

static Point3D restoreBilinearPatchHitPoint(
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

static bool testRayVsPrism(
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const Point3D &pD, const Point3D &pE, const Point3D &pF,
    float* const hitDistMin, float* const hitDistMax,
    Normal3D* const hitNormalMin, Normal3D* const hitNormalMax,
    float* const hitParam0Min, float* const hitParam0Max,
    float* const hitParam1Min, float* const hitParam1Max) {
    *hitDistMin = INFINITY;
    *hitDistMax = -INFINITY;

    const auto updateHit = [&]
    (const float t, const Normal3D &n, const uint32_t faceID, const float u, const float v) {
        const Normal3D nn = normalize(n);
        if (t < *hitDistMin) {
            *hitDistMin = t;
            *hitNormalMin = nn;
            *hitParam0Min = faceID + u;
            *hitParam1Min = v;
        }
        if (t > *hitDistMax) {
            *hitDistMax = t;
            *hitNormalMax = nn;
            *hitParam0Max = faceID + u;
            *hitParam1Max = v;
        }
    };

    float tt;
    Normal3D nn;
    float uu, vv;
    if (testRayVsTriangle(
        rayOrg, rayDir, -INFINITY, INFINITY,
        pC, pB, pA,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 0, uu, vv);
    }
    if (testRayVsTriangle(
        rayOrg, rayDir, -INFINITY, INFINITY,
        pD, pE, pF,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 1, uu, vv);
    }
    if (testRayVsBilinearPatch(
        rayOrg, rayDir, -INFINITY, INFINITY,
        pA, pB, pD, pE,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 2, uu, vv);
    }
    if (testRayVsBilinearPatch(
        rayOrg, rayDir, -INFINITY, INFINITY,
        pB, pC, pE, pF,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 3, uu, vv);
    }
    if (testRayVsBilinearPatch(
        rayOrg, rayDir, -INFINITY, INFINITY,
        pC, pA, pF, pD,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 4, uu, vv);
    }

    *hitDistMin = std::fmax(*hitDistMin, distMin);
    *hitDistMax = std::fmin(*hitDistMax, distMax);

    return *hitDistMin <= *hitDistMax && *hitDistMax > 0.0f;
}

Point3D restorePrismHitPoint(
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

void testRayVsPrism() {
    struct TestData {
        Point3D pA;
        Point3D pB;
        Point3D pC;
        Normal3D nA;
        Normal3D nB;
        Normal3D nC;

        Point3D SA(float h) const {
            return pA + h * nA;
        }
        Point3D SB(float h) const {
            return pB + h * nB;
        }
        Point3D SC(float h) const {
            return pC + h * nC;
        }
        Point3D p(const float alpha, const float beta) const {
            return (1 - alpha - beta) * pA + alpha * pB + beta * pC;
        }
        Normal3D n(const float alpha, const float beta) const {
            return (1 - alpha - beta) * nA + alpha * nB + beta * nC;
        }
        Point3D S(const float alpha, const float beta, const float h) const {
            const Point3D ret = (1 - alpha - beta) * SA(h) + alpha * SB(h) + beta * SC(h);
            return ret;
        }
    };

    const TestData test = {
        Point3D(-0.5f, -0.4f, 0.1f),
        Point3D(0.4f, 0.1f, 0.4f),
        Point3D(-0.3f, 0.5f, 0.6f),
        normalize(Normal3D(-0.3f, -0.2f, 1.0f)),
        normalize(Normal3D(0.8f, -0.3f, 0.4f)),
        normalize(Normal3D(0.4f, 0.2f, 1.0f)),
    };

    std::mt19937 rng(51231011);
    std::uniform_real_distribution<float> u01;

    AABB prismAabb;
    prismAabb
        .unify(test.pA).unify(test.pB).unify(test.pC)
        .unify(test.SA(1)).unify(test.SB(1)).unify(test.SC(1));
    const Point3D prismCenter = (prismAabb.minP + prismAabb.maxP) * 0.5f;

    constexpr uint32_t numRays = 500;
    for (int rayIdx = 0; rayIdx < numRays; ++rayIdx) {
        const Point3D rayOrg(
            0.5f * (2 * u01(rng) - 1) + prismCenter.x,
            0.5f * (2 * u01(rng) - 1) + prismCenter.y,
            0.5f * (2 * u01(rng) - 1) + prismCenter.z);
        const Vector3D rayDir = uniformSampleSphere(u01(rng), u01(rng));
        constexpr float rayLength = 1.5f;

        vdb_frame();

        constexpr float axisScale = 1.0f;
        drawAxes(axisScale);

        const auto drawWiredDottedTriangle = []
        (const Point3D &pA, const Point3D pB, const Point3D &pC) {
            drawWiredTriangle(pA, pB, pC);
            setColor(RGB(0, 1, 1));
            drawPoint(pA);
            setColor(RGB(1, 0, 1));
            drawPoint(pB);
            setColor(RGB(1, 1, 0));
            drawPoint(pC);
        };

        // World-space Shell
        setColor(RGB(0.25f));
        drawWiredTriangle(test.pA, test.pB, test.pC);
        setColor(RGB(0.0f, 0.5f, 1.0f));
        drawVector(test.pA, test.nA, 1.0f);
        drawVector(test.pB, test.nB, 1.0f);
        drawVector(test.pC, test.nC, 1.0f);
        for (int i = 1; i <= 10; ++i) {
            const float p = static_cast<float>(i) / 10;
            setColor(RGB(p));
            drawWiredDottedTriangle(test.SA(p), test.SB(p), test.SC(p));
        }

        // World-space Ray
        setColor(RGB(1.0f));
        drawCross(rayOrg, 0.05f);
        drawVector(rayOrg, rayDir, rayLength);

#if 0
        {
            const Point3D p00(-0.5f, 0.5f, 0.1f);
            const Point3D p10(0.5f, 0.6f, 0.3f);
            const Point3D p01(-0.5f, 0.5f, 0.9f);
            const Point3D p11(0.5f, 0.3f, 1.0f);
            setColor(RGB(1));
            drawLine(p00, p10);
            drawLine(p01, p11);
            drawLine(p00, p01);
            drawLine(p10, p11);

            {
                /*
                JP: 任意のuだと当然レイとは交わらない。
                    (レイがバイリニアパッチと交差する場合は)uを動かすとPa-Pbがレイと交わる箇所が見つかるはず。
                */
                const float u = 0.3f;
                const Point3D Pa = lerp(p00, p10, u);
                const Point3D Pb = lerp(p01, p11, u);

                setColor(RGB(0.25f));
                drawLine(Pa, Pb);
                Vector3D n = cross(Pb - Pa, rayDir);
                n.normalize();
                setColor(RGB(0, 1, 1));
                drawVector(Pa, n);
                float dist = dot(Pa - rayOrg, n);
            }

            /*
            JP: uに関する二次方程式を解くことでレイと交わるuを求めることができる。
            */
            const Vector3D eAB = p10 - p00;
            const Vector3D eAC = p01 - p00;
            const Vector3D eCD = p11 - p01;
            const Vector3D eOA = p00 - rayOrg;
            const Vector3D f = eCD - eAB;
            const Vector3D cr_f_d = cross(f, rayDir);
            const Vector3D cr_eAC_d = cross(eAC, rayDir);
            const float coeffs[] = {
                dot(eOA, cr_eAC_d),
                dot(eOA, cr_f_d) + dot(eAB, cr_eAC_d),
                dot(eAB, cr_f_d),
            };
            float us[2];
            solveQuadraticEquation(coeffs, 0.0f, 1.0f, us);
            const float u = us[0];

            const Point3D Pa = lerp(p00, p10, u);
            const Point3D Pb = lerp(p01, p11, u);

            setColor(RGB(1.0f));
            drawLine(Pa, Pb);
            Vector3D n = cross(Pb - Pa, rayDir);
            n.normalize();
            setColor(RGB(0, 1, 1));
            drawVector(Pa, n);
            float dist = dot(Pa - rayOrg, n);

            printf("");
        }
#endif

        const Point3D SA1 = test.SA(1);
        const Point3D SB1 = test.SB(1);
        const Point3D SC1 = test.SC(1);

        float hitDistEnter, hitDistLeave;
        Normal3D hitNormalEnter, hitNormalLeave;
        float hitParam0Enter, hitParam0Leave;
        float hitParam1Enter, hitParam1Leave;

        if (testRayVsPrism(
            rayOrg, rayDir, 0.0f, rayLength,
            test.pA, test.pB, test.pC, SA1, SB1, SC1,
            &hitDistEnter, &hitDistLeave, &hitNormalEnter, &hitNormalLeave,
            &hitParam0Enter, &hitParam0Leave,
            &hitParam1Enter, &hitParam1Leave)) {
            if (hitDistEnter > 0.0f) {
                const Point3D hp = rayOrg + hitDistEnter * rayDir;
                setColor(RGB(1, 0.5f, 0));
                drawCross(hp, 0.05f);
                setColor(RGB(0, 1, 1));
                drawVector(hp, hitNormalEnter, 0.1f);

                Normal3D hn;
                const Point3D _hp = restorePrismHitPoint(
                    test.pA, test.pB, test.pC, SA1, SB1, SC1,
                    hitParam0Enter, hitParam1Enter, &hn);
                hn.normalize();
                printf("");
            }

            if (hitDistLeave < rayLength) {
                const Point3D hp = rayOrg + hitDistLeave * rayDir;
                setColor(RGB(1, 0.5f, 0));
                drawCross(hp, 0.05f);
                setColor(RGB(0, 1, 1));
                drawVector(hp, hitNormalLeave, 0.1f);

                Normal3D hn;
                const Point3D _hp = restorePrismHitPoint(
                    test.pA, test.pB, test.pC, SA1, SB1, SC1,
                    hitParam0Leave, hitParam1Leave, &hn);
                hn.normalize();
                printf("");
            }
        }
    }
}



static void computeTextureSpaceRayCoeffs(
    const Point2D &tcA, const Point2D &tcB, const Point2D &tcC,
    const float alpha2, const float alpha1, const float alpha0,
    const float beta2, const float beta1, const float beta0,
    const float denom2, const float denom1, const float denom0,
    Point2D* const tc2, Point2D* const tc1, Point2D* const tc0) {
    *tc2 = (denom2 - alpha2 - beta2) * tcA + alpha2 * tcB + beta2 * tcC;
    *tc1 = (denom1 - alpha1 - beta1) * tcA + alpha1 * tcB + beta1 * tcC;
    *tc0 = (denom0 - alpha0 - beta0) * tcA + alpha0 * tcB + beta0 * tcC;
}

static inline float evaluateQuadraticPolynomial(
    const float a, const float b, const float c, const float x) {
    return (a * x + b) * x + c;
}

static bool testNonlinearRayVsAabb(
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

    const auto testHeightPlane = [&]
    (const float h) {
        const auto compute_u_v = [&]
        (const float h_plane, const float recDenom,
         float* const u, float* const v) {
            *u = evaluateQuadraticPolynomial(tc2.x, tc1.x, tc0.x, h_plane) * recDenom;
            *v = evaluateQuadraticPolynomial(tc2.y, tc1.y, tc0.y, h_plane) * recDenom;
        };

        if (const float denom = evaluateQuadraticPolynomial(denom2, denom1, denom0, h);
            denom != 0) {
            const float recDenom = 1.0f / denom;
            float u, v;
            compute_u_v(h, recDenom, &u, &v);
            if (u >= aabb.minP.x && u <= aabb.maxP.x && v >= aabb.minP.y && v <= aabb.maxP.y) {
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
            }
        }
    };

    // min/max height plane
    testHeightPlane(aabb.minP.z);
    testHeightPlane(aabb.maxP.z);

    const auto testUPlane = [&]
    (const float v, const float h) {
        if (v >= aabb.minP.y && v <= aabb.maxP.y && h >= aabb.minP.z && h <= aabb.maxP.z) {
            const float recDenom = 1.0f / evaluateQuadraticPolynomial(denom2, denom1, denom0, h);
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
        }
    };

    // min/max u plane
    testUPlane(vs_uLo[0], hs_uLo[0]);
    testUPlane(vs_uLo[1], hs_uLo[1]);
    testUPlane(vs_uHi[0], hs_uHi[0]);
    testUPlane(vs_uHi[1], hs_uHi[1]);

    const auto testVPlane = [&]
    (const float u, const float h) {
        if (u >= aabb.minP.x && u <= aabb.maxP.x && h >= aabb.minP.z && h <= aabb.maxP.z) {
            const float recDenom = 1.0f / evaluateQuadraticPolynomial(denom2, denom1, denom0, h);
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
        }
    };

    // min/max v plane
    testVPlane(us_vLo[0], hs_vLo[0]);
    testVPlane(us_vLo[1], hs_vLo[1]);
    testVPlane(us_vHi[0], hs_vHi[0]);
    testVPlane(us_vHi[1], hs_vHi[1]);

    *hitDistMin = std::fmax(*hitDistMin, distMin);
    *hitDistMax = std::fmin(*hitDistMax, distMax);

    return *hitDistMin <= *hitDistMax && *hitDistMax > 0.0f;
}

void testNonlinearRayVsAabb() {
    struct TestData {
        Point3D pA;
        Point3D pB;
        Point3D pC;
        Normal3D nA;
        Normal3D nB;
        Normal3D nC;
        Point2D tcA;
        Point2D tcB;
        Point2D tcC;
        AABB aabb;

        Point3D SA(float h) const {
            return pA + h * nA;
        }
        Point3D SB(float h) const {
            return pB + h * nB;
        }
        Point3D SC(float h) const {
            return pC + h * nC;
        }
        Point3D p(const float alpha, const float beta) const {
            return (1 - alpha - beta) * pA + alpha * pB + beta * pC;
        }
        Normal3D n(const float alpha, const float beta) const {
            return (1 - alpha - beta) * nA + alpha * nB + beta * nC;
        }
        Point2D tc(const float alpha, const float beta) const {
            return (1 - alpha - beta) * tcA + alpha * tcB + beta * tcC;
        }
        Point3D S(const float alpha, const float beta, const float h) const {
            const Point3D ret = (1 - alpha - beta) * SA(h) + alpha * SB(h) + beta * SC(h);
            return ret;
        }
    };

    const TestData test = {
        Point3D(-0.5f, -0.4f, 0.1f),
        Point3D(0.4f, 0.1f, 0.4f),
        Point3D(-0.3f, 0.5f, 0.6f),
        normalize(Normal3D(-0.3f, -0.2f, 1.0f)),
        normalize(Normal3D(0.8f, -0.3f, 0.4f)),
        normalize(Normal3D(0.4f, 0.2f, 1.0f)),
        Point2D(0.4f, 0.9f),
        Point2D(0.1f, 0.05f),
        Point2D(0.9f, 0.2f),
        AABB(Point3D(0.25f, 0.125f, 0.25f), Point3D(0.75f, 0.5f, 0.75f))
    };

    std::mt19937 rng(51231011);
    std::uniform_real_distribution<float> u01;

    AABB prismAabb;
    prismAabb
        .unify(test.pA).unify(test.pB).unify(test.pC)
        .unify(test.SA(1)).unify(test.SB(1)).unify(test.SC(1));
    const Point3D prismCenter = (prismAabb.minP + prismAabb.maxP) * 0.5f;

    constexpr uint32_t numRays = 500;
    for (int rayIdx = 0; rayIdx < numRays; ++rayIdx) {
        constexpr float rayLength = 1.5f;

#if 0
        const Point3D rayOrg(0, 1.06066, 1.06066);
        const Vector3D rayDir(-0.349335, -0.774794, -0.526934);

        const Point3D pA(-0.5, 0, -0.5);
        const Point3D pB(-0.5, 0, 0.5);
        const Point3D pC(0.5, 0, 0.5);
        const Normal3D nA(0, 1, 0);
        const Normal3D nB(0, 1, 0);
        const Normal3D nC(0, 1, 0);
        const Point2D tcA(0, 0);
        const Point2D tcB(0, 1);
        const Point2D tcC(1, 1);
        const AABB aabb(Point3D(0, 0.5, 0.0218128), Point3D(0.5, 1, 0.0807843));
#else
        const Point3D rayOrg(
            0.5f * (2 * u01(rng) - 1) + prismCenter.x,
            0.5f * (2 * u01(rng) - 1) + prismCenter.y,
            0.5f * (2 * u01(rng) - 1) + prismCenter.z);
        const Vector3D rayDir = uniformSampleSphere(u01(rng), u01(rng));

        const Point3D pA = test.pA;
        const Point3D pB = test.pB;
        const Point3D pC = test.pC;
        const Normal3D nA = test.nA;
        const Normal3D nB = test.nB;
        const Normal3D nC = test.nC;
        const Point2D tcA = test.tcA;
        const Point2D tcB = test.tcB;
        const Point2D tcC = test.tcC;
        const AABB aabb = test.aabb;
#endif

        const auto SA = [&](const float h) {
            return pA + h * nA;
        };
        const auto SB = [&](const float h) {
            return pB + h * nB;
        };
        const auto SC = [&](const float h) {
            return pC + h * nC;
        };
        const auto S = [&](const float alpha, const float beta, const float h) {
            const Point3D ret = (1 - alpha - beta) * SA(h) + alpha * SB(h) + beta * SC(h);
            return ret;
        };

        vdb_frame();

        constexpr float axisScale = 1.0f;
        drawAxes(axisScale);

        const auto drawWiredDottedTriangle = []
        (const Point3D &pA, const Point3D pB, const Point3D &pC) {
            drawWiredTriangle(pA, pB, pC);
            setColor(RGB(0, 1, 1));
            drawPoint(pA);
            setColor(RGB(1, 0, 1));
            drawPoint(pB);
            setColor(RGB(1, 1, 0));
            drawPoint(pC);
        };

        // World-space Shell
        setColor(RGB(0.25f));
        drawWiredTriangle(pA, pB, pC);
        setColor(RGB(0.0f, 0.5f, 1.0f));
        drawVector(pA, nA, 1.0f);
        drawVector(pB, nB, 1.0f);
        drawVector(pC, nC, 1.0f);
        for (int i = 1; i <= 10; ++i) {
            const float p = static_cast<float>(i) / 10;
            setColor(RGB(p));
            drawWiredDottedTriangle(SA(p), SB(p), SC(p));
        }

        // World-space Ray
        setColor(RGB(1.0f));
        drawCross(rayOrg, 0.05f);
        drawVector(rayOrg, rayDir, rayLength);

        constexpr Vector3D globalOffsetForTexture(0.0f, -2.0f, 0);
        drawAxes(axisScale, globalOffsetForTexture);

        // Texture-space Shell
        setColor(RGB(0.25f));
        drawWiredTriangle(
            globalOffsetForTexture + Point3D(tcA, 0.0f),
            globalOffsetForTexture + Point3D(tcB, 0.0f),
            globalOffsetForTexture + Point3D(tcC, 0.0f));
        setColor(RGB(0.0f, 0.5f, 1.0f));
        drawVector(globalOffsetForTexture + Point3D(tcA, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForTexture + Point3D(tcB, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForTexture + Point3D(tcC, 0), Normal3D(0, 0, 1), 1.0f);
        for (int i = 1; i <= 10; ++i) {
            const float p = static_cast<float>(i) / 10;
            setColor(RGB(p));
            drawWiredDottedTriangle(
                globalOffsetForTexture + Point3D(tcA, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForTexture + Point3D(tcB, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForTexture + Point3D(tcC, 0) + p * Normal3D(0, 0, 1));
        }

        // World-space AABB
        const Matrix2x2 invA = invert(Matrix2x2(tcB - tcA, tcC - tcA));
        const auto computePInWorldSpace = [&]
        (const float u, const float v, const float h) {
            const Vector2D bc = invA * (Point2D(u, v) - tcA);
            return S(bc.x, bc.y, h);
        };

        Point3D prevAabbPs[12];
        setColor(RGB(1));
        for (int i = 0; i <= 100; ++i) {
            const float t = static_cast<float>(i) / 100;
            Point3D ps[12];
            // x
            ps[0] = computePInWorldSpace(
                lerp(aabb.minP.x, aabb.maxP.x, t), aabb.minP.y, aabb.minP.z);
            ps[1] = computePInWorldSpace(
                lerp(aabb.minP.x, aabb.maxP.x, t), aabb.maxP.y, aabb.minP.z);
            ps[2] = computePInWorldSpace(
                lerp(aabb.minP.x, aabb.maxP.x, t), aabb.minP.y, aabb.maxP.z);
            ps[3] = computePInWorldSpace(
                lerp(aabb.minP.x, aabb.maxP.x, t), aabb.maxP.y, aabb.maxP.z);
            // y
            ps[4] = computePInWorldSpace(
                aabb.minP.x, lerp(aabb.minP.y, aabb.maxP.y, t), aabb.minP.z);
            ps[5] = computePInWorldSpace(
                aabb.maxP.x, lerp(aabb.minP.y, aabb.maxP.y, t), aabb.minP.z);
            ps[6] = computePInWorldSpace(
                aabb.minP.x, lerp(aabb.minP.y, aabb.maxP.y, t), aabb.maxP.z);
            ps[7] = computePInWorldSpace(
                aabb.maxP.x, lerp(aabb.minP.y, aabb.maxP.y, t), aabb.maxP.z);
            // z
            ps[8] = computePInWorldSpace(
                aabb.minP.x, aabb.minP.y, lerp(aabb.minP.z, aabb.maxP.z, t));
            ps[9] = computePInWorldSpace(
                aabb.maxP.x, aabb.minP.y, lerp(aabb.minP.z, aabb.maxP.z, t));
            ps[10] = computePInWorldSpace(
                aabb.minP.x, aabb.maxP.y, lerp(aabb.minP.z, aabb.maxP.z, t));
            ps[11] = computePInWorldSpace(
                aabb.maxP.x, aabb.maxP.y, lerp(aabb.minP.z, aabb.maxP.z, t));
            if (i > 0) {
                for (int j = 0; j < 12; ++j)
                    drawLine(prevAabbPs[j], ps[j]);
            }
            for (int j = 0; j < 12; ++j)
                prevAabbPs[j] = ps[j];
        }

        // Texture-space AABB
        setColor(RGB(1));
        drawAabb(AABB(globalOffsetForTexture + aabb.minP, globalOffsetForTexture + aabb.maxP));

        Vector3D e0, e1;
        rayDir.makeCoordinateSystem(&e0, &e1);

        float alpha2, alpha1, alpha0;
        float beta2, beta1, beta0;
        float denom2, denom1, denom0;
        computeCanonicalSpaceRayCoeffs(
            rayOrg, rayDir, e0, e1,
            pA, pB, pC,
            nA, nB, nC,
            &alpha2, &alpha1, &alpha0,
            &beta2, &beta1, &beta0,
            &denom2, &denom1, &denom0);

        Point2D tc2, tc1, tc0;
        computeTextureSpaceRayCoeffs(
            tcA, tcB, tcC,
            alpha2, alpha1, alpha0,
            beta2, beta1, beta0,
            denom2, denom1, denom0,
            &tc2, &tc1, &tc0);

        // Texture-space Ray
        std::vector<float> heightValues;
        std::vector<int32_t> indices;
        std::vector<Point3D> texPs;
        int32_t heightIdx = 0;
        for (int i = 0; i <= 500; ++i) {
            const float t = static_cast<float>(i) / 500;
            float hs[3];
            findHeight(
                pA, pB, pC,
                nA, nB, nC,
                rayOrg + t * rayLength * rayDir,
                hs);
            for (int j = 0; j < 3; ++j) {
                const float h = hs[j];
                if (!std::isfinite(h))
                    continue;
                const float h2 = pow2(h);
                const float denom = denom2 * h2 + denom1 * h + denom0;
                const Point3D p(
                    (alpha2 * h2 + alpha1 * h + alpha0) / denom,
                    (beta2 * h2 + beta1 * h + beta0) / denom,
                    h);
                const Point3D tcp((tc2 * h2 + tc1 * h + tc0) / denom, h);

                heightValues.push_back(h);
                indices.push_back(heightIdx++);
                texPs.push_back(tcp);
            }
        }
        std::vector<int32_t> negIndices;
        for (int i = 0; i <= 500; ++i) {
            const float t = -static_cast<float>(i) / 500;
            float hs[3];
            findHeight(
                pA, pB, pC,
                nA, nB, nC,
                rayOrg + t * rayLength * rayDir,
                hs);
            for (int j = 0; j < 3; ++j) {
                const float h = hs[j];
                if (!std::isfinite(h))
                    continue;
                const float h2 = pow2(h);
                const float denom = denom2 * h2 + denom1 * h + denom0;
                const Point3D p(
                    (alpha2 * h2 + alpha1 * h + alpha0) / denom,
                    (beta2 * h2 + beta1 * h + beta0) / denom,
                    h);
                const Point3D tcp((tc2 * h2 + tc1 * h + tc0) / denom, h);

                heightValues.push_back(h);
                negIndices.push_back(heightIdx++);
                texPs.push_back(tcp);
            }
        }

        std::sort(
            indices.begin(), indices.end(),
            [&](int32_t a, int32_t b) { return heightValues[a] < heightValues[b]; });
        std::sort(
            negIndices.begin(), negIndices.end(),
            [&](int32_t a, int32_t b) { return heightValues[a] < heightValues[b]; });

        Point3D prevRayPInTex = texPs[*indices.cbegin()];
        setColor(RGB(1.0f));
        drawCross(globalOffsetForTexture + texPs[0], 0.05f);
        for (auto it = ++indices.cbegin(); it != indices.cend(); ++it) {
            const Point3D &tcp = texPs[*it];
            drawLine(globalOffsetForTexture + prevRayPInTex, globalOffsetForTexture + tcp);
            prevRayPInTex = tcp;
        }
        prevRayPInTex = texPs[*negIndices.cbegin()];
        setColor(RGB(0.1f));
        for (auto it = ++negIndices.cbegin(); it != negIndices.cend(); ++it) {
            const Point3D &tcp = texPs[*it];
            drawLine(globalOffsetForTexture + prevRayPInTex, globalOffsetForTexture + tcp);
            prevRayPInTex = tcp;
        }

        const auto solveQuadraticEquation = [](
            const float a, const float b, const float c, const float xMin, const float xMax,
            float roots[2]) {
            const float coeffs[] = { c, b, a };
            const uint32_t numRoots = ::solveQuadraticEquation(coeffs, xMin, xMax, roots);
            for (int i = numRoots; i < 2; ++i)
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
            for (int i = 0; i < 2; ++i) {
                vs[i] = NAN;
                if (stc::isfinite(hs[i])) {
                    vs[i] = evaluateQuadraticPolynomial(tc2.y, tc1.y, tc0.y, hs[i])
                        / evaluateQuadraticPolynomial(denom2, denom1, denom0, hs[i]);
                }
            }
        };

        float hs_uMin[2], vs_uMin[2];
        compute_h_v(aabb.minP.x, hs_uMin, vs_uMin);
        float hs_uMax[2], vs_uMax[2];
        compute_h_v(aabb.maxP.x, hs_uMax, vs_uMax);

        const auto compute_h_u = [&]
        (const float v_plane,
         float hs[2], float us[2]) {
            solveQuadraticEquation(
                tc2.y - v_plane * denom2,
                tc1.y - v_plane * denom1,
                tc0.y - v_plane * denom0, 0.0f, 1.0f,
                hs);
            for (int i = 0; i < 2; ++i) {
                us[i] = NAN;
                if (stc::isfinite(hs[i])) {
                    us[i] = evaluateQuadraticPolynomial(tc2.x, tc1.x, tc0.x, hs[i])
                        / evaluateQuadraticPolynomial(denom2, denom1, denom0, hs[i]);
                }
            }
        };

        float hs_vMin[2], us_vMin[2];
        compute_h_u(aabb.minP.y, hs_vMin, us_vMin);
        float hs_vMax[2], us_vMax[2];
        compute_h_u(aabb.maxP.y, hs_vMax, us_vMax);

        float hitDistMin, hitDistMax;
        const bool hit = testNonlinearRayVsAabb(
            pA, pB, pC, nA, nB, nC,
            aabb,
            rayOrg, rayDir, 0.0f, rayLength,
            alpha2, alpha1, alpha0, beta2, beta1, beta0, denom2, denom1, denom0,
            tc2, tc1, tc0,
            hs_uMin, vs_uMin, hs_uMax, vs_uMax,
            hs_vMin, us_vMin, hs_vMax, us_vMax,
            &hitDistMin, &hitDistMax);
        if (hit) {
            const auto selectH = [](const float hs[3]) {
                float ret = hs[0];
                if (!std::isfinite(ret) || std::fabs(hs[1]) < std::fabs(ret))
                    ret = hs[1];
                if (!std::isfinite(ret) || std::fabs(hs[2]) < std::fabs(ret))
                    ret = hs[2];
                return ret;
            };

            //hitDistMin = std::fmax(hitDistMin, 0.0f);
            {
                const Point3D p = rayOrg + hitDistMin * rayDir;
                float hs[3];
                findHeight(
                    pA, pB, pC,
                    nA, nB, nC,
                    p,
                    hs);
                const float h = selectH(hs);
                const float h2 = pow2(h);
                const float denom = denom2 * h2 + denom1 * h + denom0;
                Point3D tcp((tc2 * h2 + tc1 * h + tc0) / denom, h);

                setColor(RGB(1, 0.5f, 0));
                drawCross(p, 0.05f);
                drawCross(globalOffsetForTexture + tcp, 0.05f);
            }
            {
                const Point3D p = rayOrg + hitDistMax * rayDir;
                float hs[3];
                findHeight(
                    pA, pB, pC,
                    nA, nB, nC,
                    p,
                    hs);
                const float h = selectH(hs);
                const float h2 = pow2(h);
                const float denom = denom2 * h2 + denom1 * h + denom0;
                Point3D tcp((tc2 * h2 + tc1 * h + tc0) / denom, h);

                setColor(RGB(1, 0.5f, 0));
                drawCross(p, 0.05f);
                drawCross(globalOffsetForTexture + tcp, 0.05f);
            }
            printf("");
        }

        printf("");
    }
}



void testTraversal() {
    struct InternalNode {
        float distMax;
        AABB aabbs[4];
    };
    struct LeafNode {
        float distMax;
        Point3D mpTL;
        Point3D mpTR;
        Point3D mpBL;
        Point3D mpBR;
    };
    using TraversalStep = std::variant<InternalNode, LeafNode>;

    const Point3D pA(-0.5f, 0, -0.5f);
    const Point3D pB(0.5f, 0, 0.5f);
    const Point3D pC(0.5f, 0, -0.5f);
    const Normal3D nA(0, 0.1f, 0);
    const Normal3D nB(0, 0.1f, 0);
    const Normal3D nC(0, 0.1f, 0);
    const Point2D tcA(0, 0);
    const Point2D tcB(1, 1);
    const Point2D tcC(1, 0);

    const Point3D rayOrg(0.384816f, 0.049543f, 0.176388f);
    const Vector3D rayDir(-0.250534f, -2.32565e-06f, -0.968108f);
    const float prismHitDistEnter = 0;
    const float prismHitDistLeave = 0.69874f;

    const float rayLength = 0.69874f;

    std::vector<TraversalStep> steps = {
InternalNode{ 0.69874f, { AABB(Point3D(0.878906f, 0.664062f, 0.482994f), Point3D(0.879883f, 0.665039f, 0.502663f)), AABB(Point3D(0.879883f, 0.664062f, 0.437476f), Point3D(0.880859f, 0.665039f, 0.488334f)), AABB(Point3D(0.878906f, 0.665039f, 0.471351f), Point3D(0.879883f, 0.666016f, 0.497536f)), AABB(Point3D(0.879883f, 0.665039f, 0.429541f), Point3D(0.880859f, 0.666016f, 0.482994f)) }},
LeafNode{0.69874f, Point3D(0.878906f, 0.664062f, 0.502663f), Point3D(0.879883f, 0.664062f, 0.488334f), Point3D(0.878906f, 0.665039f, 0.497536f), Point3D(0.879883f, 0.665039f, 0.482994f)},
LeafNode{0.69874f, Point3D(0.878906f, 0.663086f, 0.508125f), Point3D(0.879883f, 0.663086f, 0.489952f), Point3D(0.878906f, 0.664062f, 0.502663f), Point3D(0.879883f, 0.664062f, 0.488334f)},
LeafNode{0.69874f, Point3D(0.878906f, 0.662109f, 0.509529f), Point3D(0.879883f, 0.662109f, 0.489952f), Point3D(0.878906f, 0.663086f, 0.508125f), Point3D(0.879883f, 0.663086f, 0.489952f)},
LeafNode{0.69874f, Point3D(0.878906f, 0.661133f, 0.509804f), Point3D(0.879883f, 0.661133f, 0.489952f), Point3D(0.878906f, 0.662109f, 0.509529f), Point3D(0.879883f, 0.662109f, 0.489952f)},
LeafNode{0.69874f, Point3D(0.878906f, 0.660156f, 0.507546f), Point3D(0.879883f, 0.660156f, 0.487938f), Point3D(0.878906f, 0.661133f, 0.509804f), Point3D(0.879883f, 0.661133f, 0.489952f)},
LeafNode{0.69874f, Point3D(0.878906f, 0.658203f, 0.49723f), Point3D(0.879883f, 0.658203f, 0.47808f), Point3D(0.878906f, 0.65918f, 0.50341f), Point3D(0.879883f, 0.65918f, 0.483452f)},
LeafNode{0.69874f, Point3D(0.878906f, 0.65918f, 0.50341f), Point3D(0.879883f, 0.65918f, 0.483452f), Point3D(0.878906f, 0.660156f, 0.507546f), Point3D(0.879883f, 0.660156f, 0.487938f)},
LeafNode{0.69874f, Point3D(0.875977f, 0.647461f, 0.486671f), Point3D(0.876953f, 0.647461f, 0.493492f), Point3D(0.875977f, 0.648438f, 0.502235f), Point3D(0.876953f, 0.648438f, 0.504845f)},
LeafNode{0.69874f, Point3D(0.875f, 0.647461f, 0.475853f), Point3D(0.875977f, 0.647461f, 0.486671f), Point3D(0.875f, 0.648438f, 0.495232f), Point3D(0.875977f, 0.648438f, 0.502235f)},
LeafNode{0.69874f, Point3D(0.868164f, 0.621094f, 0.500725f), Point3D(0.869141f, 0.621094f, 0.497841f), Point3D(0.868164f, 0.62207f, 0.494743f), Point3D(0.869141f, 0.62207f, 0.491447f)},
LeafNode{0.69874f, Point3D(0.867188f, 0.614258f, 0.490715f), Point3D(0.868164f, 0.614258f, 0.488197f), Point3D(0.867188f, 0.615234f, 0.498116f), Point3D(0.868164f, 0.615234f, 0.495781f)},
LeafNode{0.69874f, Point3D(0.866211f, 0.614258f, 0.492164f), Point3D(0.867188f, 0.614258f, 0.490715f), Point3D(0.866211f, 0.615234f, 0.499214f), Point3D(0.867188f, 0.615234f, 0.498116f)},
LeafNode{0.69874f, Point3D(0.860352f, 0.591797f, 0.497017f), Point3D(0.861328f, 0.591797f, 0.477241f), Point3D(0.860352f, 0.592773f, 0.497017f), Point3D(0.861328f, 0.592773f, 0.477241f)},
LeafNode{0.69874f, Point3D(0.860352f, 0.59082f, 0.494545f), Point3D(0.861328f, 0.59082f, 0.474769f), Point3D(0.860352f, 0.591797f, 0.497017f), Point3D(0.861328f, 0.591797f, 0.477241f)},
LeafNode{0.69874f, Point3D(0.859375f, 0.588867f, 0.493828f), Point3D(0.860352f, 0.588867f, 0.480385f), Point3D(0.859375f, 0.589844f, 0.503532f), Point3D(0.860352f, 0.589844f, 0.489662f)},
LeafNode{0.69874f, Point3D(0.851562f, 0.55957f, 0.501854f), Point3D(0.852539f, 0.55957f, 0.474113f), Point3D(0.851562f, 0.560547f, 0.510887f), Point3D(0.852539f, 0.560547f, 0.471366f)},
LeafNode{0.69874f, Point3D(0.847656f, 0.541016f, 0.494163f), Point3D(0.848633f, 0.541016f, 0.496666f), Point3D(0.847656f, 0.541992f, 0.487846f), Point3D(0.848633f, 0.541992f, 0.495293f)},
LeafNode{0.69874f, Point3D(0.847656f, 0.540039f, 0.496086f), Point3D(0.848633f, 0.540039f, 0.493919f), Point3D(0.847656f, 0.541016f, 0.494163f), Point3D(0.848633f, 0.541016f, 0.496666f)},
LeafNode{0.69874f, Point3D(0.847656f, 0.539062f, 0.491142f), Point3D(0.848633f, 0.539062f, 0.486671f), Point3D(0.847656f, 0.540039f, 0.496086f), Point3D(0.848633f, 0.540039f, 0.493919f)},
LeafNode{0.69874f, Point3D(0.84668f, 0.536133f, 0.502861f), Point3D(0.847656f, 0.536133f, 0.502419f), Point3D(0.84668f, 0.537109f, 0.492073f), Point3D(0.847656f, 0.537109f, 0.493553f)},
LeafNode{0.69874f, Point3D(0.845703f, 0.536133f, 0.50135f), Point3D(0.84668f, 0.536133f, 0.502861f), Point3D(0.845703f, 0.537109f, 0.487541f), Point3D(0.84668f, 0.537109f, 0.492073f)},
LeafNode{0.69874f, Point3D(0.844727f, 0.530273f, 0.494347f), Point3D(0.845703f, 0.530273f, 0.504021f), Point3D(0.844727f, 0.53125f, 0.492226f), Point3D(0.845703f, 0.53125f, 0.498543f)},
LeafNode{0.69874f, Point3D(0.844727f, 0.529297f, 0.513359f), Point3D(0.845703f, 0.529297f, 0.523339f), Point3D(0.844727f, 0.530273f, 0.494347f), Point3D(0.845703f, 0.530273f, 0.504021f)},
LeafNode{0.69874f, Point3D(0.84375f, 0.529297f, 0.496574f), Point3D(0.844727f, 0.529297f, 0.513359f), Point3D(0.84375f, 0.530273f, 0.481559f), Point3D(0.844727f, 0.530273f, 0.494347f)},
LeafNode{0.69874f, Point3D(0.838867f, 0.506836f, 0.491371f), Point3D(0.839844f, 0.506836f, 0.494942f), Point3D(0.838867f, 0.507812f, 0.496315f), Point3D(0.839844f, 0.507812f, 0.500511f)},
LeafNode{0.69874f, Point3D(0.837891f, 0.506836f, 0.487251f), Point3D(0.838867f, 0.506836f, 0.491371f), Point3D(0.837891f, 0.507812f, 0.49012f), Point3D(0.838867f, 0.507812f, 0.496315f)},
LeafNode{0.69874f, Point3D(0.836914f, 0.498047f, 0.483574f), Point3D(0.837891f, 0.498047f, 0.503487f), Point3D(0.836914f, 0.499023f, 0.463111f), Point3D(0.837891f, 0.499023f, 0.463111f)},
LeafNode{0.69874f, Point3D(0.836914f, 0.49707f, 0.504051f), Point3D(0.837891f, 0.49707f, 0.524514f), Point3D(0.836914f, 0.498047f, 0.483574f), Point3D(0.837891f, 0.498047f, 0.503487f)},
LeafNode{0.69874f, Point3D(0.835938f, 0.49707f, 0.48333f), Point3D(0.836914f, 0.49707f, 0.504051f), Point3D(0.835938f, 0.498047f, 0.474434f), Point3D(0.836914f, 0.498047f, 0.483574f)},
LeafNode{0.69874f, Point3D(0.835938f, 0.496094f, 0.500999f), Point3D(0.836914f, 0.496094f, 0.523491f), Point3D(0.835938f, 0.49707f, 0.48333f), Point3D(0.836914f, 0.49707f, 0.504051f)},
LeafNode{0.69874f, Point3D(0.833008f, 0.485352f, 0.467918f), Point3D(0.833984f, 0.485352f, 0.520134f), Point3D(0.833008f, 0.486328f, 0.503487f), Point3D(0.833984f, 0.486328f, 0.535378f)},
LeafNode{0.69874f, Point3D(0.833008f, 0.484375f, 0.440818f), Point3D(0.833984f, 0.484375f, 0.51577f), Point3D(0.833008f, 0.485352f, 0.467918f), Point3D(0.833984f, 0.485352f, 0.520134f)},
LeafNode{0.69874f, Point3D(0.833008f, 0.482422f, 0.565332f), Point3D(0.833984f, 0.482422f, 0.617594f), Point3D(0.833008f, 0.483398f, 0.477821f), Point3D(0.833984f, 0.483398f, 0.556817f)},
LeafNode{0.69874f, Point3D(0.833008f, 0.483398f, 0.477821f), Point3D(0.833984f, 0.483398f, 0.556817f), Point3D(0.833008f, 0.484375f, 0.440818f), Point3D(0.833984f, 0.484375f, 0.51577f)},
LeafNode{0.69874f, Point3D(0.831055f, 0.480469f, 0.539117f), Point3D(0.832031f, 0.480469f, 0.610925f), Point3D(0.831055f, 0.481445f, 0.492622f), Point3D(0.832031f, 0.481445f, 0.564874f)},
LeafNode{0.69874f, Point3D(0.826172f, 0.455078f, 0.527871f), Point3D(0.827148f, 0.455078f, 0.477668f), Point3D(0.826172f, 0.456055f, 0.573343f), Point3D(0.827148f, 0.456055f, 0.534127f)},
LeafNode{0.69874f, Point3D(0.825195f, 0.454102f, 0.505898f), Point3D(0.826172f, 0.454102f, 0.466682f), Point3D(0.825195f, 0.455078f, 0.573343f), Point3D(0.826172f, 0.455078f, 0.527871f)},
LeafNode{0.69874f, Point3D(0.825195f, 0.453125f, 0.445777f), Point3D(0.826172f, 0.453125f, 0.434791f), Point3D(0.825195f, 0.454102f, 0.505898f), Point3D(0.826172f, 0.454102f, 0.466682f)},
LeafNode{0.69874f, Point3D(0.823242f, 0.444336f, 0.512795f), Point3D(0.824219f, 0.444336f, 0.508354f), Point3D(0.823242f, 0.445312f, 0.458732f), Point3D(0.824219f, 0.445312f, 0.462104f)},
LeafNode{0.69874f, Point3D(0.822266f, 0.444336f, 0.512795f), Point3D(0.823242f, 0.444336f, 0.512795f), Point3D(0.822266f, 0.445312f, 0.458732f), Point3D(0.823242f, 0.445312f, 0.458732f)},
LeafNode{0.69874f, Point3D(0.818359f, 0.426758f, 0.491081f), Point3D(0.819336f, 0.426758f, 0.485466f), Point3D(0.818359f, 0.427734f, 0.504067f), Point3D(0.819336f, 0.427734f, 0.502037f)},
LeafNode{0.69874f, Point3D(0.817383f, 0.424805f, 0.51194f), Point3D(0.818359f, 0.424805f, 0.494285f), Point3D(0.817383f, 0.425781f, 0.498512f), Point3D(0.818359f, 0.425781f, 0.48481f)},
LeafNode{0.69874f, Point3D(0.817383f, 0.423828f, 0.522164f), Point3D(0.818359f, 0.423828f, 0.504173f), Point3D(0.817383f, 0.424805f, 0.51194f), Point3D(0.818359f, 0.424805f, 0.494285f)},
LeafNode{0.69874f, Point3D(0.8125f, 0.40625f, 0.510414f), Point3D(0.813477f, 0.40625f, 0.492348f), Point3D(0.8125f, 0.407227f, 0.586725f), Point3D(0.813477f, 0.407227f, 0.573022f)},
LeafNode{0.69874f, Point3D(0.8125f, 0.405273f, 0.437476f), Point3D(0.813477f, 0.405273f, 0.427771f), Point3D(0.8125f, 0.40625f, 0.510414f), Point3D(0.813477f, 0.40625f, 0.492348f)},
LeafNode{0.69874f, Point3D(0.811523f, 0.405273f, 0.480613f), Point3D(0.8125f, 0.405273f, 0.437476f), Point3D(0.811523f, 0.40625f, 0.538567f), Point3D(0.8125f, 0.40625f, 0.510414f)},
LeafNode{0.69874f, Point3D(0.811523f, 0.402344f, 0.515953f), Point3D(0.8125f, 0.402344f, 0.481621f), Point3D(0.811523f, 0.40332f, 0.499153f), Point3D(0.8125f, 0.40332f, 0.452949f)},
LeafNode{0.69874f, Point3D(0.811523f, 0.401367f, 0.529458f), Point3D(0.8125f, 0.401367f, 0.505074f), Point3D(0.811523f, 0.402344f, 0.515953f), Point3D(0.8125f, 0.402344f, 0.481621f)},
LeafNode{0.69874f, Point3D(0.80957f, 0.392578f, 0.490257f), Point3D(0.810547f, 0.392578f, 0.51017f), Point3D(0.80957f, 0.393555f, 0.490257f), Point3D(0.810547f, 0.393555f, 0.490257f)},
LeafNode{0.69874f, Point3D(0.80957f, 0.393555f, 0.490257f), Point3D(0.810547f, 0.393555f, 0.490257f), Point3D(0.80957f, 0.394531f, 0.50399f), Point3D(0.810547f, 0.394531f, 0.495613f)},
LeafNode{0.69874f, Point3D(0.808594f, 0.393555f, 0.502617f), Point3D(0.80957f, 0.393555f, 0.490257f), Point3D(0.808594f, 0.394531f, 0.518074f), Point3D(0.80957f, 0.394531f, 0.50399f)},
LeafNode{0.69874f, Point3D(0.80957f, 0.391602f, 0.51017f), Point3D(0.810547f, 0.391602f, 0.546563f), Point3D(0.80957f, 0.392578f, 0.490257f), Point3D(0.810547f, 0.392578f, 0.51017f)},
LeafNode{0.69874f, Point3D(0.808594f, 0.391602f, 0.490257f), Point3D(0.80957f, 0.391602f, 0.51017f), Point3D(0.808594f, 0.392578f, 0.490394f), Point3D(0.80957f, 0.392578f, 0.490257f)},
LeafNode{0.69874f, Point3D(0.808594f, 0.390625f, 0.508263f), Point3D(0.80957f, 0.390625f, 0.54548f), Point3D(0.808594f, 0.391602f, 0.490257f), Point3D(0.80957f, 0.391602f, 0.51017f)},
LeafNode{0.69874f, Point3D(0.798828f, 0.349609f, 0.490837f), Point3D(0.799805f, 0.349609f, 0.513161f), Point3D(0.798828f, 0.350586f, 0.539254f), Point3D(0.799805f, 0.350586f, 0.560845f)},
LeafNode{0.69874f, Point3D(0.797852f, 0.349609f, 0.46923f), Point3D(0.798828f, 0.349609f, 0.490837f), Point3D(0.797852f, 0.350586f, 0.506493f), Point3D(0.798828f, 0.350586f, 0.539254f)},
LeafNode{0.69874f, Point3D(0.797852f, 0.348633f, 0.485618f), Point3D(0.798828f, 0.348633f, 0.496773f), Point3D(0.797852f, 0.349609f, 0.46923f), Point3D(0.798828f, 0.349609f, 0.490837f)},
LeafNode{0.69874f, Point3D(0.797852f, 0.347656f, 0.501335f), Point3D(0.798828f, 0.347656f, 0.507134f), Point3D(0.797852f, 0.348633f, 0.485618f), Point3D(0.798828f, 0.348633f, 0.496773f)},
LeafNode{0.69874f, Point3D(0.796875f, 0.347656f, 0.496315f), Point3D(0.797852f, 0.347656f, 0.501335f), Point3D(0.796875f, 0.348633f, 0.472526f), Point3D(0.797852f, 0.348633f, 0.485618f)},
LeafNode{0.69874f, Point3D(0.794922f, 0.334961f, 0.497566f), Point3D(0.795898f, 0.334961f, 0.495369f), Point3D(0.794922f, 0.335938f, 0.501015f), Point3D(0.795898f, 0.335938f, 0.499046f)},
LeafNode{0.69874f, Point3D(0.793945f, 0.333984f, 0.494118f), Point3D(0.794922f, 0.333984f, 0.491768f), Point3D(0.793945f, 0.334961f, 0.499763f), Point3D(0.794922f, 0.334961f, 0.497566f)},
LeafNode{0.69874f, Point3D(0.792969f, 0.333008f, 0.490959f), Point3D(0.793945f, 0.333008f, 0.486275f), Point3D(0.792969f, 0.333984f, 0.497093f), Point3D(0.793945f, 0.333984f, 0.494118f)},
LeafNode{0.69874f, Point3D(0.791992f, 0.329102f, 0.520211f), Point3D(0.792969f, 0.329102f, 0.48217f), Point3D(0.791992f, 0.330078f, 0.495506f), Point3D(0.792969f, 0.330078f, 0.472145f)},
LeafNode{0.69874f, Point3D(0.791992f, 0.326172f, 0.528908f), Point3D(0.792969f, 0.326172f, 0.487495f), Point3D(0.791992f, 0.327148f, 0.534081f), Point3D(0.792969f, 0.327148f, 0.487495f)},
LeafNode{0.69874f, Point3D(0.791992f, 0.327148f, 0.534081f), Point3D(0.792969f, 0.327148f, 0.487495f), Point3D(0.791992f, 0.328125f, 0.529885f), Point3D(0.792969f, 0.328125f, 0.484825f)},
LeafNode{0.69874f, Point3D(0.791992f, 0.325195f, 0.515312f), Point3D(0.792969f, 0.325195f, 0.481727f), Point3D(0.791992f, 0.326172f, 0.528908f), Point3D(0.792969f, 0.326172f, 0.487495f)},
LeafNode{0.69874f, Point3D(0.791992f, 0.324219f, 0.494621f), Point3D(0.792969f, 0.324219f, 0.472282f), Point3D(0.791992f, 0.325195f, 0.515312f), Point3D(0.792969f, 0.325195f, 0.481727f)},
LeafNode{0.69874f, Point3D(0.791016f, 0.324219f, 0.543038f), Point3D(0.791992f, 0.324219f, 0.494621f), Point3D(0.791016f, 0.325195f, 0.56643f), Point3D(0.791992f, 0.325195f, 0.515312f)},
LeafNode{0.69874f, Point3D(0.791016f, 0.323242f, 0.502953f), Point3D(0.791992f, 0.323242f, 0.476448f), Point3D(0.791016f, 0.324219f, 0.543038f), Point3D(0.791992f, 0.324219f, 0.494621f)},
LeafNode{0.69874f, Point3D(0.791016f, 0.322266f, 0.460807f), Point3D(0.791992f, 0.322266f, 0.454887f), Point3D(0.791016f, 0.323242f, 0.502953f), Point3D(0.791992f, 0.323242f, 0.476448f)},
LeafNode{0.69874f, Point3D(0.790039f, 0.31543f, 0.495766f), Point3D(0.791016f, 0.31543f, 0.499138f), Point3D(0.790039f, 0.316406f, 0.480842f), Point3D(0.791016f, 0.316406f, 0.487373f)},
LeafNode{0.69874f, Point3D(0.789062f, 0.31543f, 0.487724f), Point3D(0.790039f, 0.31543f, 0.495766f), Point3D(0.789062f, 0.316406f, 0.460243f), Point3D(0.790039f, 0.316406f, 0.480842f)},
LeafNode{0.69874f, Point3D(0.789062f, 0.314453f, 0.494438f), Point3D(0.790039f, 0.314453f, 0.500313f), Point3D(0.789062f, 0.31543f, 0.487724f), Point3D(0.790039f, 0.31543f, 0.495766f)},
LeafNode{0.69874f, Point3D(0.789062f, 0.3125f, 0.493217f), Point3D(0.790039f, 0.3125f, 0.49604f), Point3D(0.789062f, 0.313477f, 0.494987f), Point3D(0.790039f, 0.313477f, 0.500313f)},
LeafNode{0.69874f, Point3D(0.789062f, 0.313477f, 0.494987f), Point3D(0.790039f, 0.313477f, 0.500313f), Point3D(0.789062f, 0.314453f, 0.494438f), Point3D(0.790039f, 0.314453f, 0.500313f)},
LeafNode{0.69874f, Point3D(0.788086f, 0.311523f, 0.496773f), Point3D(0.789062f, 0.311523f, 0.493523f), Point3D(0.788086f, 0.3125f, 0.485863f), Point3D(0.789062f, 0.3125f, 0.493217f)},
LeafNode{0.69874f, Point3D(0.788086f, 0.310547f, 0.517952f), Point3D(0.789062f, 0.310547f, 0.508293f), Point3D(0.788086f, 0.311523f, 0.496773f), Point3D(0.789062f, 0.311523f, 0.493523f)},
LeafNode{0.69874f, Point3D(0.775391f, 0.259766f, 0.487739f), Point3D(0.776367f, 0.259766f, 0.4907f), Point3D(0.775391f, 0.260742f, 0.515694f), Point3D(0.776367f, 0.260742f, 0.529427f)},
LeafNode{0.69874f, Point3D(0.774414f, 0.259766f, 0.495094f), Point3D(0.775391f, 0.259766f, 0.487739f), Point3D(0.774414f, 0.260742f, 0.515694f), Point3D(0.775391f, 0.260742f, 0.515694f)},
LeafNode{0.69874f, Point3D(0.774414f, 0.254883f, 0.486595f), Point3D(0.775391f, 0.254883f, 0.499107f), Point3D(0.774414f, 0.255859f, 0.483574f), Point3D(0.775391f, 0.255859f, 0.488319f)},
LeafNode{0.69874f, Point3D(0.773438f, 0.253906f, 0.485084f), Point3D(0.774414f, 0.253906f, 0.498924f), Point3D(0.773438f, 0.254883f, 0.478004f), Point3D(0.774414f, 0.254883f, 0.486595f)},
LeafNode{0.69874f, Point3D(0.773438f, 0.25293f, 0.500038f), Point3D(0.774414f, 0.25293f, 0.513924f), Point3D(0.773438f, 0.253906f, 0.485084f), Point3D(0.774414f, 0.253906f, 0.498924f)},
LeafNode{0.69874f, Point3D(0.770508f, 0.240234f, 0.482658f), Point3D(0.771484f, 0.240234f, 0.470298f), Point3D(0.770508f, 0.241211f, 0.497642f), Point3D(0.771484f, 0.241211f, 0.484993f)},
LeafNode{0.69874f, Point3D(0.770508f, 0.241211f, 0.497642f), Point3D(0.771484f, 0.241211f, 0.484993f), Point3D(0.770508f, 0.242188f, 0.50837f), Point3D(0.771484f, 0.242188f, 0.4963f)},
LeafNode{0.69874f, Point3D(0.769531f, 0.240234f, 0.495003f), Point3D(0.770508f, 0.240234f, 0.482658f), Point3D(0.769531f, 0.241211f, 0.506966f), Point3D(0.770508f, 0.241211f, 0.497642f)},
LeafNode{0.69874f, Point3D(0.767578f, 0.231445f, 0.505135f), Point3D(0.768555f, 0.231445f, 0.500465f), Point3D(0.767578f, 0.232422f, 0.490486f), Point3D(0.768555f, 0.232422f, 0.488151f)},
LeafNode{0.69874f, Point3D(0.766602f, 0.231445f, 0.505135f), Point3D(0.767578f, 0.231445f, 0.505135f), Point3D(0.766602f, 0.232422f, 0.488472f), Point3D(0.767578f, 0.232422f, 0.490486f)},
LeafNode{0.69874f, Point3D(0.765625f, 0.222656f, 0.490486f), Point3D(0.766602f, 0.222656f, 0.491417f), Point3D(0.765625f, 0.223633f, 0.504265f), Point3D(0.766602f, 0.223633f, 0.501961f)},
LeafNode{0.69874f, Point3D(0.762695f, 0.214844f, 0.504768f), Point3D(0.763672f, 0.214844f, 0.49424f), Point3D(0.762695f, 0.21582f, 0.493492f), Point3D(0.763672f, 0.21582f, 0.481498f)},
LeafNode{0.69874f, Point3D(0.762695f, 0.213867f, 0.510857f), Point3D(0.763672f, 0.213867f, 0.501686f), Point3D(0.762695f, 0.214844f, 0.504768f), Point3D(0.763672f, 0.214844f, 0.49424f)},
LeafNode{0.69874f, Point3D(0.758789f, 0.194336f, 0.495018f), Point3D(0.759766f, 0.194336f, 0.495018f), Point3D(0.758789f, 0.195312f, 0.502358f), Point3D(0.759766f, 0.195312f, 0.501564f)},
LeafNode{0.69874f, Point3D(0.757812f, 0.194336f, 0.494652f), Point3D(0.758789f, 0.194336f, 0.495018f), Point3D(0.757812f, 0.195312f, 0.501991f), Point3D(0.758789f, 0.195312f, 0.502358f)},
LeafNode{0.69874f, Point3D(0.756836f, 0.19043f, 0.514214f), Point3D(0.757812f, 0.19043f, 0.47747f), Point3D(0.756836f, 0.191406f, 0.488319f), Point3D(0.757812f, 0.191406f, 0.471839f)},
LeafNode{0.69874f, Point3D(0.756836f, 0.189453f, 0.542641f), Point3D(0.757812f, 0.189453f, 0.503304f), Point3D(0.756836f, 0.19043f, 0.514214f), Point3D(0.757812f, 0.19043f, 0.47747f)},
LeafNode{0.69874f, Point3D(0.75f, 0.162109f, 0.489219f), Point3D(0.750977f, 0.162109f, 0.510201f), Point3D(0.75f, 0.163086f, 0.525322f), Point3D(0.750977f, 0.163086f, 0.536065f)},
LeafNode{0.69874f, Point3D(0.75f, 0.161133f, 0.448997f), Point3D(0.750977f, 0.161133f, 0.471977f), Point3D(0.75f, 0.162109f, 0.489219f), Point3D(0.750977f, 0.162109f, 0.510201f)},
LeafNode{0.69874f, Point3D(0.74707f, 0.149414f, 0.528206f), Point3D(0.748047f, 0.149414f, 0.533837f), Point3D(0.74707f, 0.150391f, 0.464714f), Point3D(0.748047f, 0.150391f, 0.482383f)},
LeafNode{0.69874f, Point3D(0.746094f, 0.149414f, 0.518059f), Point3D(0.74707f, 0.149414f, 0.528206f), Point3D(0.746094f, 0.150391f, 0.451743f), Point3D(0.74707f, 0.150391f, 0.464714f)},
LeafNode{0.69874f, Point3D(0.744141f, 0.141602f, 0.485664f), Point3D(0.745117f, 0.141602f, 0.505959f), Point3D(0.744141f, 0.142578f, 0.505959f), Point3D(0.745117f, 0.142578f, 0.518029f)},
LeafNode{0.69874f, Point3D(0.744141f, 0.140625f, 0.449851f), Point3D(0.745117f, 0.140625f, 0.477928f), Point3D(0.744141f, 0.141602f, 0.485664f), Point3D(0.745117f, 0.141602f, 0.505959f)},
LeafNode{0.69874f, Point3D(0.742188f, 0.132812f, 0.513664f), Point3D(0.743164f, 0.132812f, 0.495277f), Point3D(0.742188f, 0.133789f, 0.494301f), Point3D(0.743164f, 0.133789f, 0.488426f)},
LeafNode{0.69874f, Point3D(0.742188f, 0.131836f, 0.521355f), Point3D(0.743164f, 0.131836f, 0.500893f), Point3D(0.742188f, 0.132812f, 0.513664f), Point3D(0.743164f, 0.132812f, 0.495277f)},
LeafNode{0.69874f, Point3D(0.734375f, 0.103516f, 0.474601f), Point3D(0.735352f, 0.103516f, 0.478599f), Point3D(0.734375f, 0.104492f, 0.499062f), Point3D(0.735352f, 0.104492f, 0.49929f)},
LeafNode{0.69874f, Point3D(0.733398f, 0.103516f, 0.470588f), Point3D(0.734375f, 0.103516f, 0.474601f), Point3D(0.733398f, 0.104492f, 0.498817f), Point3D(0.734375f, 0.104492f, 0.499062f)},
LeafNode{0.69874f, Point3D(0.733398f, 0.0966797f, 0.504784f), Point3D(0.734375f, 0.0966797f, 0.49984f), Point3D(0.733398f, 0.0976562f, 0.490242f), Point3D(0.734375f, 0.0976562f, 0.476905f)},
LeafNode{0.69874f, Point3D(0.732422f, 0.0966797f, 0.505333f), Point3D(0.733398f, 0.0966797f, 0.504784f), Point3D(0.732422f, 0.0976562f, 0.493263f), Point3D(0.733398f, 0.0976562f, 0.490242f)},
LeafNode{0.69874f, Point3D(0.730469f, 0.0917969f, 0.494728f), Point3D(0.731445f, 0.0917969f, 0.501793f), Point3D(0.730469f, 0.0927734f, 0.498985f), Point3D(0.731445f, 0.0927734f, 0.50782f)},
LeafNode{0.69874f, Point3D(0.731445f, 0.0898438f, 0.486244f), Point3D(0.732422f, 0.0898438f, 0.48925f), Point3D(0.731445f, 0.0908203f, 0.493965f), Point3D(0.732422f, 0.0908203f, 0.500008f)},
LeafNode{0.69874f, Point3D(0.731445f, 0.0908203f, 0.493965f), Point3D(0.732422f, 0.0908203f, 0.500008f), Point3D(0.731445f, 0.0917969f, 0.501793f), Point3D(0.732422f, 0.0917969f, 0.506859f)},
LeafNode{0.69874f, Point3D(0.727539f, 0.078125f, 0.489921f), Point3D(0.728516f, 0.078125f, 0.496086f), Point3D(0.727539f, 0.0791016f, 0.483162f), Point3D(0.728516f, 0.0791016f, 0.492332f)},
LeafNode{0.69874f, Point3D(0.727539f, 0.0771484f, 0.492866f), Point3D(0.728516f, 0.0771484f, 0.498039f), Point3D(0.727539f, 0.078125f, 0.489921f), Point3D(0.728516f, 0.078125f, 0.496086f)},
LeafNode{0.69874f, Point3D(0.727539f, 0.0761719f, 0.492866f), Point3D(0.728516f, 0.0761719f, 0.498039f), Point3D(0.727539f, 0.0771484f, 0.492866f), Point3D(0.728516f, 0.0771484f, 0.498039f)},
LeafNode{0.69874f, Point3D(0.727539f, 0.0751953f, 0.487816f), Point3D(0.728516f, 0.0751953f, 0.493812f), Point3D(0.727539f, 0.0761719f, 0.492866f), Point3D(0.728516f, 0.0761719f, 0.498039f)},
LeafNode{0.69874f, Point3D(0.726562f, 0.0712891f, 0.505547f), Point3D(0.727539f, 0.0712891f, 0.500114f), Point3D(0.726562f, 0.0722656f, 0.49218f), Point3D(0.727539f, 0.0722656f, 0.486397f)},
LeafNode{0.69874f, Point3D(0.725586f, 0.0722656f, 0.497063f), Point3D(0.726562f, 0.0722656f, 0.49218f), Point3D(0.725586f, 0.0732422f, 0.468391f), Point3D(0.726562f, 0.0732422f, 0.469902f)},
LeafNode{0.69874f, Point3D(0.724609f, 0.0664062f, 0.491218f), Point3D(0.725586f, 0.0664062f, 0.493141f), Point3D(0.724609f, 0.0673828f, 0.506935f), Point3D(0.725586f, 0.0673828f, 0.508858f)},
LeafNode{0.69874f, Point3D(0.722656f, 0.0585938f, 0.499672f), Point3D(0.723633f, 0.0585938f, 0.474525f), Point3D(0.722656f, 0.0595703f, 0.467887f), Point3D(0.723633f, 0.0595703f, 0.439109f)},
LeafNode{0.69874f, Point3D(0.722656f, 0.0576172f, 0.51339f), Point3D(0.723633f, 0.0576172f, 0.502129f), Point3D(0.722656f, 0.0585938f, 0.499672f), Point3D(0.723633f, 0.0585938f, 0.474525f)},
LeafNode{0.69874f, Point3D(0.720703f, 0.0507812f, 0.49308f), Point3D(0.72168f, 0.0507812f, 0.500877f), Point3D(0.720703f, 0.0517578f, 0.49192f), Point3D(0.72168f, 0.0517578f, 0.496468f)},
LeafNode{0.69874f, Point3D(0.720703f, 0.0517578f, 0.49192f), Point3D(0.72168f, 0.0517578f, 0.496468f), Point3D(0.720703f, 0.0527344f, 0.503212f), Point3D(0.72168f, 0.0527344f, 0.506661f)},
LeafNode{0.69874f, Point3D(0.720703f, 0.0498047f, 0.508644f), Point3D(0.72168f, 0.0498047f, 0.521645f), Point3D(0.720703f, 0.0507812f, 0.49308f), Point3D(0.72168f, 0.0507812f, 0.500877f)},
LeafNode{0.69874f, Point3D(0.719727f, 0.0498047f, 0.501259f), Point3D(0.720703f, 0.0498047f, 0.508644f), Point3D(0.719727f, 0.0507812f, 0.494179f), Point3D(0.720703f, 0.0507812f, 0.49308f)},
LeafNode{0.69874f, Point3D(0.71875f, 0.0449219f, 0.483101f), Point3D(0.719727f, 0.0449219f, 0.520989f), Point3D(0.71875f, 0.0458984f, 0.49749f), Point3D(0.719727f, 0.0458984f, 0.529641f)},
LeafNode{0.69874f, Point3D(0.71875f, 0.0439453f, 0.454154f), Point3D(0.719727f, 0.0439453f, 0.500404f), Point3D(0.71875f, 0.0449219f, 0.483101f), Point3D(0.719727f, 0.0449219f, 0.520989f)},
LeafNode{0.69874f, Point3D(0.71875f, 0.0429688f, 0.429374f), Point3D(0.719727f, 0.0429688f, 0.473152f), Point3D(0.71875f, 0.0439453f, 0.454154f), Point3D(0.719727f, 0.0439453f, 0.500404f)},
LeafNode{0.69874f, Point3D(0.714844f, 0.03125f, 0.520623f), Point3D(0.71582f, 0.03125f, 0.491539f), Point3D(0.714844f, 0.0322266f, 0.497414f), Point3D(0.71582f, 0.0322266f, 0.471611f)},
LeafNode{0.69874f, Point3D(0.71582f, 0.0302734f, 0.504631f), Point3D(0.716797f, 0.0302734f, 0.464164f), Point3D(0.71582f, 0.03125f, 0.491539f), Point3D(0.716797f, 0.03125f, 0.456123f)},
LeafNode{0.69874f, Point3D(0.71582f, 0.0292969f, 0.504631f), Point3D(0.716797f, 0.0292969f, 0.46746f), Point3D(0.71582f, 0.0302734f, 0.504631f), Point3D(0.716797f, 0.0302734f, 0.464164f)},
LeafNode{0.69874f, Point3D(0.709961f, 0.0078125f, 0.500206f), Point3D(0.710938f, 0.0078125f, 0.494163f), Point3D(0.709961f, 0.00878906f, 0.507652f), Point3D(0.710938f, 0.00878906f, 0.50634f)},
LeafNode{0.69874f, Point3D(0.709961f, 0.00683594f, 0.485206f), Point3D(0.710938f, 0.00683594f, 0.472374f), Point3D(0.709961f, 0.0078125f, 0.500206f), Point3D(0.710938f, 0.0078125f, 0.494163f)},
LeafNode{0.69874f, Point3D(0.708984f, 0.00683594f, 0.486168f), Point3D(0.709961f, 0.00683594f, 0.485206f), Point3D(0.708984f, 0.0078125f, 0.49691f), Point3D(0.709961f, 0.0078125f, 0.500206f)},
    };

    Vector3D e0, e1;
    rayDir.makeCoordinateSystem(&e0, &e1);

    float alpha2, alpha1, alpha0;
    float beta2, beta1, beta0;
    float denom2, denom1, denom0;
    computeCanonicalSpaceRayCoeffs(
        rayOrg, rayDir, e0, e1,
        pA, pB, pC,
        nA, nB, nC,
        &alpha2, &alpha1, &alpha0,
        &beta2, &beta1, &beta0,
        &denom2, &denom1, &denom0);

    Point2D tc2, tc1, tc0;
    computeTextureSpaceRayCoeffs(
        tcA, tcB, tcC,
        alpha2, alpha1, alpha0,
        beta2, beta1, beta0,
        denom2, denom1, denom0,
        &tc2, &tc1, &tc0);

    vdb_frame();

    constexpr float axisScale = 1.0f;
    //drawAxes(axisScale);

    // Texture-space Ray
    constexpr bool visNegRay = false;
    std::vector<float> heightValues;
    std::vector<int32_t> indices;
    std::vector<Point3D> texPs;
    int32_t heightIdx = 0;
    for (int i = 0; i <= 500; ++i) {
        const float t = static_cast<float>(i) / 500;
        float hs[3];
        findHeight(
            pA, pB, pC,
            nA, nB, nC,
            rayOrg + t * rayLength * rayDir,
            hs);
        for (int j = 0; j < 3; ++j) {
            const float h = hs[j];
            if (!std::isfinite(h))
                continue;
            const float h2 = pow2(h);
            const float denom = denom2 * h2 + denom1 * h + denom0;
            const Point3D p(
                (alpha2 * h2 + alpha1 * h + alpha0) / denom,
                (beta2 * h2 + beta1 * h + beta0) / denom,
                h);
            const Point3D tcp((tc2 * h2 + tc1 * h + tc0) / denom, h);

            heightValues.push_back(h);
            indices.push_back(heightIdx++);
            texPs.push_back(tcp);
        }
    }
    std::vector<int32_t> negIndices;
    if constexpr (visNegRay) {
        for (int i = 0; i <= 500; ++i) {
            const float t = -static_cast<float>(i) / 500;
            float hs[3];
            findHeight(
                pA, pB, pC,
                nA, nB, nC,
                rayOrg + t * rayLength * rayDir,
                hs);
            for (int j = 0; j < 3; ++j) {
                const float h = hs[j];
                if (!std::isfinite(h))
                    continue;
                const float h2 = pow2(h);
                const float denom = denom2 * h2 + denom1 * h + denom0;
                const Point3D p(
                    (alpha2 * h2 + alpha1 * h + alpha0) / denom,
                    (beta2 * h2 + beta1 * h + beta0) / denom,
                    h);
                const Point3D tcp((tc2 * h2 + tc1 * h + tc0) / denom, h);

                heightValues.push_back(h);
                negIndices.push_back(heightIdx++);
                texPs.push_back(tcp);
            }
        }
    }

    std::sort(
        indices.begin(), indices.end(),
        [&](int32_t a, int32_t b) { return heightValues[a] < heightValues[b]; });
    if constexpr (visNegRay) {
        std::sort(
            negIndices.begin(), negIndices.end(),
            [&](int32_t a, int32_t b) { return heightValues[a] < heightValues[b]; });
    }

    Point3D prevRayPInTex = texPs[*indices.cbegin()];
    setColor(RGB(1.0f));
    drawCross(texPs[0], 0.05f);
    for (auto it = ++indices.cbegin(); it != indices.cend(); ++it) {
        const Point3D &tcp = texPs[*it];
        drawLine(prevRayPInTex, tcp);
        prevRayPInTex = tcp;
    }
    if constexpr (visNegRay) {
        prevRayPInTex = texPs[*negIndices.cbegin()];
        setColor(RGB(0.1f));
        for (auto it = ++negIndices.cbegin(); it != negIndices.cend(); ++it) {
            const Point3D &tcp = texPs[*it];
            drawLine(prevRayPInTex, tcp);
            prevRayPInTex = tcp;
        }
    }

    // Traversal steps
    setColor(RGB(1));
    for (int stepIdx = 0; stepIdx < steps.size(); ++stepIdx) {
        const TraversalStep &step = steps[stepIdx];
        if (std::holds_alternative<InternalNode>(step)) {
            const InternalNode &node = std::get<InternalNode>(step);
            for (int i = 0; i < 4; ++i) {
                const AABB &aabb = node.aabbs[i];
                drawAabb(aabb);

                const auto solveQuadraticEquation = [](
                    const float a, const float b, const float c, const float xMin, const float xMax,
                    float roots[2]) {
                    const float coeffs[] = { c, b, a };
                    const uint32_t numRoots = ::solveQuadraticEquation(coeffs, xMin, xMax, roots);
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
                    for (int i = 0; i < 2; ++i) {
                        us[i] = NAN;
                        if (stc::isfinite(hs[i])) {
                            us[i] = evaluateQuadraticPolynomial(tc2.x, tc1.x, tc0.x, hs[i])
                                / evaluateQuadraticPolynomial(denom2, denom1, denom0, hs[i]);
                        }
                    }
                };

                float hs_uLo[2], vs_uLo[2];
                float hs_uHi[2], vs_uHi[2];
                float hs_vLo[2], us_vLo[2];
                float hs_vHi[2], us_vHi[2];
                compute_h_v(aabb.minP.x, hs_uLo, vs_uLo);
                compute_h_v(aabb.maxP.x, hs_uHi, vs_uHi);
                compute_h_v(aabb.minP.y, hs_vLo, us_vLo);
                compute_h_v(aabb.maxP.y, hs_vHi, us_vHi);

                float hitDistMin, hitDistMax;
                if (testNonlinearRayVsAabb(
                    pA, pB, pC,
                    nA, nB, nC,
                    aabb,
                    rayOrg, rayDir, prismHitDistEnter, node.distMax,
                    alpha2, alpha1, alpha0,
                    beta2, beta1, beta0,
                    denom2, denom1, denom0,
                    tc2, tc1, tc0,
                    hs_uLo, vs_uLo,
                    hs_uHi, vs_uHi,
                    hs_vLo, us_vLo,
                    hs_vHi, us_vHi,
                    &hitDistMin, &hitDistMax)) {
                    printf("");
                }
            }
        }
        else {
            const LeafNode &node = std::get<LeafNode>(step);
            drawWiredTriangle(node.mpTL, node.mpBL, node.mpBR);
            drawWiredTriangle(node.mpTL, node.mpBR, node.mpTR);

            Point3D hpInCan;
            Point3D hpInTex;
            float hitDist;
            Normal3D hitNormal;
            if (testNonlinearRayVsMicroTriangle(
                pA, pB, pC,
                nA, nB, nC,
                tcA, tcB, tcC,
                node.mpTL, node.mpBL, node.mpBR,
                rayOrg, rayDir, prismHitDistEnter, node.distMax,
                e0, e1,
                tc2, tc1, tc0,
                denom2, denom1, denom0,
                &hpInCan, &hpInTex, &hitDist, &hitNormal)) {
                printf("");
            }
            if (testNonlinearRayVsMicroTriangle(
                pA, pB, pC,
                nA, nB, nC,
                tcA, tcB, tcC,
                node.mpTL, node.mpBR, node.mpTR,
                rayOrg, rayDir, prismHitDistEnter, node.distMax,
                e0, e1,
                tc2, tc1, tc0,
                denom2, denom1, denom0,
                &hpInCan, &hpInTex, &hitDist, &hitNormal)) {
                printf("");
            }
        }
        printf("");
    }

    printf("");
}



struct TriangleMesh {
    std::vector<shared::Vertex> vertices;
    std::vector<shared::Triangle> triangles;
};

struct FlattenedNode {
    Matrix4x4 transform;
    std::vector<uint32_t> meshIndices;
};

static void computeFlattenedNodes(
    const aiScene* scene, const Matrix4x4 &parentXfm, const aiNode* curNode,
    std::vector<FlattenedNode> &flattenedNodes) {
    aiMatrix4x4 curAiXfm = curNode->mTransformation;
    Matrix4x4 curXfm = Matrix4x4(
        Vector4D(curAiXfm.a1, curAiXfm.a2, curAiXfm.a3, curAiXfm.a4),
        Vector4D(curAiXfm.b1, curAiXfm.b2, curAiXfm.b3, curAiXfm.b4),
        Vector4D(curAiXfm.c1, curAiXfm.c2, curAiXfm.c3, curAiXfm.c4),
        Vector4D(curAiXfm.d1, curAiXfm.d2, curAiXfm.d3, curAiXfm.d4));
    FlattenedNode flattenedNode;
    flattenedNode.transform = parentXfm * transpose(curXfm);
    flattenedNode.meshIndices.resize(curNode->mNumMeshes);
    if (curNode->mNumMeshes > 0) {
        std::copy_n(curNode->mMeshes, curNode->mNumMeshes, flattenedNode.meshIndices.data());
        flattenedNodes.push_back(flattenedNode);
    }

    for (uint32_t cIdx = 0; cIdx < curNode->mNumChildren; ++cIdx)
        computeFlattenedNodes(scene, flattenedNode.transform, curNode->mChildren[cIdx], flattenedNodes);
}

static void calcTriangleVertices(
    const bvh::Geometry &geom, const uint32_t primIdx,
    Point3D* const pA, Point3D* const pB, Point3D* const pC) {
    uint32_t tri[3];
    const auto triAddr = reinterpret_cast<uintptr_t>(geom.triangles) + geom.triangleStride * primIdx;
    if (geom.triangleFormat == bvh::TriangleFormat::UI32x3) {
        tri[0] = reinterpret_cast<const uint32_t*>(triAddr)[0];
        tri[1] = reinterpret_cast<const uint32_t*>(triAddr)[1];
        tri[2] = reinterpret_cast<const uint32_t*>(triAddr)[2];
    }
    else {
        Assert(geom.triangleFormat == bvh::TriangleFormat::UI16x3, "Invalid triangle format.");
        tri[0] = reinterpret_cast<const uint16_t*>(triAddr)[0];
        tri[1] = reinterpret_cast<const uint16_t*>(triAddr)[1];
        tri[2] = reinterpret_cast<const uint16_t*>(triAddr)[2];
    }

    Point3D ps[3];
    const auto vertBaseAddr = reinterpret_cast<uintptr_t>(geom.vertices);
    Assert(geom.vertexFormat == bvh::VertexFormat::Fp32x3, "Invalid vertex format.");
    ps[0] = *reinterpret_cast<const Point3D*>(vertBaseAddr + geom.vertexStride * tri[0]);
    ps[1] = *reinterpret_cast<const Point3D*>(vertBaseAddr + geom.vertexStride * tri[1]);
    ps[2] = *reinterpret_cast<const Point3D*>(vertBaseAddr + geom.vertexStride * tri[2]);

    *pA = geom.preTransform * ps[0];
    *pB = geom.preTransform * ps[1];
    *pC = geom.preTransform * ps[2];
}

void testBvhBuilder() {
    struct TestScene {
        std::filesystem::path filePath;
        Matrix4x4 transform;
        Matrix4x4 cameraTransform;
    };

    std::map<std::string, TestScene> scenes = {
        {
            "bunny",
            {
                R"(E:/assets/McguireCGArchive/bunny/bunny.obj)",
                Matrix4x4(),
                translate3D_4x4(-0.299932f, 1.73252f, 2.4276f) *
                rotate3DY_4x4(178.68f * pi_v<float> / 180) *
                rotate3DX_4x4(22.2f * pi_v<float> / 180)
            }
        },
        {
            "dragon",
            {
                R"(E:/assets/McguireCGArchive/dragon/dragon.obj)",
                Matrix4x4(),
                translate3D_4x4(-1.08556f, 0.450182f, -0.473484f) *
                rotate3DY_4x4(68.0f * pi_v<float> / 180) *
                rotate3DX_4x4(19.0f * pi_v<float> / 180)
            }
        },
        {
            "buddha",
            {
                R"(E:/assets/McguireCGArchive/buddha/buddha.obj)",
                Matrix4x4(),
                translate3D_4x4(-0.004269f, 0.342561f, -1.34414f) *
                rotate3DY_4x4(0.0f * pi_v<float> / 180) *
                rotate3DX_4x4(12.5f * pi_v<float> / 180)
            }
        },
        {
            "white_oak",
            {
                R"(E:/assets/McguireCGArchive/white_oak/white_oak.obj)",
                scale3D_4x4(0.01f),
                translate3D_4x4(2.86811f, 4.87556f, 10.4772f) *
                rotate3DY_4x4(195.5f * pi_v<float> / 180) *
                rotate3DX_4x4(8.9f * pi_v<float> / 180)
            }
        },
        {
            "conference",
            {
                R"(E:/assets/McguireCGArchive/conference/conference.obj)",
                Matrix4x4(),
                translate3D_4x4(1579.2f, 493.793f, 321.98f) *
                rotate3DY_4x4(-120 * pi_v<float> / 180) *
                rotate3DX_4x4(20 * pi_v<float> / 180)
            }
        },
        {
            "breakfast_room",
            {
                R"(E:/assets/McguireCGArchive/breakfast_room/breakfast_room.obj)",
                Matrix4x4(),
                translate3D_4x4(4.37691f, 1.8413f, 6.35917f) *
                rotate3DY_4x4(210 * pi_v<float> / 180) *
                rotate3DX_4x4(2.8f * pi_v<float> / 180)
            }
        },
        {
            "salle_de_bain",
            {
                R"(E:/assets/McguireCGArchive/salle_de_bain/salle_de_bain.obj)",
                Matrix4x4(),
                translate3D_4x4(2.56843f, 15.9865f, 45.3711f) *
                rotate3DY_4x4(191 * pi_v<float> / 180) *
                rotate3DX_4x4(6.2f * pi_v<float> / 180)
            }
        },
        {
            "crytek_sponza",
            {
                R"(E:/assets/McguireCGArchive/sponza/sponza.obj)",
                scale3D_4x4(0.01f),
                translate3D_4x4(10.0f, 2.0f, -0.5f) *
                rotate3DY_4x4(-pi_v<float> / 2)
            }
        },
        {
            "sibenik",
            {
                R"(E:/assets/McguireCGArchive/sibenik/sibenik.obj)",
                Matrix4x4(),
                translate3D_4x4(-15.0f, -3.0f, 0.0f) *
                rotate3DY_4x4(pi_v<float> / 2) *
                rotate3DX_4x4(20 * pi_v<float> / 180)
            }
        },
        {
            "hairball",
            {
                R"(E:/assets/McguireCGArchive/hairball/hairball.obj)",
                Matrix4x4(),
                translate3D_4x4(0.0f, 0.0f, 13.0f) *
                rotate3DY_4x4(pi_v<float>)
            }
        },
        {
            "rungholt",
            {
                R"(E:/assets/McguireCGArchive/rungholt/rungholt.obj)",
                scale3D_4x4(0.1f),
                translate3D_4x4(36.1611f, 5.56238f, -20.4327f) *
                rotate3DY_4x4(-53.0f * pi_v<float> / 180) *
                rotate3DX_4x4(14.2f * pi_v<float> / 180)
            }
        },
        {
            "san_miguel",
            {
                R"(E:/assets/McguireCGArchive/San_Miguel/san-miguel.obj)",
                Matrix4x4(),
                translate3D_4x4(6.2928f, 3.05034f, 7.49142f) *
                rotate3DY_4x4(125.8f * pi_v<float> / 180) *
                rotate3DX_4x4(9.3f * pi_v<float> / 180)
            }
        },
        {
            "powerplant",
            {
                R"(E:/assets/McguireCGArchive/powerplant/powerplant.obj)",
                scale3D_4x4(0.0001f),
                translate3D_4x4(-16.5697f, 5.66694f, 14.8665f) *
                rotate3DY_4x4(125.2f * pi_v<float> / 180) *
                rotate3DX_4x4(10.5f * pi_v<float> / 180)
            }
        },
        {
            "box",
            {
                R"(E:/assets/box/box.obj)",
                Matrix4x4(),
                translate3D_4x4(3.0f, 3.0f, 3.0f) *
                rotate3DY_4x4(225 * pi_v<float> / 180) *
                rotate3DX_4x4(35.264f * pi_v<float> / 180)
            }
        },
        {
            "lowpoly_bunny",
            {
                R"(E:/assets/lowpoly_bunny/stanford_bunny_309_faces.obj)",
                scale3D_4x4(0.1f),
                translate3D_4x4(-4.60892f, 9.15149f, 11.7878f) *
                rotate3DY_4x4(161.4f * pi_v<float> / 180) *
                rotate3DX_4x4(23.6f * pi_v<float> / 180)
            }
        },
        {
            "teapot",
            {
                R"(E:/assets/McguireCGArchive/teapot/teapot.obj)",
                Matrix4x4(),
                translate3D_4x4(0.0f, 133.3f, 200.0f) *
                rotate3DY_4x4(180 * pi_v<float> / 180) *
                rotate3DX_4x4(25 * pi_v<float> / 180)
            }
        },
        {
            "one_tri",
            {
                R"(E:/assets/one_tri.obj)",
                Matrix4x4(),
                translate3D_4x4(0.0f, 0.0f, 3.0f) *
                rotate3DY_4x4(180 * pi_v<float> / 180)
            }
        },
        {
            "two_tris",
            {
                R"(E:/assets/two_tris.obj)",
                Matrix4x4(),
                translate3D_4x4(0.0f, 0.0f, 3.0f) *
                rotate3DY_4x4(180 * pi_v<float> / 180)
            }
        },
    };

    //const TestScene &scene = scenes.at("lowpoly_bunny");
    //const TestScene &scene = scenes.at("conference");
    //const TestScene &scene = scenes.at("hairball");
    //const TestScene &scene = scenes.at("breakfast_room");
    const TestScene &scene = scenes.at("powerplant");
    //const TestScene &scene = scenes.at("san_miguel");
    constexpr uint32_t maxNumIntersections = 128;
    constexpr uint32_t singleCamIdx = 0;
    constexpr bool visStats = false;

    constexpr uint32_t arity = 8;

    bvh::GeometryBVHBuildConfig config = {};
    config.splittingBudget = 0.3f;
    config.intNodeTravCost = 1.2f;
    config.primIntersectCost = 1.0f;
    config.minNumPrimsPerLeaf = 1;
    config.maxNumPrimsPerLeaf = 128;

    hpprintf("Reading: %s ... ", scene.filePath.string().c_str());
    fflush(stdout);
    Assimp::Importer importer;
    const aiScene* aiscene = importer.ReadFile(
        scene.filePath.string(),
        aiProcess_Triangulate |
        aiProcess_GenNormals |
        aiProcess_CalcTangentSpace |
        aiProcess_FlipUVs);
    if (!aiscene) {
        hpprintf("Failed to load %s.\n", scene.filePath.string().c_str());
        return;
    }
    hpprintf("done.\n");

    std::vector<TriangleMesh> meshes;
    for (uint32_t meshIdx = 0; meshIdx < aiscene->mNumMeshes; ++meshIdx) {
        const aiMesh* aiMesh = aiscene->mMeshes[meshIdx];

        std::vector<shared::Vertex> vertices(aiMesh->mNumVertices);
        for (int vIdx = 0; vIdx < vertices.size(); ++vIdx) {
            const aiVector3D &aip = aiMesh->mVertices[vIdx];
            const aiVector3D &ain = aiMesh->mNormals[vIdx];
            aiVector3D aitc0dir;
            if (aiMesh->mTangents)
                aitc0dir = aiMesh->mTangents[vIdx];
            if (!aiMesh->mTangents || !std::isfinite(aitc0dir.x)) {
                const auto makeCoordinateSystem = []
                (const Normal3D &normal, Vector3D* tangent, Vector3D* bitangent) {
                    float sign = normal.z >= 0 ? 1.0f : -1.0f;
                    const float a = -1 / (sign + normal.z);
                    const float b = normal.x * normal.y * a;
                    *tangent = Vector3D(1 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
                    *bitangent = Vector3D(b, sign + normal.y * normal.y * a, -normal.y);
                };
                Vector3D tangent, bitangent;
                makeCoordinateSystem(Normal3D(ain.x, ain.y, ain.z), &tangent, &bitangent);
                aitc0dir = aiVector3D(tangent.x, tangent.y, tangent.z);
            }
            const aiVector3D ait = aiMesh->mTextureCoords[0] ?
                aiMesh->mTextureCoords[0][vIdx] :
                aiVector3D(0.0f, 0.0f, 0.0f);

            shared::Vertex v;
            v.position = Point3D(aip.x, aip.y, aip.z);
            v.normal = normalize(Normal3D(ain.x, ain.y, ain.z));
            v.texCoord0Dir = normalize(Vector3D(aitc0dir.x, aitc0dir.y, aitc0dir.z));
            v.texCoord = Point2D(ait.x, ait.y);
            vertices[vIdx] = v;
        }

        std::vector<shared::Triangle> triangles(aiMesh->mNumFaces);
        for (int fIdx = 0; fIdx < triangles.size(); ++fIdx) {
            const aiFace &aif = aiMesh->mFaces[fIdx];
            Assert(aif.mNumIndices == 3, "Number of face vertices must be 3 here.");
            shared::Triangle tri;
            tri.index0 = aif.mIndices[0];
            tri.index1 = aif.mIndices[1];
            tri.index2 = aif.mIndices[2];
            triangles[fIdx] = tri;
        }

        TriangleMesh mesh;
        mesh.vertices = std::move(vertices);
        mesh.triangles = std::move(triangles);
        meshes.push_back(std::move(mesh));
    }

    std::vector<FlattenedNode> flattenedNodes;
    computeFlattenedNodes(aiscene, scene.transform, aiscene->mRootNode, flattenedNodes);

    std::vector<bvh::Geometry> bvhGeoms;
    uint32_t numGlobalPrimitives = 0;
    for (uint32_t i = 0; i < flattenedNodes.size(); ++i) {
        const FlattenedNode &fNode = flattenedNodes[i];
        for (uint32_t mIdx = 0; mIdx < fNode.meshIndices.size(); ++mIdx) {
            const TriangleMesh &triMesh = meshes[fNode.meshIndices[mIdx]];

            bvh::Geometry geom = {};
            geom.vertices = triMesh.vertices.data();
            geom.vertexStride = sizeof(triMesh.vertices[0]);
            geom.vertexFormat = bvh::VertexFormat::Fp32x3;
            geom.numVertices = static_cast<uint32_t>(triMesh.vertices.size());
            geom.triangles = triMesh.triangles.data();
            geom.triangleStride = sizeof(triMesh.triangles[0]);
            geom.triangleFormat = bvh::TriangleFormat::UI32x3;
            geom.numTriangles = static_cast<uint32_t>(triMesh.triangles.size());
            geom.preTransform = fNode.transform;
            bvhGeoms.push_back(geom);

            numGlobalPrimitives += geom.numTriangles;
        }
    }

    bvh::GeometryBVH<arity> bvh;
    bvh::buildGeometryBVH(
        bvhGeoms.data(), static_cast<uint32_t>(bvhGeoms.size()),
        config, &bvh);

    static bool enableVdbViz = false;
    if (enableVdbViz) {
        struct StackEntry {
            uint32_t nodeIndex;
            uint32_t depth;
        };

        struct NodeChildAddress {
            uint32_t nodeIndex;
            uint32_t slot;
        };

        constexpr uint32_t maxDepth = 12;
        std::vector<StackEntry> stack;
        std::vector<std::vector<NodeChildAddress>> triToNodeChildMap(numGlobalPrimitives);

        vdb_frame();
        drawAxes(10.0f);
        setColor(1.0f, 1.0f, 1.0f);

        stack.push_back(StackEntry{ 0, 0 });
        while (!stack.empty()) {
            const StackEntry entry = stack.back();
            stack.pop_back();

            const shared::InternalNode_T<arity> &intNode = bvh.intNodes[entry.nodeIndex];
            for (int32_t slot = 0; slot < arity; ++slot) {
                if (!intNode.getChildIsValid(slot))
                    break;

                const bool isLeaf = ((intNode.internalMask >> slot) & 0b1) == 0;
                const uint32_t lowerMask = (1 << slot) - 1;
                if (isLeaf) {
                    setColor(0.1f, 0.1f, 0.1f);
                    drawAabb(intNode.getChildAabb(slot));
                    uint32_t leafOffset = intNode.leafBaseIndex + intNode.getLeafOffset(slot);
                    uint32_t chainLength = 0;
                    while (true) {
                        //hpprintf("%3u\n", leafOffset);
                        const shared::PrimitiveReference &primRef = bvh.primRefs[leafOffset++];
                        const shared::TriangleStorage &triStorage = bvh.triStorages[primRef.storageIndex];
                        ++chainLength;
                        setColor(1.0f, 1.0f, 1.0f);
                        drawWiredTriangle(triStorage.pA, triStorage.pB, triStorage.pC);
                        triToNodeChildMap[primRef.storageIndex].push_back(
                            NodeChildAddress{ entry.nodeIndex, static_cast<uint32_t>(slot) });
                        if (primRef.isLeafEnd)
                            break;
                    }
                }
                else {
                    setColor(0.0f, 0.3f * (entry.depth + 1) / maxDepth, 0.0f);
                    drawAabb(intNode.getChildAabb(slot));
                    if (entry.depth < maxDepth) {
                        const uint32_t childIdx = intNode.intNodeChildBaseIndex + intNode.getInternalChildNumber(slot);
                        stack.push_back(StackEntry{ childIdx, entry.depth + 1 });
                    }
                    else {
                        printf("");
                    }
                }
            }
        }

        if (false) {
            // Triangle to Node Children
            for (uint32_t globalPrimIdx = 0; globalPrimIdx < numGlobalPrimitives; ++globalPrimIdx) {
                const std::vector<NodeChildAddress> &refs = triToNodeChildMap[globalPrimIdx];

                vdb_frame();
                drawAxes(10.0f);

                const shared::TriangleStorage &triStorage = bvh.triStorages[globalPrimIdx];
                setColor(1.0f, 1.0f, 1.0f);
                drawWiredTriangle(triStorage.pA, triStorage.pB, triStorage.pC);

                for (uint32_t refIdx = 0; refIdx < refs.size(); ++refIdx) {
                    const NodeChildAddress &ref = refs[refIdx];
                    const shared::InternalNode_T<arity> &intNode = bvh.intNodes[ref.nodeIndex];
                    setColor(0.1f, 0.1f, 0.1f);
                    drawAabb(intNode.getChildAabb(ref.slot));
                }
                printf("");
            }

            printf("");
        }
    }

    static bool enableTraversalTest = true;
    if (enableTraversalTest) {
        constexpr uint32_t width = 1024;
        constexpr uint32_t height = 1024;
        const float aspect = static_cast<float>(width) / height;
        const float fovY = 45 * pi_v<float> / 180;

        for (uint32_t camIdx = 0; camIdx < 30; ++camIdx) {
            if (camIdx != singleCamIdx && singleCamIdx != -1)
                continue;
            const Matrix4x4 camXfm =
                rotate3DY_4x4<float>(static_cast<float>(camIdx) / 30 * 2 * pi_v<float>) *
                scene.cameraTransform;

            std::vector<float4> image(width * height);
            double sumMaxStackDepth = 0;
            int32_t maxMaxStackDepth = -1;
            double sumAvgStackAccessDepth = 0;
            float maxAvgStackAccessDepth = -INFINITY;
            constexpr int32_t fastStackDepthLimit = 12 - 1;
            uint64_t stackMemoryAccessAmount = 0;
            for (uint32_t ipy = 0; ipy < height; ++ipy) {
                for (uint32_t ipx = 0; ipx < width; ++ipx) {
                    const float px = ipx + 0.5f;
                    const float py = ipy + 0.5f;

                    const Vector3D rayDirInLocal(
                        aspect * tan(fovY * 0.5f) * (1 - 2 * px / width),
                        tan(fovY * 0.5f) * (1 - 2 * py / height),
                        1);
                    const Point3D rayOrg = camXfm * Point3D(0, 0, 0);
                    const Vector3D rayDir = camXfm * rayDirInLocal;
                    bvh::TraversalStatistics stats = {};
                    stats.fastStackDepthLimit = fastStackDepthLimit;
                    const shared::HitObject hitObj = bvh::traverse(
                        bvh, rayOrg, rayDir, 0.0f, 1e+10f, &stats/*,
                        ipx == 691 && ipy == 458*/);

                    RGB color;
                    if (visStats) {
                        const float t = static_cast<float>(
                            stc::min(stats.numAabbTests + stats.numTriTests, maxNumIntersections)) /
                            maxNumIntersections;
                        const RGB Red(1, 0, 0);
                        const RGB Green(0, 1, 0);
                        const RGB Blue(0, 0, 1);
                        color = t < 0.5f ? lerp(Blue, Green, 2.0f * t) : lerp(Green, Red, 2.0f * t - 1.0);
                    }
                    else {
                        if (hitObj.isHit()) {
                            const bvh::Geometry &geom = bvhGeoms[hitObj.geomIndex];
                            Point3D pA, pB, pC;
                            calcTriangleVertices(geom, hitObj.primIndex, &pA, &pB, &pC);
                            const Vector3D geomNormal = normalize(cross(pB - pA, pC - pA));
                            color.r = 0.5f + 0.5f * geomNormal.x;
                            color.g = 0.5f + 0.5f * geomNormal.y;
                            color.b = 0.5f + 0.5f * geomNormal.z;
                        }
                    }

                    image[width * ipy + ipx] = float4(color.toNative(), 1.0f);
                    sumMaxStackDepth += stats.maxStackDepth;
                    maxMaxStackDepth = std::max(stats.maxStackDepth, maxMaxStackDepth);
                    sumAvgStackAccessDepth += stats.avgStackAccessDepth;
                    maxAvgStackAccessDepth = std::max(stats.avgStackAccessDepth, maxAvgStackAccessDepth);
                    stackMemoryAccessAmount += stats.stackMemoryAccessAmount;
                }
            }
            hpprintf("Avg Stack Access Depth - Avg: %.3f\n", sumAvgStackAccessDepth / (width * height));
            hpprintf("                       - Max: %.3f\n", maxAvgStackAccessDepth);
            hpprintf("Max Stack Depth - Avg: %.3f\n", sumMaxStackDepth / (width * height));
            hpprintf("                - Max: %d\n", maxMaxStackDepth);
            hpprintf(
                "Stack Memory Access: %llu [B] (#FastStackEntry: %d)",
                stackMemoryAccessAmount, fastStackDepthLimit + 1);

            SDRImageSaverConfig imageSaveConfig = {};
            imageSaveConfig.applyToneMap = false;
            imageSaveConfig.apply_sRGB_gammaCorrection = false;
            imageSaveConfig.brightnessScale = 1.0f;
            imageSaveConfig.flipY = false;
            char filename[256];
            sprintf_s(filename, "trav_test_%03u.png", camIdx);
            saveImage(
                filename,
                width, height, image.data(),
                imageSaveConfig);
        }
    }

    printf("");
}



enum class TriangleSquareIntersection2DResult {
    SquareOutsideTriangle = 0,
    SquareInsideTriangle,
    SquareOverlappingTriangle
};

TriangleSquareIntersection2DResult testTriangleRectangleIntersection2D(
    const Point2D &pA, const Point2D &pB, const Point2D &pC, bool tcFlipped, const Vector2D triEdgeNormals[3],
    const Point2D &triAabbMinP, const Point2D &triAabbMaxP,
    const Point2D &rectCenter, const Vector2D &rectHalfWidth) {
    const Vector2D vRectCenter = static_cast<Vector2D>(rectCenter);
    const Point2D relTriPs[] = {
        pA - vRectCenter,
        pB - vRectCenter,
        pC - vRectCenter,
    };

    // JP: 四角形と三角形のAABBのIntersectionを計算する。
    // EN: Test intersection between the rectangle and the triangle AABB.
    if (any(min(Point2D(rectHalfWidth), triAabbMaxP - vRectCenter) <=
            max(Point2D(-rectHalfWidth), triAabbMinP - vRectCenter)))
        return TriangleSquareIntersection2DResult::SquareOutsideTriangle;

    // JP: いずれかの三角形のエッジの法線方向に四角形があるなら四角形は三角形の外にある。
    // EN: Rectangle is outside of the triangle if the rectangle is in the normal direction of any edge.
    for (int eIdx = 0; eIdx < 3; ++eIdx) {
        Vector2D eNormal = (tcFlipped ? -1 : 1) * triEdgeNormals[eIdx];
        Bool2D b = eNormal >= Vector2D(0.0f);
        Vector2D pToCorner = Point2D((b.x ? -1 : 1) * rectHalfWidth.x, (b.y ? -1 : 1) * rectHalfWidth.y)
            - relTriPs[eIdx];
        if (dot(eNormal, pToCorner) > 0)
            return TriangleSquareIntersection2DResult::SquareOutsideTriangle;
    }

    // JP: 四角形が三角形のエッジとかぶっているかどうかを調べる。
    // EN: Test if the rectangle is overlapping with some edges of the triangle.
    for (int i = 0; i < 4; ++i) {
        Point2D corner(
            (i % 2 ? -1 : 1) * rectHalfWidth.x,
            (i / 2 ? -1 : 1) * rectHalfWidth.y);
        for (int eIdx = 0; eIdx < 3; ++eIdx) {
            const Point2D &o = relTriPs[eIdx];
            const Vector2D &e1 = relTriPs[(eIdx + 1) % 3] - o;
            Vector2D e2 = corner - o;
            if ((tcFlipped ? -1 : 1) * cross(e1, e2) < 0)
                return TriangleSquareIntersection2DResult::SquareOverlappingTriangle;
        }
    }

    // JP: それ以外の場合は四角形は三角形に囲まれている。
    // EN: Otherwise, the rectangle is encompassed by the triangle.
    return TriangleSquareIntersection2DResult::SquareInsideTriangle;
}

void testTriVsRectIntersection() {
        std::mt19937 rng(14131631);
    std::uniform_real_distribution<float> u01;

    constexpr int32_t maxLevel = 4;
    constexpr int32_t maxRes = 1 << maxLevel;
    constexpr int32_t wrapMin = -3;
    constexpr int32_t wrapMax = 3;

    const auto drawGrid = [&]
    () {
        constexpr int32_t numRepeats = wrapMax - wrapMin;
        for (int i = 1; i < numRepeats * maxRes; ++i) {
            float p = static_cast<float>(i) / maxRes;
            RGB color(RGB(sRGB_degamma_s(0.05f * (1 << tzcnt(i % maxRes)))));
            if (i % maxRes == 0)
                color = RGB(0.8f, 0.8f, 0.8f);
            setColor(color);
            drawLine(Point3D(wrapMin, p + wrapMin, 0.0f), Point3D(numRepeats + wrapMin, p + wrapMin, 0.0f));
            drawLine(Point3D(p + wrapMin, wrapMin, 0.0f), Point3D(p + wrapMin, numRepeats + wrapMin, 0.0f));
        }
        setColor(RGB(0.25f, 0.0f, 0.0f));
        drawLine(Point3D(0.0f, 0.0f, 0.0025f), Point3D(wrapMin, 0.0f, 0.0025f));
        setColor(RGB(0.0f, 0.25f, 0.0f));
        drawLine(Point3D(0.0f, 0.0f, 0.0025f), Point3D(0.0f, wrapMin, 0.0025f));
        setColor(RGB(1.0f, 0.0f, 0.0f));
        drawLine(Point3D(0.0f, 0.0f, 0.0025f), Point3D(wrapMax, 0.0f, 0.0025f));
        setColor(RGB(0.0f, 1.0f, 0.0f));
        drawLine(Point3D(0.0f, 0.0f, 0.0025f), Point3D(0.0f, wrapMax, 0.0025f));
    };

    const auto drawRect = [](const Point2D &minP, const Point2D &maxP, float z) {
        drawLine(Point3D(minP.x, minP.y, z), Point3D(maxP.x, minP.y, z));
        drawLine(Point3D(minP.x, maxP.y, z), Point3D(maxP.x, maxP.y, z));
        drawLine(Point3D(minP.x, minP.y, z), Point3D(minP.x, maxP.y, z));
        drawLine(Point3D(maxP.x, minP.y, z), Point3D(maxP.x, maxP.y, z));
    };

    const auto drawTriangle = [](const Point2D &pA, const Point2D &pB, const Point2D &pC, float z) {
        drawLine(Point3D(pA, z), Point3D(pB, z));
        drawLine(Point3D(pB, z), Point3D(pC, z));
        drawLine(Point3D(pC, z), Point3D(pA, z));
    };

    constexpr uint32_t numTests = 1000;
    for (int testIdx = 0; testIdx < numTests; ++testIdx) {
        vdb_frame();

        drawGrid();

        const float scale = u01(rng);
        const float aspect = 2 * u01(rng) - 1;
        const Point2D minP(u01(rng), u01(rng));
        Vector2D d = Vector2D(scale, scale * std::fabs(aspect));
        if (aspect < 0.0f)
            d = d.yx();
        const Point2D maxP = minP + d;

        setColor(RGB(1.0f, 1.0f, 1.0f));
        drawRect(minP, maxP, 0.01f);

        const float triScale = 2 * u01(rng);
        const Point2D triOffset = 0.5f * (minP + maxP) + Vector2D(2 * u01(rng) - 1, 2 * u01(rng) - 1);

        const Point2D tcA = triScale * Point2D(u01(rng) - 0.5f, u01(rng) - 0.5f) + triOffset;
        const Point2D tcB = triScale * Point2D(u01(rng) - 0.5f, u01(rng) - 0.5f) + triOffset;
        const Point2D tcC = triScale * Point2D(u01(rng) - 0.5f, u01(rng) - 0.5f) + triOffset;
        const bool tcFlipped = cross(tcB - tcA, tcC - tcA) < 0;

        setColor(0.01f, 0.01f, 0.01f);
        drawTriangle(tcA, tcB, tcC, 0.01f);

        const Vector2D texTriEdgeNormals[] = {
            Vector2D(tcB.y - tcA.y, tcA.x - tcB.x),
            Vector2D(tcC.y - tcB.y, tcB.x - tcC.x),
            Vector2D(tcA.y - tcC.y, tcC.x - tcA.x),
        };
        const Point2D texTriAabbMinP = min(tcA, min(tcB, tcC));
        const Point2D texTriAabbMaxP = max(tcA, max(tcB, tcC));

        const TriangleSquareIntersection2DResult isectResult =
            testTriangleRectangleIntersection2D(
                tcA, tcB, tcC, tcFlipped, texTriEdgeNormals, texTriAabbMinP, texTriAabbMaxP,
                0.5f * (minP + maxP), 0.5f * d);
        if (isectResult == TriangleSquareIntersection2DResult::SquareInsideTriangle)
            setColor(1.0f, 0.5f, 0.0f);
        else if (isectResult == TriangleSquareIntersection2DResult::SquareOutsideTriangle)
            setColor(0.25f, 0.25f, 0.25f);
        else
            setColor(0.0f, 1.0f, 1.0f);
        drawTriangle(tcA, tcB, tcC, 0.02f);

        hpprintf("");
    }
}

#endif
