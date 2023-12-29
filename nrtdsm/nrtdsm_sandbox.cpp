/*

JP: このソースコードは作者が色々試すためだけのもの。
    ここの関数は一切サンプルでは使われない。
EN: This source code is just a sand box, where the author try different things.
    Functions here are not used in the sample at all.

*/

#include "nrtdsm_shared.h"
#include "../common/common_host.h"

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
    if constexpr (showNegativeShell) {
        for (int i = 1; i <= 10; ++i) {
            const float p = -static_cast<float>(i) / 10;
            setColor(RGB(-p));
            drawWiredDottedTriangle(test.SA(p), test.SB(p), test.SC(p));
        }
    }

    // World-space Ray
    const Point3D rayOrg(0.5f, -0.5f, 1.0f);
    const Vector3D rayDir = normalize(Vector3D(-0.7f, 1.3f, -0.5f));
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
        globalOffsetForTexture + Point3D(test.tcA, 0.0f),
        globalOffsetForTexture + Point3D(test.tcB, 0.0f),
        globalOffsetForTexture + Point3D(test.tcC, 0.0f));
    setColor(RGB(0.0f, 0.5f, 1.0f));
    drawVector(globalOffsetForCanonical + Point3D(0, 0, 0), Normal3D(0, 0, 1), 1.0f);
    drawVector(globalOffsetForCanonical + Point3D(1, 0, 0), Normal3D(0, 0, 1), 1.0f);
    drawVector(globalOffsetForCanonical + Point3D(0, 1, 0), Normal3D(0, 0, 1), 1.0f);
    drawVector(globalOffsetForTexture + Point3D(test.tcA, 0), Normal3D(0, 0, 1), 1.0f);
    drawVector(globalOffsetForTexture + Point3D(test.tcB, 0), Normal3D(0, 0, 1), 1.0f);
    drawVector(globalOffsetForTexture + Point3D(test.tcC, 0), Normal3D(0, 0, 1), 1.0f);
    for (int i = 1; i <= 10; ++i) {
        const float p = static_cast<float>(i) / 10;
        setColor(RGB(p));
        drawWiredDottedTriangle(
            globalOffsetForCanonical + Point3D(0, 0, 0) + p * Normal3D(0, 0, 1),
            globalOffsetForCanonical + Point3D(1, 0, 0) + p * Normal3D(0, 0, 1),
            globalOffsetForCanonical + Point3D(0, 1, 0) + p * Normal3D(0, 0, 1));
        setColor(RGB(p));
        drawWiredDottedTriangle(
            globalOffsetForTexture + Point3D(test.tcA, 0) + p * Normal3D(0, 0, 1),
            globalOffsetForTexture + Point3D(test.tcB, 0) + p * Normal3D(0, 0, 1),
            globalOffsetForTexture + Point3D(test.tcC, 0) + p * Normal3D(0, 0, 1));
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
                globalOffsetForTexture + Point3D(test.tcA, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForTexture + Point3D(test.tcB, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForTexture + Point3D(test.tcC, 0) + p * Normal3D(0, 0, 1));
        }
    }

    Vector3D e0, e1;
    rayDir.makeCoordinateSystem(&e0, &e1);

    float alpha2, alpha1, alpha0;
    float beta2, beta1, beta0;
    float denom2, denom1, denom0;
    computeCanonicalSpaceRayCoeffs(
        rayOrg, rayDir, e0, e1,
        test.pA, test.pB, test.pC,
        test.nA, test.nB, test.nC,
        &alpha2, &alpha1, &alpha0,
        &beta2, &beta1, &beta0,
        &denom2, &denom1, &denom0);

    const Point2D tc2 =
        (denom2 - alpha2 - beta2) * test.tcA
        + alpha2 * test.tcB
        + beta2 * test.tcC;
    const Point2D tc1 =
        (denom1 - alpha1 - beta1) * test.tcA
        + alpha1 * test.tcB
        + beta1 * test.tcC;
    const Point2D tc0 =
        (denom0 - alpha0 - beta0) * test.tcA
        + alpha0 * test.tcB
        + beta0 * test.tcC;

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
            test.pA, test.pB, test.pC,
            test.nA, test.nB, test.nC,
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
    const Point3D &mpAInTc, const Point3D &mpBInTc, const Point3D &mpCInTc,
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const Vector3D &e0, const Vector3D &e1,
    const Point2D &tc2, const Point2D &tc1, const Point2D &tc0,
    const float denom2, const float denom1, const float denom0,
    Point3D* hitPointInCan, Point3D* hitPointInTc, float* hitDist, Normal3D* hitNormalInObj) {
    // JP: テクスチャー空間中のマイクロ三角形を含む平面の方程式の係数を求める。
    const Normal3D nInTc(normalize(cross(mpBInTc - mpAInTc, mpCInTc - mpAInTc)));
    const float KInTc = -dot(nInTc, static_cast<Vector3D>(mpAInTc));

    // JP: 正準空間中のマイクロ三角形を含む平面の方程式の係数を求める。
    const Normal3D nInCan(
        nInTc.x * (tcB.x - tcA.x) + nInTc.y * (tcB.y - tcA.y),
        nInTc.x * (tcC.x - tcA.x) + nInTc.y * (tcC.y - tcA.y),
        nInTc.z);
    const float KInCan = nInTc.x * tcA.x + nInTc.y * tcA.y + KInTc;
    const float minHeight = std::fmin(std::fmin(mpAInTc.z, mpBInTc.z), mpCInTc.z);
    const float maxHeight = std::fmax(std::fmax(mpAInTc.z, mpBInTc.z), mpCInTc.z);

    // JP: テクスチャー空間中のレイとマイクロ三角形を含む平面の交差判定。
    float hs[3];
    uint32_t numRoots;
    {
        const float coeffs[] = {
            nInTc.x * tc0.x + nInTc.y * tc0.y + KInTc * denom0,
            nInTc.x * tc1.x + nInTc.y * tc1.y + nInTc.z * denom0 + KInTc * denom1,
            nInTc.x * tc2.x + nInTc.y * tc2.y + nInTc.z * denom1 + KInTc * denom2,
            nInTc.z * denom2
        };
        numRoots = solveCubicEquationAnalytical(coeffs, minHeight, maxHeight, hs);
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
        const Point3D hpInTc((1 - alpha - beta) * tcA + alpha * tcB + beta * tcC, h);

        // JP: 上で求まったα, βはベース三角形における重心座標に過ぎない。
        //     求めた交点がマイクロ三角形内にあるかチェックする必要がある。
        {
            const Vector2D eAB = mpBInTc.xy() - mpAInTc.xy();
            const Vector2D eBC = mpCInTc.xy() - mpBInTc.xy();
            const Vector2D eCA = mpAInTc.xy() - mpCInTc.xy();
            const Vector2D eAP = hpInTc.xy() - mpAInTc.xy();
            const Vector2D eBP = hpInTc.xy() - mpBInTc.xy();
            const Vector2D eCP = hpInTc.xy() - mpCInTc.xy();
            const float cAB = cross(eAB, eAP);
            const float cBC = cross(eBC, eBP);
            const float cCA = cross(eCA, eCP);
            if ((cAB < 0 || cBC < 0 || cCA < 0) && (cAB >= 0 || cBC >= 0 || cCA >= 0))
                continue;
        }

        const float dist = dot(
            rayDir,
            (1 - alpha - beta) * SAh + alpha * SBh + beta * SCh - rayOrg);
        if (dist > distMin && dist < *hitDist) {
            *hitDist = dist;
            *hitPointInCan = hpInCan;
            *hitPointInTc = hpInTc;
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
    const Point3D mpAInTc(test.tc(mAlphaA, mBetaA), mhA);
    const Point3D mpBInTc(test.tc(mAlphaB, mBetaB), mhB);
    const Point3D mpCInTc(test.tc(mAlphaC, mBetaC), mhC);

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

        Vector3D e0, e1;
        rayDir.makeCoordinateSystem(&e0, &e1);

        float alpha2, alpha1, alpha0;
        float beta2, beta1, beta0;
        float denom2, denom1, denom0;
        computeCanonicalSpaceRayCoeffs(
            rayOrg, rayDir, e0, e1,
            test.pA, test.pB, test.pC,
            test.nA, test.nB, test.nC,
            &alpha2, &alpha1, &alpha0,
            &beta2, &beta1, &beta0,
            &denom2, &denom1, &denom0);

        const auto computeTcCoeffs = []
        (const Point2D &tcA, const Point2D &tcB, const Point2D &tcC,
         const float denom, const float alpha, const float beta) {
            return (denom - alpha - beta) * tcA + alpha * tcB + beta * tcC;
        };

        const Point2D tc2 = computeTcCoeffs(test.tcA, test.tcB, test.tcC, denom2, alpha2, beta2);
        const Point2D tc1 = computeTcCoeffs(test.tcA, test.tcB, test.tcC, denom1, alpha1, beta1);
        const Point2D tc0 = computeTcCoeffs(test.tcA, test.tcB, test.tcC, denom0, alpha0, beta0);

        Point3D hitPointInCan;
        Point3D hitPointInTc;
        float hitDist;
        Normal3D hitNormalInObj;
        const bool hit = testNonlinearRayVsMicroTriangle(
            test.pA, test.pB, test.pC,
            test.nA, test.nB, test.nC,
            test.tcA, test.tcB, test.tcC,
            mpAInTc, mpBInTc, mpCInTc,
            rayOrg, rayDir, 0, rayLength,
            e0, e1,
            tc2, tc1, tc0,
            denom2, denom1, denom0,
            &hitPointInCan, &hitPointInTc, &hitDist, &hitNormalInObj);

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
        if constexpr (showNegativeShell) {
            for (int i = 1; i <= 10; ++i) {
                const float p = -static_cast<float>(i) / 10;
                setColor(RGB(-p));
                drawWiredDottedTriangle(test.SA(p), test.SB(p), test.SC(p));
            }
        }

        // World-space Ray
        setColor(RGB(1.0f));
        drawCross(rayOrg, 0.05f);
        drawVector(rayOrg, rayDir, rayLength);
        if (hit) {
            setColor(RGB(1, 0.5f, 0));
            const Point3D hpA = test.S(hitPointInCan.x, hitPointInCan.y, hitPointInCan.z);
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
            prevSAToB = test.S(mAlphaAToB, mBetaAToB, mhAToB);

            const float mAlphaBToC = lerp(mAlphaB, mAlphaC, p);
            const float mBetaBToC = lerp(mBetaB, mBetaC, p);
            const float mhBToC = lerp(mhB, mhC, p);
            prevSBToC = test.S(mAlphaBToC, mBetaBToC, mhBToC);

            const float mAlphaCToA = lerp(mAlphaC, mAlphaA, p);
            const float mBetaCToA = lerp(mBetaC, mBetaA, p);
            const float mhCToA = lerp(mhC, mhA, p);
            prevSCToA = test.S(mAlphaCToA, mBetaCToA, mhCToA);

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
            const Point3D SAToB = test.S(mAlphaAToB, mBetaAToB, mhAToB);
            drawLine(prevSAToB, SAToB);
            prevSAToB = SAToB;

            const float mAlphaBToC = lerp(mAlphaB, mAlphaC, p);
            const float mBetaBToC = lerp(mBetaB, mBetaC, p);
            const float mhBToC = lerp(mhB, mhC, p);
            const Point3D SBToC = test.S(mAlphaBToC, mBetaBToC, mhBToC);
            drawLine(prevSBToC, SBToC);
            prevSBToC = SBToC;

            const float mAlphaCToA = lerp(mAlphaC, mAlphaA, p);
            const float mBetaCToA = lerp(mBetaC, mBetaA, p);
            const float mhCToA = lerp(mhC, mhA, p);
            const Point3D SCToA = test.S(mAlphaCToA, mBetaCToA, mhCToA);
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
            globalOffsetForTexture + Point3D(test.tcA, 0.0f),
            globalOffsetForTexture + Point3D(test.tcB, 0.0f),
            globalOffsetForTexture + Point3D(test.tcC, 0.0f));
        setColor(RGB(0.0f, 0.5f, 1.0f));
        drawVector(globalOffsetForCanonical + Point3D(0, 0, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForCanonical + Point3D(1, 0, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForCanonical + Point3D(0, 1, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForTexture + Point3D(test.tcA, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForTexture + Point3D(test.tcB, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForTexture + Point3D(test.tcC, 0), Normal3D(0, 0, 1), 1.0f);
        for (int i = 1; i <= 10; ++i) {
            const float p = static_cast<float>(i) / 10;
            setColor(RGB(p));
            drawWiredDottedTriangle(
                globalOffsetForCanonical + Point3D(0, 0, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForCanonical + Point3D(1, 0, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForCanonical + Point3D(0, 1, 0) + p * Normal3D(0, 0, 1));
            setColor(RGB(p));
            drawWiredDottedTriangle(
                globalOffsetForTexture + Point3D(test.tcA, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForTexture + Point3D(test.tcB, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForTexture + Point3D(test.tcC, 0) + p * Normal3D(0, 0, 1));
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
                    globalOffsetForTexture + Point3D(test.tcA, 0) + p * Normal3D(0, 0, 1),
                    globalOffsetForTexture + Point3D(test.tcB, 0) + p * Normal3D(0, 0, 1),
                    globalOffsetForTexture + Point3D(test.tcC, 0) + p * Normal3D(0, 0, 1));
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
                test.pA, test.pB, test.pC,
                test.nA, test.nB, test.nC,
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
                globalOffsetForTexture + mpAInTc,
                globalOffsetForTexture + mpBInTc,
                globalOffsetForTexture + mpCInTc);
        }

        if (hit) {
            setColor(RGB(1, 0.5f, 0));
            drawCross(globalOffsetForCanonical + hitPointInCan, 0.05f);
            drawCross(globalOffsetForTexture + hitPointInTc, 0.05f);
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

        constexpr Vector3D globalOffsetForTexture(0.0f, -2.0f, 0);
        drawAxes(axisScale, globalOffsetForTexture);

        // Texture-space Shell
        setColor(RGB(0.25f));
        drawWiredTriangle(
            globalOffsetForTexture + Point3D(test.tcA, 0.0f),
            globalOffsetForTexture + Point3D(test.tcB, 0.0f),
            globalOffsetForTexture + Point3D(test.tcC, 0.0f));
        setColor(RGB(0.0f, 0.5f, 1.0f));
        drawVector(globalOffsetForTexture + Point3D(test.tcA, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForTexture + Point3D(test.tcB, 0), Normal3D(0, 0, 1), 1.0f);
        drawVector(globalOffsetForTexture + Point3D(test.tcC, 0), Normal3D(0, 0, 1), 1.0f);
        for (int i = 1; i <= 10; ++i) {
            const float p = static_cast<float>(i) / 10;
            setColor(RGB(p));
            drawWiredDottedTriangle(
                globalOffsetForTexture + Point3D(test.tcA, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForTexture + Point3D(test.tcB, 0) + p * Normal3D(0, 0, 1),
                globalOffsetForTexture + Point3D(test.tcC, 0) + p * Normal3D(0, 0, 1));
        }

        // World-space AABB
        const Matrix2x2 invA = invert(Matrix2x2(test.tcB - test.tcA, test.tcC - test.tcA));
        const auto computePInWorldSpace = [&]
        (const float u, const float v, const float h) {
            const Vector2D bc = invA * (Point2D(u, v) - test.tcA);
            return test.S(bc.x, bc.y, h);
        };

        Point3D prevAabbPs[12];
        setColor(RGB(1));
        for (int i = 0; i <= 100; ++i) {
            const float t = static_cast<float>(i) / 100;
            Point3D ps[12];
            // x
            ps[0] = computePInWorldSpace(
                lerp(test.aabb.minP.x, test.aabb.maxP.x, t), test.aabb.minP.y, test.aabb.minP.z);
            ps[1] = computePInWorldSpace(
                lerp(test.aabb.minP.x, test.aabb.maxP.x, t), test.aabb.maxP.y, test.aabb.minP.z);
            ps[2] = computePInWorldSpace(
                lerp(test.aabb.minP.x, test.aabb.maxP.x, t), test.aabb.minP.y, test.aabb.maxP.z);
            ps[3] = computePInWorldSpace(
                lerp(test.aabb.minP.x, test.aabb.maxP.x, t), test.aabb.maxP.y, test.aabb.maxP.z);
            // y
            ps[4] = computePInWorldSpace(
                test.aabb.minP.x, lerp(test.aabb.minP.y, test.aabb.maxP.y, t), test.aabb.minP.z);
            ps[5] = computePInWorldSpace(
                test.aabb.maxP.x, lerp(test.aabb.minP.y, test.aabb.maxP.y, t), test.aabb.minP.z);
            ps[6] = computePInWorldSpace(
                test.aabb.minP.x, lerp(test.aabb.minP.y, test.aabb.maxP.y, t), test.aabb.maxP.z);
            ps[7] = computePInWorldSpace(
                test.aabb.maxP.x, lerp(test.aabb.minP.y, test.aabb.maxP.y, t), test.aabb.maxP.z);
            // z
            ps[8] = computePInWorldSpace(
                test.aabb.minP.x, test.aabb.minP.y, lerp(test.aabb.minP.z, test.aabb.maxP.z, t));
            ps[9] = computePInWorldSpace(
                test.aabb.maxP.x, test.aabb.minP.y, lerp(test.aabb.minP.z, test.aabb.maxP.z, t));
            ps[10] = computePInWorldSpace(
                test.aabb.minP.x, test.aabb.maxP.y, lerp(test.aabb.minP.z, test.aabb.maxP.z, t));
            ps[11] = computePInWorldSpace(
                test.aabb.maxP.x, test.aabb.maxP.y, lerp(test.aabb.minP.z, test.aabb.maxP.z, t));
            if (i > 0) {
                for (int j = 0; j < 12; ++j)
                    drawLine(prevAabbPs[j], ps[j]);
            }
            for (int j = 0; j < 12; ++j)
                prevAabbPs[j] = ps[j];
        }

        // Texture-space AABB
        setColor(RGB(1));
        drawAabb(AABB(globalOffsetForTexture + test.aabb.minP, globalOffsetForTexture + test.aabb.maxP));

        Vector3D e0, e1;
        rayDir.makeCoordinateSystem(&e0, &e1);

        float alpha2, alpha1, alpha0;
        float beta2, beta1, beta0;
        float denom2, denom1, denom0;
        computeCanonicalSpaceRayCoeffs(
            rayOrg, rayDir, e0, e1,
            test.pA, test.pB, test.pC,
            test.nA, test.nB, test.nC,
            &alpha2, &alpha1, &alpha0,
            &beta2, &beta1, &beta0,
            &denom2, &denom1, &denom0);

        const auto computeTcCoeffs = []
        (const Point2D &tcA, const Point2D &tcB, const Point2D &tcC,
         const float denom, const float alpha, const float beta) {
            return (denom - alpha - beta) * tcA + alpha * tcB + beta * tcC;
        };

        const Point2D tc2 = computeTcCoeffs(test.tcA, test.tcB, test.tcC, denom2, alpha2, beta2);
        const Point2D tc1 = computeTcCoeffs(test.tcA, test.tcB, test.tcC, denom1, alpha1, beta1);
        const Point2D tc0 = computeTcCoeffs(test.tcA, test.tcB, test.tcC, denom0, alpha0, beta0);

        // Texture-space Ray
        std::vector<float> heightValues;
        std::vector<int32_t> indices;
        std::vector<Point3D> texPs;
        int32_t heightIdx = 0;
        for (int i = 0; i <= 500; ++i) {
            const float t = static_cast<float>(i) / 500;
            float hs[3];
            findHeight(
                test.pA, test.pB, test.pC,
                test.nA, test.nB, test.nC,
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
                test.pA, test.pB, test.pC,
                test.nA, test.nB, test.nC,
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
        compute_h_v(test.aabb.minP.x, hs_uMin, vs_uMin);
        float hs_uMax[2], vs_uMax[2];
        compute_h_v(test.aabb.maxP.x, hs_uMax, vs_uMax);

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
        compute_h_u(test.aabb.minP.y, hs_vMin, us_vMin);
        float hs_vMax[2], us_vMax[2];
        compute_h_u(test.aabb.maxP.y, hs_vMax, us_vMax);

        float hitDistMin, hitDistMax;
        const bool hit = testNonlinearRayVsAabb(
            test.pA, test.pB, test.pC, test.nA, test.nB, test.nC,
            test.aabb,
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
                    test.pA, test.pB, test.pC,
                    test.nA, test.nB, test.nC,
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
                    test.pA, test.pB, test.pC,
                    test.nA, test.nB, test.nC,
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

#endif