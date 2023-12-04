#include "tfdm_shared.h"
#include "../common/common_host.h"

#if ENABLE_VDB

struct Texel {
    int16_t x;
    int16_t y;
    int16_t lod;

    bool operator==(const Texel &r) const {
        return x == r.x && y == r.y && lod == r.lod;
    }
    bool operator!=(const Texel &r) const {
        return x != r.x || y != r.y || lod != r.lod;
    }
};

static void down(Texel &texel) {
    --texel.lod;
    texel.x *= 2;
    texel.y *= 2;
}

static void up(Texel &texel) {
    ++texel.lod;
    texel.x = floorDiv(texel.x, 2);
    texel.y = floorDiv(texel.y, 2);
    //texel.x /= 2;
    //texel.y /= 2;
}

static void next(Texel &texel, int32_t maxDepth) {
    while (texel.lod <= maxDepth) {
        switch (2 * floorMod(texel.x, 2) + floorMod(texel.y, 2)) {
        //switch (2 * (texel.x % 2) + texel.y % 2) {
        case 1:
            --texel.y;
            ++texel.x;
            return;
        case 3:
            up(texel);
            break;
        default:
            ++texel.y;
            return;
        }
    }
}

enum class TriangleSquareIntersection2DResult {
    SquareOutsideTriangle = 0,
    SquareInsideTriangle,
    SquareOverlappingTriangle
};

static TriangleSquareIntersection2DResult testTriangleSquareIntersection2D(
    const Point2D triPs[3], bool tcFlipped, const Vector2D triEdgeNormals[3],
    const Point2D &triAabbMinP, const Point2D &triAabbMaxP,
    const Point2D &squareCenter, float squareHalfWidth) {
    const Vector2D vSquareCenter = static_cast<Vector2D>(squareCenter);
    const Point2D relTriPs[] = {
        triPs[0] - vSquareCenter,
        triPs[1] - vSquareCenter,
        triPs[2] - vSquareCenter,
    };

    // JP: テクセルのAABBと三角形のAABBのIntersectionを計算する。
    // EN: Test intersection between the texel AABB and the triangle AABB.
    if (any(min(Point2D(squareHalfWidth), triAabbMaxP - vSquareCenter) <=
            max(Point2D(-squareHalfWidth), triAabbMinP - vSquareCenter)))
        return TriangleSquareIntersection2DResult::SquareOutsideTriangle;

    // JP: いずれかの三角形のエッジの法線方向にテクセルがあるならテクセルは三角形の外にある。
    // EN: Texel is outside of the triangle if the texel is in the normal direction of any edge.
    for (int eIdx = 0; eIdx < 3; ++eIdx) {
        Vector2D eNormal = (tcFlipped ? -1 : 1) * triEdgeNormals[eIdx];
        Bool2D b = eNormal >= Vector2D(0.0f);
        Vector2D e = static_cast<Vector2D>(relTriPs[eIdx]) +
            Vector2D((b.x ? 1 : -1) * squareHalfWidth,
                     (b.y ? 1 : -1) * squareHalfWidth);
        if (dot(eNormal, e) <= 0)
            return TriangleSquareIntersection2DResult::SquareOutsideTriangle;
    }

    // JP: テクセルが三角形のエッジとかぶっているかどうかを調べる。
    // EN: Test if the texel is overlapping with some edges of the triangle.
    for (int i = 0; i < 4; ++i) {
        Point2D corner(
            (i % 2 ? -1 : 1) * squareHalfWidth,
            (i / 2 ? -1 : 1) * squareHalfWidth);
        for (int eIdx = 0; eIdx < 3; ++eIdx) {
            const Point2D &o = relTriPs[eIdx];
            const Vector2D &e1 = relTriPs[(eIdx + 1) % 3] - o;
            Vector2D e2 = corner - o;
            if ((tcFlipped ? -1 : 1) * cross(e1, e2) < 0)
                return TriangleSquareIntersection2DResult::SquareOverlappingTriangle;
        }
    }

    // JP: それ以外の場合はテクセルは三角形に囲まれている。
    // EN: Otherwise, the texel is encompassed by the triangle.
    return TriangleSquareIntersection2DResult::SquareInsideTriangle;
}

static void findRoots(
    const Point2D &triAabbMinP, const Point2D &triAabbMaxP, const int32_t maxDepth, uint32_t targetMipLevel,
    Texel* const roots, uint32_t* const numRoots) {
    using namespace shared;
    static_assert(useMultipleRootOptimization, "Naive method is not implemented.");
    const Vector2D d = triAabbMaxP - triAabbMinP;
    const uint32_t largerDim = d.y > d.x;
    int32_t startMipLevel = maxDepth - prevPowOf2Exponent(static_cast<uint32_t>(1.0f / d[largerDim])) - 1;
    startMipLevel = std::max(startMipLevel, 0);
    while (true) {
        const float res = std::pow(2.0f, static_cast<float>(maxDepth - startMipLevel));
        const int32_t minTexelX = static_cast<int32_t>(std::floor(res * triAabbMinP.x));
        const int32_t minTexelY = static_cast<int32_t>(std::floor(res * triAabbMinP.y));
        const int32_t maxTexelX = static_cast<int32_t>(std::floor(res * triAabbMaxP.x));
        const int32_t maxTexelY = static_cast<int32_t>(std::floor(res * triAabbMaxP.y));
        if ((maxTexelX - minTexelX) < 2 && (maxTexelY - minTexelY) < 2 &&
            startMipLevel >= targetMipLevel) {
            *numRoots = 0;
            for (int y = minTexelY; y <= maxTexelY; ++y) {
                for (int x = minTexelX; x <= maxTexelX; ++x) {
                    Texel &root = roots[(*numRoots)++];
                    root.x = x;
                    root.y = y;
                    root.lod = startMipLevel;
                }
            }
            break;
        }
        ++startMipLevel;
    }
}



void testFindRoots() {
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

    const auto drawTexel = [&maxLevel, &drawRect](const Texel &texel, float z) {
        const float scale = std::pow(2.0f, static_cast<float>(texel.lod - maxLevel));
        const Point2D minP(texel.x * scale, texel.y * scale);
        const Point2D maxP((texel.x + 1) * scale, (texel.y + 1) * scale);
        drawRect(minP, maxP, z);
    };

    {
        vdb_frame();

        drawGrid();

        //const Point2D minP(0.0f, 0.0f - 0.01f);
        //const Point2D maxP(1.0f, 1.0f - 0.01f);
        //const Point2D minP(0.5f, 0.5f);
        //const Point2D maxP(2.5f, 2.5f);
        //const Point2D minP(-0.935419, 0.0215215);
        //const Point2D maxP(-0.311806, 0.499992);
        //const Point2D minP(-0.961285, -0.0215919);
        //const Point2D maxP(-0.320428, 0.448254);
        //const Point2D minP(-0.0972371, 0.333274);
        //const Point2D maxP(0.354056, 0.999822);
        //const Point2D minP(-0.0986062, 0.33348);
        //const Point2D maxP(0.35231, 1.00044);
        //const Point2D minP(-0.2f, -0.3f);
        //const Point2D maxP(0.4f, 0.2f);
        //const Point2D minP(-1.2f, -1.3f);
        //const Point2D maxP(-0.6f, -0.8f);
        //const Point2D minP(-0.2f, -0.3f);
        //const Point2D maxP(1.4f, 1.2f);
        const Point2D minP(-1.00002, -1.75675e-05);
        const Point2D maxP(0, 0.999982);

        setColor(RGB(0.0f, 1.0f, 1.0f));
        drawRect(minP, maxP, 0.005f);

        uint32_t numRoots;
        Texel roots[4];
        findRoots(minP, maxP, maxLevel, maxLevel - 4, roots, &numRoots);

        setColor(RGB(1.0f, 0.5f, 0.0f));
        for (int i = 0; i < numRoots; ++i) {
            Texel endTexel = roots[i];
            next(endTexel, maxLevel);
            drawTexel(roots[i], 0.0075f);
        }

        hpprintf("");
    }

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

        const auto drawRect = [](const Point2D &minP, const Point2D &maxP, float z) {
            drawLine(Point3D(minP.x, minP.y, z), Point3D(maxP.x, minP.y, z));
            drawLine(Point3D(minP.x, maxP.y, z), Point3D(maxP.x, maxP.y, z));
            drawLine(Point3D(minP.x, minP.y, z), Point3D(minP.x, maxP.y, z));
            drawLine(Point3D(maxP.x, minP.y, z), Point3D(maxP.x, maxP.y, z));
        };

        setColor(RGB(1.0f, 1.0f, 1.0f));
        drawRect(minP, maxP, 0.01f);

        uint32_t numRoots;
        Texel roots[4];
        findRoots(minP, maxP, maxLevel, maxLevel - 4, roots, &numRoots);

        setColor(RGB(1.0f, 0.5f, 0.0f));
        for (int i = 0; i < numRoots; ++i)
            drawTexel(roots[i], 0.01f);

        hpprintf("");
    }

    exit(0);
}



void testNewtonMethod() {
    static bool drawIntermediatePoints = false;

    {
        struct TestData {
            Point3D p0;
            Point3D p1;
            Point3D p2;
            Normal3D n0;
            Normal3D n1;
            Normal3D n2;
            Point2D tc0;
            Point2D tc1;
            Point2D tc2;
            Texel texel;
            float heightUL;
            float heightUR;
            float heightBL;
            float heightBR;
        };

        constexpr int2 imgSize(4, 4);
        const float texelScale = 1.0f / imgSize.x;
        std::vector<TestData> testData = {
            //TestData{
            //    Point3D(-0.5, 0, -0.5), Point3D(0.5, 0, 0.5), Point3D(0.5, 0, -0.5),
            //    Normal3D(0, 1, 0), Normal3D(0, 1, 0), Normal3D(0, 1, 0),
            //    Point2D(0, 0), Point2D(1, 1), Point2D(1, 0),
            //    Texel{0, 1}, 0.262661, 0.271313, 0.249683, 0.249081
            //},
            //TestData{
            //    Point3D(-0.5, 0, -0.5), Point3D(-0.5, 0, 0.5), Point3D(0.5, 0, 0.5),
            //    Normal3D(0, 1, 0), Normal3D(0, 1, 0), Normal3D(0, 1, 0),
            //    Point2D(0, 0), Point2D(0, 1), Point2D(1, 1),
            //    Texel{4, 6}, 0.258854, 0.293248, 0.257206, 0.277455
            //}
            TestData{
                Point3D(0, 0.212132, 0.212132), Point3D(0, -1.31134e-08, 0.3), Point3D(0.212132, -1.31134e-08, 0.212132),
                Normal3D(0, 0.707107, 0.707107), Normal3D(0, -4.37114e-08, 1), Normal3D(0.707107, -4.37114e-08, 0.707107),
                Point2D(0, 0.25), Point2D(0, 0.5), Point2D(0.125, 0.5),
                Texel{0, 1}, 0.262661, 0.271313, 0.249683, 0.249081
            },
        };

        vdb_frame();

        constexpr float axisScale = 1;
        setColor(RGB(1, 0, 0));
        drawLine(Point3D(0, 0, 0), Point3D(axisScale, 0, 0));
        setColor(RGB(0, 1, 0));
        drawLine(Point3D(0, 0, 0), Point3D(0, axisScale, 0));
        setColor(RGB(0, 0, 1));
        drawLine(Point3D(0, 0, 0), Point3D(0, 0, axisScale));

        for (uint32_t testIdx = 0; testIdx < testData.size(); ++testIdx) {
            const TestData &test = testData[testIdx];
            Texel curTexel = test.texel;

            setColor(RGB(0.1f, 0.1f, 0.1f));
            drawLine(test.p0, test.p1);
            drawLine(test.p1, test.p2);
            drawLine(test.p2, test.p0);

            setColor(RGB(0, 1, 1));
            drawVector(test.p0, test.n0, 0.5f);
            drawVector(test.p1, test.n1, 0.5f);
            drawVector(test.p2, test.n2, 0.5f);

            const Matrix3x3 matBcToTc =
                Matrix3x3(Point3D(test.tc0, 1.0f), Point3D(test.tc1, 1.0f), Point3D(test.tc2, 1.0f));
            const Matrix3x3 matTcToBc = invert(matBcToTc);
            const Matrix3x3 matBcToPInObj = Matrix3x3(test.p0, test.p1, test.p2);
            const Matrix3x3 matTcToPInObj = matBcToPInObj * matTcToBc;
            const Matrix3x3 matBcToNInObj = Matrix3x3(test.n0, test.n1, test.n2);
            const Matrix3x3 matTcToNInObj = matBcToNInObj * matTcToBc;

            setColor(RGB(0.5f, 0.5f, 0.5f));
            for (int32_t ib = 0; ib < 100; ++ib) {
                float tA = static_cast<float>(ib) / 100;
                float tB = static_cast<float>(ib + 1) / 100;

                const auto draw = [&]
                (const Point3D &bcA, const Point3D &bcB) {
                    Point3D pA = matBcToPInObj * bcA;
                    Point3D pB = matBcToPInObj * bcB;
                    Normal3D nA = Normal3D(matBcToNInObj * bcA);
                    Normal3D nB = Normal3D(matBcToNInObj * bcB);
                    nA.normalize();
                    nB.normalize();
                    Point3D tcA = matBcToTc * bcA;
                    Point3D tcB = matBcToTc * bcB;
                    float utA = imgSize.x * tcA.x - curTexel.x;
                    float vtA = imgSize.y * tcA.y - curTexel.y;
                    float utB = imgSize.x * tcB.x - curTexel.x;
                    float vtB = imgSize.y * tcB.y - curTexel.y;
                    float hA =
                        (1 - utA) * (1 - vtA) * test.heightUL
                        + utA * (1 - vtA) * test.heightUR
                        + (1 - utA) * vtA * test.heightBL
                        + utA * vtA * test.heightBR;
                    float hB =
                        (1 - utB) * (1 - vtB) * test.heightUL
                        + utB * (1 - vtB) * test.heightUR
                        + (1 - utB) * vtB * test.heightBL
                        + utB * vtB * test.heightBR;
                    drawLine(pA + hA * nA, pB + hB * nB);
                };

                draw(Point3D(1 - tA, tA, 0.0f), Point3D(1 - tB, tB, 0.0f));
                draw(Point3D(0.0f, 1 - tA, tA), Point3D(0.0f, 1 - tB, tB));
                draw(Point3D(tA, 0.0f, 1 - tA), Point3D(tB, 0.0f, 1 - tB));
            }

            for (int32_t itc0 = 0; itc0 <= 100; ++itc0) {
                const float tc0 = static_cast<float>(itc0) / 100;
                for (int32_t itc1 = 0; itc1 <= 100; ++itc1) {
                    const float tc1 = static_cast<float>(itc1) / 100;
                    Point3D tc3D(tc0, tc1, 1.0f);

                    const auto process = [&]
                    (const Matrix3x3 &matTcToBc, const Matrix3x3 &matTcToPInObj, const Matrix3x3 &matTcToNInObj) {
                        const Point3D bc = matTcToBc * tc3D;
                        const bool isInValidRange =
                            bc[0] >= 0.0f && bc[1] >= 0.0f && bc[2] >= 0.0f &&
                            bc[0] <= 1.0f && bc[1] <= 1.0f && bc[2] <= 1.0f &&
                            (tc3D.x * imgSize.x) >= curTexel.x && (tc3D.x * imgSize.x) < curTexel.x + 1 &&
                            (tc3D.y * imgSize.y) >= curTexel.y && (tc3D.y * imgSize.y) < curTexel.y + 1;
                        float ut = imgSize.x * tc3D.x - curTexel.x;
                        float vt = imgSize.y * tc3D.y - curTexel.y;

                        const Point3D pBaseInObj = matTcToPInObj * tc3D;
                        Normal3D nInObj(matTcToNInObj * tc3D);
                        nInObj.normalize();
                        const float h =
                            (1 - ut) * (1 - vt) * test.heightUL
                            + ut * (1 - vt) * test.heightUR
                            + (1 - ut) * vt * test.heightBL
                            + ut * vt * test.heightBR;

                        if (!isInValidRange) {
                            setColor(0.02f * RGB(0.1f, 0.1f, 0.1f));
                            drawPoint(pBaseInObj);
                            setColor(0.02f * RGB(0.5f, 0.5f, 0.5f));
                            drawPoint(pBaseInObj + h * nInObj);
                        }
                    };
                    process(matTcToBc, matTcToPInObj, matTcToNInObj);
                }
            }

            for (int32_t itc0 = 0; itc0 <= 100; ++itc0) {
                const float tc0 = (test.texel.x + static_cast<float>(itc0) / 100) * texelScale;
                for (int32_t itc1 = 0; itc1 <= 100; ++itc1) {
                    const float tc1 = (test.texel.y + static_cast<float>(itc1) / 100) * texelScale;
                    Point3D tc3D(tc0, tc1, 1.0f);

                    const auto process = [&]
                    (const Matrix3x3 &matTcToBc, const Matrix3x3 &matTcToPInObj, const Matrix3x3 &matTcToNInObj) {
                        const Point3D bc = matTcToBc * tc3D;
                        const bool isInValidRange =
                            bc[0] >= 0.0f && bc[1] >= 0.0f && bc[2] >= 0.0f &&
                            bc[0] <= 1.0f && bc[1] <= 1.0f && bc[2] <= 1.0f &&
                            (tc3D.x * imgSize.x) >= curTexel.x && (tc3D.x * imgSize.x) < curTexel.x + 1 &&
                            (tc3D.y * imgSize.y) >= curTexel.y && (tc3D.y * imgSize.y) < curTexel.y + 1;
                        if (!isInValidRange)
                            return;
                        float ut = imgSize.x * tc3D.x - curTexel.x;
                        float vt = imgSize.y * tc3D.y - curTexel.y;

                        const Point3D pBaseInObj = matTcToPInObj * tc3D;
                        Normal3D nInObj(matTcToNInObj * tc3D);
                        nInObj.normalize();
                        const float h =
                            (1 - ut) * (1 - vt) * test.heightUL
                            + ut * (1 - vt) * test.heightUR
                            + (1 - ut) * vt * test.heightBL
                            + ut * vt * test.heightBR;

                        setColor(RGB(0.1f, 0.1f, 0.1f));
                        drawPoint(pBaseInObj);
                        setColor(RGB(0.5f, 0.5f, 0.5f));
                        drawPoint(pBaseInObj + h * nInObj);
                    };
                    process(matTcToBc, matTcToPInObj, matTcToNInObj);
                }
            }

            hpprintf("");
        }

        const auto testRayVsBilinearPatchIntersection = [&]
        (const TestData &test, const Point2D &avgTc,
         const Matrix3x3 &matTcToP, const Matrix3x3 &matTcToN, const Matrix3x3 &matTcToBc,
         const Point3D &rayOrg, const Vector3D &rayDir, const Vector3D &d1, const Vector3D &d2,
         float* hitDist, float* b1, float* b2, Normal3D* hitNormal) {
            Texel curTexel = test.texel;
            const Point2D texelCenter = Point2D(curTexel.x + 0.5f, curTexel.y + 0.5f) * texelScale;
            const float cornerHeightUL = test.heightUL;
            const float cornerHeightUR = test.heightUR;
            const float cornerHeightBL = test.heightBL;
            const float cornerHeightBR = test.heightBR;

            const Matrix3x2 jacobP(matTcToP[0], matTcToP[1]);
            const Matrix3x2 jacobN(matTcToN[0], matTcToN[1]);
            Point2D curGuess = avgTc;
            float hitDist2;
            Matrix3x2 jacobS;
            const Point2D hitGuessMin = texelCenter - Vector2D(0.5f, 0.5f) * texelScale;
            const Point2D hitGuessMax = texelCenter + Vector2D(0.5f, 0.5f) * texelScale;
            float prevErrDist2 = INFINITY;
            uint32_t errDistStreak = 0;
            uint32_t invalidRegionStreak = 0;
            uint32_t itr = 0;
            constexpr uint32_t numIterations = 10;
            for (; itr < numIterations; ++itr) {
                Normal3D n(matTcToN * Point3D(curGuess, 1.0f));
                const float nLength = n.length();
                n /= nLength;

                const float ut = imgSize.x * curGuess.x - curTexel.x;
                const float vt = imgSize.y * curGuess.y - curTexel.y;
                const float h =
                    (1 - ut) * (1 - vt) * cornerHeightUL
                    + ut * (1 - vt) * cornerHeightUR
                    + (1 - ut) * vt * cornerHeightBL
                    + ut * vt * cornerHeightBR;

                const Point3D S = matTcToP * Point3D(curGuess, 1.0f) + h * n;
                const Vector3D delta = S - rayOrg;
                const Vector2D F(dot(delta, d1), dot(delta, d2));
                const float errDist2 = F.sqLength();
                if (errDist2 > prevErrDist2)
                    ++errDistStreak;
                else
                    errDistStreak = 0;
                if (errDistStreak >= 2) {
                    *hitDist = INFINITY;
                    return false;
                }
                prevErrDist2 = errDist2;
                hitDist2 = sqDistance(S, rayOrg);

                const float jacobHu = imgSize.x *
                    (-(1 - vt) * cornerHeightUL + (1 - vt) * cornerHeightUR
                     - vt * cornerHeightBL + vt * cornerHeightBR);
                const float jacobHv = imgSize.y *
                    (-(1 - ut) * cornerHeightUL - ut * cornerHeightUR
                     + (1 - ut) * cornerHeightBL + ut * cornerHeightBR);

                jacobS =
                    jacobP + Matrix3x2(jacobHu * n, jacobHv * n)
                    + (h / nLength) * (jacobN - Matrix3x2(dot(jacobN[0], n) * n, dot(jacobN[1], n) * n));

                if (errDist2 < pow2(1e-5f)) {
                    Point3D bc(1 - *b1 - *b2, *b1, *b2);
                    if (bc[0] < 0.0f || bc[1] < 0.0f || bc[2] < 0.0f
                        || bc[0] > 1.0f || bc[1] > 1.0f || bc[2] > 1.0f) {
                        *hitDist = INFINITY;
                        return false;
                    }
                    *hitDist = std::sqrt(hitDist2/* / rayDir.sqLength()*/);
                    *hitNormal = static_cast<Normal3D>(normalize(cross(jacobS[1], jacobS[0])));
                    return true;
                }

                if (itr + 1 < numIterations) {
                    const Matrix2x2 jacobF(
                        Vector2D(dot(d1, jacobS[0]), dot(d2, jacobS[0])),
                        Vector2D(dot(d1, jacobS[1]), dot(d2, jacobS[1])));
                    const Matrix2x2 invJacobF = invert(jacobF);
                    const Vector2D deltaGuess = invJacobF * F;
                    curGuess -= deltaGuess;

                    Point3D bc = matTcToBc * Point3D(curGuess, 1.0f);
                    if (any(curGuess < hitGuessMin) || any(curGuess > hitGuessMax)
                        || bc[0] < 0.0f || bc[1] < 0.0f || bc[2] < 0.0f
                        || bc[0] > 1.0f || bc[1] > 1.0f || bc[2] > 1.0f) {
                        ++invalidRegionStreak;
                        if (invalidRegionStreak >= 3) {
                            *hitDist = INFINITY;
                            return false;
                        }
                        curGuess = min(max(curGuess, hitGuessMin), hitGuessMax);
                        bc = matTcToBc * Point3D(curGuess, 1.0f);
                    }
                    else {
                        invalidRegionStreak = 0;
                    }
                    *b1 = bc[1];
                    *b2 = bc[2];
                }
            }

            return false;
        };

        {
            drawIntermediatePoints = true;

            Point3D rayOrg(0, 0.5, 1.5);
            Vector3D rayDir(0.314883, -0.234673, -0.919661);
            setColor(RGB(1.0f, 1.0f, 1.0f));
            drawCross(rayOrg, 0.05f);
            drawVector(rayOrg, rayDir, 5.0f);

            Vector3D d1, d2;
            rayDir.makeCoordinateSystem(&d1, &d2);

            const TestData &test = testData[0];
            Texel curTexel = test.texel;

            const Point2D texelCenter((curTexel.x + 0.5f) / imgSize.x, (curTexel.y + 0.5f) / imgSize.y);

            const Matrix3x3 matTcToBc = invert(
                Matrix3x3(Point3D(test.tc0, 1.0f), Point3D(test.tc1, 1.0f), Point3D(test.tc2, 1.0f)));
            const Matrix3x3 matTcToPInObj = Matrix3x3(test.p0, test.p1, test.p2) * matTcToBc;
            const Matrix3x3 matTcToNInObj = Matrix3x3(test.n0, test.n1, test.n2) * matTcToBc;

            float hitDist;
            float hitBc1, hitBc2;
            Normal3D hitNormal;
            if (testRayVsBilinearPatchIntersection(
                test, texelCenter,
                matTcToPInObj, matTcToNInObj, matTcToBc,
                rayOrg, rayDir, d1, d2,
                &hitDist, &hitBc1, &hitBc2, &hitNormal)) {
                Point3D hp = rayOrg + hitDist * rayDir;
                setColor(RGB(1.0f, 0.5f, 0.0f));
                drawPoint(hp);
                drawCross(hp, 0.005f);
                setColor(RGB(0, 1, 1));
                drawVector(hp, hitNormal, 0.5f);
            }

            hpprintf("");
        }

        //for (uint32_t ix = 0; ix <= 100; ++ix) {
        //    float px = static_cast<float>(ix) / 100;
        //    for (uint32_t iz = 0; iz <= 100; ++iz) {
        //        float pz = static_cast<float>(iz) / 100;
        //        Point3D rayOrg(-1 + 2 * px, 2, -1 + 2 * pz);
        //        Vector3D rayDir(0, -1, 0);

        //        Vector3D d1, d2;
        //        rayDir.makeCoordinateSystem(&d1, &d2);
        //        for (uint32_t testIdx = 0; testIdx < testData.size(); ++testIdx) {
        //            const TestData &test = testData[testIdx];

        //            const Point2D texelCenter((curTexel.x + 0.5f) / imgSize, (curTexel.y + 0.5f) / imgSize);

        //            const Matrix3x3 matTcToBc = invert(
        //                Matrix3x3(Point3D(test.tc0, 1.0f), Point3D(test.tc1, 1.0f), Point3D(test.tc2, 1.0f)));
        //            const Matrix3x3 matTcToPInObj = Matrix3x3(test.p0, test.p1, test.p2) * matTcToBc;
        //            const Matrix3x3 matTcToNInObj = Matrix3x3(test.n0, test.n1, test.n2) * matTcToBc;

        //            float hitDist;
        //            float hitBc1, hitBc2;
        //            Normal3D hitNormal;
        //            if (testRayVsBilinearPatchIntersection(
        //                test, texelCenter,
        //                matTcToPInObj, matTcToNInObj, matTcToBc,
        //                rayOrg, rayDir, d1, d2,
        //                &hitDist, &hitBc1, &hitBc2, &hitNormal)) {
        //                Point3D hp = rayOrg + hitDist * rayDir;
        //                setColor(RGB(1.0f, 0.5f, 0.0f));
        //                drawPoint(hp);
        //                drawCross(hp, 0.005f);
        //                //setColor(RGB(0, 1, 1));
        //                //drawVector(hp, hitNormal, 0.5f);
        //            }
        //        }
        //    }
        //}

        hpprintf("");
    }

    std::mt19937 rng(14131631);
    std::uniform_real_distribution<float> u01;

    const auto cRng = [&](std::mt19937 &rng, float radius) {
        return 2 * radius * (u01(rng) - 0.5f);
    };

    constexpr uint32_t numTests = 1000;
    for (uint32_t testIdx = 0; testIdx < numTests; ++testIdx) {
        const Point3D pUL(-1 + cRng(rng, 0.1f), 0.5f + cRng(rng, 0.5f), -1 + cRng(rng, 0.1f));
        const Point3D pUR(1 + cRng(rng, 0.1f), 0.5f + cRng(rng, 0.5f), -1 + cRng(rng, 0.1f));
        const Point3D pBL(-1 + cRng(rng, 0.1f), 0.5f + cRng(rng, 0.5f), 1 + cRng(rng, 0.1f));
        const Point3D pBR(1 + cRng(rng, 0.1f), 0.5f + cRng(rng, 0.5f), 1 + cRng(rng, 0.1f));

        const Normal3D nUL = Normal3D::fromPolarYUp(2 * pi_v<float> *u01(rng), 0.7f * u01(rng));
        const Normal3D nUR = Normal3D::fromPolarYUp(2 * pi_v<float> *u01(rng), 0.7f * u01(rng));
        const Normal3D nBL = Normal3D::fromPolarYUp(2 * pi_v<float> *u01(rng), 0.7f * u01(rng));
        const Normal3D nBR = Normal3D::fromPolarYUp(2 * pi_v<float> *u01(rng), 0.7f * u01(rng));

        const Point2D tcUL(0.2f + cRng(rng, 0.2f), 0.2f + cRng(rng, 0.2f));
        const Point2D tcUR(0.8f + cRng(rng, 0.2f), 0.2f + cRng(rng, 0.2f));
        const Point2D tcBL(0.2f + cRng(rng, 0.2f), 0.8f + cRng(rng, 0.2f));
        const Point2D tcBR(0.8f + cRng(rng, 0.2f), 0.8f + cRng(rng, 0.2f));

        const float heightUL = 2 * (0.5f + cRng(rng, 0.3f));
        const float heightUR = 2 * (0.5f + cRng(rng, 0.3f));
        const float heightBL = 2 * (0.5f + cRng(rng, 0.3f));
        const float heightBR = 2 * (0.5f + cRng(rng, 0.3f));

        const Point3D rayOrg(cRng(rng, 2.0f), 3.0f + cRng(rng, 0.5f), cRng(rng, 2.0f));
        const Point3D rayDst(cRng(rng, 1.0f), 0.0f, cRng(rng, 1.0f));
        const Vector3D rayDir = normalize(rayDst - rayOrg);

        //if (testIdx != 24 &&
        //    testIdx != 25 &&
        //    testIdx != 26 &&
        //    testIdx != 42 &&
        //    testIdx != 47 &&
        //    testIdx != 48 &&
        //    testIdx != 88 &&
        //    testIdx != 130) {
        //    continue;
        //}

        vdb_frame();

        constexpr float axisScale = 5;
        setColor(RGB(1, 0, 0));
        drawLine(Point3D(0, 0, 0), Point3D(axisScale, 0, 0));
        setColor(RGB(0, 1, 0));
        drawLine(Point3D(0, 0, 0), Point3D(0, axisScale, 0));
        setColor(RGB(0, 0, 1));
        drawLine(Point3D(0, 0, 0), Point3D(0, 0, axisScale));

        setColor(RGB(0, 1, 1));
        drawVector(pUL, nUL, 0.5f);
        drawVector(pUR, nUR, 0.5f);
        drawVector(pBL, nBL, 0.5f);
        drawVector(pBR, nBR, 0.5f);

        const Matrix3x3 matTcToBc_A =
            invert(Matrix3x3(Point3D(tcUL, 1.0f), Point3D(tcBL, 1.0f), Point3D(tcBR, 1.0f)));
        const Matrix3x3 matTcToPInObj_A = Matrix3x3(pUL, pBL, pBR) * matTcToBc_A;
        const Matrix3x3 matTcToNInObj_A = Matrix3x3(nUL, nBL, nBR) * matTcToBc_A;
        const Point2D tcAvg_A = (tcUL + tcBL + tcBR) / 3;

        const Matrix3x3 matTcToBc_B =
            invert(Matrix3x3(Point3D(tcBR, 1.0f), Point3D(tcUR, 1.0f), Point3D(tcUL, 1.0f)));
        const Matrix3x3 matTcToPInObj_B = Matrix3x3(pBR, pUR, pUL) * matTcToBc_B;
        const Matrix3x3 matTcToNInObj_B = Matrix3x3(nBR, nUR, nUL) * matTcToBc_B;
        const Point2D tcAvg_B = (tcBR + tcUR + tcUL) / 3;

        for (uint32_t itc0 = 0; itc0 <= 100; ++itc0) {
            const float tc0 = static_cast<float>(itc0) / 100;
            for (uint32_t itc1 = 0; itc1 <= 100; ++itc1) {
                const float tc1 = static_cast<float>(itc1) / 100;
                Point3D tc3D(tc0, tc1, 1.0f);

                const auto process = [&]
                (const Matrix3x3 &matTcToBc, const Matrix3x3 &matTcToPInObj, const Matrix3x3 &matTcToNInObj) {
                    const Point3D bc = matTcToBc * tc3D;
                    const bool isInValidRange =
                        bc[0] >= 0.0f && bc[1] >= 0.0f && bc[2] >= 0.0f &&
                        bc[0] <= 1.0f && bc[1] <= 1.0f && bc[2] <= 1.0f;

                    const Point3D pBaseInObj = matTcToPInObj * tc3D;
                    Normal3D nInObj(matTcToNInObj * tc3D);
                    nInObj.normalize();
                    const float h =
                        (1 - tc3D[0]) * (1 - tc3D[1]) * heightUL
                        + tc3D[0] * (1 - tc3D[1]) * heightUR
                        + (1 - tc3D[0]) * tc3D[1] * heightBL
                        + tc3D[0] * tc3D[1] * heightBR;

                    setColor((isInValidRange ? 1.0f : 0.02f) * RGB(0.1f, 0.1f, 0.1f));
                    drawPoint(pBaseInObj);
                    setColor((isInValidRange ? 1.0f : 0.02f) * RGB(0.5f, 0.5f, 0.5f));
                    drawPoint(pBaseInObj + h * nInObj);
                };
                process(matTcToBc_A, matTcToPInObj_A, matTcToNInObj_A);
                process(matTcToBc_B, matTcToPInObj_B, matTcToNInObj_B);
            }
        }

        setColor(RGB(1.0f, 1.0f, 1.0f));
        drawCross(rayOrg, 0.05f);
        drawVector(rayOrg, rayDir, 5.0f);

        Vector3D d1, d2;
        rayDir.makeCoordinateSystem(&d1, &d2);

        const auto computeHitPoint = [&]
        (const Point2D &avgTc,
         const Matrix3x3 &matTcToPInObj, const Matrix3x3 &matTcToNInObj, const Matrix3x3 &matTcToBc,
         const Point3D &rayOrg, const Vector3D &rayDir,
         float* t, float* b1, float* b2, Normal3D* hitNormal) {
            constexpr uint32_t numIterations = 10;
            const Matrix3x2 jacobP(matTcToPInObj[0], matTcToPInObj[1]);
            const Matrix3x2 jacobN(matTcToNInObj[0], matTcToNInObj[1]);
            Point2D curGuess = avgTc;
            Normal3D n;
            float nLength;
            float h;
            Vector2D F;
            float errDist2;
            float hitDist2;
            {
                n = Normal3D(matTcToNInObj * Point3D(curGuess, 1.0f));
                nLength = n.length();
                n /= nLength;

                h =
                    (1 - curGuess.x) * (1 - curGuess.y) * heightUL
                    + curGuess.x * (1 - curGuess.y) * heightUR
                    + (1 - curGuess.x) * curGuess.y * heightBL
                    + curGuess.x * curGuess.y * heightBR;

                const Point3D S = matTcToPInObj * Point3D(curGuess, 1.0f) + h * n;
                const Vector3D delta = S - rayOrg;
                F = Vector2D(dot(delta, d1), dot(delta, d2));
                errDist2 = F.sqLength();
                hitDist2 = sqDistance(S, rayOrg);

                if (drawIntermediatePoints) {
                    setColor(0.25f * RGB(1, 0.5f, 0));
                    drawPoint(S);
                    drawCross(S, 0.025f);
                }
            }
            Matrix3x2 jacobS;
            float prevErrDist2 = INFINITY;
            uint32_t itr = 0;
            uint32_t invalidBcCount = 0;
            for (; itr < numIterations; ++itr) {
                const float jacobHu =
                    -(1 - curGuess.y) * heightUL + (1 - curGuess.y) * heightUR
                    - curGuess.y * heightBL + curGuess.y * heightBR;
                const float jacobHv =
                    -(1 - curGuess.x) * heightUL - curGuess.x * heightUR
                    + (1 - curGuess.x) * heightBL + curGuess.x * heightBR;

                jacobS =
                    jacobP + Matrix3x2(jacobHu * n, jacobHv * n)
                    + (h / nLength) * (jacobN - Matrix3x2(dot(jacobN[0], n) * n, dot(jacobN[1], n) * n));

                const Point3D bc = matTcToBc * Point3D(curGuess, 1.0f);
                if (bc[0] < 0.0f || bc[1] < 0.0f || bc[2] < 0.0f
                    || bc[0] > 1.0f || bc[1] > 1.0f || bc[2] > 1.0f)
                    ++invalidBcCount;
                else
                    invalidBcCount = 0;
                if (invalidBcCount >= 2) {
                    hpprintf(
                        "Test %u terminated at %u: The root is in invalid region.\n",
                        testIdx, itr);
                    *t = INFINITY;
                    return;
                }

                if (errDist2 > prevErrDist2) {
                    hpprintf(
                        "Test %u terminated at %u: The new root is farther than the previous.\n",
                        testIdx, itr);
                    *t = INFINITY;
                    return;
                }

                *b1 = bc[1];
                *b2 = bc[2];
                if (errDist2 < pow2(1e-3f))
                    break;
                prevErrDist2 = errDist2;

                if (itr + 1 < numIterations) {
                    const Matrix2x2 jacobF(
                        Vector2D(dot(d1, jacobS[0]), dot(d2, jacobS[0])),
                        Vector2D(dot(d1, jacobS[1]), dot(d2, jacobS[1])));
                    const Matrix2x2 invJacobF = invert(jacobF);
                    const Vector2D deltaGuess = invJacobF * F;
                    //curGuess -= deltaGuess;
                    const Point2D prevGuess = curGuess;
                    float coeff = 1.0f;
                    for (int decStep = 0; decStep < 3; ++decStep) {
                        curGuess = prevGuess - coeff * deltaGuess;

                        n = Normal3D(matTcToNInObj * Point3D(curGuess, 1.0f));
                        nLength = n.length();
                        n /= nLength;

                        h =
                            (1 - curGuess.x) * (1 - curGuess.y) * heightUL
                            + curGuess.x * (1 - curGuess.y) * heightUR
                            + (1 - curGuess.x) * curGuess.y * heightBL
                            + curGuess.x * curGuess.y * heightBR;

                        const Point3D S = matTcToPInObj * Point3D(curGuess, 1.0f) + h * n;
                        const Vector3D delta = S - rayOrg;
                        F = Vector2D(dot(delta, d1), dot(delta, d2));
                        errDist2 = F.sqLength();

                        if (errDist2 < (1 - 0.25f * coeff) * prevErrDist2) {
                            if (drawIntermediatePoints) {
                                setColor(0.25f * RGB(1, 0.5f, 0));
                                drawPoint(S);
                                drawCross(S, 0.025f);
                            }
                            hitDist2 = sqDistance(S, rayOrg);
                            break;
                        }

                        coeff *= 0.5f;
                    }
                }
            }

            const Point3D bc = matTcToBc * Point3D(curGuess, 1.0f);
            if (bc[0] < 0.0f || bc[1] < 0.0f || bc[2] < 0.0f
                || bc[0] > 1.0f || bc[1] > 1.0f || bc[2] > 1.0f) {
                hpprintf(
                    "Test %u terminated at %u: The root is in invalid region.\n",
                    testIdx, itr);
                *t = INFINITY;
                return;
            }

            hpprintf("Test %u Root found at %u\n", testIdx, itr);

            *t = std::sqrt(hitDist2 / rayDir.sqLength());
            *hitNormal = static_cast<Normal3D>(normalize(cross(jacobS[1], jacobS[0])));
        };

        float t_A;
        float b1_A, b2_A;
        Normal3D hitNormal_A;
        computeHitPoint(
            tcAvg_A, matTcToPInObj_A, matTcToNInObj_A, matTcToBc_A, rayOrg, rayDir,
            &t_A, &b1_A, &b2_A, &hitNormal_A);
        if (std::isfinite(t_A)) {
            Point3D hp = rayOrg + t_A * rayDir;
            setColor(RGB(1.0f, 0.5f, 0.0f));
            drawPoint(hp);
            drawCross(hp, 0.025f);
            setColor(RGB(0, 1, 1));
            drawVector(hp, hitNormal_A, 0.5f);
        }
        float t_B;
        float b1_B, b2_B;
        Normal3D hitNormal_B;
        computeHitPoint(
            tcAvg_B, matTcToPInObj_B, matTcToNInObj_B, matTcToBc_B, rayOrg, rayDir,
            &t_B, &b1_B, &b2_B, &hitNormal_B);
        if (std::isfinite(t_B)) {
            Point3D hp = rayOrg + t_B * rayDir;
            setColor(RGB(1.0f, 0.5f, 0.0f));
            drawPoint(hp);
            drawCross(hp, 0.025f);
            setColor(RGB(0, 1, 1));
            drawVector(hp, hitNormal_B, 0.5f);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        hpprintf("");
    }

    exit(0);
}

#endif