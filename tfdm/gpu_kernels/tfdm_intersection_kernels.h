#pragma once

#include "../tfdm_shared.h"

using namespace shared;

#define DEBUG_TRAVERSAL 0

CUDA_DEVICE_FUNCTION CUDA_INLINE bool isDebugPixel() {
    return optixGetLaunchIndex().x == 960 && optixGetLaunchIndex().y == 540;
    //return isCursorPixel();
}



CUDA_DEVICE_KERNEL void RT_IS_NAME(aabb)() {
    const auto sbtr = HitGroupSBTRecordData::get();
    const GeometryInstanceDataForTFDM &tfdm = plp.s->geomInstTfdmDataBuffer[sbtr.geomInstSlot];

    const AABB &aabb = tfdm.aabbBuffer[optixGetPrimitiveIndex()];
    float u, v;
    bool isFrontHit;
    const float t = aabb.intersect(
        Point3D(optixGetObjectRayOrigin()),
        Vector3D(optixGetObjectRayDirection()),
        optixGetRayTmin(), optixGetRayTmax(),
        &u, &v, &isFrontHit);
    if (!stc::isfinite(t))
        return;

    AABBAttributeSignature::reportIntersection(
        t,
        isFrontHit ? CustomHitKind_AABBFrontFace : CustomHitKind_AABBBackFace,
        u, v);
}



template <bool outputTravStats, LocalIntersectionType intersectionType>
CUDA_DEVICE_FUNCTION CUDA_INLINE void displacedSurface_generic(TraversalStats* travStats) {
    const auto sbtr = HitGroupSBTRecordData::get();
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];
    const GeometryInstanceDataForTFDM &tfdmGeomInst = plp.s->geomInstTfdmDataBuffer[sbtr.geomInstSlot];

    const uint32_t primIdx = optixGetPrimitiveIndex();

    const Triangle &tri = geomInst.triangleBuffer[primIdx];
    const Vertex &vA = geomInst.vertexBuffer[tri.index0];
    const Vertex &vB = geomInst.vertexBuffer[tri.index1];
    const Vertex &vC = geomInst.vertexBuffer[tri.index2];

    const DisplacementParameters &dispParams = tfdmGeomInst.params;
    const float baseHeight = dispParams.hOffset - dispParams.hScale * dispParams.hBias;
    const float heightScale = dispParams.hScale;

    const Matrix3x3 &texXfm = dispParams.textureTransform;
    const Point2D tcA = texXfm * vA.texCoord;
    const Point2D tcB = texXfm * vB.texCoord;
    const Point2D tcC = texXfm * vC.texCoord;
    const float triAreaInTc = cross(tcB - tcA, tcC - tcA)/* * 0.5f*/;
    const bool tcFlipped = triAreaInTc < 0;
    const float recTriAreaInTc = 1.0f / triAreaInTc;

    const Vector2D texTriEdgeNormals[] = {
        Vector2D(tcB.y - tcA.y, tcA.x - tcB.x),
        Vector2D(tcC.y - tcB.y, tcB.x - tcC.x),
        Vector2D(tcA.y - tcC.y, tcC.x - tcA.x),
    };
    const Point2D texTriAabbMinP = min(tcA, min(tcB, tcC));
    const Point2D texTriAabbMaxP = max(tcA, max(tcB, tcC));

    // JP: 位置や法線などをテクスチャー座標の関数として表すための行列を用意する。
    //     ここでのテクスチャー座標とはテクスチャートランスフォーム前のオリジナルのテクスチャー座標。
    // EN: Prepare matrices to express a position and a normal and so on as functions of the texture coordinate.
    //     Texure coordinate here is the original one before applying the texture transform.
    const TFDMTriangleAuxInfo &dispTriAuxInfo = tfdmGeomInst.dispTriAuxInfoBuffer[primIdx];
    const Matrix3x3 invTexXfm = invert(texXfm);
    const Matrix3x3 matTcToBc = dispTriAuxInfo.matTcToBc * invTexXfm;
    const Matrix3x3 matTcToPInObj =
        Matrix3x3(vA.position, vB.position, vC.position) * matTcToBc;
    const Matrix3x3 matTcToNInObj = dispTriAuxInfo.matTcToNInObj * invTexXfm;

    // JP: オブジェクト空間から(テクスチャートランスフォーム後の)UVに沿った接空間に変換する行列。
    // EN: Matrix to convert from the object space to the (post-texture transform) uv-aligned tangent space.
    const Matrix4x4 matObjToTcTang =
        Matrix4x4(Matrix3x3(texXfm[0], texXfm[1], Vector3D(0, 0, 1)), Vector3D(texXfm[2].xy()))
        * dispTriAuxInfo.matObjToTcTang;

    Normal3D hitNormal;
    float hitBcB, hitBcC;
    float tMax = optixGetRayTmax();
    const float tMin = optixGetRayTmin();

    // JP: レイと変位させたサーフェスの交叉判定はUVに沿った接空間で考える。
    // EN: Test ray vs displace surface intersection in the uv-aligned tangent space.
    // TODO?: Can we test bilinear patch in tangent space as well?
    const Vector3D rayDirInObj = Vector3D(optixGetObjectRayDirection());
    const Point3D rayOrgInTcTang = matObjToTcTang * Point3D(optixGetObjectRayOrigin());
    const Vector3D rayDirInTcTang = matObjToTcTang * rayDirInObj;
    const bool signX = rayDirInTcTang.x < 0;
    const bool signY = rayDirInTcTang.y < 0;
    Vector3D d1, d2;
    if constexpr (intersectionType == LocalIntersectionType::Bilinear ||
                  intersectionType == LocalIntersectionType::BSpline) {
        normalize(rayDirInObj).makeCoordinateSystem(&d1, &d2);
    }
    else {
        (void)d1;
        (void)d2;
    }

    const int32_t maxDepth =
        prevPowOf2Exponent(max(tfdmGeomInst.heightMapSize.x, tfdmGeomInst.heightMapSize.y));
    const int32_t targetMipLevel = dispParams.targetMipLevel;

    Texel roots[4];
    uint32_t numRoots;
    findRoots(texTriAabbMinP, texTriAabbMaxP, maxDepth, targetMipLevel, roots, &numRoots);
#if DEBUG_TRAVERSAL
    if (isDebugPixel() && getDebugPrintEnabled()) {
        printf(
            "%u-%u: TriAABB: (%g, %g) - (%g, %g), %u roots, signs: %c, %c\n",
            plp.f->frameIndex, primIdx,
            v2print(texTriAabbMinP), v2print(texTriAabbMaxP),
            numRoots, signX ? '-' : '+', signY ? '-' : '+');
    }
#endif
    for (int rootIdx = 0; rootIdx < lengthof(roots); ++rootIdx) {
        if (rootIdx >= numRoots)
            break;
        Texel curTexel = roots[rootIdx];
        Texel endTexel = curTexel;
        const int16_t initialLod = curTexel.lod;
        next(endTexel, signX, signY, initialLod);
#if DEBUG_TRAVERSAL
        if (isDebugPixel() && getDebugPrintEnabled()) {
            printf(
                "%u-%u, Root %u: [%d - %d, %d] - [%d - %d, %d]\n",
                plp.f->frameIndex, primIdx, rootIdx,
                curTexel.lod, curTexel.x, curTexel.y,
                endTexel.lod, endTexel.x, endTexel.y);
        }
#endif
        while (curTexel != endTexel) {
            const int2 imgSize = make_int2(1 << max(maxDepth - curTexel.lod, 0));
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
                        plp.f->frameIndex, primIdx, rootIdx,
                        curTexel.lod, curTexel.x, curTexel.y);
                }
#endif
                next(curTexel, signX, signY, initialLod);
                continue;
            }

            // JP: 接空間でテクセルがつくるAABBをアフィン演算を用いて計算。
            // EN: Compute the AABB of texel in the tangent space using affine arithmetic.
            AABB texelAabb;
            {
                if constexpr (outputTravStats)
                    ++travStats->numAabbTests;
                const int2 wrapIndex = make_int2(floorDiv(curTexel.x, imgSize.x), floorDiv(curTexel.y, imgSize.y));
                const uint2 wrappedTexel = curTexel.lod <= maxDepth ?
                    make_uint2(curTexel.x - wrapIndex.x * imgSize.x, curTexel.y - wrapIndex.y * imgSize.y) :
                    make_uint2(0, 0);
                const float2 minmax = tfdmGeomInst.minMaxMipMap[min(curTexel.lod, maxDepth)].read(wrappedTexel);
                const float amplitude = heightScale * (minmax.y - minmax.x);
                const float minHeight = baseHeight + heightScale * minmax.x;
                const AAFloatOn2D hBound(minHeight + 0.5f * amplitude, 0, 0, 0.5f * amplitude);

                const Point2D clippedTcMinP = max(texelCenter - Vector2D(0.5f) * texelScale, texTriAabbMinP);
                const Point2D clippedTcMaxP = min(texelCenter + Vector2D(0.5f) * texelScale, texTriAabbMaxP);
                const Vector2D clippedTcDim = clippedTcMaxP - clippedTcMinP;

                const AAFloatOn2D_Vector3D edge0(
                    Vector3D(0.0f), Vector3D(0.5f * clippedTcDim.x, 0, 0), Vector3D(0.0f), Vector3D(0.0f));
                const AAFloatOn2D_Vector3D edge1(
                    Vector3D(0.0f), Vector3D(0.0f), Vector3D(0, 0.5f * clippedTcDim.y, 0), Vector3D(0.0f));
                const AAFloatOn2D_Point3D texCoord =
                    Point3D(clippedTcMinP + 0.5f * clippedTcDim, 1.0f) + (edge0 + edge1);

                const AAFloatOn2D_Point3D pBoundInTcTang(texCoord.x, texCoord.y, AAFloatOn2D(0.0f));
                AAFloatOn2D_Vector3D nBoundInObj = static_cast<AAFloatOn2D_Vector3D>(matTcToNInObj * texCoord);
                nBoundInObj.normalize();
                const AAFloatOn2D_Vector3D nBoundInTcTang = matObjToTcTang.getUpperLeftMatrix() * nBoundInObj;
                const AAFloatOn2D_Point3D boundsInTcTang = pBoundInTcTang + hBound * nBoundInTcTang;

                const auto iaSx = boundsInTcTang.x.toIAFloat();
                const auto iaSy = boundsInTcTang.y.toIAFloat();
                const auto iaSz = boundsInTcTang.z.toIAFloat();
                texelAabb.minP = Point3D(iaSx.lo(), iaSy.lo(), iaSz.lo());
                texelAabb.maxP = Point3D(iaSx.hi(), iaSy.hi(), iaSz.hi());
            }

            // JP: レイがAABBにヒットしない場合はテクセル内のサーフェスともヒットしないため深掘りしない。
            // EN: Don't descend more when the ray does not hit the AABB since
            //     the ray will never hit the surface inside the texel
            if (!texelAabb.intersect(rayOrgInTcTang, rayDirInTcTang, tMin, tMax)) {
#if DEBUG_TRAVERSAL
                if (isDebugPixel() && getDebugPrintEnabled()) {
                    printf(
                        "%u-%u, Root %u: [%d - %d, %d] Miss AABB\n",
                        plp.f->frameIndex, primIdx, rootIdx,
                        curTexel.lod, curTexel.x, curTexel.y);
                }
#endif
                next(curTexel, signX, signY, initialLod);
                continue;
            }

            // JP: レイがAABBにヒットしているがターゲットのMIPレベルに到達していないときは下位MIPに下る。
            // EN: Descend to the lower mip when the ray hit the AABB but does not reach the target mip level.
            if (curTexel.lod > targetMipLevel) {
#if DEBUG_TRAVERSAL
                if (isDebugPixel() && getDebugPrintEnabled()) {
                    printf(
                        "%u-%u, Root %u: [%d - %d, %d] Hit AABB, down\n",
                        plp.f->frameIndex, primIdx, rootIdx,
                        curTexel.lod, curTexel.x, curTexel.y);
                }
#endif
                down(curTexel, signX, signY);
                continue;
            }

#if DEBUG_TRAVERSAL
            if (isDebugPixel() && getDebugPrintEnabled()) {
                printf(
                    "%u-%u, Root %u: [%d - %d, %d] Hit, AABB, intersect\n",
                    plp.f->frameIndex, primIdx, rootIdx,
                    curTexel.lod, curTexel.x, curTexel.y);
            }
#endif

            if constexpr (outputTravStats)
                ++travStats->numLeafTests;

            const auto sample = [&](float px, float py) {
                // No need to explicitly consider texture wrapping since the sampler is responsible for it.
                return tex2DLod<float>(tfdmGeomInst.heightMap, px / imgSize.x, py / imgSize.y, curTexel.lod);
            };

            // JP: レイと変位を加えたサーフェスとの交叉判定を行う。
            // EN: Test ray intersection against the displaced surface.
            if constexpr (intersectionType == LocalIntersectionType::Box) {
                (void)imgSize;
                (void)sample;

                float param0, param1;
                bool isF;
                const float t = texelAabb.intersect(
                    rayOrgInTcTang, rayDirInTcTang, tMin, tMax, &param0, &param1, &isF);
                if (t < tMax) {
                    Normal3D n;
                    const Point2D hp(texelAabb.restoreHitPoint(param0, param1, &n));
                    const float bcB = cross(tcC - hp, tcA - hp) * recTriAreaInTc;
                    const float bcC = cross(tcA - hp, tcB - hp) * recTriAreaInTc;
                    tMax = t;
                    hitBcB = bcB;
                    hitBcC = bcC;
                    hitNormal = static_cast<Normal3D>(n);
                }
            }
            if constexpr (intersectionType == LocalIntersectionType::TwoTriangle) {
                const float cornerHeightTL =
                    baseHeight + heightScale * sample(curTexel.x - 0.0f, curTexel.y - 0.0f);
                const float cornerHeightTR =
                    baseHeight + heightScale * sample(curTexel.x + 1.0f, curTexel.y - 0.0f);
                const float cornerHeightBL =
                    baseHeight + heightScale * sample(curTexel.x - 0.0f, curTexel.y + 1.0f);
                const float cornerHeightBR =
                    baseHeight + heightScale * sample(curTexel.x + 1.0f, curTexel.y + 1.0f);

                const Point2D tcTL(texelCenter + texelScale * Vector2D(-0.5f, -0.5f));
                const Point2D tcTR(texelCenter + texelScale * Vector2D(0.5f, -0.5f));
                const Point2D tcBL(texelCenter + texelScale * Vector2D(-0.5f, 0.5f));
                const Point2D tcBR(texelCenter + texelScale * Vector2D(0.5f, 0.5f));

                // JP: 法線はオブジェクト空間で正規化する。
                // EN: Normalize normal vectors in the object space.
                const Vector3D nTLInObj = normalize(matTcToNInObj * Vector3D(tcTL, 1.0f));
                const Vector3D nTRInObj = normalize(matTcToNInObj * Vector3D(tcTR, 1.0f));
                const Vector3D nBLInObj = normalize(matTcToNInObj * Vector3D(tcBL, 1.0f));
                const Vector3D nBRInObj = normalize(matTcToNInObj * Vector3D(tcBR, 1.0f));

                // JP: テクセルコーナーにおける高さと法線を使って四隅の座標を接空間で求める。
                // EN: Compute the coordinates of four corners in the tangent space using
                //     the height values at the corners and the normals.
                const Matrix3x3 matObjToTcTang3x3 = matObjToTcTang.getUpperLeftMatrix();
                const Point3D pTL = Point3D(tcTL, 0.0f) + cornerHeightTL * (matObjToTcTang3x3 * nTLInObj);
                const Point3D pTR = Point3D(tcTR, 0.0f) + cornerHeightTR * (matObjToTcTang3x3 * nTRInObj);
                const Point3D pBL = Point3D(tcBL, 0.0f) + cornerHeightBL * (matObjToTcTang3x3 * nBLInObj);
                const Point3D pBR = Point3D(tcBR, 0.0f) + cornerHeightBR * (matObjToTcTang3x3 * nBRInObj);

                const auto testRayVsTriangleIntersection = []
                (const Point3D &org, const Vector3D &dir, const float distMin, const float distMax,
                 const Point3D &pA, const Point3D &pB, const Point3D &pC,
                 Vector3D* const n, float* const t, float* const bcB, float* const bcC) {
                    const Vector3D eAB = pB - pA;
                    const Vector3D eCA = pA - pC;
                    *n = cross(eCA, eAB);

                    const Vector3D e2 = (1.0f / dot(*n, dir)) * (pA - org);
                    const Vector3D i = cross(dir, e2);

                    *bcB = dot(i, eCA);
                    *bcC = dot(i, eAB);
                    *t = dot(*n, e2);

                    return (
                        (*t < distMax) & (*t > distMin)
                        & (*bcB >= 0.0f) & (*bcC >= 0.0f) & (*bcB + *bcC <= 1));
                };

                float t = INFINITY;
                float mbcB, mbcC;
                Vector3D n;
                if (testRayVsTriangleIntersection(
                    rayOrgInTcTang, rayDirInTcTang, tMin, tMax, pTL, pTR, pBR, &n, &t, &mbcB, &mbcC)) {
                    if (t < tMax) {
                        const Point2D hp((1 - (mbcB + mbcC)) * tcTL + mbcB * tcTR + mbcC * tcBR);
                        const float bcB = cross(tcC - hp, tcA - hp) * recTriAreaInTc;
                        const float bcC = cross(tcA - hp, tcB - hp) * recTriAreaInTc;
                        if (bcB >= 0.0f && bcC >= 0.0f && bcB + bcC <= 1.0f) {
                            tMax = t;
                            hitBcB = bcB;
                            hitBcC = bcC;
                            hitNormal = static_cast<Normal3D>(n);
                        }
                    }
                }
                if (testRayVsTriangleIntersection(
                    rayOrgInTcTang, rayDirInTcTang, tMin, tMax, pTL, pBR, pBL, &n, &t, &mbcB, &mbcC)) {
                    if (t < tMax) {
                        const Point2D hp((1 - (mbcB + mbcC)) * tcTL + mbcB * tcBR + mbcC * tcBL);
                        const float bcB = cross(tcC - hp, tcA - hp) * recTriAreaInTc;
                        const float bcC = cross(tcA - hp, tcB - hp) * recTriAreaInTc;
                        if (bcB >= 0.0f && bcC >= 0.0f && bcB + bcC <= 1.0f) {
                            tMax = t;
                            hitBcB = bcB;
                            hitBcC = bcC;
                            hitNormal = static_cast<Normal3D>(n);
                        }
                    }
                }
            }
            if constexpr (intersectionType == LocalIntersectionType::Bilinear) {
                const float cornerHeightTL =
                    baseHeight + heightScale * sample(curTexel.x - 0.0f, curTexel.y - 0.0f);
                const float cornerHeightTR =
                    baseHeight + heightScale * sample(curTexel.x + 1.0f, curTexel.y - 0.0f);
                const float cornerHeightBL =
                    baseHeight + heightScale * sample(curTexel.x - 0.0f, curTexel.y + 1.0f);
                const float cornerHeightBR =
                    baseHeight + heightScale * sample(curTexel.x + 1.0f, curTexel.y + 1.0f);

#if DEBUG_TRAVERSAL
                if (isDebugPixel() && getDebugPrintEnabled()) {
                    printf(
                        "%u-%u: %u-%u-%u\n",
                        plp.f->frameIndex, primIdx,
                        curTexel.lod, curTexel.x, curTexel.y);
                    printf(
                        "%u-%u: v0 (%g, %g, %g, %g, %g, %g, %g, %g)\n",
                        plp.f->frameIndex, primIdx,
                        v3print(vA.position), v3print(vA.normal), v2print(vA.texCoord));
                    printf(
                        "%u-%u: v1 (%g, %g, %g, %g, %g, %g, %g, %g)\n",
                        plp.f->frameIndex, primIdx,
                        v3print(vB.position), v3print(vB.normal), v2print(vB.texCoord));
                    printf(
                        "%u-%u: v2 (%g, %g, %g, %g, %g, %g, %g, %g)\n",
                        plp.f->frameIndex, primIdx,
                        v3print(vC.position), v3print(vC.normal), v2print(vC.texCoord));

                    printf(
                        "%u-%u: Height: %g, %g, %g, %g\n",
                        plp.f->frameIndex, primIdx,
                        cornerHeightTL, cornerHeightTR, cornerHeightBL, cornerHeightBR);
                    printf(
                        "%u-%u: org: (%g, %g, %g), dir: (%g, %g, %g)\n",
                        plp.f->frameIndex, primIdx,
                        v3print(optixGetObjectRayOrigin()), v3print(rayDirInObj));
                }
#endif

                // JP: ニュートン法を使ってレイとバイリニアパッチとの交叉判定を行う。
                // EN: Test ray vs bilinear patch intersection using the Newton method.
                const auto testRayVsBilinearPatchIntersection = [&]
                (const Point2D &avgTc,
                 const Matrix3x3 &matTcToP, const Matrix3x3 &matTcToN, const Matrix3x3 &matTcToBc,
                 const Point3D &rayOrg, const Vector3D &rayDir,
                 float* hitDist, float* bcB, float* bcC, Normal3D* hitNormal) {
                    const Matrix3x2 jacobP(matTcToP[0], matTcToP[1]);
                    const Matrix3x2 jacobN(matTcToN[0], matTcToN[1]);
                    Point2D curGuess = avgTc;
                    float hitDist2;
                    Matrix3x2 jacobS;
                    const Point2D hitGuessMin = texelCenter - Vector2D(0.5f, 0.5f) * texelScale;
                    const Point2D hitGuessMax = texelCenter + Vector2D(0.5f, 0.5f) * texelScale;
                    float prevErrDist2 = INFINITY;
                    uint8_t errDistStreak = 0;
                    uint8_t invalidRegionStreak = 0;
                    uint8_t behindStreak = 0;
                    uint32_t itr = 0;
                    constexpr uint32_t numIterations = 10;
                    for (; itr < numIterations; ++itr) {
                        Normal3D n(matTcToN * Point3D(curGuess, 1.0f));
                        const float nLength = n.length();
                        n /= nLength;

                        const float ut = imgSize.x * curGuess.x - curTexel.x;
                        const float vt = imgSize.y * curGuess.y - curTexel.y;
                        const float h =
                            (1 - ut) * (1 - vt) * cornerHeightTL
                            + ut * (1 - vt) * cornerHeightTR
                            + (1 - ut) * vt * cornerHeightBL
                            + ut * vt * cornerHeightBR;

                        const Point3D S = matTcToP * Point3D(curGuess, 1.0f) + h * n;
                        const Vector3D delta = S - rayOrg;
                        const Vector2D F(dot(delta, d1), dot(delta, d2));
                        const float errDist2 = F.sqLength();
                        const float dotDirDelta = dot(rayDir, delta);
                        errDistStreak = errDist2 > prevErrDist2 ? (errDistStreak + 1) : 0;
                        behindStreak = dotDirDelta < 0 ? (behindStreak + 1) : 0;
                        if (errDistStreak >= 2 || behindStreak >= 2) {
                            *hitDist = INFINITY;
                            return false;
                        }
                        prevErrDist2 = errDist2;
                        hitDist2 = sqDistance(S, rayOrg);

                        const float jacobHu = imgSize.x *
                            (-(1 - vt) * cornerHeightTL + (1 - vt) * cornerHeightTR
                             - vt * cornerHeightBL + vt * cornerHeightBR);
                        const float jacobHv = imgSize.y *
                            (-(1 - ut) * cornerHeightTL - ut * cornerHeightTR
                             + (1 - ut) * cornerHeightBL + ut * cornerHeightBR);

                        jacobS =
                            jacobP + Matrix3x2(jacobHu * n, jacobHv * n)
                            + (h / nLength) * (jacobN - Matrix3x2(dot(jacobN[0], n) * n, dot(jacobN[1], n) * n));

                        if (errDist2 < pow2(1e-5f)) {
                            Point3D bc(1 - *bcB - *bcC, *bcB, *bcC);
                            if (bc[0] < 0.0f || bc[1] < 0.0f || bc[2] < 0.0f
                                || bc[0] > 1.0f || bc[1] > 1.0f || bc[2] > 1.0f
                                || dotDirDelta < 0) {
                                *hitDist = INFINITY;
                                return false;
                            }
                            *hitDist = std::sqrt(hitDist2/* / rayDir.sqLength()*/);
                            *hitNormal = static_cast<Normal3D>(normalize(cross(jacobS[1], jacobS[0])));
#if DEBUG_TRAVERSAL
                            if (isDebugPixel() && getDebugPrintEnabled()) {
                                printf(
                                    "%u-%u-%u: guess: (%g, %g), dist: %g, S: (%g, %g, %g), n: (%g, %g, %g)\n",
                                    plp.f->frameIndex, primIdx, itr,
                                    curGuess.x, curGuess.y, *hitDist,
                                    v3print(S), v3print(*hitNormal));
                            }
#endif
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
                            *bcB = bc[1];
                            *bcC = bc[2];
                        }
                    }

                    return false;
                };

                Normal3D n;
                float t;
                float bcB, bcC;
                if (testRayVsBilinearPatchIntersection(
                    texelCenter,
                    matTcToPInObj, matTcToNInObj, matTcToBc,
                    Point3D(optixGetObjectRayOrigin()), rayDirInObj,
                    &t, &bcB, &bcC, &n)) {
                    if (t < tMax) {
                        tMax = t;
                        hitBcB = bcB;
                        hitBcC = bcC;
                        hitNormal = n;
                    }
                }
            }
            if constexpr (intersectionType == LocalIntersectionType::BSpline) {
                Assert_NotImplemented();
            }

            next(curTexel, signX, signY, initialLod);
        }
    }

    if (tMax == optixGetRayTmax())
        return;

    DisplacedSurfaceAttributes attr = {};
    if constexpr (intersectionType == LocalIntersectionType::Box ||
                  intersectionType == LocalIntersectionType::TwoTriangle) {
        /*
        JP: 接空間で求めた法線を面との直交性を保ちつつオブジェクト空間に変換する。
            必要となる行列は、法線以外のベクトルを接空間からオブジェクト空間へと変換する行列
            (の左上3x3)の逆行列の転置である。逆行列はすでに持っているので転置のみで済む。
        EN: Transform the normal computed in the tangent space into the object space while preserving
            orthogonality to the surface.
            The required matrix is the transpose of the inverse of (the upper left 3x3 of) a matrix
            to transform vectors other than the normal from the tangent space into the object space.
            We already have the inverse matrix, so just transpose it.
        */
        attr.normalInObj = normalize(transpose(matObjToTcTang.getUpperLeftMatrix()) * hitNormal);
    }
    else {
        attr.normalInObj = hitNormal;
    }
    const uint8_t hitKind = dot(rayDirInTcTang, hitNormal) <= 0 ?
        CustomHitKind_DisplacedSurfaceFrontFace :
        CustomHitKind_DisplacedSurfaceBackFace;
    DisplacedSurfaceAttributeSignature::reportIntersection(tMax, hitKind, hitBcB, hitBcC, attr);
}
