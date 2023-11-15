#pragma once

#include "../tfdm_shared.h"

using namespace shared;



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
    if (!isfinite(t))
        return;

    AABBAttributeSignature::reportIntersection(
        t,
        isFrontHit ? CustomHitKind_AABBFrontFace : CustomHitKind_AABBBackFace,
        u, v);
}



template <LocalIntersectionType intersectionType>
CUDA_DEVICE_FUNCTION CUDA_INLINE void displacedSurface_generic() {
    bool isDebugPixel = optixGetLaunchIndex().x == 1262 && optixGetLaunchIndex().y == 878;
    //bool isDebugPixel = isCursorPixel();

    const auto sbtr = HitGroupSBTRecordData::get();
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];
    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    const Triangle &tri = geomInst.triangleBuffer[optixGetPrimitiveIndex()];
    const Vertex (&vs)[] = {
        geomInst.vertexBuffer[tri.index0],
        geomInst.vertexBuffer[tri.index1],
        geomInst.vertexBuffer[tri.index2]
    };

    const GeometryInstanceDataForTFDM &tfdm = plp.s->geomInstTfdmDataBuffer[sbtr.geomInstSlot];
    const DisplacementParameters &dispParams = tfdm.params;

    const Matrix3x3 &texXfm = dispParams.textureTransform;
    const Point2D tcs[] = {
        texXfm * vs[0].texCoord,
        texXfm * vs[1].texCoord,
        texXfm * vs[2].texCoord,
    };
    const float triAreaInTc = cross(tcs[1] - tcs[0], tcs[2] - tcs[0])/* * 0.5f*/;
    const bool tcFlipped = triAreaInTc < 0;
    const float recTriAreaInTc = 1.0f / triAreaInTc;

    const Vector2D texTriEdgeNormals[] = {
        Vector2D(tcs[1].y - tcs[0].y, tcs[0].x - tcs[1].x),
        Vector2D(tcs[2].y - tcs[1].y, tcs[1].x - tcs[2].x),
        Vector2D(tcs[0].y - tcs[2].y, tcs[2].x - tcs[0].x),
    };
    const Point2D texTriAabbMinP = min(tcs[0], min(tcs[1], tcs[2]));
    const Point2D texTriAabbMaxP = max(tcs[0], max(tcs[1], tcs[2]));

    const DisplacedTriangleAuxInfo &dispTriAuxInfo = tfdm.dispTriAuxInfoBuffer[optixGetPrimitiveIndex()];
    const Matrix3x3 invTexXfm = invert(texXfm);
    const Matrix3x3 matTcToBc = dispTriAuxInfo.matTcToBc * invTexXfm;
    const Matrix4x4 matObjToTc =
        Matrix4x4(Matrix3x3(texXfm[0], texXfm[1], Vector3D(0, 0, 1)), Vector3D(texXfm[2].xy()))
        * dispTriAuxInfo.matObjToTc;
    const Matrix3x3 matTcToPInObj =
        Matrix3x3(vs[0].position, vs[1].position, vs[2].position) * matTcToBc;
    const Matrix3x3 matTcToNInObj = dispTriAuxInfo.matTcToNInObj * invTexXfm;

    Normal3D hitNormal;
    float hitBc1, hitBc2;
    float tMax = optixGetRayTmax();
    const float tMin = optixGetRayTmin();

    // JP: レイと変位させたサーフェスの交叉判定はテクスチャー空間で考える。
    // EN: Test ray vs displace surface intersection in the texture space.
    // TODO?: Can we test bilinear patch in texture space as well?
    const Vector3D rayDirInObj = Vector3D(optixGetObjectRayDirection());
    const Point3D rayOrgInTc = matObjToTc * Point3D(optixGetObjectRayOrigin());
    const Vector3D rayDirInTc = matObjToTc * rayDirInObj;
    const bool signX = rayDirInTc.x < 0;
    const bool signY = rayDirInTc.y < 0;
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
        prevPowOf2Exponent(max(mat.heightMapSize.x, mat.heightMapSize.y));
#if USE_WORKAROUND_FOR_CUDA_BC_TEX
    const int32_t targetMipLevel = min(dispParams.targetMipLevel, maxDepth - 2);
#else
    const int32_t targetMipLevel = dispParams.targetMipLevel;
#endif

#if OUTPUT_TRAVERSAL_STATS
    uint32_t numIterations = 0;
#endif

    Texel roots[useMultipleRootOptimization ? 4 : 1];
    uint32_t numRoots;
    findRoots(texTriAabbMinP, texTriAabbMaxP, maxDepth, targetMipLevel, roots, &numRoots);
    for (int rootIdx = 0; rootIdx < lengthof(roots); ++rootIdx) {
        if (rootIdx >= numRoots)
            break;
        Texel curTexel = roots[rootIdx];
        Texel endTexel = curTexel;
        next(endTexel, signX, signY, maxDepth);
        while (curTexel != endTexel) {
#if OUTPUT_TRAVERSAL_STATS
            ++numIterations;
#endif
            const int2 imgSize = make_int2(1 << max(maxDepth - curTexel.lod, 0));
            const float texelScale = std::pow(2.0f, static_cast<float>(curTexel.lod - maxDepth));
            const Point2D texelCenter = Point2D(curTexel.x + 0.5f, curTexel.y + 0.5f) * texelScale;
            const TriangleSquareIntersection2DResult isectResult =
                testTriangleSquareIntersection2D(
                    tcs, tcFlipped, texTriEdgeNormals, texTriAabbMinP, texTriAabbMaxP,
                    texelCenter, 0.5f * texelScale);

            // JP: テクセルがベース三角形の外にある場合はテクセルをスキップ。
            // EN: Skip the texel if it is outside of the base triangle.
            if (isectResult == TriangleSquareIntersection2DResult::SquareOutsideTriangle) {
                next(curTexel, signX, signY, maxDepth);
                continue;
            }

            // JP: テクスチャー空間でテクセルがつくるAABBをアフィン演算を用いて計算。
            // EN: Compute the AABB of texel in the texture space using affine arithmetic.
            AABB texelAabb;
            {
                const int2 wrapIndex = make_int2(floorDiv(curTexel.x, imgSize.x), floorDiv(curTexel.y, imgSize.y));
                const uint2 wrappedTexel = curTexel.lod <= maxDepth ?
                    make_uint2(curTexel.x - wrapIndex.x * imgSize.x, curTexel.y - wrapIndex.y * imgSize.y) :
                    make_uint2(0, 0);
                const float2 minmax = mat.minMaxMipMap[min(curTexel.lod, maxDepth)].read(wrappedTexel);
                const float amplitude = dispParams.hScale * (minmax.y - minmax.x);
                const float minHeight = dispParams.hOffset + dispParams.hScale * (minmax.x - dispParams.hBias);
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

                const AAFloatOn2D_Point3D pBoundInTc(texCoord.x, texCoord.y, AAFloatOn2D(0.0f));
                AAFloatOn2D_Vector3D nBoundInObj = static_cast<AAFloatOn2D_Vector3D>(matTcToNInObj * texCoord);
                nBoundInObj.normalize();
                const AAFloatOn2D_Vector3D nBoundInTc = matObjToTc.getUpperLeftMatrix() * nBoundInObj;
                const AAFloatOn2D_Point3D boundsInTc = pBoundInTc + hBound * nBoundInTc;

                const auto iaSx = boundsInTc.x.toIAFloat();
                const auto iaSy = boundsInTc.y.toIAFloat();
                const auto iaSz = boundsInTc.z.toIAFloat();
                texelAabb.minP = Point3D(iaSx.lo(), iaSy.lo(), iaSz.lo());
                texelAabb.maxP = Point3D(iaSx.hi(), iaSy.hi(), iaSz.hi());
            }

            // JP: レイがAABBにヒットしない場合はテクセル内のサーフェスともヒットしないため深掘りしない。
            // EN: Don't descend more when the ray does not hit the AABB since
            //     the ray will never hit the surface inside the texel
            if (!texelAabb.intersect(rayOrgInTc, rayDirInTc, tMin, tMax)) {
                next(curTexel, signX, signY, maxDepth);
                continue;
            }

            // JP: レイがAABBにヒットしているがターゲットのMIPレベルに到達していないときは下位MIPに下る。
            // EN: Descend to the lower mip when the ray hit the AABB but does not reach the target mip level.
            if (curTexel.lod > targetMipLevel) {
                down(curTexel, signX, signY);
                continue;
            }

            const auto sample = [&](float px, float py) {
                // No need to explicitly consider texture wrapping since the sampler is responsible for it.
                return tex2DLod<float>(mat.heightMap, px / imgSize.x, py / imgSize.y, curTexel.lod);
            };

            // JP: レイと変位を加えたサーフェスとの交叉判定を行う。
            // EN: Test ray intersection against the displaced surface.
            if constexpr (intersectionType == LocalIntersectionType::Box) {
                (void)imgSize;
                (void)sample;

                float param0, param1;
                bool isF;
                const float t = texelAabb.intersect(
                    rayOrgInTc, rayDirInTc, tMin, tMax, &param0, &param1, &isF);
                if (t < tMax) {
                    Normal3D n;
                    const Point2D hp(texelAabb.restoreHitPoint(param0, param1, &n));
                    const float b1 = cross(tcs[2] - hp, tcs[0] - hp) * recTriAreaInTc;
                    const float b2 = cross(tcs[0] - hp, tcs[1] - hp) * recTriAreaInTc;
                    tMax = t;
                    hitBc1 = b1;
                    hitBc2 = b2;
                    hitNormal = static_cast<Normal3D>(n);
                }
            }
            if constexpr (intersectionType == LocalIntersectionType::TwoTriangle) {
                const float cornerHeightUL =
                    dispParams.hOffset + dispParams.hScale
                    * (sample(curTexel.x - 0.0f, curTexel.y - 0.0f) - dispParams.hBias);
                const float cornerHeightUR =
                    dispParams.hOffset + dispParams.hScale
                    * (sample(curTexel.x + 1.0f, curTexel.y - 0.0f) - dispParams.hBias);
                const float cornerHeightBL =
                    dispParams.hOffset + dispParams.hScale
                    * (sample(curTexel.x - 0.0f, curTexel.y + 1.0f) - dispParams.hBias);
                const float cornerHeightBR =
                    dispParams.hOffset + dispParams.hScale
                    * (sample(curTexel.x + 1.0f, curTexel.y + 1.0f) - dispParams.hBias);

                const Point2D tcUL(texelCenter + texelScale * Vector2D(-0.5f, -0.5f));
                const Point2D tcUR(texelCenter + texelScale * Vector2D(0.5f, -0.5f));
                const Point2D tcBL(texelCenter + texelScale * Vector2D(-0.5f, 0.5f));
                const Point2D tcBR(texelCenter + texelScale * Vector2D(0.5f, 0.5f));

                // JP: 法線はオブジェクト空間で正規化する。
                // EN: Normalize normal vectors in the object space.
                const Vector3D nULInObj = normalize(matTcToNInObj * Vector3D(tcUL, 1.0f));
                const Vector3D nURInObj = normalize(matTcToNInObj * Vector3D(tcUR, 1.0f));
                const Vector3D nBLInObj = normalize(matTcToNInObj * Vector3D(tcBL, 1.0f));
                const Vector3D nBRInObj = normalize(matTcToNInObj * Vector3D(tcBR, 1.0f));

                // JP: テクセルコーナーにおける高さと法線を使って四隅の座標をテクスチャー空間で求める。
                // EN: Compute the coordinates of four corners in the texture space using
                //     the height values at the corners and the normals.
                const Matrix3x3 matObjToTc3x3 = matObjToTc.getUpperLeftMatrix();
                const Point3D pUL = Point3D(tcUL, 0.0f) + cornerHeightUL * matObjToTc3x3 * nULInObj;
                const Point3D pUR = Point3D(tcUR, 0.0f) + cornerHeightUR * matObjToTc3x3 * nURInObj;
                const Point3D pBL = Point3D(tcBL, 0.0f) + cornerHeightBL * matObjToTc3x3 * nBLInObj;
                const Point3D pBR = Point3D(tcBR, 0.0f) + cornerHeightBR * matObjToTc3x3 * nBRInObj;

                const auto testRayVsTriangleIntersection = []
                (const Point3D &org, const Vector3D &dir, float distMin, float distMax,
                 const Point3D &p0, const Point3D &p1, const Point3D &p2,
                 Vector3D* n, float* t, float* beta, float* gamma) {
                    const Vector3D e0 = p1 - p0;
                    const Vector3D e1 = p0 - p2;
                    *n = cross(e1, e0);

                    const Vector3D e2 = (1.0f / dot(*n, dir)) * (p0 - org);
                    const Vector3D i = cross(dir, e2);

                    *beta = dot(i, e1);
                    *gamma = dot(i, e0);
                    *t = dot(*n, e2);

                    return (
                        (*t < distMax) & (*t > distMin)
                        & (*beta >= 0.0f) & (*gamma >= 0.0f) & (*beta + *gamma <= 1));
                };

                float t = INFINITY;
                float mb1, mb2;
                Vector3D n;
                if (testRayVsTriangleIntersection(
                    rayOrgInTc, rayDirInTc, tMin, tMax, pUL, pUR, pBR, &n, &t, &mb1, &mb2)) {
                    if (t < tMax) {
                        const Point2D hp((1 - (mb1 + mb2)) * tcUL + mb1 * tcUR + mb2 * tcBR);
                        const float b1 = cross(tcs[2] - hp, tcs[0] - hp) * recTriAreaInTc;
                        const float b2 = cross(tcs[0] - hp, tcs[1] - hp) * recTriAreaInTc;
                        if (b1 >= 0.0f && b2 >= 0.0f && b1 + b2 <= 1.0f) {
                            tMax = t;
                            hitBc1 = b1;
                            hitBc2 = b2;
                            hitNormal = static_cast<Normal3D>(n);
                        }
                    }
                }
                if (testRayVsTriangleIntersection(
                    rayOrgInTc, rayDirInTc, tMin, tMax, pUL, pBR, pBL, &n, &t, &mb1, &mb2)) {
                    if (t < tMax) {
                        const Point2D hp((1 - (mb1 + mb2)) * tcUL + mb1 * tcBR + mb2 * tcBL);
                        const float b1 = cross(tcs[2] - hp, tcs[0] - hp) * recTriAreaInTc;
                        const float b2 = cross(tcs[0] - hp, tcs[1] - hp) * recTriAreaInTc;
                        if (b1 >= 0.0f && b2 >= 0.0f && b1 + b2 <= 1.0f) {
                            tMax = t;
                            hitBc1 = b1;
                            hitBc2 = b2;
                            hitNormal = static_cast<Normal3D>(n);
                        }
                    }
                }
            }
            if constexpr (intersectionType == LocalIntersectionType::Bilinear) {
                const float cornerHeightUL =
                    dispParams.hOffset + dispParams.hScale
                    * (sample(curTexel.x - 0.0f, curTexel.y - 0.0f) - dispParams.hBias);
                const float cornerHeightUR =
                    dispParams.hOffset + dispParams.hScale
                    * (sample(curTexel.x + 1.0f, curTexel.y - 0.0f) - dispParams.hBias);
                const float cornerHeightBL =
                    dispParams.hOffset + dispParams.hScale
                    * (sample(curTexel.x - 0.0f, curTexel.y + 1.0f) - dispParams.hBias);
                const float cornerHeightBR =
                    dispParams.hOffset + dispParams.hScale
                    * (sample(curTexel.x + 1.0f, curTexel.y + 1.0f) - dispParams.hBias);

#define DEBUG_BILINEAR 0

#if DEBUG_BILINEAR
                if (isDebugPixel && getDebugPrintEnabled()) {
                    printf(
                        "%u-%u: %u-%u-%u\n",
                        plp.f->frameIndex, optixGetPrimitiveIndex(),
                        curTexel.lod, curTexel.x, curTexel.y);
                    printf(
                        "%u-%u: v0 (%g, %g, %g, %g, %g, %g, %g, %g)\n",
                        plp.f->frameIndex, optixGetPrimitiveIndex(),
                        v3print(vs[0].position), v3print(vs[0].normal), v2print(vs[0].texCoord));
                    printf(
                        "%u-%u: v1 (%g, %g, %g, %g, %g, %g, %g, %g)\n",
                        plp.f->frameIndex, optixGetPrimitiveIndex(),
                        v3print(vs[1].position), v3print(vs[1].normal), v2print(vs[1].texCoord));
                    printf(
                        "%u-%u: v2 (%g, %g, %g, %g, %g, %g, %g, %g)\n",
                        plp.f->frameIndex, optixGetPrimitiveIndex(),
                        v3print(vs[2].position), v3print(vs[2].normal), v2print(vs[2].texCoord));

                    printf(
                        "%u-%u: Height: %g, %g, %g, %g\n",
                        plp.f->frameIndex, optixGetPrimitiveIndex(),
                        cornerHeightUL, cornerHeightUR, cornerHeightBL, cornerHeightBR);
                    printf(
                        "%u-%u: org: (%g, %g, %g), dir: (%g, %g, %g)\n",
                        plp.f->frameIndex, optixGetPrimitiveIndex(),
                        v3print(optixGetObjectRayOrigin()), v3print(rayDirInObj));
                }
#endif

                const auto testRayVsBilinearPatchIntersection = [&]
                (const Point2D &avgTc,
                 const Matrix3x3 &matTcToP, const Matrix3x3 &matTcToN, const Matrix3x3 &matTcToBc,
                 const Point3D &rayOrg, const Vector3D &rayDir,
                 float* hitDist, float* b1, float* b2, Normal3D* hitNormal) {
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
                            (1 - ut) * (1 - vt) * cornerHeightUL
                            + ut * (1 - vt) * cornerHeightUR
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
                                || bc[0] > 1.0f || bc[1] > 1.0f || bc[2] > 1.0f
                                || dotDirDelta < 0) {
                                *hitDist = INFINITY;
                                return false;
                            }
                            *hitDist = std::sqrt(hitDist2/* / rayDir.sqLength()*/);
                            *hitNormal = static_cast<Normal3D>(normalize(cross(jacobS[1], jacobS[0])));
#if DEBUG_BILINEAR
                            if (isDebugPixel && getDebugPrintEnabled()) {
                                printf(
                                    "%u-%u-%u: guess: (%g, %g), dist: %g, S: (%g, %g, %g), n: (%g, %g, %g)\n",
                                    plp.f->frameIndex, optixGetPrimitiveIndex(), itr,
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
                            *b1 = bc[1];
                            *b2 = bc[2];
                        }
                    }

                    return false;
                };

                Normal3D n;
                float t;
                float b1, b2;
                if (testRayVsBilinearPatchIntersection(
                    texelCenter,
                    matTcToPInObj, matTcToNInObj, matTcToBc,
                    Point3D(optixGetObjectRayOrigin()), rayDirInObj,
                    &t, &b1, &b2, &n)) {
                    if (t < tMax) {
                        tMax = t;
                        hitBc1 = b1;
                        hitBc2 = b2;
                        hitNormal = n;
                    }
                }
            }
            if constexpr (intersectionType == LocalIntersectionType::BSpline) {
                Assert_NotImplemented();
            }

            next(curTexel, signX, signY, maxDepth);
        }
    }

    if (tMax == optixGetRayTmax())
        return;

    DisplacedSurfaceAttributes attr = {};
    if constexpr (intersectionType == LocalIntersectionType::Box ||
                  intersectionType == LocalIntersectionType::TwoTriangle)
        attr.normalInObj = normalize(transpose(matObjToTc.getUpperLeftMatrix()) * hitNormal);
    else
        attr.normalInObj = hitNormal;
#if OUTPUT_TRAVERSAL_STATS
    attr.numIterations = numIterations;
#endif
    const uint8_t hitKind = dot(rayDirInTc, hitNormal) <= 0 ?
        CustomHitKind_DisplacedSurfaceFrontFace :
        CustomHitKind_DisplacedSurfaceBackFace;
    DisplacedSurfaceAttributeSignature::reportIntersection(tMax, hitKind, hitBc1, hitBc2, attr);
}

CUDA_DEVICE_KERNEL void RT_IS_NAME(displacedSurface_Box)() {
    displacedSurface_generic<LocalIntersectionType::Box>();
}

CUDA_DEVICE_KERNEL void RT_IS_NAME(displacedSurface_TwoTriangle)() {
    displacedSurface_generic<LocalIntersectionType::TwoTriangle>();
}

CUDA_DEVICE_KERNEL void RT_IS_NAME(displacedSurface_Bilinear)() {
    displacedSurface_generic<LocalIntersectionType::Bilinear>();
}

CUDA_DEVICE_KERNEL void RT_IS_NAME(displacedSurface_BSpline)() {
    displacedSurface_generic<LocalIntersectionType::BSpline>();
}
