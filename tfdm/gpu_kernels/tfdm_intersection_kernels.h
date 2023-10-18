#pragma once

#include "../tfdm_shared.h"

CUDA_DEVICE_KERNEL void RT_IS_NAME(aabb)() {
    using namespace shared;

    const auto sbtr = HitGroupSBTRecordData::get();
    const TFDMData &tfdm = plp.s->tfdmDataBuffer[sbtr.geomInstSlot];

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
        static_cast<uint32_t>(isFrontHit ? CustomHitKind::AABBFrontFace : CustomHitKind::AABBBackFace),
        u, v);
}



CUDA_DEVICE_FUNCTION CUDA_INLINE void down(Texel &texel, bool signX, bool signY) {
  --texel.lod;
  texel.x *= 2;
  texel.x += signX;
  texel.y *= 2;
  texel.y += signY;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void next(Texel &texel, bool signX, bool signY, uint32_t maxDepth) {
  while (texel.lod <= maxDepth) {
    switch (2 * ((texel.x + signX) % 2) + (texel.y + signY) % 2) {
    case 1:
      texel.y += signY ? 1 : -1;
      texel.x += signX ? -1 : 1;
      return;
    case 3:
      up(texel);
      break;
    default:
      texel.y += signY ? -1 : 1;
      return;
    }
  }
}

CUDA_DEVICE_KERNEL void RT_IS_NAME(displacedSurface)() {
    using namespace shared;

    const auto sbtr = HitGroupSBTRecordData::get();
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];
    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    const Triangle &tri = geomInst.triangleBuffer[optixGetPrimitiveIndex()];
    const Vertex (&vs)[] = {
        geomInst.vertexBuffer[tri.index0],
        geomInst.vertexBuffer[tri.index1],
        geomInst.vertexBuffer[tri.index2]
    };

    Point2D tcs[] = {
        vs[0].texCoord,
        vs[1].texCoord,
        vs[2].texCoord,
    };
    const float triAreaInTc = cross(tcs[1] - tcs[0], tcs[2] - tcs[0])/* * 0.5f*/;
    if (triAreaInTc < 0)
        swap(tcs[1], tcs[2]);
    const float recTriAreaInTc = std::fabs(1.0f / triAreaInTc);

    const Vector2D texTriEdgeNormals[] = {
        Vector2D(tcs[1].y - tcs[0].y, tcs[0].x - tcs[1].x),
        Vector2D(tcs[2].y - tcs[1].y, tcs[1].x - tcs[2].x),
        Vector2D(tcs[0].y - tcs[2].y, tcs[2].x - tcs[0].x),
    };
    const Point2D texTriAabbMinP = min(tcs[0], min(tcs[1], tcs[2]));
    const Point2D texTriAabbMaxP = max(tcs[0], max(tcs[1], tcs[2]));

    const TFDMData &tfdm = plp.s->tfdmDataBuffer[sbtr.geomInstSlot];
    const DisplacedTriangleAuxInfo &dispTriAuxInfo = tfdm.dispTriAuxInfoBuffer[optixGetPrimitiveIndex()];
    const Matrix3x3 matTcToNInTc =
        dispTriAuxInfo.matObjToTc.getUpperLeftMatrix() * dispTriAuxInfo.matTcToNInObj;

    Normal3D hitNormalInTc;
    float hitBc1, hitBc2;
    float tMax = optixGetRayTmax();
    const float tMin = optixGetRayTmin();

    // JP: レイと変位させたサーフェスの交叉判定はテクスチャー空間で考える。
    // EN: Test ray vs displace surface intersection in the texture space.
    const Point3D rayOrgInTc = dispTriAuxInfo.matObjToTc * Point3D(optixGetObjectRayOrigin());
    const Vector3D rayDirInTc = dispTriAuxInfo.matObjToTc * Vector3D(optixGetObjectRayDirection());
    const bool signX = rayDirInTc.x < 0;
    const bool signY = rayDirInTc.y < 0;

    const uint32_t maxDepth =
        prevPowOf2Exponent(max(mat.heightMapSize.x, mat.heightMapSize.y));
    Texel roots[useMultipleRootOptimization ? 4 : 1];
    uint32_t numRoots;
    findRoots(texTriAabbMinP, texTriAabbMaxP, maxDepth, roots, &numRoots);
    for (int rootIdx = 0; rootIdx < lengthof(roots); ++rootIdx) {
        if (rootIdx >= numRoots)
            break;
        Texel curTexel = roots[rootIdx];
        Texel endTexel = curTexel;
        next(endTexel, signX, signY, maxDepth);
        while (curTexel != endTexel) {
            const float texelScale = 1.0f / (1 << (maxDepth - curTexel.lod));
            const Point2D texelCenter = Point2D(curTexel.x + 0.5f, curTexel.y + 0.5f) * texelScale;
            const TriangleSquareIntersection2DResult isectResult =
                testTriangleSquareIntersection2D(
                    tcs, texTriEdgeNormals, texTriAabbMinP, texTriAabbMaxP,
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
                const float2 minmax = mat.minMaxMipMap[curTexel.lod].read(int2(curTexel.x, curTexel.y));
                const float amplitude = tfdm.hScale * (minmax.y - minmax.x);
                const float minHeight = tfdm.hOffset + tfdm.hScale * (minmax.x - tfdm.hBias);
                const AAFloatOn2D hBound(minHeight + 0.5f * amplitude, 0, 0, 0.5f * amplitude);

                const AAFloatOn2D_Vector3D edge0(
                    Vector3D(0.0f), Vector3D(0.5f * texelScale, 0, 0), Vector3D(0.0f), Vector3D(0.0f));
                const AAFloatOn2D_Vector3D edge1(
                    Vector3D(0.0f), Vector3D(0.0f), Vector3D(0, 0.5f * texelScale, 0), Vector3D(0.0f));
                const AAFloatOn2D_Point3D texCoord =
                    Point3D(texelCenter.x, texelCenter.y, 1.0f) + (edge0 + edge1);

                const AAFloatOn2D_Point3D pBoundInTc(texCoord.x, texCoord.y, AAFloatOn2D(0.0f));
                AAFloatOn2D_Vector3D nBoundInTc =
                    static_cast<AAFloatOn2D_Vector3D>(matTcToNInTc * texCoord);
                nBoundInTc.normalize();
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
#if USE_WORKAROUND_FOR_CUDA_BC_TEX
            if (curTexel.lod > min(plp.f->targetMipLevel, maxDepth - 2)) {
#else
            if (curTexel.lod > plp.f->targetMipLevel) {
#endif
                down(curTexel, signX, signY);
                continue;
            }

            // JP: レイと変位を加えたサーフェスとの交叉判定を行う。
            // EN: Test ray intersection against the displaced surface.
            const auto isectType = static_cast<LocalIntersectionType>(plp.f->localIntersectionType);
            switch (isectType) {
            case LocalIntersectionType::Box: {
                float param0, param1;
                bool isF;
                const float t = texelAabb.intersect(
                    rayOrgInTc, rayDirInTc, tMin, tMax, &param0, &param1, &isF);
                if (t < tMax) {
                    Normal3D n;
                    const Point2D hp(texelAabb.restoreHitPoint(param0, param1, &n));
                    const float b1 = cross(tcs[2] - hp, tcs[0] - hp) * recTriAreaInTc;
                    const float b2 = cross(tcs[0] - hp, tcs[1] - hp) * recTriAreaInTc;
                    if (b1 >= 0.0f && b2 >= 0.0f && b1 + b2 <= 1.0f) {
                        tMax = t;
                        hitBc1 = b1;
                        hitBc2 = b2;
                        hitNormalInTc = static_cast<Normal3D>(n);
                    }
                }
                break;
            }
            case LocalIntersectionType::TwoTriangle: {
                const int2 imgSize = make_int2(1 << (maxDepth - curTexel.lod));
                const auto sample = [&](float px, float py) {
                    return tex2DLod<float>(mat.heightMap, px / imgSize.x, py / imgSize.y, curTexel.lod);
                };

                const float cornerHeightUL = sample(curTexel.x - 0.5f, curTexel.y - 0.5f);
                const float cornerHeightUR = sample(curTexel.x + 0.5f, curTexel.y - 0.5f);
                const float cornerHeightBL = sample(curTexel.x - 0.5f, curTexel.y + 0.5f);
                const float cornerHeightBR = sample(curTexel.x + 0.5f, curTexel.y + 0.5f);

                const Point2D tcUL(texelCenter + texelScale * Vector2D(-0.5f, -0.5f));
                const Point2D tcUR(texelCenter + texelScale * Vector2D(0.5f, -0.5f));
                const Point2D tcBL(texelCenter + texelScale * Vector2D(-0.5f, 0.5f));
                const Point2D tcBR(texelCenter + texelScale * Vector2D(0.5f, 0.5f));

                const Point3D pUL = Point3D(tcUL, 0.0f)
                    + (tfdm.hOffset + tfdm.hScale * (cornerHeightUL - tfdm.hBias))
                    * normalize(static_cast<Vector3D>(matTcToNInTc * Point3D(tcUL, 1.0f)));
                const Point3D pUR = Point3D(tcUR, 0.0f)
                    + (tfdm.hOffset + tfdm.hScale * (cornerHeightUR - tfdm.hBias))
                    * normalize(static_cast<Vector3D>(matTcToNInTc * Point3D(tcUR, 1.0f)));
                const Point3D pBL = Point3D(tcBL, 0.0f)
                    + (tfdm.hOffset + tfdm.hScale * (cornerHeightBL - tfdm.hBias))
                    * normalize(static_cast<Vector3D>(matTcToNInTc * Point3D(tcBL, 1.0f)));
                const Point3D pBR = Point3D(tcBR, 0.0f)
                    + (tfdm.hOffset + tfdm.hScale * (cornerHeightBR - tfdm.hBias))
                    * normalize(static_cast<Vector3D>(matTcToNInTc * Point3D(tcBR, 1.0f)));

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

                Vector3D n;
                float t;
                float mb1, mb2;
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
                            hitNormalInTc = static_cast<Normal3D>(n);
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
                            hitNormalInTc = static_cast<Normal3D>(n);
                        }
                    }
                }

                break;
            }
            case LocalIntersectionType::Bilinear:
            case LocalIntersectionType::BSpline:
                Assert_NotImplemented();
            default:
                Assert_ShouldNotBeCalled();
                break;
            }

            next(curTexel, signX, signY, maxDepth);
        }
    }

    if (tMax == optixGetRayTmax())
        return;

    const Normal3D normalInObj = normalize(dispTriAuxInfo.matTcToObj * hitNormalInTc);
    DisplacedSurfaceAttributeSignature::reportIntersection(
        tMax,
        static_cast<uint32_t>(
            dot(rayDirInTc, hitNormalInTc) <= 0 ?
            CustomHitKind::DisplacedSurfaceFrontFace :
            CustomHitKind::DisplacedSurfaceBackFace),
        hitBc1, hitBc2, normalInObj);
}
