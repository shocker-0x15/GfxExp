#define PURE_CUDA
#include "tfdm_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void generateFirstMinMaxMipMap(
    const MaterialData* const material) {
    const int2 dstPixIdx(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 srcImageSize = material->heightMapSize;
    const int2 dstImageSize = srcImageSize / 2;
    if (dstPixIdx.x >= dstImageSize.x || dstPixIdx.y >= dstImageSize.y)
        return;

    const int2 basePixIdx = dstPixIdx * 2;
    float minHeight = INFINITY;
    float maxHeight = -INFINITY;
    float height;

    const auto sample = []
    (const CUtexObject texture, const int2 &imageSize, const int2 &tc) {
        return tex2DLod<float>(
            texture,
            (tc.x + 0.5f) / imageSize.x,
            (tc.y + 0.5f) / imageSize.y,
            0.0f);
    };

    const CUtexObject heightMap = material->heightMap;

    height = sample(heightMap, srcImageSize, basePixIdx + int2(0, 0));
    minHeight = std::fmin(height, minHeight);
    maxHeight = std::fmax(height, maxHeight);

    height = sample(heightMap, srcImageSize, basePixIdx + int2(1, 0));
    minHeight = std::fmin(height, minHeight);
    maxHeight = std::fmax(height, maxHeight);

    height = sample(heightMap, srcImageSize, basePixIdx + int2(0, 1));
    minHeight = std::fmin(height, minHeight);
    maxHeight = std::fmax(height, maxHeight);

    height = sample(heightMap, srcImageSize, basePixIdx + int2(1, 1));
    minHeight = std::fmin(height, minHeight);
    maxHeight = std::fmax(height, maxHeight);

    material->minMaxMipMap[0].write(dstPixIdx, make_float2(minHeight, maxHeight));
}

CUDA_DEVICE_KERNEL void generateMinMaxMipMap(
    const MaterialData* material, const uint32_t srcMipLevel) {
    const int2 dstPixIdx(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 srcImageSize = material->heightMapSize >> (srcMipLevel + 1);
    const int2 dstImageSize = srcImageSize / 2;
    if (dstPixIdx.x >= dstImageSize.x || dstPixIdx.y >= dstImageSize.y)
        return;

    const int2 basePixIdx = 2 * dstPixIdx;
    float minHeight = INFINITY;
    float maxHeight = -INFINITY;
    float2 minMax;

    const optixu::NativeBlockBuffer2D<float2> &prevMinMaxMip = material->minMaxMipMap[srcMipLevel];

    minMax = prevMinMaxMip.read(basePixIdx + int2(0, 0));
    minHeight = std::fmin(minMax.x, minHeight);
    maxHeight = std::fmax(minMax.y, maxHeight);

    minMax = prevMinMaxMip.read(basePixIdx + int2(1, 0));
    minHeight = std::fmin(minMax.x, minHeight);
    maxHeight = std::fmax(minMax.y, maxHeight);

    minMax = prevMinMaxMip.read(basePixIdx + int2(0, 1));
    minHeight = std::fmin(minMax.x, minHeight);
    maxHeight = std::fmax(minMax.y, maxHeight);

    minMax = prevMinMaxMip.read(basePixIdx + int2(1, 1));
    minHeight = std::fmin(minMax.x, minHeight);
    maxHeight = std::fmax(minMax.y, maxHeight);

    material->minMaxMipMap[srcMipLevel + 1].write(dstPixIdx, make_float2(minHeight, maxHeight));
}



struct Texel {
    uint32_t x : 14;
    uint32_t y : 14;
    uint32_t lod : 4;

    CUDA_DEVICE_FUNCTION bool operator!=(const Texel &r) const {
        return x != r.x || y != r.y || lod != r.lod;
    }
};

CUDA_DEVICE_FUNCTION CUDA_INLINE void down(Texel &texel) {
    --texel.lod;
    texel.x *= 2;
    texel.y *= 2;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void up(Texel &texel) {
    ++texel.lod;
    texel.x /= 2;
    texel.y /= 2;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void next(Texel &texel) {
    while (true) {
        switch (2 * (texel.x % 2) + texel.y % 2) {
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

CUDA_DEVICE_FUNCTION TriangleSquareIntersection2DResult testTriangleSquareIntersection2D(
    const Point2D triPs[3], const Vector2D triEdgeNormals[3],
    const Point2D &triAabbMinP, const Point2D &triAabbMaxP,
    const Point2D &squareCenter, FloatType squareHalfWidth) {
    Vector2D vSquareCenter = static_cast<Vector2D>(squareCenter);
    Point2D relTriPs[] = {
        triPs[0] - vSquareCenter,
        triPs[1] - vSquareCenter,
        triPs[2] - vSquareCenter,
    };

    // JP: テクセルのAABBと三角形のAABBのIntersectionを計算する。
    // EN: 
    if (any(min(Point2D(squareHalfWidth), triAabbMaxP - vSquareCenter) <=
            max(Point2D(-squareHalfWidth), triAabbMinP - vSquareCenter)))
        return TriangleSquareIntersection2DResult::SquareOutsideTriangle;

    // JP: 
    // EN: 
    for (int eIdx = 0; eIdx < 3; ++eIdx) {
        const Vector2D &eNormal = triEdgeNormals[eIdx];
        Bool2D b = eNormal >= 0;
        Vector2D e = static_cast<Vector2D>(relTriPs[eIdx]) +
            Vector2D((b.x ? 1 : -1) * squareHalfWidth,
                     (b.y ? 1 : -1) * squareHalfWidth);
        if (dot(eNormal, e) <= 0)
            return TriangleSquareIntersection2DResult::SquareOutsideTriangle;
    }

    for (int i = 0; i < 4; ++i) {
        Point2D corner(
            (i % 2 ? -1 : 1) * squareHalfWidth,
            (i / 2 ? -1 : 1) * squareHalfWidth);
        for (int eIdx = 0; eIdx < 3; ++eIdx) {
            const Point2D &o = relTriPs[eIdx];
            const Vector2D &e1 = relTriPs[(eIdx + 1) % 3] - o;
            Vector2D e2 = corner - o;
            if (cross(e1, e2) < 0)
                return TriangleSquareIntersection2DResult::SquareOverlappingTriangle;
        }
    }

    return TriangleSquareIntersection2DResult::SquareInsideTriangle;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Texel root(
    const Point2D triPs[3], const Vector2D triEdgeNormals[3],
    const Point2D &triAabbMinP, const Point2D &triAabbMaxP,
    const uint32_t maxDepth) {

    Texel ret = { 0, 0, maxDepth };
    Texel curTexel = ret;
    while (true) {
        const float texelScale = 1.0f / (1 << (maxDepth - curTexel.lod));
        const Point2D texelAabbMinP(curTexel.x * texelScale, curTexel.y * texelScale);
        const Point2D texelAabbMaxP((curTexel.x + 1) * texelScale, (curTexel.y + 1) * texelScale);
        if (all(triAabbMinP >= texelAabbMinP) && all(triAabbMaxP <= texelAabbMaxP)) {
            // The texel contains the triangle.
            ret = curTexel;
            if (curTexel.lod == 0)
                break;
            down(curTexel);
        }
        else {
            const TriangleSquareIntersection2DResult isectResult =
                testTriangleSquareIntersection2D(
                    triPs, triEdgeNormals, triAabbMinP, triAabbMaxP,
                    Point2D((curTexel.x + 0.5f) * texelScale, (curTexel.y + 0.5f) * texelScale),
                    0.5f * texelScale);
            if (isectResult != TriangleSquareIntersection2DResult::SquareOutsideTriangle)
                break;
            next(curTexel);
        }
    }
    return ret;
}



CUDA_DEVICE_KERNEL void computeAABBs(
    const GeometryInstanceData* const geomInst, const MaterialData* const material) {
    const uint32_t primIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (primIndex >= geomInst->triangleBuffer.getNumElements())
        return;

    const Triangle &tri = geomInst->triangleBuffer[primIndex];
    const Vertex (&vs)[] = {
        geomInst->vertexBuffer[tri.index0],
        geomInst->vertexBuffer[tri.index1],
        geomInst->vertexBuffer[tri.index2]
    };
    //printf(
    //    "prim %u: "
    //    "p (" V3FMT "), n (" V3FMT "), tc (" V2FMT "), "
    //    "p (" V3FMT "), n (" V3FMT "), tc (" V2FMT "), "
    //    "p (" V3FMT "), n (" V3FMT "), tc (" V2FMT ")\n",
    //    primIndex,
    //    v3print(vs[0].position), v3print(vs[0].normal), v2print(vs[0].texCoord),
    //    v3print(vs[1].position), v3print(vs[1].normal), v2print(vs[1].texCoord),
    //    v3print(vs[2].position), v3print(vs[2].normal), v2print(vs[2].texCoord));

    float minHeight = INFINITY;
    float maxHeight = -INFINITY;
    {
        Point2D tcs[] = {
            vs[0].texCoord,
            vs[1].texCoord,
            vs[2].texCoord,
        };
        if (cross(tcs[1] - tcs[0], tcs[2] - tcs[0]) < 0)
            swap(tcs[1], tcs[2]);
        //printf("prim %u: (" V2FMT "), (" V2FMT "), (" V2FMT ")\n",
        //       primIndex,
        //       vector2Arg(tcs[0]), vector2Arg(tcs[1]), vector2Arg(tcs[2]));

        const Vector2D texTriEdgeNormals[] = {
            /*normalize(*/Vector2D(tcs[1].y - tcs[0].y, tcs[0].x - tcs[1].x)/*)*/,
            /*normalize(*/Vector2D(tcs[2].y - tcs[1].y, tcs[1].x - tcs[2].x)/*)*/,
            /*normalize(*/Vector2D(tcs[0].y - tcs[2].y, tcs[2].x - tcs[0].x)/*)*/,
        };
        const Point2D texTriAabbMinP = min(tcs[0], min(tcs[1], tcs[2]));
        const Point2D texTriAabbMaxP = max(tcs[0], max(tcs[1], tcs[2]));
        //printf("prim %u: (" V2FMT "), (" V2FMT ")\n",
        //       primIndex,
        //       vector2Arg(texTriAabbMinP), vector2Arg(texTriAabbMaxP));

        const uint32_t maxDepth =
            prevPowOf2Exponent(max(material->heightMapSize.x, material->heightMapSize.y)) - 1;
        Texel curTexel = root(tcs, texTriEdgeNormals, texTriAabbMinP, texTriAabbMaxP, maxDepth);
        Texel endTexel = curTexel;
        next(endTexel);
        while (curTexel != endTexel) {
            const float texelScale = 1.0f / (1 << (maxDepth - curTexel.lod));
            const TriangleSquareIntersection2DResult isectResult =
                testTriangleSquareIntersection2D(
                    tcs, texTriEdgeNormals, texTriAabbMinP, texTriAabbMaxP,
                    Point2D((curTexel.x + 0.5f) * texelScale, (curTexel.y + 0.5f) * texelScale),
                    0.5f * texelScale);
            //if (primIndex == 0)
            //    printf("step: texel %u, %u, %u: (%u)\n", curTexel.x, curTexel.y, curTexel.lod, isectResult);
            if (isectResult == TriangleSquareIntersection2DResult::SquareOutsideTriangle) {
                next(curTexel);
            }
            else if (isectResult == TriangleSquareIntersection2DResult::SquareInsideTriangle ||
                     curTexel.lod == 0) {
                const float2 minmax = material->minMaxMipMap[curTexel.lod].read(int2(curTexel.x, curTexel.y));
                minHeight = std::fmin(minHeight, minmax.x);
                maxHeight = std::fmax(maxHeight, minmax.y);
                next(curTexel);
            }
            else {
                down(curTexel);
            }
        }
    }
    //printf("prim %u: height min/max: %g/%g\n", primIndex, minHeight, maxHeight);

    const Point3D tcs3D[] = {
        Point3D(vs[0].texCoord, 1.0f),
        Point3D(vs[1].texCoord, 1.0f),
        Point3D(vs[2].texCoord, 1.0f),
    };
    const Matrix3x3 matTcToBc = invert(Matrix3x3(tcs3D[0], tcs3D[1], tcs3D[2]));
    const Matrix3x3 matBcToP(vs[0].position, vs[1].position, vs[2].position);
    const Matrix3x3 matBcToN(vs[0].normal, vs[1].normal, vs[2].normal);
    const Matrix3x3 matTcToP = matBcToP * matTcToBc;
    const Matrix3x3 matTcToN = matBcToN * matTcToBc;

    RWBuffer aabbBuffer(geomInst->aabbBuffer);

    const AAFloatOn2D hBound(
        geomInst->hOffset + geomInst->hScale * minHeight
        + 0.5f * geomInst->hScale * ((maxHeight - minHeight) - geomInst->hBias),
        0, 0,
        0.5f * geomInst->hScale * (maxHeight - minHeight));

    AABB triAabb;
    for (int vIdx = 0; vIdx < 3; ++vIdx) {
        const Point3D center =
            0.5f * tcs3D[vIdx]
            + 0.25f * tcs3D[(vIdx + 1) % 3]
            + 0.25f * tcs3D[(vIdx + 2) % 3];
        //AAFloatOn2D elem0(0, 1, 0, 0);
        //AAFloatOn2D elem1(0, 0, 1, 0);
        //AAFloatOn2D_Vector3D edge0 = 0.25f * (tcs3D[(vIdx + 1) % 3] - tcs3D[vIdx]) * elem0;
        //AAFloatOn2D_Vector3D edge1 = 0.25f * (tcs3D[(vIdx + 2) % 3] - tcs3D[vIdx]) * elem1;
        const AAFloatOn2D_Vector3D edge0(
            0, 0.25f * (tcs3D[(vIdx + 1) % 3] - tcs3D[vIdx]), 0, 0);
        const AAFloatOn2D_Vector3D edge1(
            0, 0, 0.25f * (tcs3D[(vIdx + 2) % 3] - tcs3D[vIdx]), 0);
        const AAFloatOn2D_Vector3D texCoord = static_cast<Vector3D>(center) + edge0 + edge1;

        const AAFloatOn2D_Vector3D pBound = matTcToP * texCoord;
        AAFloatOn2D_Vector3D nBound = matTcToN * texCoord;
        nBound.normalize();

        const AAFloatOn2D_Vector3D bounds = pBound + hBound * nBound;
        const auto iaSx = bounds.x.toIAFloat();
        const auto iaSy = bounds.y.toIAFloat();
        const auto iaSz = bounds.z.toIAFloat();
        triAabb.unify(AABB(
            Point3D(iaSx.lo(), iaSy.lo(), iaSz.lo()),
            Point3D(iaSx.hi(), iaSy.hi(), iaSz.hi())));
    }

    aabbBuffer[primIndex] = triAabb;
}
