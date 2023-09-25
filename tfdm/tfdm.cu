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
            prevPowOf2Exponent(max(material->heightMapSize.x, material->heightMapSize.y));
        Texel roots[useMultipleRootOptimization ? 4 : 1];
        uint32_t numRoots;
        findRoots(texTriAabbMinP, texTriAabbMaxP, maxDepth, roots, &numRoots);
#pragma unroll
        for (int rootIdx = 0; rootIdx < lengthof(roots); ++rootIdx) {
            if (rootIdx >= numRoots)
                break;
            Texel curTexel = roots[rootIdx];
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

    /*
    JP: 三角形によって与えられるUV領域上のアフィン演算は3つの平行四辺形上の演算の合成として厳密に評価できる。
    EN: 
          /\                                            /\
         /  \                                          /  \
        /    \                                        /    \
       /------\    =>    .------:  +  :------.    +  :      :
      / \    / \        /      /       \      \       \    /
     /   \  /   \      /      /         \      \       \  /
    /_____\/_____\    /______/           \______\       \/
    */
    AABB triAabb;
    for (int pgIdx = 0; pgIdx < 3; ++pgIdx) {
        const Point3D center =
            0.5f * tcs3D[pgIdx]
            + 0.25f * tcs3D[(pgIdx + 1) % 3]
            + 0.25f * tcs3D[(pgIdx + 2) % 3];
        const AAFloatOn2D_Vector3D edge0(
            Vector3D(0.0f), 0.25f * (tcs3D[(pgIdx + 1) % 3] - tcs3D[pgIdx]), Vector3D(0.0f), Vector3D(0.0f));
        const AAFloatOn2D_Vector3D edge1(
            Vector3D(0.0f), Vector3D(0.0f), 0.25f * (tcs3D[(pgIdx + 2) % 3] - tcs3D[pgIdx]), Vector3D(0.0f));
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
