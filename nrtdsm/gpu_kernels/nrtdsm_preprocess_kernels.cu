#define PURE_CUDA
#include "../nrtdsm_shared.h"

using namespace shared;

CUDA_DEVICE_FUNCTION float2 computeTexelMinMax(
    const CUtexObject heightMap, const int32_t mipLevel, const int2 &imgSize, const int2 &pixIdx) {
    const auto sample = [&](float px, float py) {
        return tex2DLod<float>(heightMap, px / imgSize.x, py / imgSize.y, mipLevel);
    };

    // TODO?: テクセルコーナー間の補間ではなくテクセルセンター間の補間とすることで
    //        Bilinearサンプル4点じゃなくてPointサンプル4点にできる？
    const float cornerHeightUL = sample(pixIdx.x - 0.0f, pixIdx.y - 0.0f);
    const float cornerHeightUR = sample(pixIdx.x + 1.0f, pixIdx.y - 0.0f);
    const float cornerHeightBL = sample(pixIdx.x - 0.0f, pixIdx.y + 1.0f);
    const float cornerHeightBR = sample(pixIdx.x + 1.0f, pixIdx.y + 1.0f);
    const float minHeight = std::fmin(std::fmin(std::fmin(
        cornerHeightUL, cornerHeightUR), cornerHeightBL), cornerHeightBR);
    const float maxHeight = std::fmax(std::fmax(std::fmax(
        cornerHeightUL, cornerHeightUR), cornerHeightBL), cornerHeightBR);

    return make_float2(minHeight, maxHeight);
}



CUDA_DEVICE_KERNEL void generateFirstMinMaxMipMap(const MaterialData* const material) {
    const int2 pixIdx(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 imgSize = material->heightMapSize;
    if (pixIdx.x >= imgSize.x || pixIdx.y >= imgSize.y)
        return;

    material->minMaxMipMap[0].write(
        pixIdx, 
        computeTexelMinMax(material->heightMap, 0, imgSize, pixIdx));
}



CUDA_DEVICE_KERNEL void generateMinMaxMipMap(
    const MaterialData* material, const uint32_t srcMipLevel) {
    const int2 dstPixIdx(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 srcImageSize = material->heightMapSize >> srcMipLevel;
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
    const GeometryInstanceData* const geomInst, const GeometryInstanceDataForNRTDSM* const nrtdsmGeomInst,
    const MaterialData* const material) {
    const uint32_t primIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (primIndex >= geomInst->triangleBuffer.getNumElements())
        return;

#define DEBUG_TRAVERSAL 0

    const Triangle &tri = geomInst->triangleBuffer[primIndex];
    const Vertex (&vs)[] = {
        geomInst->vertexBuffer[tri.index0],
        geomInst->vertexBuffer[tri.index1],
        geomInst->vertexBuffer[tri.index2]
    };
#if DEBUG_TRAVERSAL
    constexpr uint32_t debugPrimIndex = 0;
    if (primIndex == debugPrimIndex) {
        printf(
            "prim %u: "
            "p (" V3FMT "), n (" V3FMT "), tc (" V2FMT "), "
            "p (" V3FMT "), n (" V3FMT "), tc (" V2FMT "), "
            "p (" V3FMT "), n (" V3FMT "), tc (" V2FMT ")\n",
            primIndex,
            v3print(vs[0].position), v3print(vs[0].normal), v2print(vs[0].texCoord),
            v3print(vs[1].position), v3print(vs[1].normal), v2print(vs[1].texCoord),
            v3print(vs[2].position), v3print(vs[2].normal), v2print(vs[2].texCoord));
    }
#endif

    // JP: 三角形を含むテクセルのmin/maxを読み取る。
    // EN: Compute the min/max of texels overlapping with the triangle.
    float minHeight = INFINITY;
    float maxHeight = -INFINITY;
    {
        const Matrix3x3 &texXfm = nrtdsmGeomInst->params.textureTransform;
        const Point2D tcA = texXfm * vs[0].texCoord,
        const Point2D tcB = texXfm * vs[1].texCoord,
        const Point2D tcC = texXfm * vs[2].texCoord,
        const bool tcFlipped = cross(tcB - tcA, tcC - tcA) < 0;
#if DEBUG_TRAVERSAL
        if (primIndex == debugPrimIndex) {
            printf("prim %u: (" V2FMT "), (" V2FMT "), (" V2FMT ")\n",
                   primIndex,
                   v2print(tcA), v2print(tcB), v2print(tcC));
        }
#endif

        const Vector2D texTriEdgeNormals[] = {
            Vector2D(tcB.y - tcA.y, tcA.x - tcB.x),
            Vector2D(tcC.y - tcB.y, tcB.x - tcC.x),
            Vector2D(tcA.y - tcC.y, tcC.x - tcA.x),
        };
        const Point2D texTriAabbMinP = min(tcA, min(tcB, tcC));
        const Point2D texTriAabbMaxP = max(tcA, max(tcB, tcC));
#if DEBUG_TRAVERSAL
        if (primIndex == debugPrimIndex) {
            printf("prim %u: (" V2FMT "), (" V2FMT ")\n",
                   primIndex,
                   v2print(texTriAabbMinP), v2print(texTriAabbMaxP));
        }
#endif

        const int32_t maxDepth = prevPowOf2Exponent(material->heightMapSize.x);
        const int32_t targetMipLevel = nrtdsmGeomInst->params.targetMipLevel;
        Texel roots[4];
        uint32_t numRoots;
        findRoots(texTriAabbMinP, texTriAabbMaxP, maxDepth, targetMipLevel, roots, &numRoots);
#if DEBUG_TRAVERSAL
        if (primIndex == debugPrimIndex) {
            printf("prim %u: %u roots\n",
                   primIndex, numRoots);
        }
#endif
        for (int rootIdx = 0; rootIdx < lengthof(roots); ++rootIdx) {
            if (rootIdx >= numRoots)
                break;
            Texel curTexel = roots[rootIdx];
#if DEBUG_TRAVERSAL
            if (primIndex == debugPrimIndex) {
                printf("prim %u, root %d: %d - %d, %d\n",
                       primIndex, rootIdx, curTexel.lod, curTexel.x, curTexel.y);
            }
#endif
            // JP: 三角形のテクスチャー座標の範囲がかなり大きい場合は
            //     最大ミップレベルからmin/maxを読み取って処理を終了する。
            // EN: Imediately finish with reading the min/max from the maximum mip level
            //     when the texture coordinate range of the triangle is fairly large.
            if (curTexel.lod >= maxDepth) {
                const float2 minmax = material->minMaxMipMap[maxDepth].read(make_int2(0, 0));
                minHeight = minmax.x;
                maxHeight = minmax.y;
                break;
            }
            Texel endTexel = curTexel;
            const int16_t initialLod = curTexel.lod;
            next(endTexel, initialLod);
            while (curTexel != endTexel) {
                const float texelScale = 1.0f / (1 << (maxDepth - curTexel.lod));
                const TriangleSquareIntersection2DResult isectResult =
                    testTriangleSquareIntersection2D(
                        tcA, tcB, tcC, tcFlipped, texTriEdgeNormals, texTriAabbMinP, texTriAabbMaxP,
                        Point2D((curTexel.x + 0.5f) * texelScale, (curTexel.y + 0.5f) * texelScale),
                        0.5f * texelScale);
#if DEBUG_TRAVERSAL
                if (primIndex == debugPrimIndex) {
                    printf("step: texel %u, %u, %u: (%u)\n", curTexel.x, curTexel.y, curTexel.lod, isectResult);
                }
#endif
                if (isectResult == TriangleSquareIntersection2DResult::SquareOutsideTriangle) {
                    // JP: テクセルがベース三角形の外にある場合はテクセルをスキップ。
                    // EN: Skip the texel if it is outside of the base triangle.
                    next(curTexel, initialLod);
                }
                else if (isectResult == TriangleSquareIntersection2DResult::SquareInsideTriangle ||
                         curTexel.lod <= targetMipLevel) {
                    const int2 imgSize = make_int2(1 << (maxDepth - curTexel.lod));
                    const int2 wrapIndex = make_int2(floorDiv(curTexel.x, imgSize.x), floorDiv(curTexel.y, imgSize.y));
                    const uint2 wrappedTexel =
                        make_uint2(curTexel.x - wrapIndex.x * imgSize.x, curTexel.y - wrapIndex.y * imgSize.y);
                    const float2 minmax = material->minMaxMipMap[curTexel.lod].read(wrappedTexel);
                    minHeight = std::fmin(minHeight, minmax.x);
                    maxHeight = std::fmax(maxHeight, minmax.y);
                    next(curTexel, initialLod);
                }
                else {
                    down(curTexel);
                }
            }
        }
    }
#if DEBUG_TRAVERSAL
    if (primIndex == debugPrimIndex) {
        printf("prim %u: height min/max: %g/%g\n", primIndex, minHeight, maxHeight);
    }
#endif

    RWBuffer aabbBuffer(nrtdsmGeomInst->aabbBuffer);
    RWBuffer dispTriAuxInfoBuffer(nrtdsmGeomInst->dispTriAuxInfoBuffer);

    const float amplitude = nrtdsmGeomInst->params.hScale * (maxHeight - minHeight);
    minHeight = nrtdsmGeomInst->params.hOffset + nrtdsmGeomInst->params.hScale * (
        minHeight - nrtdsmGeomInst->params.hBias);

    AABB triAabb;
    triAabb.unify(vs[0].position + minHeight * vs[0].normal);
    triAabb.unify(vs[1].position + minHeight * vs[1].normal);
    triAabb.unify(vs[2].position + minHeight * vs[2].normal);
    triAabb.unify(vs[0].position + (minHeight + amplitude) * vs[0].normal);
    triAabb.unify(vs[1].position + (minHeight + amplitude) * vs[1].normal);
    triAabb.unify(vs[2].position + (minHeight + amplitude) * vs[2].normal);

    aabbBuffer[primIndex] = triAabb;
    dispTriAuxInfoBuffer[primIndex].minHeight = minHeight;
    dispTriAuxInfoBuffer[primIndex].amplitude = amplitude;
}
