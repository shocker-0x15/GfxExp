#define PURE_CUDA
#include "../nrtdsm_shared.h"

using namespace shared;

CUDA_DEVICE_FUNCTION float2 computeTexelMinMax(
    const CUtexObject heightMap, const int32_t mipLevel, const int2 &imgSize, const int2 &pixIdx)
{
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



CUDA_DEVICE_KERNEL void generateFirstMinMaxMipMap(
    const GeometryInstanceDataForNRTDSM* const nrtdsmGeomInst)
{
    const int2 pixIdx(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 imgSize = nrtdsmGeomInst->heightMapSize;
    if (pixIdx.x >= imgSize.x || pixIdx.y >= imgSize.y)
        return;

    nrtdsmGeomInst->minMaxMipMap[0].write(
        pixIdx, 
        computeTexelMinMax(nrtdsmGeomInst->heightMap, 0, imgSize, pixIdx));
}



CUDA_DEVICE_KERNEL void generateMinMaxMipMap(
    const GeometryInstanceDataForNRTDSM* const nrtdsmGeomInst, const uint32_t srcMipLevel)
{
    const int2 dstPixIdx(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 srcImageSize = nrtdsmGeomInst->heightMapSize >> static_cast<int32_t>(srcMipLevel);
    const int2 dstImageSize = srcImageSize / 2;
    if (dstPixIdx.x >= dstImageSize.x || dstPixIdx.y >= dstImageSize.y)
        return;

    const int2 basePixIdx = 2 * dstPixIdx;
    float minHeight = stc::numeric_limits<float>::infinity();
    float maxHeight = -stc::numeric_limits<float>::infinity();
    float2 minMax;

    const optixu::NativeBlockBuffer2D<float2> &prevMinMaxMip = nrtdsmGeomInst->minMaxMipMap[srcMipLevel];

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

    nrtdsmGeomInst->minMaxMipMap[srcMipLevel + 1].write(dstPixIdx, make_float2(minHeight, maxHeight));
}



CUDA_DEVICE_FUNCTION CUDA_INLINE void traverseShellBvh(
    const Point2D &tcA, const Point2D &tcB, const Point2D &tcC,
    const bool tcFlipped, const Vector2D texTriEdgeNormals[2],
    const Point2D &texTriAabbMinP, const Point2D &texTriAabbMaxP,
    const GeometryBVH_T<shellBvhArity> &shellBvh, const Vector2D &bvhShift,
    float* const minHeight, float* const maxHeight)
{
    using InternalNode = InternalNode_T<shellBvhArity>;

    uint32_t curNodeIdx = 0;
    uint32_t curStartSlot = 0;
    while (true) {
        const InternalNode &intNode = shellBvh.intNodes[curNodeIdx];
        uint32_t nextNodeIdx = 0xFFFF'FFFF;
        for (uint32_t slot = curStartSlot; slot < shellBvhArity; ++slot) {
            if (!intNode.getChildIsValid(slot))
                break;

            AABB aabb = intNode.getChildAabb(slot);
            aabb.minP.x += bvhShift.x;
            aabb.minP.y += bvhShift.y;
            aabb.maxP.x += bvhShift.x;
            aabb.maxP.y += bvhShift.y;
            const Point2D minP = aabb.minP.xy();
            const Point2D maxP = aabb.maxP.xy();
            const TriangleSquareIntersection2DResult isectResult =
                testTriangleRectangleIntersection2D(
                    tcA, tcB, tcC, tcFlipped, texTriEdgeNormals, texTriAabbMinP, texTriAabbMaxP,
                    0.5f * (minP + maxP), 0.5f * (maxP - minP));
            if (isectResult == TriangleSquareIntersection2DResult::SquareOutsideTriangle) {
                continue;
            }
            else if (isectResult == TriangleSquareIntersection2DResult::SquareInsideTriangle ||
                     intNode.getChildIsLeaf(slot)) {
                *minHeight = std::fmin(aabb.minP.z, *minHeight);
                *maxHeight = std::fmax(aabb.maxP.z, *maxHeight);
            }
            else {
                nextNodeIdx = intNode.intNodeChildBaseIndex + intNode.getInternalChildNumber(slot);
                break;
            }
        }

        if (nextNodeIdx == 0xFFFF'FFFF) {
            if (curNodeIdx == 0)
                break;
            const ParentPointer &parentPointer = shellBvh.parentPointers[curNodeIdx];
            curNodeIdx = parentPointer.index;
            curStartSlot = parentPointer.slot + 1;
        }
        else {
            curNodeIdx = nextNodeIdx;
            curStartSlot = 0;
        }
    }
}

template <bool forShellMapping>
CUDA_DEVICE_FUNCTION CUDA_INLINE void computeAABBs_generic(
    const GeometryInstanceData* const geomInst, const GeometryInstanceDataForNRTDSM* const nrtdsmGeomInst) {
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

    // JP: 三角形を含むテクセルもしくはBVHノードのmin/maxを読み取る。
    // EN: Compute the min/max of texels or BVH nodes overlapping with the triangle.
    float minHeight = stc::numeric_limits<float>::infinity();
    float maxHeight = -stc::numeric_limits<float>::infinity();
    float preScale = 1.0f;
    {
        const Matrix3x3 &texXfm = nrtdsmGeomInst->params.textureTransform;
        Vector2D uvScale;
        texXfm.decompose(&uvScale, nullptr, nullptr);
        preScale = 1.0f / std::sqrt(uvScale.x * uvScale.y);
        const Point2D tcA = texXfm * vs[0].texCoord;
        const Point2D tcB = texXfm * vs[1].texCoord;
        const Point2D tcC = texXfm * vs[2].texCoord;
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

        const GeometryBVH_T<shellBvhArity> &shellBvh = nrtdsmGeomInst->shellBvh;

        Texel roots[4];
        uint32_t numRoots;
        int32_t maxDepth;
        int32_t targetMipLevel;
        if constexpr (forShellMapping) {
            maxDepth = 0;
            targetMipLevel = 0;
            findRootsForShellMapping(texTriAabbMinP, texTriAabbMaxP, roots, &numRoots);
        }
        else {
            maxDepth = prevPowOf2Exponent(nrtdsmGeomInst->heightMapSize.x);
            targetMipLevel = nrtdsmGeomInst->params.targetMipLevel;
            findRoots(texTriAabbMinP, texTriAabbMaxP, maxDepth, targetMipLevel, roots, &numRoots);
        }
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
            if constexpr (!forShellMapping) {
                if (curTexel.lod >= maxDepth) {
                    const float2 minmax = nrtdsmGeomInst->minMaxMipMap[maxDepth].read(make_int2(0, 0));
                    minHeight = minmax.x;
                    maxHeight = minmax.y;
                    break;
                }
            }
            Texel endTexel = curTexel;
            const int16_t initialLod = curTexel.lod;
            next(endTexel, initialLod);
            bool travTerminated = false;
            while (curTexel != endTexel) {
                const float texelScale = std::pow(2.0f, static_cast<float>(curTexel.lod - maxDepth));
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
                else {
                    if constexpr (forShellMapping) {
                        if (isectResult == TriangleSquareIntersection2DResult::SquareInsideTriangle) {
                            // JP: テクセルがベース三角形に完全に含まれる場合は取り得る高さの範囲(の近似)が
                            //     BVHのルートノードから分かるのでトラバーサルを終了する。
                            // EN: If the texel is completely enclosed by the base triangle,
                            //     a (approximated) possible height range can be obtained from the BVH root node,
                            //     therefore terminate the traversal.
                            const AABB rootAabb = shellBvh.intNodes[0].getAabb();
                            minHeight = rootAabb.minP.z;
                            maxHeight = rootAabb.maxP.z;
                            travTerminated = true;
                            break;
                        }
                        else if (curTexel.lod <= 0) {
                            // JP: シェルBVHをトラバースしてベース三角形が交差する範囲の高さの範囲を求める。
                            // EN: Traverse the shell BVH to get the height range of a region to which
                            //     the base triangle intersects.
                            traverseShellBvh(
                                tcA, tcB, tcC, tcFlipped, texTriEdgeNormals, texTriAabbMinP, texTriAabbMaxP,
                                shellBvh, Vector2D(curTexel.x, curTexel.y),
                                &minHeight, &maxHeight);
                            next(curTexel, initialLod);
                        }
                        else {
                            down(curTexel);
                        }
                    }
                    else {
                        if (isectResult == TriangleSquareIntersection2DResult::SquareInsideTriangle ||
                            curTexel.lod <= targetMipLevel) {
                            const int2 imgSize = make_int2(1 << (maxDepth - curTexel.lod));
                            const int2 wrapIndex = make_int2(
                                floorDiv(curTexel.x, imgSize.x), floorDiv(curTexel.y, imgSize.y));
                            const uint2 wrappedTexel = make_uint2(
                                curTexel.x - wrapIndex.x * imgSize.x, curTexel.y - wrapIndex.y * imgSize.y);
                            const float2 minmax = nrtdsmGeomInst->minMaxMipMap[curTexel.lod].read(wrappedTexel);
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

            if (travTerminated) {
                const AABB rootAabb = nrtdsmGeomInst->shellBvh.intNodes[0].getAabb();
                minHeight = rootAabb.minP.z;
                maxHeight = rootAabb.maxP.z;
                break;
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

    const float scale = nrtdsmGeomInst->params.hScale * preScale;
    const float amplitude = scale * (maxHeight - minHeight);
    minHeight = nrtdsmGeomInst->params.hOffset + scale * (minHeight - nrtdsmGeomInst->params.hBias);

    AABB triAabb;
    if (stc::isfinite(minHeight)) {
        triAabb.unify(vs[0].position + minHeight * vs[0].normal);
        triAabb.unify(vs[1].position + minHeight * vs[1].normal);
        triAabb.unify(vs[2].position + minHeight * vs[2].normal);
        triAabb.unify(vs[0].position + (minHeight + amplitude) * vs[0].normal);
        triAabb.unify(vs[1].position + (minHeight + amplitude) * vs[1].normal);
        triAabb.unify(vs[2].position + (minHeight + amplitude) * vs[2].normal);
    }

    aabbBuffer[primIndex] = triAabb;
    dispTriAuxInfoBuffer[primIndex].minHeight = minHeight;
    dispTriAuxInfoBuffer[primIndex].amplitude = amplitude;
}



CUDA_DEVICE_KERNEL void computeAABBsForDisplacementMapping(
    const GeometryInstanceData* const geomInst, const GeometryInstanceDataForNRTDSM* const nrtdsmGeomInst) {
    computeAABBs_generic<false>(geomInst, nrtdsmGeomInst);
}

CUDA_DEVICE_KERNEL void computeAABBsForShellMapping(
    const GeometryInstanceData* const geomInst, const GeometryInstanceDataForNRTDSM* const nrtdsmGeomInst) {
    computeAABBs_generic<true>(geomInst, nrtdsmGeomInst);
}
