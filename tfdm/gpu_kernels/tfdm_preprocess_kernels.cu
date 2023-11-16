#define PURE_CUDA
#include "../tfdm_shared.h"

using namespace shared;

template <LocalIntersectionType intersectionType>
CUDA_DEVICE_FUNCTION float2 computeTexelMinMax(
    const CUtexObject heightMap, const int32_t mipLevel, const int2 &imgSize, const int2 &pixIdx) {
    const auto sample = [&](float px, float py) {
        return tex2DLod<float>(heightMap, px / imgSize.x, py / imgSize.y, mipLevel);
    };

    float minHeight = INFINITY;
    float maxHeight = -INFINITY;
    if constexpr (intersectionType == LocalIntersectionType::Box ||
                  intersectionType == LocalIntersectionType::TwoTriangle ||
                  intersectionType == LocalIntersectionType::Bilinear) {
        // TODO?: テクセルコーナー間の補間ではなくテクセルセンター間の補間とすることで
        //        Bilinearサンプル4点じゃなくてPointサンプル4点にできる？
        const float cornerHeightUL = sample(pixIdx.x - 0.0f, pixIdx.y - 0.0f);
        const float cornerHeightUR = sample(pixIdx.x + 1.0f, pixIdx.y - 0.0f);
        const float cornerHeightBL = sample(pixIdx.x - 0.0f, pixIdx.y + 1.0f);
        const float cornerHeightBR = sample(pixIdx.x + 1.0f, pixIdx.y + 1.0f);
        minHeight = std::fmin(std::fmin(std::fmin(cornerHeightUL, cornerHeightUR), cornerHeightBL), cornerHeightBR);
        maxHeight = std::fmax(std::fmax(std::fmax(cornerHeightUL, cornerHeightUR), cornerHeightBL), cornerHeightBR);
    }
    if constexpr (intersectionType == LocalIntersectionType::BSpline) {
        //const float cornerHeightUL = sample(pixIdx.x - 0.5f, pixIdx.y - 0.5f);
        //const float cornerHeightUC = sample(pixIdx.x + 0.5f, pixIdx.y - 0.5f);
        //const float cornerHeightUR = sample(pixIdx.x + 1.5f, pixIdx.y - 0.5f);
        //const float cornerHeightCL = sample(pixIdx.x - 0.5f, pixIdx.y + 0.5f);
        //const float cornerHeightCC = sample(pixIdx.x + 0.5f, pixIdx.y + 0.5f);
        //const float cornerHeightCR = sample(pixIdx.x + 1.5f, pixIdx.y + 0.5f);
        //const float cornerHeightBL = sample(pixIdx.x - 0.5f, pixIdx.y + 1.5f);
        //const float cornerHeightBC = sample(pixIdx.x + 0.5f, pixIdx.y + 1.5f);
        //const float cornerHeightBR = sample(pixIdx.x + 1.5f, pixIdx.y + 1.5f);
        Assert_NotImplemented();
    }

    return make_float2(minHeight, maxHeight);
}



template <LocalIntersectionType intersectionType>
CUDA_DEVICE_FUNCTION void generateFirstMinMaxMipMap_generic(const MaterialData* const material) {
    const int2 pixIdx(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 imgSize = material->heightMapSize;
    if (pixIdx.x >= imgSize.x || pixIdx.y >= imgSize.y)
        return;

    material->minMaxMipMap[0].write(
        pixIdx, 
        computeTexelMinMax<intersectionType>(material->heightMap, 0, imgSize, pixIdx));
}

CUDA_DEVICE_KERNEL void generateFirstMinMaxMipMap_Box(const MaterialData* const material) {
    generateFirstMinMaxMipMap_generic<LocalIntersectionType::Box>(material);
}

CUDA_DEVICE_KERNEL void generateFirstMinMaxMipMap_TwoTriangle(const MaterialData* const material) {
    generateFirstMinMaxMipMap_generic<LocalIntersectionType::TwoTriangle>(material);
}

CUDA_DEVICE_KERNEL void generateFirstMinMaxMipMap_Bilinear(const MaterialData* const material) {
    generateFirstMinMaxMipMap_generic<LocalIntersectionType::Bilinear>(material);
}

CUDA_DEVICE_KERNEL void generateFirstMinMaxMipMap_BSpline(const MaterialData* const material) {
    generateFirstMinMaxMipMap_generic<LocalIntersectionType::BSpline>(material);
}



template <LocalIntersectionType intersectionType>
CUDA_DEVICE_FUNCTION void generateMinMaxMipMap_generic(
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

    // JP: 常に最高解像度のMIPレベルしか使わないのなら不要。
    // EN: This is not necessary when using only the finest mip level.
    if (dstImageSize.x >= 4 || !USE_WORKAROUND_FOR_CUDA_BC_TEX) {
        const float2 minMaxOfThisMipTexel = computeTexelMinMax<intersectionType>(
            material->heightMap, srcMipLevel + 1, dstImageSize, dstPixIdx);
        minHeight = std::fmin(minHeight, minMaxOfThisMipTexel.x);
        maxHeight = std::fmax(maxHeight, minMaxOfThisMipTexel.y);
    }

    material->minMaxMipMap[srcMipLevel + 1].write(dstPixIdx, make_float2(minHeight, maxHeight));
}

CUDA_DEVICE_KERNEL void generateMinMaxMipMap_Box(
    const MaterialData* material, const uint32_t srcMipLevel) {
    generateMinMaxMipMap_generic<LocalIntersectionType::Box>(material, srcMipLevel);
}

CUDA_DEVICE_KERNEL void generateMinMaxMipMap_TwoTriangle(
    const MaterialData* material, const uint32_t srcMipLevel) {
    generateMinMaxMipMap_generic<LocalIntersectionType::TwoTriangle>(material, srcMipLevel);
}

CUDA_DEVICE_KERNEL void generateMinMaxMipMap_Bilinear(
    const MaterialData* material, const uint32_t srcMipLevel) {
    generateMinMaxMipMap_generic<LocalIntersectionType::Bilinear>(material, srcMipLevel);
}

CUDA_DEVICE_KERNEL void generateMinMaxMipMap_BSpline(
    const MaterialData* material, const uint32_t srcMipLevel) {
    generateMinMaxMipMap_generic<LocalIntersectionType::BSpline>(material, srcMipLevel);
}



CUDA_DEVICE_KERNEL void computeAABBs(
    const GeometryInstanceData* const geomInst, const GeometryInstanceDataForTFDM* const tfdmGeomInst,
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
        const Matrix3x3 &texXfm = tfdmGeomInst->params.textureTransform;
        const Point2D tcs[] = {
            texXfm * vs[0].texCoord,
            texXfm * vs[1].texCoord,
            texXfm * vs[2].texCoord,
        };
        const bool tcFlipped = cross(tcs[1] - tcs[0], tcs[2] - tcs[0]) < 0;
#if DEBUG_TRAVERSAL
        if (primIndex == debugPrimIndex) {
            printf("prim %u: (" V2FMT "), (" V2FMT "), (" V2FMT ")\n",
                   primIndex,
                   v2print(tcs[0]), v2print(tcs[1]), v2print(tcs[2]));
        }
#endif

        const Vector2D texTriEdgeNormals[] = {
            Vector2D(tcs[1].y - tcs[0].y, tcs[0].x - tcs[1].x),
            Vector2D(tcs[2].y - tcs[1].y, tcs[1].x - tcs[2].x),
            Vector2D(tcs[0].y - tcs[2].y, tcs[2].x - tcs[0].x),
        };
        const Point2D texTriAabbMinP = min(tcs[0], min(tcs[1], tcs[2]));
        const Point2D texTriAabbMaxP = max(tcs[0], max(tcs[1], tcs[2]));
#if DEBUG_TRAVERSAL
        if (primIndex == debugPrimIndex) {
            printf("prim %u: (" V2FMT "), (" V2FMT ")\n",
                   primIndex,
                   v2print(texTriAabbMinP), v2print(texTriAabbMaxP));
        }
#endif

        const int32_t maxDepth = prevPowOf2Exponent(material->heightMapSize.x);
        //constexpr int32_t targetMipLevel = 0;
#if USE_WORKAROUND_FOR_CUDA_BC_TEX
        const int32_t targetMipLevel = min(tfdmGeomInst->params.targetMipLevel, maxDepth - 2);
#else
        const int32_t targetMipLevel = tfdmGeomInst->params.targetMipLevel;
#endif
        Texel roots[useMultipleRootOptimization ? 4 : 1];
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
                        tcs, tcFlipped, texTriEdgeNormals, texTriAabbMinP, texTriAabbMaxP,
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

    const Point3D tcs3D[] = {
        Point3D(vs[0].texCoord, 1.0f),
        Point3D(vs[1].texCoord, 1.0f),
        Point3D(vs[2].texCoord, 1.0f),
    };
    const DisplacedTriangleAuxInfo &dispTriAuxInfo = tfdmGeomInst->dispTriAuxInfoBuffer[primIndex];
    const Matrix3x3 matBcToPInObj(vs[0].position, vs[1].position, vs[2].position);
    const Matrix3x3 matTcToPInObj = matBcToPInObj * dispTriAuxInfo.matTcToBc;
    const Matrix3x3 &matTcToNInObj = dispTriAuxInfo.matTcToNInObj;

    RWBuffer aabbBuffer(tfdmGeomInst->aabbBuffer);

    const float amplitude = tfdmGeomInst->params.hScale * (maxHeight - minHeight);
    minHeight = tfdmGeomInst->params.hOffset + tfdmGeomInst->params.hScale * (minHeight - tfdmGeomInst->params.hBias);
    const AAFloatOn2D hBound(minHeight + 0.5f * amplitude, 0, 0, 0.5f * amplitude);

    /*
    JP: 三角形によって与えられるUV領域上のアフィン演算は3つの平行四辺形上の演算の合成として厳密に評価できる。
    EN: Affine arithmetic on the triangle can be performed strictly by considering the triangle as
        an union of three overlapping parallelograms.
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
        const AAFloatOn2D_Point3D texCoord = center + (edge0 + edge1);

        AAFloatOn2D_Point3D pBoundInObj = matTcToPInObj * texCoord;
        AAFloatOn2D_Vector3D nBoundInObj = static_cast<AAFloatOn2D_Vector3D>(matTcToNInObj * texCoord);
        nBoundInObj.normalize();

        const AAFloatOn2D_Point3D boundsInObj = pBoundInObj + hBound * nBoundInObj;
        const auto iaSx = boundsInObj.x.toIAFloat();
        const auto iaSy = boundsInObj.y.toIAFloat();
        const auto iaSz = boundsInObj.z.toIAFloat();
        triAabb.unify(AABB(
            Point3D(iaSx.lo(), iaSy.lo(), iaSz.lo()),
            Point3D(iaSx.hi(), iaSy.hi(), iaSz.hi())));
    }
#if DEBUG_TRAVERSAL
    if (primIndex == debugPrimIndex) {
        printf(
            "prim %u: triAabb: (%g, %g, %g) - (%g, %g, %g)\n",
            primIndex,
            v3print(triAabb.minP), v3print(triAabb.maxP));
    }
#endif

    aabbBuffer[primIndex] = triAabb;
}
