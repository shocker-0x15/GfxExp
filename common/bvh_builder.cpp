#include "bvh_builder.h"
#include "common_host.h"

namespace bvh {

enum class PrimitiveType {
    Geometric = 0,
    Instance,
};



template <PrimitiveType primType>
struct BuilderInput;

template <>
struct BuilderInput<PrimitiveType::Geometric> {
    const Geometry* geometries;
    uint32_t numGeometries;
    float splittingBudget;
    float intNodeTravCost;
    float primIntersectCost;
    uint32_t minNumPrimsPerLeaf : 16;
    uint32_t maxNumPrimsPerLeaf : 16;
};

template <>
struct BuilderInput<PrimitiveType::Instance> {
    const Instance* instances;
    uint32_t numInstances;
    float rebraidingBudget;
    uint32_t minNumPrimsPerLeaf : 16;
    uint32_t maxNumPrimsPerLeaf : 16;
};



template <uint32_t arity, PrimitiveType primType>
struct BVHAlias;
template <uint32_t arity>
struct BVHAlias<arity, PrimitiveType::Geometric> {
    using type = GeometryBVH<arity>;
};
template <uint32_t arity>
struct BVHAlias<arity, PrimitiveType::Instance> {
    using type = InstanceBVH<arity>;
};



constexpr int32_t numObjBins = 16;
constexpr int32_t objBinIdxBitWidth = 4;
constexpr int32_t numObjPlanes = 15;
constexpr int32_t objPlaneIdxBitWidth = 4;
constexpr int32_t numSpaBins = 32;
constexpr int32_t spaBinIdxBitWidth = 5;
constexpr int32_t numSpaPlanes = 31;
constexpr int32_t spaPlaneIdxBitWidth = 5;
constexpr int32_t planeIdxBitWidth = std::max(objPlaneIdxBitWidth, spaPlaneIdxBitWidth);

struct PrimitiveReference {
    AABB box;
    union {
        struct {
            uint32_t geomIndex;
            uint32_t primIndex;
        };
        struct {
            uint32_t instIndex;
            uint32_t nodeIndex;
        };
    };

    PrimitiveReference() : geomIndex(0), primIndex(0) {}
};

union PrimSplitInfo {
    struct {
        uint32_t binIdxX : objBinIdxBitWidth;
        uint32_t binIdxY : objBinIdxBitWidth;
        uint32_t binIdxZ : objBinIdxBitWidth;
    };
    struct {
        uint32_t entryBinIdxX : spaBinIdxBitWidth;
        uint32_t exitBinIdxX : spaBinIdxBitWidth;
        uint32_t entryBinIdxY : spaBinIdxBitWidth;
        uint32_t exitBinIdxY : spaBinIdxBitWidth;
        uint32_t entryBinIdxZ : spaBinIdxBitWidth;
        uint32_t exitBinIdxZ : spaBinIdxBitWidth;
    };
    struct {
        uint32_t isRight : 1;
    };

    void setBinIndex(int32_t dim, uint32_t idx) {
        if (dim == 0)
            binIdxX = idx;
        else if (dim == 1)
            binIdxY = idx;
        else /*if (dim == 2)*/
            binIdxZ = idx;
    }
    uint32_t getBinIndex(int32_t dim) const {
        if (dim == 0)
            return binIdxX;
        else if (dim == 1)
            return binIdxY;
        else /*if (dim == 2)*/
            return binIdxZ;
    }
    void setEntryBinIndex(int32_t dim, uint32_t idx) {
        if (dim == 0)
            entryBinIdxX = idx;
        else if (dim == 1)
            entryBinIdxY = idx;
        else /*if (dim == 2)*/
            entryBinIdxZ = idx;
    }
    uint32_t getEntryBinIndex(int32_t dim) const {
        if (dim == 0)
            return entryBinIdxX;
        else if (dim == 1)
            return entryBinIdxY;
        else /*if (dim == 2)*/
            return entryBinIdxZ;
    }
    void setExitBinIndex(int32_t dim, uint32_t idx) {
        if (dim == 0)
            exitBinIdxX = idx;
        else if (dim == 1)
            exitBinIdxY = idx;
        else /*if (dim == 2)*/
            exitBinIdxZ = idx;
    }
    uint32_t getExitBinIndex(int32_t dim) const {
        if (dim == 0)
            return exitBinIdxX;
        else if (dim == 1)
            return exitBinIdxY;
        else /*if (dim == 2)*/
            return exitBinIdxZ;
    }
};

struct SplitTask {
    AABB geomAabb;
    AABB centAabb;
    std::span<PrimitiveReference> primRefs;
    std::span<PrimSplitInfo> primSplitInfos;
    uint32_t numActualElems;
    uint32_t parentIndex;
    uint32_t slotInParent : 4;
    uint32_t isSplittable : 1;
};

struct SplitInfo {
    uint32_t leftPrimCount;
    uint32_t rightPrimCount;
    AABB leftAabb;
    AABB rightAabb;
    float cost;
    uint32_t dim : 2;
    uint32_t planeIndex : planeIdxBitWidth;
    uint32_t isSpecialSplit : 1;
};

template <uint32_t arity>
struct TempInternalNode_T {
    struct Child {
        AABB aabb;
        uint32_t index;
        uint32_t numLeaves;
    } children[arity];
};



static void calcTriangleVertices(
    const BuilderInput<PrimitiveType::Geometric> &buildInput,
    const uint32_t geomIdx, const uint32_t primIdx,
    Point3D* const pA, Point3D* const pB, Point3D* const pC) {
    const Geometry &geom = buildInput.geometries[geomIdx];
    const auto tri = reinterpret_cast<const uint32_t*>(
        geom.triangles + geom.triangleStride * primIdx);

    *pA = geom.preTransform *
        *reinterpret_cast<const Point3D*>(geom.vertices + geom.vertexStride * tri[0]);
    *pB = geom.preTransform *
        *reinterpret_cast<const Point3D*>(geom.vertices + geom.vertexStride * tri[1]);
    *pC = geom.preTransform *
        *reinterpret_cast<const Point3D*>(geom.vertices + geom.vertexStride * tri[2]);
}



static void findBestObjectSplit(
    const std::span<PrimitiveReference> primRefs, const std::span<PrimSplitInfo> primSplitInfos,
    const uint32_t numPrimRefs, const AABB &centAabb,
    SplitInfo* const splitInfo) {
    AABB binAabbs[numObjBins][3];
    uint32_t binPrimCounts[numObjBins][3];

    // EN: Initialize bins.
    for (int32_t binIdx = 0; binIdx < numObjBins; ++binIdx) {
        for (int32_t dim = 0; dim < 3; ++dim) {
            binAabbs[binIdx][dim] = AABB();
            binPrimCounts[binIdx][dim] = 0;
        }
    }

    // EN: Perform binning.
    for (uint32_t primRefIdx = 0; primRefIdx < numPrimRefs; ++primRefIdx) {
        const PrimitiveReference &primRef = primRefs[primRefIdx];
        PrimSplitInfo &primSplitInfo = primSplitInfos[primRefIdx];
        const Point3D np = centAabb.normalize(primRef.box.getCenter());
        const uint3 binIdx3D = min(make_uint3(numObjBins * np.toNative()), numObjBins - 1);
        for (int32_t dim = 0; dim < 3; ++dim) {
            const uint32_t binIdx = binIdx3D[dim];
            binAabbs[binIdx][dim].unify(primRef.box);
            ++binPrimCounts[binIdx][dim];
            primSplitInfo.setBinIndex(dim, binIdx);
        }
    }

    // EN: Compute the AABB and the number of primitives for the right side of each split plane.
    AABB rightAabbs[numObjPlanes][3];
    uint32_t rightPrimCounts[numObjPlanes][3];
    {
        AABB accRightAabbs[3];
        uint32_t accRightPrimCounts[3] = { 0, 0, 0 };
        for (int32_t planeIdx = numObjPlanes - 1; planeIdx >= 0; --planeIdx) {
            const int32_t binIdx = planeIdx + 1;
            for (int32_t dim = 0; dim < 3; ++dim) {
                accRightAabbs[dim].unify(binAabbs[binIdx][dim]);
                accRightPrimCounts[dim] += binPrimCounts[binIdx][dim];
                rightAabbs[planeIdx][dim] = accRightAabbs[dim];
                rightPrimCounts[planeIdx][dim] = accRightPrimCounts[dim];
            }
        }
    }

    // EN: Find the best plane index for each dimension.
    int32_t bestPlaneIndices[3] = { -1, -1, -1 };
    float bestCosts[3] = { INFINITY, INFINITY, INFINITY };
    uint32_t bestLeftPrimCounts[3] = { 0, 0, 0 };
    uint32_t bestRightPrimCounts[3] = { 0, 0, 0 };
    AABB bestLeftAabbs[3];
    AABB bestRightAabbs[3];
    {
        AABB accLeftAabbs[3];
        uint32_t accLeftPrimCounts[3] = { 0, 0, 0 };
        for (int32_t planeIdx = 0; planeIdx < numObjPlanes; ++planeIdx) {
            const uint32_t binIdx = planeIdx;
            for (int32_t dim = 0; dim < 3; ++dim) {
                AABB &accLeftAabb = accLeftAabbs[dim];
                accLeftAabb.unify(binAabbs[binIdx][dim]);
                accLeftPrimCounts[dim] += binPrimCounts[binIdx][dim];
                const float leftArea = accLeftAabb.calcHalfSurfaceArea();
                const uint32_t leftPrimCount = accLeftPrimCounts[dim];
                const AABB &rightAabb = rightAabbs[planeIdx][dim];
                const float rightArea = rightAabb.calcHalfSurfaceArea();
                const uint32_t rightPrimCount = rightPrimCounts[planeIdx][dim];

                const float cost = leftArea * leftPrimCount + rightArea * rightPrimCount;
                if (cost < bestCosts[dim]) {
                    bestPlaneIndices[dim] = planeIdx;
                    bestCosts[dim] = cost;
                    bestLeftPrimCounts[dim] = leftPrimCount;
                    bestRightPrimCounts[dim] = rightPrimCount;
                    bestLeftAabbs[dim] = accLeftAabb;
                    bestRightAabbs[dim] = rightAabb;
                }
            }
        }
    }

    // EN: Determine the best split.
    const uint32_t bestDim =
        static_cast<uint32_t>(std::distance(bestCosts, std::min_element(bestCosts, bestCosts + 3)));
    splitInfo->dim = bestDim;
    splitInfo->planeIndex = bestPlaneIndices[bestDim];
    splitInfo->cost = bestCosts[bestDim];
    splitInfo->leftPrimCount = bestLeftPrimCounts[bestDim];
    splitInfo->rightPrimCount = bestRightPrimCounts[bestDim];
    splitInfo->leftAabb = bestLeftAabbs[bestDim];
    splitInfo->rightAabb = bestRightAabbs[bestDim];
    splitInfo->isSpecialSplit = false;

    Assert(
        std::isinf(splitInfo->cost) ||
        (numPrimRefs == splitInfo->leftPrimCount + splitInfo->rightPrimCount),
        "Object splitting should not change the number of primitive references.");
}

static void findBestSpatialSplit(
    const std::span<PrimitiveReference> primRefs,
    const uint32_t numPrims, const AABB &geomAabb,
    SplitInfo* const splitInfo) {
    AABB binAabbs[numSpaBins][3];
    uint32_t binPrimEntryCounts[numSpaBins][3];
    uint32_t binPrimExitCounts[numSpaBins][3];
    const Vector3D planePosCoeff = (geomAabb.maxP - geomAabb.minP) / numSpaBins;

    // EN: Initialize bins.
    for (int32_t binIdx = 0; binIdx < numSpaBins; ++binIdx) {
        for (int32_t dim = 0; dim < 3; ++dim) {
            binAabbs[binIdx][dim] = AABB();
            binPrimEntryCounts[binIdx][dim] = 0;
            binPrimExitCounts[binIdx][dim] = 0;
        }
    }

    // EN: Perform binning.
    for (uint32_t primRefIdx = 0; primRefIdx < numPrims; ++primRefIdx) {
        const PrimitiveReference &primRef = primRefs[primRefIdx];
        const Point3D entryNp = geomAabb.normalize(primRef.box.minP);
        const Point3D exitNp = geomAabb.normalize(primRef.box.maxP);
        const uint3 entryBinIdx3D = min(make_uint3(numSpaBins * entryNp.toNative()), numSpaBins - 1);
        const uint3 exitBinIdx3D = min(make_uint3(numSpaBins * exitNp.toNative()), numSpaBins - 1);
        for (int32_t dim = 0; dim < 3; ++dim) {
            const uint32_t entryBinIdx = entryBinIdx3D[dim];
            const uint32_t exitBinIdx = exitBinIdx3D[dim];
            for (int32_t binIdx = entryBinIdx; binIdx <= static_cast<int32_t>(exitBinIdx); ++binIdx)
                binAabbs[binIdx][dim].unify(primRef.box);
            ++binPrimEntryCounts[entryBinIdx][dim];
            ++binPrimExitCounts[exitBinIdx][dim];
        }
    }

    // EN: Compute the AABB and the number of primitives for the right side of each split plane.
    AABB rightAabbs[numSpaPlanes][3];
    uint32_t rightPrimCounts[numSpaPlanes][3];
    {
        AABB accRightAabbs[3];
        uint32_t accRightPrimCounts[3] = { 0, 0, 0 };
        for (int32_t planeIdx = numSpaPlanes - 1; planeIdx >= 0; --planeIdx) {
            const int32_t binIdx = planeIdx + 1;
            for (int32_t dim = 0; dim < 3; ++dim) {
                accRightAabbs[dim].unify(binAabbs[binIdx][dim]);
                accRightPrimCounts[dim] += binPrimExitCounts[binIdx][dim];
                AABB rightAabb = accRightAabbs[dim];
                rightAabb.minP[dim] = geomAabb.minP[dim] + (planeIdx + 1) * planePosCoeff[dim];
                rightAabbs[planeIdx][dim] = rightAabb;
                rightPrimCounts[planeIdx][dim] = accRightPrimCounts[dim];
            }
        }
    }

    // EN: Find the best plane index for each dimension.
    int32_t bestPlaneIndices[3] = { -1, -1, -1 };
    float bestCosts[3] = { INFINITY, INFINITY, INFINITY };
    uint32_t bestLeftPrimCounts[3] = { 0, 0, 0 };
    uint32_t bestRightPrimCounts[3] = { 0, 0, 0 };
    AABB bestLeftAabbs[3];
    AABB bestRightAabbs[3];
    {
        AABB accLeftAabbs[3];
        uint32_t accLeftPrimCounts[3] = { 0, 0, 0 };
        for (int32_t planeIdx = 0; planeIdx < numSpaPlanes; ++planeIdx) {
            const uint32_t binIdx = planeIdx;
            for (int32_t dim = 0; dim < 3; ++dim) {
                accLeftAabbs[dim].unify(binAabbs[binIdx][dim]);
                accLeftPrimCounts[dim] += binPrimEntryCounts[binIdx][dim];
                AABB leftAabb = accLeftAabbs[dim];
                leftAabb.maxP[dim] = geomAabb.minP[dim] + (planeIdx + 1) * planePosCoeff[dim];
                const float leftArea = leftAabb.calcHalfSurfaceArea();
                const uint32_t leftPrimCount = accLeftPrimCounts[dim];
                const AABB &rightAabb = rightAabbs[planeIdx][dim];
                const float rightArea = rightAabb.calcHalfSurfaceArea();
                const uint32_t rightPrimCount = rightPrimCounts[planeIdx][dim];

                const float cost = leftArea * leftPrimCount + rightArea * rightPrimCount;
                if (cost < bestCosts[dim]) {
                    bestPlaneIndices[dim] = planeIdx;
                    bestCosts[dim] = cost;
                    bestLeftPrimCounts[dim] = leftPrimCount;
                    bestRightPrimCounts[dim] = rightPrimCount;
                    bestLeftAabbs[dim] = leftAabb;
                    bestRightAabbs[dim] = rightAabb;
                }
            }
        }
    }

    // EN: Determine the best split.
    const uint32_t bestDim =
        static_cast<uint32_t>(std::distance(bestCosts, std::min_element(bestCosts, bestCosts + 3)));
    splitInfo->dim = bestDim;
    splitInfo->planeIndex = bestPlaneIndices[bestDim];
    splitInfo->cost = bestCosts[bestDim];
    splitInfo->leftPrimCount = bestLeftPrimCounts[bestDim];
    splitInfo->rightPrimCount = bestRightPrimCounts[bestDim];
    splitInfo->leftAabb = bestLeftAabbs[bestDim];
    splitInfo->rightAabb = bestRightAabbs[bestDim];
    splitInfo->isSpecialSplit = true;
}



static void performPartition(
    const SplitTask &splitTask, const std::function<bool(uint32_t)> &pred,
    const uint32_t minNumPrimsPerLeaf,
    const uint32_t leftPrimCount, const uint32_t rightPrimCount,
    SplitTask* const leftTask, SplitTask* const rightTask) {
    *leftTask = {};
    *rightTask = {};

    const uint32_t numActualElems = leftPrimCount + rightPrimCount;
    Assert(leftPrimCount > 0 && rightPrimCount > 0, "Invalid split.");
    uint32_t leftIdx = 0;
    uint32_t rightIdx = numActualElems - 1;
    while (leftIdx < rightIdx) {
        while (leftIdx < rightIdx && pred(leftIdx)) {
            const PrimitiveReference &primRef = splitTask.primRefs[leftIdx];
            leftTask->geomAabb.unify(primRef.box);
            leftTask->centAabb.unify(primRef.box.getCenter());
            ++leftIdx;
        }

        while (leftIdx < rightIdx && !pred(rightIdx)) {
            const PrimitiveReference &primRef = splitTask.primRefs[rightIdx];
            rightTask->geomAabb.unify(primRef.box);
            rightTask->centAabb.unify(primRef.box.getCenter());
            --rightIdx;
        }

        if (leftIdx < rightIdx) {
            std::swap(splitTask.primRefs[leftIdx], splitTask.primRefs[rightIdx]);
            std::swap(splitTask.primSplitInfos[leftIdx], splitTask.primSplitInfos[rightIdx]);
        }
        else {
            const PrimitiveReference &primRef = splitTask.primRefs[rightIdx];
            rightTask->geomAabb.unify(primRef.box);
            rightTask->centAabb.unify(primRef.box.getCenter());
        }
    }
    Assert(leftIdx == leftPrimCount, "Partition is inconsistent with splitting.");

    const uint32_t numPrimsReserved = static_cast<uint32_t>(splitTask.primRefs.size());
    const uint32_t numLeftPrimsReserved = std::max(
        static_cast<uint32_t>(
            numPrimsReserved * static_cast<float>(leftPrimCount) / (leftPrimCount + rightPrimCount)),
        leftPrimCount);
    const uint32_t numRightPrimsReserved = numPrimsReserved - numLeftPrimsReserved;

    if (leftPrimCount < numLeftPrimsReserved) {
        std::copy_backward(
            splitTask.primRefs.begin() + leftPrimCount,
            splitTask.primRefs.begin() + numActualElems,
            splitTask.primRefs.begin() + numLeftPrimsReserved + rightPrimCount);
#if 1
        for (uint32_t i = leftPrimCount; i < numLeftPrimsReserved; ++i)
            splitTask.primRefs[i] = {};
#endif
    }

    leftTask->primRefs = splitTask.primRefs.subspan(0, numLeftPrimsReserved);
    leftTask->primSplitInfos = splitTask.primSplitInfos.subspan(0, numLeftPrimsReserved);
    leftTask->numActualElems = leftPrimCount;
    leftTask->isSplittable = leftPrimCount > minNumPrimsPerLeaf;

    rightTask->primRefs = splitTask.primRefs.subspan(numLeftPrimsReserved, numRightPrimsReserved);
    rightTask->primSplitInfos = splitTask.primSplitInfos.subspan(numLeftPrimsReserved, numRightPrimsReserved);
    rightTask->numActualElems = rightPrimCount;
    rightTask->isSplittable = rightPrimCount > minNumPrimsPerLeaf;
}

static void performObjectSplit(
    const SplitTask &splitTask, const SplitInfo &splitInfo, const uint32_t minNumPrimsPerLeaf,
    SplitTask* const leftTask, SplitTask* const rightTask) {
    const uint32_t splitDim = splitInfo.dim;
    const auto pred = [&splitTask, &splitInfo, splitDim]
    (uint32_t idx) {
        const PrimSplitInfo &primSplitInfo = splitTask.primSplitInfos[idx];
        return primSplitInfo.getBinIndex(splitDim) <= splitInfo.planeIndex;
    };

    performPartition(
        splitTask, pred,
        minNumPrimsPerLeaf,
        splitInfo.leftPrimCount, splitInfo.rightPrimCount,
        leftTask, rightTask);
}

static void splitTriangle(
    Point3D pA, Point3D pB, Point3D pC,
    const float splitPlane, const uint32_t splitAxis,
    AABB* const bbA, AABB* const bbB) {
    uint32_t mask =
        (((pC[splitAxis] >= splitPlane) << 2) |
         ((pB[splitAxis] >= splitPlane) << 1) |
         ((pA[splitAxis] >= splitPlane) << 0));
    bool lrSwap = false;
    if (pA[splitAxis] >= splitPlane) {
        mask = ~mask & 0b111;
        lrSwap = true;
    }
    if (popcnt(mask) == 1) {
        Point3D temp = pA;
        if (mask == 0b010) {
            pA = pB;
            pB = temp;
        }
        else { // 0b100
            pA = pC;
            pC = temp;
        }
        lrSwap ^= true;
    }

    const float tAB = (splitPlane - pA[splitAxis]) / (pB[splitAxis] - pA[splitAxis]);
    const Point3D pAB = pA + tAB * (pB - pA);
    const float tAC = (splitPlane - pA[splitAxis]) / (pC[splitAxis] - pA[splitAxis]);
    const Point3D pAC = pA + tAC * (pC - pA);

    AABB aabb;
    aabb.unify(pAB).unify(pAC);
    *bbA = unify(aabb, pA);
    *bbB = unify(aabb, pB).unify(pC);

    if (lrSwap)
        stc::swap(*bbA, *bbB);
}

static void performSpatialSplit(
    const BuilderInput<PrimitiveType::Geometric> &buildInput,
    const SplitTask &splitTask, const SplitInfo &splitInfo,
    SplitTask* const leftTask, SplitTask* const rightTask) {
    const uint32_t splitDim = splitInfo.dim;
    const uint32_t splitPlaneIdx = splitInfo.planeIndex;
    const float binCoeff = (splitTask.geomAabb.maxP[splitDim] - splitTask.geomAabb.minP[splitDim]) / numSpaBins;
    const float splitPlane = splitTask.geomAabb.minP[splitDim] + (splitPlaneIdx + 1) * binCoeff;

    const float addToLeftPartialCost =
        splitInfo.rightAabb.calcHalfSurfaceArea() * (splitInfo.rightPrimCount - 1);
    const float addToRightPartialCost =
        splitInfo.leftAabb.calcHalfSurfaceArea() * (splitInfo.leftPrimCount - 1);

    const uint32_t numPrimsReserved = static_cast<uint32_t>(splitTask.primRefs.size());
    uint32_t leftPrimCount = 0;
    uint32_t rightPrimCount = 0;
    uint32_t curNumPrims = splitTask.numActualElems;
    for (uint32_t primRefIdx = 0; primRefIdx < splitTask.numActualElems; ++primRefIdx) {
        PrimitiveReference &primRef = splitTask.primRefs[primRefIdx];
        PrimSplitInfo &primSplitInfo = splitTask.primSplitInfos[primRefIdx];

        const float fEntryBinIdx =
            (primRef.box.minP[splitDim] - splitTask.geomAabb.minP[splitDim]) / binCoeff;
        const uint32_t entryBinIdx = std::min(
            static_cast<uint32_t>(fEntryBinIdx),
            static_cast<uint32_t>(numSpaBins - 1));
        const float fExitBinIdx =
            (primRef.box.maxP[splitDim] - splitTask.geomAabb.minP[splitDim]) / binCoeff;
        const uint32_t exitBinIdx = std::min(
            static_cast<uint32_t>(fExitBinIdx),
            static_cast<uint32_t>(numSpaBins - 1));

        if (entryBinIdx <= splitPlaneIdx && exitBinIdx > splitPlaneIdx) {
            // EN: Evaluate reference unsplitting.
            const float splitCost = splitInfo.cost;
#if 1
            const float addToLeftCost =
                unify(splitInfo.leftAabb, primRef.box).calcHalfSurfaceArea() * splitInfo.leftPrimCount +
                addToLeftPartialCost;
            const float addToRightCost =
                addToRightPartialCost +
                unify(splitInfo.rightAabb, primRef.box).calcHalfSurfaceArea() * splitInfo.rightPrimCount;
#else
            const float addToLeftCost = INFINITY;
            const float addToRightCost = INFINITY;
#endif
            if (splitCost < addToLeftCost && splitCost < addToRightCost &&
                curNumPrims < numPrimsReserved) {
                primSplitInfo.isRight = false;

                Point3D pA, pB, pC;
                calcTriangleVertices(
                    buildInput, primRef.geomIndex, primRef.primIndex,
                    &pA, &pB, &pC);

                PrimitiveReference &newPrimRef = splitTask.primRefs[curNumPrims];
                PrimSplitInfo &newPrimSplitInfo = splitTask.primSplitInfos[curNumPrims];
                AABB leftAabb, rightAabb;
                splitTriangle(pA, pB, pC, splitPlane, splitDim, &leftAabb, &rightAabb);
                leftAabb.intersect(primRef.box);
                rightAabb.intersect(primRef.box);
                primRef.box = leftAabb;
                newPrimRef.box = rightAabb;
                newPrimRef.geomIndex = primRef.geomIndex;
                newPrimRef.primIndex = primRef.primIndex;
                newPrimSplitInfo.isRight = true;

                ++leftPrimCount;
                ++rightPrimCount;
                ++curNumPrims;
            }
            else if (addToLeftCost < addToRightCost) {
                primSplitInfo.isRight = false;
                ++leftPrimCount;
            }
            else {
                primSplitInfo.isRight = true;
                ++rightPrimCount;
            }
        }
        else {
            if (entryBinIdx <= splitPlaneIdx) {
                primSplitInfo.isRight = false;
                ++leftPrimCount;
            }
            else {
                primSplitInfo.isRight = true;
                ++rightPrimCount;
            }
        }
    }

    const auto pred = [&splitTask]
    (uint32_t idx) {
        const PrimSplitInfo &primSplitInfo = splitTask.primSplitInfos[idx];
        return !primSplitInfo.isRight;
    };

    performPartition(
        splitTask, pred,
        buildInput.minNumPrimsPerLeaf,
        leftPrimCount, rightPrimCount,
        leftTask, rightTask);
}



template <uint32_t arity, PrimitiveType primType>
static void buildBVH(
    const BuilderInput<primType> &buildInput,
    typename BVHAlias<arity, primType>::type* const bvh) {
    using TempInternalNode = TempInternalNode_T<arity>;
    using InternalNode = shared::InternalNode_T<arity>;

    uint32_t numInputPrimitives = 0;
    std::vector<uint32_t> inputPrimOffsets;
    if constexpr (primType == PrimitiveType::Geometric) {
        inputPrimOffsets.resize(buildInput.numGeometries);
        for (uint32_t geomIdx = 0; geomIdx < buildInput.numGeometries; ++geomIdx) {
            const Geometry &geom = buildInput.geometries[geomIdx];
            inputPrimOffsets[geomIdx] = numInputPrimitives;
            numInputPrimitives += geom.numTriangles;
        }
    }
    else /*if constexpr (primType == PrimitiveType::Instance)*/ {
        (void)inputPrimOffsets;
        numInputPrimitives = buildInput.numInstances;
    }

    const auto extractGeomAndPrimIndex = [&inputPrimOffsets]
    (const uint32_t inputPrimIdx,
     uint32_t* const geomIdx, uint32_t* const primIdx) {
        const uint32_t numGeoms = static_cast<uint32_t>(inputPrimOffsets.size());
        *geomIdx = 0;
        for (int d = nextPowerOf2(numGeoms) >> 1; d >= 1; d >>= 1) {
            if (*geomIdx + d >= numGeoms)
                continue;
            if (inputPrimOffsets[*geomIdx + d] <= inputPrimIdx)
                *geomIdx += d;
        }
        *primIdx = inputPrimIdx - inputPrimOffsets[*geomIdx];
    };

    const uint32_t minNumPrimsPerLeaf = buildInput.minNumPrimsPerLeaf;
    const uint32_t maxNumPrimsPerLeaf = buildInput.maxNumPrimsPerLeaf;

    uint32_t numPrimRefsAllocated;
    float intTravCost;
    float primIsectCost;
    if constexpr (primType == PrimitiveType::Geometric) {
        numPrimRefsAllocated = std::max(
            numInputPrimitives,
            static_cast<uint32_t>((1.0f + buildInput.splittingBudget) * numInputPrimitives));
        intTravCost = buildInput.intNodeTravCost;
        primIsectCost = buildInput.primIntersectCost;
    }
    else /*if constexpr (primType == PrimitiveType::Instance)*/ {
        numPrimRefsAllocated = std::max(
            numInputPrimitives,
            static_cast<uint32_t>((1.0f + buildInput.rebraidingBudget) * numInputPrimitives));
        intTravCost = 1.0f;
        primIsectCost = 100.0f;
    }

    // EN: Initialize primitive references.
    std::vector<PrimitiveReference> primRefsMem(numPrimRefsAllocated);
    std::vector<PrimSplitInfo> primSplitInfosMem(numPrimRefsAllocated);
    std::span<PrimitiveReference> primRefs = primRefsMem;
    std::span<PrimSplitInfo> primSplitInfos = primSplitInfosMem;
    if constexpr (primType == PrimitiveType::Geometric) {
        for (uint32_t inputPrimIdx = 0; inputPrimIdx < numInputPrimitives; ++inputPrimIdx) {
            uint32_t geomIdx, primIdx;
            extractGeomAndPrimIndex(inputPrimIdx, &geomIdx, &primIdx);

            Point3D pA, pB, pC;
            calcTriangleVertices(
                buildInput, geomIdx, primIdx,
                &pA, &pB, &pC);

            PrimitiveReference primRef = {};
            primRef.box.unify(pA).unify(pB).unify(pC);
            primRef.geomIndex = geomIdx;
            primRef.primIndex = primIdx;
            primRefs[inputPrimIdx] = primRef;
        }
    }
    else /*if constexpr (primType == PrimitiveType::Instance)*/ {
        (void)extractGeomAndPrimIndex;
        for (uint32_t instIdx = 0; instIdx < numInputPrimitives; ++instIdx) {
            const Instance &inst = buildInput.instances[instIdx];
            auto &blas = reinterpret_cast<const GeometryBVH<arity> &>(inst.bvhAddress);
            const shared::InternalNode_T<arity> &rootNode = blas.intNodes[0];

            PrimitiveReference primRef = {};
            primRef.box = inst.rotFromObj * rootNode.getAabb() + inst.transFromObj;
            primRef.instIndex = instIdx;
            primRef.nodeIndex = 0;
            primRefs[instIdx] = primRef;
        }
    }

    std::vector<SplitTask> stack;

    // EN: Set up the root task to initialize the top-down build.
    {
        SplitTask rootTask = {};
        rootTask.geomAabb = AABB();
        rootTask.centAabb = AABB();
        for (uint32_t globalPrimIdx = 0; globalPrimIdx < numInputPrimitives; ++globalPrimIdx) {
            const PrimitiveReference &primRef = primRefs[globalPrimIdx];
            rootTask.geomAabb.unify(primRef.box);
            rootTask.centAabb.unify(primRef.box.getCenter());
        }
        rootTask.primRefs = primRefs;
        rootTask.primSplitInfos = primSplitInfos;
        rootTask.numActualElems = numInputPrimitives;
        rootTask.parentIndex = UINT32_MAX;
        rootTask.slotInParent = 0;
        rootTask.isSplittable = numInputPrimitives > 1;
        stack.push_back(rootTask);
    }

    const bool allowPrimRefIncrease = numPrimRefsAllocated > numInputPrimitives;
    const float rootSA = stack.back().geomAabb.calcHalfSurfaceArea();
    std::vector<TempInternalNode> tempIntNodes;

    // EN: Build a temporary BVH using the top-down approach.
    while (!stack.empty()) {
        const SplitTask task = stack.back();
        stack.pop_back();

        // EN: Try to split the current segment until we get the full arity.
        SplitTask children[arity];
        children[0] = task;
        uint32_t numChildren = 1;
        while (numChildren < arity) {
            // EN: Choose a child with the maximum surface area to split.
            float maxArea = -INFINITY;
            uint32_t slotToSplit = UINT32_MAX;
            for (uint32_t slot = 0; slot < numChildren; ++slot) {
                const SplitTask &child = children[slot];
                if (!child.isSplittable)
                    continue;
                const float area = child.geomAabb.calcHalfSurfaceArea();
                if (area > maxArea) {
                    maxArea = area;
                    slotToSplit = slot;
                }
            }
            if (slotToSplit == UINT32_MAX)
                break;

            SplitTask taskToSplit = children[slotToSplit];

            const uint32_t numPrimRefsInSubSeg = taskToSplit.numActualElems;
            const float geomSA = taskToSplit.geomAabb.calcHalfSurfaceArea();
            const float leafCost = geomSA * numPrimRefsInSubSeg * primIsectCost;

            // EN: Evaluate an object split cost.
            SplitInfo splitInfo;
            findBestObjectSplit(
                taskToSplit.primRefs, taskToSplit.primSplitInfos,
                numPrimRefsInSubSeg, taskToSplit.centAabb,
                &splitInfo);
            float splitCost = geomSA * intTravCost + splitInfo.cost * primIsectCost;
            const bool objSplitSuccess = !std::isinf(splitInfo.cost);

            if constexpr (primType == PrimitiveType::Geometric) {
                if (allowPrimRefIncrease && objSplitSuccess &&
                    numPrimRefsInSubSeg < taskToSplit.primRefs.size()) {
                    const AABB overlappedAabb = intersect(splitInfo.leftAabb, splitInfo.rightAabb);
                    const float overlappedSA = overlappedAabb.isValid() ?
                        overlappedAabb.calcHalfSurfaceArea() : 0.0f;
                    constexpr float splittingThreshold = 1e-5f;
                    if (overlappedSA / rootSA > splittingThreshold) {
                        // EN: Evaluate a spatial split cost.
                        SplitInfo spaSplitInfo;
                        findBestSpatialSplit(
                            taskToSplit.primRefs,
                            numPrimRefsInSubSeg, taskToSplit.geomAabb,
                            &spaSplitInfo);
                        const float spaSplitCost = geomSA * intTravCost + spaSplitInfo.cost * primIsectCost;
                        if (spaSplitCost < splitCost) {
                            splitInfo = spaSplitInfo;
                            splitCost = spaSplitCost;
                        }
                    }
                }
            }
            else /*if constexpr (primType == PrimitiveType::Instance)*/ {
                Assert_NotImplemented();
            }

            // EN: When the leaf cost is less than the split cost, mark this sub-segment as non-splittable.
            if (leafCost < splitCost &&
                numPrimRefsInSubSeg <= maxNumPrimsPerLeaf) {
                children[slotToSplit].isSplittable = false;
                continue;
            }

            // EN: Perform actual splitting on the current sub-segment based on the best split we found.
            SplitTask leftTask, rightTask;
            if (objSplitSuccess) {
                if (splitInfo.isSpecialSplit) {
                    if constexpr (primType == PrimitiveType::Geometric) {
                        performSpatialSplit(
                            buildInput,
                            taskToSplit, splitInfo,
                            &leftTask, &rightTask);
                    }
                    else /*if constexpr (primType == PrimitiveType::Instance)*/ {
                        Assert_NotImplemented();
                    }
                }
                else {
                    performObjectSplit(
                        taskToSplit, splitInfo, minNumPrimsPerLeaf,
                        &leftTask, &rightTask);
                }
            }
            else {
                // EN: When splitting failed, fall back to simple equal splitting.
                const uint32_t leftPrimCount = numPrimRefsInSubSeg / 2;
                const uint32_t rightPrimCount = numPrimRefsInSubSeg - leftPrimCount;
                const auto pred = [&leftPrimCount]
                (uint32_t idx) {
                    return idx < leftPrimCount;
                };
                performPartition(
                    taskToSplit, pred,
                    minNumPrimsPerLeaf,
                    leftPrimCount, rightPrimCount,
                    &leftTask, &rightTask);
            }
            children[slotToSplit] = leftTask;
            children[numChildren] = rightTask;

            ++numChildren;
        }

        // EN: When the current segment ended with no splitting, make a leaf node.
        if (numChildren == 1 && task.parentIndex != UINT32_MAX) {
            TempInternalNode &parentNode = tempIntNodes[task.parentIndex];
            typename TempInternalNode::Child &selfSlot = parentNode.children[task.slotInParent];
            selfSlot.index = static_cast<uint32_t>(std::distance(
                primRefs.data(), task.primRefs.data()));
            selfSlot.numLeaves = task.numActualElems;
            continue;
        }

        std::stable_sort(
            children, children + numChildren,
            [](const SplitTask &a, const SplitTask &b) {
            return a.numActualElems > b.numActualElems;
        });

        // EN: Allocate an internal node and set the index to the parent.
        const uint32_t intNodeIdx = static_cast<uint32_t>(tempIntNodes.size());
        if (task.parentIndex != UINT32_MAX) {
            TempInternalNode &parentNode = tempIntNodes[task.parentIndex];
            typename TempInternalNode::Child &selfSlot = parentNode.children[task.slotInParent];
            selfSlot.index = intNodeIdx;
        }
        tempIntNodes.resize(tempIntNodes.size() + 1);

        // EN: Make the internal node.
        TempInternalNode &intNode = tempIntNodes[intNodeIdx];
        for (uint32_t slot = 0; slot < numChildren; ++slot) {
            SplitTask &childTask = children[slot];
            typename TempInternalNode::Child &child = intNode.children[slot];
            child.aabb = childTask.geomAabb;
            if (childTask.isSplittable) {
                childTask.parentIndex = intNodeIdx;
                childTask.slotInParent = slot;
                stack.push_back(childTask);

                child.numLeaves = 0;
            }
            else {
                child.index = static_cast<uint32_t>(std::distance(
                    primRefs.data(), childTask.primRefs.data()));
                child.numLeaves = childTask.numActualElems;
                Assert(child.numLeaves > 0, "Invalid number of leaves as a leaf node.");
            }
        }
        for (uint32_t slot = numChildren; slot < arity; ++slot) {
            typename TempInternalNode::Child &child = intNode.children[slot];
            child.aabb = AABB();
            child.index = UINT32_MAX;
            child.numLeaves = 0;
        }
    }

    // EN: Finished to build the temporary BVH, now we convert it to the final BVH.

    std::vector<shared::TriangleStorage> triStorages;
    if constexpr (primType == PrimitiveType::Geometric) {
        // EN: Create triangle storages.
        triStorages.resize(numInputPrimitives);
        for (uint32_t inputPrimIdx = 0; inputPrimIdx < numInputPrimitives; ++inputPrimIdx) {
            uint32_t geomIdx, primIdx;
            extractGeomAndPrimIndex(inputPrimIdx, &geomIdx, &primIdx);

            Point3D pA, pB, pC;
            calcTriangleVertices(
                buildInput, geomIdx, primIdx,
                &pA, &pB, &pC);

            shared::TriangleStorage &triStorage = triStorages[inputPrimIdx];
            triStorage = {};
            triStorage.pA = pA;
            triStorage.pB = pB;
            triStorage.pC = pC;
            triStorage.geomIndex = geomIdx;
            triStorage.primIndex = primIdx;
        }
    }
    else /*if constexpr (primType == PrimitiveType::Instance)*/ {
        (void)triStorages;
        Assert_NotImplemented();
    }

    const uint32_t numIntNodes = static_cast<uint32_t>(tempIntNodes.size());

    // EN: Compute mapping from the temporary BVH to the final BVH.
    std::vector<uint32_t> dstIntNodeIndices(numIntNodes);
    std::vector<uint32_t> leafChildBlockIndices(numIntNodes);
    dstIntNodeIndices[0] = 0;
    uint32_t intChildBlockIdx = 1;
    uint32_t leafChildBlockIdx = 0;
    for (uint32_t intNodeIdx = 0; intNodeIdx < numIntNodes; ++intNodeIdx) {
        const TempInternalNode &intNode = tempIntNodes[intNodeIdx];
        leafChildBlockIndices[intNodeIdx] = leafChildBlockIdx;
        uint32_t intChildCount = 0;
        for (uint32_t slot = 0; slot < arity; ++slot) {
            const typename TempInternalNode::Child &child = intNode.children[slot];
            if (child.index == UINT32_MAX)
                break;
            if (child.numLeaves > 0) {
                leafChildBlockIdx += child.numLeaves;
            }
            else {
                dstIntNodeIndices[child.index] = intChildBlockIdx + intChildCount;
                ++intChildCount;
            }
        }
        intChildBlockIdx += intChildCount;
    }

#if defined(_DEBUG)
    std::vector<std::vector<uint32_t>> primToPrimRefMap(numInputPrimitives);
#endif

    // EN: Create internal nodes and primitive references.
    const uint32_t numFinalPrimRefs = leafChildBlockIdx;
    std::vector<InternalNode> dstIntNodes(numIntNodes);
    std::vector<shared::PrimitiveReference> dstPrimRefs(numFinalPrimRefs);
    for (uint32_t srcIntNodeIdx = 0; srcIntNodeIdx < numIntNodes; ++srcIntNodeIdx) {
        const uint32_t dstIntNodeIdx = dstIntNodeIndices[srcIntNodeIdx];
        const TempInternalNode &srcIntNode = tempIntNodes[srcIntNodeIdx];
        InternalNode &dstIntNode = dstIntNodes[dstIntNodeIdx];

        AABB quantAabb;
        uint32_t internalMask = 0;
        uint32_t firstIntChildSlot = UINT32_MAX;
        uint32_t primRefOffset = leafChildBlockIndices[srcIntNodeIdx];
        uint32_t numValidChilren = 0;
        for (uint32_t slot = 0; slot < arity; ++slot) {
            const typename TempInternalNode::Child &srcChild = srcIntNode.children[slot];
            if (srcChild.index == UINT32_MAX)
                break;

            ++numValidChilren;
            quantAabb.unify(srcChild.aabb);
            if (srcChild.numLeaves > 0) {
                for (uint32_t primRefIdx = 0; primRefIdx < srcChild.numLeaves; ++primRefIdx) {
                    const PrimitiveReference &srcPrimRef = primRefs[srcChild.index + primRefIdx];
                    shared::PrimitiveReference &dstPrimRef = dstPrimRefs[primRefOffset + primRefIdx];
                    dstPrimRef.storageIndex = inputPrimOffsets[srcPrimRef.geomIndex] + srcPrimRef.primIndex;
                    dstPrimRef.isLeafEnd = primRefIdx == srcChild.numLeaves - 1;

#if defined(_DEBUG)
                    primToPrimRefMap[dstPrimRef.storageIndex].push_back(srcChild.index + primRefIdx);
#endif
                }
                primRefOffset += srcChild.numLeaves;
            }
            else {
                internalMask |= 1 << slot;
                if (firstIntChildSlot == UINT32_MAX)
                    firstIntChildSlot = slot;
            }
        }

        dstIntNode.setQuantizationAabb(quantAabb);
        dstIntNode.internalMask = internalMask;

        if (firstIntChildSlot != UINT32_MAX)
            dstIntNode.intNodeChildBaseIndex = dstIntNodeIndices[srcIntNode.children[firstIntChildSlot].index];
        else
            dstIntNode.intNodeChildBaseIndex = UINT32_MAX;

        if (~internalMask & ((1 << numValidChilren) - 1))
            dstIntNode.leafBaseIndex = leafChildBlockIndices[srcIntNodeIdx];
        else
            dstIntNode.leafBaseIndex = UINT32_MAX;

        uint32_t leafOffset = 0;
        for (uint32_t slot = 0; slot < arity; ++slot) {
            const typename TempInternalNode::Child &srcChild = srcIntNode.children[slot];
            if (srcChild.index != UINT32_MAX) {
                dstIntNode.setChildAabb(slot, srcChild.aabb);
                typename InternalNode::ChildMeta childMeta;
                if (srcChild.numLeaves > 0) {
                    childMeta.setLeafOffset(leafOffset);
                    leafOffset += srcChild.numLeaves;
                }
                dstIntNode.setChildMeta(slot, childMeta);
            }
            else {
                dstIntNode.setInvalidChildBox(slot);
            }
        }
    }

    // Debug Visualization
#if ENABLE_VDB && defined(_DEBUG)
    if (false) {
        if constexpr (primType == PrimitiveType::Geometric) {
            // Triangle to Primitive References
            for (uint32_t inputPrimIdx = 0; inputPrimIdx < numInputPrimitives; ++inputPrimIdx) {
                const std::vector<uint32_t> &refs = primToPrimRefMap[inputPrimIdx];

                vdb_frame();
                drawAxes(10.0f);

                const shared::TriangleStorage &triStorage = triStorages[inputPrimIdx];
                setColor(1.0f, 1.0f, 1.0f);
                drawWiredTriangle(triStorage.pA, triStorage.pB, triStorage.pC);

                for (uint32_t i = 0; i < refs.size(); ++i) {
                    const PrimitiveReference &primRef = primRefs[refs[i]];
                    setColor(0.1f, 0.1f, 0.1f);
                    drawAabb(primRef.box);
                }
                if (refs.size() > 1)
                    printf("");
                printf("");
            }
        }
        else /*if constexpr (primType == PrimitiveType::Instance)*/ {

        }
    }
#endif

    bvh->intNodes = std::move(dstIntNodes);
    if constexpr (primType == PrimitiveType::Geometric) {
        bvh->primRefs = std::move(dstPrimRefs);
        bvh->triStorages = std::move(triStorages);
        bvh->numGeoms = buildInput.numGeometries;
        bvh->totalNumPrims = numInputPrimitives;
    } 
    else /*if constexpr (primType == PrimitiveType::Instance)*/ {
        Assert_NotImplemented();
        bvh->numInsts = numInputPrimitives;
    }
}



template <uint32_t arity>
void buildGeometryBVH(
    const Geometry* const geoms, const uint32_t numGeoms,
    const GeometryBVHBuildConfig &config, GeometryBVH<arity>* const bvh) {
    BuilderInput<PrimitiveType::Geometric> input = {};
    input.geometries = geoms;
    input.numGeometries = numGeoms;
    input.splittingBudget = config.splittingBudget;
    input.intNodeTravCost = config.intNodeTravCost;
    input.primIntersectCost = config.primIntersectCost;
    input.minNumPrimsPerLeaf = config.minNumPrimsPerLeaf;
    input.maxNumPrimsPerLeaf = config.maxNumPrimsPerLeaf;
    buildBVH<arity, PrimitiveType::Geometric>(input, bvh);
}

template void buildGeometryBVH<2>(
    const Geometry* const geoms, const uint32_t numGeoms,
    const GeometryBVHBuildConfig &config, GeometryBVH<2>* const bvh);
template void buildGeometryBVH<4>(
    const Geometry* const geoms, const uint32_t numGeoms,
    const GeometryBVHBuildConfig &config, GeometryBVH<4>* const bvh);
template void buildGeometryBVH<8>(
    const Geometry* const geoms, const uint32_t numGeoms,
    const GeometryBVHBuildConfig &config, GeometryBVH<8>* const bvh);



template <uint32_t arity>
void buildInstanceBVH(
    const Instance* const insts, const uint32_t numInsts,
    const InstanceBVHBuildConfig &config, InstanceBVH<arity>* const bvh) {
    BuilderInput<PrimitiveType::Instance> input = {};
    input.instances = insts;
    input.numInstances = numInsts;
    input.rebraidingBudget = config.rebraidingBudget;
    input.minNumPrimsPerLeaf = 1;
    input.maxNumPrimsPerLeaf = 1;
    buildBVH<arity, PrimitiveType::Instance>(input, bvh);
}

template void buildInstanceBVH<2>(
    const Instance* const insts, const uint32_t numInsts,
    const InstanceBVHBuildConfig &config, InstanceBVH<2>* const bvh);
template void buildInstanceBVH<4>(
    const Instance* const insts, const uint32_t numInsts,
    const InstanceBVHBuildConfig &config, InstanceBVH<4>* const bvh);
template void buildInstanceBVH<8>(
    const Instance* const insts, const uint32_t numInsts,
    const InstanceBVHBuildConfig &config, InstanceBVH<8>* const bvh);



#define SWAP(A, B)\
    if (keys[A] > keys[B]) {\
        stc::swap(keys[A], keys[B]);\
        stc::swap(values[A], values[B]);\
    }

template <typename KeyType, typename ValueType>
static inline void sort(KeyType (&keys)[2], ValueType (&values)[2]) {
    SWAP(0, 1);
}

template <typename KeyType, typename ValueType>
static inline void sort(KeyType (&keys)[4], ValueType (&values)[4]) {
    SWAP(0, 2); SWAP(1, 3);
    SWAP(0, 1); SWAP(2, 3);
    SWAP(1, 2);
}

template <typename KeyType, typename ValueType>
static inline void sort(KeyType (&keys)[8], ValueType (&values)[8]) {
    SWAP(0, 2); SWAP(1, 3); SWAP(4, 6); SWAP(5, 7);
    SWAP(0, 4); SWAP(1, 5); SWAP(2, 6); SWAP(3, 7);
    SWAP(0, 1); SWAP(2, 3); SWAP(4, 5); SWAP(6, 7);
    SWAP(2, 4); SWAP(3, 5);
    SWAP(1, 4); SWAP(3, 6);
    SWAP(1, 2); SWAP(3, 4); SWAP(5, 6);
}

#undef SWAP

#define SWAP_ORDER(A, B, W)\
    if (keys[A] > keys[B]) {\
        stc::swap(keys[A], keys[B]);\
        constexpr uint32_t mask = (1 << W) - 1;\
        const uint32_t offsetA = A * W;\
        const uint32_t offsetB = B * W;\
        const uint32_t vA = (*values >> offsetA) & mask;\
        const uint32_t vB = (*values >> offsetB) & mask;\
        *values &= ~(mask << offsetA);\
        *values |= (vB << offsetA);\
        *values &= ~(mask << offsetB);\
        *values |= (vA << offsetB);\
    }

template <typename KeyType>
static inline void sortOrder(KeyType (&keys)[2], uint32_t* const values) {
    SWAP_ORDER(0, 1, 1);
}

template <typename KeyType>
static inline void sortOrder(KeyType (&keys)[4], uint32_t* const values) {
    SWAP_ORDER(0, 2, 2); SWAP_ORDER(1, 3, 2);
    SWAP_ORDER(0, 1, 2); SWAP_ORDER(2, 3, 2);
    SWAP_ORDER(1, 2, 2);
}

template <typename KeyType>
static inline void sortOrder(KeyType (&keys)[8], uint32_t* const values) {
    SWAP_ORDER(0, 2, 3); SWAP_ORDER(1, 3, 3); SWAP_ORDER(4, 6, 3); SWAP_ORDER(5, 7, 3);
    SWAP_ORDER(0, 4, 3); SWAP_ORDER(1, 5, 3); SWAP_ORDER(2, 6, 3); SWAP_ORDER(3, 7, 3);
    SWAP_ORDER(0, 1, 3); SWAP_ORDER(2, 3, 3); SWAP_ORDER(4, 5, 3); SWAP_ORDER(6, 7, 3);
    SWAP_ORDER(2, 4, 3); SWAP_ORDER(3, 5, 3);
    SWAP_ORDER(1, 4, 3); SWAP_ORDER(3, 6, 3);
    SWAP_ORDER(1, 2, 3); SWAP_ORDER(3, 4, 3); SWAP_ORDER(5, 6, 3);
}

#undef SWAP_ORDER

static bool testRayVsTriangle(
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    float* const hitDist, Normal3D* const hitNormal, float* const bcB, float* const bcC) {
    const Vector3D eAB = pB - pA;
    const Vector3D eCA = pA - pC;
    *hitNormal = static_cast<Normal3D>(cross(eCA, eAB));

    const Vector3D e = (1.0f / dot(*hitNormal, rayDir)) * (pA - rayOrg);
    const Vector3D i = cross(rayDir, e);

    *bcB = dot(i, eCA);
    *bcC = dot(i, eAB);
    *hitDist = dot(*hitNormal, e);

    return
        ((*hitDist < distMax) && (*hitDist > distMin)
         && (*bcB >= 0.0f) && (*bcC >= 0.0f) && (*bcB + *bcC <= 1));
}

template <uint32_t arity>
inline shared::HitObject __traverse(
    const GeometryBVH<arity> &bvh,
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    TraversalStatistics* const stats, const bool debugPrint) {
    using namespace shared;
    using InternalNode = InternalNode_T<arity>;

    HitObject ret = {};
    ret.dist = distMax;
    ret.instIndex = UINT32_MAX;
    ret.instUserData = 0;
    ret.geomIndex = UINT32_MAX;
    ret.primIndex = UINT32_MAX;
    ret.bcA = NAN;
    ret.bcB = NAN;
    ret.bcC = NAN;

    if (stats) {
        stats->numAabbTests = 0;
        stats->numTriTests = 0;
    }

#define USE_COMPRESSED_STACK 1

    double sumStackAccessDepth = 0.0;
    uint32_t numStackAccesses = 0;
    int32_t maxStackDepth = -1;
    uint32_t numIterations = 0;
    const int32_t fastStackDepthLimit = stats ? stats->fastStackDepthLimit : 0;
    uint32_t stackMemoryAccessAmount = 0;

#if USE_COMPRESSED_STACK
    union Entry {
        struct {
            uint32_t baseIndex : 31;
            uint32_t isLeafGroup : 1;
            uint32_t orderInfo : 28;
            uint32_t numItems : 4;
        };
        uint32_t asUInts[2];
    };

    Entry stack[32];
    uint8_t leafOffsets[arity];
    int32_t stackIdx = 0;
    Entry curGroup = { 0, 0, 0, 1 };
    while (true) {
        static constexpr uint32_t orderBitWidth = tzcntConst(arity);
        static constexpr uint32_t orderMask = (1 << orderBitWidth) - 1;

        if (curGroup.numItems == 0) {
            if (stackIdx == 0)
                break;
            curGroup = stack[--stackIdx];
            if (stats) {
                sumStackAccessDepth += stackIdx;
                ++numStackAccesses;
                if (stackIdx > fastStackDepthLimit)
                    stackMemoryAccessAmount += sizeof(Entry);
            }
            if (debugPrint) {
                hpprintf("Pop (%u): %u - [", stackIdx, curGroup.baseIndex);
                for (uint32_t i = 0; i < curGroup.numItems; ++i)
                    hpprintf(
                        "%u%s",
                        (curGroup.orderInfo >> (orderBitWidth * i)) & orderMask,
                        i + 1 < curGroup.numItems ? ", " : "");
                hpprintf("]\n");
            }
        }

        ++numIterations;

        Entry curTriGroup = {};
        if (curGroup.isLeafGroup) {
            curTriGroup = curGroup;
            curGroup.numItems = 0;
        }
        else {
            Assert(curGroup.numItems > 0, "No items anymore.");
            const uint32_t nodeIdx = curGroup.baseIndex + (curGroup.orderInfo & orderMask);
            curGroup.orderInfo >>= orderBitWidth;
            --curGroup.numItems;

            const InternalNode &intNode = bvh.intNodes[nodeIdx];
            if (debugPrint)
                hpprintf(
                    "Int %u: %u, %u\n",
                    nodeIdx, intNode.intNodeChildBaseIndex, intNode.leafBaseIndex);

            Entry newGroup = {};
            {
                uint32_t keys[arity];
                uint32_t orderInfo = 0;
                uint32_t numIntHits = 0;
                uint32_t numLeafHits = 0;
                for (uint32_t slot = 0; slot < arity; ++slot) {
                    if (!intNode.getChildIsValid(slot)) {
                        for (; slot < arity; ++slot)
                            keys[slot] = floatToOrderedUInt(INFINITY);
                        break;
                    }

                    if (stats)
                        ++stats->numAabbTests;
                    const AABB &aabb = intNode.getChildAabb(slot);
                    float hitDistMin, hitDistMax;
                    if (aabb.intersect(rayOrg, rayDir, distMin, ret.dist, &hitDistMin, &hitDistMax)) {
                        bool const isLeaf = intNode.getChildIsLeaf(slot);
                        const float dist = 0.5f * (hitDistMin + hitDistMax);
                        keys[slot] = (floatToOrderedUInt(dist) >> 1) | (!isLeaf << 31);
                        if (isLeaf) {
                            orderInfo |= (slot << (orderBitWidth * slot));
                            ++numLeafHits;
                            if (debugPrint)
                                hpprintf("  %u: %g, leaf\n", slot, dist);
                        }
                        else {
                            const uint32_t nthIntChild = intNode.getInternalChildIndex(slot);
                            orderInfo |= (nthIntChild << (orderBitWidth * slot));
                            ++numIntHits;
                            if (debugPrint)
                                hpprintf("  %u: %g, %u th int child\n", slot, dist, nthIntChild);
                        }
                    }
                    else {
                        keys[slot] = floatToOrderedUInt(INFINITY);
                    }
                }

                if (numIntHits + numLeafHits > 0)
                    sortOrder(keys, &orderInfo);

                if (numLeafHits > 0) {
                    curTriGroup.numItems = numLeafHits;
                    curTriGroup.baseIndex = intNode.leafBaseIndex;
                    curTriGroup.isLeafGroup = true;
                    curTriGroup.orderInfo = orderInfo;

                    //#pragma unroll
                    for (uint32_t slot = 0; slot < arity; ++slot)
                        leafOffsets[slot] = intNode.childMetas[slot].getLeafOffset();

                    if (debugPrint) {
                        hpprintf("  Order (Leaf): [");
                        for (uint32_t i = 0; i < curTriGroup.numItems; ++i)
                            hpprintf(
                                "%u%s",
                                (curTriGroup.orderInfo >> (orderBitWidth * i)) & orderMask,
                                i + 1 < curTriGroup.numItems ? ", " : "");
                        hpprintf("]\n");
                    }
                }
                if (numIntHits > 0) {
                    newGroup.numItems = numIntHits;
                    newGroup.baseIndex = intNode.intNodeChildBaseIndex;
                    newGroup.isLeafGroup = false;
                    newGroup.orderInfo = orderInfo >> (orderBitWidth * numLeafHits);

                    if (debugPrint) {
                        hpprintf("  Order (Int): [");
                        for (uint32_t i = 0; i < newGroup.numItems; ++i)
                            hpprintf(
                                "%u th%s",
                                (newGroup.orderInfo >> (orderBitWidth * i)) & orderMask,
                                i + 1 < newGroup.numItems ? ", " : "");
                        hpprintf("]\n");
                    }
                }
            }

            if (newGroup.numItems > 0) {
                if (curGroup.numItems > 0) {
                    if (stats) {
                        sumStackAccessDepth += stackIdx;
                        ++numStackAccesses;
                        maxStackDepth = std::max(stackIdx, maxStackDepth);
                        if (stackIdx > fastStackDepthLimit)
                            stackMemoryAccessAmount += sizeof(Entry);
                    }
                    stack[stackIdx++] = curGroup;
                    if (debugPrint) {
                        hpprintf("Push (%u): %u - [", stackIdx, curGroup.baseIndex);
                        for (uint32_t i = 0; i < curGroup.numItems; ++i)
                            hpprintf(
                                "%u%s",
                                (curGroup.orderInfo >> (orderBitWidth * i)) & orderMask,
                                i + 1 < curGroup.numItems ? ", " : "");
                        hpprintf("]\n");
                    }
                }
                curGroup = newGroup;
            }
        }

        if (curTriGroup.numItems > 0) {
            const uint32_t slot = curTriGroup.orderInfo & orderMask;
            const uint32_t primRefIdx = curTriGroup.baseIndex + leafOffsets[slot]++;
            if (stats)
                ++stats->numTriTests;
            const shared::PrimitiveReference primRef = bvh.primRefs[primRefIdx];
            const TriangleStorage &triStorage = bvh.triStorages[primRef.storageIndex];
            float hitDist;
            float hitBcB, hitBcC;
            Normal3D hitNormal;
            const bool hit = testRayVsTriangle(
                rayOrg, rayDir, distMin, ret.dist,
                triStorage.pA, triStorage.pB, triStorage.pC,
                &hitDist, &hitNormal, &hitBcB, &hitBcC);
            if (hit) {
                ret.dist = hitDist;
                ret.geomIndex = triStorage.geomIndex;
                ret.primIndex = triStorage.primIndex;
                ret.bcA = 1.0f - (hitBcB + hitBcC);
                ret.bcB = hitBcB;
                ret.bcC = hitBcC;
            }
            if (primRef.isLeafEnd) {
                curTriGroup.orderInfo >>= orderBitWidth;
                --curTriGroup.numItems;
            }
            if (debugPrint)
                hpprintf(
                    "Leaf %u: tri %u: %s (%g)\n",
                    primRefIdx, primRef.storageIndex, hit ? "hit" : "miss",
                    hit ? ret.dist : INFINITY);

            if (curTriGroup.numItems > 0) {
                if (curGroup.numItems > 0) {
                    if (stats) {
                        sumStackAccessDepth += stackIdx;
                        ++numStackAccesses;
                        maxStackDepth = std::max(stackIdx, maxStackDepth);
                        if (stackIdx > fastStackDepthLimit)
                            stackMemoryAccessAmount += sizeof(Entry);
                    }
                    stack[stackIdx++] = curGroup;
                    if (debugPrint) {
                        hpprintf("Push (%u): %u - [", stackIdx, curGroup.baseIndex);
                        for (uint32_t i = 0; i < curGroup.numItems; ++i)
                            hpprintf(
                                "%u%s",
                                (curGroup.orderInfo >> (orderBitWidth * i)) & orderMask,
                                i + 1 < curGroup.numItems ? ", " : "");
                        hpprintf("]\n");
                    }
                }
                curGroup = curTriGroup;
            }
        }
    }
#else
    union Entry {
        struct {
            uint32_t index : 31;
            uint32_t isLeaf : 1;
        };
        uint32_t asUInt;
    };

    Entry stack[64];
    int32_t stackIdx = 0;
    Entry curEntry = { 0, 0 };
    while (true) {
        if (curEntry.asUInt == UINT32_MAX) {
            if (stackIdx == 0)
                break;
            curEntry = stack[--stackIdx];
            if (stats) {
                sumStackAccessDepth += stackIdx;
                ++numStackAccesses;
                if (stackIdx > fastStackDepthLimit)
                    stackMemoryAccessAmount += sizeof(Entry);
            }
            if (debugPrint)
                hpprintf("Pop (%u)\n", stackIdx);
        }

        ++numIterations;

        if (curEntry.isLeaf) {
            if (stats)
                ++stats->numTriTests;
            const shared::PrimitiveReference primRef = bvh.primRefs[curEntry.index];
            const TriangleStorage &triStorage = bvh.triStorages[primRef.storageIndex];
            float hitDist;
            float hitBcB, hitBcC;
            Normal3D hitNormal;
            const bool hit = testRayVsTriangle(
                rayOrg, rayDir, distMin, ret.dist,
                triStorage.pA, triStorage.pB, triStorage.pC,
                &hitDist, &hitNormal, &hitBcB, &hitBcC);
            if (hit) {
                ret.dist = hitDist;
                ret.geomIndex = triStorage.geomIndex;
                ret.primIndex = triStorage.primIndex;
                ret.bcA = 1.0f - (hitBcB + hitBcC);
                ret.bcB = hitBcB;
                ret.bcC = hitBcC;
            }
            if (debugPrint)
                hpprintf(
                    "Leaf %u: tri %u: %s (%g)\n",
                    curEntry.index, primRef.storageIndex, hit ? "hit" : "miss",
                    hit ? ret.dist : INFINITY);
            if (primRef.isLeafEnd)
                curEntry.asUInt = UINT32_MAX;
            else
                ++curEntry.index;
            continue;
        }

        const InternalNode &intNode = bvh.intNodes[curEntry.index];
        if (debugPrint)
            hpprintf(
                "Int %u: %u, %u\n",
                curEntry.index, intNode.intNodeChildBaseIndex, intNode.leafBaseIndex);

        uint32_t keys[arity];
        Entry entries[arity];
        uint32_t numHits = 0;
        for (uint32_t slot = 0; slot < arity; ++slot) {
            if (!intNode.getChildIsValid(slot)) {
                for (; slot < arity; ++slot)
                    keys[slot] = floatToOrderedUInt(INFINITY);
                break;
            }

            if (stats)
                ++stats->numAabbTests;
            const AABB &aabb = intNode.getChildAabb(slot);
            float hitDistMin, hitDistMax;
            if (aabb.intersect(rayOrg, rayDir, distMin, ret.dist, &hitDistMin, &hitDistMax)) {
                Entry entry;
                entry.isLeaf = intNode.getChildIsLeaf(slot);
                const float dist = 0.5f * (hitDistMin + hitDistMax);
                if (entry.isLeaf) {
                    entry.index = intNode.leafBaseIndex + intNode.getLeafOffset(slot);
                    if (debugPrint)
                        hpprintf("  %u: %g, leaf\n", slot, dist);
                }
                else {
                    const uint32_t nthIntChild = intNode.getInternalChildIndex(slot);
                    entry.index = intNode.intNodeChildBaseIndex + nthIntChild;
                    if (debugPrint)
                        hpprintf("  %u: %g, %u th int child\n", slot, dist, nthIntChild);
                }
                entries[slot] = entry;
                keys[slot] = (floatToOrderedUInt(dist) >> 1) | (!entry.isLeaf << 31);
                ++numHits;
            }
            else {
                keys[slot] = floatToOrderedUInt(INFINITY);
            }
        }

        curEntry.asUInt = UINT32_MAX;
        if (numHits > 0) {
            sort(keys, entries);
            for (uint32_t i = numHits - 1; i > 0; --i) {
                if (stats) {
                    sumStackAccessDepth += stackIdx;
                    ++numStackAccesses;
                    if (stackIdx > fastStackDepthLimit)
                        stackMemoryAccessAmount += sizeof(Entry);
                }
                stack[stackIdx++] = entries[i];
                if (debugPrint)
                    hpprintf("Push (%u)\n", stackIdx);
            }
            if (stats)
                maxStackDepth = std::max(stackIdx - 1, maxStackDepth);
            curEntry = entries[0];
        }
    }
#endif

    if (stats) {
        stats->avgStackAccessDepth = numStackAccesses > 0 ?
            static_cast<float>(sumStackAccessDepth / numStackAccesses) : 0.0f;
        stats->maxStackDepth = maxStackDepth;
        stats->stackMemoryAccessAmount = stackMemoryAccessAmount;
    }

    return ret;
}

// Workaround for Intellisense.
// https://developercommunity.visualstudio.com/t/The-provide-sample-template-arguments-f/10564187
template <uint32_t arity>
shared::HitObject traverse(
    const GeometryBVH<arity> &bvh,
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    TraversalStatistics* const stats, const bool debugPrint) {
    return __traverse(
        bvh,
        rayOrg, rayDir, distMin, distMax,
        stats, debugPrint);
}

template shared::HitObject traverse<2>(
    const GeometryBVH<2> &bvh,
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    TraversalStatistics* const stats, const bool debugPrint);
template shared::HitObject traverse<4>(
    const GeometryBVH<4> &bvh,
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    TraversalStatistics* const stats, const bool debugPrint);
template shared::HitObject traverse<8>(
    const GeometryBVH<8> &bvh,
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    TraversalStatistics* const stats, const bool debugPrint);

}
