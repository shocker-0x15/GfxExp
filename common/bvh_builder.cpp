#include "bvh_builder.h"
#include "common_host.h"

namespace bvh {

enum class PrimitiveType {
	Triangle = 0,
};



template <PrimitiveType primType>
struct BuilderInput;

template <>
struct BuilderInput<PrimitiveType::Triangle> {
	const Geometry* geometries;
	uint32_t numGeometries;
	float splittingBudget;
	float intNodeTravCost;
	float primIntersectCost;
	uint32_t minNumPrimsPerLeaf : 16;
	uint32_t maxNumPrimsPerLeaf : 16;
};



template <uint32_t arity, PrimitiveType primType>
using BVH = GeometryBVH<arity>;



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
	};
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
	uint32_t slotInParent;
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
	uint32_t isSpecialSplit;
};

template <uint32_t arity>
struct TempInternalNode {
	struct Child {
		AABB aabb;
		uint32_t index;
		uint32_t numLeaves;
	} children[arity];
};



void findBestObjectSplit(
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

void findBestSpatialSplit(
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
			for (int32_t binIdx = entryBinIdx; binIdx <= exitBinIdx; ++binIdx)
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



void performPartition(
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

void performObjectSplit(
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

void splitTriangle(
	Point3D pA, Point3D pB, Point3D pC,
	const float splitPlane, const uint32_t splitAxis,
	AABB* const bbA, AABB* const bbB) {
	uint32_t mask = (((pC[splitAxis] >= splitPlane) << 2) |
					 ((pB[splitAxis] >= splitPlane) << 1) |
					 ((pA[splitAxis] >= splitPlane) << 0));
	if (pA[splitAxis] >= splitPlane)
		mask = ~mask & 0b111;
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
	}

	float tAB = (splitPlane - pA[splitAxis]) / (pB[splitAxis] - pA[splitAxis]);
	Point3D pAB = pA + tAB * (pB - pA);
	float tAC = (splitPlane - pA[splitAxis]) / (pC[splitAxis] - pA[splitAxis]);
	Point3D pAC = pA + tAC * (pC - pA);

	AABB aabb;
	aabb.unify(pAB).unify(pAC);
	*bbA = unify(aabb, pA);
	*bbB = unify(aabb, pB).unify(pC);
}

void performSpatialSplit(
	const BuilderInput<PrimitiveType::Triangle> &buildInput,
	const SplitTask &splitTask, const SplitInfo &splitInfo,
	SplitTask* const leftTask, SplitTask* const rightTask) {
	const uint32_t splitDim = splitInfo.dim;
	const uint32_t splitPlaneIdx = splitInfo.planeIndex;
	const float binCoeff = (splitTask.geomAabb.maxP[splitDim] - splitTask.geomAabb.minP[splitDim]) / numSpaBins;
	const float splitPlane = splitTask.geomAabb.minP[splitDim] + (splitPlaneIdx + 1) * binCoeff;

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

		if (entryBinIdx <= splitPlaneIdx && exitBinIdx > splitPlaneIdx &&
			curNumPrims < numPrimsReserved) {
			primSplitInfo.isRight = false;

			const Geometry &geom = buildInput.geometries[primRef.geomIndex];
			const auto tri = reinterpret_cast<const uint32_t*>(
				geom.triangles + geom.triangleStride * primRef.primIndex);

			const Point3D pA = geom.preTransform *
				*reinterpret_cast<const Point3D*>(geom.vertices + geom.vertexStride * tri[0]);
			const Point3D pB = geom.preTransform *
				*reinterpret_cast<const Point3D*>(geom.vertices + geom.vertexStride * tri[1]);
			const Point3D pC = geom.preTransform *
				*reinterpret_cast<const Point3D*>(geom.vertices + geom.vertexStride * tri[2]);

			PrimitiveReference &newPrimRef = splitTask.primRefs[curNumPrims];
			PrimSplitInfo &newPrimSplitInfo = splitTask.primSplitInfos[curNumPrims];
			splitTriangle(pA, pB, pC, splitPlane, splitDim, &primRef.box, &newPrimRef.box);
			newPrimRef.geomIndex = primRef.geomIndex;
			newPrimRef.primIndex = primRef.primIndex;
			newPrimSplitInfo.isRight = true;

			++leftPrimCount;
			++rightPrimCount;
			++curNumPrims;
		}
		else {
			if (entryBinIdx > splitPlaneIdx) {
				primSplitInfo.isRight = true;
				++rightPrimCount;
			}
			else {
				primSplitInfo.isRight = false;
				++leftPrimCount;
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
	BVH<arity, primType>* const bvh)
{
	uint32_t numGlobalPrimitives = 0;
	std::vector<uint32_t> globalPrimOffsets;
	if constexpr (primType == PrimitiveType::Triangle) {
		globalPrimOffsets.resize(buildInput.numGeometries);
		for (uint32_t geomIdx = 0; geomIdx < buildInput.numGeometries; ++geomIdx) {
			const Geometry &geom = buildInput.geometries[geomIdx];
			globalPrimOffsets[geomIdx] = numGlobalPrimitives;
			numGlobalPrimitives += geom.numTriangles;
		}
	}
	else {
		(void)globalPrimOffsets;
	}

	const auto extractGeomAndPrimIndex = [&globalPrimOffsets]
	(const uint32_t globalPrimIdx,
	 uint32_t* const geomIdx, uint32_t* const primIdx) {
		const uint32_t numGeoms = static_cast<uint32_t>(globalPrimOffsets.size());
		*geomIdx = 0;
		for (int d = nextPowerOf2(numGeoms) >> 1; d >= 1; d >>= 1) {
			if (*geomIdx + d >= numGeoms)
				continue;
			if (globalPrimOffsets[*geomIdx + d] <= globalPrimIdx)
				*geomIdx += d;
		}
		*primIdx = globalPrimIdx - globalPrimOffsets[*geomIdx];
	};

	const bool allowPrimRefIncrease = buildInput.splittingBudget > 0.0f;
	const uint32_t numGlobalPrimsAllocated = std::max(
		numGlobalPrimitives,
		static_cast<uint32_t>((1.0f + buildInput.splittingBudget) * numGlobalPrimitives));
	std::vector<PrimitiveReference> primRefsMem(numGlobalPrimsAllocated);
	std::vector<PrimSplitInfo> primSplitInfosMem(numGlobalPrimsAllocated);
	std::span<PrimitiveReference> primRefs = primRefsMem;
	std::span<PrimSplitInfo> primSplitInfos = primSplitInfosMem;
	if constexpr (primType == PrimitiveType::Triangle) {
		for (uint32_t globalPrimIdx = 0; globalPrimIdx < numGlobalPrimitives; ++globalPrimIdx) {
			uint32_t geomIdx, primIdx;
			extractGeomAndPrimIndex(globalPrimIdx, &geomIdx, &primIdx);
			const Geometry &geom = buildInput.geometries[geomIdx];
			const auto tri = reinterpret_cast<const uint32_t*>(
				geom.triangles + geom.triangleStride * primIdx);

			const Point3D pA = geom.preTransform *
				*reinterpret_cast<const Point3D*>(geom.vertices + geom.vertexStride * tri[0]);
			const Point3D pB = geom.preTransform *
				*reinterpret_cast<const Point3D*>(geom.vertices + geom.vertexStride * tri[1]);
			const Point3D pC = geom.preTransform *
				*reinterpret_cast<const Point3D*>(geom.vertices + geom.vertexStride * tri[2]);

			PrimitiveReference primRef = {};
			primRef.box.unify(pA).unify(pB).unify(pC);
			primRef.geomIndex = geomIdx;
			primRef.primIndex = primIdx;
			primRefs[globalPrimIdx] = primRef;
		}
	}
	else {
		Assert_NotImplemented();
	}

	const float intTravCost = buildInput.intNodeTravCost;
	const float primIsectCost = buildInput.primIntersectCost;

	std::vector<TempInternalNode<arity>> tempIntNodes;

	std::vector<SplitTask> stack;
	{
		SplitTask rootTask = {};
		rootTask.geomAabb = AABB();
		rootTask.centAabb = AABB();
		for (uint32_t globalPrimIdx = 0; globalPrimIdx < numGlobalPrimitives; ++globalPrimIdx) {
			const PrimitiveReference &primRef = primRefs[globalPrimIdx];
			rootTask.geomAabb.unify(primRef.box);
			rootTask.centAabb.unify(primRef.box.getCenter());
		}
		rootTask.primRefs = primRefs;
		rootTask.primSplitInfos = primSplitInfos;
		rootTask.numActualElems = numGlobalPrimitives;
		rootTask.parentIndex = UINT32_MAX;
		rootTask.slotInParent = 0;
		rootTask.isSplittable = numGlobalPrimitives > 1;
		stack.push_back(rootTask);
	}

	const float rootSA = stack.back().geomAabb.calcHalfSurfaceArea();

	while (!stack.empty()) {
		const SplitTask task = stack.back();
		stack.pop_back();

		SplitTask children[arity];
		children[0] = task;
		uint32_t numChildren = 1;
		while (numChildren < arity) {
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

			const uint32_t numPrimRefs = taskToSplit.numActualElems;
			const float geomSA = taskToSplit.geomAabb.calcHalfSurfaceArea();
			const float leafCost = geomSA * numPrimRefs * primIsectCost;

			SplitInfo splitInfo;
			findBestObjectSplit(
				taskToSplit.primRefs, taskToSplit.primSplitInfos,
				numPrimRefs, taskToSplit.centAabb,
				&splitInfo);
			float splitCost = geomSA * intTravCost + splitInfo.cost * primIsectCost;
			const bool objSplitSuccess = !std::isinf(splitInfo.cost);

			if constexpr (primType == PrimitiveType::Triangle) {
				if (allowPrimRefIncrease && objSplitSuccess) {
					const AABB overlappedAabb = intersect(splitInfo.leftAabb, splitInfo.rightAabb);
					const float overlappedSA = overlappedAabb.isValid() ?
						overlappedAabb.calcHalfSurfaceArea() : 0.0f;
					constexpr float splittingThreshold = 1e-5f;
					if (overlappedSA / rootSA > splittingThreshold) {
						SplitInfo spaSplitInfo;
						findBestSpatialSplit(
							taskToSplit.primRefs,
							numPrimRefs, taskToSplit.geomAabb,
							&spaSplitInfo);
						const float spaSplitCost = geomSA * intTravCost + spaSplitInfo.cost * primIsectCost;
						if (spaSplitCost < splitCost) {
							splitInfo = spaSplitInfo;
							splitCost = spaSplitCost;
						}
					}
				}
			}
			else {
				Assert_NotImplemented();
			}

			if (leafCost < splitCost &&
				numPrimRefs <= buildInput.maxNumPrimsPerLeaf) {
				children[slotToSplit].isSplittable = false;
				continue;
			}

			SplitTask leftTask, rightTask;
			if (objSplitSuccess) {
				if (splitInfo.isSpecialSplit) {
					if constexpr (primType == PrimitiveType::Triangle) {
						performSpatialSplit(
							buildInput,
							taskToSplit, splitInfo,
							&leftTask, &rightTask);
					}
					else {
						Assert_NotImplemented();
					}
				}
				else {
					performObjectSplit(
						taskToSplit, splitInfo, buildInput.minNumPrimsPerLeaf,
						&leftTask, &rightTask);
				}
			}
			else {
				const auto pred = [&numPrimRefs]
				(uint32_t idx) {
					return idx < numPrimRefs;
				};
				const uint32_t leftPrimCount = numPrimRefs / 2;
				const uint32_t rightPrimCount = numPrimRefs - leftPrimCount;
				performPartition(
					taskToSplit, pred,
					buildInput.minNumPrimsPerLeaf,
					leftPrimCount, rightPrimCount,
					&leftTask, &rightTask);
			}
			children[slotToSplit] = leftTask;
			children[numChildren] = rightTask;

			++numChildren;
		}

		if (numChildren == 1 && task.parentIndex != UINT32_MAX) {
			TempInternalNode<arity> &parentNode = tempIntNodes[task.parentIndex];
			typename TempInternalNode<arity>::Child &selfSlot = parentNode.children[task.slotInParent];
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

		const uint32_t intNodeIdx = static_cast<uint32_t>(tempIntNodes.size());
		if (task.parentIndex != UINT32_MAX) {
			TempInternalNode<arity> &parentNode = tempIntNodes[task.parentIndex];
			typename TempInternalNode<arity>::Child &selfSlot = parentNode.children[task.slotInParent];
			selfSlot.index = intNodeIdx;
		}

		tempIntNodes.resize(tempIntNodes.size() + 1);
		TempInternalNode<arity> &intNode = tempIntNodes[intNodeIdx];

		for (uint32_t slot = 0; slot < numChildren; ++slot) {
			SplitTask &childTask = children[slot];
			typename TempInternalNode<arity>::Child &child = intNode.children[slot];
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
			typename TempInternalNode<arity>::Child &child = intNode.children[slot];
			child.aabb = AABB();
			child.index = UINT32_MAX;
			child.numLeaves = 0;
		}
	}

	// EN: Finished to build the temporary BVH, now we convert it to the final BVH.

	// EN: Create triangle storages.
	std::vector<shared::TriangleStorage> triStorages(numGlobalPrimitives);
	for (uint32_t globalPrimIdx = 0; globalPrimIdx < numGlobalPrimitives; ++globalPrimIdx) {
		uint32_t geomIdx, primIdx;
		extractGeomAndPrimIndex(globalPrimIdx, &geomIdx, &primIdx);
		const Geometry &geom = buildInput.geometries[geomIdx];
		const auto tri = reinterpret_cast<const uint32_t*>(
			geom.triangles + geom.triangleStride * primIdx);

		const Point3D pA = geom.preTransform *
			*reinterpret_cast<const Point3D*>(geom.vertices + geom.vertexStride * tri[0]);
		const Point3D pB = geom.preTransform *
			*reinterpret_cast<const Point3D*>(geom.vertices + geom.vertexStride * tri[1]);
		const Point3D pC = geom.preTransform *
			*reinterpret_cast<const Point3D*>(geom.vertices + geom.vertexStride * tri[2]);

		shared::TriangleStorage &triStorage = triStorages[globalPrimIdx];
		triStorage = {};
		triStorage.pA = pA;
		triStorage.pB = pB;
		triStorage.pC = pC;
		triStorage.geomIndex = geomIdx;
		triStorage.primIndex = primIdx;
	}

	const uint32_t numIntNodes = static_cast<uint32_t>(tempIntNodes.size());

	// Compute mapping from the source tree to the destination tree.
	std::vector<uint32_t> dstIntNodeIndices(numIntNodes);
	std::vector<uint32_t> leafChildBlockIndices(numIntNodes);
	dstIntNodeIndices[0] = 0;
	uint32_t intChildBlockIdx = 1;
	uint32_t leafChildBlockIdx = 0;
	for (uint32_t intNodeIdx = 0; intNodeIdx < numIntNodes; ++intNodeIdx) {
		const TempInternalNode<arity> &intNode = tempIntNodes[intNodeIdx];
		leafChildBlockIndices[intNodeIdx] = leafChildBlockIdx;
		uint32_t intChildCount = 0;
		for (uint32_t slot = 0; slot < arity; ++slot) {
			const TempInternalNode<arity>::Child &child = intNode.children[slot];
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
	std::vector<std::vector<uint32_t>> triToPrimRefMap(numGlobalPrimitives);
#endif

	// EN: Create internal nodes and primitive references.
	const uint32_t numPrimRefs = leafChildBlockIdx;
	std::vector<shared::InternalNode_T<arity>> dstIntNodes(numIntNodes);
	std::vector<shared::PrimitiveReference> dstPrimRefs(numPrimRefs);
	for (uint32_t srcIntNodeIdx = 0; srcIntNodeIdx < numIntNodes; ++srcIntNodeIdx) {
		const uint32_t dstIntNodeIdx = dstIntNodeIndices[srcIntNodeIdx];
		const TempInternalNode<arity> &srcIntNode = tempIntNodes[srcIntNodeIdx];
		shared::InternalNode_T<arity> &dstIntNode = dstIntNodes[dstIntNodeIdx];

		AABB quantizationBox;
		uint32_t leafMask = 0;
		uint32_t firstIntChildSlot = UINT32_MAX;
		uint32_t primRefOffset = leafChildBlockIndices[srcIntNodeIdx];
		for (uint32_t slot = 0; slot < arity; ++slot) {
			const typename TempInternalNode<arity>::Child &srcChild = srcIntNode.children[slot];
			if (srcChild.index == UINT32_MAX)
				break;
			quantizationBox.unify(srcChild.aabb);
			if (srcChild.numLeaves > 0) {
				leafMask |= 1 << slot;

				for (uint32_t primRefIdx = 0; primRefIdx < srcChild.numLeaves; ++primRefIdx) {
					const PrimitiveReference &srcPrimRef = primRefs[srcChild.index + primRefIdx];
					shared::PrimitiveReference &dstPrimRef = dstPrimRefs[primRefOffset + primRefIdx];
					dstPrimRef.storageIndex = globalPrimOffsets[srcPrimRef.geomIndex] + srcPrimRef.primIndex;
					dstPrimRef.isLeafEnd = primRefIdx == srcChild.numLeaves - 1;

#if defined(_DEBUG)
					triToPrimRefMap[dstPrimRef.storageIndex].push_back(srcChild.index + primRefIdx);
#endif
				}
				primRefOffset += srcChild.numLeaves;
			}
			else {
				if (firstIntChildSlot == UINT32_MAX)
					firstIntChildSlot = slot;
			}
		}

		dstIntNode.quantBoxMinP = quantizationBox.minP;
		dstIntNode.quantBoxSizeCoeff = (quantizationBox.maxP - quantizationBox.minP) / 255.0f;
		dstIntNode.leafMask = leafMask;

		if (firstIntChildSlot != UINT32_MAX)
			dstIntNode.intNodeChildBaseIndex = dstIntNodeIndices[srcIntNode.children[firstIntChildSlot].index];
		else
			dstIntNode.intNodeChildBaseIndex = UINT32_MAX;

		if (leafMask)
			dstIntNode.leafBaseIndex = leafChildBlockIndices[srcIntNodeIdx];
		else
			dstIntNode.leafBaseIndex = UINT32_MAX;

		for (uint32_t slot = 0; slot < arity; ++slot) {
			const typename TempInternalNode<arity>::Child &srcChild = srcIntNode.children[slot];
			typename shared::InternalNode_T<arity>::Child &dstChild = dstIntNode.children[slot];
			if (srcChild.index != UINT32_MAX) {
				const Point3D nMinP = quantizationBox.normalize(srcChild.aabb.minP);
				const Point3D nMaxP = quantizationBox.normalize(srcChild.aabb.maxP);
				const uint3 qMinP = make_uint3(255 * nMinP.toNative());
				const uint3 qMaxP = min(make_uint3(255 * nMaxP.toNative()) + 1, 255);
				dstChild.minX = qMinP.x;
				dstChild.minY = qMinP.y;
				dstChild.minZ = qMinP.z;
				dstChild.maxX = qMaxP.x;
				dstChild.maxY = qMaxP.y;
				dstChild.maxZ = qMaxP.z;
			}
			else {
				dstChild.minX = dstChild.minY = dstChild.minZ = 255;
				dstChild.maxX = dstChild.maxY = dstChild.maxZ = 0;
			}
		}
	}

	// Debug Visualization
#if ENABLE_VDB && defined(_DEBUG)
	// Triangle to Primitive References
	if (false) {
		for (uint32_t globalPrimIdx = 0; globalPrimIdx < numGlobalPrimitives; ++globalPrimIdx) {
			const std::vector<uint32_t> &refs = triToPrimRefMap[globalPrimIdx];

			vdb_frame();
			drawAxes(10.0f);

			const shared::TriangleStorage &triStorage = triStorages[globalPrimIdx];
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
#endif

	bvh->intNodes = std::move(dstIntNodes);
	bvh->primRefs = std::move(dstPrimRefs);
	bvh->triStorages = std::move(triStorages);
}



template <uint32_t arity>
void buildGeometryBVH(
	const Geometry* const geoms, const uint32_t numGeoms,
	const GeometryBVHBuildConfig &config, GeometryBVH<arity>* const bvh) {
	BuilderInput<PrimitiveType::Triangle> input = {};
	input.geometries = geoms;
	input.numGeometries = numGeoms;
	input.splittingBudget = config.splittingBudget;
	input.intNodeTravCost = config.intNodeTravCost;
	input.primIntersectCost = config.primIntersectCost;
	input.minNumPrimsPerLeaf = config.minNumPrimsPerLeaf;
	input.maxNumPrimsPerLeaf = config.maxNumPrimsPerLeaf;
	buildBVH(input, bvh);
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

}
