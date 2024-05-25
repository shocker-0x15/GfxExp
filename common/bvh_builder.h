#pragma once

#include "common_shared.h"

namespace bvh {

template <uint32_t arity>
struct GeometryBVH {
    std::vector<shared::InternalNode_T<arity>> intNodes;
    std::vector<shared::TriangleStorage> triStorages;
    std::vector<shared::PrimitiveReference> primRefs;
    uint32_t numGeoms;
    uint32_t totalNumPrims;
};

enum class VertexFormat {
    Fp32x3 = 0,
};

enum class TriangleFormat {
    UI32x3 = 0,
    UI16x3,
};

struct Geometry {
    const void* vertices;
    uint32_t vertexStride;
    VertexFormat vertexFormat;
    uint32_t numVertices;
    const void* triangles;
    uint32_t triangleStride;
    TriangleFormat triangleFormat;
    uint32_t numTriangles;
    Matrix4x4 preTransform;
};

struct GeometryBVHBuildConfig {
    float splittingBudget;
    float intNodeTravCost;
    float primIntersectCost;
    uint32_t minNumPrimsPerLeaf;
    uint32_t maxNumPrimsPerLeaf;
};

template <uint32_t arity>
void buildGeometryBVH(
    const Geometry* const geoms, const uint32_t numGeoms,
    const GeometryBVHBuildConfig &config, GeometryBVH<arity>* const bvh);



template <uint32_t arity>
struct InstanceBVH {
    std::vector<shared::InternalNode_T<arity>> intNodes;
    std::vector<shared::InstanceReference> instRefs;
    uint32_t numInsts;
};

struct Instance {
    Matrix3x3 rotFromObj;
    Vector3D transFromObj;
    uintptr_t bvhAddress;
    uint32_t userData;
};

struct InstanceBVHBuildConfig {
    float rebraidingBudget;
};

template <uint32_t arity>
void buildInstanceBVH(
    const Instance* const insts, const uint32_t numInsts,
    const InstanceBVHBuildConfig &config, InstanceBVH<arity>* const bvh);



struct TraversalStatistics {
    uint32_t numAabbTests;
    uint32_t numTriTests;
    float avgStackAccessDepth;
    int32_t maxStackDepth;
    int32_t fastStackDepthLimit; // input
    uint32_t stackMemoryAccessAmount;
};

template <uint32_t arity>
shared::HitObject traverse(
    const GeometryBVH<arity> &bvh,
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    TraversalStatistics* const stats = nullptr, const bool debugPrint = false);

}
