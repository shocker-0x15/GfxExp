#pragma once

#include "common_shared.h"

namespace bvh {

template <uint32_t arity>
struct GeometryBVH {
    std::vector<shared::InternalNode_T<arity>> intNodes;
    std::vector<shared::PrimitiveReference> primRefs;
    std::vector<shared::TriangleStorage> triStorages;
    uint32_t numGeoms;
    uint32_t totalNumPrims;
};

struct Geometry {
    const uint8_t* vertices;
    uint32_t vertexStride;
    uint32_t numVertices;
    const uint8_t* triangles;
    uint32_t triangleStride;
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
    uint32_t maxStackDepth;
};

template <uint32_t arity>
shared::HitObject traverse(
    const GeometryBVH<arity> &bvh,
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    TraversalStatistics* const stats = nullptr);

}
