#include "tfdm_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void computeAABBs(
    const GeometryInstanceData* geomInst) {
    const uint32_t primIndex = blockDim.x * blockIdx.x + threadIdx.x;

    const Triangle &tri = geomInst->triangleBuffer[primIndex];
    const Vertex (&vs)[] = {
        geomInst->vertexBuffer[tri.index0],
        geomInst->vertexBuffer[tri.index1],
        geomInst->vertexBuffer[tri.index2]
    };
    const Point3D tcs[] = {
        Point3D(vs[0].texCoord, 1.0f),
        Point3D(vs[1].texCoord, 1.0f),
        Point3D(vs[2].texCoord, 1.0f),
    };

    const Matrix3x3 matTcToBc = invert(Matrix3x3(tcs[0], tcs[1], tcs[2]));
    const Matrix3x3 matBcToP(vs[0].position, vs[1].position, vs[2].position);
    const Matrix3x3 matBcToN(vs[0].normal, vs[1].normal, vs[2].normal);
    const Matrix3x3 matTcToP = matBcToP * matTcToBc;
    const Matrix3x3 matTcToN = matBcToN * matTcToBc;

    for (int vIdx = 0; vIdx < 3; ++vIdx) {
        Point3D center =
            0.5f * tcs[vIdx]
            + 0.25f * tcs[(vIdx + 1) % 3]
            + 0.25f * tcs[(vIdx + 2) % 3];
        //AAFloatOn2D elem0(0, 1, 0, 0);
        //AAFloatOn2D elem1(0, 0, 1, 0);
        //AAFloatOn2D_Vector3D edge0 = 0.25f * (tcs[(vIdx + 1) % 3] - tcs[vIdx]) * elem0;
        //AAFloatOn2D_Vector3D edge1 = 0.25f * (tcs[(vIdx + 2) % 3] - tcs[vIdx]) * elem1;
        AAFloatOn2D_Vector3D edge0(
            0, 0.25f * (tcs[(vIdx + 1) % 3] - tcs[vIdx]), 0, 0);
        AAFloatOn2D_Vector3D edge1(
            0, 0, 0.25f * (tcs[(vIdx + 2) % 3] - tcs[vIdx]), 0);
        AAFloatOn2D_Vector3D texCoord = static_cast<Vector3D>(center) + edge0 + edge1;
    }
}