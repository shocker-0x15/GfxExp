#pragma once

#include "../nrtdsm_shared.h"

using namespace shared;

CUDA_DEVICE_FUNCTION CUDA_INLINE bool testRayVsTriangle(
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
        ((*hitDist < distMax) & (*hitDist > distMin)
         & (*bcB >= 0.0f) & (*bcC >= 0.0f) & (*bcB + *bcC <= 1));
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D restoreTriangleHitPoint(
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const float bcB, const float bcC, Normal3D* const hitNormal) {
    *hitNormal = static_cast<Normal3D>(cross(pB - pA, pC - pA));
    return (1 - (bcB + bcC)) * pA + bcB * pB + bcC * pC;
}

// Reference: Chapter 8. Cool Patches: A Geometric Approach to Ray/Bilinear Patch Intersections
//            Ray Tracing Gems
CUDA_DEVICE_FUNCTION CUDA_INLINE bool testRayVsBilinearPatch(
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const Point3D &pA, const Point3D &pB, const Point3D &pC, const Point3D &pD,
    float* const hitDist, Normal3D* const hitNormal, float* const u, float* const v) {
    const Vector3D eAB = pB - pA;
    const Vector3D eAC = pC - pA;
    const Vector3D eBD = pD - pB;
    const Vector3D eCD = pD - pC;
    const Vector3D pARel = pA - rayOrg;
    const Vector3D pBRel = pB - rayOrg;

    float u1, u2;
    {
        const Vector3D qN = cross(eAB, -eCD);
        const float a = dot(qN, rayDir);
        float b = dot(cross(pBRel, rayDir), eBD);
        const float c = dot(cross(pARel, rayDir), eAC);
        b -= a + c;

        const float det = pow2(b) - 4 * a * c;
        if (det < 0)
            return false;

        if (a == 0) {
            u1 = -c / b;
            u2 = -1;
        }
        else {
            const float sqrtDet = std::sqrt(det);
            const float temp = -0.5f * (b + std::copysign(sqrtDet, b));
            u1 = temp / a;
            u2 = c / temp;
        }
    }

    *hitDist = distMax;

    const auto find_v_t = [&](const float uu) {
        if (uu >= 0 && uu <= 1) {
            const Vector3D pAB = lerp(pARel, pBRel, uu);
            const Vector3D pCD = lerp(eAC, eBD, uu);
            Vector3D n = cross(rayDir, pCD);
            const float recDet = 1.0f / dot(n, n);
            n = cross(n, pAB);
            const float tt = dot(n, pCD) * recDet;
            const float vv = dot(n, rayDir) * recDet;
            if (vv >= 0 && vv <= 1 && tt > distMin && tt < *hitDist) {
                *hitDist = tt;
                *u = uu;
                *v = vv;
            }
        }
    };

    find_v_t(u1);
    find_v_t(u2);
    if (*hitDist == distMax)
        return false;

    const Vector3D dpdu = lerp(eAB, eCD, *v);
    const Vector3D dpdv = lerp(eAC, eBD, *u);
    *hitNormal = static_cast<Normal3D>(cross(dpdu, dpdv));

    return true;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D restoreBilinearPatchHitPoint(
    const Point3D &pA, const Point3D &pB, const Point3D &pC, const Point3D &pD,
    const float u, const float v, Normal3D* const hitNormal) {
    const Vector3D eAB = pB - pA;
    const Vector3D eAC = pC - pA;
    const Vector3D eBD = pD - pB;
    const Vector3D eCD = pD - pC;
    const Vector3D dpdu = lerp(eAB, eCD, v);
    const Vector3D dpdv = lerp(eAC, eBD, u);
    *hitNormal = static_cast<Normal3D>(cross(dpdu, dpdv));
    return (1 - u) * (1 - v) * pA + u * (1 - v) * pB + (1 - u) * v * pC + u * v * pD;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE bool testRayVsPrism(
    const Point3D &rayOrg, const Vector3D &rayDir, const float distMin, const float distMax,
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const Point3D &pD, const Point3D &pE, const Point3D &pF,
    float* const hitDist, float* const hitParam0, float* const hitParam1,
    bool* isFrontHit) {
    *hitDist = distMax;

    const auto updateHit = [&]
    (const float t, const Normal3D &n, uint32_t faceID, const float u, const float v) {
        *hitDist = t;
        *hitParam0 = faceID + u;
        *hitParam1 = v;
        //*hitNormal = normalize(n);
        *isFrontHit = dot(n, rayDir) < 0;
    };

    float tt;
    Normal3D nn;
    float uu, vv;
    if (testRayVsTriangle(
        rayOrg, rayDir, distMin, *hitDist,
        pC, pB, pA,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 0, uu, vv);
    }
    if (testRayVsTriangle(
        rayOrg, rayDir, distMin, *hitDist,
        pD, pE, pF,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 1, uu, vv);
    }
    if (testRayVsBilinearPatch(
        rayOrg, rayDir, distMin, *hitDist,
        pA, pB, pD, pE,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 2, uu, vv);
    }
    if (testRayVsBilinearPatch(
        rayOrg, rayDir, distMin, *hitDist,
        pB, pC, pE, pF,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 3, uu, vv);
    }
    if (testRayVsBilinearPatch(
        rayOrg, rayDir, distMin, *hitDist,
        pC, pA, pF, pD,
        &tt, &nn, &uu, &vv)) {
        updateHit(tt, nn, 4, uu, vv);
    }

    return *hitDist < distMax;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D restorePrismHitPoint(
    const Point3D &pA, const Point3D &pB, const Point3D &pC,
    const Point3D &pD, const Point3D &pE, const Point3D &pF,
    const float hitParam0, const float hitParam1,
    Normal3D* const hitNormal) {
    const uint32_t faceID = static_cast<uint32_t>(hitParam0);
    const float u = std::fmod(hitParam0, 1.0f);
    const float v = std::fmod(hitParam1, 1.0f);
    if (faceID == 0)
        return restoreTriangleHitPoint(pC, pB, pA, u, v, hitNormal);
    else if (faceID == 1)
        return restoreTriangleHitPoint(pD, pE, pF, u, v, hitNormal);
    else if (faceID == 2)
        return restoreBilinearPatchHitPoint(pA, pB, pD, pE, u, v, hitNormal);
    else if (faceID == 3)
        return restoreBilinearPatchHitPoint(pB, pC, pE, pF, u, v, hitNormal);
    else if (faceID == 4)
        return restoreBilinearPatchHitPoint(pC, pA, pF, pD, u, v, hitNormal);
    return Point3D(NAN);
}



CUDA_DEVICE_KERNEL void RT_IS_NAME(prism)() {
    const auto sbtr = HitGroupSBTRecordData::get();
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];
    const GeometryInstanceDataForNRTDSM &nrtdsmGeomInst = plp.s->geomInstNrtdsmDataBuffer[sbtr.geomInstSlot];

    const uint32_t primIdx = optixGetPrimitiveIndex();
    const Triangle &tri = geomInst.triangleBuffer[primIdx];
    const Vertex &vA = geomInst.vertexBuffer[tri.index0];
    const Vertex &vB = geomInst.vertexBuffer[tri.index1];
    const Vertex &vC = geomInst.vertexBuffer[tri.index2];
    const NRTDSMTriangleAuxInfo &dispTriAuxInfo = nrtdsmGeomInst.dispTriAuxInfoBuffer[primIdx];
    const float minHeight = dispTriAuxInfo.minHeight;
    const float maxHeight = minHeight + dispTriAuxInfo.amplitude;
    const Point3D pA = vA.position + minHeight * vA.normal;
    const Point3D pB = vB.position + minHeight * vB.normal;
    const Point3D pC = vC.position + minHeight * vC.normal;
    const Point3D pD = vA.position + maxHeight * vA.normal;
    const Point3D pE = vB.position + maxHeight * vB.normal;
    const Point3D pF = vC.position + maxHeight * vC.normal;
    float hitDist;
    float hitParam0, hitParam1;
    bool isFrontHit;
    const bool hit = testRayVsPrism(
        Point3D(optixGetObjectRayOrigin()), Vector3D(optixGetObjectRayDirection()),
        optixGetRayTmin(), optixGetRayTmax(),
        pA, pB, pC,
        pD, pE, pF,
        &hitDist, &hitParam0, &hitParam1, &isFrontHit);
    if (!hit)
        return;

    //if (isCursorPixel() && getDebugPrintEnabled()) {
    //    printf("frame %u\n", plp.f->frameIndex);
    //    printf("height: %g, %g\n", minHeight, maxHeight);
    //    printf(
    //        "org: (%g, %g, %g), dir: (%g, %g, %g)\n",
    //        v3print(optixGetObjectRayOrigin()),
    //        v3print(optixGetObjectRayDirection()));
    //    printf("dist: %g - %g\n", optixGetRayTmin(), optixGetRayTmax());
    //    printf(
    //        "(%g, %g, %g), (%g, %g, %g), (%g, %g, %g)\n",
    //        v3print(pA), v3print(pB), v3print(pC));
    //    printf(
    //        "(%g, %g, %g), (%g, %g, %g), (%g, %g, %g)\n",
    //        v3print(pD), v3print(pE), v3print(pF));
    //    printf("params: %g, %g\n", hitParam0, hitParam1);
    //}

    PrismAttributeSignature::reportIntersection(
        hitDist,
        isFrontHit ? CustomHitKind_PrismFrontFace : CustomHitKind_PrismBackFace,
        hitParam0, hitParam1);
}
