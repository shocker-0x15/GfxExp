#include "../restir_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility);
}

static constexpr bool useMIS_RIS = true;



template <bool withTemporalRIS, bool withSpatialRIS, bool useUnbiasedEstimator>
CUDA_DEVICE_FUNCTION CUDA_INLINE void traceShadowRays() {
    int2 launchIndex = make_int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    uint32_t curBufIdx = plp.f->bufferIndex;
    uint32_t prevBufIdx;
    optixu::BlockBuffer2D<Reservoir<LightSample>, 0> curReservoirs =
        plp.s->reservoirBuffer[plp.currentReservoirIndex];
    optixu::BlockBuffer2D<Reservoir<LightSample>, 0> prevReservoirs;
    optixu::NativeBlockBuffer2D<SampleVisibility> curSampleVisBuffer =
        plp.s->sampleVisibilityBuffer[curBufIdx];
    optixu::NativeBlockBuffer2D<SampleVisibility> prevSampleVisBuffer;
    GBuffer0 gBuffer0 = plp.s->GBuffer0[curBufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[curBufIdx].read(launchIndex);
    GBuffer2 gBuffer2 = plp.s->GBuffer2[curBufIdx].read(launchIndex);
    if constexpr (withTemporalRIS || withSpatialRIS) {
        prevBufIdx = (curBufIdx + 1) % 2;
        prevReservoirs = plp.s->reservoirBuffer[(plp.currentReservoirIndex + 1) % 2];
        prevSampleVisBuffer = plp.s->sampleVisibilityBuffer[prevBufIdx];
    }
    else {
        (void)prevBufIdx;
        (void)prevReservoirs;
        (void)prevSampleVisBuffer;
    }

    Point3D positionInWorld = gBuffer0.positionInWorld;
    Normal3D shadingNormalInWorld = gBuffer1.normalInWorld;
    uint32_t materialSlot = gBuffer2.materialSlot;

    if (materialSlot == 0xFFFFFFFF)
        return;

    // TODO?: Use true geometric normal.
    Normal3D geometricNormalInWorld = shadingNormalInWorld;
    Vector3D vOut = plp.f->camera.position - positionInWorld;
    float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

    ReferenceFrame shadingFrame(shadingNormalInWorld);
    positionInWorld = offsetRayOriginNaive(positionInWorld, frontHit * geometricNormalInWorld);
    float dist = length(vOut);

    SampleVisibility sampleVis;

    // New sample for the current pixel
    LightSample newSample;
    bool newSampleIsValid;
    {
        const Reservoir<LightSample> /*&*/reservoir = curReservoirs[launchIndex];
        newSample = reservoir.getSample();
        newSampleIsValid = reservoir.getSumWeights() > 0.0f;
        if (newSampleIsValid)
            sampleVis.newSample = evaluateVisibility<ReSTIRRayType>(positionInWorld, newSample);
    }

    // Temporal Sample
    int2 tNbCoord;
    Point3D tNbPositionInWorld;
    bool temporalSampleIsValid;
    if constexpr (withTemporalRIS) {
        Vector2D motionVector = gBuffer2.motionVector;
        tNbCoord = make_int2(launchIndex.x + 0.5f - motionVector.x,
                             launchIndex.y + 0.5f - motionVector.y);
        // JP: 隣接ピクセルのジオメトリ・マテリアルがあまりに異なる場合に候補サンプルを再利用すると
        //     バイアスが増えてしまうため、そのようなピクセルは棄却する。
        // EN: Reusing candidates from neighboring pixels with substantially different geometry/material
        //     leads to increased bias. Reject such a pixel.
        sampleVis.temporalPassedHeuristic =
            testNeighbor<true>(prevBufIdx, tNbCoord, dist, shadingNormalInWorld);
        if (sampleVis.temporalPassedHeuristic) {
            Reservoir<LightSample> neighbor;
            LightSample temporalSample;
            if (plp.f->reuseVisibilityForTemporal && !useUnbiasedEstimator) {
                SampleVisibility prevSampleVis = prevSampleVisBuffer.read(tNbCoord);
                sampleVis.temporalSample = prevSampleVis.selectedSample;
            }
            else {
                neighbor = prevReservoirs[tNbCoord];
                temporalSample = neighbor.getSample();
                temporalSampleIsValid = neighbor.getSumWeights() > 0.0f;
                if (temporalSampleIsValid)
                    sampleVis.temporalSample =
                        evaluateVisibility<ReSTIRRayType>(positionInWorld, temporalSample);
            }

            if constexpr (useUnbiasedEstimator) {
                GBuffer0 nbGBuffer0 = plp.s->GBuffer0[prevBufIdx].read(tNbCoord);
                GBuffer1 nbGBuffer1 = plp.s->GBuffer1[prevBufIdx].read(tNbCoord);
                Point3D nbPositionInWorld = nbGBuffer0.positionInWorld;
                Normal3D nbShadingNormalInWorld = nbGBuffer1.normalInWorld;

                // TODO?: Use true geometric normal.
                Normal3D nbGeometricNormalInWorld = nbShadingNormalInWorld;
                Vector3D nbVOut = plp.f->prevCamera.position - nbPositionInWorld;
                float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
                tNbPositionInWorld = offsetRayOriginNaive(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);

                if (newSampleIsValid)
                    sampleVis.newSampleOnTemporal =
                        evaluateVisibility<ReSTIRRayType>(tNbPositionInWorld, newSample);

                if (temporalSampleIsValid)
                    sampleVis.temporalSampleOnCurrent =
                        evaluateVisibility<ReSTIRRayType>(positionInWorld, temporalSample);
            }
            else {
                (void)tNbPositionInWorld;
            }
        }
    }
    else {
        (void)tNbCoord;
        (void)tNbPositionInWorld;
        (void)temporalSampleIsValid;
    }

    // Spatiotemporal Sample
    int2 stNbCoord;
    Point3D stNbPositionInWorld;
    bool spatiotemporalSampleIsValid;
    if constexpr (withSpatialRIS) {
        // JP: 周辺ピクセルの座標をランダムに決定。
        // EN: Randomly determine the coordinates of a neighboring pixel.
        float radius = plp.f->spatialNeighborRadius;
        float deltaX, deltaY;
        if (plp.f->useLowDiscrepancyNeighbors) {
            uint32_t deltaIndex = plp.spatialNeighborBaseIndex +
                5 * launchIndex.x + 7 * launchIndex.y;
            Vector2D delta = plp.s->spatialNeighborDeltas[deltaIndex % 1024];
            deltaX = radius * delta.x;
            deltaY = radius * delta.y;
        }
        else {
            PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);
            radius *= std::sqrt(rng.getFloat0cTo1o());
            float angle = 2 * Pi * rng.getFloat0cTo1o();
            deltaX = radius * std::cos(angle);
            deltaY = radius * std::sin(angle);
            // JP: シェーディング時に同じ近傍を得るためにRNGのステート変化は保存しない。
            // EN: Not store RNG's state changes to get the same neighbor when shading.
        }
        stNbCoord = make_int2(launchIndex.x + 0.5f + deltaX,
                              launchIndex.y + 0.5f + deltaY);

        // JP: 隣接ピクセルのジオメトリ・マテリアルがあまりに異なる場合に候補サンプルを再利用すると
        //     バイアスが増えてしまうため、そのようなピクセルは棄却する。
        // EN: Reusing candidates from neighboring pixels with substantially different geometry/material
        //     leads to increased bias. Reject such a pixel.
        sampleVis.spatiotemporalPassedHeuristic =
            testNeighbor<true>(prevBufIdx, stNbCoord, dist, shadingNormalInWorld);
        sampleVis.spatiotemporalPassedHeuristic &= stNbCoord.x != launchIndex.x || stNbCoord.y != launchIndex.y;
        if (sampleVis.spatiotemporalPassedHeuristic) {
            bool reused = false;
            if (plp.f->reuseVisibilityForSpatiotemporal && !useUnbiasedEstimator) {
                float threshold2 = pow2(plp.f->radiusThresholdForSpatialVisReuse);
                float dist2 = pow2(deltaX) + pow2(deltaY);
                reused = dist2 < threshold2;
            }

            Reservoir<LightSample> neighbor;
            LightSample spatiotemporalSample;
            if (reused) {
                SampleVisibility prevSampleVis = prevSampleVisBuffer.read(stNbCoord);
                sampleVis.spatiotemporalSample = prevSampleVis.selectedSample;
            }
            else {
                neighbor = prevReservoirs[stNbCoord];
                spatiotemporalSample = neighbor.getSample();
                spatiotemporalSampleIsValid = neighbor.getSumWeights() > 0.0f;
                if (spatiotemporalSampleIsValid)
                    sampleVis.spatiotemporalSample =
                       evaluateVisibility<ReSTIRRayType>(positionInWorld, spatiotemporalSample);
            }

            if constexpr (useUnbiasedEstimator) {
                GBuffer0 nbGBuffer0 = plp.s->GBuffer0[prevBufIdx].read(stNbCoord);
                GBuffer1 nbGBuffer1 = plp.s->GBuffer1[prevBufIdx].read(stNbCoord);
                Point3D nbPositionInWorld = nbGBuffer0.positionInWorld;
                Normal3D nbShadingNormalInWorld = nbGBuffer1.normalInWorld;

                // TODO?: Use true geometric normal.
                Normal3D nbGeometricNormalInWorld = nbShadingNormalInWorld;
                Vector3D nbVOut = plp.f->prevCamera.position - nbPositionInWorld;
                float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
                stNbPositionInWorld = offsetRayOriginNaive(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);

                if (newSampleIsValid)
                    sampleVis.newSampleOnSpatiotemporal =
                        evaluateVisibility<ReSTIRRayType>(stNbPositionInWorld, newSample);

                if (spatiotemporalSampleIsValid)
                    sampleVis.spatiotemporalSampleOnCurrent =
                        evaluateVisibility<ReSTIRRayType>(positionInWorld, spatiotemporalSample);
            }
            else {
                (void)stNbPositionInWorld;
            }
        }
    }
    else {
        (void)stNbCoord;
        (void)stNbPositionInWorld;
        (void)spatiotemporalSampleIsValid;
    }

    if constexpr (useUnbiasedEstimator && withTemporalRIS && withSpatialRIS) {
        if (sampleVis.temporalPassedHeuristic && sampleVis.spatiotemporalPassedHeuristic) {
            if (temporalSampleIsValid) {
                const Reservoir<LightSample> /*&*/tNeighbor = prevReservoirs[tNbCoord];
                sampleVis.temporalSampleOnSpatiotemporal =
                    evaluateVisibility<ReSTIRRayType>(stNbPositionInWorld, tNeighbor.getSample());
            }
            if (spatiotemporalSampleIsValid) {
                const Reservoir<LightSample> /*&*/stNeighbor = prevReservoirs[stNbCoord];
                sampleVis.spatiotemporalSampleOnTemporal =
                    evaluateVisibility<ReSTIRRayType>(tNbPositionInWorld, stNeighbor.getSample());
            }
        }
    }

    curSampleVisBuffer.write(launchIndex, sampleVis);
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRays)() {
    traceShadowRays<false, false, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRaysWithTemporalReuseBiased)() {
    traceShadowRays<true, false, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRaysWithSpatialReuseBiased)() {
    traceShadowRays<false, true, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRaysWithSpatioTemporalReuseBiased)() {
    traceShadowRays<true, true, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRaysWithTemporalReuseUnbiased)() {
    traceShadowRays<true, false, true>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRaysWithSpatialReuseUnbiased)() {
    traceShadowRays<false, true, true>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(traceShadowRaysWithSpatioTemporalReuseUnbiased)() {
    traceShadowRays<true, true, true>();
}



enum class SampleType {
    New = 0,
    Temporal,
    Spatiotemporal
};

template <SampleType sampleType, bool withTemporalRIS, bool withSpatialRIS>
CUDA_DEVICE_FUNCTION CUDA_INLINE float computeMISWeight(
    const int2 &launchIndex, uint32_t prevBufIdx, const optixu::BlockBuffer2D<Reservoir<LightSample>, 0> &prevReservoirs,
    uint32_t maxPrevStreamLength, const SampleVisibility &sampleVis,
    uint32_t selfStreamLength, const Point3D &positionInWorld, const Vector3D &vOutLocal,
    const ReferenceFrame &shadingFrame, const BSDF &bsdf,
    const int2 &tNbCoord, const int2 &stNbCoord,
    uint32_t streamLength, const LightSample &lightSample, float sampleTargetDensity) {
    float numMisWeight = sampleTargetDensity;
    float denomMisWeight = numMisWeight * streamLength;

    if constexpr (sampleType != SampleType::New) {
        // JP: 与えられたサンプルを現在のシェーディング点で得る確率密度を計算する。
        // EN: Compute a probability density to get the given sample at the current shading point.
        RGB cont = performDirectLighting<ReSTIRRayType, false>(
            positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample);
        float targetDensity = convertToWeight(cont);
        if (plp.f->useUnbiasedEstimator) {
            targetDensity *= sampleType == SampleType::Temporal ?
                sampleVis.temporalSampleOnCurrent :
                sampleVis.spatiotemporalSampleOnCurrent;
        }

        if constexpr (useMIS_RIS) {
            denomMisWeight += targetDensity * selfStreamLength;
        }
        else {
            if (targetDensity > 0.0f)
                denomMisWeight += selfStreamLength;
        }
    }

    if constexpr (sampleType != SampleType::Temporal && withTemporalRIS) {
        if (sampleVis.temporalPassedHeuristic) {
            // JP: 前のフレームで対応するシェーディング点の情報を復元する。
            // EN: Reconstruct the information of the corresponding shading point in the previous frame.
            Point3D nbPositionInWorld;
            Vector3D nbVOutLocal;
            ReferenceFrame nbShadingFrame;
            BSDF nbBsdf;
            {
                GBuffer0 nbGBuffer0 = plp.s->GBuffer0[prevBufIdx].read(tNbCoord);
                GBuffer1 nbGBuffer1 = plp.s->GBuffer1[prevBufIdx].read(tNbCoord);
                GBuffer2 nbGBuffer2 = plp.s->GBuffer2[prevBufIdx].read(tNbCoord);
                nbPositionInWorld = nbGBuffer0.positionInWorld;
                Normal3D nbShadingNormalInWorld = nbGBuffer1.normalInWorld;
                Point2D nbTexCoord(nbGBuffer0.texCoord_x, nbGBuffer1.texCoord_y);
                uint32_t nbMaterialSlot = nbGBuffer2.materialSlot;

                const MaterialData &nbMat = plp.s->materialDataBuffer[nbMaterialSlot];

                // TODO?: Use true geometric normal.
                Normal3D nbGeometricNormalInWorld = nbShadingNormalInWorld;
                Vector3D nbVOut = plp.f->prevCamera.position - nbPositionInWorld;
                float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

                nbBsdf.setup(nbMat, nbTexCoord, 0.0f);
                nbShadingFrame = ReferenceFrame(nbShadingNormalInWorld);
                nbPositionInWorld = offsetRayOriginNaive(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);
                float nbDist = length(nbVOut);
                nbVOut /= nbDist;
                nbVOutLocal = nbShadingFrame.toLocal(nbVOut);
            }

            // JP: 与えられたサンプルを前のフレームで対応するシェーディング点で得る確率密度を計算する。
            // EN: Compute a probability density to get the given sample at the corresponding shading point
            //     in the previous frame.
            RGB cont = performDirectLighting<ReSTIRRayType, false>(
                nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, lightSample);
            float nbTargetDensity = convertToWeight(cont);
            if (plp.f->useUnbiasedEstimator) {
                nbTargetDensity *= sampleType == SampleType::New ?
                    sampleVis.newSampleOnTemporal :
                    sampleVis.spatiotemporalSampleOnTemporal;
            }

            const Reservoir<LightSample> /*&*/neighbor = prevReservoirs[tNbCoord];
            uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
            if constexpr (useMIS_RIS) {
                denomMisWeight += nbTargetDensity * nbStreamLength;
            }
            else {
                if (nbTargetDensity > 0.0f)
                    denomMisWeight += nbStreamLength;
            }
        }
    }

    if constexpr (sampleType != SampleType::Spatiotemporal && withSpatialRIS) {
        if (sampleVis.spatiotemporalPassedHeuristic) {
            // JP: 近傍のシェーディング点(前フレーム)の情報を復元する。
            // EN: Reconstruct the information of a shading point on the neighbor (in the previous frame).
            Point3D nbPositionInWorld;
            Vector3D nbVOutLocal;
            ReferenceFrame nbShadingFrame;
            BSDF nbBsdf;
            {
                GBuffer0 nbGBuffer0 = plp.s->GBuffer0[prevBufIdx].read(stNbCoord);
                GBuffer1 nbGBuffer1 = plp.s->GBuffer1[prevBufIdx].read(stNbCoord);
                GBuffer2 nbGBuffer2 = plp.s->GBuffer2[prevBufIdx].read(stNbCoord);
                nbPositionInWorld = nbGBuffer0.positionInWorld;
                Normal3D nbShadingNormalInWorld = nbGBuffer1.normalInWorld;
                Point2D nbTexCoord(nbGBuffer0.texCoord_x, nbGBuffer1.texCoord_y);
                uint32_t nbMaterialSlot = nbGBuffer2.materialSlot;

                const MaterialData &nbMat = plp.s->materialDataBuffer[nbMaterialSlot];

                // TODO?: Use true geometric normal.
                Normal3D nbGeometricNormalInWorld = nbShadingNormalInWorld;
                Vector3D nbVOut = plp.f->prevCamera.position - nbPositionInWorld;
                float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

                nbBsdf.setup(nbMat, nbTexCoord, 0.0f);
                nbShadingFrame = ReferenceFrame(nbShadingNormalInWorld);
                nbPositionInWorld = offsetRayOriginNaive(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);
                float nbDist = length(nbVOut);
                nbVOut /= nbDist;
                nbVOutLocal = nbShadingFrame.toLocal(nbVOut);
            }

            // JP: 与えられたサンプルを近傍のシェーディング点(前フレーム)で得る確率密度を計算する。
            // EN: Compute a probability density to get the given sample at a shading point on the neighbor
            //     (in the previous frame).
            RGB cont = performDirectLighting<ReSTIRRayType, false>(
                nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, lightSample);
            float nbTargetDensity = convertToWeight(cont);
            if (plp.f->useUnbiasedEstimator) {
                nbTargetDensity *= sampleType == SampleType::New ?
                    sampleVis.newSampleOnSpatiotemporal :
                    sampleVis.temporalSampleOnSpatiotemporal;
            }

            const Reservoir<LightSample> /*&*/neighbor = prevReservoirs[stNbCoord];
            uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
            if constexpr (useMIS_RIS) {
                denomMisWeight += nbTargetDensity * nbStreamLength;
            }
            else {
                if (nbTargetDensity > 0.0f)
                    denomMisWeight += nbStreamLength;
            }
        }
    }

    return numMisWeight / denomMisWeight;
}

template <bool withTemporalRIS, bool withSpatialRIS>
CUDA_DEVICE_FUNCTION CUDA_INLINE void shadeAndResample() {
    int2 launchIndex = make_int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    uint32_t curBufIdx = plp.f->bufferIndex;
    uint32_t prevBufIdx;
    optixu::BlockBuffer2D<Reservoir<LightSample>, 0> curReservoirs = plp.s->reservoirBuffer[plp.currentReservoirIndex];
    optixu::BlockBuffer2D<Reservoir<LightSample>, 0> prevReservoirs;
    optixu::NativeBlockBuffer2D<ReservoirInfo> curReservoirInfos = plp.s->reservoirInfoBuffer[plp.currentReservoirIndex];
    optixu::NativeBlockBuffer2D<ReservoirInfo> prevReservoirInfos;
    optixu::NativeBlockBuffer2D<SampleVisibility> curSampleVisBuffer = plp.s->sampleVisibilityBuffer[curBufIdx];
    GBuffer0 gBuffer0 = plp.s->GBuffer0[curBufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[curBufIdx].read(launchIndex);
    GBuffer2 gBuffer2 = plp.s->GBuffer2[curBufIdx].read(launchIndex);
    if constexpr (withTemporalRIS || withSpatialRIS) {
        prevBufIdx = (curBufIdx + 1) % 2;
        uint32_t prevResIndex = (plp.currentReservoirIndex + 1) % 2;
        prevReservoirs = plp.s->reservoirBuffer[prevResIndex];
        prevReservoirInfos = plp.s->reservoirInfoBuffer[prevResIndex];
    }
    else {
        (void)prevBufIdx;
        (void)prevReservoirs;
        (void)prevReservoirInfos;
    }

    Point2D texCoord(gBuffer0.texCoord_x, gBuffer1.texCoord_y);
    uint32_t materialSlot = gBuffer2.materialSlot;

    RGB contribution(0.01f, 0.01f, 0.01f);
    if (materialSlot != 0xFFFFFFFF) {
        PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

        Point3D positionInWorld = gBuffer0.positionInWorld;
        Normal3D shadingNormalInWorld = gBuffer1.normalInWorld;

        int2 tNbCoord;
        if constexpr (withTemporalRIS) {
            Vector2D motionVector = gBuffer2.motionVector;
            tNbCoord = make_int2(launchIndex.x + 0.5f - motionVector.x,
                                 launchIndex.y + 0.5f - motionVector.y);
        }
        else {
            (void)tNbCoord;
        }
        int2 stNbCoord;
        if constexpr (withSpatialRIS) {
            // JP: 周辺ピクセルの座標をランダムに決定。
            // EN: Randomly determine the coordinates of a neighboring pixel.
            float radius = plp.f->spatialNeighborRadius;
            float deltaX, deltaY;
            if (plp.f->useLowDiscrepancyNeighbors) {
                uint32_t deltaIndex = plp.spatialNeighborBaseIndex +
                    5 * launchIndex.x + 7 * launchIndex.y;
                Vector2D delta = plp.s->spatialNeighborDeltas[deltaIndex % 1024];
                deltaX = radius * delta.x;
                deltaY = radius * delta.y;
            }
            else {
                radius *= std::sqrt(rng.getFloat0cTo1o());
                float angle = 2 * Pi * rng.getFloat0cTo1o();
                deltaX = radius * std::cos(angle);
                deltaY = radius * std::sin(angle);
            }
            stNbCoord = make_int2(launchIndex.x + 0.5f + deltaX,
                                  launchIndex.y + 0.5f + deltaY);
        }
        else {
            (void)stNbCoord;
        }

        const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

        // TODO?: Use true geometric normal.
        Normal3D geometricNormalInWorld = shadingNormalInWorld;
        Vector3D vOut = plp.f->camera.position - positionInWorld;
        float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

        BSDF bsdf;
        bsdf.setup(mat, texCoord, 0.0f);
        ReferenceFrame shadingFrame(shadingNormalInWorld);
        positionInWorld = offsetRayOriginNaive(positionInWorld, frontHit * geometricNormalInWorld);
        float dist = length(vOut);
        vOut /= dist;
        Vector3D vOutLocal = shadingFrame.toLocal(vOut);

        // JP: 光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from a light source directly seeing.
        contribution = RGB(0.0f);
        if (vOutLocal.z > 0) {
            RGB emittance(0.0f, 0.0f, 0.0f);
            if (mat.emittance) {
                float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
                emittance = RGB(getXYZ(texValue));
            }
            contribution += emittance / Pi;
        }

        SampleVisibility sampleVis = curSampleVisBuffer.read(launchIndex);

        float selectedTargetDensity = 0.0f;
        Reservoir<LightSample> combinedReservoir;
        uint32_t combinedStreamLength = 0;
        combinedReservoir.initialize();

        RGB directCont(0.0f, 0.0f, 0.0f);
        float selectedMisWeight = 0.0f;

        const Reservoir<LightSample> /*&*/selfRes = curReservoirs[launchIndex];
        const ReservoirInfo selfResInfo = curReservoirInfos.read(launchIndex);
        uint32_t selfStreamLength = selfRes.getStreamLength();
        uint32_t maxPrevStreamLength;
        if constexpr (withTemporalRIS || withSpatialRIS)
            maxPrevStreamLength = 20 * selfStreamLength;
        else
            (void)maxPrevStreamLength;

        // New sample for the current pixel.
        {
            if (selfResInfo.recPDFEstimate > 0.0f && sampleVis.newSample) {
                LightSample lightSample = selfRes.getSample();
                RGB cont = performDirectLighting<ReSTIRRayType, false>(
                    positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample);
                float targetDensity = convertToWeight(cont);

                float misWeight;
                if constexpr (withTemporalRIS || withSpatialRIS)
                    misWeight = computeMISWeight<SampleType::New, withTemporalRIS, withSpatialRIS>(
                        launchIndex, prevBufIdx, prevReservoirs,
                        maxPrevStreamLength, sampleVis,
                        selfStreamLength, positionInWorld, vOutLocal, shadingFrame, bsdf,
                        tNbCoord, stNbCoord,
                        selfStreamLength, lightSample, selfResInfo.targetDensity);
                else
                    misWeight = 1.0f / selfStreamLength;

                float weight = selfRes.getSumWeights();
                directCont += (misWeight * selfResInfo.recPDFEstimate * selfStreamLength) * cont;
                combinedReservoir = selfRes;
                selectedTargetDensity = targetDensity;
                selectedMisWeight = misWeight;
                sampleVis.selectedSample = sampleVis.newSample;
            }
            combinedStreamLength = selfStreamLength;
        }

        // Temporal Sample
        if constexpr (withTemporalRIS) {
            if (sampleVis.temporalPassedHeuristic) {
                const Reservoir<LightSample> /*&*/neighbor = prevReservoirs[tNbCoord];
                const ReservoirInfo neighborInfo = prevReservoirInfos.read(tNbCoord);
                // JP: 際限なく過去フレームで得たサンプルがウェイトを増やさないように、
                //     前フレームのストリーム長を、現在フレームのReservoirに対して20倍までに制限する。
                // EN: Limit the stream length of the previous frame by 20 times of that of the current frame
                //     in order to avoid a sample obtained in the past getting an unlimited weight.
                uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
                if (neighborInfo.recPDFEstimate > 0.0f) {
                    // JP: 隣接ピクセルが持つ候補サンプルの「現在の」ピクセルにおける確率密度を計算する。
                    // EN: Calculate the probability density at the "current" pixel of the candidate sample
                    //     the neighboring pixel holds.
                    LightSample nbLightSample = neighbor.getSample();
                    RGB cont = performDirectLighting<ReSTIRRayType, false>(
                        positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
                    float targetDensity = convertToWeight(cont);

                    float misWeight = computeMISWeight<SampleType::Temporal, withTemporalRIS, withSpatialRIS>(
                        launchIndex, prevBufIdx, prevReservoirs,
                        maxPrevStreamLength, sampleVis,
                        selfStreamLength, positionInWorld, vOutLocal, shadingFrame, bsdf,
                        tNbCoord, stNbCoord,
                        nbStreamLength, nbLightSample, neighborInfo.targetDensity);

                    float weight = targetDensity * neighborInfo.recPDFEstimate * nbStreamLength;
                    directCont += (sampleVis.temporalSample * misWeight * neighborInfo.recPDFEstimate * nbStreamLength) * cont;
                    if (combinedReservoir.update(nbLightSample, weight, rng.getFloat0cTo1o())) {
                        selectedTargetDensity = targetDensity;
                        selectedMisWeight = misWeight;
                        sampleVis.selectedSample = sampleVis.temporalSample;
                    }
                }
                combinedStreamLength += nbStreamLength;
            }
        }

        // Spatiotemporal Sample
        if constexpr (withSpatialRIS) {
            if (sampleVis.spatiotemporalPassedHeuristic) {
                const Reservoir<LightSample> /*&*/neighbor = prevReservoirs[stNbCoord];
                const ReservoirInfo neighborInfo = prevReservoirInfos.read(stNbCoord);
                // JP: 際限なく過去フレームで得たサンプルがウェイトを増やさないように、
                //     前フレームのストリーム長を、現在フレームのReservoirに対して20倍までに制限する。
                // EN: Limit the stream length of the previous frame by 20 times of that of the current frame
                //     in order to avoid a sample obtained in the past getting an unlimited weight.
                uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
                if (neighborInfo.recPDFEstimate > 0.0f) {
                    // JP: 隣接ピクセルが持つ候補サンプルの「現在の」ピクセルにおける確率密度を計算する。
                    // EN: Calculate the probability density at the "current" pixel of the candidate sample
                    //     the neighboring pixel holds.
                    LightSample nbLightSample = neighbor.getSample();
                    RGB cont = performDirectLighting<ReSTIRRayType, false>(
                        positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
                    float targetDensity = convertToWeight(cont);

                    float misWeight = computeMISWeight<SampleType::Spatiotemporal, withTemporalRIS, withSpatialRIS>(
                        launchIndex, prevBufIdx, prevReservoirs,
                        maxPrevStreamLength, sampleVis,
                        selfStreamLength, positionInWorld, vOutLocal, shadingFrame, bsdf,
                        tNbCoord, stNbCoord,
                        nbStreamLength, nbLightSample, neighborInfo.targetDensity);

                    // JP: 隣接ピクセルと現在のピクセルではターゲットPDFが異なるためサンプルはウェイトを持つ。
                    // EN: The sample has a weight since the target PDFs of the neighboring pixel and the current
                    //     are the different.
                    float weight = targetDensity * neighborInfo.recPDFEstimate * nbStreamLength;
                    directCont +=
                        (sampleVis.spatiotemporalSample * misWeight * neighborInfo.recPDFEstimate * nbStreamLength) * cont;
                    if (combinedReservoir.update(nbLightSample, weight, rng.getFloat0cTo1o())) {
                        selectedTargetDensity = targetDensity;
                        selectedMisWeight = misWeight;
                        sampleVis.selectedSample = sampleVis.spatiotemporalSample;
                    }
                }
                combinedStreamLength += nbStreamLength;
            }
        }

        combinedReservoir.setStreamLength(combinedStreamLength);
        contribution += directCont;

        // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
        // EN: Calculate the estimate of the reciprocal of the probability density that the current sample survives.
        float recPDFEstimate = selectedMisWeight * combinedReservoir.getSumWeights() / selectedTargetDensity;
        if (!isfinite(recPDFEstimate) || (plp.f->reuseVisibility && !sampleVis.selectedSample)) {
            recPDFEstimate = 0.0f;
            selectedTargetDensity = 0.0f;
        }

        ReservoirInfo reservoirInfo;
        reservoirInfo.recPDFEstimate = recPDFEstimate;
        reservoirInfo.targetDensity = selectedTargetDensity;

        curSampleVisBuffer.write(launchIndex, sampleVis);
        curReservoirs[launchIndex] = combinedReservoir;
        curReservoirInfos.write(launchIndex, reservoirInfo);
        plp.s->rngBuffer.write(launchIndex, rng);
    }
    else {
        // JP: 環境光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from the environmental light source directly seeing.
        if (plp.s->envLightTexture && plp.f->enableEnvLight) {
            float u = texCoord.x, v = texCoord.y;
            float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, u, v, 0.0f);
            RGB luminance = plp.f->envLightPowerCoeff * RGB(getXYZ(texValue));
            contribution = luminance;
        }
    }

    RGB prevColorResult(0.0f, 0.0f, 0.0f);
    if (plp.f->numAccumFrames > 0)
        prevColorResult = RGB(getXYZ(plp.s->beautyAccumBuffer.read(launchIndex)));
    float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
    RGB colorResult = (1 - curWeight) * prevColorResult + curWeight * contribution;
    plp.s->beautyAccumBuffer.write(launchIndex, make_float4(colorResult.toNative(), 1.0f));
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(shadeAndResample)() {
    shadeAndResample<false, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(shadeAndResampleWithTemporalReuse)() {
    shadeAndResample<true, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(shadeAndResampleWithSpatialReuse)() {
    shadeAndResample<false, true>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(shadeAndResampleWithSpatioTemporalReuse)() {
    shadeAndResample<true, true>();
}
