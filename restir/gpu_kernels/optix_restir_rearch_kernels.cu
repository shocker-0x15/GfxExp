#include "../restir_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility);
}

static constexpr bool useMIS_RIS = true;



template <bool withTemporalRIS, bool withSpatialRIS, bool useUnbiasedEstimator>
CUDA_DEVICE_FUNCTION CUDA_INLINE void traceShadowRays() {
    const int2 launchIndex = make_int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t curBufIdx = plp.f->bufferIndex;
    uint32_t prevBufIdx;
    const optixu::BlockBuffer2D<Reservoir<LightSample>, 0> curReservoirs =
        plp.s->reservoirBufferArray[plp.currentReservoirIndex];
    optixu::BlockBuffer2D<Reservoir<LightSample>, 0> prevReservoirs;
    const optixu::NativeBlockBuffer2D<SampleVisibility> curSampleVisBuffer =
        plp.s->sampleVisibilityBufferArray[curBufIdx];
    optixu::NativeBlockBuffer2D<SampleVisibility> prevSampleVisBuffer;
    if constexpr (withTemporalRIS || withSpatialRIS) {
        prevBufIdx = (curBufIdx + 1) % 2;
        prevReservoirs = plp.s->reservoirBufferArray[(plp.currentReservoirIndex + 1) % 2];
        prevSampleVisBuffer = plp.s->sampleVisibilityBufferArray[prevBufIdx];
    }
    else {
        (void)prevBufIdx;
        (void)prevReservoirs;
        (void)prevSampleVisBuffer;
    }

    const GBuffer0Elements gb0Elems = plp.s->GBuffer0[curBufIdx].read(launchIndex);
    if (gb0Elems.instSlot == 0xFFFFFFFF)
        return;

    const GBuffer2Elements gb2Elems = plp.s->GBuffer2[curBufIdx].read(launchIndex);
    const GBuffer3Elements gb3Elems = plp.s->GBuffer3[curBufIdx].read(launchIndex);

    Point3D positionInWorld = gb2Elems.positionInWorld;
    const Normal3D geometricNormalInWorld = decodeNormal(gb2Elems.qGeometricNormal);
    const Normal3D shadingNormalInWorld = decodeNormal(gb3Elems.qShadingNormal);

    const Vector3D vOut = plp.f->camera.position - positionInWorld;
    const float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
    // Offsetting assumes BRDF.
    positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);
    const float dist = length(vOut);

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
        const GBuffer1Elements gb1Elems = plp.s->GBuffer1[curBufIdx].read(launchIndex);
        const Vector2D motionVector = gb1Elems.motionVector;
        tNbCoord = make_int2(launchIndex.x + 0.5f - motionVector.x,
                             launchIndex.y + 0.5f - motionVector.y);
        // JP: 隣接ピクセルのジオメトリ・マテリアルがあまりに異なる場合に候補サンプルを再利用すると
        //     バイアスが増えてしまうため、そのようなピクセルは棄却する。
        // EN: Reusing candidates from neighboring pixels with substantially different geometry/material
        //     leads to increased bias. Reject such a pixel.
        sampleVis.temporalPassedHeuristic =
            testNeighbor<true>(prevBufIdx, tNbCoord, dist, shadingNormalInWorld);
        if (sampleVis.temporalPassedHeuristic) {
            LightSample temporalSample;
            if (plp.f->reuseVisibilityForTemporal && !useUnbiasedEstimator) {
                const SampleVisibility prevSampleVis = prevSampleVisBuffer.read(tNbCoord);
                sampleVis.temporalSample = prevSampleVis.selectedSample;
            }
            else {
                const Reservoir<LightSample> neighbor = prevReservoirs[tNbCoord];
                temporalSample = neighbor.getSample();
                temporalSampleIsValid = neighbor.getSumWeights() > 0.0f;
                if (temporalSampleIsValid)
                    sampleVis.temporalSample =
                        evaluateVisibility<ReSTIRRayType>(positionInWorld, temporalSample);
            }

            if constexpr (useUnbiasedEstimator) {
                const GBuffer2Elements nbGb2Elems = plp.s->GBuffer2[prevBufIdx].read(tNbCoord);
                const Point3D nbPositionInWorld = nbGb2Elems.positionInWorld;

                const Normal3D nbGeometricNormalInWorld = decodeNormal(nbGb2Elems.qGeometricNormal);
                const Vector3D nbVOut = plp.f->prevCamera.position - nbPositionInWorld;
                const float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
                tNbPositionInWorld = offsetRayOrigin(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);

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
            const uint32_t deltaIndex = plp.spatialNeighborBaseIndex +
                5 * launchIndex.x + 7 * launchIndex.y;
            const Vector2D delta = plp.s->spatialNeighborDeltas[deltaIndex % 1024];
            deltaX = radius * delta.x;
            deltaY = radius * delta.y;
        }
        else {
            PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);
            radius *= std::sqrt(rng.getFloat0cTo1o());
            const float angle = 2 * Pi * rng.getFloat0cTo1o();
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
                const float threshold2 = pow2(plp.f->radiusThresholdForSpatialVisReuse);
                const float dist2 = pow2(deltaX) + pow2(deltaY);
                reused = dist2 < threshold2;
            }

            LightSample spatiotemporalSample;
            if (reused) {
                const SampleVisibility prevSampleVis = prevSampleVisBuffer.read(stNbCoord);
                sampleVis.spatiotemporalSample = prevSampleVis.selectedSample;
            }
            else {
                const Reservoir<LightSample> neighbor = prevReservoirs[stNbCoord];
                spatiotemporalSample = neighbor.getSample();
                spatiotemporalSampleIsValid = neighbor.getSumWeights() > 0.0f;
                if (spatiotemporalSampleIsValid)
                    sampleVis.spatiotemporalSample =
                       evaluateVisibility<ReSTIRRayType>(positionInWorld, spatiotemporalSample);
            }

            if constexpr (useUnbiasedEstimator) {
                const GBuffer2Elements nbGb2Elems = plp.s->GBuffer2[prevBufIdx].read(stNbCoord);
                const Point3D nbPositionInWorld = nbGb2Elems.positionInWorld;

                const Normal3D nbGeometricNormalInWorld = decodeNormal(nbGb2Elems.qGeometricNormal);
                const Vector3D nbVOut = plp.f->prevCamera.position - nbPositionInWorld;
                const float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
                stNbPositionInWorld = offsetRayOrigin(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);

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
    const int2 &launchIndex, uint32_t prevBufIdx,
    const optixu::BlockBuffer2D<Reservoir<LightSample>, 0> &prevReservoirs,
    uint32_t maxPrevStreamLength, const SampleVisibility &sampleVis,
    uint32_t selfStreamLength, const Point3D &positionInWorld, const Vector3D &vOutLocal,
    const ReferenceFrame &shadingFrame, const BSDF &bsdf,
    const int2 &tNbCoord, const int2 &stNbCoord,
    uint32_t streamLength, const LightSample &lightSample, float sampleTargetDensity) {
    const float numMisWeight = sampleTargetDensity;
    float denomMisWeight = numMisWeight * streamLength;

    if constexpr (sampleType != SampleType::New) {
        // JP: 与えられたサンプルを現在のシェーディング点で得る確率密度を計算する。
        // EN: Compute a probability density to get the given sample at the current shading point.
        const RGB cont = performDirectLighting<ReSTIRRayType, false>(
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
                const GBuffer2Elements nbGb2Elems = plp.s->GBuffer2[prevBufIdx].read(tNbCoord);
                const GBuffer3Elements nbGb3Elems = plp.s->GBuffer3[prevBufIdx].read(tNbCoord);

                nbPositionInWorld = nbGb2Elems.positionInWorld;
                const Normal3D nbGeometricNormalInWorld = decodeNormal(nbGb2Elems.qGeometricNormal);
                const Vector3D nbVOut = normalize(plp.f->prevCamera.position - nbPositionInWorld);
                const float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
                nbPositionInWorld = offsetRayOrigin(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);

                const MaterialData &nbMat = plp.s->materialDataBuffer[nbGb3Elems.matSlot];
                const Point2D nbTexCoord = decodeTexCoords(nbGb3Elems.qTexCoord);
                nbBsdf.setup(nbMat, nbTexCoord, 0.0f);

                nbShadingFrame = ReferenceFrame(
                    decodeNormal(nbGb3Elems.qShadingNormal), decodeVector(nbGb3Elems.qShadingTangent));
                nbVOutLocal = nbShadingFrame.toLocal(nbVOut);
            }

            // JP: 与えられたサンプルを前のフレームで対応するシェーディング点で得る確率密度を計算する。
            // EN: Compute a probability density to get the given sample at the corresponding shading point
            //     in the previous frame.
            const RGB cont = performDirectLighting<ReSTIRRayType, false>(
                nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, lightSample);
            float nbTargetDensity = convertToWeight(cont);
            if (plp.f->useUnbiasedEstimator) {
                nbTargetDensity *= sampleType == SampleType::New ?
                    sampleVis.newSampleOnTemporal :
                    sampleVis.spatiotemporalSampleOnTemporal;
            }

            const Reservoir<LightSample> /*&*/neighbor = prevReservoirs[tNbCoord];
            const uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
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
                const GBuffer2Elements nbGb2Elems = plp.s->GBuffer2[prevBufIdx].read(stNbCoord);
                const GBuffer3Elements nbGb3Elems = plp.s->GBuffer3[prevBufIdx].read(stNbCoord);

                nbPositionInWorld = nbGb2Elems.positionInWorld;
                const Normal3D nbGeometricNormalInWorld = decodeNormal(nbGb2Elems.qGeometricNormal);
                const Vector3D nbVOut = normalize(plp.f->prevCamera.position - nbPositionInWorld);
                const float nbFrontHit = dot(nbVOut, nbGeometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
                nbPositionInWorld = offsetRayOrigin(nbPositionInWorld, nbFrontHit * nbGeometricNormalInWorld);

                const MaterialData &nbMat = plp.s->materialDataBuffer[nbGb3Elems.matSlot];
                const Point2D nbTexCoord = decodeTexCoords(nbGb3Elems.qTexCoord);
                nbBsdf.setup(nbMat, nbTexCoord, 0.0f);

                nbShadingFrame = ReferenceFrame(
                    decodeNormal(nbGb3Elems.qShadingNormal), decodeVector(nbGb3Elems.qShadingTangent));
                nbVOutLocal = nbShadingFrame.toLocal(nbVOut);
            }

            // JP: 与えられたサンプルを近傍のシェーディング点(前フレーム)で得る確率密度を計算する。
            // EN: Compute a probability density to get the given sample at a shading point on the neighbor
            //     (in the previous frame).
            const RGB cont = performDirectLighting<ReSTIRRayType, false>(
                nbPositionInWorld, nbVOutLocal, nbShadingFrame, nbBsdf, lightSample);
            float nbTargetDensity = convertToWeight(cont);
            if (plp.f->useUnbiasedEstimator) {
                nbTargetDensity *= sampleType == SampleType::New ?
                    sampleVis.newSampleOnSpatiotemporal :
                    sampleVis.temporalSampleOnSpatiotemporal;
            }

            const Reservoir<LightSample> /*&*/neighbor = prevReservoirs[stNbCoord];
            const uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
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
    const int2 launchIndex = make_int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t curBufIdx = plp.f->bufferIndex;
    uint32_t prevBufIdx;
    optixu::BlockBuffer2D<Reservoir<LightSample>, 0> curReservoirs =
        plp.s->reservoirBufferArray[plp.currentReservoirIndex];
    optixu::BlockBuffer2D<Reservoir<LightSample>, 0> prevReservoirs;
    const optixu::NativeBlockBuffer2D<ReservoirInfo> curReservoirInfos =
        plp.s->reservoirInfoBufferArray[plp.currentReservoirIndex];
    optixu::NativeBlockBuffer2D<ReservoirInfo> prevReservoirInfos;
    const optixu::NativeBlockBuffer2D<SampleVisibility> curSampleVisBuffer =
        plp.s->sampleVisibilityBufferArray[curBufIdx];
    if constexpr (withTemporalRIS || withSpatialRIS) {
        prevBufIdx = (curBufIdx + 1) % 2;
        uint32_t prevResIndex = (plp.currentReservoirIndex + 1) % 2;
        prevReservoirs = plp.s->reservoirBufferArray[prevResIndex];
        prevReservoirInfos = plp.s->reservoirInfoBufferArray[prevResIndex];
    }
    else {
        (void)prevBufIdx;
        (void)prevReservoirs;
        (void)prevReservoirInfos;
    }

    const GBuffer0Elements gb0Elems = plp.s->GBuffer0[curBufIdx].read(launchIndex);
    const GBuffer3Elements gb3Elems = plp.s->GBuffer3[curBufIdx].read(launchIndex);
    const Point2D texCoord = decodeTexCoords(gb3Elems.qTexCoord);

    RGB contribution(0.01f, 0.01f, 0.01f);
    if (gb0Elems.instSlot != 0xFFFFFFFF) {
        PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

        int2 tNbCoord;
        if constexpr (withTemporalRIS) {
            const GBuffer1Elements gb1Elems = plp.s->GBuffer1[curBufIdx].read(launchIndex);
            tNbCoord = make_int2(launchIndex.x + 0.5f - gb1Elems.motionVector.x,
                                 launchIndex.y + 0.5f - gb1Elems.motionVector.y);
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
                const Vector2D delta = plp.s->spatialNeighborDeltas[deltaIndex % 1024];
                deltaX = radius * delta.x;
                deltaY = radius * delta.y;
            }
            else {
                radius *= std::sqrt(rng.getFloat0cTo1o());
                const float angle = 2 * Pi * rng.getFloat0cTo1o();
                deltaX = radius * std::cos(angle);
                deltaY = radius * std::sin(angle);
            }
            stNbCoord = make_int2(launchIndex.x + 0.5f + deltaX,
                                  launchIndex.y + 0.5f + deltaY);
        }
        else {
            (void)stNbCoord;
        }

        const GBuffer2Elements gb2Elems = plp.s->GBuffer2[curBufIdx].read(launchIndex);

        Point3D positionInWorld = gb2Elems.positionInWorld;
        const Normal3D geometricNormalInWorld = decodeNormal(gb2Elems.qGeometricNormal);

        const Vector3D vOut = normalize(plp.f->camera.position - positionInWorld);
        const float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
        // Offsetting assumes BRDF.
        positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);

        const ReferenceFrame shadingFrame(
            decodeNormal(gb3Elems.qShadingNormal), decodeVector(gb3Elems.qShadingTangent));
        const Vector3D vOutLocal = shadingFrame.toLocal(vOut);

        const MaterialData &mat = plp.s->materialDataBuffer[gb3Elems.matSlot];
        BSDF bsdf;
        bsdf.setup(mat, texCoord, 0.0f);

        // JP: 光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from a light source directly seeing.
        contribution = RGB(0.0f);
        if (vOutLocal.z > 0) {
            RGB emittance(0.0f, 0.0f, 0.0f);
            if (mat.emittance) {
                const float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
                emittance = RGB(getXYZ(texValue));
            }
            contribution += emittance / Pi;
        }

        SampleVisibility sampleVis = curSampleVisBuffer.read(launchIndex);

        float selectedTargetDensity = 0.0f;
        Reservoir<LightSample> combinedReservoir;
        uint32_t combinedStreamLength = 0;
        combinedReservoir.initialize(LightSample());

        RGB directCont(0.0f, 0.0f, 0.0f);
        float selectedMisWeight = 0.0f;

        const Reservoir<LightSample> /*&*/selfRes = curReservoirs[launchIndex];
        const ReservoirInfo selfResInfo = curReservoirInfos.read(launchIndex);
        const uint32_t selfStreamLength = selfRes.getStreamLength();
        uint32_t maxPrevStreamLength;
        if constexpr (withTemporalRIS || withSpatialRIS)
            maxPrevStreamLength = 20 * selfStreamLength;
        else
            (void)maxPrevStreamLength;

        // New sample for the current pixel.
        {
            if (selfResInfo.recPDFEstimate > 0.0f && sampleVis.newSample) {
                const LightSample lightSample = selfRes.getSample();
                const RGB cont = performDirectLighting<ReSTIRRayType, false>(
                    positionInWorld, vOutLocal, shadingFrame, bsdf, lightSample);
                const float targetDensity = convertToWeight(cont);

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

                const float weight = selfRes.getSumWeights();
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
                const uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
                if (neighborInfo.recPDFEstimate > 0.0f) {
                    // JP: 隣接ピクセルが持つ候補サンプルの「現在の」ピクセルにおける確率密度を計算する。
                    // EN: Calculate the probability density at the "current" pixel of the candidate sample
                    //     the neighboring pixel holds.
                    const LightSample nbLightSample = neighbor.getSample();
                    const RGB cont = performDirectLighting<ReSTIRRayType, false>(
                        positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
                    const float targetDensity = convertToWeight(cont);

                    const float misWeight = computeMISWeight<SampleType::Temporal, withTemporalRIS, withSpatialRIS>(
                        launchIndex, prevBufIdx, prevReservoirs,
                        maxPrevStreamLength, sampleVis,
                        selfStreamLength, positionInWorld, vOutLocal, shadingFrame, bsdf,
                        tNbCoord, stNbCoord,
                        nbStreamLength, nbLightSample, neighborInfo.targetDensity);

                    const float weight = targetDensity * neighborInfo.recPDFEstimate * nbStreamLength;
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
                const uint32_t nbStreamLength = min(neighbor.getStreamLength(), maxPrevStreamLength);
                if (neighborInfo.recPDFEstimate > 0.0f) {
                    // JP: 隣接ピクセルが持つ候補サンプルの「現在の」ピクセルにおける確率密度を計算する。
                    // EN: Calculate the probability density at the "current" pixel of the candidate sample
                    //     the neighboring pixel holds.
                    const LightSample nbLightSample = neighbor.getSample();
                    const RGB cont = performDirectLighting<ReSTIRRayType, false>(
                        positionInWorld, vOutLocal, shadingFrame, bsdf, nbLightSample);
                    const float targetDensity = convertToWeight(cont);

                    const float misWeight = computeMISWeight<SampleType::Spatiotemporal, withTemporalRIS, withSpatialRIS>(
                        launchIndex, prevBufIdx, prevReservoirs,
                        maxPrevStreamLength, sampleVis,
                        selfStreamLength, positionInWorld, vOutLocal, shadingFrame, bsdf,
                        tNbCoord, stNbCoord,
                        nbStreamLength, nbLightSample, neighborInfo.targetDensity);

                    // JP: 隣接ピクセルと現在のピクセルではターゲットPDFが異なるためサンプルはウェイトを持つ。
                    // EN: The sample has a weight since the target PDFs of the neighboring pixel and the current
                    //     are the different.
                    const float weight = targetDensity * neighborInfo.recPDFEstimate * nbStreamLength;
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
        if (!stc::isfinite(recPDFEstimate) || (plp.f->reuseVisibility && !sampleVis.selectedSample)) {
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
            const float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, texCoord.x, texCoord.y, 0.0f);
            const RGB luminance = plp.f->envLightPowerCoeff * RGB(getXYZ(texValue));
            contribution = luminance;
        }
    }

    RGB prevColorResult(0.0f, 0.0f, 0.0f);
    if (plp.f->numAccumFrames > 0)
        prevColorResult = RGB(getXYZ(plp.s->beautyAccumBuffer.read(launchIndex)));
    const float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
    const RGB colorResult = (1 - curWeight) * prevColorResult + curWeight * contribution;
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
