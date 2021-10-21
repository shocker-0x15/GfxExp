#include "regir_shared.h"
#include "../common/common_device.cuh"

using namespace shared;

CUDA_DEVICE_MEM static PipelineLaunchParameters plp;

CUDA_DEVICE_FUNCTION void sampleLight(
    float ul, bool sampleEnvLight, float u0, float u1,
    LightSample* lightSample, float* areaPDensity) {
    CUtexObject texEmittance = 0;
    float3 emittance = make_float3(0.0f, 0.0f, 0.0f);
    float2 texCoord;
    if (sampleEnvLight) {
        float u, v;
        float uvPDF;
        plp.s->envLightImportanceMap.sample(u0, u1, &u, &v, &uvPDF);
        float phi = 2 * Pi * u;
        float theta = Pi * v;

        float posPhi = phi - plp.f->envLightRotation;
        posPhi = posPhi - floorf(posPhi / (2 * Pi)) * 2 * Pi;

        float3 direction = fromPolarYUp(posPhi, theta);
        float3 position = make_float3(direction.x, direction.y, direction.z);
        lightSample->position = position;
        lightSample->atInfinity = true;

        lightSample->normal = -position;

        // JP: テクスチャー空間中のPDFを面積に関するものに変換する。
        // EN: convert the PDF in texture space to one with respect to area.
        // The true value is: lim_{l to inf} uvPDF / (2 * Pi * Pi * std::sin(theta)) / l^2
        *areaPDensity = uvPDF / (2 * Pi * Pi * std::sin(theta));

        texEmittance = plp.s->envLightTexture;
        // JP: 環境マップテクスチャーの値に係数をかけて、通常の光源と同じように返り値を光束発散度
        //     として扱えるようにする。
        // EN: Multiply a coefficient to make the return value possible to be handled as luminous emittance.
        emittance = make_float3(Pi * plp.f->envLightPowerCoeff);
        texCoord.x = u;
        texCoord.y = v;
    }
    else {
        float lightProb = 1.0f;

        // JP: まずはインスタンスをサンプルする。
        // EN: First, sample an instance.
        float instProb;
        float uGeomInst;
        uint32_t instIndex = plp.s->lightInstDist.sample(ul, &instProb, &uGeomInst);
        lightProb *= instProb;
        const InstanceData &inst = plp.f->instanceDataBuffer[instIndex];

        // JP: 次にサンプルしたインスタンスに属するジオメトリインスタンスをサンプルする。
        // EN: Next, sample a geometry instance which belongs to the sampled instance.
        float geomInstProb;
        float uPrim;
        uint32_t geomInstIndexInInst = inst.lightGeomInstDist.sample(uGeomInst, &geomInstProb, &uPrim);
        uint32_t geomInstIndex = inst.geomInstSlots[geomInstIndexInInst];
        lightProb *= geomInstProb;
        const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[geomInstIndex];

        // JP: 最後に、サンプルしたジオメトリインスタンスに属するプリミティブをサンプルする。
        // EN: Finally, sample a primitive which belongs to the sampled geometry instance.
        float primProb;
        uint32_t primIndex = geomInst.emitterPrimDist.sample(uPrim, &primProb);
        lightProb *= primProb;

        // Uniform sampling on unit triangle
        // A Low-Distortion Map Between Triangle and Square
        float t0 = 0.5f * u0;
        float t1 = 0.5f * u1;
        float offset = t1 - t0;
        if (offset > 0)
            t1 += offset;
        else
            t0 -= offset;
        float t2 = 1 - (t0 + t1);

        //printf("%u-%u-%u: %g\n", instIndex, geomInstIndex, primIndex, lightProb);

        const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

        const shared::Triangle &tri = geomInst.triangleBuffer[primIndex];
        const shared::Vertex (&v)[3] = {
            geomInst.vertexBuffer[tri.index0],
            geomInst.vertexBuffer[tri.index1],
            geomInst.vertexBuffer[tri.index2]
        };
        float3 p[3] = {
            inst.transform * v[0].position,
            inst.transform * v[1].position,
            inst.transform * v[2].position,
        };

        float3 geomNormal = cross(p[1] - p[0], p[2] - p[0]);
        lightSample->position = t0 * p[0] + t1 * p[1] + t2 * p[2];
        lightSample->atInfinity = false;
        float recArea = 1.0f / length(geomNormal);
        //lightSample->normal = geomNormal * recArea;
        lightSample->normal = t0 * v[0].normal + t1 * v[1].normal + t2 * v[2].normal;
        lightSample->normal = normalize(inst.normalMatrix * lightSample->normal);
        recArea *= 2;
        *areaPDensity = lightProb * recArea;

        //printf("%u-%u-%u: (%g, %g, %g), PDF: %g\n", instIndex, geomInstIndex, primIndex,
        //       mat.emittance.x, mat.emittance.y, mat.emittance.z, *areaPDensity);

        //printf("%u-%u-%u: (%g, %g, %g), (%g, %g, %g)\n", instIndex, geomInstIndex, primIndex,
        //       lightPosition->x, lightPosition->y, lightPosition->z,
        //       lightNormal->x, lightNormal->y, lightNormal->z);

        if (mat.emittance) {
            texEmittance = mat.emittance;
            emittance = make_float3(1.0f, 1.0f, 1.0f);
            texCoord = t0 * v[0].texCoord + t1 * v[1].texCoord + t2 * v[2].texCoord;
        }
    }

    if (texEmittance) {
        float4 texValue = tex2DLod<float4>(texEmittance, texCoord.x, texCoord.y, 0.0f);
        emittance *= make_float3(texValue);
    }
    lightSample->emittance = emittance;
}

// TODO: セルの中央だけのサンプリングだと、セルの中央が光源の裏側に回ってしまっている場合に、
//       寄与の可能性のあるサンプルを棄却してしまう。代表点をランダムに決定するなどで解決できそうだが、
//       PDFが毎回変わるのでそれを考慮する必要あり？
CUDA_DEVICE_FUNCTION float3 sampleIntensity(
    const float3 &shadingPoint, float minSquaredDistance,
    float uLight, bool sampleEnvLight, float uPos0, float uPos1,
    LightSample* lightSample, float* probDensity) {
    sampleLight(uLight, sampleEnvLight, uPos0, uPos1,
                lightSample, probDensity);

    float3 shadowRayDir = lightSample->atInfinity ?
        lightSample->position :
        (lightSample->position - shadingPoint);
    float dist2 = sqLength(shadowRayDir);
    float dist = std::sqrt(dist2);
    shadowRayDir /= dist;

    float lpCos = dot(-shadowRayDir, lightSample->normal);

    if (lpCos > 0) {
        float3 Le = lightSample->emittance / Pi;
        float3 ret = Le * (lpCos / dist2);
        return ret;
    }
    else {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
}

template <bool useTemporalReuse>
CUDA_DEVICE_FUNCTION void buildCellReservoirsAndTemporalReuse(const PipelineLaunchParameters &_plp, uint32_t frameIndex) {
    plp = _plp;

    uint32_t linearThreadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t cellLinearIndex = linearThreadIndex / kNumLightSlotsPerCell;
    uint32_t lastAccessFrameIndex = plp.s->lastAccessFrameIndices[cellLinearIndex];
    if (linearThreadIndex == 0)
        *plp.f->numActiveCells = 0;
    plp.s->perCellNumAccesses[cellLinearIndex] = 0;
    if (frameIndex - lastAccessFrameIndex > 8)
        return;

    //uint32_t lightSlotIndex = linearThreadIndex % kNumLightSlotsPerCell;
    uint32_t iz = cellLinearIndex / (plp.s->gridDimension.x * plp.s->gridDimension.y);
    uint32_t iy = (cellLinearIndex % (plp.s->gridDimension.x * plp.s->gridDimension.y)) / plp.s->gridDimension.x;
    uint32_t ix = cellLinearIndex % plp.s->gridDimension.x;
    float3 cellCenter = plp.s->gridOrigin + make_float3(
        (ix + 0.5f) * plp.s->gridCellSize.x,
        (iy + 0.5f) * plp.s->gridCellSize.y,
        (iz + 0.5f) * plp.s->gridCellSize.z);
    const float minSquaredDistance = sqLength(0.5f * plp.s->gridCellSize);

    uint32_t bufferIndex = plp.f->bufferIndex;
    Reservoir<LightSample>* curReservoirs = plp.s->reservoirs[bufferIndex];
    ReservoirInfo* curReservoirInfos = plp.s->reservoirInfos[bufferIndex];

    PCG32RNG rng = plp.s->lightSlotRngs[linearThreadIndex];

    float selectedTargetPDensity = 0.0f;
    Reservoir<LightSample> reservoir;
    reservoir.initialize();

    // JP: セルの代表点に到達する光度をターゲットPDFとしてStreaming RISを実行。
    // EN: Perform streaming RIS with luminous intensity reaching to a cell's representative point
    //     as the target PDF.
    const uint32_t numCandidates = 1 << plp.f->log2NumCandidatesPerLightSlot;
    for (int candIdx = 0; candIdx < numCandidates; ++candIdx) {
        // JP: 環境光テクスチャーが設定されている場合は一定の確率でサンプルする。
        //     ダイバージェンスを抑えるために、ループの最初とそれ以外で環境光かそれ以外のサンプリングを分ける。
        //     ただし、そもそもReGIRは2段階のRISにおいてVisibilityを一切考慮していないため、環境光は(特に高いエネルギーの場合)、
        //     Reservoir中のサンプルに無駄なものを増やしてしまい、むしろ分散が増える傾向にある。
        //     環境光のサンプリングは別で行うほうが良いかもしれない。
        // EN: Sample an environmental light texture with a fixed probability if it is set.
        //     Separate sampling from the environmental light and the others to
        //     the beginning of the loop and the rest to avoid divergence.
        //     However in the first place, ReGIR doesn't take visibility into account at all during two-stage RIS,
        //     therefore an environmental light (particularly with a high-energy case) tends to increase useless
        //     samples in reservoirs, resulting in high variance.
        //     Separated environmental light sampling may be preferred.
        float uLight = rng.getFloat0cTo1o();
        bool sampleEnvLight = false;
        float probToSampleCurLightType = 1.0f;
        if (plp.s->envLightTexture && plp.f->enableEnvLight) {
            float prob = min(max(probToSampleEnvLight * numCandidates - candIdx, 0.0f), 1.0f);
            if (uLight < prob) {
                probToSampleCurLightType = probToSampleEnvLight;
                uLight = uLight / prob;
                sampleEnvLight = true;
            }
            else {
                probToSampleCurLightType = 1.0f - probToSampleEnvLight;
                uLight = (uLight - prob) / (1 - prob);
            }
        }

        // JP: 候補サンプルを生成して、ターゲットPDFを計算する。
        //     ターゲットPDFは正規化されていなくても良い。
        // EN: Generate a candidate sample then calculate the target PDF for it.
        //     Target PDF doesn't require to be normalized.
        LightSample lightSample;
        float areaPDensity;
        float3 cont = sampleIntensity(
            cellCenter, minSquaredDistance,
            uLight, sampleEnvLight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
            &lightSample, &areaPDensity);
        areaPDensity *= probToSampleCurLightType;
        float targetPDensity = convertToWeight(cont);

        // JP: 候補サンプル生成用のPDFとターゲットPDFは異なるためサンプルにはウェイトがかかる。
        // EN: The sample has a weight since the PDF to generate the candidate sample and the target PDF are
        //     different.
        float weight = targetPDensity / areaPDensity;
        //if (ix == 7 && iz == 7 && iy == 0) {
        //    printf("%2u, %2u, %2u, %3u, %u: %g, %g\n", ix, iy, iz, lightSlotIndex, candIdx,
        //           areaPDensity, targetPDensity);
        //}
        if (reservoir.update(lightSample, weight, rng.getFloat0cTo1o()))
            selectedTargetPDensity = targetPDensity;
    }

    // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
    // EN: Calculate the estimate of the reciprocal of the probability density that the current sample suvives.
    float recPDFEstimate = reservoir.getSumWeights() / (selectedTargetPDensity * reservoir.getStreamLength());
    if (!isfinite(recPDFEstimate)) {
        recPDFEstimate = 0.0f;
        selectedTargetPDensity = 0.0f;
    }

    // JP: 元の文献では過去数フレーム分のストリーム長で正規化されたReservoirを保持して、それらを結合しているが、
    //     ここでは正規化は行わず現在フレームと過去フレームの累積Reservoirの2つを結合する。
    // EN: The original literature suggests using stream length normalized reservoirs of several previous
    //     frames, then combine them, but here it doesn't use normalization and combines two reservoirs, one from
    //     the current frame and the other is the accumulation of the previous frames.
    if constexpr (useTemporalReuse) {
        uint32_t prevBufferIndex = (bufferIndex + 1) % 2;
        const Reservoir<LightSample>* prevReservoirs = plp.s->reservoirs[prevBufferIndex];
        const ReservoirInfo* prevReservoirInfos = plp.s->reservoirInfos[prevBufferIndex];

        uint32_t selfStreamLength = reservoir.getStreamLength();
        if (recPDFEstimate == 0.0f)
            reservoir.initialize();
        uint32_t combinedStreamLength = selfStreamLength;
        uint32_t maxNumPrevSamples = 20 * selfStreamLength;

        // JP: 際限なく過去フレームで得たサンプルがウェイトを増やさないように、
        //     前フレームのストリーム長を、現在フレームのReservoirに対して20倍までに制限する。
        // EN: Limit the stream length of the previous frame by 20 times of that of the current frame
        //     in order to avoid a sample obtained in the past getting a unlimited weight.
        // TODO: 光源アニメーションがある場合には前フレームと今のフレームでターゲットPDFが異なるので
        //       ウェイトを調整するべき？
        const Reservoir<LightSample> &prevReservoir = prevReservoirs[linearThreadIndex];
        const ReservoirInfo &prevResInfo = prevReservoirInfos[linearThreadIndex];
        const LightSample &prevLightSample = prevReservoir.getSample();
        float prevTargetDensity = prevResInfo.targetDensity;
        uint32_t prevStreamLength = min(prevReservoir.getStreamLength(), maxNumPrevSamples);
        float lengthCorrection = static_cast<float>(prevStreamLength) / prevReservoir.getStreamLength();
        float weight = lengthCorrection * prevReservoir.getSumWeights(); // New target PDF and prev target PDF are the same here.
        if (reservoir.update(prevLightSample, weight, rng.getFloat0cTo1o()))
            selectedTargetPDensity = prevTargetDensity;
        combinedStreamLength += prevStreamLength;
        reservoir.setStreamLength(combinedStreamLength);

        // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
        // EN: Calculate the estimate of the reciprocal of the probability density that the current sample suvives.
        float weightForEstimate = 1.0f / reservoir.getStreamLength();
        recPDFEstimate = weightForEstimate * reservoir.getSumWeights() / selectedTargetPDensity;
        if (!isfinite(recPDFEstimate)) {
            recPDFEstimate = 0.0f;
            selectedTargetPDensity = 0.0f;
        }
    }

    ReservoirInfo resInfo;
    resInfo.recPDFEstimate = recPDFEstimate;
    resInfo.targetDensity = selectedTargetPDensity;

    plp.s->lightSlotRngs[linearThreadIndex] = rng;
    curReservoirs[linearThreadIndex] = reservoir;
    curReservoirInfos[linearThreadIndex] = resInfo;
}

CUDA_DEVICE_KERNEL void buildCellReservoirs(PipelineLaunchParameters _plp, uint32_t frameIndex) {
    buildCellReservoirsAndTemporalReuse<false>(_plp, frameIndex);
}

CUDA_DEVICE_KERNEL void buildCellReservoirsAndTemporalReuse(PipelineLaunchParameters _plp, uint32_t frameIndex) {
    buildCellReservoirsAndTemporalReuse<true>(_plp, frameIndex);
}

CUDA_DEVICE_KERNEL void updateLastAccessFrameIndices(PipelineLaunchParameters _plp, uint32_t frameIndex) {
    plp = _plp;

    // JP: 現在のフレーム中でアクセスされたセルにフレーム番号を記録する。
    // EN: Record the frame number to cells that accessed in the current frame.
    uint32_t linearThreadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t cellLinearIndex = linearThreadIndex;
    uint32_t perCellNumAccesses = plp.s->perCellNumAccesses[cellLinearIndex];
    if (perCellNumAccesses > 0)
        plp.s->lastAccessFrameIndices[cellLinearIndex] = frameIndex;

    uint32_t numActiveCellsInGroup = __popc(__ballot_sync(0xFFFFFFFF, perCellNumAccesses > 0));
    if (threadIdx.x == 0 && numActiveCellsInGroup > 0)
        atomicAdd(plp.f->numActiveCells, numActiveCellsInGroup);
}
