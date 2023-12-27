#define PURE_CUDA
#include "../restir_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void performLightPreSampling() {
    const uint32_t linearThreadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    //const uint32_t subsetIndex = linearThreadIndex / lightSubsetSize;
    const uint32_t indexInSubset = linearThreadIndex % lightSubsetSize;
    PCG32RNG rng = plp.s->lightPreSamplingRngs[linearThreadIndex];

    // JP: 環境光テクスチャーが設定されている場合は一定の確率でサンプルする。
    //     ダイバージェンスを抑えるために、サブセットの最初とそれ以外で環境光かそれ以外のサンプリングを分ける。
    // EN: Sample an environmental light texture with a fixed probability if it is set.
    //     Separate sampling from the environmental light and the others to
    //     the beginning of the subset and the rest to avoid divergence.
    float probToSampleCurLightType = 1.0f;
    bool sampleEnvLight = false;
    if (plp.s->envLightTexture && plp.f->enableEnvLight) {
        if (plp.s->lightInstDist.integral() > 0.0f) {
            sampleEnvLight = indexInSubset < probToSampleEnvLight * lightSubsetSize;
            //sampleEnvLight = subsetIndex < probToSampleEnvLight * numLightSubsets;
            probToSampleCurLightType = sampleEnvLight ?
                probToSampleEnvLight : (1 - probToSampleEnvLight);
        }
        else {
            sampleEnvLight = true;
        }
    }

    PreSampledLight preSampledLight;
    sampleLight<false>(
        Point3D(0.0f),
        rng.getFloat0cTo1o(), sampleEnvLight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &preSampledLight.sample, &preSampledLight.areaPDensity);
    preSampledLight.areaPDensity *= probToSampleCurLightType;

    plp.s->lightPreSamplingRngs[linearThreadIndex] = rng;
    plp.s->preSampledLights[linearThreadIndex] = preSampledLight;
}



CUDA_DEVICE_KERNEL void performPerPixelRIS() {
    const int2 launchIndex = make_int2(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const uint32_t curBufIdx = plp.f->bufferIndex;

    // JP: タイルごとに共通のライトサブセットを選択することでメモリアクセスのコヒーレンシーを改善する。
    // EN: Select a common light subset for each tile to improve memory access coherency.
    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);
    CUDA_SHARED_MEM uint32_t sm_perTileLightSubsetIndex;
    if (threadIdx.x == 0 && threadIdx.y == 0)
        sm_perTileLightSubsetIndex = mapPrimarySampleToDiscrete(rng.getFloat0cTo1o(), numLightSubsets);
    __syncthreads();
    const uint32_t perTileLightSubsetIndex = sm_perTileLightSubsetIndex;
    const PreSampledLight* const lightSubSet = &plp.s->preSampledLights[perTileLightSubsetIndex * lightSubsetSize];

    const GBuffer0Elements gb0Elems = plp.s->GBuffer0[curBufIdx].read(launchIndex);
    if (gb0Elems.instSlot == 0xFFFFFFFF)
        return;

    const GBuffer2Elements gb2Elems = plp.s->GBuffer2[curBufIdx].read(launchIndex);
    const GBuffer3Elements gb3Elems = plp.s->GBuffer3[curBufIdx].read(launchIndex);

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
    const Point2D texCoord = decodeTexCoords(gb3Elems.qTexCoord);
    BSDF bsdf;
    bsdf.setup(mat, texCoord, 0.0f);

    const uint32_t curResIndex = plp.currentReservoirIndex;
    Reservoir<LightSample> reservoir;
    reservoir.initialize(LightSample());

    // JP: Unshadowed ContributionをターゲットPDFとしてStreaming RISを実行。
    // EN: Perform streaming RIS with unshadowed contribution as the target PDF.
    float selectedTargetDensity = 0.0f;
    const uint32_t numCandidates = 1 << plp.f->log2NumCandidateSamples;
    for (int i = 0; i < numCandidates; ++i) {
        const uint32_t lightIndex = mapPrimarySampleToDiscrete(rng.getFloat0cTo1o(), lightSubsetSize);
        const PreSampledLight &preSampledLight = lightSubSet[lightIndex];

        // JP: 候補サンプルを生成して、ターゲットPDFを計算する。
        //     ターゲットPDFは正規化されていなくても良い。
        // EN: Generate a candidate sample then calculate the target PDF for it.
        //     Target PDF doesn't require to be normalized.
        const RGB cont = performDirectLighting<ReSTIRRayType, false>(
            positionInWorld, vOutLocal, shadingFrame, bsdf,
            preSampledLight.sample);
        const float targetDensity = convertToWeight(cont);

        // JP: 候補サンプル生成用のPDFとターゲットPDFは異なるためサンプルにはウェイトがかかる。
        // EN: The sample has a weight since the PDF to generate the candidate sample and the target PDF are
        //     different.
        const float weight = targetDensity / preSampledLight.areaPDensity;
        if (reservoir.update(preSampledLight.sample, weight, rng.getFloat0cTo1o()))
            selectedTargetDensity = targetDensity;
    }

    // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
    // EN: Calculate the estimate of the reciprocal of the probability density that the current sample survives.
    float recPDFEstimate = reservoir.getSumWeights() / (selectedTargetDensity * reservoir.getStreamLength());
    if (!stc::isfinite(recPDFEstimate)) {
        recPDFEstimate = 0.0f;
        selectedTargetDensity = 0.0f;
    }

    ReservoirInfo reservoirInfo;
    reservoirInfo.recPDFEstimate = recPDFEstimate;
    reservoirInfo.targetDensity = selectedTargetDensity;

    plp.s->rngBuffer.write(launchIndex, rng);
    plp.s->reservoirBufferArray[curResIndex][launchIndex] = reservoir;
    plp.s->reservoirInfoBufferArray[curResIndex].write(launchIndex, reservoirInfo);
}
