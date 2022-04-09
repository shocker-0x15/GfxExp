#define PURE_CUDA
#include "restir_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void performLightPreSampling() {
    uint32_t linearThreadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    //uint32_t subsetIndex = linearThreadIndex / lightSubsetSize;
    uint32_t indexInSubset = linearThreadIndex % lightSubsetSize;
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
            probToSampleCurLightType = sampleEnvLight ?
                probToSampleEnvLight : (1 - probToSampleEnvLight);
        }
        else {
            sampleEnvLight = true;
        }
    }

    PreSampledLight preSampledLight;
    sampleLight<false>(
        make_float3(0.0f),
        rng.getFloat0cTo1o(), sampleEnvLight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &preSampledLight.sample, &preSampledLight.areaPDensity);
    preSampledLight.areaPDensity *= probToSampleCurLightType;

    plp.s->lightPreSamplingRngs[linearThreadIndex] = rng;
    plp.s->preSampledLights[linearThreadIndex] = preSampledLight;
}



CUDA_DEVICE_KERNEL void performPerPixelRIS() {
    int2 launchIndex = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
                                 blockDim.y * blockIdx.y + threadIdx.y);

    uint32_t curBufIdx = plp.f->bufferIndex;
    GBuffer0 gBuffer0 = plp.s->GBuffer0[curBufIdx].read(launchIndex);
    GBuffer1 gBuffer1 = plp.s->GBuffer1[curBufIdx].read(launchIndex);
    GBuffer2 gBuffer2 = plp.s->GBuffer2[curBufIdx].read(launchIndex);

    float3 positionInWorld = gBuffer0.positionInWorld;
    float3 shadingNormalInWorld = gBuffer1.normalInWorld;
    float2 texCoord = make_float2(gBuffer0.texCoord_x, gBuffer1.texCoord_y);
    uint32_t materialSlot = gBuffer2.materialSlot;

    // JP: タイルごとに共通のライトサブセットを選択することでメモリアクセスのコヒーレンシーを改善する。
    // EN: Select a common light subset for each tile to improve memory access coherency.
    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);
    CUDA_SHARED_MEM uint32_t sm_perTileLightSubsetIndex;
    if (threadIdx.x == 0 && threadIdx.y == 0)
        sm_perTileLightSubsetIndex = mapPrimarySampleToDiscrete(rng.getFloat0cTo1o(), numLightSubsets);
    __syncthreads();
    uint32_t perTileLightSubsetIndex = sm_perTileLightSubsetIndex;
    const PreSampledLight* lightSubSet = &plp.s->preSampledLights[perTileLightSubsetIndex * lightSubsetSize];

    if (materialSlot == 0xFFFFFFFF)
        return;

    const MaterialData &mat = plp.s->materialDataBuffer[materialSlot];

    // TODO?: Use true geometric normal.
    float3 geometricNormalInWorld = shadingNormalInWorld;
    float3 vOut = plp.f->camera.position - positionInWorld;
    float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

    BSDF bsdf;
    bsdf.setup(mat, texCoord);
    ReferenceFrame shadingFrame(shadingNormalInWorld);
    positionInWorld = offsetRayOriginNaive(positionInWorld, frontHit * geometricNormalInWorld);
    float dist = length(vOut);
    vOut /= dist;
    float3 vOutLocal = shadingFrame.toLocal(vOut);

    uint32_t curResIndex = plp.currentReservoirIndex;
    Reservoir<LightSample> reservoir;
    reservoir.initialize();

    // JP: Unshadowed ContributionをターゲットPDFとしてStreaming RISを実行。
    // EN: Perform streaming RIS with unshadowed contribution as the target PDF.
    float selectedTargetDensity = 0.0f;
    uint32_t numCandidates = 1 << plp.f->log2NumCandidateSamples;
    for (int i = 0; i < numCandidates; ++i) {
        uint32_t lightIndex = mapPrimarySampleToDiscrete(rng.getFloat0cTo1o(), lightSubsetSize);
        const PreSampledLight &preSampledLight = lightSubSet[lightIndex];

        // JP: 候補サンプルを生成して、ターゲットPDFを計算する。
        //     ターゲットPDFは正規化されていなくても良い。
        // EN: Generate a candidate sample then calculate the target PDF for it.
        //     Target PDF doesn't require to be normalized.
        float3 cont = performDirectLighting<false>(
            positionInWorld, vOutLocal, shadingFrame, bsdf,
            preSampledLight.sample);
        float targetDensity = convertToWeight(cont);

        // JP: 候補サンプル生成用のPDFとターゲットPDFは異なるためサンプルにはウェイトがかかる。
        // EN: The sample has a weight since the PDF to generate the candidate sample and the target PDF are
        //     different.
        float weight = targetDensity / preSampledLight.areaPDensity;
        if (reservoir.update(preSampledLight.sample, weight, rng.getFloat0cTo1o()))
            selectedTargetDensity = targetDensity;
    }

    // JP: 現在のサンプルが生き残る確率密度の逆数の推定値を計算する。
    // EN: Calculate the estimate of the reciprocal of the probability density that the current sample suvives.
    float recPDFEstimate = reservoir.getSumWeights() / (selectedTargetDensity * reservoir.getStreamLength());
    if (!isfinite(recPDFEstimate)) {
        recPDFEstimate = 0.0f;
        selectedTargetDensity = 0.0f;
    }

    ReservoirInfo reservoirInfo;
    reservoirInfo.recPDFEstimate = recPDFEstimate;
    reservoirInfo.targetDensity = selectedTargetDensity;

    plp.s->rngBuffer.write(launchIndex, rng);
    plp.s->reservoirBuffer[curResIndex][launchIndex] = reservoir;
    plp.s->reservoirInfoBuffer[curResIndex].write(launchIndex, reservoirInfo);
}
