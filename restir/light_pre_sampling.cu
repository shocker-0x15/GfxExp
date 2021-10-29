#include "restir_shared.h"
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

CUDA_DEVICE_KERNEL void performLightPreSampling(PipelineLaunchParameters _plp) {
    plp = _plp;

    uint32_t linearThreadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    //uint32_t subsetIndex = linearThreadIndex / lightSubsetSize;
    uint32_t indexInSubset = linearThreadIndex % lightSubsetSize;
    PCG32RNG rng = plp.s->lightPreSamplingRngs[linearThreadIndex];
    float probToSampleCurLightType = 1.0f;
    bool sampleEnvLight = false;
    if (plp.s->envLightTexture && plp.f->enableEnvLight) {
        sampleEnvLight = indexInSubset < probToSampleEnvLight * lightSubsetSize;
        probToSampleCurLightType = sampleEnvLight ?
            probToSampleEnvLight : (1 - probToSampleEnvLight);
    }
    PreSampledLight preSampledLight;
    sampleLight(
        rng.getFloat0cTo1o(), sampleEnvLight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &preSampledLight.sample, &preSampledLight.areaPDensity);
    preSampledLight.areaPDensity *= probToSampleCurLightType;

    plp.s->lightPreSamplingRngs[linearThreadIndex] = rng;
    plp.s->preSampledLights[linearThreadIndex] = preSampledLight;
}



CUDA_DEVICE_KERNEL void samplePerTileLightSubsetIndices(PipelineLaunchParameters _plp) {
    plp = _plp;

    int2 tileIndex = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
                               blockDim.y * blockIdx.y + threadIdx.y);
    if (tileIndex.x >= plp.s->numTiles.x ||
        tileIndex.y >= plp.s->numTiles.y)
        return;

    int2 cornerPixelIndex = tileIndex * make_int2(tileSizeX, tileSizeY);
    PCG32RNG rng = plp.s->rngBuffer.read(cornerPixelIndex);

    uint32_t lightSubsetIndex = mapPrimarySampleToDiscrete(rng.getFloat0cTo1o(), numLightSubsets);

    plp.s->perTileLightSubsetIndexBuffer.write(tileIndex, lightSubsetIndex);
    plp.s->rngBuffer.write(cornerPixelIndex, rng);
}