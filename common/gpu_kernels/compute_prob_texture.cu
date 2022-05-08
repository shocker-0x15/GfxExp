#define PURE_CUDA
#include "../common_device.cuh"

using namespace shared;

CUDA_DEVICE_KERNEL void computeProbabilityTextureFirstMip(
    const float* probs, uint32_t numElems,
    ProbabilityTexture* probTex, optixu::NativeBlockBuffer2D<float> dstMip) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint2 dims = computeProbabilityTextureDimentions(numElems);
    probTex->setDimensions(dims);
    uint2 idx2D = probTex->compute2DFrom1D(linearIndex);
    float value = 0.0f;
    if (linearIndex < numElems)
        value = probs[linearIndex];
    if (idx2D.x < dims.x && idx2D.y < dims.y)
        dstMip.write(idx2D, value);
}

CUDA_DEVICE_KERNEL void computeTriangleProbabilities(
    GeometryInstanceData* geomInst, uint32_t numTriangles,
    const MaterialData* materialDataBuffer,
    optixu::NativeBlockBuffer2D<float> dstMip) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint2 dims = computeProbabilityTextureDimentions(numTriangles);
    if (linearIndex == 0)
        geomInst->emitterPrimDist.setDimensions(dims);
    uint2 idx2D = compute2DFrom1D(dims, linearIndex);
    float importance = 0.0f;
    if (linearIndex < numTriangles) {
        const MaterialData &mat = materialDataBuffer[geomInst->materialSlot];
        const Triangle &tri = geomInst->triangleBuffer[linearIndex];
        const Vertex (&v)[3] = {
            geomInst->vertexBuffer[tri.index0],
            geomInst->vertexBuffer[tri.index1],
            geomInst->vertexBuffer[tri.index2]
        };

        float3 normal = cross(v[1].position - v[0].position, v[2].position - v[0].position);
        float area = length(normal);

        // TODO: もっと正確な推定の実装。テクスチャー空間中の面積に応じてMIPレベルを選択する？
        float3 emittanceEstimate = make_float3(0.0f, 0.0f, 0.0f);
        emittanceEstimate += getXYZ(tex2DLod<float4>(mat.emittance, v[0].texCoord.x, v[0].texCoord.y, 0));
        emittanceEstimate += getXYZ(tex2DLod<float4>(mat.emittance, v[1].texCoord.x, v[1].texCoord.y, 0));
        emittanceEstimate += getXYZ(tex2DLod<float4>(mat.emittance, v[2].texCoord.x, v[2].texCoord.y, 0));
        emittanceEstimate /= 3;

        importance = sRGB_calcLuminance(emittanceEstimate) * area;
    }
    if (idx2D.x < dims.x && idx2D.y < dims.y)
        dstMip.write(idx2D, importance);
}

CUDA_DEVICE_KERNEL void computeGeomInstProbabilities(
    InstanceData* inst, uint32_t instIdx, uint32_t numGeomInsts,
    const GeometryInstanceData* geometryInstanceDataBuffer,
    optixu::NativeBlockBuffer2D<float> dstMip) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint2 dims = computeProbabilityTextureDimentions(numGeomInsts);
    if (linearIndex == 0)
        inst->lightGeomInstDist.setDimensions(dims);
    uint2 idx2D = compute2DFrom1D(dims, linearIndex);
    float importance = 0.0f;
    if (linearIndex < numGeomInsts) {
        uint32_t slot = inst->geomInstSlots[linearIndex];
        const GeometryInstanceData &geomInst = geometryInstanceDataBuffer[slot];
        importance = geomInst.emitterPrimDist.integral();
        //printf("%5u-%5u: %g\n", instIdx, slot, importance);
    }
    if (idx2D.x < dims.x && idx2D.y < dims.y)
        dstMip.write(idx2D, importance);
}

CUDA_DEVICE_KERNEL void computeInstProbabilities(
    ProbabilityTexture* lightInstDist, uint32_t numInsts,
    const InstanceData* instanceDataBuffer,
    optixu::NativeBlockBuffer2D<float> dstMip) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint2 dims = computeProbabilityTextureDimentions(numInsts);
    if (linearIndex == 0)
        lightInstDist->setDimensions(dims);
    uint2 idx2D = compute2DFrom1D(dims, linearIndex);
    float importance = 0.0f;
    if (linearIndex < numInsts) {
        const InstanceData &inst = instanceDataBuffer[linearIndex];
        importance = inst.lightGeomInstDist.integral();
    }
    if (idx2D.x < dims.x && idx2D.y < dims.y)
        dstMip.write(idx2D, importance);
}

CUDA_DEVICE_KERNEL void computeProbabilityTextureMip(
    const ProbabilityTexture* probTex, uint32_t dstMipLevel,
    optixu::NativeBlockBuffer2D<float> srcMip,
    optixu::NativeBlockBuffer2D<float> dstMip) {
    uint32_t numMipLevels = probTex->calcNumMipLevels();
    if (dstMipLevel >= numMipLevels)
        return;

    uint2 srcDims = probTex->getDimensions() >> (dstMipLevel - 1);
    uint2 dstDims = max(probTex->getDimensions() >> dstMipLevel, make_uint2(1, 1));
    uint2 globalIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                   blockDim.y * blockIdx.y + threadIdx.y);
    uint2 ul = 2 * globalIndex;
    uint2 ur = ul + make_uint2(1, 0);
    uint2 ll = ul + make_uint2(0, 1);
    uint2 lr = ll + make_uint2(1, 0);
    float sum = 0.0f;
    sum += (ul.x < srcDims.x && ul.y < srcDims.y) ? srcMip.read(ul) : 0.0f;
    sum += (ur.x < srcDims.x && ur.y < srcDims.y) ? srcMip.read(ur) : 0.0f;
    sum += (ll.x < srcDims.x && ll.y < srcDims.y) ? srcMip.read(ll) : 0.0f;
    sum += (lr.x < srcDims.x && lr.y < srcDims.y) ? srcMip.read(lr) : 0.0f;
    if (globalIndex.x < dstDims.x && globalIndex.y < dstDims.y)
        dstMip.write(globalIndex, sum);
}

CUDA_DEVICE_KERNEL void testProbabilityTexture(
    const ProbabilityTexture* probTex, PCG32RNG* rngs, uint32_t numThreads,
    uint32_t* histogram) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    PCG32RNG rng = rngs[linearIndex];
    float prob;
    uint32_t sampledIndex = probTex->sample(rng.getFloat0cTo1o(), &prob);
    atomicAdd(&histogram[sampledIndex], 1u);
    rngs[linearIndex] = rng;
}
