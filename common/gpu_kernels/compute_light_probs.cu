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



CUDA_DEVICE_FUNCTION CUDA_INLINE float computeTriangleImportance(
    GeometryInstanceData* geomInst, uint32_t triIndex,
    const MaterialData* materialDataBuffer) {
    const MaterialData &mat = materialDataBuffer[geomInst->materialSlot];
    const Triangle &tri = geomInst->triangleBuffer[triIndex];
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

    float importance = sRGB_calcLuminance(emittanceEstimate) * area;
    return importance;
}

CUDA_DEVICE_KERNEL void computeTriangleProbTexture(
    GeometryInstanceData* geomInst, uint32_t numTriangles,
    const MaterialData* materialDataBuffer,
    optixu::NativeBlockBuffer2D<float> dstMip) {
#if USE_PROBABILITY_TEXTURE
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint2 dims = computeProbabilityTextureDimentions(numTriangles);
    if (linearIndex == 0)
        geomInst->emitterPrimDist.setDimensions(dims);
    uint2 idx2D = compute2DFrom1D(dims, linearIndex);
    float importance = 0.0f;
    if (linearIndex < numTriangles)
        importance = computeTriangleImportance(geomInst, linearIndex, materialDataBuffer);
    if (idx2D.x < dims.x && idx2D.y < dims.y)
        dstMip.write(idx2D, importance);
#endif
}

CUDA_DEVICE_KERNEL void computeTriangleProbBuffer(
    GeometryInstanceData* geomInst, uint32_t numTriangles,
    const MaterialData* materialDataBuffer) {
#if !USE_PROBABILITY_TEXTURE
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex == 0)
        geomInst->emitterPrimDist.setNumValues(numTriangles);
    if (linearIndex < numTriangles) {
        float importance = computeTriangleImportance(geomInst, linearIndex, materialDataBuffer);
        geomInst->emitterPrimDist.setWeightAt(linearIndex, importance);
    }
#endif
}



CUDA_DEVICE_FUNCTION CUDA_INLINE float computeGeomInstImportance(
    InstanceData* inst,
    const GeometryInstanceData* geometryInstanceDataBuffer, uint32_t geomInstIndex) {
    uint32_t slot = inst->geomInstSlots[geomInstIndex];
    const GeometryInstanceData &geomInst = geometryInstanceDataBuffer[slot];
    float importance = geomInst.emitterPrimDist.integral();
    return importance;
}

CUDA_DEVICE_KERNEL void computeGeomInstProbTexture(
    InstanceData* inst, uint32_t instIdx, uint32_t numGeomInsts,
    const GeometryInstanceData* geometryInstanceDataBuffer,
    optixu::NativeBlockBuffer2D<float> dstMip) {
#if USE_PROBABILITY_TEXTURE
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint2 dims = computeProbabilityTextureDimentions(numGeomInsts);
    if (linearIndex == 0)
        inst->lightGeomInstDist.setDimensions(dims);
    uint2 idx2D = compute2DFrom1D(dims, linearIndex);
    float importance = 0.0f;
    if (linearIndex < numGeomInsts)
        importance = computeGeomInstImportance(inst, geometryInstanceDataBuffer, linearIndex);
    if (idx2D.x < dims.x && idx2D.y < dims.y)
        dstMip.write(idx2D, importance);
#endif
}

CUDA_DEVICE_KERNEL void computeGeomInstProbBuffer(
    InstanceData* inst, uint32_t instIdx, uint32_t numGeomInsts,
    const GeometryInstanceData* geometryInstanceDataBuffer) {
#if !USE_PROBABILITY_TEXTURE
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex == 0)
        inst->lightGeomInstDist.setNumValues(numGeomInsts);
    if (linearIndex < numGeomInsts) {
        float importance = computeGeomInstImportance(inst, geometryInstanceDataBuffer, linearIndex);
        inst->lightGeomInstDist.setWeightAt(linearIndex, importance);
    }
#endif
}



// TODO: instSlot?
CUDA_DEVICE_FUNCTION CUDA_INLINE float computeInstImportance(
    const InstanceData* instanceDataBuffer, uint32_t instIndex) {
    const InstanceData &inst = instanceDataBuffer[instIndex];
    float3 scale;
    inst.transform.decompose(&scale, nullptr, nullptr);
    float uniformScale = scale.x;
    float importance = inst.lightGeomInstDist.integral();
    return importance;
}

CUDA_DEVICE_KERNEL void computeInstProbTexture(
    ProbabilityTexture* lightInstDist, uint32_t numInsts,
    const InstanceData* instanceDataBuffer,
    optixu::NativeBlockBuffer2D<float> dstMip) {
#if USE_PROBABILITY_TEXTURE
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint2 dims = computeProbabilityTextureDimentions(numInsts);
    if (linearIndex == 0)
        lightInstDist->setDimensions(dims);
    uint2 idx2D = compute2DFrom1D(dims, linearIndex);
    float importance = 0.0f;
    if (linearIndex < numInsts)
        importance = computeInstImportance(instanceDataBuffer, linearIndex);
    if (idx2D.x < dims.x && idx2D.y < dims.y)
        dstMip.write(idx2D, importance);
#endif
}

CUDA_DEVICE_KERNEL void computeInstProbBuffer(
    DiscreteDistribution1D* lightInstDist, uint32_t numInsts,
    const InstanceData* instanceDataBuffer) {
#if !USE_PROBABILITY_TEXTURE
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex == 0)
        lightInstDist->setNumValues(numInsts);
    if (linearIndex < numInsts) {
        float importance = computeInstImportance(instanceDataBuffer, linearIndex);
        lightInstDist->setWeightAt(linearIndex, importance);
    }
#endif
}



CUDA_DEVICE_KERNEL void computeProbabilityTextureMip(
    ProbabilityTexture* probTex, uint32_t dstMipLevel,
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
    if (globalIndex.x < dstDims.x && globalIndex.y < dstDims.y) {
        dstMip.write(globalIndex, sum);
        if (dstMipLevel == numMipLevels - 1)
            probTex->setIntegral(sum);
    }
}

CUDA_DEVICE_KERNEL void finalizeDiscreteDistribution1D(
    DiscreteDistribution1D* lightInstDist) {
#if !defined(USE_WALKER_ALIAS_METHOD)
    if (threadIdx.x == 0)
        lightInstDist->finalize();
#endif
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
