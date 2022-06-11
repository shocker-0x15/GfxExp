#define PURE_CUDA
#include "../svgf_shared.h"

using namespace shared;

CUDA_DEVICE_FUNCTION CUDA_INLINE float calcDepthWeight(
    float nbDepth, float depth,
    float dzdx, float dzdy, int32_t dx, int32_t dy) {
    constexpr float sigma_z = 1.0f;
    constexpr float eps = 1e-6f;
    return std::exp(-std::fabs(nbDepth - depth) / (sigma_z * std::fabs(dzdx * dx + dzdy * dy) + eps));
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float calcNormalWeight(
    const float3 &nbNormal, const float3 &normal) {
    constexpr float sigma_n = 128;
    return std::pow(std::fmax(0.0f, dot(nbNormal, normal)), sigma_n);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float calcLuminanceWeight(
    float nbLuminance, float luminance,
    float localMeanStdDev) {
    constexpr float sigma_l = 4.0f;
    constexpr float eps = 1e-6f;
    return std::exp(-std::fabs(nbLuminance - luminance) / (sigma_l * localMeanStdDev + eps));
}



CUDA_DEVICE_FUNCTION CUDA_INLINE void estimateVariance_generic() {
    const auto glPix = [](int2 pix) {
        return make_int2(pix.x, plp.s->imageSize.y - 1 - pix.y);
    };

    int2 launchIndex = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
                                 blockDim.y * blockIdx.y + threadIdx.y);
    int2 imageSize = plp.s->imageSize;
    int2 pix = make_int2(launchIndex.x, launchIndex.y);
    bool valid = pix.x >= 0 && pix.y >= 0 && pix.x < imageSize.x && pix.y < imageSize.y;
    if (!valid)
        return;

    uint32_t curBufIdx = plp.f->bufferIndex;
    const StaticPipelineLaunchParameters::TemporalSet &staticTemporalSet =
        plp.s->temporalSets[curBufIdx];
    const PerFramePipelineLaunchParameters::TemporalSet &perFrameTemporalSet =
        plp.f->temporalSets[curBufIdx];

    MomentPair_SampleInfo momentPair_sampleInfo =
        staticTemporalSet.momentPair_sampleInfo_buffer.read(pix);
    float firstMoment = momentPair_sampleInfo.firstMoment;
    float secondMoment = momentPair_sampleInfo.secondMoment;
    const SampleInfo &sampleInfo = momentPair_sampleInfo.sampleInfo;

    if (sampleInfo.count < 4) {
        // JP: 空間的な分散推定へのフォールバック
        // EN: Fallback to spatial estimate of variance
        constexpr float filterKernel[] = {
            0.00598, 0.060626, 0.241843, 0.383103, 0.241843, 0.060626, 0.00598
        };

        constexpr float centerWeight = pow2(filterKernel[3]);
        float sumFirstMoments = centerWeight * firstMoment;
        float sumSecondMoments = centerWeight * secondMoment;

        float depth = perFrameTemporalSet.depthBuffer.read(glPix(pix));
        int32_t dx = pix.x < imageSize.x / 2 ? 1 : -1;
        int32_t dy = pix.y < imageSize.y / 2 ? 1 : -1;
        float hnbDepth = perFrameTemporalSet.depthBuffer.read(glPix(pix + make_int2(dx, 0)));
        float vnbDepth = perFrameTemporalSet.depthBuffer.read(glPix(pix + make_int2(0, dy)));
        float dzdx = (hnbDepth - depth) * dx;
        float dzdy = (vnbDepth - depth) * dy;
        float3 normal = perFrameTemporalSet.GBuffer1.read(glPix(pix)).normalInWorld;

        // 7x7 Bilateral Filter driven by depth and normal.
        float sumWeights = centerWeight;
        for (int i = -3; i <= 3; ++i) {
            int nbPixY = pix.y + i;
            if (nbPixY < 0 || nbPixY >= imageSize.y)
                continue;
            float hy = filterKernel[i + 3];

            for (int j = -3; j <= 3; ++j) {
                int nbPixX = pix.x + j;
                if (nbPixX < 0 || nbPixX >= imageSize.x)
                    continue;

                if (i == 0 && j == 0)
                    continue;

                float hx = filterKernel[j + 3];

                int2 nbPix = make_int2(nbPixX, nbPixY);
                float nbDepth = perFrameTemporalSet.depthBuffer.read(glPix(nbPix));
                float3 nbNormal = perFrameTemporalSet.GBuffer1.read(glPix(nbPix)).normalInWorld;

                float wz = calcDepthWeight(nbDepth, depth, dzdx, dzdy, j, i);
                float wn = calcNormalWeight(nbNormal, normal);
                float weight = hx * hy * wz * wn;

                MomentPair_SampleInfo nb_momentPair_sampleInfo =
                    staticTemporalSet.momentPair_sampleInfo_buffer.read(nbPix);
                sumFirstMoments += weight * nb_momentPair_sampleInfo.firstMoment;
                sumSecondMoments += weight * nb_momentPair_sampleInfo.secondMoment;
                sumWeights += weight;
            }
        }
        firstMoment = sumFirstMoments / sumWeights;
        secondMoment = sumSecondMoments / sumWeights;

        //if (plp.f->mousePosition == launchIndex) {
        //    printf("%2u (%4u, %4u): norm: (%g, %g, %g), 1st: %g, 2nd: %g\n",
        //           plp.f->frameIndex, launchIndex.x, launchIndex.y,
        //           vector3Arg(normal),
        //           firstMoment, secondMoment);
        //}
    }

    // V[X] = E[X^2] - E[X]^2
    float variance = max(secondMoment - pow2(firstMoment), 0.0f);
    plp.s->lighting_variance_buffers[0].writeComp<3>(pix, variance);
}

CUDA_DEVICE_KERNEL void estimateVariance() {
    estimateVariance_generic();
}



enum ATrousKernelType {
    ATrousKernelType_Box3x3 = 0,
    ATrousKernelType_Gauss3x3,
    ATrousKernelType_Gauss5x5,
};

template <ATrousKernelType kernelType>
struct ATrousKernel {};

template <>
struct ATrousKernel<ATrousKernelType_Box3x3> {
    CUDA_DEVICE_FUNCTION constexpr static float Weights(uint32_t idx) {
        constexpr float _Weights[] = {
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
        };
        return _Weights[idx];
    }
    CUDA_DEVICE_FUNCTION constexpr static int2 Offsets(uint32_t idx) {
        constexpr int2 _Offsets[] = {
            int2{-1, -1}, int2{0, -1}, int2{1, -1},
            int2{-1,  0}, int2{0,  0}, int2{1,  0},
            int2{-1,  1}, int2{0,  1}, int2{1,  1},
        };
        return _Offsets[idx];
    }
    CUDA_DEVICE_FUNCTION constexpr static uint32_t Size() {
        return 9;
    }
    static constexpr uint32_t centerIndex = 4;
};
template <>
struct ATrousKernel<ATrousKernelType_Gauss3x3> {
    CUDA_DEVICE_FUNCTION constexpr static float Weights(uint32_t idx) {
        constexpr float _Weights[] = {
            1 / 16.0f, 1 / 8.0f, 1 / 16.0f,
            1 / 8.0f, 1 / 4.0f, 1 / 8.0f,
            1 / 16.0f, 1 / 8.0f, 1 / 16.0f,
        };
        return _Weights[idx];
    }
    CUDA_DEVICE_FUNCTION constexpr static int2 Offsets(uint32_t idx) {
        constexpr int2 _Offsets[] = {
            int2{-1, -1}, int2{0, -1}, int2{1, -1},
            int2{-1,  0}, int2{0,  0}, int2{1,  0},
            int2{-1,  1}, int2{0,  1}, int2{1,  1},
        };
        return _Offsets[idx];
    }
    CUDA_DEVICE_FUNCTION constexpr static uint32_t Size() {
        return 9;
    }
    static constexpr uint32_t centerIndex = 4;
};
template <>
struct ATrousKernel<ATrousKernelType_Gauss5x5> {
    CUDA_DEVICE_FUNCTION constexpr static float Weights(uint32_t idx) {
        constexpr float _Weights[] = {
            1 / 256.0f,  4 / 256.0f,  6 / 256.0f,  4 / 256.0f, 1 / 256.0f,
            4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f,
            6 / 256.0f, 24 / 256.0f, 36 / 256.0f, 24 / 256.0f, 6 / 256.0f,
            4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f,
            1 / 256.0f,  4 / 256.0f,  6 / 256.0f,  4 / 256.0f, 1 / 256.0f,
        };
        return _Weights[idx];
    }
    CUDA_DEVICE_FUNCTION constexpr static int2 Offsets(uint32_t idx) {
        constexpr int2 _Offsets[] = {
            int2{-2, -2}, int2{-1, -2}, int2{0, -2}, int2{1, -2}, int2{2, -2},
            int2{-2, -1}, int2{-1, -1}, int2{0, -1}, int2{1, -1}, int2{2, -1},
            int2{-2,  0}, int2{-1,  0}, int2{0,  0}, int2{1,  0}, int2{2,  0},
            int2{-2,  1}, int2{-1,  1}, int2{0,  1}, int2{1,  1}, int2{2,  1},
            int2{-2,  2}, int2{-1,  2}, int2{0,  2}, int2{1,  2}, int2{2,  2},
        };
        return _Offsets[idx];
    }
    CUDA_DEVICE_FUNCTION constexpr static uint32_t Size() {
        return 25;
    }
    static constexpr uint32_t centerIndex = 12;
};

template <ATrousKernelType kernelType>
CUDA_DEVICE_FUNCTION void applyATrousFilter_generic(uint32_t filterStageIndex) {
    int2 launchIndex = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
                                 blockDim.y * blockIdx.y + threadIdx.y);
    int2 pix = make_int2(launchIndex.x, launchIndex.y);
    int2 imageSize = plp.s->imageSize;
    bool valid = pix.x >= 0 && pix.y >= 0 && pix.x < imageSize.x && pix.y < imageSize.y;
    if (!valid)
        return;

    constexpr int32_t stepWidths[] = {
#if 1
        1, 2, 4, 8, 16,
#else
        1, 2, 5, 11, 24
#endif
    };
    const int32_t stepWidth = stepWidths[filterStageIndex];

    uint32_t curBufIdx = plp.f->bufferIndex;
    //const StaticPipelineLaunchParameters::TemporalSet &staticTemporalSet =
    //    plp.s->temporalSets[curBufIdx];
    const PerFramePipelineLaunchParameters::TemporalSet &perFrameTemporalSet =
        plp.f->temporalSets[curBufIdx];

    optixu::NativeBlockBuffer2D<Lighting_Variance> &src_lighting_variance_buffer =
        plp.s->lighting_variance_buffers[filterStageIndex % 2];
    optixu::NativeBlockBuffer2D<Lighting_Variance> &dst_lighting_variance_buffer =
        plp.s->lighting_variance_buffers[(filterStageIndex + 1) % 2];



    // ----------------------------------------------------------------
    // JP: 現在のピクセルの情報を求める。
    // EN: Obtain the current pixel's information.

    Lighting_Variance src_lighting_var = src_lighting_variance_buffer.read(pix);
    float luminance = sRGB_calcLuminance(src_lighting_var.noisyLighting);

    float depth = perFrameTemporalSet.depthBuffer.read(glPix(pix));
    int32_t dx = pix.x < imageSize.x / 2 ? 1 : -1;
    int32_t dy = pix.y < imageSize.y / 2 ? 1 : -1;
    float hnbDepth = perFrameTemporalSet.depthBuffer.read(glPix(pix + make_int2(dx, 0)));
    float vnbDepth = perFrameTemporalSet.depthBuffer.read(glPix(pix + make_int2(0, dy)));
    float dzdx = (hnbDepth - depth) * dx;
    float dzdy = (vnbDepth - depth) * dy;
    float3 normal = perFrameTemporalSet.GBuffer1.read(glPix(pix)).normalInWorld;

    // JP: 安定化のため分散は3x3のガウシアンフィルターにかける。
    // EN: Apply 3x3 Gaussian filter to variance for stabilization.
    constexpr float gaussKernel[] = {
        1 / 4.0f, 1 / 2.0f, 1 / 4.0f
    };
    float localMeanVar = 0.0f;
#pragma unroll
    for (int i = -1; i <= 1; ++i) {
        int nbPixY = clamp(pix.y + i, 0, imageSize.y - 1);
        float hy = gaussKernel[i + 1];
        for (int j = -1; j <= 1; ++j) {
            int nbPixX = clamp(pix.x + j, 0, imageSize.x - 1);
            float hx = gaussKernel[j + 1];
            int2 nbPix = make_int2(nbPixX, nbPixY);
            localMeanVar += hx * hy * src_lighting_variance_buffer.read(nbPix).variance;
        }
    }
    float localMeanStdDev = std::sqrt(localMeanVar);

    // END: Obtain the current pixel's information.
    // ----------------------------------------------------------------



    // JP: カラーと分散をA-Trousフィルターにかける。
    // EN: Apply A-Trous filter to color and variance.
    using Kernel = ATrousKernel<kernelType>;
    constexpr float centerWeight = Kernel::Weights(Kernel::centerIndex);
    float sumWeights = centerWeight;
    Lighting_Variance dst_lighting_var;
    dst_lighting_var.denoisedLighting = centerWeight * src_lighting_var.noisyLighting;
    dst_lighting_var.variance = pow2(centerWeight) * src_lighting_var.variance;
#pragma unroll
    for (int i = 0; i < Kernel::Size(); ++i) {
        if (i == Kernel::centerIndex)
            continue;

        int2 offset = make_int2(Kernel::Offsets(i).x * stepWidth, Kernel::Offsets(i).y * stepWidth);
        int2 nbPix = make_int2(pix.x + offset.x, pix.y + offset.y);
        if (nbPix.x < 0 || nbPix.x >= imageSize.x ||
            nbPix.y < 0 || nbPix.y >= imageSize.y)
            continue;

        float h = Kernel::Weights(i);

        float nbDepth = perFrameTemporalSet.depthBuffer.read(glPix(nbPix));
        float3 nbNormal = perFrameTemporalSet.GBuffer1.read(glPix(nbPix)).normalInWorld;

        float wz = calcDepthWeight(nbDepth, depth, dzdx, dzdy, offset.x, offset.y);
        float wn = calcNormalWeight(nbNormal, normal);
        //if (h * wz * wn < 1e-6f)
        //    continue;

        Lighting_Variance nb_lighting_var = src_lighting_variance_buffer.read(nbPix);
        float nbLuminance = sRGB_calcLuminance(nb_lighting_var.noisyLighting);
        float wl = calcLuminanceWeight(nbLuminance, luminance, localMeanStdDev);

        float weight = h * wz * wn * wl;
        dst_lighting_var.denoisedLighting += weight * nb_lighting_var.noisyLighting;
        dst_lighting_var.variance += pow2(weight) * nb_lighting_var.variance;
        sumWeights += weight;
    }
    dst_lighting_var.denoisedLighting /= sumWeights;
    dst_lighting_var.variance /= pow2(sumWeights);

    // TODO: Varianceは最終レベルは出力の必要なし。
    dst_lighting_variance_buffer.write(pix, dst_lighting_var);

    if (filterStageIndex == 0)
        plp.s->prevNoisyLightingBuffer.write(pix, dst_lighting_var);
}

CUDA_DEVICE_KERNEL void applyATrousFilter_box3x3(uint32_t filterStageIndex) {
    applyATrousFilter_generic<ATrousKernelType_Box3x3>(filterStageIndex);
}



CUDA_DEVICE_FUNCTION void reprojectPreviousAccumulation(
    const optixu::NativeBlockBuffer2D<float4> &prevFinalLightingBuffer, float2 prevScreenPos,
    float3* prevFinalLighting, bool* outOfScreen) {
    *prevFinalLighting = make_float3(0.0f, 0.0f, 0.0f);
    *outOfScreen = (prevScreenPos.x < 0.0f || prevScreenPos.y < 0.0f ||
                    prevScreenPos.x >= 1.0f || prevScreenPos.y >= 1.0f);
    if (*outOfScreen)
        return;

    int2 imageSize = plp.s->imageSize;

    float2 prevViewportPos = make_float2(imageSize.x * prevScreenPos.x, imageSize.y * prevScreenPos.y);
    int2 prevPixPos = make_int2(prevViewportPos);

    int2 ulPos = make_int2(prevPixPos.x, prevPixPos.y);
    int2 urPos = make_int2(min(prevPixPos.x + 1, imageSize.x - 1), prevPixPos.y);
    int2 llPos = make_int2(prevPixPos.x, min(prevPixPos.y + 1, imageSize.y - 1));
    int2 lrPos = make_int2(min(prevPixPos.x + 1, imageSize.x - 1),
                           min(prevPixPos.y + 1, imageSize.y - 1));

    float sumWeights = 0.0f;
    float s = clamp((prevViewportPos.x - 0.5f) - prevPixPos.x, 0.0f, 1.0f);
    float t = clamp((prevViewportPos.y - 0.5f) - prevPixPos.y, 0.0f, 1.0f);

    //{
    //    int2 launchIndex = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
    //                                 blockDim.y * blockIdx.y + threadIdx.y);
    //    if (launchIndex == plp.f->mousePosition) {
    //        printf("m: %4u, %4u, prev: %6.1f, %6.1f: %.3f, %.3f\n",
    //               vector2Arg(launchIndex), vector2Arg(prevViewportPos),
    //               s, t);
    //    }
    //}

    // Upper Left
    {
        float weight = (1 - s) * (1 - t);
        *prevFinalLighting += weight * make_float3(prevFinalLightingBuffer.read(glPix(ulPos)));
        sumWeights += weight;
    }
    // Upper Right
    {
        float weight = s * (1 - t);
        *prevFinalLighting += weight * make_float3(prevFinalLightingBuffer.read(glPix(urPos)));
        sumWeights += weight;
    }
    // Lower Left
    {
        float weight = (1 - s) * t;
        *prevFinalLighting += weight * make_float3(prevFinalLightingBuffer.read(glPix(llPos)));
        sumWeights += weight;
    }
    // Lower Right
    {
        float weight = s * t;
        *prevFinalLighting += weight * make_float3(prevFinalLightingBuffer.read(glPix(lrPos)));
        sumWeights += weight;
    }

    *prevFinalLighting = safeDivide(*prevFinalLighting, sumWeights);
}

CUDA_DEVICE_KERNEL void applyAlbedoModulationAndTemporalAntiAliasing(uint32_t numFilteringStages) {
    int2 launchIndex = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
                                 blockDim.y * blockIdx.y + threadIdx.y);
    int2 imageSize = plp.s->imageSize;
    int2 pix = make_int2(launchIndex.x, launchIndex.y);
    bool valid = pix.x >= 0 && pix.y >= 0 && pix.x < imageSize.x && pix.y < imageSize.y;
    if (!valid)
        return;

    uint32_t curBufIdx = plp.f->bufferIndex;
    uint32_t prevBufIdx = (plp.f->bufferIndex + 1) % 2;
    //const StaticPipelineLaunchParameters::TemporalSet &curStaticTemporalSet =
    //    plp.s->temporalSets[curBufIdx];
    const PerFramePipelineLaunchParameters::TemporalSet &curPerFrameTemporalSet =
        plp.f->temporalSets[curBufIdx];
    const PerFramePipelineLaunchParameters::TemporalSet &prevPerFrameTemporalSet =
        plp.f->temporalSets[prevBufIdx];

    optixu::NativeBlockBuffer2D<Lighting_Variance> &src_lighting_variance_buffer =
        plp.s->lighting_variance_buffers[plp.f->enableSVGF ? numFilteringStages % 2 : 0];

    Lighting_Variance src_lighting_var = src_lighting_variance_buffer.read(pix);
    Albedo albedo = plp.s->albedoBuffer.read(pix);
    float3 finalLighting = src_lighting_var.denoisedLighting;
    if (plp.f->modulateAlbedo)
        finalLighting *= albedo.dhReflectance;

    if (plp.f->enableTemporalAA && !plp.f->isFirstFrame) {
        GBuffer2 gBuffer2 = curPerFrameTemporalSet.GBuffer2.read(glPix(pix));
        float2 prevScreenPos = gBuffer2.prevScreenPos;

        const optixu::NativeBlockBuffer2D<float4> &prevFinalLightingBuffer =
            prevPerFrameTemporalSet.finalLightingBuffer;

        float3 prevFinalLighting;
        bool prevWasOutOfScreen;
        reprojectPreviousAccumulation(
            prevFinalLightingBuffer, prevScreenPos,
            &prevFinalLighting, &prevWasOutOfScreen);

        float3 nbBoxMin = finalLighting;
        float3 nbBoxMax = finalLighting;
        float3 nbCrossMin = finalLighting;
        float3 nbCrossMax = finalLighting;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0)
                    continue;
                int2 nbPix = make_int2(clamp<int32_t>(pix.x + j, 0, imageSize.x - 1),
                                       clamp<int32_t>(pix.y + i, 0, imageSize.y - 1));
                Lighting_Variance nbSrc_lighting_var = src_lighting_variance_buffer.read(nbPix);
                Albedo nbAlbedo = plp.s->albedoBuffer.read(nbPix);
                float3 nbValue = nbSrc_lighting_var.denoisedLighting;
                if (plp.f->modulateAlbedo)
                    nbValue *= nbAlbedo.dhReflectance;
                nbBoxMin = min(nbBoxMin, nbValue);
                nbBoxMax = max(nbBoxMax, nbValue);
                if (i == 0 || j == 0) {
                    nbCrossMin = min(nbCrossMin, nbValue);
                    nbCrossMax = max(nbCrossMax, nbValue);
                }
            }
        }
        float3 nbMin = 0.5f * (nbBoxMin + nbCrossMin);
        float3 nbMax = 0.5f * (nbBoxMax + nbCrossMax);
        prevFinalLighting = clamp(prevFinalLighting, nbMin, nbMax);

        float curWeight = 1.0f / plp.f->taaHistoryLength; // Exponential Moving Average
        //if (sampleCount < plp.f->taaHistoryLength) // Cumulative Moving Average
        //    curWeight = 1.0f / sampleCount;
        float prevWeight = 1.0f - curWeight;
        finalLighting = prevWeight * prevFinalLighting + curWeight * finalLighting;
    }

    const optixu::NativeBlockBuffer2D<float4> &curFinalLightingBuffer =
        curPerFrameTemporalSet.finalLightingBuffer;
    curFinalLightingBuffer.write(glPix(pix), make_float4(finalLighting, 1.0f));
}
