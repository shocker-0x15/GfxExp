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
    const Normal3D &nbNormal, const Normal3D &normal) {
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

    const int2 launchIndex = make_int2(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 imageSize = plp.s->imageSize;
    const int2 pix = make_int2(launchIndex.x, launchIndex.y);
    const bool valid = pix.x >= 0 && pix.y >= 0 && pix.x < imageSize.x && pix.y < imageSize.y;
    if (!valid)
        return;

    const uint32_t curBufIdx = plp.f->bufferIndex;
    const StaticPipelineLaunchParameters::TemporalSet &staticTemporalSet =
        plp.s->temporalSets[curBufIdx];
    const PerFramePipelineLaunchParameters::TemporalSet &perFrameTemporalSet =
        plp.f->temporalSets[curBufIdx];

    const uint32_t matSlot = perFrameTemporalSet.GBuffer2.read(glPix(pix)).materialSlot;
    if (matSlot == 0xFFFFFFFF)
        return;

    const MomentPair_SampleInfo momentPair_sampleInfo =
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

        const float depth = perFrameTemporalSet.depthBuffer.read(glPix(pix));
        const int32_t dx = pix.x < imageSize.x / 2 ? 1 : -1;
        const int32_t dy = pix.y < imageSize.y / 2 ? 1 : -1;
        const float hnbDepth = perFrameTemporalSet.depthBuffer.read(glPix(pix + make_int2(dx, 0)));
        const float vnbDepth = perFrameTemporalSet.depthBuffer.read(glPix(pix + make_int2(0, dy)));
        const float dzdx = (hnbDepth - depth) * dx;
        const float dzdy = (vnbDepth - depth) * dy;
        const Normal3D normal = perFrameTemporalSet.GBuffer1.read(glPix(pix)).normalInWorld;

        // 7x7 Bilateral Filter driven by depth and normal.
        float sumWeights = centerWeight;
        for (int i = -3; i <= 3; ++i) {
            const int nbPixY = pix.y + i;
            if (nbPixY < 0 || nbPixY >= imageSize.y)
                continue;
            const float hy = filterKernel[i + 3];

            for (int j = -3; j <= 3; ++j) {
                const int nbPixX = pix.x + j;
                if (nbPixX < 0 || nbPixX >= imageSize.x)
                    continue;

                if (i == 0 && j == 0)
                    continue;

                const float hx = filterKernel[j + 3];

                const int2 nbPix = make_int2(nbPixX, nbPixY);
                const float nbDepth = perFrameTemporalSet.depthBuffer.read(glPix(nbPix));
                if (nbDepth == 1.0f)
                    continue;
                const Normal3D nbNormal = perFrameTemporalSet.GBuffer1.read(glPix(nbPix)).normalInWorld;

                const float wz = calcDepthWeight(nbDepth, depth, dzdx, dzdy, j, i);
                const float wn = calcNormalWeight(nbNormal, normal);
                const float weight = hx * hy * wz * wn;

                const MomentPair_SampleInfo nb_momentPair_sampleInfo =
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
    const float variance = max(secondMoment - pow2(firstMoment), 0.0f);
    constexpr size_t byteOffset = offsetof(Lighting_Variance, variance);
    plp.s->lighting_variance_buffers[0].writePartially<byteOffset>(pix, variance);
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
    const int2 launchIndex = make_int2(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 pix = make_int2(launchIndex.x, launchIndex.y);
    const int2 imageSize = plp.s->imageSize;
    const bool valid = pix.x >= 0 && pix.y >= 0 && pix.x < imageSize.x && pix.y < imageSize.y;
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

    const uint32_t curBufIdx = plp.f->bufferIndex;
    //const StaticPipelineLaunchParameters::TemporalSet &staticTemporalSet =
    //    plp.s->temporalSets[curBufIdx];
    const PerFramePipelineLaunchParameters::TemporalSet &perFrameTemporalSet =
        plp.f->temporalSets[curBufIdx];

    optixu::NativeBlockBuffer2D<Lighting_Variance> &src_lighting_variance_buffer =
        plp.s->lighting_variance_buffers[filterStageIndex % 2];
    optixu::NativeBlockBuffer2D<Lighting_Variance> &dst_lighting_variance_buffer =
        plp.s->lighting_variance_buffers[(filterStageIndex + 1) % 2];

    const uint32_t matSlot = perFrameTemporalSet.GBuffer2.read(glPix(pix)).materialSlot;
    if (matSlot == 0xFFFFFFFF)
        return;



    // ----------------------------------------------------------------
    // JP: 現在のピクセルの情報を求める。
    // EN: Obtain the current pixel's information.

    const Lighting_Variance src_lighting_var = src_lighting_variance_buffer.read(pix);
    if (filterStageIndex == 0 && !plp.f->feedback1stFilteredResult)
        plp.s->prevNoisyLightingBuffer.write(pix, src_lighting_var);
    const float luminance = sRGB_calcLuminance(src_lighting_var.noisyLighting);

    const float depth = perFrameTemporalSet.depthBuffer.read(glPix(pix));
    const int32_t dx = pix.x < imageSize.x / 2 ? 1 : -1;
    const int32_t dy = pix.y < imageSize.y / 2 ? 1 : -1;
    const float hnbDepth = perFrameTemporalSet.depthBuffer.read(glPix(pix + make_int2(dx, 0)));
    const float vnbDepth = perFrameTemporalSet.depthBuffer.read(glPix(pix + make_int2(0, dy)));
    const float dzdx = (hnbDepth - depth) * dx;
    const float dzdy = (vnbDepth - depth) * dy;
    const Normal3D normal = perFrameTemporalSet.GBuffer1.read(glPix(pix)).normalInWorld;

    // JP: 安定化のため分散は3x3のガウシアンフィルターにかける。
    // EN: Apply 3x3 Gaussian filter to variance for stabilization.
    constexpr float gaussKernel[] = {
        1 / 4.0f, 1 / 2.0f, 1 / 4.0f
    };
    float sumLocalVars = 0.0f;
    float sumVarWeights = 0.0f;
#pragma unroll
    for (int i = -1; i <= 1; ++i) {
        const int nbPixY = clamp(pix.y + i, 0, imageSize.y - 1);
        const float hy = gaussKernel[i + 1];
        for (int j = -1; j <= 1; ++j) {
            const int nbPixX = clamp(pix.x + j, 0, imageSize.x - 1);
            const float hx = gaussKernel[j + 1];
            const int2 nbPix = make_int2(nbPixX, nbPixY);
            const float weight = hx * hy;
            sumLocalVars += weight * src_lighting_variance_buffer.read(nbPix).variance;
            sumVarWeights += weight;
        }
    }
    const float localMeanStdDev = std::sqrt(sumLocalVars / sumVarWeights);

    // END: Obtain the current pixel's information.
    // ----------------------------------------------------------------



    // JP: カラーと分散をà-trousフィルターにかける。
    // EN: Apply à-trous filter to color and variance.
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

        const int2 offset = make_int2(Kernel::Offsets(i).x * stepWidth, Kernel::Offsets(i).y * stepWidth);
        const int2 nbPix = make_int2(pix.x + offset.x, pix.y + offset.y);
        if (nbPix.x < 0 || nbPix.x >= imageSize.x ||
            nbPix.y < 0 || nbPix.y >= imageSize.y)
            continue;

        const float h = Kernel::Weights(i);

        const float nbDepth = perFrameTemporalSet.depthBuffer.read(glPix(nbPix));
        if (nbDepth == 1.0f)
            continue;
        const Normal3D nbNormal = perFrameTemporalSet.GBuffer1.read(glPix(nbPix)).normalInWorld;

        const float wz = calcDepthWeight(nbDepth, depth, dzdx, dzdy, offset.x, offset.y);
        const float wn = calcNormalWeight(nbNormal, normal);
        //if (h * wz * wn < 1e-6f)
        //    continue;

        const Lighting_Variance nb_lighting_var = src_lighting_variance_buffer.read(nbPix);
        const float nbLuminance = sRGB_calcLuminance(nb_lighting_var.noisyLighting);
        const float wl = calcLuminanceWeight(nbLuminance, luminance, localMeanStdDev);

        const float weight = h * wz * wn * wl;
        dst_lighting_var.denoisedLighting += weight * nb_lighting_var.noisyLighting;
        dst_lighting_var.variance += pow2(weight) * nb_lighting_var.variance;
        sumWeights += weight;
    }
    dst_lighting_var.denoisedLighting /= sumWeights;
    dst_lighting_var.variance /= pow2(sumWeights);

    dst_lighting_variance_buffer.write(pix, dst_lighting_var);

    if (filterStageIndex == 0 && plp.f->feedback1stFilteredResult)
        plp.s->prevNoisyLightingBuffer.write(pix, dst_lighting_var);
}

CUDA_DEVICE_KERNEL void applyATrousFilter_box3x3(uint32_t filterStageIndex) {
    applyATrousFilter_generic<ATrousKernelType_Box3x3>(filterStageIndex);
}



// for the case where SVGF is disabled and temporal accumulation is enabled.
CUDA_DEVICE_KERNEL void feedbackNoisyLighting() {
    const int2 launchIndex = make_int2(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 pix = make_int2(launchIndex.x, launchIndex.y);
    const int2 imageSize = plp.s->imageSize;
    const bool valid = pix.x >= 0 && pix.y >= 0 && pix.x < imageSize.x && pix.y < imageSize.y;
    if (!valid)
        return;

    const optixu::NativeBlockBuffer2D<Lighting_Variance> &src_lighting_variance_buffer =
        plp.s->lighting_variance_buffers[0];

    const Lighting_Variance lighting_var = src_lighting_variance_buffer.read(pix);
    plp.s->prevNoisyLightingBuffer.write(launchIndex, lighting_var);
}



CUDA_DEVICE_KERNEL void fillBackground(uint32_t numFilteringStages) {
    const int2 launchIndex = make_int2(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 imageSize = plp.s->imageSize;
    const int2 pix = make_int2(launchIndex.x, launchIndex.y);
    const bool valid = pix.x >= 0 && pix.y >= 0 && pix.x < imageSize.x && pix.y < imageSize.y;
    if (!valid)
        return;

    const uint32_t curBufIdx = plp.f->bufferIndex;
    const uint32_t prevBufIdx = (plp.f->bufferIndex + 1) % 2;
    const PerFramePipelineLaunchParameters::TemporalSet &curPerFrameTemporalSet =
        plp.f->temporalSets[curBufIdx];
    const PerFramePipelineLaunchParameters::TemporalSet &prevPerFrameTemporalSet =
        plp.f->temporalSets[prevBufIdx];

    const optixu::NativeBlockBuffer2D<Lighting_Variance> &dst_lighting_variance_buffer =
        plp.s->lighting_variance_buffers[plp.f->enableSVGF ? numFilteringStages % 2 : 0];

    const bool useEnvLight = plp.s->envLightTexture && plp.f->enableEnvLight;

    const GBuffer2Elements gb2Elems = curPerFrameTemporalSet.GBuffer2.read(glPix(pix));
    if (gb2Elems.materialSlot != 0xFFFFFFFF)
        return;

    const auto computeEnvDirection = [&]
    (const PerspectiveCamera &tgtCamera, int2 tgtPix, bool withSubPixelOffset) {
        const float2 fpix = make_float2(
            tgtPix.x + (withSubPixelOffset ? tgtCamera.subPixelOffset.x : 0.5f),
            tgtPix.y + (withSubPixelOffset ? (1 - tgtCamera.subPixelOffset.y) : 0.5f));
        const float x = fpix.x / imageSize.x;
        const float y = fpix.y / imageSize.y;
        const float vh = 2 * std::tan(tgtCamera.fovY * 0.5f);
        const float vw = tgtCamera.aspect * vh;

        const Vector3D direction = normalize(tgtCamera.orientation * Vector3D(vw * (0.5f - x), vh * (0.5f - y), 1));
        return direction;
    };

    const auto computeEnvMapTexCoord = [&]
    (const Vector3D &direction) {
        float posPhi, posTheta;
        toPolarYUp(direction, &posPhi, &posTheta);

        float phi = posPhi + plp.f->envLightRotation;
        if (useEnvLight)
            phi += plp.f->envLightRotation;

        float u = phi / (2 * Pi);
        u -= floorf(u);
        const float v = posTheta / Pi;

        return Point2D(u, v);
    };

    RGB finalLighting(0.001f, 0.001f, 0.001f);
    if (useEnvLight) {
        const Vector3D direction = computeEnvDirection(curPerFrameTemporalSet.camera, pix, true);
        const Point2D texCoord = computeEnvMapTexCoord(direction);
        const float4 texValue = tex2DLod<float4>(
            plp.s->envLightTexture, texCoord.x, texCoord.y, 0.0f);
        finalLighting = plp.f->envLightPowerCoeff * RGB(getXYZ(texValue));
    }

    Vector3D direction = computeEnvDirection(curPerFrameTemporalSet.camera, pix, false);
    const PerspectiveCamera &prevCamera = prevPerFrameTemporalSet.camera;
    direction = transpose(prevCamera.orientation) * direction;
    direction /= direction.z;
    const float vh = 2 * std::tan(prevCamera.fovY * 0.5f);
    const float vw = prevCamera.aspect * vh;
    const Point2D prevScreenPos(0.5f - direction.x / vw, 0.5f - direction.y / vh);

    Lighting_Variance lighting_var;
    lighting_var.denoisedLighting = finalLighting;
    lighting_var.variance = 0.0f;

    Albedo albedo;
    albedo.dhReflectance = RGB(1.0f, 1.0f, 1.0f);

    dst_lighting_variance_buffer.write(pix, lighting_var);
    plp.s->albedoBuffer.write(pix, albedo);
    curPerFrameTemporalSet.GBuffer2.writePartially<0>(glPix(pix), prevScreenPos);
}



CUDA_DEVICE_FUNCTION void reprojectPreviousAccumulation(
    const optixu::NativeBlockBuffer2D<float4> &prevFinalLightingBuffer, Point2D prevScreenPos,
    RGB* prevFinalLighting, bool* outOfScreen) {
    *prevFinalLighting = RGB(0.0f, 0.0f, 0.0f);
    *outOfScreen = (prevScreenPos.x < 0.0f || prevScreenPos.y < 0.0f ||
                    prevScreenPos.x >= 1.0f || prevScreenPos.y >= 1.0f);
    if (*outOfScreen)
        return;

    const int2 imageSize = plp.s->imageSize;

    const Point2D prevViewportPos(imageSize.x * prevScreenPos.x, imageSize.y * prevScreenPos.y);
    const int2 prevPixPos = make_int2(prevViewportPos.x, prevViewportPos.y);
    const Vector2D fDelta = prevViewportPos - (Point2D(prevPixPos.x, prevPixPos.y) + Point2D(0.5f));
    const int2 delta = make_int2(
        fDelta.x < 0 ? -1 : 1,
        fDelta.y < 0 ? -1 : 1);

    const int2 basePos = make_int2(prevPixPos.x, prevPixPos.y);
    const int2 dxPos = make_int2(clamp(prevPixPos.x + delta.x, 0, imageSize.x - 1), prevPixPos.y);
    const int2 dyPos = make_int2(prevPixPos.x, clamp(prevPixPos.y + delta.y, 0, imageSize.y - 1));
    const int2 dxdyPos = make_int2(
        clamp(prevPixPos.x + delta.x, 0, imageSize.x - 1),
        clamp(prevPixPos.y + delta.y, 0, imageSize.y - 1));

    float sumWeights = 0.0f;
    const float s = std::fabs(fDelta.x);
    const float t = std::fabs(fDelta.y);

    //{
    //    int2 launchIndex = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
    //                                 blockDim.y * blockIdx.y + threadIdx.y);
    //    if (launchIndex == plp.f->mousePosition) {
    //        printf("m: %4u, %4u, prev: %6.1f, %6.1f: %.3f, %.3f\n",
    //               vector2Arg(launchIndex), vector2Arg(prevViewportPos),
    //               s, t);
    //    }
    //}

    // Base
    {
        const float weight = (1 - s) * (1 - t);
        *prevFinalLighting += weight * RGB(getXYZ(prevFinalLightingBuffer.read(glPix(basePos))));
        sumWeights += weight;
    }
    // dx
    {
        const float weight = s * (1 - t);
        *prevFinalLighting += weight * RGB(getXYZ(prevFinalLightingBuffer.read(glPix(dxPos))));
        sumWeights += weight;
    }
    // dy
    {
        const float weight = (1 - s) * t;
        *prevFinalLighting += weight * RGB(getXYZ(prevFinalLightingBuffer.read(glPix(dyPos))));
        sumWeights += weight;
    }
    // dxdy
    {
        const float weight = s * t;
        *prevFinalLighting += weight * RGB(getXYZ(prevFinalLightingBuffer.read(glPix(dxdyPos))));
        sumWeights += weight;
    }

    *prevFinalLighting = safeDivide(*prevFinalLighting, sumWeights);
}

CUDA_DEVICE_KERNEL void applyAlbedoModulationAndTemporalAntiAliasing(uint32_t numFilteringStages) {
    const int2 launchIndex = make_int2(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    const int2 imageSize = plp.s->imageSize;
    const int2 pix = make_int2(launchIndex.x, launchIndex.y);
    const bool valid = pix.x >= 0 && pix.y >= 0 && pix.x < imageSize.x && pix.y < imageSize.y;
    if (!valid)
        return;

    const uint32_t curBufIdx = plp.f->bufferIndex;
    const uint32_t prevBufIdx = (plp.f->bufferIndex + 1) % 2;
    //const StaticPipelineLaunchParameters::TemporalSet &curStaticTemporalSet =
    //    plp.s->temporalSets[curBufIdx];
    const PerFramePipelineLaunchParameters::TemporalSet &curPerFrameTemporalSet =
        plp.f->temporalSets[curBufIdx];
    const PerFramePipelineLaunchParameters::TemporalSet &prevPerFrameTemporalSet =
        plp.f->temporalSets[prevBufIdx];

    const optixu::NativeBlockBuffer2D<Lighting_Variance> &src_lighting_variance_buffer =
        plp.s->lighting_variance_buffers[plp.f->enableSVGF ? numFilteringStages % 2 : 0];

    const Lighting_Variance src_lighting_var = src_lighting_variance_buffer.read(pix);
    const Albedo albedo = plp.s->albedoBuffer.read(pix);
    RGB finalLighting = src_lighting_var.denoisedLighting;
    if (plp.f->modulateAlbedo)
        finalLighting *= albedo.dhReflectance;

    if (plp.f->enableTemporalAA && !plp.f->isFirstFrame) {
        const optixu::NativeBlockBuffer2D<float4> &prevFinalLightingBuffer =
            prevPerFrameTemporalSet.finalLightingBuffer;

        const GBuffer2Elements gb2Elems = curPerFrameTemporalSet.GBuffer2.read(glPix(pix));
        RGB prevFinalLighting;
        bool prevWasOutOfScreen;
        reprojectPreviousAccumulation(
            prevFinalLightingBuffer, gb2Elems.prevScreenPos,
            &prevFinalLighting, &prevWasOutOfScreen);

        RGB nbBoxMin = finalLighting;
        RGB nbBoxMax = finalLighting;
        RGB nbCrossMin = finalLighting;
        RGB nbCrossMax = finalLighting;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0)
                    continue;
                const int2 nbPix = make_int2(
                    clamp<int32_t>(pix.x + j, 0, imageSize.x - 1),
                    clamp<int32_t>(pix.y + i, 0, imageSize.y - 1));
                const Lighting_Variance nbSrc_lighting_var = src_lighting_variance_buffer.read(nbPix);
                const Albedo nbAlbedo = plp.s->albedoBuffer.read(nbPix);
                RGB nbValue = nbSrc_lighting_var.denoisedLighting;
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
        const RGB nbMin = 0.5f * (nbBoxMin + nbCrossMin);
        const RGB nbMax = 0.5f * (nbBoxMax + nbCrossMax);
        prevFinalLighting = clamp(prevFinalLighting, nbMin, nbMax);

        const float curWeight = 1.0f / plp.f->taaHistoryLength; // Exponential Moving Average
        //if (sampleCount < plp.f->taaHistoryLength) // Cumulative Moving Average
        //    curWeight = 1.0f / sampleCount;
        const float prevWeight = 1.0f - curWeight;
        finalLighting = prevWeight * prevFinalLighting + curWeight * finalLighting;
    }

    const optixu::NativeBlockBuffer2D<float4> &curFinalLightingBuffer =
        curPerFrameTemporalSet.finalLightingBuffer;
    curFinalLightingBuffer.write(glPix(pix), make_float4(finalLighting.toNative(), 1.0f));
}
