#include "../neural_radiance_caching_shared.h"

using namespace shared;

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility);
}



CUDA_DEVICE_FUNCTION CUDA_INLINE void convertToPolar(const Vector3D &dir, float* phi, float* theta) {
    float z = std::fmin(std::fmax(dir.z, -1.0f), 1.0f);
    *theta = std::acos(z);
    *phi = std::atan2(dir.y, dir.x);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void createRadianceQuery(
    const Point3D &positionInWorld, const Normal3D &normalInWorld, const Vector3D &scatteredDirInWorld,
    float roughness, const RGB &diffuseReflectance, const RGB &specularReflectance,
    RadianceQuery* query) {
    float phi, theta;
    query->position = plp.s->sceneAABB->normalize(positionInWorld);
    convertToPolar(Vector3D(normalInWorld), &phi, &theta);
    query->normal_phi = phi;
    query->normal_theta = theta;
    convertToPolar(scatteredDirInWorld, &phi, &theta);
    query->vOut_phi = phi;
    query->vOut_theta = theta;
    query->roughness = 1 - std::exp(-roughness);
    query->diffuseReflectance = diffuseReflectance;
    query->specularReflectance = specularReflectance;
}

static constexpr bool useSolidAngleSampling = false;

CUDA_DEVICE_FUNCTION CUDA_INLINE RGB performNextEventEstimation(
    const Point3D &shadingPoint, const Vector3D &vOutLocal, const ReferenceFrame &shadingFrame,
    const BSDF &bsdf, PCG32RNG &rng) {
    float uLight = rng.getFloat0cTo1o();
    bool selectEnvLight = false;
    float probToSampleCurLightType = 1.0f;
    if (plp.s->envLightTexture && plp.f->enableEnvLight) {
        if (plp.s->lightInstDist.integral() > 0.0f) {
            if (uLight < probToSampleEnvLight) {
                probToSampleCurLightType = probToSampleEnvLight;
                uLight /= probToSampleCurLightType;
                selectEnvLight = true;
            }
            else {
                probToSampleCurLightType = 1.0f - probToSampleEnvLight;
                uLight = (uLight - probToSampleEnvLight) / probToSampleCurLightType;
            }
        }
        else {
            selectEnvLight = true;
        }
    }
    LightSample lightSample;
    float areaPDensity;
    sampleLight<useSolidAngleSampling>(
        shadingPoint,
        uLight, selectEnvLight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &lightSample, &areaPDensity);
    areaPDensity *= probToSampleCurLightType;

    Vector3D shadowRay = lightSample.atInfinity ?
        Vector3D(lightSample.position) :
        (lightSample.position - shadingPoint);
    const float dist2 = shadowRay.sqLength();
    shadowRay /= std::sqrt(dist2);
    const Vector3D vInLocal = shadingFrame.toLocal(shadowRay);
    const float lpCos = std::fabs(dot(shadowRay, lightSample.normal));
    float bsdfPDensity = bsdf.evaluatePDF(vOutLocal, vInLocal) * lpCos / dist2;
    if (!stc::isfinite(bsdfPDensity))
        bsdfPDensity = 0.0f;
    const float lightPDensity = areaPDensity;
    const float misWeight = pow2(lightPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
    RGB ret(0.0f);
    if (areaPDensity > 0.0f)
        ret = performDirectLighting<PathTracingRayType, true>(
            shadingPoint, vOutLocal, shadingFrame, bsdf, lightSample) * (misWeight / areaPDensity);
    //if (!allFinite(ret)) {
    //    printf("mis: %g / %g, p:(%g, %g, %g), v:(%g, %g, %g)\n",
    //           misWeight, areaPDensity,
    //           shadingPoint.x, shadingPoint.y, shadingPoint.z,
    //           vOutLocal.x, vOutLocal.y, vOutLocal.z);
    //}

    return ret;
}

template <bool useNRC>
CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_raygen_generic() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = plp.f->bufferIndex;

    const GBuffer0Elements gb0Elems = plp.s->GBuffer0[bufIdx].read(launchIndex);
    const uint32_t instSlot = gb0Elems.instSlot;
    const float bcB = decodeBarycentric(gb0Elems.qbcB);
    const float bcC = decodeBarycentric(gb0Elems.qbcC);

    const PerspectiveCamera &camera = plp.f->camera;

    uint32_t linearTileIndex;
    bool isTrainingPath;
    bool isUnbiasedTrainingTile;
    if constexpr (useNRC) {
        const uint2 tileSize = *plp.s->tileSize[bufIdx];
        const uint32_t numPixelsInTile = tileSize.x * tileSize.y;

        // JP: 動的サイズのタイルごとに1つトレーニングパスを選ぶ。
        // EN: choose a training path for each dynamic-sized tile.
        const uint2 localIndex = launchIndex % tileSize;
        const uint32_t localLinearIndex = localIndex.y * tileSize.x + localIndex.x;
        isTrainingPath = (localLinearIndex + *plp.s->offsetToSelectTrainingPath) % numPixelsInTile == 0;

        const uint2 numTiles = (plp.s->imageSize + tileSize - 1) / tileSize;
        const uint2 tileIndex = launchIndex / tileSize;
        linearTileIndex = tileIndex.y * numTiles.x + tileIndex.x;

        // JP: トレーニングパスの16本に1本はセルフトレーニングを使用しないUnbiasedパスとする。
        // EN: Make one path out of every 16 training paths not use self-training and unbiased.
        const uint2 tileGroupSize = make_uint2(4, 4);
        const uint2 localTileIndex = tileIndex % tileGroupSize;
        const uint32_t localLinearTileIndex = localTileIndex.y * tileGroupSize.x + localTileIndex.x;
        isUnbiasedTrainingTile = (localLinearTileIndex + *plp.s->offsetToSelectUnbiasedTile) % 16 == 0;
    }
    else {
        (void)linearTileIndex;
        (void)isTrainingPath;
        (void)isUnbiasedTrainingTile;
    }

    const bool useEnvLight = plp.s->envLightTexture && plp.f->enableEnvLight;
    RGB contribution(0.001f, 0.001f, 0.001f);
    bool renderingPathEndsWithCache = false;
    uint32_t pathLength = 1;
    if (instSlot != 0xFFFFFFFF) {
        const uint32_t geomInstSlot = gb0Elems.geomInstSlot;
        const InstanceData &inst = plp.s->instanceDataBufferArray[bufIdx][instSlot];
        const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[geomInstSlot];
        Point3D positionInWorld;
        Normal3D geometricNormalInWorld;
        Normal3D shadingNormalInWorld;
        Vector3D texCoord0DirInWorld;
        Point2D texCoord;
        computeSurfacePoint(
            inst, geomInst,
            gb0Elems.primIndex, bcB, bcC,
            &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
            &geometricNormalInWorld, &texCoord);

        RGB alpha(1.0f);
        const float initImportance = sRGB_calcLuminance(alpha);
        PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

        // JP: 最初の交点におけるシェーディング。
        // EN: Shading on the first hit.
        Vector3D vIn;
        float dirPDensity;
        float primaryPathSpread;
        RGB localThroughput;
        uint32_t trainDataIndex;
        {
            const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

            Vector3D vOut = camera.position - positionInWorld;
            const float primaryDist2 = vOut.sqLength();
            vOut /= std::sqrt(primaryDist2);
            const float primaryDotVN = dot(vOut, geometricNormalInWorld);
            const float frontHit = primaryDotVN >= 0.0f ? 1.0f : -1.0f;
            // Offsetting assumes BRDF.
            positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);

            if constexpr (useNRC)
                primaryPathSpread = primaryDist2 / (4 * pi_v<float> * std::fabs(primaryDotVN));

            ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
            if (plp.f->enableBumpMapping) {
                const Normal3D modLocalNormal = mat.readModifiedNormal(mat.normal, mat.normalDimInfo, texCoord, 0.0f);
                applyBumpMapping(modLocalNormal, &shadingFrame);
            }
            const Vector3D vOutLocal = shadingFrame.toLocal(vOut);

            // JP: 光源を直接見ている場合の寄与を蓄積。
            // EN: Accumulate the contribution from a light source directly seeing.
            contribution = RGB(0.0f);
            if (vOutLocal.z > 0 && mat.emittance) {
                const float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
                const RGB emittance(getXYZ(texValue));
                contribution += alpha * emittance / pi_v<float>;
            }

            BSDF bsdf;
            bsdf.setup(mat, texCoord, 0.0f);

            // Next event estimation (explicit light sampling) on the first hit.
            const RGB directContNEE = performNextEventEstimation(
                positionInWorld, vOutLocal, shadingFrame, bsdf, rng);
            contribution += alpha * directContNEE;

            // generate a next ray.
            Vector3D vInLocal;
            localThroughput = bsdf.sampleThroughput(
                vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
                &vInLocal, &dirPDensity);
            alpha *= localThroughput;
            vIn = shadingFrame.fromLocal(vInLocal);

            if constexpr (useNRC) {
                // JP: 訓練データエントリーの確保。
                // EN: Allocate a training data entry.
                if (isTrainingPath) {
                    trainDataIndex = atomicAdd(plp.s->numTrainingData[bufIdx], 1u);

                    if (trainDataIndex < trainBufferSize) {
                        float roughness;
                        RGB diffuseReflectance, specularReflectance;
                        bsdf.getSurfaceParameters(
                            &diffuseReflectance, &specularReflectance, &roughness);

                        RadianceQuery radQuery;
                        createRadianceQuery(
                            positionInWorld, shadingFrame.normal, vOut,
                            roughness, diffuseReflectance, specularReflectance,
                            &radQuery);
                        plp.s->trainRadianceQueryBuffer[0][trainDataIndex] = radQuery;

                        TrainingVertexInfo vertInfo;
                        vertInfo.localThroughput = localThroughput;
                        vertInfo.prevVertexDataIndex = invalidVertexDataIndex;
                        vertInfo.pathLength = pathLength;
                        plp.s->trainVertexInfoBuffer[trainDataIndex] = vertInfo;

                        // JP: 現在の頂点に対する直接照明(NEE)によるScattered Radianceでターゲット値を初期化。
                        // EN: Initialize a target value by scattered radiance at the current vertex
                        //     by direct lighting (NEE).
                        plp.s->trainTargetBuffer[0][trainDataIndex] = directContNEE;
                        //if (!allFinite(directContNEE))
                        //    printf("NEE: (%g, %g, %g)\n",
                        //           directContNEE.x, directContNEE.y, directContNEE.z);
                    }
                    else {
                        trainDataIndex = invalidVertexDataIndex;
                    }
                }
            }
            else {
                (void)primaryPathSpread;
                (void)trainDataIndex;
            }
        }

        // Path extension loop
        PathTraceWriteOnlyPayload woPayload = {};
        PathTraceWriteOnlyPayload* woPayloadPtr = &woPayload;
        PathTraceReadWritePayload<useNRC> rwPayload = {};
        PathTraceReadWritePayload<useNRC>* rwPayloadPtr = &rwPayload;
        rwPayload.rng = rng;
        rwPayload.initImportance = initImportance;
        rwPayload.alpha = alpha;
        rwPayload.contribution = contribution;
        rwPayload.prevDirPDensity = dirPDensity;
        if constexpr (useNRC) {
            rwPayload.linearTileIndex = linearTileIndex;
            rwPayload.primaryPathSpread = primaryPathSpread;
            rwPayload.curSqrtPathSpread = 0.0f;
            rwPayload.prevLocalThroughput = localThroughput;
            rwPayload.prevTrainDataIndex = trainDataIndex;
            rwPayload.renderingPathEndsWithCache = false;
            rwPayload.isTrainingPath = isTrainingPath;
            rwPayload.isUnbiasedTrainingTile = isUnbiasedTrainingTile;
            rwPayload.trainingSuffixEndsWithCache = false;
        }
        rwPayload.pathLength = pathLength;
        Point3D rayOrg = positionInWorld;
        Vector3D rayDir = vIn;
        while (true) {
            const bool isValidSampling = rwPayload.prevDirPDensity > 0.0f && stc::isfinite(rwPayload.prevDirPDensity);
            if (!isValidSampling)
                break;

            ++rwPayload.pathLength;
            // JP: 通常のパストレーシングとNRCを正しく比較するには(特に通常のパストレーシングにおいて)
            //     反射回数制限を解除する必要がある。
            // EN: Disabling the limitation in the number of bounces (particularly for the base path tracing)
            //     is required to properly compare the base path tracing and NRC.
            if (rwPayload.pathLength >= plp.f->maxPathLength && plp.f->maxPathLength > 0)
                rwPayload.maxLengthTerminate = true;
            rwPayload.terminate = true;

            constexpr PathTracingRayType pathTraceRayType = useNRC ?
                PathTracingRayType::NRC : PathTracingRayType::Baseline;
            PathTraceRayPayloadSignature<useNRC>::trace(
                plp.f->travHandle, rayOrg.toNative(), rayDir.toNative(),
                0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
                pathTraceRayType, maxNumRayTypes, pathTraceRayType,
                woPayloadPtr, rwPayloadPtr);
            if (rwPayload.terminate)
                break;
            rayOrg = woPayload.nextOrigin;
            rayDir = woPayload.nextDirection;
        }
        contribution = rwPayload.contribution;

        plp.s->rngBuffer.write(launchIndex, rwPayload.rng);

        if constexpr (useNRC) {
            renderingPathEndsWithCache = rwPayload.renderingPathEndsWithCache;
            pathLength = rwPayload.pathLength;
            if (rwPayload.isTrainingPath && !rwPayload.trainingSuffixEndsWithCache) {
                TrainingSuffixTerminalInfo terminalInfo;
                terminalInfo.prevVertexDataIndex = rwPayload.prevTrainDataIndex;
                terminalInfo.hasQuery = false;
                terminalInfo.pathLength = rwPayload.pathLength;
                plp.s->trainSuffixTerminalInfoBuffer[rwPayload.linearTileIndex] = terminalInfo;
            }
        }
    }
    else {
        // JP: 環境光源を直接見ている場合の寄与を蓄積。
        // EN: Accumulate the contribution from the environmental light source directly seeing.
        if (useEnvLight) {
            const float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, bcB, bcC, 0.0f);
            const RGB luminance = plp.f->envLightPowerCoeff * RGB(getXYZ(texValue));
            contribution = luminance;
        }
    }

    if constexpr (useNRC) {
        const uint32_t linearIndex = launchIndex.y * plp.s->imageSize.x + launchIndex.x;

        // JP: 無限遠にレイが飛んだか、ロシアンルーレットによってパストレースが完了したケース。
        // EN: When a ray goes infinity or the path ends with Russain roulette.
        if (!renderingPathEndsWithCache) {
            TerminalInfo terminalInfo;
            terminalInfo.alpha = RGB(0.0f, 0.0f, 0.0f);
            terminalInfo.pathLength = pathLength;
            terminalInfo.hasQuery = false;
            terminalInfo.isTrainingPixel = isTrainingPath;
            terminalInfo.isUnbiasedTile = isUnbiasedTrainingTile;
            plp.s->inferenceTerminalInfoBuffer[linearIndex] = terminalInfo;
        }

        plp.s->perFrameContributionBuffer[linearIndex] = contribution;
    }
    else {
        (void)renderingPathEndsWithCache;
        (void)pathLength;

        RGB prevColorResult = RGB(0.0f, 0.0f, 0.0f);
        if (plp.f->numAccumFrames > 0)
            prevColorResult = RGB(getXYZ(plp.s->beautyAccumBuffer.read(launchIndex)));
        const float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
        const RGB colorResult = (1 - curWeight) * prevColorResult + curWeight * contribution;
        plp.s->beautyAccumBuffer.write(launchIndex, make_float4(colorResult.toNative(), 1.0f));
    }
}

template <bool useNRC>
CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_closestHit_generic() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = plp.f->bufferIndex;

    const auto sbtr = HitGroupSBTRecordData::get();
    const InstanceData &inst = plp.s->instanceDataBufferArray[bufIdx][optixGetInstanceId()];
    const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];

    PathTraceWriteOnlyPayload* woPayload;
    PathTraceReadWritePayload<useNRC>* rwPayload;
    PathTraceRayPayloadSignature<useNRC>::get(&woPayload, &rwPayload);
    PCG32RNG &rng = rwPayload->rng;

    const Point3D rayOrigin(optixGetWorldRayOrigin());

    const auto hp = HitPointParameter::get();
    Point3D positionInWorld;
    Normal3D shadingNormalInWorld;
    Vector3D texCoord0DirInWorld;
    Normal3D geometricNormalInWorld;
    Point2D texCoord;
    float hypAreaPDensity;
    computeSurfacePoint<true, useSolidAngleSampling>(
        inst, geomInst, hp.primIndex, hp.bcB, hp.bcC,
        rayOrigin,
        &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
        &geometricNormalInWorld, &texCoord, &hypAreaPDensity);

    const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

    const Vector3D vOut = normalize(-Vector3D(optixGetWorldRayDirection()));
    const float frontHit = dot(vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

    ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
    if (plp.f->enableBumpMapping) {
        const Normal3D modLocalNormal = mat.readModifiedNormal(mat.normal, mat.normalDimInfo, texCoord, 0.0f);
        applyBumpMapping(modLocalNormal, &shadingFrame);
    }
    positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);
    const Vector3D vOutLocal = shadingFrame.toLocal(vOut);
    //if (!allFinite(vOutLocal)) {
    //    printf("(%g, %g, %g), (%g, %g, %g), (%g, %g, %g)\n",
    //           shadingFrame.tangent.x, shadingFrame.tangent.y, shadingFrame.tangent.z,
    //           shadingFrame.bitangent.x, shadingFrame.bitangent.y, shadingFrame.bitangent.z,
    //           shadingFrame.normal.x, shadingFrame.normal.y, shadingFrame.normal.z);
    //}

    const float dist2 = sqDistance(rayOrigin, positionInWorld);
    if constexpr (useNRC)
        rwPayload->curSqrtPathSpread += std::sqrt(dist2 / (rwPayload->prevDirPDensity * std::fabs(vOutLocal.z)));

    // Implicit Light Sampling
    if (vOutLocal.z > 0 && mat.emittance) {
        const float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.x, texCoord.y, 0.0f);
        const RGB emittance(getXYZ(texValue));
        const float lightPDensity = hypAreaPDensity * dist2 / vOutLocal.z;
        const float bsdfPDensity = rwPayload->prevDirPDensity;
        const float misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
        const RGB directContImplicit = emittance * (misWeight / pi_v<float>);
        rwPayload->contribution += rwPayload->alpha * directContImplicit;

        if constexpr (useNRC) {
            // JP: 1つ前の頂点に対する直接照明(Implicit)によるScattered Radianceをターゲット値に加算。
            // EN: Accumulate scattered radiance at the previous vertex by direct lighting (implicit)
            //     to the target value.
            if (rwPayload->isTrainingPath && rwPayload->prevTrainDataIndex != invalidVertexDataIndex) {
                plp.s->trainTargetBuffer[0][rwPayload->prevTrainDataIndex] +=
                    rwPayload->prevLocalThroughput * directContImplicit;
                //if (!allFinite(rwPayload->prevLocalThroughput) ||
                //    !allFinite(directContImplicit))
                //    printf("Implicit: (%g, %g, %g), (%g, %g, %g)\n",
                //           rwPayload->prevLocalThroughput.x,
                //           rwPayload->prevLocalThroughput.y,
                //           rwPayload->prevLocalThroughput.z,
                //           directContImplicit.x,
                //           directContImplicit.y,
                //           directContImplicit.z);
            }
        }
    }

    // Russian roulette
    bool performRR = true;
    bool terminatedByRR = false;
    float recContinueProb = 1.0f;
    if constexpr (useNRC) {
        if (rwPayload->isTrainingPath)
            performRR = rwPayload->pathLength > 2;
    }
    if (performRR) {
        float continueProb = std::fmin(sRGB_calcLuminance(rwPayload->alpha) / rwPayload->initImportance, 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb || rwPayload->maxLengthTerminate) {
            if constexpr (useNRC) {
                if (rwPayload->renderingPathEndsWithCache &&
                    rwPayload->isTrainingPath && rwPayload->isUnbiasedTrainingTile)
                    return;
                terminatedByRR = true;
            }
            else {
                return;
            }
        }
        recContinueProb = 1.0f / continueProb;
    }

    BSDF bsdf;
    bsdf.setup(mat, texCoord, 0.0f);

    if constexpr (useNRC) {
        bool endsWithCache = false;
        const bool pathIsSpreadEnough =
            pow2(rwPayload->curSqrtPathSpread) > pathTerminationFactor * rwPayload->primaryPathSpread;
        endsWithCache |= pathIsSpreadEnough;
        if (rwPayload->renderingPathEndsWithCache &&
            rwPayload->isTrainingPath && rwPayload->isUnbiasedTrainingTile)
            endsWithCache = false;

        if (endsWithCache) {
            const uint32_t linearIndex = launchIndex.y * plp.s->imageSize.x + launchIndex.x;

            float roughness;
            RGB diffuseReflectance, specularReflectance;
            bsdf.getSurfaceParameters(
                &diffuseReflectance, &specularReflectance, &roughness);

            // JP: Radianceクエリーのための情報を記録する。
            // EN: Store information for radiance query.
            RadianceQuery radQuery;
            createRadianceQuery(
                positionInWorld, shadingFrame.normal, vOut,
                roughness, diffuseReflectance, specularReflectance,
                &radQuery);

            if (!rwPayload->renderingPathEndsWithCache) {
                plp.s->inferenceRadianceQueryBuffer[linearIndex] = radQuery;

                TerminalInfo terminalInfo;
                terminalInfo.alpha = rwPayload->alpha;
                terminalInfo.pathLength = rwPayload->pathLength;
                terminalInfo.hasQuery = true;
                terminalInfo.isTrainingPixel = rwPayload->isTrainingPath;
                terminalInfo.isUnbiasedTile = rwPayload->isUnbiasedTrainingTile;
                plp.s->inferenceTerminalInfoBuffer[linearIndex] = terminalInfo;

                rwPayload->renderingPathEndsWithCache = true;
                if (rwPayload->isTrainingPath)
                    rwPayload->curSqrtPathSpread = 0;
                else
                    return;
            }
            else {
                // JP: 訓練データバッファーがフルの場合は既にTraining Suffixは終了したことになっている。
                // EN: The training suffix should have been ended if the training data buffer is full.
                if (!rwPayload->trainingSuffixEndsWithCache) {
                    uint32_t offset = plp.s->imageSize.x * plp.s->imageSize.y;
                    plp.s->inferenceRadianceQueryBuffer[offset + rwPayload->linearTileIndex] = radQuery;

                    // JP: 直前のTraining VertexへのリンクとともにTraining Suffixを終了させる。
                    // EN: Finish the training suffix with the link to the previous training vertex.
                    TrainingSuffixTerminalInfo terminalInfo;
                    terminalInfo.prevVertexDataIndex = rwPayload->prevTrainDataIndex;
                    terminalInfo.hasQuery = true;
                    terminalInfo.pathLength = rwPayload->pathLength;
                    plp.s->trainSuffixTerminalInfoBuffer[rwPayload->linearTileIndex] = terminalInfo;

                    rwPayload->trainingSuffixEndsWithCache = true;
                }
                return;
            }
        }
    }

    if constexpr (useNRC) {
        if (terminatedByRR)
            return;
    }
    rwPayload->alpha *= recContinueProb;
    if constexpr (useNRC) {
        if (rwPayload->isTrainingPath && rwPayload->prevTrainDataIndex != invalidVertexDataIndex)
            plp.s->trainVertexInfoBuffer[rwPayload->prevTrainDataIndex].localThroughput *= recContinueProb;
    }

    // Next Event Estimation (Explicit Light Sampling)
    const RGB directContNEE = performNextEventEstimation(
        positionInWorld, vOutLocal, shadingFrame, bsdf, rng);
    rwPayload->contribution += rwPayload->alpha * directContNEE;

    // generate a next ray.
    Vector3D vInLocal;
    float dirPDensity;
    const RGB localThroughput = bsdf.sampleThroughput(
        vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &vInLocal, &dirPDensity);
    rwPayload->alpha *= localThroughput;
    const Vector3D vIn = shadingFrame.fromLocal(vInLocal);

    woPayload->nextOrigin = positionInWorld;
    woPayload->nextDirection = vIn;
    rwPayload->prevDirPDensity = dirPDensity;
    if constexpr (useNRC)
        rwPayload->prevLocalThroughput = localThroughput;
    rwPayload->terminate = false;

    if constexpr (useNRC) {
        // JP: 訓練データエントリーの確保。
        // EN: Allocate a training data entry.
        if (rwPayload->isTrainingPath && !rwPayload->trainingSuffixEndsWithCache) {
            const uint32_t trainDataIndex = atomicAdd(plp.s->numTrainingData[bufIdx], 1u);

            // TODO?: 訓練データ数の正確な推定のためにtrainingSuffixEndsWithCacheのチェックをここに持ってくる？

            float roughness;
            RGB diffuseReflectance, specularReflectance;
            bsdf.getSurfaceParameters(
                &diffuseReflectance, &specularReflectance, &roughness);

            RadianceQuery radQuery;
            createRadianceQuery(
                positionInWorld, shadingFrame.normal, vOut,
                roughness, diffuseReflectance, specularReflectance,
                &radQuery);

            if (trainDataIndex < trainBufferSize) {
                plp.s->trainRadianceQueryBuffer[0][trainDataIndex] = radQuery;

                // JP: ローカルスループットと前のTraining Vertexへのリンクを記録。
                // EN: Record the local throughput and the link to the previous training vertex.
                TrainingVertexInfo vertInfo;
                vertInfo.localThroughput = localThroughput;
                vertInfo.prevVertexDataIndex = rwPayload->prevTrainDataIndex;
                vertInfo.pathLength = rwPayload->pathLength;
                plp.s->trainVertexInfoBuffer[trainDataIndex] = vertInfo;

                // JP: 現在の頂点に対する直接照明(NEE)によるScattered Radianceでターゲット値を初期化。
                // EN: Initialize a target value by scattered radiance at the current vertex by
                //     direct lighting (NEE).
                plp.s->trainTargetBuffer[0][trainDataIndex] = directContNEE;
                //if (!allFinite(directContNEE))
                //    printf("NEE: (%g, %g, %g)\n",
                //           directContNEE.x, directContNEE.y, directContNEE.z);

                rwPayload->prevTrainDataIndex = trainDataIndex;
            }
            // JP: 訓練データがバッファーを溢れた場合は強制的にTraining Suffixを終了させる。
            // EN: Forcefully end the training suffix if the training data buffer become full.
            else {
                const uint32_t offset = plp.s->imageSize.x * plp.s->imageSize.y;
                plp.s->inferenceRadianceQueryBuffer[offset + rwPayload->linearTileIndex] = radQuery;

                TrainingSuffixTerminalInfo terminalInfo;
                terminalInfo.prevVertexDataIndex = rwPayload->prevTrainDataIndex;
                terminalInfo.hasQuery = true;
                terminalInfo.pathLength = rwPayload->pathLength;
                plp.s->trainSuffixTerminalInfoBuffer[rwPayload->linearTileIndex] = terminalInfo;

                rwPayload->trainingSuffixEndsWithCache = true;
            }
        }
    }
}

template <bool useNRC>
CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_miss_generic() {
    if (!plp.s->envLightTexture || !plp.f->enableEnvLight)
        return;

    PathTraceReadWritePayload<useNRC>* rwPayload;
    PathTraceRayPayloadSignature<useNRC>::get(nullptr, &rwPayload);

    const Vector3D rayDir = normalize(Vector3D(optixGetWorldRayDirection()));
    float posPhi, theta;
    toPolarYUp(rayDir, &posPhi, &theta);

    float phi = posPhi + plp.f->envLightRotation;
    phi = phi - floorf(phi / (2 * pi_v<float>)) * 2 * pi_v<float>;
    const Point2D texCoord(phi / (2 * pi_v<float>), theta / pi_v<float>);

    // Implicit Light Sampling
    const float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, texCoord.x, texCoord.y, 0.0f);
    const RGB luminance = plp.f->envLightPowerCoeff * RGB(getXYZ(texValue));
    const float uvPDF = plp.s->envLightImportanceMap.evaluatePDF(texCoord.x, texCoord.y);
    const float hypAreaPDensity = uvPDF / (2 * pi_v<float> * pi_v<float> * std::sin(theta));
    const float lightPDensity = probToSampleEnvLight * hypAreaPDensity;
    const float bsdfPDensity = rwPayload->prevDirPDensity;
    const float misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
    const RGB directContImplicit = misWeight * luminance;
    rwPayload->contribution += rwPayload->alpha * directContImplicit;

    if constexpr (useNRC) {
        // JP: 1つ前の頂点に対する直接照明(Implicit)によるScattered Radianceをターゲット値に加算。
        // EN: Accumulate scattered radiance at the previous vertex by direct lighting (implicit)
        //     to the target value.
        if (rwPayload->isTrainingPath && rwPayload->prevTrainDataIndex != invalidVertexDataIndex) {
            plp.s->trainTargetBuffer[0][rwPayload->prevTrainDataIndex] +=
                rwPayload->prevLocalThroughput * directContImplicit;
            //if (!allFinite(rwPayload->prevLocalThroughput) ||
            //    !allFinite(directContImplicit))
            //    printf("Implicit: (%g, %g, %g), (%g, %g, %g)\n",
            //           rwPayload->prevLocalThroughput.x,
            //           rwPayload->prevLocalThroughput.y,
            //           rwPayload->prevLocalThroughput.z,
            //           directContImplicit.x,
            //           directContImplicit.y,
            //           directContImplicit.z);
        }
    }
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTraceBaseline)() {
    pathTrace_raygen_generic<false>();
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTraceBaseline)() {
    pathTrace_closestHit_generic<false>();
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(pathTraceBaseline)() {
    pathTrace_miss_generic<false>();
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTraceNRC)() {
    pathTrace_raygen_generic<true>();
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTraceNRC)() {
    pathTrace_closestHit_generic<true>();
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(pathTraceNRC)() {
    pathTrace_miss_generic<true>();
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(visualizePrediction)() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t linearIndex = launchIndex.y * plp.s->imageSize.x + launchIndex.x;
    const uint32_t bufIdx = plp.f->bufferIndex;

    const GBuffer0Elements gb0Elems = plp.s->GBuffer0[bufIdx].read(launchIndex);
    const uint32_t instSlot = gb0Elems.instSlot;
    const float bcB = decodeBarycentric(gb0Elems.qbcB);
    const float bcC = decodeBarycentric(gb0Elems.qbcC);

    const PerspectiveCamera &camera = plp.f->camera;

    if (instSlot != 0xFFFFFFFF) {
        const uint32_t geomInstSlot = gb0Elems.geomInstSlot;
        const InstanceData &inst = plp.s->instanceDataBufferArray[bufIdx][instSlot];
        const GeometryInstanceData &geomInst = plp.s->geometryInstanceDataBuffer[geomInstSlot];
        Point3D positionInWorld;
        Normal3D geometricNormalInWorld;
        Normal3D shadingNormalInWorld;
        Vector3D texCoord0DirInWorld;
        Point2D texCoord;
        computeSurfacePoint(
            inst, geomInst,
            gb0Elems.primIndex, bcB, bcC,
            &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
            &geometricNormalInWorld, &texCoord);

        const MaterialData &mat = plp.s->materialDataBuffer[geomInst.materialSlot];

        Vector3D vOut = camera.position - positionInWorld;
        float primaryDist2 = vOut.sqLength();
        vOut /= std::sqrt(primaryDist2);
        const float primaryDotVN = dot(vOut, geometricNormalInWorld);
        const float frontHit = primaryDotVN >= 0.0f ? 1.0f : -1.0f;
        // Offsetting assumes BRDF.
        positionInWorld = offsetRayOrigin(positionInWorld, frontHit * geometricNormalInWorld);

        ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
        if (plp.f->enableBumpMapping) {
            const Normal3D modLocalNormal = mat.readModifiedNormal(mat.normal, mat.normalDimInfo, texCoord, 0.0f);
            applyBumpMapping(modLocalNormal, &shadingFrame);
        }

        BSDF bsdf;
        bsdf.setup(mat, texCoord, 0.0f);

        float roughness;
        RGB diffuseReflectance, specularReflectance;
        bsdf.getSurfaceParameters(
            &diffuseReflectance, &specularReflectance, &roughness);

        RadianceQuery radQuery;
        createRadianceQuery(
            positionInWorld, shadingFrame.normal, vOut,
            roughness, diffuseReflectance, specularReflectance,
            &radQuery);

        plp.s->inferenceRadianceQueryBuffer[linearIndex] = radQuery;
    }
    else {
        //// JP: 環境光源を直接見ている場合の寄与を蓄積。
        //// EN: Accumulate the contribution from the environmental light source directly seeing.
        //if (useEnvLight) {
        //    const float4 texValue = tex2DLod<float4>(plp.s->envLightTexture, bcB, bcC, 0.0f);
        //    const RGB luminance = plp.f->envLightPowerCoeff * RGB(getXYZ(texValue));
        //    contribution = luminance;
        //}
    }

    TerminalInfo terminalInfo;
    terminalInfo.alpha = RGB(1.0f);
    terminalInfo.pathLength = 1;
    terminalInfo.hasQuery = instSlot != 0xFFFFFFFF;
    terminalInfo.isTrainingPixel = false;
    terminalInfo.isUnbiasedTile = false;
    plp.s->inferenceTerminalInfoBuffer[linearIndex] = terminalInfo;
}
