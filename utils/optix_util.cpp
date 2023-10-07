/*

   Copyright 2023 Shin Watanabe

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

#include "optix_util_private.h"

namespace optixu {
    void devPrintf(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
#if defined(OPTIXU_Platform_Windows_MSVC)
        char str[4096];
        vsnprintf_s(str, sizeof(str), _TRUNCATE, fmt, args);
        OutputDebugString(str);
#else
        vprintf_s(fmt, args);
#endif
        va_end(args);
    }



    // static
    Context Context::create(CUcontext cuContext, uint32_t logLevel, EnableValidation enableValidation) {
        return (new _Context(cuContext, logLevel, enableValidation))->getPublicType();
    }

    void Context::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    void Context::setLogCallback(
        OptixLogCallback callback, void* callbackData, uint32_t logLevel) const {
        m->throwRuntimeError(logLevel <= 4, "Valid range for logLevel is [0, 4].");
        if (callback)
            OPTIX_CHECK(optixDeviceContextSetLogCallback(m->rawContext, callback, callbackData, logLevel));
        else
            OPTIX_CHECK(optixDeviceContextSetLogCallback(m->rawContext, &logCallBack, nullptr, logLevel));
    }

    void Context::setName(const std::string &name) const {
        m->setName(name);
    }

    const char* Context::getName() const {
        return m->getRegisteredName();
    }



    template <typename T>
    Context Object<T>::getContext() const {
        return m->getContext()->getPublicType();
    }

    template <typename T>
    void Object<T>::setName(const std::string &name) const {
        m->setName(name);
    }

    template <typename T>
    const char* Object<T>::getName() const {
        return m->getRegisteredName();
    }



    Material Context::createMaterial() const {
        return (new _Material(m))->getPublicType();
    }

    Scene Context::createScene() const {
        return (new _Scene(m))->getPublicType();
    }

    Pipeline Context::createPipeline() const {
        return (new _Pipeline(m))->getPublicType();
    }

    Denoiser Context::createDenoiser(
        OptixDenoiserModelKind modelKind,
        GuideAlbedo guideAlbedo,
        GuideNormal guideNormal,
        OptixDenoiserAlphaMode alphaMode) const {
        return (new _Denoiser(m, modelKind, guideAlbedo, guideNormal, alphaMode))->getPublicType();
    }

    uint32_t Context::getRTCoreVersion() const {
        return m->rtCoreVersion;
    }

    uint32_t Context::getShaderExecutionReorderingFlags() const {
        return m->shaderExecutionReorderingFlags;
    }

    CUcontext Context::getCUcontext() const {
        return m->cuContext;
    }



    void Material::Priv::setRecordHeader(
        const _Pipeline* pipeline, uint32_t rayType, uint8_t* record, SizeAlign* curSizeAlign) const {
        Key key{ pipeline, rayType };
        throwRuntimeError(
            programs.count(key),
            "No hit group is set to the pipeline %s, ray type %u",
            pipeline->getName().c_str(), rayType);
        const _HitProgramGroup* hitGroup = programs.at(key);
        *curSizeAlign = SizeAlign(OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);
        hitGroup->packHeader(record);
    }

    void Material::Priv::setRecordData(uint8_t* record, SizeAlign* curSizeAlign) const {
        uint32_t offset;
        curSizeAlign->add(userDataSizeAlign, &offset);
        std::memcpy(record + offset, userData.data(), userDataSizeAlign.size);
    }

    void Material::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    void Material::setHitGroup(uint32_t rayType, HitProgramGroup hitGroup) const {
        const _Pipeline* _pipeline = extract(hitGroup)->getPipeline();
        m->throwRuntimeError(_pipeline, "Invalid pipeline %p.", _pipeline);

        _Material::Key key{ _pipeline, rayType };
        m->programs[key] = extract(hitGroup);
    }

    void Material::setUserData(const void* data, uint32_t size, uint32_t alignment) const {
        m->throwRuntimeError(
            size <= s_maxMaterialUserDataSize,
            "Maximum user data size for Material is %u bytes.",
            s_maxMaterialUserDataSize);
        m->throwRuntimeError(
            alignment > 0 && alignment <= OPTIX_SBT_RECORD_ALIGNMENT,
            "Valid alignment range is [1, %u].",
            OPTIX_SBT_RECORD_ALIGNMENT);
        m->userDataSizeAlign = SizeAlign(size, alignment);
        m->userData.resize(size);
        std::memcpy(m->userData.data(), data, size);
    }

    HitProgramGroup Material::getHitGroup(Pipeline pipeline, uint32_t rayType) const {
        auto _pipeline = extract(pipeline);
        m->throwRuntimeError(_pipeline, "Invalid pipeline %p.", _pipeline);

        _Material::Key key{ _pipeline, rayType };
        m->throwRuntimeError(
            m->programs.count(key),
            "Hit group is not set for the pipeline %s, rayType %u.",
            _pipeline->getName().c_str(), rayType);
        return m->programs.at(key)->getPublicType();
    }

    void Material::getUserData(void* data, uint32_t* size, uint32_t* alignment) const {
        if (data)
            std::memcpy(data, m->userData.data(), m->userDataSizeAlign.size);
        if (size)
            *size = m->userDataSizeAlign.size;
        if (alignment)
            *alignment = m->userDataSizeAlign.alignment;
    }



    void Scene::Priv::addGAS(_GeometryAccelerationStructure* gas) {
        geomASs[gas->getSerialID()] = gas;
    }

    void Scene::Priv::removeGAS(_GeometryAccelerationStructure* gas) {
        geomASs.erase(gas->getSerialID());
    }

    void Scene::Priv::markSBTLayoutDirty() {
        sbtLayoutIsUpToDate = false;

        for (_InstanceAccelerationStructure* _ias : instASs)
            _ias->markDirty(true);
    }

    uint32_t Scene::Priv::getSBTOffset(_GeometryAccelerationStructure* gas, uint32_t matSetIdx) {
        SBTOffsetKey key = SBTOffsetKey{ gas->getSerialID(), matSetIdx };
        throwRuntimeError(
            sbtOffsets.count(key),
            "GAS %s: material set index %u is out of bounds.",
            gas->getName().c_str(), matSetIdx);
        return sbtOffsets.at(key);
    }

    void Scene::Priv::setupHitGroupSBT(
        CUstream stream, const _Pipeline* pipeline, const BufferView &sbt, void* hostMem) {
        throwRuntimeError(
            sbt.sizeInBytes() >= singleRecordSize * numSBTRecords,
            "Hit group shader binding table size is not enough.");

        auto records = reinterpret_cast<uint8_t*>(hostMem);

        for (const std::pair<uint32_t, _GeometryAccelerationStructure*> &gas : geomASs) {
            uint32_t numMatSets = gas.second->getNumMaterialSets();
            for (uint32_t matSetIdx = 0; matSetIdx < numMatSets; ++matSetIdx) {
                uint32_t numRecords = gas.second->fillSBTRecords(pipeline, matSetIdx, records);
                records += numRecords * singleRecordSize;
            }
        }

        CUDADRV_CHECK(cuMemcpyHtoDAsync(sbt.getCUdeviceptr(), hostMem, sbt.sizeInBytes(), stream));
    }

    bool Scene::Priv::isReady(bool* hasMotionAS) {
        *hasMotionAS = false;
        for (const std::pair<uint32_t, _GeometryAccelerationStructure*> &gas : geomASs) {
            *hasMotionAS |= gas.second->hasMotion();
            if (!gas.second->isReady())
                return false;
        }

        for (_Transform* tr : transforms) {
            if (!tr->isReady())
                return false;
        }

        for (_InstanceAccelerationStructure* ias : instASs) {
            *hasMotionAS |= ias->hasMotion();
            if (!ias->isReady())
                return false;
        }

        if (!sbtLayoutIsUpToDate)
            return false;

        return true;
    }

    void Scene::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    OpacityMicroMapArray Scene::createOpacityMicroMapArray() const {
        return (new _OpacityMicroMapArray(m))->getPublicType();
    }

    DisplacementMicroMapArray Scene::createDisplacementMicroMapArray() const {
        return (new _DisplacementMicroMapArray(m))->getPublicType();
    }

    GeometryInstance Scene::createGeometryInstance(GeometryType geomType) const {
        m->throwRuntimeError(
            geomType == GeometryType::Triangles ||
            geomType == GeometryType::LinearSegments ||
            geomType == GeometryType::QuadraticBSplines ||
            geomType == GeometryType::FlatQuadraticBSplines ||
            geomType == GeometryType::CubicBSplines ||
            geomType == GeometryType::CatmullRomSplines ||
            geomType == GeometryType::CubicBezier ||
            geomType == GeometryType::Spheres ||
            geomType == GeometryType::CustomPrimitives,
            "Invalid geometry type: %u.",
            static_cast<uint32_t>(geomType));
        return (new _GeometryInstance(m, geomType))->getPublicType();
    }

    GeometryAccelerationStructure Scene::createGeometryAccelerationStructure(GeometryType geomType) const {
        m->throwRuntimeError(
            geomType == GeometryType::Triangles ||
            geomType == GeometryType::LinearSegments ||
            geomType == GeometryType::QuadraticBSplines ||
            geomType == GeometryType::FlatQuadraticBSplines ||
            geomType == GeometryType::CubicBSplines ||
            geomType == GeometryType::CatmullRomSplines ||
            geomType == GeometryType::CubicBezier ||
            geomType == GeometryType::Spheres ||
            geomType == GeometryType::CustomPrimitives,
            "Invalid geometry type: %u.",
            static_cast<uint32_t>(geomType));
        // JP: GASを生成するだけならSBTレイアウトには影響を与えないので無効化は不要。
        // EN: Only generating a GAS doesn't affect a SBT layout, no need to invalidate it.
        optixuAssert(m->geomASs.count(m->nextGeomASSerialID) == 0,
                     "Too many GAS creation beyond expectation has been done.");
        return (new _GeometryAccelerationStructure(m, m->nextGeomASSerialID++, geomType))->getPublicType();
    }

    Transform Scene::createTransform() const {
        return (new _Transform(m))->getPublicType();
    }

    Instance Scene::createInstance() const {
        return (new _Instance(m))->getPublicType();
    }

    InstanceAccelerationStructure Scene::createInstanceAccelerationStructure() const {
        return (new _InstanceAccelerationStructure(m))->getPublicType();
    }

    void Scene::markShaderBindingTableLayoutDirty() const {
        m->markSBTLayoutDirty();
    }

    void Scene::generateShaderBindingTableLayout(size_t* memorySize) const {
        if (m->sbtLayoutIsUpToDate) {
            *memorySize = m->singleRecordSize * std::max(m->numSBTRecords, 1u);
            return;
        }

        uint32_t sbtOffset = 0;
        m->sbtOffsets.clear();
        SizeAlign maxRecordSizeAlign;
        maxRecordSizeAlign += SizeAlign(OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);
        // JP: GASの仮想アドレスが実行の度に変わる環境でSBTのレイアウトを固定するため、
        //     GASはアドレスではなくシリアルIDに紐付けられている。
        // EN: A GAS is associated to its serial ID instead of its address to make SBT layout fixed
        //     in an environment where GAS's virtual address changes run to run.
        for (const std::pair<uint32_t, _GeometryAccelerationStructure*> &gas : m->geomASs) {
            uint32_t numMatSets = gas.second->getNumMaterialSets();
            for (uint32_t matSetIdx = 0; matSetIdx < numMatSets; ++matSetIdx) {
                SizeAlign gasRecordSizeAlign;
                uint32_t gasNumSBTRecords;
                gas.second->calcSBTRequirements(matSetIdx, &gasRecordSizeAlign, &gasNumSBTRecords);
                maxRecordSizeAlign = max(maxRecordSizeAlign, gasRecordSizeAlign);
                _Scene::SBTOffsetKey key = { gas.first, matSetIdx };
                m->sbtOffsets[key] = sbtOffset;
                sbtOffset += gasNumSBTRecords;
            }
        }
        maxRecordSizeAlign.alignUp();
        m->singleRecordSize = maxRecordSizeAlign.size;
        m->numSBTRecords = sbtOffset;
        m->sbtLayoutIsUpToDate = true;

        *memorySize = m->singleRecordSize * std::max(m->numSBTRecords, 1u);
    }

    bool Scene::shaderBindingTableLayoutIsReady() const {
        return m->sbtLayoutIsUpToDate;
    }



    void OpacityMicroMapArray::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    void OpacityMicroMapArray::setConfiguration(OptixOpacityMicromapFlags config) const {
        bool changed = false;
        changed = m->flags != config;
        m->flags = config;

        if (changed) {
            m->memoryUsageComputed = false;
            m->available = false;
        }
    }

    void OpacityMicroMapArray::computeMemoryUsage(
        const OptixOpacityMicromapHistogramEntry* microMapHistogramEntries,
        uint32_t numMicroMapHistogramEntries,
        OptixMicromapBufferSizes* memoryRequirement) const {
        m->microMapHistogramEntries.resize(numMicroMapHistogramEntries);
        std::copy_n(microMapHistogramEntries, numMicroMapHistogramEntries, m->microMapHistogramEntries.data());

        m->buildInput = {};
        m->buildInput.flags = m->flags;
        m->buildInput.micromapHistogramEntries = m->microMapHistogramEntries.data();
        m->buildInput.numMicromapHistogramEntries = static_cast<uint32_t>(m->microMapHistogramEntries.size());
        OPTIX_CHECK(optixOpacityMicromapArrayComputeMemoryUsage(
            m->getRawContext(), &m->buildInput, &m->memoryRequirement));

        *memoryRequirement = m->memoryRequirement;

        m->memoryUsageComputed = true;
        m->available = false;
    }

    void OpacityMicroMapArray::setBuffers(
        const BufferView &rawOmmBuffer, const BufferView &perMicroMapDescBuffer,
        const BufferView &outputBuffer) const {
        m->throwRuntimeError(
            outputBuffer.sizeInBytes() >= m->memoryRequirement.outputSizeInBytes,
            "Size of the given buffer is not enough.");
        m->rawOmmBuffer = rawOmmBuffer;
        m->perMicroMapDescBuffer = perMicroMapDescBuffer;
        m->outputBuffer = outputBuffer;

        m->buildInput.inputBuffer = m->rawOmmBuffer.getCUdeviceptr();
        m->buildInput.perMicromapDescBuffer = m->perMicroMapDescBuffer.getCUdeviceptr();
        m->buildInput.perMicromapDescStrideInBytes = m->perMicroMapDescBuffer.stride();

        m->buffersSet = true;
        m->available = false;
    }

    void OpacityMicroMapArray::markDirty() const {
        m->memoryUsageComputed = false;
        m->available = false;
    }

    void OpacityMicroMapArray::rebuild(
        CUstream stream, const BufferView &scratchBuffer) const {
        m->throwRuntimeError(
            m->memoryUsageComputed, "You need to call computeMemoryUsage() before rebuild.");
        m->throwRuntimeError(
            m->buffersSet, "You need to call setBuffers() before rebuild.");
        m->throwRuntimeError(
            scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempSizeInBytes,
            "Size of the given scratch buffer is not enough.");

        OptixMicromapBuffers buffers = {};
        buffers.output = m->outputBuffer.getCUdeviceptr();
        buffers.outputSizeInBytes = m->memoryRequirement.outputSizeInBytes;
        buffers.temp = scratchBuffer.getCUdeviceptr();
        buffers.tempSizeInBytes = m->memoryRequirement.tempSizeInBytes;
        OPTIX_CHECK(optixOpacityMicromapArrayBuild(m->getRawContext(), stream, &m->buildInput, &buffers));

        m->available = true;
    }

    bool OpacityMicroMapArray::isReady() const {
        return m->isReady();
    }

    BufferView OpacityMicroMapArray::getOutputBuffer() const {
        return m->getBuffer();
    }

    OptixOpacityMicromapFlags OpacityMicroMapArray::getConfiguration() const {
        return m->flags;
    }



    void DisplacementMicroMapArray::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    void DisplacementMicroMapArray::setConfiguration(OptixDisplacementMicromapFlags config) const {
        bool changed = false;
        changed = m->flags != config;
        m->flags = config;

        if (changed) {
            m->memoryUsageComputed = false;
            m->available = false;
        }
    }

    void DisplacementMicroMapArray::computeMemoryUsage(
        const OptixDisplacementMicromapHistogramEntry* microMapHistogramEntries,
        uint32_t numMicroMapHistogramEntries,
        OptixMicromapBufferSizes* memoryRequirement) const {
        m->microMapHistogramEntries.resize(numMicroMapHistogramEntries);
        std::copy_n(microMapHistogramEntries, numMicroMapHistogramEntries, m->microMapHistogramEntries.data());

        m->buildInput = {};
        m->buildInput.flags = m->flags;
        m->buildInput.displacementMicromapHistogramEntries = m->microMapHistogramEntries.data();
        m->buildInput.numDisplacementMicromapHistogramEntries =
            static_cast<uint32_t>(m->microMapHistogramEntries.size());
        OPTIX_CHECK(optixDisplacementMicromapArrayComputeMemoryUsage(
            m->getRawContext(), &m->buildInput, &m->memoryRequirement));

        *memoryRequirement = m->memoryRequirement;

        m->memoryUsageComputed = true;
        m->available = false;
    }

    void DisplacementMicroMapArray::setBuffers(
        const BufferView &rawDmmBuffer, const BufferView &perMicroMapDescBuffer,
        const BufferView &outputBuffer) const {
        m->throwRuntimeError(
            outputBuffer.sizeInBytes() >= m->memoryRequirement.outputSizeInBytes,
            "Size of the given buffer is not enough.");
        m->rawDmmBuffer = rawDmmBuffer;
        m->perMicroMapDescBuffer = perMicroMapDescBuffer;
        m->outputBuffer = outputBuffer;

        m->buildInput.displacementValuesBuffer = m->rawDmmBuffer.getCUdeviceptr();
        m->buildInput.perDisplacementMicromapDescBuffer = m->perMicroMapDescBuffer.getCUdeviceptr();
        m->buildInput.perDisplacementMicromapDescStrideInBytes = m->perMicroMapDescBuffer.stride();

        m->buffersSet = true;
        m->available = false;
    }

    void DisplacementMicroMapArray::markDirty() const {
        m->memoryUsageComputed = false;
        m->available = false;
    }

    void DisplacementMicroMapArray::rebuild(
        CUstream stream, const BufferView &scratchBuffer) const {
        m->throwRuntimeError(
            m->memoryUsageComputed, "You need to call computeMemoryUsage() before rebuild.");
        m->throwRuntimeError(
            m->buffersSet, "You need to call setBuffers() before rebuild.");
        m->throwRuntimeError(
            scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempSizeInBytes,
            "Size of the given scratch buffer is not enough.");

        OptixMicromapBuffers buffers = {};
        buffers.output = m->outputBuffer.getCUdeviceptr();
        buffers.outputSizeInBytes = m->memoryRequirement.outputSizeInBytes;
        buffers.temp = scratchBuffer.getCUdeviceptr();
        buffers.tempSizeInBytes = m->memoryRequirement.tempSizeInBytes;
        OPTIX_CHECK(optixDisplacementMicromapArrayBuild(m->getRawContext(), stream, &m->buildInput, &buffers));

        m->available = true;
    }

    bool DisplacementMicroMapArray::isReady() const {
        return m->isReady();
    }

    BufferView DisplacementMicroMapArray::getOutputBuffer() const {
        return m->getBuffer();
    }

    OptixDisplacementMicromapFlags DisplacementMicroMapArray::getConfiguration() const {
        return m->flags;
    }



    void GeometryInstance::Priv::fillBuildInput(OptixBuildInput* input, CUdeviceptr preTransform) const {
        *input = OptixBuildInput{};

        if (std::holds_alternative<TriangleGeometry>(geometry)) {
            auto &geom = std::get<TriangleGeometry>(geometry);
            throwRuntimeError(
                (geom.indexFormat != OPTIX_INDICES_FORMAT_NONE) == geom.triangleBuffer.isValid(),
                "Triangle buffer must be provided if using a index format other than None, "
                "otherwise must not be provided.");

            uint32_t vertexStride = geom.vertexBuffers[0].stride();
            uint32_t numVertices = static_cast<uint32_t>(geom.vertexBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.vertexBufferArray[i] = geom.vertexBuffers[i].getCUdeviceptr();
                throwRuntimeError(
                    geom.vertexBuffers[i].isValid(),
                    "Vertex buffer for motion step %u is not set.",
                    i);
                throwRuntimeError(
                    geom.vertexBuffers[i].numElements() == numVertices,
                    "Num elements for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.vertexBuffers[i].stride() == vertexStride,
                    "Vertex stride for motion step %u doesn't match that of 0.",
                    i);
            }

            input->type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            OptixBuildInputTriangleArray &triArray = input->triangleArray;

            triArray.vertexBuffers = geom.vertexBufferArray;
            triArray.numVertices = numVertices;
            triArray.vertexFormat = geom.vertexFormat;
            triArray.vertexStrideInBytes = vertexStride;

            uint32_t numTriangles;
            if (geom.indexFormat != OPTIX_INDICES_FORMAT_NONE) {
                numTriangles = static_cast<uint32_t>(geom.triangleBuffer.numElements());
                triArray.indexBuffer = geom.triangleBuffer.getCUdeviceptr();
                triArray.indexStrideInBytes = geom.triangleBuffer.stride();
                triArray.numIndexTriplets = numTriangles;
            }
            else {
                numTriangles = numVertices / 3;
                triArray.indexBuffer = 0;
                triArray.indexStrideInBytes = 0;
                triArray.numIndexTriplets = 0;
            }
            triArray.indexFormat = geom.indexFormat;
            triArray.primitiveIndexOffset = primitiveIndexOffset;

            if (geom.opacityMicroMapArray) {
                OptixBuildInputOpacityMicromap &ommInput = triArray.opacityMicromap;
                ommInput.opacityMicromapArray = geom.opacityMicroMapArray->getBuffer().getCUdeviceptr();
                ommInput.micromapUsageCounts = geom.opacityMicroMapUsageCounts.data();
                ommInput.numMicromapUsageCounts = static_cast<uint32_t>(geom.opacityMicroMapUsageCounts.size());
                ommInput.indexingMode = geom.opacityMicroMapIndexingMode;
                if (ommInput.indexingMode == OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED) {
                    ommInput.indexBuffer = geom.opacityMicroMapIndexBuffer.getCUdeviceptr();
                    ommInput.indexStrideInBytes = geom.opacityMicroMapIndexBuffer.stride();
                    ommInput.indexSizeInBytes = geom.opacityMicroMapIndexSize;
                    ommInput.indexOffset = geom.opacityMicroMapIndexOffset;
                }
            }

            if (geom.displacementMicroMapArray) {
                throwRuntimeError(
                    geom.displacementVertexDirectionBuffer.isValid(),
                    "Vertex direction buffer must be provided.");
                throwRuntimeError(
                    geom.displacementVertexDirectionBuffer.numElements() == numVertices,
                    "Num elements of the vertex direction buffer doesn't match that of the vertices.");
                throwRuntimeError(
                    geom.displacementVertexBiasAndScaleBuffer.isValid(),
                    "Vertex bias and scale buffer must be provided.");
                throwRuntimeError(
                    geom.displacementVertexBiasAndScaleBuffer.numElements() == numVertices,
                    "Num elements of the vertex bias and scale buffer doesn't match that of the vertices.");
                throwRuntimeError(
                    geom.displacementTriangleFlagsBuffer.isValid(),
                    "Triangle flags buffer must be provided.");
                throwRuntimeError(
                    geom.displacementTriangleFlagsBuffer.numElements() == numTriangles,
                    "Num elements of the triangle flags buffer doesn't match that of the triangles.");

                OptixBuildInputDisplacementMicromap &dmmInput = triArray.displacementMicromap;
                dmmInput.vertexDirectionsBuffer = geom.displacementVertexDirectionBuffer.getCUdeviceptr();
                dmmInput.vertexDirectionStrideInBytes = geom.displacementVertexDirectionBuffer.stride();
                dmmInput.vertexDirectionFormat = geom.displacementVertexDirectionFormat;
                dmmInput.vertexBiasAndScaleBuffer = geom.displacementVertexBiasAndScaleBuffer.getCUdeviceptr();
                dmmInput.vertexBiasAndScaleStrideInBytes = geom.displacementVertexBiasAndScaleBuffer.stride();
                dmmInput.vertexBiasAndScaleFormat = geom.displacementVertexBiasAndScaleFormat;
                dmmInput.triangleFlagsBuffer = geom.displacementTriangleFlagsBuffer.getCUdeviceptr();
                dmmInput.triangleFlagsStrideInBytes = geom.displacementTriangleFlagsBuffer.stride();
                dmmInput.displacementMicromapArray = geom.displacementMicroMapArray->getBuffer().getCUdeviceptr();
                dmmInput.displacementMicromapUsageCounts = geom.displacementMicroMapUsageCounts.data();
                dmmInput.numDisplacementMicromapUsageCounts =
                    static_cast<uint32_t>(geom.displacementMicroMapUsageCounts.size());
                dmmInput.indexingMode = geom.displacementMicroMapIndexingMode;
                if (dmmInput.indexingMode == OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED) {
                    dmmInput.displacementMicromapIndexBuffer = geom.displacementMicroMapIndexBuffer.getCUdeviceptr();
                    dmmInput.displacementMicromapIndexStrideInBytes = geom.displacementMicroMapIndexBuffer.stride();
                    dmmInput.displacementMicromapIndexSizeInBytes = geom.displacementMicroMapIndexSize;
                    dmmInput.displacementMicromapIndexOffset = geom.displacementMicroMapIndexOffset;
                }
            }

            triArray.numSbtRecords = static_cast<uint32_t>(buildInputFlags.size());
            if (triArray.numSbtRecords > 1) {
                triArray.sbtIndexOffsetBuffer = geom.materialIndexBuffer.getCUdeviceptr();
                triArray.sbtIndexOffsetSizeInBytes = geom.materialIndexSize;
                triArray.sbtIndexOffsetStrideInBytes = geom.materialIndexBuffer.stride();
            }
            else {
                triArray.sbtIndexOffsetBuffer = 0; // No per-primitive record
                triArray.sbtIndexOffsetSizeInBytes = 0; // No effect
                triArray.sbtIndexOffsetStrideInBytes = 0; // No effect
            }

            triArray.preTransform = preTransform;
            triArray.transformFormat = preTransform ?
                OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 : OPTIX_TRANSFORM_FORMAT_NONE;

            triArray.flags = reinterpret_cast<const uint32_t*>(buildInputFlags.data());
        }
        else if (std::holds_alternative<CurveGeometry>(geometry)) {
            auto &geom = std::get<CurveGeometry>(geometry);
            throwRuntimeError(geom.segmentIndexBuffer.isValid(), "Segment index buffer must be provided.");

            uint32_t vertexStride = geom.vertexBuffers[0].stride();
            uint32_t widthStride = geom.widthBuffers[0].stride();
            bool normalBufferAvailable = geomType == GeometryType::FlatQuadraticBSplines;
            bool normalBufferSet = false;
            uint32_t normalStride = 0;
            if (normalBufferAvailable) {
                normalBufferSet = geom.normalBuffers[0].isValid();
                normalStride = geom.normalBuffers[0].stride();
            }
            uint32_t numVertices = static_cast<uint32_t>(geom.vertexBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.vertexBufferArray[i] = geom.vertexBuffers[i].getCUdeviceptr();
                geom.widthBufferArray[i] = geom.widthBuffers[i].getCUdeviceptr();
                throwRuntimeError(
                    geom.vertexBuffers[i].isValid(),
                    "Vertex buffer for motion step %u is not set.",
                    i);
                throwRuntimeError(
                    geom.vertexBuffers[i].numElements() == numVertices,
                    "Num elements of the vertex buffer for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.vertexBuffers[i].stride() == vertexStride,
                    "Vertex stride for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.widthBuffers[i].isValid(),
                    "Width buffer for motion step %u is not set.",
                    i);
                throwRuntimeError(
                    geom.widthBuffers[i].numElements() == numVertices,
                    "Num elements of the width buffer for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.widthBuffers[i].stride() == widthStride,
                    "Width stride for motion step %u doesn't match that of 0.",
                    i);
                if (normalBufferAvailable) {
                    throwRuntimeError(
                        normalBufferSet == geom.normalBuffers[i].isValid(),
                        "Normal buffer for motion step %u is not set (/ set) while the step 0 is set (/ not set).",
                        i);
                    if (normalBufferSet) {
                        geom.normalBufferArray[i] = geom.normalBuffers[i].getCUdeviceptr();
                        throwRuntimeError(
                            geom.normalBuffers[i].numElements() == numVertices,
                            "Num elements of the normal buffer for motion step %u doesn't match that of 0.",
                            i);
                        throwRuntimeError(
                            geom.normalBuffers[i].stride() == normalStride,
                            "Normal stride for motion step %u doesn't match that of 0.",
                            i);
                    }
                }
            }

            input->type = OPTIX_BUILD_INPUT_TYPE_CURVES;
            OptixBuildInputCurveArray &curveArray = input->curveArray;

            if (geomType == GeometryType::LinearSegments)
                curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
            else if (geomType == GeometryType::QuadraticBSplines)
                curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
            else if (geomType == GeometryType::FlatQuadraticBSplines)
                curveArray.curveType = OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE;
            else if (geomType == GeometryType::CubicBSplines)
                curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
            else if (geomType == GeometryType::CatmullRomSplines)
                curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM;
            else if (geomType == GeometryType::CubicBezier)
                curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER;
            else
                optixuAssert_ShouldNotBeCalled();
            curveArray.endcapFlags = geom.endcapFlags;

            curveArray.vertexBuffers  = geom.vertexBufferArray;
            curveArray.vertexStrideInBytes = vertexStride;
            curveArray.widthBuffers = geom.widthBufferArray;
            curveArray.widthStrideInBytes = widthStride;
            if (normalBufferSet) {
                curveArray.normalBuffers = geom.normalBufferArray;
                curveArray.normalStrideInBytes = normalStride;
            }
            curveArray.numVertices = numVertices;

            curveArray.indexBuffer = geom.segmentIndexBuffer.getCUdeviceptr();
            curveArray.indexStrideInBytes = geom.segmentIndexBuffer.stride();
            curveArray.numPrimitives = static_cast<uint32_t>(geom.segmentIndexBuffer.numElements());
            curveArray.primitiveIndexOffset = primitiveIndexOffset;

            curveArray.flag = static_cast<uint32_t>(buildInputFlags[0]);
        }
        else if (std::holds_alternative<SphereGeometry>(geometry)) {
            auto &geom = std::get<SphereGeometry>(geometry);

            uint32_t centerStride = geom.centerBuffers[0].stride();
            uint32_t radiusStride = geom.radiusBuffers[0].stride();
            uint32_t numSpheres = static_cast<uint32_t>(geom.centerBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.centerBufferArray[i] = geom.centerBuffers[i].getCUdeviceptr();
                geom.radiusBufferArray[i] = geom.radiusBuffers[i].getCUdeviceptr();
                throwRuntimeError(
                    geom.centerBuffers[i].isValid(),
                    "Center buffer for motion step %u is not set.",
                    i);
                throwRuntimeError(
                    geom.centerBuffers[i].numElements() == numSpheres,
                    "Num elements of the center buffer for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.centerBuffers[i].stride() == centerStride,
                    "Center stride for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.radiusBuffers[i].isValid(),
                    "Radius buffer for motion step %u is not set.",
                    i);
                throwRuntimeError(
                    geom.radiusBuffers[i].numElements() == numSpheres,
                    "Num elements of the radius buffer for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.radiusBuffers[i].stride() == radiusStride,
                    "Radius stride for motion step %u doesn't match that of 0.",
                    i);
            }

            input->type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
            OptixBuildInputSphereArray &sphereArray = input->sphereArray;

            sphereArray.vertexBuffers = geom.centerBufferArray;
            sphereArray.vertexStrideInBytes = centerStride;
            sphereArray.radiusBuffers = geom.radiusBufferArray;
            sphereArray.radiusStrideInBytes = radiusStride;
            sphereArray.numVertices = numSpheres;
            sphereArray.singleRadius = geom.useSingleRadius;

            sphereArray.primitiveIndexOffset = primitiveIndexOffset;

            sphereArray.numSbtRecords = static_cast<uint32_t>(buildInputFlags.size());
            if (sphereArray.numSbtRecords > 1) {
                sphereArray.sbtIndexOffsetBuffer = geom.materialIndexBuffer.getCUdeviceptr();
                sphereArray.sbtIndexOffsetSizeInBytes = geom.materialIndexSize;
                sphereArray.sbtIndexOffsetStrideInBytes = geom.materialIndexBuffer.stride();
            }
            else {
                sphereArray.sbtIndexOffsetBuffer = 0; // No per-primitive record
                sphereArray.sbtIndexOffsetSizeInBytes = 0; // No effect
                sphereArray.sbtIndexOffsetStrideInBytes = 0; // No effect
            }

            sphereArray.flags = reinterpret_cast<const uint32_t*>(buildInputFlags.data());
        }
        else if (std::holds_alternative<CustomPrimitiveGeometry>(geometry)) {
            auto &geom = std::get<CustomPrimitiveGeometry>(geometry);

            uint32_t stride = geom.primitiveAabbBuffers[0].stride();
            uint32_t numPrimitives = static_cast<uint32_t>(geom.primitiveAabbBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.primitiveAabbBufferArray[i] = geom.primitiveAabbBuffers[i].getCUdeviceptr();
                throwRuntimeError(
                    geom.primitiveAabbBuffers[i].isValid(),
                    "AABB buffer for motion step %u is not set.",
                    i);
                throwRuntimeError(
                    geom.primitiveAabbBuffers[i].numElements() == numPrimitives,
                    "Num elements for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.primitiveAabbBuffers[i].stride() == stride,
                    "Stride for motion step %u doesn't match that of 0.",
                    i);
            }

            input->type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            OptixBuildInputCustomPrimitiveArray &customPrimArray = input->customPrimitiveArray;

            customPrimArray.aabbBuffers = geom.primitiveAabbBufferArray;
            customPrimArray.numPrimitives = numPrimitives;
            customPrimArray.strideInBytes = stride;
            customPrimArray.primitiveIndexOffset = primitiveIndexOffset;

            customPrimArray.numSbtRecords = static_cast<uint32_t>(buildInputFlags.size());
            if (customPrimArray.numSbtRecords > 1) {
                customPrimArray.sbtIndexOffsetBuffer = geom.materialIndexBuffer.getCUdeviceptr();
                customPrimArray.sbtIndexOffsetSizeInBytes = geom.materialIndexSize;
                customPrimArray.sbtIndexOffsetStrideInBytes = geom.materialIndexBuffer.stride();
            }
            else {
                customPrimArray.sbtIndexOffsetBuffer = 0; // No per-primitive record
                customPrimArray.sbtIndexOffsetSizeInBytes = 0; // No effect
                customPrimArray.sbtIndexOffsetStrideInBytes = 0; // No effect
            }

            customPrimArray.flags = reinterpret_cast<const uint32_t*>(buildInputFlags.data());
        }
        else {
            optixuAssert_ShouldNotBeCalled();
        }
    }

    void GeometryInstance::Priv::updateBuildInput(OptixBuildInput* input, CUdeviceptr preTransform) const {
        if (std::holds_alternative<TriangleGeometry>(geometry)) {
            auto &geom = std::get<TriangleGeometry>(geometry);

            uint32_t vertexStride = geom.vertexBuffers[0].stride();
            uint32_t numVertices = static_cast<uint32_t>(geom.vertexBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.vertexBufferArray[i] = geom.vertexBuffers[i].getCUdeviceptr();
                throwRuntimeError(
                    geom.vertexBuffers[i].isValid(),
                    "Vertex buffer for motion step %u is not set.",
                    i);
                throwRuntimeError(
                    geom.vertexBuffers[i].numElements() == numVertices,
                    "Num elements for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.vertexBuffers[i].stride() == vertexStride,
                    "Vertex stride for motion step %u doesn't match that of 0.",
                    i);
            }

            OptixBuildInputTriangleArray &triArray = input->triangleArray;

            triArray.vertexBuffers = geom.vertexBufferArray;

            uint32_t numTriangles;
            if (geom.indexFormat != OPTIX_INDICES_FORMAT_NONE) {
                numTriangles = static_cast<uint32_t>(geom.triangleBuffer.numElements());
                triArray.indexBuffer = geom.triangleBuffer.getCUdeviceptr();
            }
            else {
                numTriangles = numVertices / 3;
            }

            if (geom.opacityMicroMapArray) {
                // TODO: どの情報が更新可能なのか調べる。
                throwRuntimeError(geom.opacityMicroMapArray->isReady(), "OMM array is not ready.");
                OptixBuildInputOpacityMicromap &ommInput = triArray.opacityMicromap;
                ommInput.opacityMicromapArray = geom.opacityMicroMapArray->getBuffer().getCUdeviceptr();
                ommInput.micromapUsageCounts = geom.opacityMicroMapUsageCounts.data();
                ommInput.numMicromapUsageCounts = static_cast<uint32_t>(geom.opacityMicroMapUsageCounts.size());
                ommInput.indexingMode = geom.opacityMicroMapIndexingMode;
                if (ommInput.indexingMode == OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED) {
                    ommInput.indexBuffer = geom.opacityMicroMapIndexBuffer.getCUdeviceptr();
                    ommInput.indexStrideInBytes = geom.opacityMicroMapIndexBuffer.stride();
                    ommInput.indexSizeInBytes = geom.opacityMicroMapIndexSize;
                    ommInput.indexOffset = geom.opacityMicroMapIndexOffset;
                }
            }

            if (geom.displacementMicroMapArray) {
                throwRuntimeError(
                    geom.displacementVertexDirectionBuffer.isValid(),
                    "Vertex direction buffer must be provided.");
                throwRuntimeError(
                    geom.displacementVertexDirectionBuffer.numElements() == numVertices,
                    "Num elements of the vertex direction buffer doesn't match that of the vertices.");
                throwRuntimeError(
                    geom.displacementVertexBiasAndScaleBuffer.isValid(),
                    "Vertex bias and scale buffer must be provided.");
                throwRuntimeError(
                    geom.displacementVertexBiasAndScaleBuffer.numElements() == numVertices,
                    "Num elements of the vertex bias and scale buffer doesn't match that of the vertices.");
                throwRuntimeError(
                    geom.displacementTriangleFlagsBuffer.isValid(),
                    "Triangle flags buffer must be provided.");
                throwRuntimeError(
                    geom.displacementTriangleFlagsBuffer.numElements() == numTriangles,
                    "Num elements of the triangle flags buffer doesn't match that of the triangles.");

                // TODO: どの情報が更新可能なのか調べる。
                throwRuntimeError(geom.displacementMicroMapArray->isReady(), "DMM array is not ready.");
                OptixBuildInputDisplacementMicromap &dmmInput = triArray.displacementMicromap;
                dmmInput.vertexDirectionsBuffer = geom.displacementVertexDirectionBuffer.getCUdeviceptr();
                dmmInput.vertexDirectionStrideInBytes = geom.displacementVertexDirectionBuffer.stride();
                dmmInput.vertexDirectionFormat = geom.displacementVertexDirectionFormat;
                dmmInput.vertexBiasAndScaleBuffer = geom.displacementVertexBiasAndScaleBuffer.getCUdeviceptr();
                dmmInput.vertexBiasAndScaleStrideInBytes = geom.displacementVertexBiasAndScaleBuffer.stride();
                dmmInput.vertexBiasAndScaleFormat = geom.displacementVertexBiasAndScaleFormat;
                dmmInput.triangleFlagsBuffer = geom.displacementTriangleFlagsBuffer.getCUdeviceptr();
                dmmInput.triangleFlagsStrideInBytes = geom.displacementTriangleFlagsBuffer.stride();
                dmmInput.displacementMicromapArray = geom.displacementMicroMapArray->getBuffer().getCUdeviceptr();
                dmmInput.displacementMicromapUsageCounts = geom.displacementMicroMapUsageCounts.data();
                dmmInput.numDisplacementMicromapUsageCounts =
                    static_cast<uint32_t>(geom.displacementMicroMapUsageCounts.size());
                dmmInput.indexingMode = geom.displacementMicroMapIndexingMode;
                if (dmmInput.indexingMode == OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED) {
                    dmmInput.displacementMicromapIndexBuffer = geom.displacementMicroMapIndexBuffer.getCUdeviceptr();
                    dmmInput.displacementMicromapIndexStrideInBytes = geom.displacementMicroMapIndexBuffer.stride();
                    dmmInput.displacementMicromapIndexSizeInBytes = geom.displacementMicroMapIndexSize;
                    dmmInput.displacementMicromapIndexOffset = geom.displacementMicroMapIndexOffset;
                }
            }

            if (triArray.numSbtRecords > 1)
                triArray.sbtIndexOffsetBuffer = geom.materialIndexBuffer.getCUdeviceptr();

            triArray.preTransform = preTransform;
            triArray.transformFormat = preTransform ?
                OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 : OPTIX_TRANSFORM_FORMAT_NONE;
        }
        else if (std::holds_alternative<CurveGeometry>(geometry)) {
            auto &geom = std::get<CurveGeometry>(geometry);

            uint32_t vertexStride = geom.vertexBuffers[0].stride();
            uint32_t widthStride = geom.widthBuffers[0].stride();
            bool normalBufferAvailable = geomType == GeometryType::FlatQuadraticBSplines;
            bool normalBufferSet = false;
            uint32_t normalStride = 0;
            if (normalBufferAvailable) {
                normalBufferSet = geom.normalBuffers[0].isValid();
                normalStride = geom.normalBuffers[0].stride();
            }
            uint32_t numVertices = static_cast<uint32_t>(geom.vertexBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.vertexBufferArray[i] = geom.vertexBuffers[i].getCUdeviceptr();
                geom.widthBufferArray[i] = geom.widthBuffers[i].getCUdeviceptr();
                throwRuntimeError(
                    geom.vertexBuffers[i].isValid(),
                    "Vertex buffer for motion step %u is not set.",
                    i);
                throwRuntimeError(
                    geom.vertexBuffers[i].numElements() == numVertices,
                    "Num elements of the vertex buffer for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.vertexBuffers[i].stride() == vertexStride,
                    "Vertex stride for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.widthBuffers[i].isValid(),
                    "Width buffer for motion step %u is not set.",
                    i);
                throwRuntimeError(
                    geom.widthBuffers[i].numElements() == numVertices,
                    "Num elements of the width buffer for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.widthBuffers[i].stride() == widthStride,
                    "Width stride for motion step %u doesn't match that of 0.",
                    i);
                if (normalBufferAvailable) {
                    throwRuntimeError(
                        normalBufferSet == geom.normalBuffers[i].isValid(),
                        "Normal buffer for motion step %u is not set (/ set) while the step 0 is set (/ not set).",
                        i);
                    if (normalBufferSet) {
                        geom.normalBufferArray[i] = geom.normalBuffers[i].getCUdeviceptr();
                        throwRuntimeError(
                            geom.normalBuffers[i].numElements() == numVertices,
                            "Num elements of the normal buffer for motion step %u doesn't match that of 0.",
                            i);
                        throwRuntimeError(
                            geom.normalBuffers[i].stride() == normalStride,
                            "Normal stride for motion step %u doesn't match that of 0.",
                            i);
                    }
                }
            }

            OptixBuildInputCurveArray &curveArray = input->curveArray;

            curveArray.vertexBuffers = geom.vertexBufferArray;
            curveArray.widthBuffers = geom.widthBufferArray;
            if (normalBufferSet) {
                curveArray.normalBuffers = geom.normalBufferArray;
                curveArray.normalStrideInBytes = normalStride;
            }

            curveArray.indexBuffer = geom.segmentIndexBuffer.getCUdeviceptr();
        }
        else if (std::holds_alternative<SphereGeometry>(geometry)) {
            auto &geom = std::get<SphereGeometry>(geometry);

            uint32_t centerStride = geom.centerBuffers[0].stride();
            uint32_t radiusStride = geom.radiusBuffers[0].stride();
            uint32_t numSpheres = static_cast<uint32_t>(geom.centerBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.centerBufferArray[i] = geom.centerBuffers[i].getCUdeviceptr();
                geom.radiusBufferArray[i] = geom.radiusBuffers[i].getCUdeviceptr();
                throwRuntimeError(
                    geom.centerBuffers[i].isValid(),
                    "Center buffer for motion step %u is not set.",
                    i);
                throwRuntimeError(
                    geom.centerBuffers[i].numElements() == numSpheres,
                    "Num elements of the center buffer for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.centerBuffers[i].stride() == centerStride,
                    "Center stride for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.radiusBuffers[i].isValid(),
                    "Radius buffer for motion step %u is not set.",
                    i);
                throwRuntimeError(
                    geom.radiusBuffers[i].numElements() == numSpheres,
                    "Num elements of the radius buffer for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.radiusBuffers[i].stride() == radiusStride,
                    "Radius stride for motion step %u doesn't match that of 0.",
                    i);
            }

            OptixBuildInputSphereArray &sphereArray = input->sphereArray;

            sphereArray.vertexBuffers = geom.centerBufferArray;
            sphereArray.radiusBuffers = geom.radiusBufferArray;
        }
        else if (std::holds_alternative<CustomPrimitiveGeometry>(geometry)) {
            auto &geom = std::get<CustomPrimitiveGeometry>(geometry);

            uint32_t stride = geom.primitiveAabbBuffers[0].stride();
            uint32_t numPrimitives = static_cast<uint32_t>(geom.primitiveAabbBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.primitiveAabbBufferArray[i] = geom.primitiveAabbBuffers[i].getCUdeviceptr();
                throwRuntimeError(
                    geom.primitiveAabbBuffers[i].isValid(),
                    "AABB buffer for motion step %u is not set.",
                    i);
                throwRuntimeError(
                    geom.primitiveAabbBuffers[i].numElements() == numPrimitives,
                    "Num elements for motion step %u doesn't match that of 0.",
                    i);
                throwRuntimeError(
                    geom.primitiveAabbBuffers[i].stride() == stride,
                    "Stride for motion step %u doesn't match that of 0.",
                    i);
            }

            OptixBuildInputCustomPrimitiveArray &customPrimArray = input->customPrimitiveArray;

            customPrimArray.aabbBuffers = geom.primitiveAabbBufferArray;

            if (customPrimArray.numSbtRecords > 1)
                customPrimArray.sbtIndexOffsetBuffer = geom.materialIndexBuffer.getCUdeviceptr();
        }
        else {
            optixuAssert_ShouldNotBeCalled();
        }
    }

    void GeometryInstance::Priv::calcSBTRequirements(
        uint32_t gasMatSetIdx,
        const SizeAlign &gasUserDataSizeAlign,
        const SizeAlign &gasChildUserDataSizeAlign,
        SizeAlign* maxRecordSizeAlign, uint32_t* numSBTRecords) const {
        *maxRecordSizeAlign = SizeAlign();
        for (int matIdx = 0; matIdx < materials.size(); ++matIdx) {
            throwRuntimeError(
                materials[matIdx][0],
                "Default material (== material set 0) is not set for the slot %u.",
                matIdx);
            uint32_t matSetIdx = gasMatSetIdx < materials[matIdx].size() ? gasMatSetIdx : 0;
            const _Material* mat = materials[matIdx][matSetIdx];
            if (!mat)
                mat = materials[matIdx][0];
            SizeAlign recordSizeAlign(OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);
            recordSizeAlign += gasUserDataSizeAlign;
            recordSizeAlign += gasChildUserDataSizeAlign;
            recordSizeAlign += mat->getUserDataSizeAlign();
            *maxRecordSizeAlign = max(*maxRecordSizeAlign, recordSizeAlign);
        }
        *maxRecordSizeAlign += userDataSizeAlign;
        *numSBTRecords = static_cast<uint32_t>(buildInputFlags.size());
    }

    uint32_t GeometryInstance::Priv::fillSBTRecords(
        const _Pipeline* pipeline, uint32_t gasMatSetIdx,
        const void* gasUserData, const SizeAlign &gasUserDataSizeAlign,
        const void* gasChildUserData, const SizeAlign &gasChildUserDataSizeAlign,
        uint32_t numRayTypes, uint8_t* records) const {
        uint32_t numMaterials = static_cast<uint32_t>(materials.size());
        for (uint32_t matIdx = 0; matIdx < numMaterials; ++matIdx) {
            throwRuntimeError(
                materials[matIdx][0],
                "Default material (== material set 0) is not set for material %u.",
                matIdx);
            uint32_t matSetIdx = gasMatSetIdx < materials[matIdx].size() ? gasMatSetIdx : 0;
            const _Material* mat = materials[matIdx][matSetIdx];
            if (!mat)
                mat = materials[matIdx][0];
            for (uint32_t rIdx = 0; rIdx < numRayTypes; ++rIdx) {
                SizeAlign curSizeAlign;
                mat->setRecordHeader(pipeline, rIdx, records, &curSizeAlign);
                uint32_t offset;
                curSizeAlign.add(gasUserDataSizeAlign, &offset);
                std::memcpy(records + offset, gasUserData, gasUserDataSizeAlign.size);
                curSizeAlign.add(gasChildUserDataSizeAlign, &offset);
                std::memcpy(records + offset, gasChildUserData, gasChildUserDataSizeAlign.size);
                curSizeAlign.add(userDataSizeAlign, &offset);
                std::memcpy(records + offset, userData.data(), userDataSizeAlign.size);
                mat->setRecordData(records, &curSizeAlign);
                records += scene->getSingleRecordSize();
            }
        }

        return numMaterials * numRayTypes;
    }

    void GeometryInstance::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    void GeometryInstance::setNumMotionSteps(uint32_t n) const {
        n = std::max(n, 1u);
        if (std::holds_alternative<Priv::TriangleGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
            delete[] geom.vertexBuffers;
            delete[] geom.vertexBufferArray;
            geom.vertexBufferArray = new CUdeviceptr[n];
            geom.vertexBuffers = new BufferView[n];
        }
        else if (std::holds_alternative<Priv::CurveGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
            if (m->geomType == GeometryType::FlatQuadraticBSplines) {
                delete[] geom.normalBuffers;
                delete[] geom.normalBufferArray;
            }
            delete[] geom.widthBuffers;
            delete[] geom.widthBufferArray;
            delete[] geom.vertexBuffers;
            delete[] geom.vertexBufferArray;
            geom.vertexBufferArray = new CUdeviceptr[n];
            geom.vertexBuffers = new BufferView[n];
            geom.widthBufferArray = new CUdeviceptr[n];
            geom.widthBuffers = new BufferView[n];
            if (m->geomType == GeometryType::FlatQuadraticBSplines) {
                geom.normalBufferArray = new CUdeviceptr[n];
                geom.normalBuffers = new BufferView[n];
            }
        }
        else if (std::holds_alternative<Priv::SphereGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::SphereGeometry>(m->geometry);
            delete[] geom.radiusBuffers;
            delete[] geom.radiusBufferArray;
            delete[] geom.centerBuffers;
            delete[] geom.centerBufferArray;
            geom.centerBufferArray = new CUdeviceptr[n];
            geom.centerBuffers = new BufferView[n];
            geom.radiusBufferArray = new CUdeviceptr[n];
            geom.radiusBuffers = new BufferView[n];
        }
        else if (std::holds_alternative<Priv::CustomPrimitiveGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::CustomPrimitiveGeometry>(m->geometry);
            delete[] geom.primitiveAabbBuffers;
            delete[] geom.primitiveAabbBufferArray;
            geom.primitiveAabbBufferArray = new CUdeviceptr[n];
            geom.primitiveAabbBuffers = new BufferView[n];
        }
        else {
            optixuAssert_ShouldNotBeCalled();
        }
        m->numMotionSteps = n;
    }

    void GeometryInstance::setVertexFormat(OptixVertexFormat format) const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::TriangleGeometry>(m->geometry),
            "This geometry instance was created not for triangles.");
        auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
        geom.vertexFormat = format;
    }

    void GeometryInstance::setVertexBuffer(const BufferView &vertexBuffer, uint32_t motionStep) const {
        m->throwRuntimeError(
            !std::holds_alternative<Priv::CustomPrimitiveGeometry>(m->geometry),
            "This geometry instance was created not for triangles or curves.");
        m->throwRuntimeError(
            motionStep < m->numMotionSteps,
            "motionStep %u is out of bounds [0, %u).",
            motionStep, m->numMotionSteps);
        if (std::holds_alternative<Priv::TriangleGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
            geom.vertexBuffers[motionStep] = vertexBuffer;
        }
        else if (std::holds_alternative<Priv::CurveGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
            geom.vertexBuffers[motionStep] = vertexBuffer;
        }
        else if (std::holds_alternative<Priv::SphereGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::SphereGeometry>(m->geometry);
            geom.centerBuffers[motionStep] = vertexBuffer;
        }
    }

    void GeometryInstance::setWidthBuffer(const BufferView &widthBuffer, uint32_t motionStep) const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::CurveGeometry>(m->geometry),
            "This geometry instance was created not for curves.");
        m->throwRuntimeError(motionStep < m->numMotionSteps, "motionStep %u is out of bounds [0, %u).",
                             motionStep, m->numMotionSteps);
        auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
        geom.widthBuffers[motionStep] = widthBuffer;
    }

    void GeometryInstance::setNormalBuffer(const BufferView &normalBuffer, uint32_t motionStep) const {
        m->throwRuntimeError(
            m->geomType == GeometryType::FlatQuadraticBSplines,
            "This geometry instance was created not for flat quadratic B-splines.");
        m->throwRuntimeError(motionStep < m->numMotionSteps, "motionStep %u is out of bounds [0, %u).",
                             motionStep, m->numMotionSteps);
        auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
        geom.normalBuffers[motionStep] = normalBuffer;
    }

    void GeometryInstance::setRadiusBuffer(const BufferView &radiusBuffer, uint32_t motionStep) const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::SphereGeometry>(m->geometry),
            "This geometry instance was created not for spheres.");
        m->throwRuntimeError(
            motionStep < m->numMotionSteps,
            "motionStep %u is out of bounds [0, %u).",
            motionStep, m->numMotionSteps);
        auto &geom = std::get<Priv::SphereGeometry>(m->geometry);
        geom.radiusBuffers[motionStep] = radiusBuffer;
    }

    void GeometryInstance::setTriangleBuffer(
        const BufferView &triangleBuffer, OptixIndicesFormat format) const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::TriangleGeometry>(m->geometry),
            "This geometry instance was created not for triangles.");
        auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
        geom.triangleBuffer = triangleBuffer;
        geom.indexFormat = format;
    }

    void GeometryInstance::setOpacityMicroMapArray(
        OpacityMicroMapArray opacityMicroMapArray,
        const OptixOpacityMicromapUsageCount* ommUsageCounts, uint32_t numOmmUsageCounts,
        const BufferView &ommIndexBuffer,
        IndexSize indexSize, uint32_t indexOffset) const {
        uint32_t indexSizeInBytes = 1 << static_cast<uint32_t>(indexSize);
        m->throwRuntimeError(
            std::holds_alternative<Priv::TriangleGeometry>(m->geometry),
            "This geometry instance was created not for triangles.");
        m->throwRuntimeError(
            indexSize == IndexSize::k2Bytes || indexSize == IndexSize::k4Bytes,
            "Invalid index size.");
        m->throwRuntimeError(
            !ommIndexBuffer.isValid() || ommIndexBuffer.stride() >= indexSizeInBytes,
            "Buffer's stride is smaller than the given index size.");
        auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
        geom.opacityMicroMapArray = extract(opacityMicroMapArray);
        if (opacityMicroMapArray) {
            geom.opacityMicroMapIndexBuffer = ommIndexBuffer;
            geom.opacityMicroMapIndexingMode = ommIndexBuffer.isValid() ?
                OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED :
                OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_LINEAR;
            geom.opacityMicroMapIndexSize = indexSizeInBytes;
            geom.opacityMicroMapIndexOffset = indexOffset;
            geom.opacityMicroMapUsageCounts.resize(numOmmUsageCounts);
            std::copy_n(ommUsageCounts, numOmmUsageCounts, geom.opacityMicroMapUsageCounts.data());
        }
        else {
            geom.opacityMicroMapIndexBuffer = BufferView();
            geom.opacityMicroMapIndexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_NONE;
            geom.opacityMicroMapIndexSize = 0;
            geom.opacityMicroMapUsageCounts.clear();
        }
    }

    void GeometryInstance::setDisplacementMicroMapArray(
        const BufferView &vertexDirectionBuffer,
        const BufferView &vertexBiasAndScaleBuffer,
        const BufferView &triangleFlagsBuffer,
        DisplacementMicroMapArray displacementMicroMapArray,
        const OptixDisplacementMicromapUsageCount* dmmUsageCounts, uint32_t numDmmUsageCounts,
        const BufferView &dmmIndexBuffer,
        IndexSize indexSize, uint32_t indexOffset,
        OptixDisplacementMicromapDirectionFormat vertexDirectionFormat,
        OptixDisplacementMicromapBiasAndScaleFormat vertexBiasAndScaleFormat) const {
        uint32_t indexSizeInBytes = 1 << static_cast<uint32_t>(indexSize);
        m->throwRuntimeError(
            std::holds_alternative<Priv::TriangleGeometry>(m->geometry),
            "This geometry instance was created not for triangles.");
        m->throwRuntimeError(
            indexSize == IndexSize::k2Bytes || indexSize == IndexSize::k4Bytes,
            "Invalid index size.");
        m->throwRuntimeError(
            !dmmIndexBuffer.isValid() || dmmIndexBuffer.stride() >= indexSizeInBytes,
            "Buffer's stride is smaller than the given index size.");
        auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
        geom.displacementVertexDirectionBuffer = vertexDirectionBuffer;
        geom.displacementVertexBiasAndScaleBuffer = vertexBiasAndScaleBuffer;
        geom.displacementTriangleFlagsBuffer = triangleFlagsBuffer;
        geom.displacementVertexDirectionFormat = vertexDirectionFormat;
        geom.displacementVertexBiasAndScaleFormat = vertexBiasAndScaleFormat;
        geom.displacementMicroMapArray = extract(displacementMicroMapArray);
        if (displacementMicroMapArray) {
            geom.displacementMicroMapIndexBuffer = dmmIndexBuffer;
            geom.displacementMicroMapIndexingMode = dmmIndexBuffer.isValid() ?
                OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_INDEXED :
                OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_LINEAR;
            geom.displacementMicroMapIndexSize = indexSizeInBytes;
            geom.displacementMicroMapIndexOffset = indexOffset;
            geom.displacementMicroMapUsageCounts.resize(numDmmUsageCounts);
            std::copy_n(dmmUsageCounts, numDmmUsageCounts, geom.displacementMicroMapUsageCounts.data());
        }
        else {
            geom.displacementMicroMapIndexBuffer = BufferView();
            geom.displacementMicroMapIndexingMode = OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_NONE;
            geom.displacementMicroMapIndexSize = 0;
            geom.displacementMicroMapUsageCounts.clear();
        }
    }

    void GeometryInstance::setSegmentIndexBuffer(const BufferView &segmentIndexBuffer) const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::CurveGeometry>(m->geometry),
            "This geometry instance was created not for curves.");
        auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
        geom.segmentIndexBuffer = segmentIndexBuffer;
    }

    void GeometryInstance::setCurveEndcapFlags(OptixCurveEndcapFlags endcapFlags) const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::CurveGeometry>(m->geometry),
            "This geometry instance was created not for curves.");
        auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
        geom.endcapFlags = endcapFlags;
    }

    void GeometryInstance::setSingleRadius(UseSingleRadius useSingleRadius) const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::SphereGeometry>(m->geometry),
            "This geometry instance was created not for spheres.");
        auto &geom = std::get<Priv::SphereGeometry>(m->geometry);
        geom.useSingleRadius = useSingleRadius;
    }

    void GeometryInstance::setCustomPrimitiveAABBBuffer(
        const BufferView &primitiveAABBBuffer, uint32_t motionStep) const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::CustomPrimitiveGeometry>(m->geometry),
            "This geometry instance was created not for custom primitives.");
        m->throwRuntimeError(
            motionStep < m->numMotionSteps,
            "motionStep %u is out of bounds [0, %u).",
            motionStep, m->numMotionSteps);
        auto &geom = std::get<Priv::CustomPrimitiveGeometry>(m->geometry);
        geom.primitiveAabbBuffers[motionStep] = primitiveAABBBuffer;
    }

    void GeometryInstance::setPrimitiveIndexOffset(uint32_t offset) const {
        m->primitiveIndexOffset = offset;
    }

    void GeometryInstance::setNumMaterials(
        uint32_t numMaterials, const BufferView &matIndexBuffer, IndexSize indexSize) const {
        uint32_t indexSizeInBytes = 1 << static_cast<uint32_t>(indexSize);
        m->throwRuntimeError(
            !std::holds_alternative<Priv::CurveGeometry>(m->geometry),
            "Geometry instance for curves is not allowed to have multiple materials.");
        m->throwRuntimeError(
            numMaterials > 0, "Invalid number of materials %u.", numMaterials);
        m->throwRuntimeError(
            (numMaterials == 1) != matIndexBuffer.isValid(),
            "Material index offset buffer must be provided when multiple materials are used.");
        m->throwRuntimeError(
            indexSizeInBytes >= 1 && indexSizeInBytes <= 4,
            "Invalid index size.");
        m->throwRuntimeError(
            !matIndexBuffer.isValid() || matIndexBuffer.stride() >= indexSizeInBytes,
            "Buffer's stride is smaller than the given index size.");
        m->buildInputFlags.resize(numMaterials, OPTIX_GEOMETRY_FLAG_NONE);
        if (std::holds_alternative<Priv::TriangleGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
            geom.materialIndexBuffer = matIndexBuffer;
            geom.materialIndexSize = indexSizeInBytes;
        }
        else if (std::holds_alternative<Priv::SphereGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::SphereGeometry>(m->geometry);
            geom.materialIndexBuffer = matIndexBuffer;
            geom.materialIndexSize = indexSizeInBytes;
        }
        else if (std::holds_alternative<Priv::CustomPrimitiveGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::CustomPrimitiveGeometry>(m->geometry);
            geom.materialIndexBuffer = matIndexBuffer;
            geom.materialIndexSize = indexSizeInBytes;
        }
        else {
            optixuAssert_ShouldNotBeCalled();
        }
        uint32_t prevNumMaterials = static_cast<uint32_t>(m->materials.size());
        m->materials.resize(numMaterials);
        for (int matIdx = prevNumMaterials; matIdx < m->materials.size(); ++matIdx)
            m->materials[matIdx].resize(1, nullptr);
    }

    void GeometryInstance::setGeometryFlags(uint32_t matIdx, OptixGeometryFlags flags) const {
        size_t numMaterials = m->materials.size();
        m->throwRuntimeError(
            matIdx < numMaterials, "Out of material bounds [0, %u).",
            static_cast<uint32_t>(numMaterials));

        m->buildInputFlags[matIdx] = flags;
    }

    void GeometryInstance::setMaterial(uint32_t matSetIdx, uint32_t matIdx, Material mat) const {
        size_t numMaterials = m->materials.size();
        m->throwRuntimeError(
            matIdx < numMaterials, "Out of material bounds [0, %u).",
            static_cast<uint32_t>(numMaterials));

        uint32_t prevNumMatSets = static_cast<uint32_t>(m->materials[matIdx].size());
        if (matSetIdx >= prevNumMatSets)
            m->materials[matIdx].resize(matSetIdx + 1, nullptr);
        m->materials[matIdx][matSetIdx] = extract(mat);
    }

    void GeometryInstance::setUserData(const void* data, uint32_t size, uint32_t alignment) const {
        m->throwRuntimeError(
            size <= s_maxGeometryInstanceUserDataSize,
            "Maximum user data size for GeometryInstance is %u bytes.",
            s_maxGeometryInstanceUserDataSize);
        m->throwRuntimeError(
            alignment > 0 && alignment <= OPTIX_SBT_RECORD_ALIGNMENT,
            "Valid alignment range is [1, %u].",
            OPTIX_SBT_RECORD_ALIGNMENT);
        if (m->userDataSizeAlign.size != size ||
            m->userDataSizeAlign.alignment != alignment)
            m->scene->markSBTLayoutDirty();
        m->userDataSizeAlign = SizeAlign(size, alignment);
        m->userData.resize(size);
        std::memcpy(m->userData.data(), data, size);
    }

    uint32_t GeometryInstance::getNumMotionSteps() const {
        return m->numMotionSteps;
    }

    OptixVertexFormat GeometryInstance::getVertexFormat() const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::TriangleGeometry>(m->geometry),
            "This geometry instance was created not for triangles.");
        const auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
        return geom.vertexFormat;
    }

    BufferView GeometryInstance::getVertexBuffer(uint32_t motionStep) {
        m->throwRuntimeError(
            !std::holds_alternative<Priv::CustomPrimitiveGeometry>(m->geometry),
            "This geometry instance was created not for triangles or curves.");
        m->throwRuntimeError(
            motionStep < m->numMotionSteps,
            "motionStep %u is out of bounds [0, %u).",
            motionStep, m->numMotionSteps);
        if (std::holds_alternative<Priv::TriangleGeometry>(m->geometry)) {
            const auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
            return geom.vertexBuffers[motionStep];
        }
        else if (std::holds_alternative<Priv::CurveGeometry>(m->geometry)) {
            const auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
            return geom.vertexBuffers[motionStep];
        }
        optixuAssert_ShouldNotBeCalled();
        return BufferView();
    }

    BufferView GeometryInstance::getWidthBuffer(uint32_t motionStep) {
        m->throwRuntimeError(
            std::holds_alternative<Priv::CurveGeometry>(m->geometry),
            "This geometry instance was created not for curves.");
        m->throwRuntimeError(
            motionStep < m->numMotionSteps,
            "motionStep %u is out of bounds [0, %u).",
            motionStep, m->numMotionSteps);
        const auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
        return geom.widthBuffers[motionStep];
    }

    BufferView GeometryInstance::getNormalBuffer(uint32_t motionStep) {
        m->throwRuntimeError(
            m->geomType == GeometryType::FlatQuadraticBSplines,
            "This geometry instance was created not for flat quadratic B-splines.");
        m->throwRuntimeError(
            motionStep < m->numMotionSteps,
            "motionStep %u is out of bounds [0, %u).",
            motionStep, m->numMotionSteps);
        const auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
        return geom.normalBuffers[motionStep];
    }

    BufferView GeometryInstance::getRadiusBuffer(uint32_t motionStep) {
        m->throwRuntimeError(
            std::holds_alternative<Priv::SphereGeometry>(m->geometry),
            "This geometry instance was created not for spheres.");
        m->throwRuntimeError(
            motionStep < m->numMotionSteps,
            "motionStep %u is out of bounds [0, %u).",
            motionStep, m->numMotionSteps);
        const auto &geom = std::get<Priv::SphereGeometry>(m->geometry);
        return geom.radiusBuffers[motionStep];
    }

    BufferView GeometryInstance::getTriangleBuffer(OptixIndicesFormat* format) const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::TriangleGeometry>(m->geometry),
            "This geometry instance was created not for triangles.");
        const auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
        if (format)
            *format = geom.indexFormat;
        return geom.triangleBuffer;
    }

    OpacityMicroMapArray GeometryInstance::getOpacityMicroMapArray(
        BufferView* ommIndexBuffer,
        IndexSize* indexSize, uint32_t* indexOffset) const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::TriangleGeometry>(m->geometry),
            "This geometry instance was created not for triangles.");
        const auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
        if (ommIndexBuffer)
            *ommIndexBuffer = geom.opacityMicroMapIndexBuffer;
        if (indexSize)
            *indexSize = convertToIndexSizeEnum(geom.opacityMicroMapIndexSize);
        if (indexOffset)
            *indexOffset = geom.opacityMicroMapIndexOffset;
        return geom.opacityMicroMapArray->getPublicType();
    }

    DisplacementMicroMapArray GeometryInstance::getDisplacementMicroMapArray(
        BufferView* vertexDirectionBuffer,
        BufferView* vertexBiasAndScaleBuffer,
        BufferView* triangleFlagsBuffer,
        BufferView* dmmIndexBuffer,
        IndexSize* indexSize, uint32_t* indexOffset,
        OptixDisplacementMicromapDirectionFormat* vertexDirectionFormat,
        OptixDisplacementMicromapBiasAndScaleFormat* vertexBiasAndScaleFormat) const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::TriangleGeometry>(m->geometry),
            "This geometry instance was created not for triangles.");
        const auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
        if (vertexDirectionBuffer)
            *vertexDirectionBuffer = geom.displacementVertexDirectionBuffer;
        if (vertexBiasAndScaleBuffer)
            *vertexBiasAndScaleBuffer = geom.displacementVertexBiasAndScaleBuffer;
        if (triangleFlagsBuffer)
            *triangleFlagsBuffer = geom.displacementTriangleFlagsBuffer;
        if (dmmIndexBuffer)
            *dmmIndexBuffer = geom.displacementMicroMapIndexBuffer;
        if (indexSize)
            *indexSize = convertToIndexSizeEnum(geom.opacityMicroMapIndexSize);
        if (indexOffset)
            *indexOffset = geom.opacityMicroMapIndexOffset;
        if (vertexDirectionFormat)
            *vertexDirectionFormat = geom.displacementVertexDirectionFormat;
        if (vertexBiasAndScaleFormat)
            *vertexBiasAndScaleFormat = geom.displacementVertexBiasAndScaleFormat;
        return geom.displacementMicroMapArray->getPublicType();
    }

    BufferView GeometryInstance::getSegmentIndexBuffer() const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::CurveGeometry>(m->geometry),
            "This geometry instance was created not for curves.");
        const auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
        return geom.segmentIndexBuffer;
    }

    BufferView GeometryInstance::getCustomPrimitiveAABBBuffer(uint32_t motionStep) const {
        m->throwRuntimeError(
            std::holds_alternative<Priv::CustomPrimitiveGeometry>(m->geometry),
            "This geometry instance was created not for custom primitives.");
        m->throwRuntimeError(
            motionStep < m->numMotionSteps,
            "motionStep %u is out of bounds [0, %u).",
            motionStep, m->numMotionSteps);
        const auto &geom = std::get<Priv::CustomPrimitiveGeometry>(m->geometry);
        return geom.primitiveAabbBuffers[motionStep];
    }

    uint32_t GeometryInstance::getPrimitiveIndexOffset() const {
        return m->primitiveIndexOffset;
    }

    uint32_t GeometryInstance::getNumMaterials(BufferView* matIndexBuffer, IndexSize* indexSize) const {
        if (matIndexBuffer || indexSize) {
            if (std::holds_alternative<Priv::TriangleGeometry>(m->geometry)) {
                const auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
                if (matIndexBuffer)
                    *matIndexBuffer = geom.materialIndexBuffer;
                if (indexSize)
                    *indexSize = convertToIndexSizeEnum(geom.materialIndexSize);
            }
            else if (std::holds_alternative<Priv::SphereGeometry>(m->geometry)) {
                const auto &geom = std::get<Priv::SphereGeometry>(m->geometry);
                if (matIndexBuffer)
                    *matIndexBuffer = geom.materialIndexBuffer;
                if (indexSize)
                    *indexSize = convertToIndexSizeEnum(geom.materialIndexSize);
            }
            else if (std::holds_alternative<Priv::CustomPrimitiveGeometry>(m->geometry)) {
                const auto &geom = std::get<Priv::CustomPrimitiveGeometry>(m->geometry);
                if (matIndexBuffer)
                    *matIndexBuffer = geom.materialIndexBuffer;
                if (indexSize)
                    *indexSize = convertToIndexSizeEnum(geom.materialIndexSize);
            }
            else {
                if (matIndexBuffer)
                    *matIndexBuffer = BufferView();
                if (indexSize)
                    *indexSize = IndexSize::None;
            }
        }
        return static_cast<uint32_t>(m->materials.size());
    }

    OptixGeometryFlags GeometryInstance::getGeometryFlags(uint32_t matIdx) const {
        size_t numMaterials = m->materials.size();
        m->throwRuntimeError(
            matIdx < numMaterials, "Out of material bounds [0, %u).",
            static_cast<uint32_t>(numMaterials));
        return m->buildInputFlags[matIdx];
    }

    Material GeometryInstance::getMaterial(uint32_t matSetIdx, uint32_t matIdx) const {
        size_t numMaterials = m->materials.size();
        m->throwRuntimeError(
            matIdx < numMaterials, "Out of material bounds [0, %u).",
            static_cast<uint32_t>(numMaterials));
        size_t numMatSets = m->materials[matIdx].size();
        m->throwRuntimeError(
            matSetIdx < numMatSets, "Out of material set bounds [0, %u).",
            static_cast<uint32_t>(numMatSets));

        return m->materials[matIdx][matSetIdx]->getPublicType();
    }

    void GeometryInstance::getUserData(void* data, uint32_t* size, uint32_t* alignment) const {
        if (data)
            std::memcpy(data, m->userData.data(), m->userDataSizeAlign.size);
        if (size)
            *size = m->userDataSizeAlign.size;
        if (alignment)
            *alignment = m->userDataSizeAlign.alignment;
    }



    void GeometryAccelerationStructure::Priv::calcSBTRequirements(
        uint32_t matSetIdx, SizeAlign* maxRecordSizeAlign, uint32_t* numSBTRecords) const {
        *maxRecordSizeAlign = SizeAlign();
        *numSBTRecords = 0;
        for (const Child &child : children) {
            SizeAlign geomInstRecordSizeAlign;
            uint32_t geomInstNumSBTRecords;
            child.geomInst->calcSBTRequirements(
                matSetIdx,
                userDataSizeAlign,
                child.userDataSizeAlign,
                &geomInstRecordSizeAlign, &geomInstNumSBTRecords);
            geomInstRecordSizeAlign += child.userDataSizeAlign;
            *maxRecordSizeAlign = max(*maxRecordSizeAlign, geomInstRecordSizeAlign);
            *numSBTRecords += geomInstNumSBTRecords;
        }
        *maxRecordSizeAlign += userDataSizeAlign;
        *numSBTRecords *= numRayTypesPerMaterialSet[matSetIdx];
    }

    uint32_t GeometryAccelerationStructure::Priv::fillSBTRecords(
        const _Pipeline* pipeline, uint32_t matSetIdx, uint8_t* records) const {
        throwRuntimeError(
            matSetIdx < numRayTypesPerMaterialSet.size(),
            "Material set index %u is out of bounds [0, %u).",
            matSetIdx, static_cast<uint32_t>(numRayTypesPerMaterialSet.size()));

        uint32_t numRayTypes = numRayTypesPerMaterialSet[matSetIdx];
        uint32_t sumRecords = 0;
        for (uint32_t sbtGasIdx = 0; sbtGasIdx < children.size(); ++sbtGasIdx) {
            const Child &child = children[sbtGasIdx];
            uint32_t numRecords = child.geomInst->fillSBTRecords(
                pipeline, matSetIdx,
                userData.data(), userDataSizeAlign,
                child.userData.data(), child.userDataSizeAlign,
                numRayTypes, records);
            records += numRecords * scene->getSingleRecordSize();
            sumRecords += numRecords;
        }

        return sumRecords;
    }

    void GeometryAccelerationStructure::Priv::markDirty() {
        readyToBuild = false;
        available = false;
        readyToCompact = false;
        compactedAvailable = false;
    }

    void GeometryAccelerationStructure::destroy() {
        if (m) {
            m->scene->markSBTLayoutDirty();
            delete m;
        }
        m = nullptr;
    }

    void GeometryAccelerationStructure::setConfiguration(
        ASTradeoff tradeoff,
        AllowUpdate allowUpdate,
        AllowCompaction allowCompaction,
        AllowRandomVertexAccess allowRandomVertexAccess,
        AllowOpacityMicroMapUpdate allowOpacityMicroMapUpdate,
        AllowDisableOpacityMicroMaps allowDisableOpacityMicroMaps) const {
        m->throwRuntimeError(
            m->geomType != GeometryType::CustomPrimitives || !allowRandomVertexAccess,
            "Random vertex access is the feature only for triangle/curve/sphere GAS.");
        bool changed = false;
        changed |= m->tradeoff != tradeoff;
        m->tradeoff = tradeoff;
        changed |= m->allowUpdate != allowUpdate;
        m->allowUpdate = allowUpdate;
        changed |= m->allowCompaction != allowCompaction;
        m->allowCompaction = allowCompaction;
        changed |= m->allowRandomVertexAccess != allowRandomVertexAccess;
        m->allowRandomVertexAccess = allowRandomVertexAccess;
        changed |= m->allowOpacityMicroMapUpdate != allowOpacityMicroMapUpdate;
        m->allowOpacityMicroMapUpdate = allowOpacityMicroMapUpdate;
        changed |= m->allowDisableOpacityMicroMaps != allowDisableOpacityMicroMaps;
        m->allowDisableOpacityMicroMaps = allowDisableOpacityMicroMaps;

        if (changed)
            m->markDirty();
    }

    void GeometryAccelerationStructure::setMotionOptions(
        uint32_t numKeys, float timeBegin, float timeEnd, OptixMotionFlags flags) const {
        m->buildOptions.motionOptions.numKeys = numKeys;
        m->buildOptions.motionOptions.timeBegin = timeBegin;
        m->buildOptions.motionOptions.timeEnd = timeEnd;
        m->buildOptions.motionOptions.flags = flags;

        m->markDirty();
    }

    void GeometryAccelerationStructure::addChild(
        GeometryInstance geomInst, CUdeviceptr preTransform,
        const void* data, uint32_t size, uint32_t alignment) const {
        auto _geomInst = extract(geomInst);
        m->throwRuntimeError(
            _geomInst,
            "Invalid geometry instance %p.",
            _geomInst);
        m->throwRuntimeError(
            _geomInst->getScene() == m->scene,
            "Scene mismatch for the given geometry instance %s.",
            _geomInst->getName().c_str());
        const char* geomTypeStrs[] = {
            "triangles",
            "linear segments",
            "quadratic B-splines",
            "flat quadratic B-splines",
            "cubic B-splines",
            "Catmull-Rom splines",
            "cubic Bezier splines",
            "custom primitives" };
        m->throwRuntimeError(
            _geomInst->getGeometryType() == m->geomType,
            "This GAS was created for %s.",
            geomTypeStrs[static_cast<uint32_t>(m->geomType)]);
        m->throwRuntimeError(
            m->geomType == GeometryType::Triangles || preTransform == 0,
            "Pre-transform is valid only for triangles.");
        Priv::Child child;
        child.geomInst = _geomInst;
        child.preTransform = preTransform;
        auto idx = std::find(m->children.cbegin(), m->children.cend(), child);
        m->throwRuntimeError(
            idx == m->children.cend(),
            "Geometry instance %s with transform %p has been already added.",
            _geomInst->getName().c_str(), preTransform);
        child.userDataSizeAlign = SizeAlign(size, alignment);
        child.userData.resize(size);
        std::memcpy(child.userData.data(), data, size);

        m->children.push_back(std::move(child));

        m->markDirty();
        m->scene->markSBTLayoutDirty();
    }

    void GeometryAccelerationStructure::removeChildAt(uint32_t index) const {
        uint32_t numChildren = static_cast<uint32_t>(m->children.size());
        m->throwRuntimeError(
            index < numChildren,
            "Index is out of bounds [0, %u).]",
            numChildren);

        m->children.erase(m->children.cbegin() + index);

        m->markDirty();
        m->scene->markSBTLayoutDirty();
    }

    void GeometryAccelerationStructure::clearChildren() const {
        m->children.clear();

        m->markDirty();
        m->scene->markSBTLayoutDirty();
    }

    void GeometryAccelerationStructure::markDirty() const {
        m->markDirty();
        m->scene->markSBTLayoutDirty();
    }

    void GeometryAccelerationStructure::setNumMaterialSets(uint32_t numMatSets) const {
        m->numRayTypesPerMaterialSet.resize(numMatSets, 0);

        m->scene->markSBTLayoutDirty();
    }

    void GeometryAccelerationStructure::setNumRayTypes(uint32_t matSetIdx, uint32_t numRayTypes) const {
        uint32_t numMatSets = static_cast<uint32_t>(m->numRayTypesPerMaterialSet.size());
        m->throwRuntimeError(
            matSetIdx < numMatSets,
            "Material set index %u is out of bounds [0, %u).",
            matSetIdx, numMatSets);
        m->numRayTypesPerMaterialSet[matSetIdx] = numRayTypes;

        m->scene->markSBTLayoutDirty();
    }

    void GeometryAccelerationStructure::prepareForBuild(OptixAccelBufferSizes* memoryRequirement) const {
        m->buildInputs.resize(m->children.size(), OptixBuildInput{});
        uint32_t childIdx = 0;
        uint32_t numMotionSteps = std::max<uint32_t>(m->buildOptions.motionOptions.numKeys, 1u);
        for (const Priv::Child &child : m->children) {
            child.geomInst->fillBuildInput(&m->buildInputs[childIdx++], child.preTransform);
            uint32_t childNumMotionSteps = child.geomInst->getNumMotionSteps();
            m->throwRuntimeError(
                childNumMotionSteps == numMotionSteps,
                "This GAS has %u motion steps but the GeometryInstance %s has the number %u.",
                numMotionSteps, child.geomInst->getName().c_str(), childNumMotionSteps);
        }

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        m->buildOptions.buildFlags = 0;
        if (m->tradeoff == ASTradeoff::PreferFastTrace)
            m->buildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        else if (m->tradeoff == ASTradeoff::PreferFastBuild)
            m->buildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        m->buildOptions.buildFlags |=
            (m->allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0)
            | (m->allowCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0)
            | (m->allowRandomVertexAccess ? OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS : 0)
            | (m->allowOpacityMicroMapUpdate ? OPTIX_BUILD_FLAG_ALLOW_OPACITY_MICROMAP_UPDATE : 0)
            | (m->allowDisableOpacityMicroMaps ? OPTIX_BUILD_FLAG_ALLOW_DISABLE_OPACITY_MICROMAPS : 0);

        uint32_t numBuildInputs = static_cast<uint32_t>(m->buildInputs.size());
        if (numBuildInputs > 0)
            OPTIX_CHECK(optixAccelComputeMemoryUsage(
                m->getRawContext(), &m->buildOptions,
                m->buildInputs.data(), numBuildInputs,
                &m->memoryRequirement));
        else
            m->memoryRequirement = {};

        *memoryRequirement = m->memoryRequirement;

        m->readyToBuild = true;
    }

    OptixTraversableHandle GeometryAccelerationStructure::rebuild(
        CUstream stream, const BufferView &accelBuffer, const BufferView &scratchBuffer) const {
        m->throwRuntimeError(
            m->readyToBuild, "You need to call prepareForBuild() before rebuild.");
        m->throwRuntimeError(
            accelBuffer.sizeInBytes() >= m->memoryRequirement.outputSizeInBytes,
            "Size of the given buffer is not enough.");
        m->throwRuntimeError(
            scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempSizeInBytes,
            "Size of the given scratch buffer is not enough.");

        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        // JP: アップデートの意味でリビルドするときはprepareForBuild()を呼ばないため
        //     ビルド入力を更新する処理をここにも書いておく必要がある。
        // EN: User is not required to call prepareForBuild() when performing rebuild
        //     for purpose of update so updating build inputs should be here.
        uint32_t childIdx = 0;
        for (const Priv::Child &child : m->children)
            child.geomInst->updateBuildInput(&m->buildInputs[childIdx++], child.preTransform);

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        uint32_t numBuildInputs = static_cast<uint32_t>(m->buildInputs.size());
        if (numBuildInputs > 0) {
            OPTIX_CHECK(optixAccelBuild(
                m->getRawContext(), stream,
                &m->buildOptions, m->buildInputs.data(), numBuildInputs,
                scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
                &m->handle,
                compactionEnabled ? &m->propertyCompactedSize : nullptr,
                compactionEnabled ? 1 : 0));
            CUDADRV_CHECK(cuEventRecord(m->finishEvent, stream));
        }
        else {
            m->handle = 0;
        }

        m->accelBuffer = accelBuffer;
        m->available = true;
        m->readyToCompact = false;
        m->compactedHandle = 0;
        m->compactedAvailable = false;

        return m->handle;
    }

    void GeometryAccelerationStructure::prepareForCompact(size_t* compactedAccelBufferSize) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        m->throwRuntimeError(
            compactionEnabled,
            "This AS does not allow compaction.");
        m->throwRuntimeError(
            m->available,
            "Uncompacted AS has not been built yet.");

        if (m->compactedAvailable)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        // EN: Wait the completion of rebuild/update then obtain the size after coompaction.
        // TODO: ? stream
        uint32_t numBuildInputs = static_cast<uint32_t>(m->buildInputs.size());
        if (numBuildInputs > 0) {
            CUDADRV_CHECK(cuEventSynchronize(m->finishEvent));
            CUDADRV_CHECK(cuMemcpyDtoH(
                &m->compactedSize, m->propertyCompactedSize.result, sizeof(m->compactedSize)));
        }
        else {
            m->compactedSize = 0;
        }

        *compactedAccelBufferSize = m->compactedSize;

        m->readyToCompact = true;
    }

    OptixTraversableHandle GeometryAccelerationStructure::compact(
        CUstream stream, const BufferView &compactedAccelBuffer) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        m->throwRuntimeError(
            compactionEnabled,
            "This AS does not allow compaction.");
        m->throwRuntimeError(
            m->readyToCompact,
            "You need to call prepareForCompact() before compaction.");
        m->throwRuntimeError(
            m->available,
            "Uncompacted AS has not been built yet.");
        m->throwRuntimeError(
            compactedAccelBuffer.sizeInBytes() >= m->compactedSize,
            "Size of the given buffer is not enough.");

        uint32_t numBuildInputs = static_cast<uint32_t>(m->buildInputs.size());
        if (numBuildInputs > 0) {
            OPTIX_CHECK(optixAccelCompact(
                m->getRawContext(), stream,
                m->handle, compactedAccelBuffer.getCUdeviceptr(), compactedAccelBuffer.sizeInBytes(),
                &m->compactedHandle));
            CUDADRV_CHECK(cuEventRecord(m->finishEvent, stream));
        }
        else {
            m->compactedHandle = 0;
        }

        m->compactedAccelBuffer = compactedAccelBuffer;
        m->compactedAvailable = true;

        return m->compactedHandle;
    }

    void GeometryAccelerationStructure::removeUncompacted() const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m->compactedAvailable || !compactionEnabled)
            return;

        uint32_t numBuildInputs = static_cast<uint32_t>(m->buildInputs.size());
        if (numBuildInputs > 0)
            CUDADRV_CHECK(cuEventSynchronize(m->finishEvent));

        m->handle = 0;
        m->available = false;
    }

    void GeometryAccelerationStructure::update(CUstream stream, const BufferView &scratchBuffer) const {
        bool updateEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;
        m->throwRuntimeError(
            updateEnabled,
            "This AS does not allow update.");
        m->throwRuntimeError(
            m->available || m->compactedAvailable,
            "AS has not been built yet.");
        m->throwRuntimeError(
            scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempUpdateSizeInBytes,
            "Size of the given scratch buffer is not enough.");

        uint32_t childIdx = 0;
        for (const Priv::Child &child : m->children)
            child.geomInst->updateBuildInput(&m->buildInputs[childIdx++], child.preTransform);

        const BufferView &accelBuffer = m->compactedAvailable ? m->compactedAccelBuffer : m->accelBuffer;
        OptixTraversableHandle handle = m->compactedAvailable ? m->compactedHandle : m->handle;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OptixTraversableHandle tempHandle = handle;
        uint32_t numBuildInputs = static_cast<uint32_t>(m->buildInputs.size());
        if (numBuildInputs > 0)
            OPTIX_CHECK(optixAccelBuild(
                m->getRawContext(), stream,
                &m->buildOptions, m->buildInputs.data(), numBuildInputs,
                scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
                &tempHandle,
                nullptr, 0));
        else
            tempHandle = 0;
        optixuAssert(
            tempHandle == handle,
            "GAS %s: Update should not change the handle itself, what's going on?",
            getName());
    }

    void GeometryAccelerationStructure::setChildUserData(
        uint32_t index, const void* data, uint32_t size, uint32_t alignment) const {
        uint32_t numChildren = static_cast<uint32_t>(m->children.size());
        m->throwRuntimeError(
            index < numChildren,
            "Index is out of bounds [0, %u).]",
            numChildren);
        m->throwRuntimeError(
            size <= s_maxGASChildUserDataSize,
            "Maximum user data size for GAS child is %u bytes.",
            s_maxGASChildUserDataSize);
        m->throwRuntimeError(
            alignment > 0 && alignment <= OPTIX_SBT_RECORD_ALIGNMENT,
            "Valid alignment range is [1, %u].",
            OPTIX_SBT_RECORD_ALIGNMENT);
        Priv::Child &child = m->children[index];
        if (child.userDataSizeAlign.size != size ||
            child.userDataSizeAlign.alignment != alignment)
            m->scene->markSBTLayoutDirty();
        child.userDataSizeAlign = SizeAlign(size, alignment);
        child.userData.resize(size);
        std::memcpy(child.userData.data(), data, size);
    }

    void GeometryAccelerationStructure::setUserData(
        const void* data, uint32_t size, uint32_t alignment) const {
        m->throwRuntimeError(
            size <= s_maxGASUserDataSize,
            "Maximum user data size for GAS is %u bytes.",
            s_maxGASUserDataSize);
        m->throwRuntimeError(
            alignment > 0 && alignment <= OPTIX_SBT_RECORD_ALIGNMENT,
            "Valid alignment range is [1, %u].",
            OPTIX_SBT_RECORD_ALIGNMENT);
        if (m->userDataSizeAlign.size != size ||
            m->userDataSizeAlign.alignment != alignment)
            m->scene->markSBTLayoutDirty();
        m->userDataSizeAlign = SizeAlign(size, alignment);
        m->userData.resize(size);
        std::memcpy(m->userData.data(), data, size);
    }

    bool GeometryAccelerationStructure::isReady() const {
        return m->isReady();
    }

    OptixTraversableHandle GeometryAccelerationStructure::getHandle() const {
        return m->getHandle();
    }

    void GeometryAccelerationStructure::getConfiguration(
        ASTradeoff* tradeOff,
        AllowUpdate* allowUpdate,
        AllowCompaction* allowCompaction,
        AllowRandomVertexAccess* allowRandomVertexAccess,
        AllowOpacityMicroMapUpdate* allowOpacityMicroMapUpdate,
        AllowDisableOpacityMicroMaps* allowDisableOpacityMicroMaps) const {
        if (tradeOff)
            *tradeOff = m->tradeoff;
        if (allowUpdate)
            *allowUpdate = AllowUpdate(m->allowUpdate);
        if (allowCompaction)
            *allowCompaction = AllowCompaction(m->allowCompaction);
        if (allowRandomVertexAccess)
            *allowRandomVertexAccess = AllowRandomVertexAccess(m->allowRandomVertexAccess);
        if (allowOpacityMicroMapUpdate)
            *allowOpacityMicroMapUpdate = AllowOpacityMicroMapUpdate(m->allowOpacityMicroMapUpdate);
        if (allowDisableOpacityMicroMaps)
            *allowDisableOpacityMicroMaps = AllowDisableOpacityMicroMaps(m->allowDisableOpacityMicroMaps);
    }

    void GeometryAccelerationStructure::getMotionOptions(
        uint32_t* numKeys, float* timeBegin, float* timeEnd, OptixMotionFlags* flags) const {
        if (numKeys)
            *numKeys = m->buildOptions.motionOptions.numKeys;
        if (timeBegin)
            *timeBegin = m->buildOptions.motionOptions.timeBegin;
        if (timeEnd)
            *timeEnd = m->buildOptions.motionOptions.timeEnd;
        if (flags)
            *flags = static_cast<OptixMotionFlags>(m->buildOptions.motionOptions.flags);
    }

    uint32_t GeometryAccelerationStructure::getNumChildren() const {
        return static_cast<uint32_t>(m->children.size());
    }

    uint32_t GeometryAccelerationStructure::findChildIndex(
        GeometryInstance geomInst, CUdeviceptr preTransform) const {
        auto _geomInst = extract(geomInst);
        m->throwRuntimeError(
            _geomInst,
            "Invalid geometry instance %p.",
            _geomInst);
        m->throwRuntimeError(
            _geomInst->getScene() == m->scene,
            "Scene mismatch for the given geometry instance %s.",
            _geomInst->getName().c_str());
        Priv::Child child;
        child.geomInst = _geomInst;
        child.preTransform = preTransform;
        auto idx = std::find(m->children.cbegin(), m->children.cend(), child);
        if (idx == m->children.cend())
            return 0xFFFFFFFF;

        return static_cast<uint32_t>(std::distance(m->children.cbegin(), idx));
    }

    GeometryInstance GeometryAccelerationStructure::getChild(
        uint32_t index, CUdeviceptr* preTransform) const {
        uint32_t numChildren = static_cast<uint32_t>(m->children.size());
        m->throwRuntimeError(
            index < numChildren,
            "Index is out of bounds [0, %u).]",
            numChildren);
        if (preTransform)
            *preTransform = m->children[index].preTransform;
        return m->children[index].geomInst->getPublicType();
    }

    uint32_t GeometryAccelerationStructure::getNumMaterialSets() const {
        return static_cast<uint32_t>(m->numRayTypesPerMaterialSet.size());
    }

    uint32_t GeometryAccelerationStructure::getNumRayTypes(uint32_t matSetIdx) const {
        uint32_t numMatSets = static_cast<uint32_t>(m->numRayTypesPerMaterialSet.size());
        m->throwRuntimeError(
            matSetIdx < numMatSets,
            "Material set index %u is out of bounds [0, %u).",
            matSetIdx, numMatSets);
        return m->numRayTypesPerMaterialSet[matSetIdx];
    }

    void GeometryAccelerationStructure::getChildUserData(
        uint32_t index, void* data, uint32_t* size, uint32_t* alignment) const {
        uint32_t numChildren = static_cast<uint32_t>(m->children.size());
        m->throwRuntimeError(
            index < numChildren,
            "Index is out of bounds [0, %u).]",
            numChildren);
        const Priv::Child &child = m->children[index];
        if (data)
            std::memcpy(data, child.userData.data(), child.userDataSizeAlign.size);
        if (size)
            *size = child.userDataSizeAlign.size;
        if (alignment)
            *alignment = child.userDataSizeAlign.alignment;
    }

    void GeometryAccelerationStructure::getUserData(
        void* data, uint32_t* size, uint32_t* alignment) const {
        if (data)
            std::memcpy(data, m->userData.data(), m->userDataSizeAlign.size);
        if (size)
            *size = m->userDataSizeAlign.size;
        if (alignment)
            *alignment = m->userDataSizeAlign.alignment;
    }



    _GeometryAccelerationStructure* Transform::Priv::getDescendantGAS() const {
        if (std::holds_alternative<_GeometryAccelerationStructure*>(child))
            return std::get<_GeometryAccelerationStructure*>(child);
        else if (std::holds_alternative<_InstanceAccelerationStructure*>(child))
            return nullptr;
        else if (std::holds_alternative<_Transform*>(child))
            return std::get<_Transform*>(child)->getDescendantGAS();
        optixuAssert_ShouldNotBeCalled();
        return nullptr;
    }

    void Transform::Priv::markDirty() {
        available = false;
    }

    void Transform::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    void Transform::setConfiguration(
        TransformType type, uint32_t numKeys,
        size_t* transformSize) const {
        m->type = type;
        numKeys = std::max(numKeys, 2u);
        if (m->type == TransformType::MatrixMotion) {
            m->dataSize = sizeof(OptixMatrixMotionTransform) +
                (numKeys - 2) * 12 * sizeof(float);
            m->options.numKeys = numKeys;
            m->data = new uint8_t[m->dataSize];
            std::memset(m->data, 0, m->dataSize);
            auto motionData = reinterpret_cast<float*>(m->data + offsetof(OptixMatrixMotionTransform, transform));
            for (uint32_t i = 0; i < numKeys; ++i) {
                float* dataPerKey = motionData + 12 * i;
                dataPerKey[0] = 1.0f; dataPerKey[1] = 0.0f; dataPerKey[2] = 0.0f; dataPerKey[3] = 0.0f;
                dataPerKey[4] = 1.0f; dataPerKey[5] = 0.0f; dataPerKey[6] = 0.0f; dataPerKey[7] = 0.0f;
                dataPerKey[8] = 1.0f; dataPerKey[9] = 0.0f; dataPerKey[10] = 0.0f; dataPerKey[11] = 0.0f;
            }
        }
        else if (m->type == TransformType::SRTMotion) {
            m->dataSize = sizeof(OptixSRTMotionTransform) +
                (numKeys - 2) * sizeof(OptixSRTData);
            m->options.numKeys = numKeys;
            m->data = new uint8_t[m->dataSize];
            std::memset(m->data, 0, m->dataSize);
            auto motionData = reinterpret_cast<OptixSRTData*>(
                m->data + offsetof(OptixSRTMotionTransform, srtData));
            for (uint32_t i = 0; i < numKeys; ++i) {
                OptixSRTData* dataPerKey = motionData + i;
                dataPerKey->sx = dataPerKey->sy = dataPerKey->sz = 1.0f;
                dataPerKey->a = dataPerKey->b = dataPerKey->c = 0.0f;
                dataPerKey->pvx = dataPerKey->pvy = dataPerKey->pvz = 0.0f;
                dataPerKey->qx = dataPerKey->qy = dataPerKey->qz = 0.0f;
                dataPerKey->qw = 1.0f;
                dataPerKey->tx = dataPerKey->ty = dataPerKey->tz = 0.0f;
            }
        }
        else if (m->type == TransformType::Static) {
            m->dataSize = sizeof(OptixStaticTransform);
            m->data = new uint8_t[m->dataSize];
            std::memset(m->data, 0, m->dataSize);
            auto xfm = reinterpret_cast<OptixStaticTransform*>(m->data);
            xfm->transform[0] = 1.0f; xfm->transform[1] = 0.0f; xfm->transform[2] = 0.0f; xfm->transform[3] = 0.0f;
            xfm->transform[4] = 1.0f; xfm->transform[5] = 0.0f; xfm->transform[6] = 0.0f; xfm->transform[7] = 0.0f;
            xfm->transform[8] = 1.0f; xfm->transform[9] = 0.0f; xfm->transform[10] = 0.0f; xfm->transform[11] = 0.0f;
            xfm->invTransform[0] = 1.0f; xfm->invTransform[1] = 0.0f; xfm->invTransform[2] = 0.0f; xfm->invTransform[3] = 0.0f;
            xfm->invTransform[4] = 1.0f; xfm->invTransform[5] = 0.0f; xfm->invTransform[6] = 0.0f; xfm->invTransform[7] = 0.0f;
            xfm->invTransform[8] = 1.0f; xfm->invTransform[9] = 0.0f; xfm->invTransform[10] = 0.0f; xfm->invTransform[11] = 0.0f;
        }
        else {
            m->throwRuntimeError(false, "Invalid transform type %u.", static_cast<uint32_t>(type));
        }

        *transformSize = m->dataSize;

        markDirty();
    }

    void Transform::setMotionOptions(float timeBegin, float timeEnd, OptixMotionFlags flags) const {
        m->options.timeBegin = timeBegin;
        m->options.timeEnd = timeEnd;
        m->options.flags = flags;

        markDirty();
    }

    void Transform::setMatrixMotionKey(uint32_t keyIdx, const float matrix[12]) const {
        m->throwRuntimeError(
            m->type == TransformType::MatrixMotion,
            "This transform has been configured as matrix motion transform.");
        m->throwRuntimeError(
            keyIdx <= m->options.numKeys,
            "Number of motion keys was set to %u", m->options.numKeys);
        auto motionData = reinterpret_cast<float*>(
            m->data + offsetof(OptixMatrixMotionTransform, transform));
        float* dataPerKey = motionData + 12 * keyIdx;

        std::copy_n(matrix, 12, dataPerKey);

        markDirty();
    }

    void Transform::setSRTMotionKey(
        uint32_t keyIdx,
        const float scale[3], const float orientation[4], const float translation[3]) const {
        m->throwRuntimeError(
            m->type == TransformType::SRTMotion,
            "This transform has been configured as SRT motion transform.");
        m->throwRuntimeError(
            keyIdx <= m->options.numKeys,
            "Number of motion keys was set to %u", m->options.numKeys);
        auto motionData = reinterpret_cast<OptixSRTData*>(
            m->data + offsetof(OptixSRTMotionTransform, srtData));
        OptixSRTData* dataPerKey = motionData + keyIdx;

        dataPerKey->sx = scale[0];
        dataPerKey->sy = scale[1];
        dataPerKey->sz = scale[2];
        dataPerKey->a = dataPerKey->b = dataPerKey->c = 0.0f;
        dataPerKey->pvx = dataPerKey->pvy = dataPerKey->pvz = 0.0f;
        std::copy_n(orientation, 4, &dataPerKey->qx);
        std::copy_n(translation, 3, &dataPerKey->tx);

        markDirty();
    }

    void Transform::setStaticTransform(const float matrix[12]) const {
        m->throwRuntimeError(
            m->type == TransformType::Static,
            "This transform has been configured as static transform.");
        float invDet = 1.0f / (matrix[ 0] * matrix[ 5] * matrix[10] +
                               matrix[ 1] * matrix[ 6] * matrix[ 8] +
                               matrix[ 2] * matrix[ 4] * matrix[ 9] -
                               matrix[ 2] * matrix[ 5] * matrix[ 8] -
                               matrix[ 1] * matrix[ 4] * matrix[10] -
                               matrix[ 0] * matrix[ 6] * matrix[ 9]);
        m->throwRuntimeError(invDet != 0.0f, "Given matrix is not invertible.");

        auto xfm = reinterpret_cast<OptixStaticTransform*>(m->data);

        std::copy_n(matrix, 12, xfm->transform);

        float invMat[12];
        invMat[ 0] = invDet * (matrix[ 5] * matrix[10] - matrix[ 6] * matrix[ 9]);
        invMat[ 1] = invDet * (matrix[ 2] * matrix[ 9] - matrix[ 1] * matrix[10]);
        invMat[ 2] = invDet * (matrix[ 1] * matrix[ 6] - matrix[ 2] * matrix[ 5]);
        invMat[ 3] = -matrix[3];
        invMat[ 4] = invDet * (matrix[ 6] * matrix[ 8] - matrix[ 4] * matrix[10]);
        invMat[ 5] = invDet * (matrix[ 0] * matrix[10] - matrix[ 2] * matrix[ 8]);
        invMat[ 6] = invDet * (matrix[ 2] * matrix[ 4] - matrix[ 0] * matrix[ 6]);
        invMat[ 7] = -matrix[7];
        invMat[ 8] = invDet * (matrix[ 4] * matrix[ 9] - matrix[ 5] * matrix[ 8]);
        invMat[ 9] = invDet * (matrix[ 1] * matrix[ 8] - matrix[ 0] * matrix[ 9]);
        invMat[10] = invDet * (matrix[ 0] * matrix[ 5] - matrix[ 1] * matrix[ 4]);
        invMat[11] = -matrix[11];
        std::copy_n(invMat, 12, xfm->invTransform);

        markDirty();
    }

    void Transform::setChild(GeometryAccelerationStructure child) const {
        auto _child = extract(child);
        m->throwRuntimeError(
            _child,
            "Invalid GAS %p.",
            _child);
        m->throwRuntimeError(
            _child->getScene() == m->scene,
            "Scene mismatch for the given GAS %s.",
            _child->getName().c_str());
        m->child = _child;

        markDirty();
    }

    void Transform::setChild(InstanceAccelerationStructure child) const {
        auto _child = extract(child);
        m->throwRuntimeError(
            _child,
            "Invalid IAS %p.",
            _child);
        m->throwRuntimeError(
            _child->getScene() == m->scene,
            "Scene mismatch for the given IAS %s.",
            _child->getName().c_str());
        m->child = _child;

        markDirty();
    }

    void Transform::setChild(Transform child) const {
        auto _child = extract(child);
        m->throwRuntimeError(
            _child,
            "Invalid transform %p.",
            _child);
        m->throwRuntimeError(
            _child->getScene() == m->scene,
            "Scene mismatch for the given transform %s.",
            _child->getName().c_str());
        m->child = _child;

        markDirty();
    }

    void Transform::markDirty() const {
        return m->markDirty();
    }

    OptixTraversableHandle Transform::rebuild(CUstream stream, const BufferView &trDeviceMem) const {
        m->throwRuntimeError(
            m->type != TransformType::Invalid,
            "Transform type is invalid.");
        m->throwRuntimeError(
            trDeviceMem.sizeInBytes() >= m->dataSize,
            "Size of the given buffer is not enough.");
        m->throwRuntimeError(
            !std::holds_alternative<void*>(m->child),
            "Child is invalid.");

        OptixTraversableHandle childHandle = 0;
        if (std::holds_alternative<_GeometryAccelerationStructure*>(m->child))
            childHandle = std::get<_GeometryAccelerationStructure*>(m->child)->getHandle();
        else if (std::holds_alternative<_InstanceAccelerationStructure*>(m->child))
            childHandle = std::get<_InstanceAccelerationStructure*>(m->child)->getHandle();
        else if (std::holds_alternative<_Transform*>(m->child))
            childHandle = std::get<_Transform*>(m->child)->getHandle();
        else 
            optixuAssert_ShouldNotBeCalled();

        OptixTraversableType travType;
        if (m->type == TransformType::MatrixMotion) {
            auto tr = reinterpret_cast<OptixMatrixMotionTransform*>(m->data);
            tr->child = childHandle;
            tr->motionOptions = m->options;
            travType = OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM;
        }
        else if (m->type == TransformType::SRTMotion) {
            auto tr = reinterpret_cast<OptixSRTMotionTransform*>(m->data);
            tr->child = childHandle;
            tr->motionOptions = m->options;
            travType = OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM;
        }
        else if (m->type == TransformType::Static) {
            auto tr = reinterpret_cast<OptixStaticTransform*>(m->data);
            tr->child = childHandle;
            travType = OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM;
        }
        else {
            optixuAssert_ShouldNotBeCalled();
        }

        CUDADRV_CHECK(cuMemcpyHtoDAsync(trDeviceMem.getCUdeviceptr(), m->data, m->dataSize, stream));
        OPTIX_CHECK(optixConvertPointerToTraversableHandle(
            m->getRawContext(), trDeviceMem.getCUdeviceptr(),
            travType,
            &m->handle));
        m->available = true;

        return m->handle;
    }

    bool Transform::isReady() const {
        return m->isReady();
    }

    OptixTraversableHandle Transform::getHandle() const {
        return m->getHandle();
    }

    void Transform::getConfiguration(TransformType* type, uint32_t* numKeys) const {
        if (type)
            *type = m->type;
        if (numKeys)
            *numKeys = m->options.numKeys;
    }

    void Transform::getMotionOptions(float* timeBegin, float* timeEnd, OptixMotionFlags* flags) const {
        if (timeBegin)
            *timeBegin = m->options.timeBegin;
        if (timeEnd)
            *timeEnd = m->options.timeEnd;
        if (flags)
            *flags = static_cast<OptixMotionFlags>(m->options.flags);
    }

    void Transform::getMatrixMotionKey(uint32_t keyIdx, float matrix[12]) const {
        m->throwRuntimeError(
            m->type == TransformType::MatrixMotion,
            "This transform has been configured as matrix motion transform.");
        m->throwRuntimeError(
            keyIdx <= m->options.numKeys,
            "Number of motion keys was set to %u",
            m->options.numKeys);
        auto motionData = reinterpret_cast<const float*>(
            m->data + offsetof(OptixMatrixMotionTransform, transform));
        const float* dataPerKey = motionData + 12 * keyIdx;

        std::copy_n(dataPerKey, 12, matrix);
    }

    void Transform::getSRTMotionKey(
        uint32_t keyIdx, float scale[3], float orientation[4], float translation[3]) const {
        m->throwRuntimeError(
            m->type == TransformType::SRTMotion,
            "This transform has been configured as SRT motion transform.");
        m->throwRuntimeError(
            keyIdx <= m->options.numKeys,
            "Number of motion keys was set to %u",
            m->options.numKeys);
        auto motionData = reinterpret_cast<const OptixSRTData*>(
            m->data + offsetof(OptixSRTMotionTransform, srtData));
        const OptixSRTData* dataPerKey = motionData + keyIdx;

        scale[0] = dataPerKey->sx;
        scale[1] = dataPerKey->sy;
        scale[2] = dataPerKey->sz;
        std::copy_n(&dataPerKey->qx, 4, orientation);
        std::copy_n(&dataPerKey->tx, 3, translation);
    }

    void Transform::getStaticTransform(float matrix[12]) const {
        m->throwRuntimeError(
            m->type == TransformType::Static,
            "This transform has been configured as static transform.");

        auto xfm = reinterpret_cast<const OptixStaticTransform*>(m->data);

        std::copy_n(xfm->transform, 12, matrix);
    }

    ChildType Transform::getChildType() const {
        if (std::holds_alternative<_GeometryAccelerationStructure*>(m->child))
            return ChildType::GAS;
        else if (std::holds_alternative<_InstanceAccelerationStructure*>(m->child))
            return ChildType::IAS;
        else if (std::holds_alternative<_Transform*>(m->child))
            return ChildType::Transform;
        return ChildType::Invalid;
    }

    template <typename T>
    T Transform::getChild() const {
        m->throwRuntimeError(
            std::holds_alternative<typename T::Priv*>(m->child),
            "Given type is inconsistent with the stored type.");
        return std::get<typename T::Priv*>(m->child)->getPublicType();
    }
    template GeometryAccelerationStructure Transform::getChild<GeometryAccelerationStructure>() const;
    template InstanceAccelerationStructure Transform::getChild<InstanceAccelerationStructure>() const;
    template Transform Transform::getChild<Transform>() const;



    void Instance::Priv::fillInstance(OptixInstance* instance) const {
        throwRuntimeError(!std::holds_alternative<void*>(child), "Child has not been set.");

        *instance = {};
        std::copy_n(instTransform, 12, instance->transform);
        instance->instanceId = id;

        if (std::holds_alternative<_GeometryAccelerationStructure*>(child)) {
            auto gas = std::get<_GeometryAccelerationStructure*>(child);
            throwRuntimeError(gas->isReady(), "GAS %s is not ready.", gas->getName().c_str());
            instance->traversableHandle = gas->getHandle();
            instance->sbtOffset = scene->getSBTOffset(gas, matSetIndex);
        }
        else if (std::holds_alternative<_InstanceAccelerationStructure*>(child)) {
            auto ias = std::get<_InstanceAccelerationStructure*>(child);
            throwRuntimeError(ias->isReady(), "IAS %s is not ready.", ias->getName().c_str());
            instance->traversableHandle = ias->getHandle();
            instance->sbtOffset = 0;
        }
        else if (std::holds_alternative<_Transform*>(child)) {
            auto xfm = std::get<_Transform*>(child);
            throwRuntimeError(xfm->isReady(), "Transform %s is not ready.", xfm->getName().c_str());
            instance->traversableHandle = xfm->getHandle();
            _GeometryAccelerationStructure* desGas = xfm->getDescendantGAS();
            if (desGas)
                instance->sbtOffset = scene->getSBTOffset(desGas, matSetIndex);
            else
                instance->sbtOffset = 0;
        }
        else {
            optixuAssert_ShouldNotBeCalled();
        }

        instance->visibilityMask = visibilityMask;
        instance->flags = flags;
    }

    void Instance::Priv::updateInstance(OptixInstance* instance) const {
        std::copy_n(instTransform, 12, instance->transform);
        instance->instanceId = id;

        if (std::holds_alternative<_GeometryAccelerationStructure*>(child)) {
            auto gas = std::get<_GeometryAccelerationStructure*>(child);
            throwRuntimeError(gas->isReady(), "GAS %s is not ready.", gas->getName().c_str());
            instance->sbtOffset = scene->getSBTOffset(gas, matSetIndex);
        }
        else if (std::holds_alternative<_InstanceAccelerationStructure*>(child)) {
            auto ias = std::get<_InstanceAccelerationStructure*>(child);
            throwRuntimeError(ias->isReady(), "IAS %s is not ready.", ias->getName().c_str());
            instance->sbtOffset = 0;
        }
        else if (std::holds_alternative<_Transform*>(child)) {
            auto xfm = std::get<_Transform*>(child);
            throwRuntimeError(xfm->isReady(), "Transform %s is not ready.", xfm->getName().c_str());
            _GeometryAccelerationStructure* desGas = xfm->getDescendantGAS();
            if (desGas)
                instance->sbtOffset = scene->getSBTOffset(desGas, matSetIndex);
            else
                instance->sbtOffset = 0;
        }
        else {
            optixuAssert_ShouldNotBeCalled();
        }

        instance->visibilityMask = visibilityMask;
        instance->flags = flags;
    }

    bool Instance::Priv::isMotionAS() const {
        if (std::holds_alternative<_GeometryAccelerationStructure*>(child))
            std::get<_GeometryAccelerationStructure*>(child)->hasMotion();
        else if (std::holds_alternative<_InstanceAccelerationStructure*>(child))
            std::get<_InstanceAccelerationStructure*>(child)->hasMotion();
        return false;
    }

    bool Instance::Priv::isTransform() const {
        return std::holds_alternative<_Transform*>(child);
    }

    void Instance::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    void Instance::setChild(GeometryAccelerationStructure child, uint32_t matSetIdx) const {
        auto _child = extract(child);
        m->throwRuntimeError(
            _child,
            "Invalid GAS %p.",
            _child);
        m->throwRuntimeError(
            _child->getScene() == m->scene,
            "Scene mismatch for the given GAS %s.",
            _child->getName().c_str());
        m->child = _child;
        m->matSetIndex = matSetIdx;
    }

    void Instance::setChild(InstanceAccelerationStructure child) const {
        auto _child = extract(child);
        m->throwRuntimeError(
            _child,
            "Invalid IAS %p.",
            _child);
        m->throwRuntimeError(
            _child->getScene() == m->scene,
            "Scene mismatch for the given IAS %s.",
            _child->getName().c_str());
        m->child = _child;
        m->matSetIndex = 0;
    }

    void Instance::setChild(Transform child, uint32_t matSetIdx) const {
        auto _child = extract(child);
        m->throwRuntimeError(
            _child,
            "Invalid transform %p.",
            _child);
        m->throwRuntimeError(
            _child->getScene() == m->scene,
            "Scene mismatch for the given transform %s.",
            _child->getName().c_str());
        m->child = _child;
        m->matSetIndex = matSetIdx;
    }

    void Instance::setTransform(const float transform[12]) const {
        std::copy_n(transform, 12, m->instTransform);
    }

    void Instance::setID(uint32_t value) const {
        uint32_t maxInstanceID = m->scene->getContext()->getMaxInstanceID();
        m->throwRuntimeError(
            value <= maxInstanceID,
            "Max instance ID value is 0x%08x.",
            maxInstanceID);
        m->id = value;
    }

    void Instance::setVisibilityMask(uint32_t mask) const {
        uint32_t numVisibilityMaskBits = m->scene->getContext()->getNumVisibilityMaskBits();
        m->throwRuntimeError(
            (mask >> numVisibilityMaskBits) == 0,
            "Number of visibility mask bits is %u.",
            numVisibilityMaskBits);
        m->visibilityMask = mask;
    }

    void Instance::setFlags(OptixInstanceFlags flags) const {
        m->flags = flags;
    }

    void Instance::setMaterialSetIndex(uint32_t matSetIdx) const {
        m->matSetIndex = matSetIdx;
    }

    ChildType Instance::getChildType() const {
        if (std::holds_alternative<_GeometryAccelerationStructure*>(m->child))
            return ChildType::GAS;
        else if (std::holds_alternative<_InstanceAccelerationStructure*>(m->child))
            return ChildType::IAS;
        else if (std::holds_alternative<_Transform*>(m->child))
            return ChildType::Transform;
        return ChildType::Invalid;
    }

    template <typename T>
    T Instance::getChild() const {
        m->throwRuntimeError(
            std::holds_alternative<typename T::Priv*>(m->child),
            "Given type is inconsistent with the stored type.");
        return std::get<typename T::Priv*>(m->child)->getPublicType();
    }
    template GeometryAccelerationStructure Instance::getChild<GeometryAccelerationStructure>() const;
    template InstanceAccelerationStructure Instance::getChild<InstanceAccelerationStructure>() const;
    template Transform Instance::getChild<Transform>() const;

    uint32_t Instance::getID() const {
        return m->id;
    }

    uint32_t Instance::getVisibilityMask() const {
        return m->visibilityMask;
    }

    OptixInstanceFlags Instance::getFlags() const {
        return m->flags;
    }

    void Instance::getTransform(float transform[12]) const {
        std::copy_n(m->instTransform, 12, transform);
    }

    uint32_t Instance::getMaterialSetIndex() const {
        return m->matSetIndex;
    }



    void InstanceAccelerationStructure::Priv::markDirty(bool readyToBuild) {
        readyToBuild = readyToBuild;
        available = false;
        readyToCompact = false;
        compactedAvailable = false;
    }

    void InstanceAccelerationStructure::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    void InstanceAccelerationStructure::setConfiguration(
        ASTradeoff tradeoff,
        AllowUpdate allowUpdate,
        AllowCompaction allowCompaction,
        AllowRandomInstanceAccess allowRandomInstanceAccess) const {
        bool changed = false;
        changed |= m->tradeoff != tradeoff;
        m->tradeoff = tradeoff;
        changed |= m->allowUpdate != allowUpdate;
        m->allowUpdate = allowUpdate;
        changed |= m->allowCompaction != allowCompaction;
        m->allowCompaction = allowCompaction;
        changed |= m->allowRandomInstanceAccess != allowRandomInstanceAccess;
        m->allowRandomInstanceAccess = allowRandomInstanceAccess;

        if (changed)
            m->markDirty(false);
    }

    void InstanceAccelerationStructure::setMotionOptions(
        uint32_t numKeys, float timeBegin, float timeEnd, OptixMotionFlags flags) const {
        m->buildOptions.motionOptions.numKeys = numKeys;
        m->buildOptions.motionOptions.timeBegin = timeBegin;
        m->buildOptions.motionOptions.timeEnd = timeEnd;
        m->buildOptions.motionOptions.flags = flags;

        m->markDirty(false);
    }

    void InstanceAccelerationStructure::addChild(Instance instance) const {
        _Instance* _inst = extract(instance);
        m->throwRuntimeError(
            _inst,
            "Invalid instance %p.");
        m->throwRuntimeError(
            _inst->getScene() == m->scene,
            "Scene mismatch for the given instance %s.",
            _inst->getName().c_str());
        auto idx = std::find(m->children.cbegin(), m->children.cend(), _inst);
        m->throwRuntimeError(
            idx == m->children.cend(),
            "Instance %s has been already added.",
            _inst->getName().c_str());

        m->children.push_back(_inst);

        m->markDirty(false);
    }

    void InstanceAccelerationStructure::removeChildAt(uint32_t index) const {
        uint32_t numChildren = static_cast<uint32_t>(m->children.size());
        m->throwRuntimeError(
            index < numChildren,
            "Index is out of bounds [0, %u).]",
            numChildren);

        m->children.erase(m->children.cbegin() + index);

        m->markDirty(false);
    }

    void InstanceAccelerationStructure::clearChildren() const {
        m->children.clear();

        m->markDirty(false);
    }

    void InstanceAccelerationStructure::markDirty() const {
        m->markDirty(false);
    }

    void InstanceAccelerationStructure::prepareForBuild(OptixAccelBufferSizes* memoryRequirement) const {
        m->instances.resize(m->children.size());

        // Fill the build input.
        {
            m->buildInput = OptixBuildInput{};
            m->buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            OptixBuildInputInstanceArray &instArray = m->buildInput.instanceArray;
            instArray.instances = 0;
            instArray.numInstances = static_cast<uint32_t>(m->children.size());
        }

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        m->buildOptions.buildFlags = 0;
        if (m->tradeoff == ASTradeoff::PreferFastTrace)
            m->buildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        else if (m->tradeoff == ASTradeoff::PreferFastBuild)
            m->buildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        m->buildOptions.buildFlags |= 
            (m->allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0)
            | (m->allowCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0)
            | (m->allowRandomInstanceAccess ? OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS : 0);

        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            m->getRawContext(), &m->buildOptions,
            &m->buildInput, 1,
            &m->memoryRequirement));

        *memoryRequirement = m->memoryRequirement;

        m->readyToBuild = true;
    }

    OptixTraversableHandle InstanceAccelerationStructure::rebuild(
        CUstream stream, const BufferView &instanceBuffer,
        const BufferView &accelBuffer, const BufferView &scratchBuffer) const {
        m->throwRuntimeError(
            m->readyToBuild, "You need to call prepareForBuild() before rebuild.");
        m->throwRuntimeError(
            accelBuffer.sizeInBytes() >= m->memoryRequirement.outputSizeInBytes,
            "Size of the given buffer is not enough.");
        m->throwRuntimeError(
            scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempSizeInBytes,
            "Size of the given scratch buffer is not enough.");
        m->throwRuntimeError(
            instanceBuffer.sizeInBytes() >= m->instances.size() * sizeof(OptixInstance),
            "Size of the given instance buffer is not enough.");
        m->throwRuntimeError(
            m->scene->sbtLayoutGenerationDone(),
            "Shader binding table layout generation has not been done.");

        uint32_t childIdx = 0;
        for (const _Instance* child : m->children)
            child->fillInstance(&m->instances[childIdx++]);
        CUDADRV_CHECK(cuMemcpyHtoDAsync(
            instanceBuffer.getCUdeviceptr(), m->instances.data(),
            m->instances.size() * sizeof(OptixInstance),
            stream));
        m->buildInput.instanceArray.instances = m->children.size() > 0 ? instanceBuffer.getCUdeviceptr() : 0;

        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(
            m->getRawContext(), stream, &m->buildOptions, &m->buildInput, 1,
            scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
            accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
            &m->handle,
            compactionEnabled ? &m->propertyCompactedSize : nullptr,
            compactionEnabled ? 1 : 0));
        CUDADRV_CHECK(cuEventRecord(m->finishEvent, stream));

        m->instanceBuffer = instanceBuffer;
        m->accelBuffer = accelBuffer;
        m->available = true;
        m->readyToCompact = false;
        m->compactedHandle = 0;
        m->compactedAvailable = false;

        return m->handle;
    }

    void InstanceAccelerationStructure::prepareForCompact(size_t* compactedAccelBufferSize) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        m->throwRuntimeError(compactionEnabled, "This AS does not allow compaction.");
        m->throwRuntimeError(m->available, "Uncompacted AS has not been built yet.");

        if (m->compactedAvailable)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        // EN: Wait the completion of rebuild/update then obtain the size after coompaction.
        // TODO: ? stream
        CUDADRV_CHECK(cuEventSynchronize(m->finishEvent));
        CUDADRV_CHECK(cuMemcpyDtoH(
            &m->compactedSize, m->propertyCompactedSize.result, sizeof(m->compactedSize)));

        *compactedAccelBufferSize = m->compactedSize;

        m->readyToCompact = true;
    }

    OptixTraversableHandle InstanceAccelerationStructure::compact(
        CUstream stream, const BufferView &compactedAccelBuffer) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        m->throwRuntimeError(
            compactionEnabled,
            "This AS does not allow compaction.");
        m->throwRuntimeError(
            m->readyToCompact,
            "You need to call prepareForCompact() before compaction.");
        m->throwRuntimeError(
            m->available,
            "Uncompacted AS has not been built yet.");
        m->throwRuntimeError(
            compactedAccelBuffer.sizeInBytes() >= m->compactedSize,
            "Size of the given buffer is not enough.");

        OPTIX_CHECK(optixAccelCompact(
            m->getRawContext(), stream,
            m->handle, compactedAccelBuffer.getCUdeviceptr(), compactedAccelBuffer.sizeInBytes(),
            &m->compactedHandle));
        CUDADRV_CHECK(cuEventRecord(m->finishEvent, stream));

        m->compactedAccelBuffer = compactedAccelBuffer;
        m->compactedAvailable = true;

        return m->compactedHandle;
    }

    void InstanceAccelerationStructure::removeUncompacted() const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m->compactedAvailable || !compactionEnabled)
            return;

        CUDADRV_CHECK(cuEventSynchronize(m->finishEvent));

        m->handle = 0;
        m->available = false;
    }

    void InstanceAccelerationStructure::update(CUstream stream, const BufferView &scratchBuffer) const {
        bool updateEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;
        m->throwRuntimeError(
            updateEnabled,
            "This AS does not allow update.");
        m->throwRuntimeError(
            m->available || m->compactedAvailable,
            "AS has not been built yet.");
        m->throwRuntimeError(
            scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempUpdateSizeInBytes,
            "Size of the given scratch buffer is not enough.");

        uint32_t childIdx = 0;
        for (const _Instance* child : m->children)
            child->updateInstance(&m->instances[childIdx++]);
        CUDADRV_CHECK(cuMemcpyHtoDAsync(
            m->instanceBuffer.getCUdeviceptr(), m->instances.data(),
            m->instances.size() * sizeof(OptixInstance),
            stream));

        const BufferView &accelBuffer = m->compactedAvailable ? m->compactedAccelBuffer : m->accelBuffer;
        OptixTraversableHandle handle = m->compactedAvailable ? m->compactedHandle : m->handle;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OptixTraversableHandle tempHandle = handle;
        OPTIX_CHECK(optixAccelBuild(
            m->getRawContext(), stream,
            &m->buildOptions, &m->buildInput, 1,
            scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
            accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
            &tempHandle,
            nullptr, 0));
        optixuAssert(
            tempHandle == handle,
            "IAS %s: Update should not change the handle itself, what's going on?",
            getName());
    }

    bool InstanceAccelerationStructure::isReady() const {
        return m->isReady();
    }

    OptixTraversableHandle InstanceAccelerationStructure::getHandle() const {
        return m->getHandle();
    }

    void InstanceAccelerationStructure::getConfiguration(
        ASTradeoff* tradeOff,
        AllowUpdate* allowUpdate,
        AllowCompaction* allowCompaction,
        AllowRandomInstanceAccess* allowRandomInstanceAccess) const {
        if (tradeOff)
            *tradeOff = m->tradeoff;
        if (allowUpdate)
            *allowUpdate = AllowUpdate(m->allowUpdate);
        if (allowCompaction)
            *allowCompaction = AllowCompaction(m->allowCompaction);
        if (allowRandomInstanceAccess)
            *allowRandomInstanceAccess = AllowRandomInstanceAccess(m->allowRandomInstanceAccess);
    }

    void InstanceAccelerationStructure::getMotionOptions(
        uint32_t* numKeys, float* timeBegin, float* timeEnd, OptixMotionFlags* flags) const {
        if (numKeys)
            *numKeys = m->buildOptions.motionOptions.numKeys;
        if (timeBegin)
            *timeBegin = m->buildOptions.motionOptions.timeBegin;
        if (timeEnd)
            *timeEnd = m->buildOptions.motionOptions.timeEnd;
        if (flags)
            *flags = static_cast<OptixMotionFlags>(m->buildOptions.motionOptions.flags);
    }

    uint32_t InstanceAccelerationStructure::getNumChildren() const {
        return static_cast<uint32_t>(m->children.size());
    }

    uint32_t InstanceAccelerationStructure::findChildIndex(Instance instance) const {
        _Instance* _inst = extract(instance);
        m->throwRuntimeError(
            _inst,
            "Invalid instance %p.",
            _inst);
        m->throwRuntimeError(
            _inst->getScene() == m->scene,
            "Scene mismatch for the given instance %s.",
            _inst->getName().c_str());
        auto idx = std::find(m->children.cbegin(), m->children.cend(), _inst);
        if (idx == m->children.cend())
            return 0xFFFFFFFF;

        return static_cast<uint32_t>(std::distance(m->children.cbegin(), idx));
    }

    Instance InstanceAccelerationStructure::getChild(uint32_t index) const {
        uint32_t numChildren = static_cast<uint32_t>(m->children.size());
        m->throwRuntimeError(
            index < numChildren,
            "Index is out of bounds [0, %u).]",
            numChildren);
        return m->children[index]->getPublicType();
    }



    Pipeline::Priv::~Priv() {
        if (pipelineLinked)
            optixPipelineDestroy(rawPipeline);
        for (auto it = modulesForBuiltinIS.begin(); it != modulesForBuiltinIS.end(); ++it)
            it->second->getPublicType().destroy();
        modulesForBuiltinIS.clear();
        context->unregisterName(this);
    }

    void Pipeline::Priv::markDirty() {
        if (pipelineLinked)
            OPTIX_CHECK(optixPipelineDestroy(rawPipeline));
        pipelineLinked = false;
    }

    Module Pipeline::Priv::createModule(
        const char* data, size_t size,
        int32_t maxRegisterCount,
        OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel,
        OptixModuleCompileBoundValueEntry* boundValues, uint32_t numBoundValues,
        const PayloadType* payloadTypes, uint32_t numPayloadTypes) {
        std::vector<OptixPayloadType> optixPayloadTypes(numPayloadTypes);
        for (uint32_t i = 0; i < numPayloadTypes; ++i)
            optixPayloadTypes[i] = payloadTypes[i].getRawType();

        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount = maxRegisterCount;
        moduleCompileOptions.optLevel = optLevel;
        moduleCompileOptions.debugLevel = debugLevel;
        moduleCompileOptions.boundValues = boundValues;
        moduleCompileOptions.numBoundValues = numBoundValues;
        moduleCompileOptions.payloadTypes = numPayloadTypes ? optixPayloadTypes.data() : nullptr;
        moduleCompileOptions.numPayloadTypes = numPayloadTypes;

        OptixModule rawModule;

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixModuleCreate(
            getRawContext(),
            &moduleCompileOptions,
            &pipelineCompileOptions,
            data, size,
            log, &logSize,
            &rawModule));

        return (new _Module(this, rawModule))->getPublicType();
    }

    OptixModule Pipeline::Priv::getModuleForBuiltin(
        OptixPrimitiveType primType, OptixCurveEndcapFlags endcapFlags,
        ASTradeoff tradeoff, bool allowUpdate, bool allowCompaction, bool allowRandomVertexAccess) {
        if (primType != OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR &&
            primType != OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE &&
            primType != OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE &&
            primType != OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE &&
            primType != OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM &&
            primType != OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER &&
            primType != OPTIX_PRIMITIVE_TYPE_SPHERE)
            return nullptr;

        KeyForBuiltinISModule key{ primType, endcapFlags, OPTIX_BUILD_FLAG_NONE };
        if (modulesForBuiltinIS.count(key) == 0) {
            OptixModuleCompileOptions moduleCompileOptions = {};
            moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

            OptixBuiltinISOptions builtinISOptions = {};
            builtinISOptions.builtinISModuleType = primType;
            builtinISOptions.curveEndcapFlags = endcapFlags;
            builtinISOptions.buildFlags = 0;
            if (tradeoff == ASTradeoff::PreferFastTrace)
                builtinISOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            else if (tradeoff == ASTradeoff::PreferFastBuild)
                builtinISOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
            builtinISOptions.buildFlags |=
                (allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0)
                | (allowCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0)
                | (allowRandomVertexAccess ? OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS : 0);
            builtinISOptions.usesMotionBlur = pipelineCompileOptions.usesMotionBlur;

            OptixModule rawModule;
            OPTIX_CHECK(optixBuiltinISModuleGet(
                context->getRawContext(),
                &moduleCompileOptions,
                &pipelineCompileOptions,
                &builtinISOptions,
                &rawModule));

            modulesForBuiltinIS[key] = new _Module(this, rawModule);
        }

        return modulesForBuiltinIS.at(key)->getRawModule();
    }

    void Pipeline::Priv::createProgram(
        const OptixProgramGroupDesc &desc, const OptixProgramGroupOptions &options,
        OptixProgramGroup* group) {
        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            getRawContext(),
            &desc, 1, // num program groups
            &options,
            log, &logSize,
            group));
        programGroups.insert(*group);

        markDirty();
    }

    void Pipeline::Priv::destroyProgram(OptixProgramGroup group) {
        optixuAssert(programGroups.count(group) > 0, "This program group has not been registered.");
        programGroups.erase(group);
        OPTIX_CHECK(optixProgramGroupDestroy(group));

        markDirty();
    }

    void Pipeline::Priv::setupShaderBindingTable(CUstream stream) {
        if (!sbtIsUpToDate) {
            throwRuntimeError(rayGenProgram, "Ray generation program is not set.");
            for (uint32_t i = 0; i < numMissRayTypes; ++i)
                throwRuntimeError(missPrograms[i], "Miss program is not set for ray type %d.", i);
            for (uint32_t i = 0; i < numCallablePrograms; ++i)
                throwRuntimeError(callablePrograms[i], "Callable program is not set for index %d.", i);

            auto records = reinterpret_cast<uint8_t*>(sbtHostMem);
            size_t offset = 0;

            size_t rayGenRecordOffset = offset;
            rayGenProgram->packHeader(records + offset);
            offset += OPTIX_SBT_RECORD_HEADER_SIZE;

            size_t exceptionRecordOffset = offset;
            if (exceptionProgram)
                exceptionProgram->packHeader(records + offset);
            offset += OPTIX_SBT_RECORD_HEADER_SIZE;

            CUdeviceptr missRecordOffset = offset;
            for (uint32_t i = 0; i < numMissRayTypes; ++i) {
                missPrograms[i]->packHeader(records + offset);
                offset += OPTIX_SBT_RECORD_HEADER_SIZE;
            }

            CUdeviceptr callableRecordOffset = offset;
            for (uint32_t i = 0; i < numCallablePrograms; ++i) {
                callablePrograms[i]->packHeader(records + offset);
                offset += OPTIX_SBT_RECORD_HEADER_SIZE;
            }

            CUDADRV_CHECK(cuMemcpyHtoDAsync(sbt.getCUdeviceptr(), sbtHostMem, sbt.sizeInBytes(), stream));

            CUdeviceptr baseAddress = sbt.getCUdeviceptr();
            sbtParams.raygenRecord = baseAddress + rayGenRecordOffset;
            sbtParams.exceptionRecord = exceptionProgram ? baseAddress + exceptionRecordOffset : 0;
            sbtParams.missRecordBase = baseAddress + missRecordOffset;
            sbtParams.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
            sbtParams.missRecordCount = numMissRayTypes;
            sbtParams.callablesRecordBase = numCallablePrograms ? baseAddress + callableRecordOffset : 0;
            sbtParams.callablesRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
            sbtParams.callablesRecordCount = numCallablePrograms;

            sbtIsUpToDate = true;
        }

        if (!hitGroupSbtIsUpToDate) {
            scene->setupHitGroupSBT(stream, this, hitGroupSbt, hitGroupSbtHostMem);

            sbtParams.hitgroupRecordBase = hitGroupSbt.getCUdeviceptr();
            sbtParams.hitgroupRecordStrideInBytes = scene->getSingleRecordSize();
            sbtParams.hitgroupRecordCount =
                static_cast<uint32_t>(hitGroupSbt.sizeInBytes() / scene->getSingleRecordSize());

            hitGroupSbtIsUpToDate = true;
        }
    }

    void Pipeline::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    void Pipeline::setPipelineOptions(
        uint32_t numPayloadValuesInDwords, uint32_t numAttributeValuesInDwords,
        const char* launchParamsVariableName, size_t sizeOfLaunchParams,
        OptixTraversableGraphFlags traversableGraphFlags,
        OptixExceptionFlags exceptionFlags,
        OptixPrimitiveTypeFlags supportedPrimitiveTypeFlags,
        UseMotionBlur useMotionBlur, UseOpacityMicroMaps useOpacityMicroMaps) const {
        m->throwRuntimeError(
            !m->pipelineLinked,
            "Changing pipeline options after linking is not supported yet.");

        // JP: パイプライン中のモジュール、そしてパイプライン自体に共通なコンパイルオプションの設定。
        // EN: Set pipeline compile options common among modules in the pipeline and the pipeline itself.
        m->pipelineCompileOptions = {};
        m->pipelineCompileOptions.numPayloadValues = numPayloadValuesInDwords;
        m->pipelineCompileOptions.numAttributeValues = numAttributeValuesInDwords;
        m->pipelineCompileOptions.pipelineLaunchParamsVariableName = launchParamsVariableName;
        m->pipelineCompileOptions.usesMotionBlur = useMotionBlur;
        m->pipelineCompileOptions.allowOpacityMicromaps = useOpacityMicroMaps;
        m->pipelineCompileOptions.traversableGraphFlags = traversableGraphFlags;
        m->pipelineCompileOptions.exceptionFlags = exceptionFlags;
        m->pipelineCompileOptions.usesPrimitiveTypeFlags = supportedPrimitiveTypeFlags;

        m->sizeOfPipelineLaunchParams = sizeOfLaunchParams;
    }

    Module Pipeline::createModuleFromPTXString(
        const std::string &ptxString, int32_t maxRegisterCount,
        OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel,
        OptixModuleCompileBoundValueEntry* boundValues, uint32_t numBoundValues,
        const PayloadType* payloadTypes, uint32_t numPayloadTypes) const {
        return m->createModule(
            ptxString.c_str(), ptxString.size(),
            maxRegisterCount,
            optLevel, debugLevel,
            boundValues, numBoundValues,
            payloadTypes, numPayloadTypes);
    }

    Module Pipeline::createModuleFromOptixIR(
        const std::vector<char> &irBin, int32_t maxRegisterCount,
        OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel,
        OptixModuleCompileBoundValueEntry* boundValues, uint32_t numBoundValues,
        const PayloadType* payloadTypes, uint32_t numPayloadTypes) const {
        return m->createModule(
            irBin.data(), irBin.size(),
            maxRegisterCount,
            optLevel, debugLevel,
            boundValues, numBoundValues,
            payloadTypes, numPayloadTypes);
    }

    Program Pipeline::createRayGenProgram(Module module, const char* entryFunctionName) const {
        _Module* _module = extract(module);
        m->throwRuntimeError(
            _module && entryFunctionName,
            "Either of RayGen module or entry function name is not provided.");
        m->throwRuntimeError(
            _module->getPipeline() == m,
            "Pipeline mismatch for the given module %s.",
            _module->getName().c_str());

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = _module->getRawModule();
        desc.raygen.entryFunctionName = entryFunctionName;

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _Program(m, group, desc.kind))->getPublicType();
    }

    Program Pipeline::createExceptionProgram(Module module, const char* entryFunctionName) const {
        _Module* _module = extract(module);
        m->throwRuntimeError(
            _module && entryFunctionName,
            "Either of Exception module or entry function name is not provided.");
        m->throwRuntimeError(
            _module->getPipeline() == m,
            "Pipeline mismatch for the given module %s.",
            _module->getName().c_str());

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        desc.exception.module = _module->getRawModule();
        desc.exception.entryFunctionName = entryFunctionName;

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _Program(m, group, desc.kind))->getPublicType();
    }

    Program Pipeline::createMissProgram(
        Module module, const char* entryFunctionName,
        const PayloadType &payloadType) const {
        _Module* _module = extract(module);
        m->throwRuntimeError(
            (_module != nullptr) == (entryFunctionName != nullptr),
            "Either of Miss module or entry function name is not provided.");
        if (_module)
            m->throwRuntimeError(
                _module->getPipeline() == m,
                "Pipeline mismatch for the given module %s.",
                _module->getName().c_str());

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        if (_module)
            desc.miss.module = _module->getRawModule();
        desc.miss.entryFunctionName = entryFunctionName;

        OptixProgramGroupOptions options = {};
        OptixPayloadType optixPayloadType = payloadType.getRawType();
        if (payloadType.numDwords > 0)
            options.payloadType = &optixPayloadType;

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _Program(m, group, desc.kind))->getPublicType();
    }

    HitProgramGroup Pipeline::createHitProgramGroupForTriangleIS(
        Module module_CH, const char* entryFunctionNameCH,
        Module module_AH, const char* entryFunctionNameAH,
        const PayloadType &payloadType) const {
        _Module* _module_CH = extract(module_CH);
        _Module* _module_AH = extract(module_AH);
        m->throwRuntimeError(
            (_module_CH != nullptr) == (entryFunctionNameCH != nullptr),
            "Either of CH module or entry function name is not provided.");
        m->throwRuntimeError(
            (_module_AH != nullptr) == (entryFunctionNameAH != nullptr),
            "Either of AH module or entry function name is not provided.");
        m->throwRuntimeError(
            entryFunctionNameCH || entryFunctionNameAH,
            "Either of CH/AH entry function name must be provided.");
        if (_module_CH)
            m->throwRuntimeError(
                _module_CH->getPipeline() == m,
                "Pipeline mismatch for the given CH module %s.",
                _module_CH->getName().c_str());
        if (_module_AH)
            m->throwRuntimeError(
                _module_AH->getPipeline() == m,
                "Pipeline mismatch for the given AH module %s.",
                _module_AH->getName().c_str());

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (entryFunctionNameCH && _module_CH) {
            desc.hitgroup.moduleCH = _module_CH->getRawModule();
            desc.hitgroup.entryFunctionNameCH = entryFunctionNameCH;
        }
        if (entryFunctionNameAH && _module_AH) {
            desc.hitgroup.moduleAH = _module_AH->getRawModule();
            desc.hitgroup.entryFunctionNameAH = entryFunctionNameAH;
        }

        OptixProgramGroupOptions options = {};
        OptixPayloadType optixPayloadType = payloadType.getRawType();
        if (payloadType.numDwords > 0)
            options.payloadType = &optixPayloadType;

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _HitProgramGroup(m, group))->getPublicType();
    }

    HitProgramGroup Pipeline::createHitProgramGroupForCurveIS(
        OptixPrimitiveType curveType, OptixCurveEndcapFlags endcapFlags,
        Module module_CH, const char* entryFunctionNameCH,
        Module module_AH, const char* entryFunctionNameAH,
        ASTradeoff tradeoff, AllowUpdate allowUpdate, AllowCompaction allowCompaction,
        AllowRandomVertexAccess allowRandomVertexAccess,
        const PayloadType &payloadType) const {
        m->throwRuntimeError(
            curveType != OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR ||
            curveType != OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE ||
            curveType != OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE ||
            curveType != OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE ||
            curveType != OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM ||
            curveType != OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER,
            "This is a hit program group for curves.");
        _Module* _module_CH = extract(module_CH);
        _Module* _module_AH = extract(module_AH);
        m->throwRuntimeError(
            (_module_CH != nullptr) == (entryFunctionNameCH != nullptr),
            "Either of CH module or entry function name is not provided.");
        m->throwRuntimeError(
            (_module_AH != nullptr) == (entryFunctionNameAH != nullptr),
            "Either of AH module or entry function name is not provided.");
        m->throwRuntimeError(
            entryFunctionNameCH || entryFunctionNameAH,
            "Either of CH/AH entry function name must be provided.");
        if (_module_CH)
            m->throwRuntimeError(
                _module_CH->getPipeline() == m,
                "Pipeline mismatch for the given CH module %s.",
                _module_CH->getName().c_str());
        if (_module_AH)
            m->throwRuntimeError(
                _module_AH->getPipeline() == m,
                "Pipeline mismatch for the given AH module %s.",
                _module_AH->getName().c_str());

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (entryFunctionNameCH && _module_CH) {
            desc.hitgroup.moduleCH = _module_CH->getRawModule();
            desc.hitgroup.entryFunctionNameCH = entryFunctionNameCH;
        }
        if (entryFunctionNameAH && _module_AH) {
            desc.hitgroup.moduleAH = _module_AH->getRawModule();
            desc.hitgroup.entryFunctionNameAH = entryFunctionNameAH;
        }
        desc.hitgroup.moduleIS = m->getModuleForBuiltin(
            curveType, endcapFlags,
            tradeoff, allowUpdate, allowCompaction, allowRandomVertexAccess);
        desc.hitgroup.entryFunctionNameIS = nullptr;

        OptixProgramGroupOptions options = {};
        OptixPayloadType optixPayloadType = payloadType.getRawType();
        if (payloadType.numDwords > 0)
            options.payloadType = &optixPayloadType;

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _HitProgramGroup(m, group))->getPublicType();
    }

    HitProgramGroup Pipeline::createHitProgramGroupForSphereIS(
        Module module_CH, const char* entryFunctionNameCH,
        Module module_AH, const char* entryFunctionNameAH,
        ASTradeoff tradeoff, AllowUpdate allowUpdate, AllowCompaction allowCompaction,
        AllowRandomVertexAccess allowRandomVertexAccess,
        const PayloadType &payloadType) const {
        _Module* _module_CH = extract(module_CH);
        _Module* _module_AH = extract(module_AH);
        m->throwRuntimeError(
            (_module_CH != nullptr) == (entryFunctionNameCH != nullptr),
            "Either of CH module or entry function name is not provided.");
        m->throwRuntimeError(
            (_module_AH != nullptr) == (entryFunctionNameAH != nullptr),
            "Either of AH module or entry function name is not provided.");
        m->throwRuntimeError(
            entryFunctionNameCH || entryFunctionNameAH,
            "Either of CH/AH entry function name must be provided.");
        if (_module_CH)
            m->throwRuntimeError(
                _module_CH->getPipeline() == m,
                "Pipeline mismatch for the given CH module %s.",
                _module_CH->getName().c_str());
        if (_module_AH)
            m->throwRuntimeError(
                _module_AH->getPipeline() == m,
                "Pipeline mismatch for the given AH module %s.",
                _module_AH->getName().c_str());

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (entryFunctionNameCH && _module_CH) {
            desc.hitgroup.moduleCH = _module_CH->getRawModule();
            desc.hitgroup.entryFunctionNameCH = entryFunctionNameCH;
        }
        if (entryFunctionNameAH && _module_AH) {
            desc.hitgroup.moduleAH = _module_AH->getRawModule();
            desc.hitgroup.entryFunctionNameAH = entryFunctionNameAH;
        }
        desc.hitgroup.moduleIS = m->getModuleForBuiltin(
            OPTIX_PRIMITIVE_TYPE_SPHERE, OPTIX_CURVE_ENDCAP_DEFAULT,
            tradeoff, allowUpdate, allowCompaction, allowRandomVertexAccess);
        desc.hitgroup.entryFunctionNameIS = nullptr;

        OptixProgramGroupOptions options = {};
        OptixPayloadType optixPayloadType = payloadType.getRawType();
        if (payloadType.numDwords > 0)
            options.payloadType = &optixPayloadType;

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _HitProgramGroup(m, group))->getPublicType();
    }

    HitProgramGroup Pipeline::createHitProgramGroupForCustomIS(
        Module module_CH, const char* entryFunctionNameCH,
        Module module_AH, const char* entryFunctionNameAH,
        Module module_IS, const char* entryFunctionNameIS,
        const PayloadType &payloadType) const {
        _Module* _module_CH = extract(module_CH);
        _Module* _module_AH = extract(module_AH);
        _Module* _module_IS = extract(module_IS);
        m->throwRuntimeError(
            (_module_CH != nullptr) == (entryFunctionNameCH != nullptr),
            "Either of CH module or entry function name is not provided.");
        m->throwRuntimeError(
            (_module_AH != nullptr) == (entryFunctionNameAH != nullptr),
            "Either of AH module or entry function name is not provided.");
        m->throwRuntimeError(
            _module_IS != nullptr && entryFunctionNameIS != nullptr,
            "Intersection program must be provided for custom primitives.");
        m->throwRuntimeError(
            entryFunctionNameCH || entryFunctionNameAH,
            "Either of CH/AH entry function name must be provided.");
        if (_module_CH)
            m->throwRuntimeError(
                _module_CH->getPipeline() == m,
                "Pipeline mismatch for the given CH module %s.",
                _module_CH->getName().c_str());
        if (_module_AH)
            m->throwRuntimeError(
                _module_AH->getPipeline() == m,
                "Pipeline mismatch for the given AH module %s.",
                _module_AH->getName().c_str());
        m->throwRuntimeError(
            _module_IS->getPipeline() == m,
            "Pipeline mismatch for the given IS module %s.",
            _module_IS->getName().c_str());

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (entryFunctionNameCH && _module_CH) {
            desc.hitgroup.moduleCH = _module_CH->getRawModule();
            desc.hitgroup.entryFunctionNameCH = entryFunctionNameCH;
        }
        if (entryFunctionNameAH && _module_AH) {
            desc.hitgroup.moduleAH = _module_AH->getRawModule();
            desc.hitgroup.entryFunctionNameAH = entryFunctionNameAH;
        }
        desc.hitgroup.moduleIS = _module_IS->getRawModule();
        desc.hitgroup.entryFunctionNameIS = entryFunctionNameIS;

        OptixProgramGroupOptions options = {};
        OptixPayloadType optixPayloadType = payloadType.getRawType();
        if (payloadType.numDwords > 0)
            options.payloadType = &optixPayloadType;

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _HitProgramGroup(m, group))->getPublicType();
    }

    HitProgramGroup Pipeline::createEmptyHitProgramGroup() const {
        OptixProgramGroupDesc desc = {};

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _HitProgramGroup(m, group))->getPublicType();
    }

    CallableProgramGroup Pipeline::createCallableProgramGroup(
        Module module_DC, const char* entryFunctionNameDC,
        Module module_CC, const char* entryFunctionNameCC,
        const PayloadType &payloadType) const {
        _Module* _module_DC = extract(module_DC);
        _Module* _module_CC = extract(module_CC);
        m->throwRuntimeError(
            (_module_DC != nullptr) == (entryFunctionNameDC != nullptr),
            "Either of DC module or entry function name is not provided.");
        m->throwRuntimeError(
            (_module_CC != nullptr) == (entryFunctionNameCC != nullptr),
            "Either of CC module or entry function name is not provided.");
        m->throwRuntimeError(
            entryFunctionNameDC || entryFunctionNameCC,
            "Either of CC/DC entry function name must be provided.");
        if (_module_DC)
            m->throwRuntimeError(
                _module_DC->getPipeline() == m,
                "Pipeline mismatch for the given DC module %s.",
                _module_DC->getName().c_str());
        if (_module_CC)
            m->throwRuntimeError(
                _module_CC->getPipeline() == m,
                "Pipeline mismatch for the given CC module %s.",
                _module_CC->getName().c_str());

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        if (entryFunctionNameDC && _module_DC) {
            desc.callables.moduleDC = _module_DC->getRawModule();
            desc.callables.entryFunctionNameDC = entryFunctionNameDC;
        }
        if (entryFunctionNameCC && _module_CC) {
            desc.callables.moduleCC = _module_CC->getRawModule();
            desc.callables.entryFunctionNameCC = entryFunctionNameCC;
        }

        OptixProgramGroupOptions options = {};
        OptixPayloadType optixPayloadType = payloadType.getRawType();
        if (payloadType.numDwords > 0)
            options.payloadType = &optixPayloadType;

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _CallableProgramGroup(m, group))->getPublicType();
    }

    void Pipeline::link(uint32_t maxTraceDepth) const {
        m->throwRuntimeError(!m->pipelineLinked, "This pipeline has been already linked.");

        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = maxTraceDepth;

        std::vector<OptixProgramGroup> groups;
        groups.resize(m->programGroups.size());
        std::copy(m->programGroups.cbegin(), m->programGroups.cend(), groups.begin());

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(
            m->getRawContext(),
            &m->pipelineCompileOptions,
            &pipelineLinkOptions,
            groups.data(), static_cast<uint32_t>(groups.size()),
            log, &logSize,
            &m->rawPipeline));

        m->pipelineLinked = true;
    }

    void Pipeline::setNumMissRayTypes(uint32_t numMissRayTypes) const {
        m->numMissRayTypes = numMissRayTypes;
        m->missPrograms.resize(m->numMissRayTypes);
        m->sbtLayoutIsUpToDate = false;
    }

    void Pipeline::setNumCallablePrograms(uint32_t numCallablePrograms) const {
        m->numCallablePrograms = numCallablePrograms;
        m->callablePrograms.resize(m->numCallablePrograms);
        m->sbtLayoutIsUpToDate = false;
    }

    void Pipeline::generateShaderBindingTableLayout(size_t* memorySize) const {
        if (m->sbtLayoutIsUpToDate) {
            *memorySize = m->sbtSize;
            return;
        }

        m->sbtSize = 0;
        m->sbtSize += OPTIX_SBT_RECORD_HEADER_SIZE; // RayGen
        m->sbtSize += OPTIX_SBT_RECORD_HEADER_SIZE; // Exception
        m->sbtSize += OPTIX_SBT_RECORD_HEADER_SIZE * m->numMissRayTypes; // Miss
        m->sbtSize += OPTIX_SBT_RECORD_HEADER_SIZE * m->numCallablePrograms; // Callable
        m->sbtLayoutIsUpToDate = true;

        *memorySize = m->sbtSize;
    }

    void Pipeline::setRayGenerationProgram(Program program) const {
        _Program* _program = extract(program);
        m->throwRuntimeError(
            _program, "Invalid program %p.", _program);
        m->throwRuntimeError(
            _program->getPipeline() == m,
            "Pipeline mismatch for the given program %s.",
            _program->getName().c_str());

        m->rayGenProgram = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setExceptionProgram(Program program) const {
        _Program* _program = extract(program);
        m->throwRuntimeError(
            _program,
            "Invalid program %p.",
            _program);
        m->throwRuntimeError(
            _program->getPipeline() == m,
            "Pipeline mismatch for the given program %s.",
            _program->getName().c_str());

        m->exceptionProgram = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setMissProgram(uint32_t rayType, Program program) const {
        _Program* _program = extract(program);
        m->throwRuntimeError(
            rayType < m->numMissRayTypes,
            "Invalid ray type.");
        m->throwRuntimeError(
            _program,
            "Invalid program %p.",
            _program);
        m->throwRuntimeError(
            _program->getPipeline() == m,
            "Pipeline mismatch for the given program %s.",
            _program->getName().c_str());

        m->missPrograms[rayType] = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setCallableProgram(uint32_t index, CallableProgramGroup program) const {
        _CallableProgramGroup* _program = extract(program);
        m->throwRuntimeError(
            index < m->numCallablePrograms,
            "Invalid callable program index.");
        m->throwRuntimeError(
            _program,
            "Invalid program %p.", _program);
        m->throwRuntimeError(
            _program->getPipeline() == m,
            "Pipeline mismatch for the given program group %s.",
            _program->getName().c_str());

        m->callablePrograms[index] = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setShaderBindingTable(const BufferView &shaderBindingTable, void* hostMem) const {
        m->throwRuntimeError(
            shaderBindingTable.sizeInBytes() >= m->sbtSize,
            "Hit group shader binding table size is not enough.");
        m->throwRuntimeError(
            hostMem,
            "Host-side SBT counterpart must be provided.");
        m->sbt = shaderBindingTable;
        m->sbtHostMem = hostMem;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setScene(const Scene &scene) const {
        m->scene = extract(scene);
        m->hitGroupSbt = BufferView();
        m->hitGroupSbtIsUpToDate = false;
    }

    void Pipeline::setHitGroupShaderBindingTable(
        const BufferView &shaderBindingTable, void* hostMem) const {
        m->throwRuntimeError(
            hostMem,
            "Host-side hit group SBT counterpart must be provided.");
        m->hitGroupSbt = shaderBindingTable;
        m->hitGroupSbtHostMem = hostMem;
        m->hitGroupSbtIsUpToDate = false;
    }

    void Pipeline::markHitGroupShaderBindingTableDirty() const {
        m->hitGroupSbtIsUpToDate = false;
    }

    void Pipeline::setStackSize(
        uint32_t directCallableStackSizeFromTraversal,
        uint32_t directCallableStackSizeFromState,
        uint32_t continuationStackSize,
        uint32_t maxTraversableGraphDepth) const {
        if (m->pipelineCompileOptions.traversableGraphFlags &
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING)
            maxTraversableGraphDepth = 2;
        else if (m->pipelineCompileOptions.traversableGraphFlags &
                 OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS)
            maxTraversableGraphDepth = 1;
        OPTIX_CHECK(optixPipelineSetStackSize(
            m->rawPipeline,
            directCallableStackSizeFromTraversal,
            directCallableStackSizeFromState,
            continuationStackSize,
            maxTraversableGraphDepth));
    }

    void Pipeline::launch(
        CUstream stream, CUdeviceptr plpOnDevice,
        uint32_t dimX, uint32_t dimY, uint32_t dimZ) const {
        m->throwRuntimeError(
            m->sbtLayoutIsUpToDate,
            "Shader binding table layout is outdated.");
        m->throwRuntimeError(
            m->sbt.isValid(),
            "Shader binding table is not set.");
        m->throwRuntimeError(
            m->sbt.sizeInBytes() >= m->sbtSize,
            "Shader binding table size is not enough.");
        m->throwRuntimeError(
            m->scene,
            "Scene is not set.");
        bool hasMotionAS;
        m->throwRuntimeError(
            m->scene->isReady(&hasMotionAS),
            "Scene is not ready.");
        m->throwRuntimeError(
            m->pipelineCompileOptions.usesMotionBlur || !hasMotionAS,
            "Scene has a motion AS but the pipeline has not been configured for motion.");
        m->throwRuntimeError(
            m->hitGroupSbt.isValid(),
            "Hitgroup shader binding table is not set.");
        m->throwRuntimeError(
            m->pipelineLinked,
            "Pipeline has not been linked yet.");

        m->setupShaderBindingTable(stream);

        OPTIX_CHECK(optixLaunch(
            m->rawPipeline, stream, plpOnDevice, m->sizeOfPipelineLaunchParams,
            &m->sbtParams, dimX, dimY, dimZ));
    }

    Scene Pipeline::getScene() const {
        if (m->scene)
            return m->scene->getPublicType();
        else
            return Scene();
    }



    void Module::destroy() {
        if (m) {
            OPTIX_CHECK(optixModuleDestroy(m->rawModule));
            delete m;
        }
        m = nullptr;
    }



    void Program::destroy() {
        if (m) {
            m->pipeline->destroyProgram(m->rawGroup);
            delete m;
        }
        m = nullptr;
    }

    uint32_t Program::getStackSize() const {
        return m->stackSize;
    }



    void HitProgramGroup::destroy() {
        if (m) {
            m->pipeline->destroyProgram(m->rawGroup);
            delete m;
        }
        m = nullptr;
    }

    uint32_t HitProgramGroup::getCHStackSize() const {
        return m->stackSizeCH;
    }

    uint32_t HitProgramGroup::getAHStackSize() const {
        return m->stackSizeAH;
    }

    uint32_t HitProgramGroup::getISStackSize() const {
        return m->stackSizeIS;
    }



    void CallableProgramGroup::destroy() {
        if (m) {
            m->pipeline->destroyProgram(m->rawGroup);
            delete m;
        }
        m = nullptr;
    }

    uint32_t CallableProgramGroup::getDCStackSize() const {
        return m->stackSizeDC;
    }

    uint32_t CallableProgramGroup::getCCStackSize() const {
        return m->stackSizeCC;
    }



    void DenoisingTask::getOutputTile(
        int32_t* offsetX, int32_t* offsetY, int32_t* width, int32_t* height) const {
        _DenoisingTask _task(*this);
        *offsetX = _task.outputOffsetX;
        *offsetY = _task.outputOffsetY;
        *width = _task.outputWidth;
        *height = _task.outputHeight;
    }



    void Denoiser::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    void Denoiser::prepare(
        uint32_t imageWidth, uint32_t imageHeight, uint32_t tileWidth, uint32_t tileHeight,
        DenoiserSizes* sizes, uint32_t* numTasks) const {
        m->throwRuntimeError(tileWidth <= imageWidth && tileHeight <= imageHeight,
                             "Tile width/height must be equal to or smaller than the image size.");

        if (tileWidth == 0)
            tileWidth = imageWidth;
        if (tileHeight == 0)
            tileHeight = imageHeight;

        m->useTiling = tileWidth < imageWidth || tileHeight < imageHeight;

        m->imageWidth = imageWidth;
        m->imageHeight = imageHeight;
        m->tileWidth = tileWidth;
        m->tileHeight = tileHeight;
        OptixDenoiserSizes rawSizes;
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(m->rawDenoiser, tileWidth, tileHeight, &rawSizes));
        m->sizes.stateSize = rawSizes.stateSizeInBytes;
        m->sizes.scratchSize = m->useTiling ?
            rawSizes.withOverlapScratchSizeInBytes : rawSizes.withoutOverlapScratchSizeInBytes;
        if (m->modelKind == OPTIX_DENOISER_MODEL_KIND_HDR ||
            m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL) {
            m->sizes.normalizerSize = sizeof(float);
            m->sizes.scratchSizeForComputeNormalizer = rawSizes.computeIntensitySizeInBytes;
        }
        else if (m->modelKind == OPTIX_DENOISER_MODEL_KIND_AOV ||
                 m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV ||
                 m->modelKind == OPTIX_DENOISER_MODEL_KIND_UPSCALE2X ||
                 m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X) {
            m->sizes.normalizerSize = 3 * sizeof(float);
            m->sizes.scratchSizeForComputeNormalizer = rawSizes.computeAverageColorSizeInBytes;
        }
        else {
            m->sizes.normalizerSize = 0;
            m->sizes.scratchSizeForComputeNormalizer = 0;
        }
        m->sizes.internalGuideLayerPixelSize = rawSizes.internalGuideLayerPixelSizeInBytes;
        m->overlapWidth = rawSizes.overlapWindowSizeInPixels;
        m->inputWidth = std::min(tileWidth + 2 * m->overlapWidth, imageWidth);
        m->inputHeight = std::min(tileHeight + 2 * m->overlapWidth, imageHeight);

        *sizes = m->sizes;

        *numTasks = 0;
        for (int32_t outputOffsetY = 0; outputOffsetY < static_cast<int32_t>(imageHeight);) {
            int32_t outputHeight = tileHeight;
            if (outputOffsetY == 0)
                outputHeight += m->overlapWidth;

            for (int32_t outputOffsetX = 0; outputOffsetX < static_cast<int32_t>(imageWidth);) {
                int32_t outputWidth = tileWidth;
                if (outputOffsetX == 0)
                    outputWidth += m->overlapWidth;

                ++*numTasks;

                outputOffsetX += outputWidth;
            }

            outputOffsetY += outputHeight;
        }

        m->tasksAreReady = false;
        m->stateIsReady = false;
        m->imageSizeSet = true;
    }

    void Denoiser::getTasks(DenoisingTask* tasks) const {
        m->throwRuntimeError(m->imageSizeSet, "Call prepare() before this function.");

        uint32_t taskIdx = 0;
        for (int32_t outputOffsetY = 0; outputOffsetY < static_cast<int32_t>(m->imageHeight);) {
            int32_t outputHeight = m->tileHeight;
            if (outputOffsetY == 0)
                outputHeight += m->overlapWidth;
            if (outputOffsetY + outputHeight > static_cast<int32_t>(m->imageHeight))
                outputHeight = m->imageHeight - outputOffsetY;

            int32_t inputOffsetY = std::max(outputOffsetY - m->overlapWidth, 0);
            if (inputOffsetY + m->inputHeight > m->imageHeight)
                inputOffsetY = m->imageHeight - m->inputHeight;

            for (int32_t outputOffsetX = 0; outputOffsetX < static_cast<int32_t>(m->imageWidth);) {
                int32_t outputWidth = m->tileWidth;
                if (outputOffsetX == 0)
                    outputWidth += m->overlapWidth;
                if (outputOffsetX + outputWidth > static_cast<int32_t>(m->imageWidth))
                    outputWidth = m->imageWidth - outputOffsetX;

                int32_t inputOffsetX = std::max(outputOffsetX - m->overlapWidth, 0);
                if (inputOffsetX + m->inputWidth > m->imageWidth)
                    inputOffsetX = m->imageWidth - m->inputWidth;

                _DenoisingTask task;
                task.inputOffsetX = inputOffsetX;
                task.inputOffsetY = inputOffsetY;
                task.outputOffsetX = outputOffsetX;
                task.outputOffsetY = outputOffsetY;
                task.outputWidth = outputWidth;
                task.outputHeight = outputHeight;
                tasks[taskIdx++] = task;

                outputOffsetX += outputWidth;
            }

            outputOffsetY += outputHeight;
        }

        m->tasksAreReady = true;
    }

    void Denoiser::setupState(
        CUstream stream, const BufferView &stateBuffer, const BufferView &scratchBuffer) const {
        m->throwRuntimeError(
            m->imageSizeSet,
            "Call prepare() before this function.");
        m->throwRuntimeError(
            stateBuffer.sizeInBytes() >= m->sizes.stateSize,
            "Size of the given state buffer is not enough.");
        m->throwRuntimeError(
            scratchBuffer.sizeInBytes() >= m->sizes.scratchSize,
            "Size of the given scratch buffer is not enough.");
        OPTIX_CHECK(optixDenoiserSetup(
            m->rawDenoiser, stream,
            m->inputWidth, m->inputHeight,
            stateBuffer.getCUdeviceptr(), stateBuffer.sizeInBytes(),
            scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes()));

        m->stateBuffer = stateBuffer;
        m->scratchBuffer = scratchBuffer;
        m->stateIsReady = true;
    }

    void Denoiser::computeNormalizer(
        CUstream stream,
        const BufferView &noisyBeauty, OptixPixelFormat beautyFormat,
        const BufferView &scratchBuffer, CUdeviceptr normalizer) const {
        m->throwRuntimeError(
            m->imageSizeSet,
            "Call prepare() before this function.");
        m->throwRuntimeError(
            scratchBuffer.sizeInBytes() >= m->sizes.scratchSizeForComputeNormalizer,
            "Size of the given scratch buffer is not enough.");

        OptixImage2D colorLayer = {};
        colorLayer.data = noisyBeauty.getCUdeviceptr();
        colorLayer.width = m->imageWidth;
        colorLayer.height = m->imageHeight;
        colorLayer.format = beautyFormat;
        colorLayer.pixelStrideInBytes = getPixelSize(beautyFormat);
        colorLayer.rowStrideInBytes = colorLayer.pixelStrideInBytes * m->imageWidth;

        if (m->modelKind == OPTIX_DENOISER_MODEL_KIND_HDR ||
            m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL)
            OPTIX_CHECK(optixDenoiserComputeIntensity(
                m->rawDenoiser, stream,
                &colorLayer, normalizer,
                scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes()));
        else if (m->modelKind == OPTIX_DENOISER_MODEL_KIND_AOV ||
                 m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV ||
                 m->modelKind == OPTIX_DENOISER_MODEL_KIND_UPSCALE2X ||
                 m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X)
            OPTIX_CHECK(optixDenoiserComputeAverageColor(
                m->rawDenoiser, stream,
                &colorLayer, normalizer,
                scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes()));
    }

    void Denoiser::invoke(
        CUstream stream, const DenoisingTask &task,
        const DenoiserInputBuffers &inputBuffers, IsFirstFrame isFirstFrame,
        CUdeviceptr normalizer, float blendFactor,
        const BufferView &denoisedBeauty, const BufferView* denoisedAovs,
        const BufferView &internalGuideLayerForNextFrame) const {
        bool isTemporal =
            m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL ||
            m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV ||
            m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X;
        bool performUpscale =
            m->modelKind == OPTIX_DENOISER_MODEL_KIND_UPSCALE2X ||
            m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X;
        bool requireInternalGuideLayer =
            m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV ||
            m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X;
        m->throwRuntimeError(
            m->tasksAreReady,
            "You need to call getTasks() before invoke.");
        m->throwRuntimeError(
            m->stateIsReady,
            "You need to call setupState() before invoke.");
        m->throwRuntimeError(
            inputBuffers.noisyBeauty.isValid() && denoisedBeauty.isValid(),
            "Both of noisy/denoised beauty buffers must be provided.");
        m->throwRuntimeError(
            inputBuffers.numAovs == 0 || (inputBuffers.noisyAovs && denoisedAovs),
            "Both of noisy/denoised AOV buffers must be provided.");
        m->throwRuntimeError(
            inputBuffers.numAovs == 0 || (inputBuffers.aovTypes),
            "AOV types must be provided.");
        for (uint32_t i = 0; i < inputBuffers.numAovs; ++i) {
            m->throwRuntimeError(
                inputBuffers.noisyAovs[i].isValid() && denoisedAovs[i].isValid(),
                "Either of AOV %u input/output buffer is invalid.", i);
        }
        m->throwRuntimeError(
            normalizer != 0 || m->modelKind == OPTIX_DENOISER_MODEL_KIND_LDR,
            "Normalizer must be provided for this denoiser.");
        m->throwRuntimeError(
            !m->guideAlbedo || inputBuffers.albedo.isValid(),
            "Denoiser requires the albedo buffer.");
        m->throwRuntimeError(
            !m->guideNormal || inputBuffers.normal.isValid(),
            "Denoiser requires the normal buffer.");
        if (requireInternalGuideLayer) {
            m->throwRuntimeError(
                inputBuffers.previousInternalGuideLayer.isValid(),
                "Denoiser requires the previous internal guide layer buffer.");
            m->throwRuntimeError(
                internalGuideLayerForNextFrame.isValid(),
                "Denoiser requires a buffer to output the internal guide layer for the next frame.");
        }
        m->throwRuntimeError(
            !isTemporal || inputBuffers.flow.isValid(),
            "Temporal denoiser requires the flow buffer.");
        m->throwRuntimeError(
            (!isTemporal || m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X) ||
            inputBuffers.previousDenoisedBeauty.isValid(),
            "Denoiser requires the previous denoised beauty buffer.");

        OptixDenoiserParams params = {};
        if (m->modelKind == OPTIX_DENOISER_MODEL_KIND_HDR ||
            m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL)
            params.hdrIntensity = normalizer;
        else if (m->modelKind == OPTIX_DENOISER_MODEL_KIND_AOV ||
                 m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV ||
                 m->modelKind == OPTIX_DENOISER_MODEL_KIND_UPSCALE2X ||
                 m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X)
            params.hdrAverageColor = normalizer;
        params.blendFactor = blendFactor;
        params.temporalModeUsePreviousLayers = !isFirstFrame;

        _DenoisingTask _task(task);

        const auto setUpInputLayer = [&]
        (OptixPixelFormat format, CUdeviceptr baseAddress, OptixImage2D* layer) {
            uint32_t pixelStride = getPixelSize(format);
            *layer = {};
            layer->rowStrideInBytes = m->imageWidth * pixelStride;
            layer->pixelStrideInBytes = pixelStride;
            uintptr_t addressOffset =
                _task.inputOffsetY * layer->rowStrideInBytes +
                _task.inputOffsetX * pixelStride;
            layer->data = baseAddress + addressOffset;
            layer->width = m->inputWidth;
            layer->height = m->inputHeight;
            layer->format = format;
        };
        const auto setUpOutputLayer = [&]
        (OptixPixelFormat format, CUdeviceptr baseAddress, OptixImage2D* layer) {
            uint32_t scale = performUpscale ? 2 : 1;
            uint32_t pixelStride = format == OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER ?
                static_cast<uint32_t>(m->sizes.internalGuideLayerPixelSize) :
                getPixelSize(format);
            *layer = {};
            layer->rowStrideInBytes = scale * m->imageWidth * pixelStride;
            layer->pixelStrideInBytes = pixelStride;
            uintptr_t addressOffset =
                scale * _task.outputOffsetY * layer->rowStrideInBytes +
                scale * _task.outputOffsetX * pixelStride;
            layer->data = baseAddress + addressOffset;
            layer->width = scale * _task.outputWidth;
            layer->height = scale * _task.outputHeight;
            layer->format = format;
        };

        // TODO: 入出力画像のrowStrideを指定できるようにする。

        OptixDenoiserGuideLayer guideLayer = {};
        if (m->guideAlbedo)
            setUpInputLayer(inputBuffers.albedoFormat, inputBuffers.albedo.getCUdeviceptr(), &guideLayer.albedo);
        if (m->guideNormal)
            setUpInputLayer(inputBuffers.normalFormat, inputBuffers.normal.getCUdeviceptr(), &guideLayer.normal);
        if (isTemporal) {
            setUpInputLayer(inputBuffers.flowFormat, inputBuffers.flow.getCUdeviceptr(), &guideLayer.flow);
            if (inputBuffers.flowTrustworthiness.isValid())
                setUpInputLayer(
                    inputBuffers.flowTrustworthinessFormat,
                    inputBuffers.flowTrustworthiness.getCUdeviceptr(),
                    &guideLayer.flowTrustworthiness);
        }
        if (requireInternalGuideLayer) {
            setUpOutputLayer(
                OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER,
                inputBuffers.previousInternalGuideLayer.getCUdeviceptr(),
                &guideLayer.previousOutputInternalGuideLayer);
            setUpOutputLayer(
                OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER,
                internalGuideLayerForNextFrame.getCUdeviceptr(),
                &guideLayer.outputInternalGuideLayer);
        }

        struct LayerInfo {
            BufferView input;
            BufferView output;
            BufferView previousOuput;
            OptixPixelFormat format;
            OptixDenoiserAOVType aovType;
        };
        std::vector<LayerInfo> layerInfos(1 + inputBuffers.numAovs);
        layerInfos[0] = LayerInfo{
            inputBuffers.noisyBeauty,
            denoisedBeauty,
            inputBuffers.previousDenoisedBeauty,
            inputBuffers.beautyFormat,
            OPTIX_DENOISER_AOV_TYPE_BEAUTY };
        for (uint32_t i = 0; i < inputBuffers.numAovs; ++i) {
            layerInfos[i + 1] = LayerInfo{
                inputBuffers.noisyAovs[i],
                denoisedAovs[i],
                inputBuffers.previousDenoisedAovs[i],
                inputBuffers.aovFormats[i],
                inputBuffers.aovTypes[i] };
        }

        std::vector<OptixDenoiserLayer> denoiserLayers(1 + inputBuffers.numAovs);
        for (uint32_t layerIdx = 0; layerIdx < 1 + inputBuffers.numAovs; ++layerIdx) {
            const LayerInfo &layerInfo = layerInfos[layerIdx];
            OptixDenoiserLayer &denoiserLayer = denoiserLayers[layerIdx];
            std::memset(&denoiserLayer, 0, sizeof(denoiserLayer));

            setUpInputLayer(layerInfo.format, layerInfo.input.getCUdeviceptr(), &denoiserLayer.input);
            if (isTemporal)
                setUpOutputLayer(
                    layerInfo.format, layerInfo.previousOuput.getCUdeviceptr(),
                    &denoiserLayer.previousOutput);
            setUpOutputLayer(layerInfo.format, layerInfo.output.getCUdeviceptr(), &denoiserLayer.output);
            denoiserLayer.type = layerInfo.aovType;
        }

        int32_t offsetXInWorkingTile = _task.outputOffsetX - _task.inputOffsetX;
        int32_t offsetYInWorkingTile = _task.outputOffsetY - _task.inputOffsetY;
        OPTIX_CHECK(optixDenoiserInvoke(
            m->rawDenoiser, stream,
            &params,
            m->stateBuffer.getCUdeviceptr(), m->stateBuffer.sizeInBytes(),
            &guideLayer,
            denoiserLayers.data(), 1 + inputBuffers.numAovs,
            offsetXInWorkingTile, offsetYInWorkingTile,
            m->scratchBuffer.getCUdeviceptr(), m->scratchBuffer.sizeInBytes()));
    }
}
