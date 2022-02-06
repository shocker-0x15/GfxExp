/*

   Copyright 2021 Shin Watanabe

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



    // Define common interfaces.
#define OPTIXU_PREPROCESS_OBJECT(Type) \
    Context Type::getContext() const { \
        return m->getContext()->getPublicType(); \
    } \
    void Type::setName(const std::string &name) const { \
        m->setName(name); \
    } \
    const char* Type::getName() const { \
        return m->getRegisteredName(); \
    }
    OPTIXU_PREPROCESS_OBJECTS();
#undef OPTIXU_PREPROCESS_OBJECT



    // static
    Context Context::create(CUcontext cuContext, uint32_t logLevel, bool enableValidation) {
        return (new _Context(cuContext, logLevel, enableValidation))->getPublicType();
    }

    void Context::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    void Context::setLogCallback(OptixLogCallback callback, void* callbackData, uint32_t logLevel) const {
        m->throwRuntimeError(logLevel <= 4, "Valid range for logLevel is [0, 4].");
        if (callback)
            OPTIX_CHECK(optixDeviceContextSetLogCallback(m->rawContext, callback, callbackData, logLevel));
        else
            OPTIX_CHECK(optixDeviceContextSetLogCallback(m->rawContext, &logCallBack, nullptr, logLevel));
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

    Denoiser Context::createDenoiser(OptixDenoiserModelKind modelKind, bool guideAlbedo, bool guideNormal) const {
        return (new _Denoiser(m, modelKind, guideAlbedo, guideNormal))->getPublicType();
    }

    CUcontext Context::getCUcontext() const {
        return m->cuContext;
    }



    void Material::Priv::setRecordHeader(const _Pipeline* pipeline, uint32_t rayType, uint8_t* record, SizeAlign* curSizeAlign) const {
        Key key{ pipeline, rayType };
        throwRuntimeError(programs.count(key), "No hit group is set to the pipeline %s, ray type %u",
                          pipeline->getName().c_str(), rayType);
        const _ProgramGroup* hitGroup = programs.at(key);
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

    void Material::setHitGroup(uint32_t rayType, ProgramGroup hitGroup) const {
        const _Pipeline* _pipeline = extract(hitGroup)->getPipeline();
        m->throwRuntimeError(_pipeline, "Invalid pipeline %p.", _pipeline);

        _Material::Key key{ _pipeline, rayType };
        m->programs[key] = extract(hitGroup);
    }

    void Material::setUserData(const void* data, uint32_t size, uint32_t alignment) const {
        m->throwRuntimeError(size <= s_maxMaterialUserDataSize,
                             "Maximum user data size for Material is %u bytes.", s_maxMaterialUserDataSize);
        m->throwRuntimeError(alignment > 0 && alignment <= OPTIX_SBT_RECORD_ALIGNMENT,
                             "Valid alignment range is [1, %u].", OPTIX_SBT_RECORD_ALIGNMENT);
        m->userDataSizeAlign = SizeAlign(size, alignment);
        m->userData.resize(size);
        std::memcpy(m->userData.data(), data, size);
    }

    ProgramGroup Material::getHitGroup(Pipeline pipeline, uint32_t rayType) const {
        auto _pipeline = extract(pipeline);
        m->throwRuntimeError(_pipeline, "Invalid pipeline %p.", _pipeline);

        _Material::Key key{ _pipeline, rayType };
        m->throwRuntimeError(m->programs.count(key), "Hit group is not set for the pipeline %s, rayType %u.",
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
        throwRuntimeError(sbtOffsets.count(key), "GAS %s: material set index %u is out of bounds.",
                          gas->getName().c_str(), matSetIdx);
        return sbtOffsets.at(key);
    }

    void Scene::Priv::setupHitGroupSBT(CUstream stream, const _Pipeline* pipeline, const BufferView &sbt, void* hostMem) {
        throwRuntimeError(sbt.sizeInBytes() >= singleRecordSize * numSBTRecords,
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

    GeometryInstance Scene::createGeometryInstance(GeometryType geomType) const {
        m->throwRuntimeError(geomType == GeometryType::Triangles ||
                             geomType == GeometryType::LinearSegments ||
                             geomType == GeometryType::QuadraticBSplines ||
                             geomType == GeometryType::CubicBSplines ||
                             geomType == GeometryType::CatmullRomSplines ||
                             geomType == GeometryType::CustomPrimitives,
                             "Invalid geometry type: %u.", static_cast<uint32_t>(geomType));
        return (new _GeometryInstance(m, geomType))->getPublicType();
    }

    GeometryAccelerationStructure Scene::createGeometryAccelerationStructure(GeometryType geomType) const {
        m->throwRuntimeError(geomType == GeometryType::Triangles ||
                             geomType == GeometryType::LinearSegments ||
                             geomType == GeometryType::QuadraticBSplines ||
                             geomType == GeometryType::CubicBSplines ||
                             geomType == GeometryType::CatmullRomSplines ||
                             geomType == GeometryType::CustomPrimitives,
                             "Invalid geometry type: %u.", static_cast<uint32_t>(geomType));
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



    void GeometryInstance::Priv::fillBuildInput(OptixBuildInput* input, CUdeviceptr preTransform) const {
        *input = OptixBuildInput{};

        if (std::holds_alternative<TriangleGeometry>(geometry)) {
            auto &geom = std::get<TriangleGeometry>(geometry);
            throwRuntimeError((geom.indexFormat != OPTIX_INDICES_FORMAT_NONE) == geom.triangleBuffer.isValid(),
                              "Triangle buffer must be provided if using a index format other than None, otherwise must not be provided.");

            uint32_t vertexStride = geom.vertexBuffers[0].stride();
            uint32_t numElements = static_cast<uint32_t>(geom.vertexBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.vertexBufferArray[i] = geom.vertexBuffers[i].getCUdeviceptr();
                throwRuntimeError(geom.vertexBuffers[i].isValid(), "Vertex buffer for motion step %u is not set.", i);
                throwRuntimeError(geom.vertexBuffers[i].numElements() == numElements, "Num elements for motion step %u doesn't match that of 0.", i);
                throwRuntimeError(geom.vertexBuffers[i].stride() == vertexStride, "Vertex stride for motion step %u doesn't match that of 0.", i);
            }

            input->type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            OptixBuildInputTriangleArray &triArray = input->triangleArray;

            triArray.vertexBuffers = geom.vertexBufferArray;
            triArray.numVertices = numElements;
            triArray.vertexFormat = geom.vertexFormat;
            triArray.vertexStrideInBytes = vertexStride;

            if (geom.indexFormat != OPTIX_INDICES_FORMAT_NONE) {
                triArray.indexBuffer = geom.triangleBuffer.getCUdeviceptr();
                triArray.indexStrideInBytes = geom.triangleBuffer.stride();
                triArray.numIndexTriplets = static_cast<uint32_t>(geom.triangleBuffer.numElements());
            }
            else {
                triArray.indexBuffer = 0;
                triArray.indexStrideInBytes = 0;
                triArray.numIndexTriplets = 0;
            }
            triArray.indexFormat = geom.indexFormat;
            triArray.primitiveIndexOffset = primitiveIndexOffset;

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
            triArray.transformFormat = preTransform ? OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 : OPTIX_TRANSFORM_FORMAT_NONE;

            triArray.flags = reinterpret_cast<const uint32_t*>(buildInputFlags.data());
        }
        else if (std::holds_alternative<CurveGeometry>(geometry)) {
            auto &geom = std::get<CurveGeometry>(geometry);
            throwRuntimeError(geom.segmentIndexBuffer.isValid(), "Segment index buffer must be provided.");

            uint32_t vertexStride = geom.vertexBuffers[0].stride();
            uint32_t widthStride = geom.widthBuffers[0].stride();
            uint32_t numElements = static_cast<uint32_t>(geom.vertexBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.vertexBufferArray[i] = geom.vertexBuffers[i].getCUdeviceptr();
                geom.widthBufferArray[i] = geom.widthBuffers[i].getCUdeviceptr();
                throwRuntimeError(geom.vertexBuffers[i].isValid(), "Vertex buffer for motion step %u is not set.", i);
                throwRuntimeError(geom.vertexBuffers[i].numElements() == numElements, "Num elements of the vertex buffer for motion step %u doesn't match that of 0.", i);
                throwRuntimeError(geom.vertexBuffers[i].stride() == vertexStride, "Vertex stride for motion step %u doesn't match that of 0.", i);
                throwRuntimeError(geom.widthBuffers[i].isValid(), "Width buffer for motion step %u is not set.", i);
                throwRuntimeError(geom.widthBuffers[i].numElements() == numElements, "Num elements of the width buffer for motion step %u doesn't match that of 0.", i);
                throwRuntimeError(geom.widthBuffers[i].stride() == widthStride, "Width stride for motion step %u doesn't match that of 0.", i);
            }

            input->type = OPTIX_BUILD_INPUT_TYPE_CURVES;
            OptixBuildInputCurveArray &curveArray = input->curveArray;

            if (geomType == GeometryType::LinearSegments)
                curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
            else if (geomType == GeometryType::QuadraticBSplines)
                curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
            else if (geomType == GeometryType::CubicBSplines)
                curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
            else if (geomType == GeometryType::CatmullRomSplines)
                curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM;
            else
                optixuAssert_ShouldNotBeCalled();
            curveArray.endcapFlags = geom.endcapFlags;

            curveArray.vertexBuffers  = geom.vertexBufferArray;
            curveArray.vertexStrideInBytes = vertexStride;
            curveArray.widthBuffers = geom.widthBufferArray;
            curveArray.widthStrideInBytes = widthStride;
            curveArray.normalBuffers = 0; // Optix just reserves normal fields for future use.
            curveArray.normalStrideInBytes = 0;
            curveArray.numVertices = numElements;

            curveArray.indexBuffer = geom.segmentIndexBuffer.getCUdeviceptr();
            curveArray.indexStrideInBytes = geom.segmentIndexBuffer.stride();
            curveArray.numPrimitives = static_cast<uint32_t>(geom.segmentIndexBuffer.numElements());
            curveArray.primitiveIndexOffset = primitiveIndexOffset;

            curveArray.flag = static_cast<uint32_t>(buildInputFlags[0]);
        }
        else if (std::holds_alternative<CustomPrimitiveGeometry>(geometry)) {
            auto &geom = std::get<CustomPrimitiveGeometry>(geometry);

            uint32_t stride = geom.primitiveAabbBuffers[0].stride();
            uint32_t numElements = static_cast<uint32_t>(geom.primitiveAabbBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.primitiveAabbBufferArray[i] = geom.primitiveAabbBuffers[i].getCUdeviceptr();
                throwRuntimeError(geom.primitiveAabbBuffers[i].isValid(), "AABB buffer for motion step %u is not set.", i);
                throwRuntimeError(geom.primitiveAabbBuffers[i].numElements() == numElements, "Num elements for motion step %u doesn't match that of 0.", i);
                throwRuntimeError(geom.primitiveAabbBuffers[i].stride() == stride, "Stride for motion step %u doesn't match that of 0.", i);
            }

            input->type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            OptixBuildInputCustomPrimitiveArray &customPrimArray = input->customPrimitiveArray;

            customPrimArray.aabbBuffers = geom.primitiveAabbBufferArray;
            customPrimArray.numPrimitives = numElements;
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
            uint32_t numElements = static_cast<uint32_t>(geom.vertexBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.vertexBufferArray[i] = geom.vertexBuffers[i].getCUdeviceptr();
                throwRuntimeError(geom.vertexBuffers[i].isValid(), "Vertex buffer for motion step %u is not set.", i);
                throwRuntimeError(geom.vertexBuffers[i].numElements() == numElements, "Num elements for motion step %u doesn't match that of 0.", i);
                throwRuntimeError(geom.vertexBuffers[i].stride() == vertexStride, "Vertex stride for motion step %u doesn't match that of 0.", i);
            }

            OptixBuildInputTriangleArray &triArray = input->triangleArray;

            triArray.vertexBuffers = geom.vertexBufferArray;

            if (geom.indexFormat != OPTIX_INDICES_FORMAT_NONE)
                triArray.indexBuffer = geom.triangleBuffer.getCUdeviceptr();

            if (triArray.numSbtRecords > 1)
                triArray.sbtIndexOffsetBuffer = geom.materialIndexBuffer.getCUdeviceptr();

            triArray.preTransform = preTransform;
            triArray.transformFormat = preTransform ? OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 : OPTIX_TRANSFORM_FORMAT_NONE;
        }
        else if (std::holds_alternative<CurveGeometry>(geometry)) {
            auto &geom = std::get<CurveGeometry>(geometry);

            uint32_t vertexStride = geom.vertexBuffers[0].stride();
            uint32_t widthStride = geom.widthBuffers[0].stride();
            uint32_t numElements = static_cast<uint32_t>(geom.vertexBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.vertexBufferArray[i] = geom.vertexBuffers[i].getCUdeviceptr();
                geom.widthBufferArray[i] = geom.widthBuffers[i].getCUdeviceptr();
                throwRuntimeError(geom.vertexBuffers[i].isValid(), "Vertex buffer for motion step %u is not set.", i);
                throwRuntimeError(geom.vertexBuffers[i].numElements() == numElements, "Num elements of the vertex buffer for motion step %u doesn't match that of 0.", i);
                throwRuntimeError(geom.vertexBuffers[i].stride() == vertexStride, "Vertex stride for motion step %u doesn't match that of 0.", i);
                throwRuntimeError(geom.widthBuffers[i].isValid(), "Width buffer for motion step %u is not set.", i);
                throwRuntimeError(geom.widthBuffers[i].numElements() == numElements, "Num elements of the width buffer for motion step %u doesn't match that of 0.", i);
                throwRuntimeError(geom.widthBuffers[i].stride() == widthStride, "Width stride for motion step %u doesn't match that of 0.", i);
            }

            OptixBuildInputCurveArray &curveArray = input->curveArray;

            curveArray.vertexBuffers = geom.vertexBufferArray;
            curveArray.widthBuffers = geom.widthBufferArray;
            curveArray.normalBuffers = 0; // Optix just reserves these fields for future use.

            curveArray.indexBuffer = geom.segmentIndexBuffer.getCUdeviceptr();
        }
        else if (std::holds_alternative<CustomPrimitiveGeometry>(geometry)) {
            auto &geom = std::get<CustomPrimitiveGeometry>(geometry);

            uint32_t stride = geom.primitiveAabbBuffers[0].stride();
            uint32_t numElements = static_cast<uint32_t>(geom.primitiveAabbBuffers[0].numElements());
            for (uint32_t i = 0; i < numMotionSteps; ++i) {
                geom.primitiveAabbBufferArray[i] = geom.primitiveAabbBuffers[i].getCUdeviceptr();
                throwRuntimeError(geom.primitiveAabbBuffers[i].isValid(), "AABB buffer for motion step %u is not set.", i);
                throwRuntimeError(geom.primitiveAabbBuffers[i].numElements() == numElements, "Num elements for motion step %u doesn't match that of 0.", i);
                throwRuntimeError(geom.primitiveAabbBuffers[i].stride() == stride, "Stride for motion step %u doesn't match that of 0.", i);
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

    void GeometryInstance::Priv::calcSBTRequirements(uint32_t gasMatSetIdx,
                                                     const SizeAlign &gasUserDataSizeAlign,
                                                     const SizeAlign &gasChildUserDataSizeAlign,
                                                     SizeAlign* maxRecordSizeAlign, uint32_t* numSBTRecords) const {
        *maxRecordSizeAlign = SizeAlign();
        for (int matIdx = 0; matIdx < materials.size(); ++matIdx) {
            throwRuntimeError(materials[matIdx][0], "Default material (== material set 0) is not set for the slot %u.", matIdx);
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

    uint32_t GeometryInstance::Priv::fillSBTRecords(const _Pipeline* pipeline, uint32_t gasMatSetIdx,
                                                    const void* gasUserData, const SizeAlign &gasUserDataSizeAlign,
                                                    const void* gasChildUserData, const SizeAlign &gasChildUserDataSizeAlign,
                                                    uint32_t numRayTypes, uint8_t* records) const {
        uint32_t numMaterials = static_cast<uint32_t>(materials.size());
        for (uint32_t matIdx = 0; matIdx < numMaterials; ++matIdx) {
            throwRuntimeError(materials[matIdx][0], "Default material (== material set 0) is not set for material %u.", matIdx);
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
            delete[] geom.widthBuffers;
            delete[] geom.widthBufferArray;
            delete[] geom.vertexBuffers;
            delete[] geom.vertexBufferArray;
            geom.vertexBufferArray = new CUdeviceptr[n];
            geom.vertexBuffers = new BufferView[n];
            geom.widthBufferArray = new CUdeviceptr[n];
            geom.widthBuffers = new BufferView[n];
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
        m->throwRuntimeError(std::holds_alternative<Priv::TriangleGeometry>(m->geometry),
                             "This geometry instance was created not for triangles.");
        auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
        geom.vertexFormat = format;
    }

    void GeometryInstance::setVertexBuffer(const BufferView &vertexBuffer, uint32_t motionStep) const {
        m->throwRuntimeError(!std::holds_alternative<Priv::CustomPrimitiveGeometry>(m->geometry),
                             "This geometry instance was created not for triangles or curves.");
        m->throwRuntimeError(motionStep < m->numMotionSteps, "motionStep %u is out of bounds [0, %u).",
                             motionStep, m->numMotionSteps);
        if (std::holds_alternative<Priv::TriangleGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
            geom.vertexBuffers[motionStep] = vertexBuffer;
        }
        else if (std::holds_alternative<Priv::CurveGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
            geom.vertexBuffers[motionStep] = vertexBuffer;
        }
    }

    void GeometryInstance::setWidthBuffer(const BufferView &widthBuffer, uint32_t motionStep) const {
        m->throwRuntimeError(std::holds_alternative<Priv::CurveGeometry>(m->geometry),
                             "This geometry instance was created not for curves.");
        m->throwRuntimeError(motionStep < m->numMotionSteps, "motionStep %u is out of bounds [0, %u).",
                             motionStep, m->numMotionSteps);
        auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
        geom.widthBuffers[motionStep] = widthBuffer;
    }

    void GeometryInstance::setTriangleBuffer(const BufferView &triangleBuffer, OptixIndicesFormat format) const {
        m->throwRuntimeError(std::holds_alternative<Priv::TriangleGeometry>(m->geometry),
                             "This geometry instance was created not for triangles.");
        auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
        geom.triangleBuffer = triangleBuffer;
        geom.indexFormat = format;
    }

    void GeometryInstance::setSegmentIndexBuffer(const BufferView &segmentIndexBuffer) const {
        m->throwRuntimeError(std::holds_alternative<Priv::CurveGeometry>(m->geometry),
                             "This geometry instance was created not for curves.");
        auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
        geom.segmentIndexBuffer = segmentIndexBuffer;
    }

    void GeometryInstance::setCurveEndcapFlags(OptixCurveEndcapFlags endcapFlags) const {
        m->throwRuntimeError(std::holds_alternative<Priv::CurveGeometry>(m->geometry),
                             "This geometry instance was created not for curves.");
        auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
        geom.endcapFlags = endcapFlags;
    }

    void GeometryInstance::setCustomPrimitiveAABBBuffer(const BufferView &primitiveAABBBuffer, uint32_t motionStep) const {
        m->throwRuntimeError(std::holds_alternative<Priv::CustomPrimitiveGeometry>(m->geometry),
                             "This geometry instance was created not for custom primitives.");
        m->throwRuntimeError(motionStep < m->numMotionSteps, "motionStep %u is out of bounds [0, %u).",
                             motionStep, m->numMotionSteps);
        auto &geom = std::get<Priv::CustomPrimitiveGeometry>(m->geometry);
        geom.primitiveAabbBuffers[motionStep] = primitiveAABBBuffer;
    }

    void GeometryInstance::setPrimitiveIndexOffset(uint32_t offset) const {
        m->primitiveIndexOffset = offset;
    }

    void GeometryInstance::setNumMaterials(uint32_t numMaterials, const BufferView &matIndexBuffer, uint32_t indexSize) const {
        m->throwRuntimeError(!std::holds_alternative<Priv::CurveGeometry>(m->geometry),
                             "Geometry instance for curves is not allowed to have multiple materials.");
        m->throwRuntimeError(numMaterials > 0, "Invalid number of materials %u.", numMaterials);
        m->throwRuntimeError((numMaterials == 1) != matIndexBuffer.isValid(),
                             "Material index offset buffer must be provided when multiple materials are used.");
        m->throwRuntimeError(indexSize >= 1 && indexSize <= 4,
                             "Invalid index offset size.");
        if (matIndexBuffer.isValid())
            m->throwRuntimeError(matIndexBuffer.stride() >= indexSize,
                                 "Buffer's stride is smaller than the given index offset size.");
        m->buildInputFlags.resize(numMaterials, OPTIX_GEOMETRY_FLAG_NONE);
        if (std::holds_alternative<Priv::TriangleGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
            geom.materialIndexBuffer = matIndexBuffer;
            geom.materialIndexSize = indexSize;
        }
        else if (std::holds_alternative<Priv::CustomPrimitiveGeometry>(m->geometry)) {
            auto &geom = std::get<Priv::CustomPrimitiveGeometry>(m->geometry);
            geom.materialIndexBuffer = matIndexBuffer;
            geom.materialIndexSize = indexSize;
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
        m->throwRuntimeError(matIdx < numMaterials, "Out of material bounds [0, %u).",
                             static_cast<uint32_t>(numMaterials));

        m->buildInputFlags[matIdx] = flags;
    }

    void GeometryInstance::setMaterial(uint32_t matSetIdx, uint32_t matIdx, Material mat) const {
        size_t numMaterials = m->materials.size();
        m->throwRuntimeError(matIdx < numMaterials, "Out of material bounds [0, %u).",
                             static_cast<uint32_t>(numMaterials));

        uint32_t prevNumMatSets = static_cast<uint32_t>(m->materials[matIdx].size());
        if (matSetIdx >= prevNumMatSets)
            m->materials[matIdx].resize(matSetIdx + 1, nullptr);
        m->materials[matIdx][matSetIdx] = extract(mat);
    }

    void GeometryInstance::setUserData(const void* data, uint32_t size, uint32_t alignment) const {
        m->throwRuntimeError(size <= s_maxGeometryInstanceUserDataSize,
                             "Maximum user data size for GeometryInstance is %u bytes.", s_maxGeometryInstanceUserDataSize);
        m->throwRuntimeError(alignment > 0 && alignment <= OPTIX_SBT_RECORD_ALIGNMENT,
                             "Valid alignment range is [1, %u].", OPTIX_SBT_RECORD_ALIGNMENT);
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
        m->throwRuntimeError(std::holds_alternative<Priv::TriangleGeometry>(m->geometry),
                             "This geometry instance was created not for triangles.");
        const auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
        return geom.vertexFormat;
    }

    BufferView GeometryInstance::getVertexBuffer(uint32_t motionStep) {
        m->throwRuntimeError(!std::holds_alternative<Priv::CustomPrimitiveGeometry>(m->geometry),
                             "This geometry instance was created not for triangles or curves.");
        m->throwRuntimeError(motionStep < m->numMotionSteps, "motionStep %u is out of bounds [0, %u).",
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
        m->throwRuntimeError(std::holds_alternative<Priv::CurveGeometry>(m->geometry),
                             "This geometry instance was created not for curves.");
        m->throwRuntimeError(motionStep < m->numMotionSteps, "motionStep %u is out of bounds [0, %u).",
                             motionStep, m->numMotionSteps);
        const auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
        return geom.widthBuffers[motionStep];
    }

    BufferView GeometryInstance::getTriangleBuffer(OptixIndicesFormat* format) const {
        m->throwRuntimeError(std::holds_alternative<Priv::TriangleGeometry>(m->geometry),
                             "This geometry instance was created not for triangles.");
        const auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
        if (format)
            *format = geom.indexFormat;
        return geom.triangleBuffer;
    }

    BufferView GeometryInstance::getSegmentIndexBuffer() const {
        m->throwRuntimeError(std::holds_alternative<Priv::CurveGeometry>(m->geometry),
                             "This geometry instance was created not for curves.");
        const auto &geom = std::get<Priv::CurveGeometry>(m->geometry);
        return geom.segmentIndexBuffer;
    }

    BufferView GeometryInstance::getCustomPrimitiveAABBBuffer(uint32_t motionStep) const {
        m->throwRuntimeError(std::holds_alternative<Priv::CustomPrimitiveGeometry>(m->geometry),
                             "This geometry instance was created not for custom primitives.");
        m->throwRuntimeError(motionStep < m->numMotionSteps, "motionStep %u is out of bounds [0, %u).",
                             motionStep, m->numMotionSteps);
        const auto &geom = std::get<Priv::CustomPrimitiveGeometry>(m->geometry);
        return geom.primitiveAabbBuffers[motionStep];
    }

    uint32_t GeometryInstance::getPrimitiveIndexOffset() const {
        return m->primitiveIndexOffset;
    }

    uint32_t GeometryInstance::getNumMaterials(BufferView* matIndexBuffer, uint32_t* indexSize) const {
        if (matIndexBuffer || indexSize) {
            if (std::holds_alternative<Priv::TriangleGeometry>(m->geometry)) {
                const auto &geom = std::get<Priv::TriangleGeometry>(m->geometry);
                if (matIndexBuffer)
                    *matIndexBuffer = geom.materialIndexBuffer;
                if (indexSize)
                    *indexSize = geom.materialIndexSize;
            }
            else if (std::holds_alternative<Priv::CustomPrimitiveGeometry>(m->geometry)) {
                const auto &geom = std::get<Priv::CustomPrimitiveGeometry>(m->geometry);
                if (matIndexBuffer)
                    *matIndexBuffer = geom.materialIndexBuffer;
                if (indexSize)
                    *indexSize = geom.materialIndexSize;
            }
            else {
                if (matIndexBuffer)
                    *matIndexBuffer = BufferView();
                if (indexSize)
                    *indexSize = 0;
            }
        }
        return static_cast<uint32_t>(m->materials.size());
    }

    OptixGeometryFlags GeometryInstance::getGeometryFlags(uint32_t matIdx) const {
        size_t numMaterials = m->materials.size();
        m->throwRuntimeError(matIdx < numMaterials, "Out of material bounds [0, %u).",
                             static_cast<uint32_t>(numMaterials));
        return m->buildInputFlags[matIdx];
    }

    Material GeometryInstance::getMaterial(uint32_t matSetIdx, uint32_t matIdx) const {
        size_t numMaterials = m->materials.size();
        m->throwRuntimeError(matIdx < numMaterials, "Out of material bounds [0, %u).",
                             static_cast<uint32_t>(numMaterials));
        size_t numMatSets = m->materials[matIdx].size();
        m->throwRuntimeError(matSetIdx < numMatSets, "Out of material set bounds [0, %u).",
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



    void GeometryAccelerationStructure::Priv::calcSBTRequirements(uint32_t matSetIdx, SizeAlign* maxRecordSizeAlign, uint32_t* numSBTRecords) const {
        *maxRecordSizeAlign = SizeAlign();
        *numSBTRecords = 0;
        for (const Child &child : children) {
            SizeAlign geomInstRecordSizeAlign;
            uint32_t geomInstNumSBTRecords;
            child.geomInst->calcSBTRequirements(matSetIdx,
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

    uint32_t GeometryAccelerationStructure::Priv::fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, uint8_t* records) const {
        throwRuntimeError(matSetIdx < numRayTypesPerMaterialSet.size(),
                          "Material set index %u is out of bounds [0, %u).",
                          matSetIdx, static_cast<uint32_t>(numRayTypesPerMaterialSet.size()));

        uint32_t numRayTypes = numRayTypesPerMaterialSet[matSetIdx];
        uint32_t sumRecords = 0;
        for (uint32_t sbtGasIdx = 0; sbtGasIdx < children.size(); ++sbtGasIdx) {
            const Child &child = children[sbtGasIdx];
            uint32_t numRecords = child.geomInst->fillSBTRecords(pipeline, matSetIdx,
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

    void GeometryAccelerationStructure::setConfiguration(ASTradeoff tradeoff,
                                                         bool allowUpdate,
                                                         bool allowCompaction,
                                                         bool allowRandomVertexAccess) const {
        m->throwRuntimeError(m->geomType != GeometryType::CustomPrimitives || !allowRandomVertexAccess,
                             "Random vertex access is the feature only for triangle/curve GAS.");
        bool changed = false;
        changed |= m->tradeoff != tradeoff;
        m->tradeoff = tradeoff;
        changed |= m->allowUpdate != allowUpdate;
        m->allowUpdate = allowUpdate;
        changed |= m->allowCompaction != allowCompaction;
        m->allowCompaction = allowCompaction;
        changed |= m->allowRandomVertexAccess != allowRandomVertexAccess;
        m->allowRandomVertexAccess = allowRandomVertexAccess;

        if (changed)
            m->markDirty();
    }

    void GeometryAccelerationStructure::setMotionOptions(uint32_t numKeys, float timeBegin, float timeEnd, OptixMotionFlags flags) const {
        m->buildOptions.motionOptions.numKeys = numKeys;
        m->buildOptions.motionOptions.timeBegin = timeBegin;
        m->buildOptions.motionOptions.timeEnd = timeEnd;
        m->buildOptions.motionOptions.flags = flags;

        m->markDirty();
    }

    void GeometryAccelerationStructure::addChild(GeometryInstance geomInst, CUdeviceptr preTransform,
                                                 const void* data, uint32_t size, uint32_t alignment) const {
        auto _geomInst = extract(geomInst);
        m->throwRuntimeError(_geomInst, "Invalid geometry instance %p.", _geomInst);
        m->throwRuntimeError(_geomInst->getScene() == m->scene, "Scene mismatch for the given geometry instance %s.",
                             _geomInst->getName().c_str());
        const char* geomTypeStrs[] = {
            "triangles",
            "linear segments",
            "quadratic B-splines",
            "cubic B-splines",
            "Catmull-Rom splines",
            "custom primitives" };
        m->throwRuntimeError(_geomInst->getGeometryType() == m->geomType,
                             "This GAS was created for %s.", geomTypeStrs[static_cast<uint32_t>(m->geomType)]);
        m->throwRuntimeError(m->geomType == GeometryType::Triangles || preTransform == 0,
                             "Pre-transform is valid only for triangles.");
        Priv::Child child;
        child.geomInst = _geomInst;
        child.preTransform = preTransform;
        auto idx = std::find(m->children.cbegin(), m->children.cend(), child);
        m->throwRuntimeError(idx == m->children.cend(), "Geometry instance %s with transform %p has been already added.",
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
        m->throwRuntimeError(index < numChildren, "Index is out of bounds [0, %u).]",
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
        m->throwRuntimeError(matSetIdx < numMatSets,
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
            m->throwRuntimeError(childNumMotionSteps == numMotionSteps,
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
            | (m->allowRandomVertexAccess ? OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS : 0);

        uint32_t numBuildInputs = static_cast<uint32_t>(m->buildInputs.size());
        if (numBuildInputs > 0)
            OPTIX_CHECK(optixAccelComputeMemoryUsage(m->getRawContext(), &m->buildOptions,
                                                     m->buildInputs.data(), numBuildInputs,
                                                     &m->memoryRequirement));
        else
            m->memoryRequirement = {};

        *memoryRequirement = m->memoryRequirement;

        m->readyToBuild = true;
    }

    OptixTraversableHandle GeometryAccelerationStructure::rebuild(CUstream stream, const BufferView &accelBuffer, const BufferView &scratchBuffer) const {
        m->throwRuntimeError(m->readyToBuild, "You need to call prepareForBuild() before rebuild.");
        m->throwRuntimeError(accelBuffer.sizeInBytes() >= m->memoryRequirement.outputSizeInBytes,
                             "Size of the given buffer is not enough.");
        m->throwRuntimeError(scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempSizeInBytes,
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
            OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
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
        m->throwRuntimeError(compactionEnabled, "This AS does not allow compaction.");
        m->throwRuntimeError(m->available, "Uncompacted AS has not been built yet.");

        if (m->compactedAvailable)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        // EN: Wait the completion of rebuild/update then obtain the size after coompaction.
        // TODO: ? stream
        uint32_t numBuildInputs = static_cast<uint32_t>(m->buildInputs.size());
        if (numBuildInputs > 0) {
            CUDADRV_CHECK(cuEventSynchronize(m->finishEvent));
            CUDADRV_CHECK(cuMemcpyDtoH(&m->compactedSize, m->propertyCompactedSize.result, sizeof(m->compactedSize)));
        }
        else {
            m->compactedSize = 0;
        }

        *compactedAccelBufferSize = m->compactedSize;

        m->readyToCompact = true;
    }

    OptixTraversableHandle GeometryAccelerationStructure::compact(CUstream stream, const BufferView &compactedAccelBuffer) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        m->throwRuntimeError(compactionEnabled, "This AS does not allow compaction.");
        m->throwRuntimeError(m->readyToCompact, "You need to call prepareForCompact() before compaction.");
        m->throwRuntimeError(m->available, "Uncompacted AS has not been built yet.");
        m->throwRuntimeError(compactedAccelBuffer.sizeInBytes() >= m->compactedSize,
                             "Size of the given buffer is not enough.");

        uint32_t numBuildInputs = static_cast<uint32_t>(m->buildInputs.size());
        if (numBuildInputs > 0) {
            OPTIX_CHECK(optixAccelCompact(m->getRawContext(), stream,
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
        m->throwRuntimeError(updateEnabled, "This AS does not allow update.");
        m->throwRuntimeError(m->available || m->compactedAvailable, "AS has not been built yet.");
        m->throwRuntimeError(scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempUpdateSizeInBytes,
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
            OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
                                        &m->buildOptions, m->buildInputs.data(), numBuildInputs,
                                        scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                        accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
                                        &tempHandle,
                                        nullptr, 0));
        else
            tempHandle = 0;
        optixuAssert(tempHandle == handle, "GAS %s: Update should not change the handle itself, what's going on?", getName());
    }

    void GeometryAccelerationStructure::setChildUserData(uint32_t index, const void* data, uint32_t size, uint32_t alignment) const {
        uint32_t numChildren = static_cast<uint32_t>(m->children.size());
        m->throwRuntimeError(index < numChildren, "Index is out of bounds [0, %u).]",
                             numChildren);
        m->throwRuntimeError(size <= s_maxGASChildUserDataSize,
                             "Maximum user data size for GAS child is %u bytes.", s_maxGASChildUserDataSize);
        m->throwRuntimeError(alignment > 0 && alignment <= OPTIX_SBT_RECORD_ALIGNMENT,
                             "Valid alignment range is [1, %u].", OPTIX_SBT_RECORD_ALIGNMENT);
        Priv::Child &child = m->children[index];
        if (child.userDataSizeAlign.size != size ||
            child.userDataSizeAlign.alignment != alignment)
            m->scene->markSBTLayoutDirty();
        child.userDataSizeAlign = SizeAlign(size, alignment);
        child.userData.resize(size);
        std::memcpy(child.userData.data(), data, size);
    }

    void GeometryAccelerationStructure::setUserData(const void* data, uint32_t size, uint32_t alignment) const {
        m->throwRuntimeError(size <= s_maxGASUserDataSize,
                             "Maximum user data size for GAS is %u bytes.", s_maxGASUserDataSize);
        m->throwRuntimeError(alignment > 0 && alignment <= OPTIX_SBT_RECORD_ALIGNMENT,
                             "Valid alignment range is [1, %u].", OPTIX_SBT_RECORD_ALIGNMENT);
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

    void GeometryAccelerationStructure::getConfiguration(ASTradeoff* tradeOff, bool* allowUpdate, bool* allowCompaction, bool* allowRandomVertexAccess) const {
        if (tradeOff)
            *tradeOff = m->tradeoff;
        if (allowUpdate)
            *allowUpdate = m->allowUpdate;
        if (allowCompaction)
            *allowCompaction = m->allowCompaction;
        if (allowRandomVertexAccess)
            *allowRandomVertexAccess = m->allowRandomVertexAccess;
    }

    void GeometryAccelerationStructure::getMotionOptions(uint32_t* numKeys, float* timeBegin, float* timeEnd, OptixMotionFlags* flags) const {
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

    uint32_t GeometryAccelerationStructure::findChildIndex(GeometryInstance geomInst, CUdeviceptr preTransform) const {
        auto _geomInst = extract(geomInst);
        m->throwRuntimeError(_geomInst, "Invalid geometry instance %p.", _geomInst);
        m->throwRuntimeError(_geomInst->getScene() == m->scene, "Scene mismatch for the given geometry instance %s.",
                             _geomInst->getName().c_str());
        Priv::Child child;
        child.geomInst = _geomInst;
        child.preTransform = preTransform;
        auto idx = std::find(m->children.cbegin(), m->children.cend(), child);
        if (idx == m->children.cend())
            return 0xFFFFFFFF;

        return static_cast<uint32_t>(std::distance(m->children.cbegin(), idx));
    }

    GeometryInstance GeometryAccelerationStructure::getChild(uint32_t index, CUdeviceptr* preTransform) const {
        uint32_t numChildren = static_cast<uint32_t>(m->children.size());
        m->throwRuntimeError(index < numChildren, "Index is out of bounds [0, %u).]",
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
        m->throwRuntimeError(matSetIdx < numMatSets,
                             "Material set index %u is out of bounds [0, %u).",
                             matSetIdx, numMatSets);
        return m->numRayTypesPerMaterialSet[matSetIdx];
    }

    void GeometryAccelerationStructure::getChildUserData(uint32_t index, void* data, uint32_t* size, uint32_t* alignment) const {
        uint32_t numChildren = static_cast<uint32_t>(m->children.size());
        m->throwRuntimeError(index < numChildren, "Index is out of bounds [0, %u).]",
                             numChildren);
        const Priv::Child &child = m->children[index];
        if (data)
            std::memcpy(data, child.userData.data(), child.userDataSizeAlign.size);
        if (size)
            *size = child.userDataSizeAlign.size;
        if (alignment)
            *alignment = child.userDataSizeAlign.alignment;
    }

    void GeometryAccelerationStructure::getUserData(void* data, uint32_t* size, uint32_t* alignment) const {
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

    void Transform::setConfiguration(TransformType type, uint32_t numKeys,
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
            auto motionData = reinterpret_cast<OptixSRTData*>(m->data + offsetof(OptixSRTMotionTransform, srtData));
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
        m->throwRuntimeError(m->type == TransformType::MatrixMotion,
                             "This transform has been configured as matrix motion transform.");
        m->throwRuntimeError(keyIdx <= m->options.numKeys,
                             "Number of motion keys was set to %u", m->options.numKeys);
        auto motionData = reinterpret_cast<float*>(m->data + offsetof(OptixMatrixMotionTransform, transform));
        float* dataPerKey = motionData + 12 * keyIdx;

        std::copy_n(matrix, 12, dataPerKey);

        markDirty();
    }

    void Transform::setSRTMotionKey(uint32_t keyIdx, const float scale[3], const float orientation[4], const float translation[3]) const {
        m->throwRuntimeError(m->type == TransformType::SRTMotion,
                             "This transform has been configured as SRT motion transform.");
        m->throwRuntimeError(keyIdx <= m->options.numKeys,
                             "Number of motion keys was set to %u", m->options.numKeys);
        auto motionData = reinterpret_cast<OptixSRTData*>(m->data + offsetof(OptixSRTMotionTransform, srtData));
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
        m->throwRuntimeError(m->type == TransformType::Static,
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
        m->throwRuntimeError(_child, "Invalid GAS %p.", _child);
        m->throwRuntimeError(_child->getScene() == m->scene, "Scene mismatch for the given GAS %s.",
                             _child->getName().c_str());
        m->child = _child;

        markDirty();
    }

    void Transform::setChild(InstanceAccelerationStructure child) const {
        auto _child = extract(child);
        m->throwRuntimeError(_child, "Invalid IAS %p.", _child);
        m->throwRuntimeError(_child->getScene() == m->scene, "Scene mismatch for the given IAS %s.",
                             _child->getName().c_str());
        m->child = _child;

        markDirty();
    }

    void Transform::setChild(Transform child) const {
        auto _child = extract(child);
        m->throwRuntimeError(_child, "Invalid transform %p.", _child);
        m->throwRuntimeError(_child->getScene() == m->scene, "Scene mismatch for the given transform %s.",
                             _child->getName().c_str());
        m->child = _child;

        markDirty();
    }

    void Transform::markDirty() const {
        return m->markDirty();
    }

    OptixTraversableHandle Transform::rebuild(CUstream stream, const BufferView &trDeviceMem) const {
        m->throwRuntimeError(m->type != TransformType::Invalid, "Transform type is invalid.");
        m->throwRuntimeError(trDeviceMem.sizeInBytes() >= m->dataSize,
                             "Size of the given buffer is not enough.");
        m->throwRuntimeError(!std::holds_alternative<void*>(m->child), "Child is invalid.");

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
        OPTIX_CHECK(optixConvertPointerToTraversableHandle(m->getRawContext(), trDeviceMem.getCUdeviceptr(),
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
        m->throwRuntimeError(m->type == TransformType::MatrixMotion,
                             "This transform has been configured as matrix motion transform.");
        m->throwRuntimeError(keyIdx <= m->options.numKeys,
                             "Number of motion keys was set to %u", m->options.numKeys);
        auto motionData = reinterpret_cast<const float*>(m->data + offsetof(OptixMatrixMotionTransform, transform));
        const float* dataPerKey = motionData + 12 * keyIdx;

        std::copy_n(dataPerKey, 12, matrix);
    }

    void Transform::getSRTMotionKey(uint32_t keyIdx, float scale[3], float orientation[4], float translation[3]) const {
        m->throwRuntimeError(m->type == TransformType::SRTMotion,
                             "This transform has been configured as SRT motion transform.");
        m->throwRuntimeError(keyIdx <= m->options.numKeys,
                             "Number of motion keys was set to %u", m->options.numKeys);
        auto motionData = reinterpret_cast<const OptixSRTData*>(m->data + offsetof(OptixSRTMotionTransform, srtData));
        const OptixSRTData* dataPerKey = motionData + keyIdx;

        scale[0] = dataPerKey->sx;
        scale[1] = dataPerKey->sy;
        scale[2] = dataPerKey->sz;
        std::copy_n(&dataPerKey->qx, 4, orientation);
        std::copy_n(&dataPerKey->tx, 3, translation);
    }

    void Transform::getStaticTransform(float matrix[12]) const {
        m->throwRuntimeError(m->type == TransformType::Static,
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
        m->throwRuntimeError(std::holds_alternative<typename T::Priv*>(m->child),
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
        m->throwRuntimeError(_child, "Invalid GAS %p.", _child);
        m->throwRuntimeError(_child->getScene() == m->scene, "Scene mismatch for the given GAS %s.",
                             _child->getName().c_str());
        m->child = _child;
        m->matSetIndex = matSetIdx;
    }

    void Instance::setChild(InstanceAccelerationStructure child) const {
        auto _child = extract(child);
        m->throwRuntimeError(_child, "Invalid IAS %p.", _child);
        m->throwRuntimeError(_child->getScene() == m->scene, "Scene mismatch for the given IAS %s.",
                             _child->getName().c_str());
        m->child = _child;
        m->matSetIndex = 0;
    }

    void Instance::setChild(Transform child, uint32_t matSetIdx) const {
        auto _child = extract(child);
        m->throwRuntimeError(_child, "Invalid transform %p.", _child);
        m->throwRuntimeError(_child->getScene() == m->scene, "Scene mismatch for the given transform %s.",
                             _child->getName().c_str());
        m->child = _child;
        m->matSetIndex = matSetIdx;
    }

    void Instance::setTransform(const float transform[12]) const {
        std::copy_n(transform, 12, m->instTransform);
    }

    void Instance::setID(uint32_t value) const {
        uint32_t maxInstanceID = m->scene->getContext()->getMaxInstanceID();
        m->throwRuntimeError(value <= maxInstanceID,
                             "Max instance ID value is 0x%08x.", maxInstanceID);
        m->id = value;
    }

    void Instance::setVisibilityMask(uint32_t mask) const {
        uint32_t numVisibilityMaskBits = m->scene->getContext()->getNumVisibilityMaskBits();
        m->throwRuntimeError((mask >> numVisibilityMaskBits) == 0,
                             "Number of visibility mask bits is %u.", numVisibilityMaskBits);
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
        m->throwRuntimeError(std::holds_alternative<typename T::Priv*>(m->child),
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

    void InstanceAccelerationStructure::setConfiguration(ASTradeoff tradeoff,
                                                         bool allowUpdate,
                                                         bool allowCompaction,
                                                         bool allowRandomInstanceAccess) const {
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

    void InstanceAccelerationStructure::setMotionOptions(uint32_t numKeys, float timeBegin, float timeEnd, OptixMotionFlags flags) const {
        m->buildOptions.motionOptions.numKeys = numKeys;
        m->buildOptions.motionOptions.timeBegin = timeBegin;
        m->buildOptions.motionOptions.timeEnd = timeEnd;
        m->buildOptions.motionOptions.flags = flags;

        m->markDirty(false);
    }

    void InstanceAccelerationStructure::addChild(Instance instance) const {
        _Instance* _inst = extract(instance);
        m->throwRuntimeError(_inst, "Invalid instance %p.");
        m->throwRuntimeError(_inst->getScene() == m->scene, "Scene mismatch for the given instance %s.",
                             _inst->getName().c_str());
        auto idx = std::find(m->children.cbegin(), m->children.cend(), _inst);
        m->throwRuntimeError(idx == m->children.cend(), "Instance %s has been already added.",
                             _inst->getName().c_str());

        m->children.push_back(_inst);

        m->markDirty(false);
    }

    void InstanceAccelerationStructure::removeChildAt(uint32_t index) const {
        uint32_t numChildren = static_cast<uint32_t>(m->children.size());
        m->throwRuntimeError(index < numChildren, "Index is out of bounds [0, %u).]",
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

        OPTIX_CHECK(optixAccelComputeMemoryUsage(m->getRawContext(), &m->buildOptions,
                                                 &m->buildInput, 1,
                                                 &m->memoryRequirement));

        *memoryRequirement = m->memoryRequirement;

        m->readyToBuild = true;
    }

    OptixTraversableHandle InstanceAccelerationStructure::rebuild(CUstream stream, const BufferView &instanceBuffer,
                                                                  const BufferView &accelBuffer, const BufferView &scratchBuffer) const {
        m->throwRuntimeError(m->readyToBuild, "You need to call prepareForBuild() before rebuild.");
        m->throwRuntimeError(accelBuffer.sizeInBytes() >= m->memoryRequirement.outputSizeInBytes,
                             "Size of the given buffer is not enough.");
        m->throwRuntimeError(scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempSizeInBytes,
                             "Size of the given scratch buffer is not enough.");
        m->throwRuntimeError(instanceBuffer.sizeInBytes() >= m->instances.size() * sizeof(OptixInstance),
                             "Size of the given instance buffer is not enough.");
        m->throwRuntimeError(m->scene->sbtLayoutGenerationDone(),
                             "Shader binding table layout generation has not been done.");

        uint32_t childIdx = 0;
        for (const _Instance* child : m->children)
            child->fillInstance(&m->instances[childIdx++]);
        CUDADRV_CHECK(cuMemcpyHtoDAsync(instanceBuffer.getCUdeviceptr(), m->instances.data(),
                                        m->instances.size() * sizeof(OptixInstance),
                                        stream));
        m->buildInput.instanceArray.instances = m->children.size() > 0 ? instanceBuffer.getCUdeviceptr() : 0;

        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream, &m->buildOptions, &m->buildInput, 1,
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
        CUDADRV_CHECK(cuMemcpyDtoH(&m->compactedSize, m->propertyCompactedSize.result, sizeof(m->compactedSize)));

        *compactedAccelBufferSize = m->compactedSize;

        m->readyToCompact = true;
    }

    OptixTraversableHandle InstanceAccelerationStructure::compact(CUstream stream, const BufferView &compactedAccelBuffer) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        m->throwRuntimeError(compactionEnabled, "This AS does not allow compaction.");
        m->throwRuntimeError(m->readyToCompact, "You need to call prepareForCompact() before compaction.");
        m->throwRuntimeError(m->available, "Uncompacted AS has not been built yet.");
        m->throwRuntimeError(compactedAccelBuffer.sizeInBytes() >= m->compactedSize,
                             "Size of the given buffer is not enough.");

        OPTIX_CHECK(optixAccelCompact(m->getRawContext(), stream,
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
        m->throwRuntimeError(updateEnabled, "This AS does not allow update.");
        m->throwRuntimeError(m->available || m->compactedAvailable, "AS has not been built yet.");
        m->throwRuntimeError(scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempUpdateSizeInBytes,
                             "Size of the given scratch buffer is not enough.");

        uint32_t childIdx = 0;
        for (const _Instance* child : m->children)
            child->updateInstance(&m->instances[childIdx++]);
        CUDADRV_CHECK(cuMemcpyHtoDAsync(m->instanceBuffer.getCUdeviceptr(), m->instances.data(),
                                        m->instances.size() * sizeof(OptixInstance),
                                        stream));

        const BufferView &accelBuffer = m->compactedAvailable ? m->compactedAccelBuffer : m->accelBuffer;
        OptixTraversableHandle handle = m->compactedAvailable ? m->compactedHandle : m->handle;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OptixTraversableHandle tempHandle = handle;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
                                    &m->buildOptions, &m->buildInput, 1,
                                    scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                    accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
                                    &tempHandle,
                                    nullptr, 0));
        optixuAssert(tempHandle == handle, "IAS %s: Update should not change the handle itself, what's going on?", getName());
    }

    bool InstanceAccelerationStructure::isReady() const {
        return m->isReady();
    }

    OptixTraversableHandle InstanceAccelerationStructure::getHandle() const {
        return m->getHandle();
    }

    void InstanceAccelerationStructure::getConfiguration(ASTradeoff* tradeOff, bool* allowUpdate, bool* allowCompaction) const {
        if (tradeOff)
            *tradeOff = m->tradeoff;
        if (allowUpdate)
            *allowUpdate = m->allowUpdate;
        if (allowCompaction)
            *allowCompaction = m->allowCompaction;
    }

    void InstanceAccelerationStructure::getMotionOptions(uint32_t* numKeys, float* timeBegin, float* timeEnd, OptixMotionFlags* flags) const {
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
        m->throwRuntimeError(_inst, "Invalid instance %p.", _inst);
        m->throwRuntimeError(_inst->getScene() == m->scene, "Scene mismatch for the given instance %s.",
                             _inst->getName().c_str());
        auto idx = std::find(m->children.cbegin(), m->children.cend(), _inst);
        if (idx == m->children.cend())
            return 0xFFFFFFFF;

        return static_cast<uint32_t>(std::distance(m->children.cbegin(), idx));
    }

    Instance InstanceAccelerationStructure::getChild(uint32_t index) const {
        uint32_t numChildren = static_cast<uint32_t>(m->children.size());
        m->throwRuntimeError(index < numChildren, "Index is out of bounds [0, %u).]",
                             numChildren);
        return m->children[index]->getPublicType();
    }



    Pipeline::Priv::~Priv() {
        if (pipelineLinked)
            optixPipelineDestroy(rawPipeline);
        for (auto it = modulesForCurveIS.begin(); it != modulesForCurveIS.end(); ++it)
            it->second->getPublicType().destroy();
        modulesForCurveIS.clear();
        context->unregisterName(this);
    }
    
    void Pipeline::Priv::markDirty() {
        if (pipelineLinked)
            OPTIX_CHECK(optixPipelineDestroy(rawPipeline));
        pipelineLinked = false;
    }

    OptixModule Pipeline::Priv::getModuleForCurves(
        OptixPrimitiveType curveType, OptixCurveEndcapFlags endcapFlags,
        ASTradeoff tradeoff, bool allowUpdate, bool allowCompaction, bool allowRandomVertexAccess) {
        if (curveType == OPTIX_PRIMITIVE_TYPE_TRIANGLE || curveType == OPTIX_PRIMITIVE_TYPE_CUSTOM)
            return nullptr;

        KeyForCurveModule key{ curveType, endcapFlags, OPTIX_BUILD_FLAG_NONE };
        if (modulesForCurveIS.count(key) == 0) {
            OptixModuleCompileOptions moduleCompileOptions = {};
            moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

            OptixBuiltinISOptions builtinISOptions = {};
            builtinISOptions.builtinISModuleType = curveType;
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
            OPTIX_CHECK(optixBuiltinISModuleGet(context->getRawContext(),
                                                &moduleCompileOptions,
                                                &pipelineCompileOptions,
                                                &builtinISOptions,
                                                &rawModule));

            modulesForCurveIS[key] = new _Module(this, rawModule);
        }

        return modulesForCurveIS.at(key)->getRawModule();
    }
    
    void Pipeline::Priv::createProgram(const OptixProgramGroupDesc &desc, const OptixProgramGroupOptions &options, OptixProgramGroup* group) {
        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(getRawContext(),
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
            sbtParams.hitgroupRecordCount = static_cast<uint32_t>(hitGroupSbt.sizeInBytes() / scene->getSingleRecordSize());

            hitGroupSbtIsUpToDate = true;
        }
    }

    void Pipeline::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    void Pipeline::setPipelineOptions(uint32_t numPayloadValuesInDwords, uint32_t numAttributeValuesInDwords,
                                      const char* launchParamsVariableName, size_t sizeOfLaunchParams,
                                      bool useMotionBlur,
                                      OptixTraversableGraphFlags traversableGraphFlags,
                                      OptixExceptionFlags exceptionFlags,
                                      OptixPrimitiveTypeFlags supportedPrimitiveTypeFlags) const {
        m->throwRuntimeError(!m->pipelineLinked, "Changing pipeline options after linking is not supported yet.");

        // JP: パイプライン中のモジュール、そしてパイプライン自体に共通なコンパイルオプションの設定。
        // EN: Set pipeline compile options common among modules in the pipeline and the pipeline itself.
        m->pipelineCompileOptions = {};
        m->pipelineCompileOptions.numPayloadValues = numPayloadValuesInDwords;
        m->pipelineCompileOptions.numAttributeValues = numAttributeValuesInDwords;
        m->pipelineCompileOptions.pipelineLaunchParamsVariableName = launchParamsVariableName;
        m->pipelineCompileOptions.usesMotionBlur = useMotionBlur;
        m->pipelineCompileOptions.traversableGraphFlags = traversableGraphFlags;
        m->pipelineCompileOptions.exceptionFlags = exceptionFlags;
        m->pipelineCompileOptions.usesPrimitiveTypeFlags = supportedPrimitiveTypeFlags;

        m->sizeOfPipelineLaunchParams = sizeOfLaunchParams;
    }

    Module Pipeline::createModuleFromPTXString(const std::string &ptxString, int32_t maxRegisterCount,
                                               OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel,
                                               OptixModuleCompileBoundValueEntry* boundValues, uint32_t numBoundValues,
                                               const PayloadType* payloadTypes, uint32_t numPayloadTypes) const {
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
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(m->getRawContext(),
                                                 &moduleCompileOptions,
                                                 &m->pipelineCompileOptions,
                                                 ptxString.c_str(), ptxString.size(),
                                                 log, &logSize,
                                                 &rawModule));

        return (new _Module(m, rawModule))->getPublicType();
    }

    ProgramGroup Pipeline::createRayGenProgram(Module module, const char* entryFunctionName) const {
        _Module* _module = extract(module);
        m->throwRuntimeError(_module && entryFunctionName,
                             "Either of RayGen module or entry function name is not provided.");
        m->throwRuntimeError(_module->getPipeline() == m,
                             "Pipeline mismatch for the given module %s.", _module->getName().c_str());

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = _module->getRawModule();
        desc.raygen.entryFunctionName = entryFunctionName;

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createExceptionProgram(Module module, const char* entryFunctionName) const {
        _Module* _module = extract(module);
        m->throwRuntimeError(_module && entryFunctionName,
                             "Either of Exception module or entry function name is not provided.");
        m->throwRuntimeError(_module->getPipeline() == m,
                             "Pipeline mismatch for the given module %s.", _module->getName().c_str());

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        desc.exception.module = _module->getRawModule();
        desc.exception.entryFunctionName = entryFunctionName;

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createMissProgram(
        Module module, const char* entryFunctionName,
        const PayloadType &payloadType) const {
        _Module* _module = extract(module);
        m->throwRuntimeError((_module != nullptr) == (entryFunctionName != nullptr),
                             "Either of Miss module or entry function name is not provided.");
        if (_module)
            m->throwRuntimeError(_module->getPipeline() == m,
                                 "Pipeline mismatch for the given module %s.", _module->getName().c_str());

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

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createHitProgramGroupForTriangleIS(
        Module module_CH, const char* entryFunctionNameCH,
        Module module_AH, const char* entryFunctionNameAH,
        const PayloadType &payloadType) const {
        _Module* _module_CH = extract(module_CH);
        _Module* _module_AH = extract(module_AH);
        m->throwRuntimeError((_module_CH != nullptr) == (entryFunctionNameCH != nullptr),
                             "Either of CH module or entry function name is not provided.");
        m->throwRuntimeError((_module_AH != nullptr) == (entryFunctionNameAH != nullptr),
                             "Either of AH module or entry function name is not provided.");
        m->throwRuntimeError(entryFunctionNameCH || entryFunctionNameAH,
                             "Either of CH/AH entry function name must be provided.");
        if (_module_CH)
            m->throwRuntimeError(_module_CH->getPipeline() == m,
                                 "Pipeline mismatch for the given CH module %s.",
                                 _module_CH->getName().c_str());
        if (_module_AH)
            m->throwRuntimeError(_module_AH->getPipeline() == m,
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

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createHitProgramGroupForCurveIS(
        OptixPrimitiveType curveType, OptixCurveEndcapFlags endcapFlags,
        Module module_CH, const char* entryFunctionNameCH,
        Module module_AH, const char* entryFunctionNameAH,
        ASTradeoff tradeoff, bool allowUpdate, bool allowCompaction, bool allowRandomVertexAccess,
        const PayloadType &payloadType) const {
        m->throwRuntimeError(curveType != OPTIX_PRIMITIVE_TYPE_TRIANGLE && curveType != OPTIX_PRIMITIVE_TYPE_CUSTOM,
                             "Use the createHitProgramGroupForTriangleIS() or createHitProgramGroupForCustomIS() for triangles or custom primitives respectively.");
        _Module* _module_CH = extract(module_CH);
        _Module* _module_AH = extract(module_AH);
        m->throwRuntimeError((_module_CH != nullptr) == (entryFunctionNameCH != nullptr),
                             "Either of CH module or entry function name is not provided.");
        m->throwRuntimeError((_module_AH != nullptr) == (entryFunctionNameAH != nullptr),
                             "Either of AH module or entry function name is not provided.");
        m->throwRuntimeError(entryFunctionNameCH || entryFunctionNameAH,
                             "Either of CH/AH entry function name must be provided.");
        if (_module_CH)
            m->throwRuntimeError(_module_CH->getPipeline() == m,
                                 "Pipeline mismatch for the given CH module %s.",
                                 _module_CH->getName().c_str());
        if (_module_AH)
            m->throwRuntimeError(_module_AH->getPipeline() == m,
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
        desc.hitgroup.moduleIS = m->getModuleForCurves(
            curveType, endcapFlags,
            tradeoff, allowUpdate, allowCompaction, allowRandomVertexAccess);
        desc.hitgroup.entryFunctionNameIS = nullptr;

        OptixProgramGroupOptions options = {};
        OptixPayloadType optixPayloadType = payloadType.getRawType();
        if (payloadType.numDwords > 0)
            options.payloadType = &optixPayloadType;

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createHitProgramGroupForCustomIS(
        Module module_CH, const char* entryFunctionNameCH,
        Module module_AH, const char* entryFunctionNameAH,
        Module module_IS, const char* entryFunctionNameIS,
        const PayloadType &payloadType) const {
        _Module* _module_CH = extract(module_CH);
        _Module* _module_AH = extract(module_AH);
        _Module* _module_IS = extract(module_IS);
        m->throwRuntimeError((_module_CH != nullptr) == (entryFunctionNameCH != nullptr),
                             "Either of CH module or entry function name is not provided.");
        m->throwRuntimeError((_module_AH != nullptr) == (entryFunctionNameAH != nullptr),
                             "Either of AH module or entry function name is not provided.");
        m->throwRuntimeError(_module_IS != nullptr && entryFunctionNameIS != nullptr,
                             "Intersection program must be provided for custom primitives.");
        m->throwRuntimeError(entryFunctionNameCH || entryFunctionNameAH,
                             "Either of CH/AH entry function name must be provided.");
        if (_module_CH)
            m->throwRuntimeError(_module_CH->getPipeline() == m,
                                 "Pipeline mismatch for the given CH module %s.",
                                 _module_CH->getName().c_str());
        if (_module_AH)
            m->throwRuntimeError(_module_AH->getPipeline() == m,
                                 "Pipeline mismatch for the given AH module %s.",
                                 _module_AH->getName().c_str());
        m->throwRuntimeError(_module_IS->getPipeline() == m,
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

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createEmptyHitProgramGroup() const {
        OptixProgramGroupDesc desc = {};

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createCallableProgramGroup(
        Module module_DC, const char* entryFunctionNameDC,
        Module module_CC, const char* entryFunctionNameCC,
        const PayloadType &payloadType) const {
        _Module* _module_DC = extract(module_DC);
        _Module* _module_CC = extract(module_CC);
        m->throwRuntimeError((_module_DC != nullptr) == (entryFunctionNameDC != nullptr),
                             "Either of DC module or entry function name is not provided.");
        m->throwRuntimeError((_module_CC != nullptr) == (entryFunctionNameCC != nullptr),
                             "Either of CC module or entry function name is not provided.");
        m->throwRuntimeError(entryFunctionNameDC || entryFunctionNameCC,
                             "Either of CC/DC entry function name must be provided.");
        if (_module_DC)
            m->throwRuntimeError(_module_DC->getPipeline() == m,
                                 "Pipeline mismatch for the given DC module %s.",
                                 _module_DC->getName().c_str());
        if (_module_CC)
            m->throwRuntimeError(_module_CC->getPipeline() == m,
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

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    void Pipeline::link(uint32_t maxTraceDepth, OptixCompileDebugLevel debugLevel) const {
        m->throwRuntimeError(!m->pipelineLinked, "This pipeline has been already linked.");

        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = maxTraceDepth;
        pipelineLinkOptions.debugLevel = debugLevel;

        std::vector<OptixProgramGroup> groups;
        groups.resize(m->programGroups.size());
        std::copy(m->programGroups.cbegin(), m->programGroups.cend(), groups.begin());

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(m->getRawContext(),
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

    void Pipeline::setRayGenerationProgram(ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        m->throwRuntimeError(_program, "Invalid program %p.", _program);
        m->throwRuntimeError(_program->getPipeline() == m, "Pipeline mismatch for the given program %s.",
                             _program->getName().c_str());

        m->rayGenProgram = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setExceptionProgram(ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        m->throwRuntimeError(_program, "Invalid program %p.", _program);
        m->throwRuntimeError(_program->getPipeline() == m, "Pipeline mismatch for the given program %s.",
                             _program->getName().c_str());

        m->exceptionProgram = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setMissProgram(uint32_t rayType, ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        m->throwRuntimeError(rayType < m->numMissRayTypes, "Invalid ray type.");
        m->throwRuntimeError(_program, "Invalid program %p.", _program);
        m->throwRuntimeError(_program->getPipeline() == m, "Pipeline mismatch for the given program %s.",
                             _program->getName().c_str());

        m->missPrograms[rayType] = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setCallableProgram(uint32_t index, ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        m->throwRuntimeError(index < m->numCallablePrograms, "Invalid callable program index.");
        m->throwRuntimeError(_program, "Invalid program %p.", _program);
        m->throwRuntimeError(_program->getPipeline() == m, "Pipeline mismatch for the given program group %s.",
                             _program->getName().c_str());

        m->callablePrograms[index] = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setShaderBindingTable(const BufferView &shaderBindingTable, void* hostMem) const {
        m->throwRuntimeError(shaderBindingTable.sizeInBytes() >= m->sbtSize,
                             "Hit group shader binding table size is not enough.");
        m->throwRuntimeError(hostMem, "Host-side SBT counterpart must be provided.");
        m->sbt = shaderBindingTable;
        m->sbtHostMem = hostMem;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setScene(const Scene &scene) const {
        m->scene = extract(scene);
        m->hitGroupSbt = BufferView();
        m->hitGroupSbtIsUpToDate = false;
    }

    void Pipeline::setHitGroupShaderBindingTable(const BufferView &shaderBindingTable, void* hostMem) const {
        m->throwRuntimeError(hostMem, "Host-side hit group SBT counterpart must be provided.");
        m->hitGroupSbt = shaderBindingTable;
        m->hitGroupSbtHostMem = hostMem;
        m->hitGroupSbtIsUpToDate = false;
    }

    void Pipeline::markHitGroupShaderBindingTableDirty() const {
        m->hitGroupSbtIsUpToDate = false;
    }

    void Pipeline::setStackSize(uint32_t directCallableStackSizeFromTraversal,
                                uint32_t directCallableStackSizeFromState,
                                uint32_t continuationStackSize,
                                uint32_t maxTraversableGraphDepth) const {
        if (m->pipelineCompileOptions.traversableGraphFlags & OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING)
            maxTraversableGraphDepth = 2;
        else if (m->pipelineCompileOptions.traversableGraphFlags == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS)
            maxTraversableGraphDepth = 1;
        OPTIX_CHECK(optixPipelineSetStackSize(m->rawPipeline,
                                              directCallableStackSizeFromTraversal,
                                              directCallableStackSizeFromState,
                                              continuationStackSize,
                                              maxTraversableGraphDepth));
    }

    void Pipeline::launch(CUstream stream, CUdeviceptr plpOnDevice, uint32_t dimX, uint32_t dimY, uint32_t dimZ) const {
        m->throwRuntimeError(m->sbtLayoutIsUpToDate, "Shader binding table layout is outdated.");
        m->throwRuntimeError(m->sbt.isValid(), "Shader binding table is not set.");
        m->throwRuntimeError(m->sbt.sizeInBytes() >= m->sbtSize, "Shader binding table size is not enough.");
        m->throwRuntimeError(m->scene, "Scene is not set.");
        bool hasMotionAS;
        m->throwRuntimeError(m->scene->isReady(&hasMotionAS), "Scene is not ready.");
        m->throwRuntimeError(m->pipelineCompileOptions.usesMotionBlur || !hasMotionAS,
                             "Scene has a motion AS but the pipeline has not been configured for motion.");
        m->throwRuntimeError(m->hitGroupSbt.isValid(), "Hitgroup shader binding table is not set.");
        m->throwRuntimeError(m->pipelineLinked, "Pipeline has not been linked yet.");

        m->setupShaderBindingTable(stream);

        OPTIX_CHECK(optixLaunch(m->rawPipeline, stream, plpOnDevice, m->sizeOfPipelineLaunchParams,
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



    void ProgramGroup::destroy() {
        if (m) {
            m->pipeline->destroyProgram(m->rawGroup);
            delete m;
        }
        m = nullptr;
    }

    void ProgramGroup::getStackSize(OptixStackSizes* sizes) const {
        OPTIX_CHECK(optixProgramGroupGetStackSize(m->rawGroup, sizes));
    }



    void Denoiser::Priv::invoke(CUstream stream,
                                bool denoiseAlpha, CUdeviceptr hdrIntensity, CUdeviceptr hdrAverageColor, float blendFactor,
                                const BufferView &noisyBeauty, OptixPixelFormat beautyFormat,
                                const BufferView* noisyAovs, OptixPixelFormat* aovFormats, uint32_t numAovs,
                                const BufferView &albedo, OptixPixelFormat albedoFormat,
                                const BufferView &normal, OptixPixelFormat normalFormat,
                                const BufferView &flow, OptixPixelFormat flowFormat,
                                const BufferView &previousDenoisedBeauty,
                                const BufferView* previousDenoisedAovs,
                                const BufferView &denoisedBeauty,
                                const BufferView* denoisedAovs,
                                const DenoisingTask &task) const {
        throwRuntimeError(stateIsReady, "You need to call setupState() before invoke.");
        throwRuntimeError(noisyBeauty.isValid(), "Input noisy beauty buffer must be provided.");
        throwRuntimeError(denoisedBeauty.isValid(), "Denoised beauty buffer must be provided.");
        throwRuntimeError(numAovs == 0 || (noisyAovs && denoisedAovs), "Both of noisy/denoised AOV buffers must be provided.");
        for (uint32_t i = 0; i < numAovs; ++i) {
            throwRuntimeError(noisyAovs[i].isValid() && denoisedAovs[i].isValid(),
                              "Either of AOV %u input/output buffer is invalid.", i);
        }
        if (guideAlbedo)
            throwRuntimeError(albedo.isValid(), "Denoiser requires albedo buffer.");
        if (guideNormal)
            throwRuntimeError(normal.isValid(), "Denoiser requires normal buffer.");
        if (modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL ||
            modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV)
            throwRuntimeError(flow.isValid() && previousDenoisedBeauty.isValid(),
                              "Denoiser requires flow buffer and the previous denoised beauty buffer.");
        OptixDenoiserParams params = {};
        params.denoiseAlpha = denoiseAlpha;
        params.hdrIntensity = hdrIntensity;
        params.hdrAverageColor = hdrAverageColor;
        params.blendFactor = blendFactor;

        _DenoisingTask _task(task);

        const auto setupInputLayer = [&]
        (OptixPixelFormat format, CUdeviceptr baseAddress, OptixImage2D* layer) {
            uint32_t pixelStride = getPixelSize(format);
            *layer = {};
            layer->rowStrideInBytes = imageWidth * pixelStride;
            layer->pixelStrideInBytes = pixelStride;
            uint32_t addressOffset = _task.inputOffsetY * layer->rowStrideInBytes + _task.inputOffsetX * pixelStride;
            layer->data = baseAddress + addressOffset;
            layer->width = maxInputWidth;
            layer->height = maxInputHeight;
            layer->format = format;
        };
        const auto setupOutputLayer = [&]
        (OptixPixelFormat format, CUdeviceptr baseAddress, OptixImage2D* layer) {
            uint32_t pixelStride = getPixelSize(format);
            *layer = {};
            layer->rowStrideInBytes = imageWidth * pixelStride;
            layer->pixelStrideInBytes = pixelStride;
            uint32_t addressOffset = _task.outputOffsetY * layer->rowStrideInBytes + _task.outputOffsetX * pixelStride;
            layer->data = baseAddress + addressOffset;
            layer->width = _task.outputWidth;
            layer->height = _task.outputHeight;
            layer->format = format;
        };

        // TODO: 入出力画像のrowStrideを指定できるようにする。

        OptixDenoiserGuideLayer guideLayer = {};
        if (guideAlbedo)
            setupInputLayer(albedoFormat, albedo.getCUdeviceptr(), &guideLayer.albedo);
        if (guideNormal)
            setupInputLayer(normalFormat, normal.getCUdeviceptr(), &guideLayer.normal);
        if (modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL ||
            modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV)
            setupInputLayer(flowFormat, flow.getCUdeviceptr(), &guideLayer.flow);

        struct LayerInfo {
            BufferView input;
            BufferView output;
            BufferView previousOuput;
            OptixPixelFormat format;
        };
        std::vector<LayerInfo> layerInfos(1 + numAovs);
        layerInfos[0] = LayerInfo{ noisyBeauty, denoisedBeauty, previousDenoisedBeauty, beautyFormat };
        for (uint32_t i = 0; i < numAovs; ++i)
            layerInfos[i + 1] = LayerInfo{ noisyAovs[i], denoisedAovs[i], previousDenoisedAovs[i], aovFormats[i] };

        std::vector<OptixDenoiserLayer> denoiserLayers(1 + numAovs);
        for (uint32_t layerIdx = 0; layerIdx < 1 + numAovs; ++layerIdx) {
            const LayerInfo &layerInfo = layerInfos[layerIdx];
            OptixDenoiserLayer &denoiserLayer = denoiserLayers[layerIdx];
            std::memset(&denoiserLayer, 0, sizeof(denoiserLayer));

            setupInputLayer(layerInfo.format, layerInfo.input.getCUdeviceptr(), &denoiserLayer.input);
            if (modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL ||
                modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV)
                setupOutputLayer(layerInfo.format, layerInfo.previousOuput.getCUdeviceptr(), &denoiserLayer.previousOutput);
            setupOutputLayer(layerInfo.format, layerInfo.output.getCUdeviceptr(), &denoiserLayer.output);
        }

        int32_t offsetX = _task.outputOffsetX - _task.inputOffsetX;
        int32_t offsetY = _task.outputOffsetY - _task.inputOffsetY;
        OPTIX_CHECK(optixDenoiserInvoke(rawDenoiser, stream,
                                        &params,
                                        stateBuffer.getCUdeviceptr(), stateBuffer.sizeInBytes(),
                                        &guideLayer,
                                        denoiserLayers.data(), 1 + numAovs,
                                        offsetX, offsetY,
                                        scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes()));
    }

    void Denoiser::destroy() {
        if (m)
            delete m;
        m = nullptr;
    }

    void Denoiser::prepare(uint32_t imageWidth, uint32_t imageHeight, uint32_t tileWidth, uint32_t tileHeight,
                           size_t* stateBufferSize, size_t* scratchBufferSize, size_t* scratchBufferSizeForComputeIntensity,
                           uint32_t* numTasks) const {
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
        OptixDenoiserSizes sizes;
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(m->rawDenoiser, tileWidth, tileHeight, &sizes));
        m->stateSize = sizes.stateSizeInBytes;
        m->scratchSize = m->useTiling ?
            sizes.withOverlapScratchSizeInBytes : sizes.withoutOverlapScratchSizeInBytes;
        if (m->modelKind == OPTIX_DENOISER_MODEL_KIND_AOV ||
            m->modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV)
            m->scratchSizeForComputeAverageColor = sizeof(int32_t) * (3 + 3 * m->imageWidth * m->imageHeight);
        else
            m->scratchSizeForComputeIntensity = sizeof(int32_t) * (2 + m->imageWidth * m->imageHeight);
        m->overlapWidth = sizes.overlapWindowSizeInPixels;
        m->maxInputWidth = std::min(tileWidth + 2 * m->overlapWidth, imageWidth);
        m->maxInputHeight = std::min(tileHeight + 2 * m->overlapWidth, imageHeight);

        *stateBufferSize = m->stateSize;
        *scratchBufferSize = m->scratchSize;
        *scratchBufferSizeForComputeIntensity = m->scratchSizeForComputeIntensity;

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
            if (inputOffsetY + m->maxInputHeight > m->imageHeight)
                inputOffsetY = m->imageHeight - m->maxInputHeight;

            for (int32_t outputOffsetX = 0; outputOffsetX < static_cast<int32_t>(m->imageWidth);) {
                int32_t outputWidth = m->tileWidth;
                if (outputOffsetX == 0)
                    outputWidth += m->overlapWidth;
                if (outputOffsetX + outputWidth > static_cast<int32_t>(m->imageWidth))
                    outputWidth = m->imageWidth - outputOffsetX;

                int32_t inputOffsetX = std::max(outputOffsetX - m->overlapWidth, 0);
                if (inputOffsetX + m->maxInputWidth > m->imageWidth)
                    inputOffsetX = m->imageWidth - m->maxInputWidth;

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
    }

    void Denoiser::setupState(CUstream stream, const BufferView &stateBuffer, const BufferView &scratchBuffer) const {
        m->throwRuntimeError(m->imageSizeSet, "Call prepare() before this function.");
        m->throwRuntimeError(stateBuffer.sizeInBytes() >= m->stateSize,
                             "Size of the given state buffer is not enough.");
        m->throwRuntimeError(scratchBuffer.sizeInBytes() >= m->scratchSize,
                             "Size of the given scratch buffer is not enough.");
        uint32_t maxInputWidth = m->useTiling ? (m->tileWidth + 2 * m->overlapWidth) : m->imageWidth;
        uint32_t maxInputHeight = m->useTiling ? (m->tileHeight + 2 * m->overlapWidth) : m->imageHeight;
        OPTIX_CHECK(optixDenoiserSetup(m->rawDenoiser, stream,
                                       maxInputWidth, maxInputHeight,
                                       stateBuffer.getCUdeviceptr(), stateBuffer.sizeInBytes(),
                                       scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes()));

        m->stateBuffer = stateBuffer;
        m->scratchBuffer = scratchBuffer;
        m->stateIsReady = true;
    }

    void Denoiser::computeIntensity(CUstream stream,
                                    const BufferView &noisyBeauty, OptixPixelFormat beautyFormat,
                                    const BufferView &scratchBuffer, CUdeviceptr outputIntensity) const {
        m->throwRuntimeError(scratchBuffer.sizeInBytes() >= m->scratchSizeForComputeIntensity,
                             "Size of the given scratch buffer is not enough.");

        OptixImage2D colorLayer = {};
        colorLayer.data = noisyBeauty.getCUdeviceptr();
        colorLayer.width = m->imageWidth;
        colorLayer.height = m->imageHeight;
        colorLayer.format = beautyFormat;
        colorLayer.pixelStrideInBytes = getPixelSize(beautyFormat);
        colorLayer.rowStrideInBytes = colorLayer.pixelStrideInBytes * m->imageWidth;

        OPTIX_CHECK(optixDenoiserComputeIntensity(
            m->rawDenoiser, stream,
            &colorLayer, outputIntensity,
            scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes()));
    }

    void Denoiser::computeAverageColor(CUstream stream,
                                       const BufferView &noisyBeauty, OptixPixelFormat beautyFormat,
                                       const BufferView &scratchBuffer, CUdeviceptr outputAverageColor) const {
        m->throwRuntimeError(scratchBuffer.sizeInBytes() >= m->scratchSizeForComputeAverageColor,
                             "Size of the given scratch buffer is not enough.");

        OptixImage2D colorLayer = {};
        colorLayer.data = noisyBeauty.getCUdeviceptr();
        colorLayer.width = m->imageWidth;
        colorLayer.height = m->imageHeight;
        colorLayer.format = beautyFormat;
        colorLayer.pixelStrideInBytes = getPixelSize(beautyFormat);
        colorLayer.rowStrideInBytes = colorLayer.pixelStrideInBytes * m->imageWidth;

        OPTIX_CHECK(optixDenoiserComputeAverageColor(
            m->rawDenoiser, stream,
            &colorLayer, outputAverageColor,
            scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes()));
    }

    void Denoiser::invoke(CUstream stream,
                          bool denoiseAlpha, CUdeviceptr hdrIntensity, float blendFactor,
                          const BufferView &noisyBeauty, OptixPixelFormat beautyFormat,
                          const BufferView &albedo, OptixPixelFormat albedoFormat,
                          const BufferView &normal, OptixPixelFormat normalFormat,
                          const BufferView &flow, OptixPixelFormat flowFormat,
                          const BufferView &previousDenoisedBeauty,
                          const BufferView &denoisedBeauty,
                          const DenoisingTask &task) const {
        m->invoke(stream,
                  denoiseAlpha, hdrIntensity, 0, blendFactor,
                  noisyBeauty, beautyFormat, nullptr, nullptr, 0,
                  albedo, albedoFormat,
                  normal, normalFormat,
                  flow, flowFormat,
                  previousDenoisedBeauty, nullptr,
                  denoisedBeauty, nullptr,
                  task);
    }

    void Denoiser::invoke(CUstream stream,
                          bool denoiseAlpha, CUdeviceptr hdrAverageColor, float blendFactor,
                          const BufferView &noisyBeauty, OptixPixelFormat beautyFormat,
                          const BufferView* noisyAovs, OptixPixelFormat* aovFormats, uint32_t numAovs,
                          const BufferView &albedo, OptixPixelFormat albedoFormat,
                          const BufferView &normal, OptixPixelFormat normalFormat,
                          const BufferView &flow, OptixPixelFormat flowFormat,
                          const BufferView &previousDenoisedBeauty, const BufferView* previousDenoisedAovs,
                          const BufferView &denoisedBeauty, const BufferView* denoisedAovs,
                          const DenoisingTask &task) const {
        m->invoke(stream,
                  denoiseAlpha, 0, hdrAverageColor, blendFactor,
                  noisyBeauty, beautyFormat, noisyAovs, aovFormats, numAovs,
                  albedo, albedoFormat,
                  normal, normalFormat,
                  flow, flowFormat,
                  previousDenoisedBeauty, previousDenoisedAovs,
                  denoisedBeauty, denoisedAovs,
                  task);
    }
}
