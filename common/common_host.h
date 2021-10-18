#pragma once

#include "common_shared.h"

#include <fstream>
#include <sstream>
#include <array>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <random>
#include <filesystem>
#include <functional>
#include <thread>
#include <chrono>
#include <variant>

#include "stopwatch.h"

#if 1
#   define hpprintf(fmt, ...) do { devPrintf(fmt, ##__VA_ARGS__); printf(fmt, ##__VA_ARGS__); } while (0)
#else
#   define hpprintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif

template <typename T, typename Deleter, typename ...ArgTypes>
std::shared_ptr<T> make_shared_with_deleter(const Deleter &deleter, ArgTypes&&... args) {
    return std::shared_ptr<T>(new T(std::forward<ArgTypes>(args)...),
                              deleter);
}

std::filesystem::path getExecutableDirectory();

std::string readTxtFile(const std::filesystem::path& filepath);



template <typename RealType>
class DiscreteDistribution1DTemplate {
    cudau::TypedBuffer<RealType> m_PMF;
#if defined(USE_WALKER_ALIAS_METHOD)
    cudau::TypedBuffer<shared::AliasTableEntry<RealType>> m_aliasTable;
    cudau::TypedBuffer<shared::AliasValueMap<RealType>> m_valueMaps;
#else
    cudau::TypedBuffer<RealType> m_CDF;
#endif
    RealType m_integral;
    uint32_t m_numValues;

public:
    void initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues);
    void finalize() {
#if defined(USE_WALKER_ALIAS_METHOD)
        if (m_valueMaps.isInitialized() && m_aliasTable.isInitialized() && m_PMF.isInitialized()) {
            m_valueMaps.finalize();
            m_aliasTable.finalize();
            m_PMF.finalize();
        }
#else
        if (m_CDF.isInitialized() && m_PMF.isInitialized()) {
            m_CDF.finalize();
            m_PMF.finalize();
        }
#endif
    }

    DiscreteDistribution1DTemplate &operator=(DiscreteDistribution1DTemplate &&v) {
        m_PMF = std::move(v.m_PMF);
#if defined(USE_WALKER_ALIAS_METHOD)
        m_aliasTable = std::move(v.m_aliasTable);
        m_valueMaps = std::move(v.m_valueMaps);
#else
        m_CDF = std::move(v.m_CDF);
#endif
        m_integral = v.m_integral;
        m_numValues = v.m_numValues;
        return *this;
    }

    RealType getIntengral() const {
        return m_integral;
    }

    void getDeviceType(shared::DiscreteDistribution1DTemplate<RealType>* instance) const {
#if defined(USE_WALKER_ALIAS_METHOD)
        if (m_PMF.isInitialized() && m_aliasTable.isInitialized() && m_valueMaps.isInitialized())
            new (instance) shared::DiscreteDistribution1DTemplate<RealType>(
                m_PMF.getDevicePointer(), m_aliasTable.getDevicePointer(), m_valueMaps.getDevicePointer(),
                m_integral, m_numValues);
#else
        if (m_PMF.isInitialized() && m_CDF.isInitialized())
            new (instance) shared::DiscreteDistribution1DTemplate<RealType>(
                m_PMF.getDevicePointer(), m_CDF.getDevicePointer(), m_integral, m_numValues);
#endif
    }
};



template <typename RealType>
class RegularConstantContinuousDistribution1DTemplate {
    cudau::TypedBuffer<RealType> m_PDF;
#if defined(USE_WALKER_ALIAS_METHOD)
    cudau::TypedBuffer<shared::AliasTableEntry<RealType>> m_aliasTable;
    cudau::TypedBuffer<shared::AliasValueMap<RealType>> m_valueMaps;
#else
    cudau::TypedBuffer<RealType> m_CDF;
#endif
    RealType m_integral;
    uint32_t m_numValues;

public:
    void initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues);
    void finalize(CUcontext cuContext) {
#if defined(USE_WALKER_ALIAS_METHOD)
        if (m_valueMaps.isInitialized() && m_aliasTable.isInitialized() && m_PDF.isInitialized()) {
            m_valueMaps.finalize();
            m_aliasTable.finalize();
            m_PDF.finalize();
        }
#else
        if (m_CDF.isInitialized() && m_PDF.isInitialized()) {
            m_CDF.finalize();
            m_PDF.finalize();
        }
#endif
    }

    RegularConstantContinuousDistribution1DTemplate &operator=(RegularConstantContinuousDistribution1DTemplate &&v) {
        m_PDF = std::move(v.m_PDF);
#if defined(USE_WALKER_ALIAS_METHOD)
        m_aliasTable = std::move(v.m_aliasTable);
        m_valueMaps = std::move(v.m_valueMaps);
#else
        m_CDF = std::move(v.m_CDF);
#endif
        m_integral = v.m_integral;
        m_numValues = v.m_numValues;
        return *this;
    }

    RealType getIntegral() const { return m_integral; }
    uint32_t getNumValues() const { return m_numValues; }

    void getDeviceType(shared::RegularConstantContinuousDistribution1DTemplate<RealType>* instance) const {
#if defined(USE_WALKER_ALIAS_METHOD)
        new (instance) shared::RegularConstantContinuousDistribution1DTemplate<RealType>(
            m_PDF.getDevicePointer(), m_aliasTable.getDevicePointer(), m_valueMaps.getDevicePointer(),
            m_integral, m_numValues);
#else
        new (instance) shared::RegularConstantContinuousDistribution1DTemplate<RealType>(
            m_PDF.getDevicePointer(), m_CDF.getDevicePointer(), m_integral, m_numValues);
#endif
    }
};



template <typename RealType>
class RegularConstantContinuousDistribution2DTemplate {
    cudau::TypedBuffer<shared::RegularConstantContinuousDistribution1DTemplate<RealType>> m_raw1DDists;
    RegularConstantContinuousDistribution1DTemplate<RealType>* m_1DDists;
    RegularConstantContinuousDistribution1DTemplate<RealType> m_top1DDist;

public:
    RegularConstantContinuousDistribution2DTemplate() : m_1DDists(nullptr) {}

    RegularConstantContinuousDistribution2DTemplate &operator=(RegularConstantContinuousDistribution2DTemplate &&v) {
        m_raw1DDists = std::move(v.m_raw1DDists);
        m_1DDists = std::move(v.m_1DDists);
        m_top1DDist = std::move(v.m_top1DDist);
        return *this;
    }

    void initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numD1, size_t numD2);
    void finalize(CUcontext cuContext) {
        m_top1DDist.finalize(cuContext);

        for (int i = m_top1DDist.getNumValues() - 1; i >= 0; --i) {
            m_1DDists[i].finalize(cuContext);
        }

        m_raw1DDists.finalize();
        delete[] m_1DDists;
        m_1DDists = nullptr;
    }

    bool isInitialized() const { return m_1DDists != nullptr; }

    void getDeviceType(shared::RegularConstantContinuousDistribution2DTemplate<RealType>* instance) const {
        shared::RegularConstantContinuousDistribution1DTemplate<RealType> top1DDist;
        m_top1DDist.getDeviceType(&top1DDist);
        new (instance) shared::RegularConstantContinuousDistribution2DTemplate<RealType>(
            m_raw1DDists.getDevicePointer(), top1DDist);
    }
};



using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;
using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float>;
using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float>;



struct MovingAverageTime {
    float values[60];
    uint32_t index;
    uint32_t numValidValues;
    MovingAverageTime() : index(0), numValidValues(0) {}
    void append(float value) {
        values[index] = value;
        index = (index + 1) % lengthof(values);
        numValidValues = std::min<uint32_t>(numValidValues + 1, static_cast<uint32_t>(lengthof(values)));
    }
    float getAverage() const {
        float sum = 0.0f;
        for (uint32_t i = 0; i < numValidValues; ++i)
            sum += values[(index - 1 - i + lengthof(values)) % lengthof(values)];
        return numValidValues > 0 ? sum / numValidValues : 0.0f;
    }
};



class SlotFinder {
    uint32_t m_numLayers;
    uint32_t m_numLowestFlagBins;
    uint32_t m_numTotalCompiledFlagBins;
    uint32_t* m_flagBins;
    uint32_t* m_offsetsToOR_AND;
    uint32_t* m_numUsedFlagsUnderBinList;
    uint32_t* m_offsetsToNumUsedFlags;
    uint32_t* m_numFlagsInLayerList;

    SlotFinder(const SlotFinder &) = delete;
    SlotFinder &operator=(const SlotFinder &) = delete;

    void aggregate();

    uint32_t getNumLayers() const {
        return m_numLayers;
    }

    const uint32_t* getOffsetsToOR_AND() const {
        return m_offsetsToOR_AND;
    }

    const uint32_t* getOffsetsToNumUsedFlags() const {
        return m_offsetsToNumUsedFlags;
    }

    const uint32_t* getNumFlagsInLayerList() const {
        return m_numFlagsInLayerList;
    }

public:
    static constexpr uint32_t InvalidSlotIndex = 0xFFFFFFFF;

    SlotFinder() :
        m_numLayers(0), m_numLowestFlagBins(0), m_numTotalCompiledFlagBins(0),
        m_flagBins(nullptr), m_offsetsToOR_AND(nullptr),
        m_numUsedFlagsUnderBinList(nullptr), m_offsetsToNumUsedFlags(nullptr),
        m_numFlagsInLayerList(nullptr) {
    }
    ~SlotFinder() {
    }

    void initialize(uint32_t numSlots);

    void finalize();

    SlotFinder &operator=(SlotFinder &&inst) {
        finalize();

        m_numLayers = inst.m_numLayers;
        m_numLowestFlagBins = inst.m_numLowestFlagBins;
        m_numTotalCompiledFlagBins = inst.m_numTotalCompiledFlagBins;
        m_flagBins = inst.m_flagBins;
        m_offsetsToOR_AND = inst.m_offsetsToOR_AND;
        m_numUsedFlagsUnderBinList = inst.m_numUsedFlagsUnderBinList;
        m_offsetsToNumUsedFlags = inst.m_offsetsToNumUsedFlags;
        m_numFlagsInLayerList = inst.m_numFlagsInLayerList;
        inst.m_flagBins = nullptr;
        inst.m_offsetsToOR_AND = nullptr;
        inst.m_numUsedFlagsUnderBinList = nullptr;
        inst.m_offsetsToNumUsedFlags = nullptr;
        inst.m_numFlagsInLayerList = nullptr;

        return *this;
    }
    SlotFinder(SlotFinder &&inst) {
        *this = std::move(inst);
    }

    void resize(uint32_t numSlots);

    void reset() {
        std::fill_n(m_flagBins, m_numLowestFlagBins + m_numTotalCompiledFlagBins, 0);
        std::fill_n(m_numUsedFlagsUnderBinList, m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2, 0);
    }



    void setInUse(uint32_t slotIdx);

    void setNotInUse(uint32_t slotIdx);

    bool getUsage(uint32_t slotIdx) const {
        uint32_t binIdx = slotIdx / 32;
        uint32_t flagIdxInBin = slotIdx % 32;
        uint32_t flagBin = m_flagBins[binIdx];

        return (bool)((flagBin >> flagIdxInBin) & 0x1);
    }

    uint32_t getFirstAvailableSlot() const;

    uint32_t getFirstUsedSlot() const;

    uint32_t find_nthUsedSlot(uint32_t n) const;

    uint32_t getNumSlots() const {
        return m_numFlagsInLayerList[0];
    }

    uint32_t getNumUsed() const {
        return m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[m_numLayers - 1]];
    }

    void debugPrint() const;
};



void saveImage(const std::filesystem::path &filepath, uint32_t width, uint32_t height, const uint32_t* data);

void saveImage(const std::filesystem::path &filepath, uint32_t width, uint32_t height, const float4* data,
               bool applyToneMap, bool apply_sRGB_gammaCorrection);

void saveImage(const std::filesystem::path &filepath,
               uint32_t width, cudau::TypedBuffer<float4> &buffer,
               bool applyToneMap, bool apply_sRGB_gammaCorrection);

void saveImage(const std::filesystem::path &filepath,
               cudau::Array &array,
               bool applyToneMap, bool apply_sRGB_gammaCorrection);

template <uint32_t log2BlockWidth>
void saveImage(const std::filesystem::path &filepath,
               optixu::HostBlockBuffer2D<float4, log2BlockWidth> &buffer,
               bool applyToneMap, bool apply_sRGB_gammaCorrection) {
    uint32_t width = buffer.getWidth();
    uint32_t height = buffer.getHeight();
    auto data = new float4[width * height];
    buffer.map();
    for (int y = 0; y < static_cast<int32_t>(height); ++y) {
        for (int x = 0; x < static_cast<int32_t>(width); ++x) {
            data[y * width + x] = buffer(x, y);
        }
    }
    buffer.unmap();
    saveImage(filepath, width, height, data, applyToneMap, apply_sRGB_gammaCorrection);
    delete[] data;
}
