#include "common_host.h"

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include "../common/dds_loader.h"
#include "../../ext/stb_image.h"
#include "tinyexr.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../ext/stb_image_write.h"

void devPrintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char str[4096];
    vsnprintf_s(str, sizeof(str), _TRUNCATE, fmt, args);
    va_end(args);
    OutputDebugString(str);
}



std::filesystem::path getExecutableDirectory() {
    static std::filesystem::path ret;

    static bool done = false;
    if (!done) {
#if defined(HP_Platform_Windows_MSVC)
        TCHAR filepath[1024];
        auto length = GetModuleFileName(NULL, filepath, 1024);
        Assert(length > 0, "Failed to query the executable path.");

        ret = filepath;
#else
        static_assert(false, "Not implemented");
#endif
        ret = ret.remove_filename();

        done = true;
    }

    return ret;
}



std::string readTxtFile(const std::filesystem::path& filepath) {
    std::ifstream ifs;
    ifs.open(filepath, std::ios::in);
    if (ifs.fail())
        return "";

    std::stringstream sstream;
    sstream << ifs.rdbuf();

    return std::string(sstream.str());
}



template <typename RealType>
void DiscreteDistribution1DTemplate<RealType>::
initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues) {
    Assert(!m_isInitialized, "Already initialized!");
    m_numValues = static_cast<uint32_t>(numValues);
    if (m_numValues == 0) {
        m_integral = 0.0f;
        return;
    }

#if defined(USE_WALKER_ALIAS_METHOD)
    m_weights.initialize(cuContext, type, m_numValues);
    m_aliasTable.initialize(cuContext, type, m_numValues);
    m_valueMaps.initialize(cuContext, type, m_numValues);

    if (values == nullptr) {
        m_integral = 0.0f;
        m_isInitialized = true;
        return;
    }

    RealType* weights = m_weights.map();
    std::memcpy(weights, values, sizeof(RealType) * m_numValues);
    m_weights.unmap();

    CompensatedSum<RealType> sum(0);
    for (uint32_t i = 0; i < m_numValues; ++i)
        sum += values[i];
    RealType avgWeight = sum / m_numValues;
    m_integral = sum;

    struct IndexAndWeight {
        uint32_t index;
        RealType weight;
        IndexAndWeight() {}
        IndexAndWeight(uint32_t _index, RealType _weight) :
            index(_index), weight(_weight) {}
    };

    std::vector<IndexAndWeight> smallGroup;
    std::vector<IndexAndWeight> largeGroup;
    for (uint32_t i = 0; i < m_numValues; ++i) {
        RealType weight = values[i];
        IndexAndWeight entry(i, weight);
        if (weight <= avgWeight)
            smallGroup.push_back(entry);
        else
            largeGroup.push_back(entry);
    }
    shared::AliasTableEntry<RealType>* aliasTable = m_aliasTable.map();
    shared::AliasValueMap<RealType>* valueMaps = m_valueMaps.map();
    for (int i = 0; !smallGroup.empty() && !largeGroup.empty(); ++i) {
        IndexAndWeight smallPair = smallGroup.back();
        smallGroup.pop_back();
        IndexAndWeight &largePair = largeGroup.back();
        uint32_t secondIndex = largePair.index;
        RealType reducedWeight = (largePair.weight + smallPair.weight) - avgWeight;
        largePair.weight = reducedWeight;
        if (largePair.weight <= avgWeight) {
            smallGroup.push_back(largePair);
            largeGroup.pop_back();
        }
        RealType probToPickFirst = smallPair.weight / avgWeight;
        aliasTable[smallPair.index] = shared::AliasTableEntry<RealType>(secondIndex, probToPickFirst);

        shared::AliasValueMap<RealType> valueMap;
        RealType probToPickSecond = 1 - probToPickFirst;
        valueMap.scaleForFirst = avgWeight / values[smallPair.index];
        valueMap.scaleForSecond = avgWeight / values[secondIndex];
        valueMap.offsetForSecond = (reducedWeight - smallPair.weight) / values[secondIndex];
        valueMaps[smallPair.index] = valueMap;
    }
    while (!smallGroup.empty() || !largeGroup.empty()) {
        IndexAndWeight pair;
        if (!smallGroup.empty()) {
            pair = smallGroup.back();
            smallGroup.pop_back();
        }
        else {
            pair = largeGroup.back();
            largeGroup.pop_back();
        }
        aliasTable[pair.index] = shared::AliasTableEntry<RealType>(0xFFFFFFFF, 1.0f);

        shared::AliasValueMap<RealType> valueMap;
        valueMap.scaleForFirst = avgWeight / values[pair.index];
        valueMap.scaleForSecond = 0;
        valueMap.offsetForSecond = 0;
        valueMaps[pair.index] = valueMap;
    }
    m_valueMaps.unmap();
    m_aliasTable.unmap();
#else
    m_weights.initialize(cuContext, type, m_numValues);
    m_CDF.initialize(cuContext, type, m_numValues);

    if (values == nullptr) {
        m_integral = 0.0f;
        m_isInitialized = true;
        return;
    }

    RealType* weights = m_weights.map();
    std::memcpy(weights, values, sizeof(RealType) * m_numValues);
    m_weights.unmap();

    RealType* CDF = m_CDF.map();

    CompensatedSum<RealType> sum(0);
    for (uint32_t i = 0; i < m_numValues; ++i) {
        CDF[i] = sum;
        sum += values[i];
    }
    m_integral = sum;

    m_CDF.unmap();
#endif

    m_isInitialized = true;
}

template class DiscreteDistribution1DTemplate<float>;



template <typename RealType>
void RegularConstantContinuousDistribution1DTemplate<RealType>::
initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues) {
    Assert(!m_isInitialized, "Already initialized!");
    m_numValues = static_cast<uint32_t>(numValues);
#if defined(USE_WALKER_ALIAS_METHOD)
    m_PDF.initialize(cuContext, type, m_numValues);
    m_aliasTable.initialize(cuContext, type, m_numValues);
    m_valueMaps.initialize(cuContext, type, m_numValues);

    RealType* PDF = m_PDF.map();
    std::memcpy(PDF, values, sizeof(RealType) * m_numValues);

    CompensatedSum<RealType> sum(0);
    for (uint32_t i = 0; i < m_numValues; ++i)
        sum += values[i];
    RealType avgWeight = sum / m_numValues;
    m_integral = avgWeight;

    for (uint32_t i = 0; i < m_numValues; ++i)
        PDF[i] /= m_integral;
    m_PDF.unmap();

    struct IndexAndWeight {
        uint32_t index;
        RealType weight;
        IndexAndWeight() {}
        IndexAndWeight(uint32_t _index, RealType _weight) :
            index(_index), weight(_weight) {}
    };

    std::vector<IndexAndWeight> smallGroup;
    std::vector<IndexAndWeight> largeGroup;
    for (uint32_t i = 0; i < m_numValues; ++i) {
        RealType weight = values[i];
        IndexAndWeight entry(i, weight);
        if (weight <= avgWeight)
            smallGroup.push_back(entry);
        else
            largeGroup.push_back(entry);
    }

    shared::AliasTableEntry<RealType>* aliasTable = m_aliasTable.map();
    shared::AliasValueMap<RealType>* valueMaps = m_valueMaps.map();
    for (int i = 0; !smallGroup.empty() && !largeGroup.empty(); ++i) {
        IndexAndWeight smallPair = smallGroup.back();
        smallGroup.pop_back();
        IndexAndWeight &largePair = largeGroup.back();
        uint32_t secondIndex = largePair.index;
        RealType reducedWeight = (largePair.weight + smallPair.weight) - avgWeight;
        largePair.weight = reducedWeight;
        if (largePair.weight <= avgWeight) {
            smallGroup.push_back(largePair);
            largeGroup.pop_back();
        }
        RealType probToPickFirst = smallPair.weight / avgWeight;
        aliasTable[smallPair.index] = shared::AliasTableEntry<RealType>(secondIndex, probToPickFirst);

        shared::AliasValueMap<RealType> valueMap;
        RealType probToPickSecond = 1 - probToPickFirst;
        valueMap.scaleForFirst = avgWeight / values[smallPair.index];
        valueMap.scaleForSecond = avgWeight / values[secondIndex];
        valueMap.offsetForSecond = (reducedWeight - smallPair.weight) / values[secondIndex];
        valueMaps[smallPair.index] = valueMap;
    }
    while (!smallGroup.empty() || !largeGroup.empty()) {
        IndexAndWeight pair;
        if (!smallGroup.empty()) {
            pair = smallGroup.back();
            smallGroup.pop_back();
        }
        else {
            pair = largeGroup.back();
            largeGroup.pop_back();
        }
        aliasTable[pair.index] = shared::AliasTableEntry<RealType>(0xFFFFFFFF, 1.0f);

        shared::AliasValueMap<RealType> valueMap;
        valueMap.scaleForFirst = avgWeight / values[pair.index];
        valueMap.scaleForSecond = 0;
        valueMap.offsetForSecond = 0;
        valueMaps[pair.index] = valueMap;
    }
    m_valueMaps.unmap();
    m_aliasTable.unmap();
#else
    m_PDF.initialize(cuContext, type, m_numValues);
    m_CDF.initialize(cuContext, type, m_numValues + 1);

    RealType* PDF = m_PDF.map();
    RealType* CDF = m_CDF.map();
    std::memcpy(PDF, values, sizeof(RealType) * m_numValues);

    CompensatedSum<RealType> sum{ 0 };
    for (uint32_t i = 0; i < m_numValues; ++i) {
        CDF[i] = sum;
        sum += PDF[i] / m_numValues;
    }
    m_integral = sum;
    for (uint32_t i = 0; i < m_numValues; ++i) {
        PDF[i] /= m_integral;
        CDF[i] /= m_integral;
    }
    CDF[m_numValues] = 1.0f;

    m_CDF.unmap();
    m_PDF.unmap();
#endif

    m_isInitialized = true;
}

template class RegularConstantContinuousDistribution1DTemplate<float>;



template <typename RealType>
void RegularConstantContinuousDistribution2DTemplate<RealType>::
initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numD1, size_t numD2) {
    Assert(!m_isInitialized, "Already initialized!");
    m_1DDists = new RegularConstantContinuousDistribution1DTemplate<RealType>[numD2];
    m_raw1DDists.initialize(cuContext, type, static_cast<uint32_t>(numD2));

    shared::RegularConstantContinuousDistribution1DTemplate<RealType>* rawDists = m_raw1DDists.map();

    // JP: まず各行に関するDistribution1Dを作成する。
    // EN: First, create Distribution1D's for every rows.
    CompensatedSum<RealType> sum(0);
    RealType* integrals = new RealType[numD2];
    for (uint32_t i = 0; i < numD2; ++i) {
        RegularConstantContinuousDistribution1DTemplate<RealType> &dist = m_1DDists[i];
        dist.initialize(cuContext, type, values + i * numD1, numD1);
        dist.getDeviceType(&rawDists[i]);
        integrals[i] = dist.getIntegral();
        sum += integrals[i];
    }

    // JP: 各行の積分値を用いてDistribution1Dを作成する。
    // EN: create a Distribution1D using integral values of each row.
    m_top1DDist.initialize(cuContext, type, integrals, numD2);
    delete[] integrals;

    Assert(std::isfinite(m_top1DDist.getIntegral()), "invalid integral value.");

    m_raw1DDists.unmap();

    m_isInitialized = true;
}

template class RegularConstantContinuousDistribution2DTemplate<float>;



void ProbabilityTexture::initialize(CUcontext cuContext, size_t numValues) {
    Assert(!m_isInitialized, "Already initialized!");
    cudau::TextureSampler sampler;
    sampler.setXyFilterMode(cudau::TextureFilterMode::Point);
    sampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler.setReadMode(cudau::TextureReadMode::ElementType);

    uint2 dims = shared::computeProbabilityTextureDimentions(numValues);
    uint32_t numMipLevels = nextPowOf2Exponent(dims.x) + 1;
    m_cuArray.initialize2D(
        cuContext, cudau::ArrayElementType::Float32, 1,
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
        dims.x, dims.y, numMipLevels);
    m_cuTexObj = sampler.createTextureObject(m_cuArray);

    m_isInitialized = true;
}



void SlotFinder::initialize(uint32_t numSlots) {
    m_numLayers = 1;
    m_numLowestFlagBins = nextMultiplierForPowOf2(numSlots, 5);

    // e.g. factor 4
    // 0 | 1101 | 0011 | 1001 | 1011 | 0010 | 1010 | 0000 | 1011 | 1110 | 0101 | 111* | **** | **** | **** | **** | **** | 43 flags
    // OR bins:
    // 1 | 1      1      1      1    | 1      1      0      1    | 1      1      1      *    | *      *      *      *    | 11
    // 2 | 1                           1                           1                           *                         | 3
    // AND bins
    // 1 | 0      0      0      0    | 0      0      0      0    | 0      0      1      *    | *      *      *      *    | 11
    // 2 | 0                           0                           0                           *                         | 3
    //
    // numSlots: 43
    // numLowestFlagBins: 11
    // numLayers: 3
    //
    // Memory Order
    // LowestFlagBins (layer 0) | OR, AND Bins (layer 1) | ... | OR, AND Bins (layer n-1)
    // Offset Pair to OR, AND (layer 0) | ... | Offset Pair to OR, AND (layer n-1)
    // NumUsedFlags (layer 0) | ... | NumUsedFlags (layer n-1)
    // Offset to NumUsedFlags (layer 0) | ... | Offset to NumUsedFlags (layer n-1)
    // NumFlags (layer 0) | ... | NumFlags (layer n-1)

    uint32_t numFlagBinsInLayer = m_numLowestFlagBins;
    m_numTotalCompiledFlagBins = 0;
    while (numFlagBinsInLayer > 1) {
        ++m_numLayers;
        numFlagBinsInLayer = nextMultiplierForPowOf2(numFlagBinsInLayer, 5);
        m_numTotalCompiledFlagBins += 2 * numFlagBinsInLayer; // OR bins and AND bins
    }

    size_t memSize = sizeof(uint32_t) *
        ((m_numLowestFlagBins + m_numTotalCompiledFlagBins) +
         m_numLayers * 2 +
         (m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2) +
         m_numLayers +
         m_numLayers);
    void* mem = malloc(memSize);

    uintptr_t memHead = (uintptr_t)mem;
    m_flagBins = (uint32_t*)memHead;
    memHead += sizeof(uint32_t) * (m_numLowestFlagBins + m_numTotalCompiledFlagBins);

    m_offsetsToOR_AND = (uint32_t*)memHead;
    memHead += sizeof(uint32_t) * m_numLayers * 2;

    m_numUsedFlagsUnderBinList = (uint32_t*)memHead;
    memHead += sizeof(uint32_t) * (m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2);

    m_offsetsToNumUsedFlags = (uint32_t*)memHead;
    memHead += sizeof(uint32_t) * m_numLayers;

    m_numFlagsInLayerList = (uint32_t*)memHead;

    uint32_t layer = 0;
    uint32_t offsetToOR_AND = 0;
    uint32_t offsetToNumUsedFlags = 0;
    {
        m_numFlagsInLayerList[layer] = numSlots;

        numFlagBinsInLayer = nextMultiplierForPowOf2(numSlots, 5);

        m_offsetsToOR_AND[2 * layer + 0] = offsetToOR_AND;
        m_offsetsToOR_AND[2 * layer + 1] = offsetToOR_AND;
        m_offsetsToNumUsedFlags[layer] = offsetToNumUsedFlags;

        offsetToOR_AND += numFlagBinsInLayer;
        offsetToNumUsedFlags += numFlagBinsInLayer;
    }
    while (numFlagBinsInLayer > 1) {
        ++layer;
        m_numFlagsInLayerList[layer] = numFlagBinsInLayer;

        numFlagBinsInLayer = nextMultiplierForPowOf2(numFlagBinsInLayer, 5);

        m_offsetsToOR_AND[2 * layer + 0] = offsetToOR_AND;
        m_offsetsToOR_AND[2 * layer + 1] = offsetToOR_AND + numFlagBinsInLayer;
        m_offsetsToNumUsedFlags[layer] = offsetToNumUsedFlags;

        offsetToOR_AND += 2 * numFlagBinsInLayer;
        offsetToNumUsedFlags += numFlagBinsInLayer;
    }

    std::fill_n(m_flagBins, m_numLowestFlagBins + m_numTotalCompiledFlagBins, 0);
    std::fill_n(m_numUsedFlagsUnderBinList, m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2, 0);
}

void SlotFinder::finalize() {
    if (m_flagBins)
        free(m_flagBins);
    m_flagBins = nullptr;
}

void SlotFinder::aggregate() {
    uint32_t offsetToOR_last = m_offsetsToOR_AND[2 * 0 + 0];
    uint32_t offsetToAND_last = m_offsetsToOR_AND[2 * 0 + 1];
    uint32_t offsetToNumUsedFlags_last = m_offsetsToNumUsedFlags[0];
    for (int layer = 1; layer < static_cast<int32_t>(m_numLayers); ++layer) {
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
        uint32_t offsetToOR = m_offsetsToOR_AND[2 * layer + 0];
        uint32_t offsetToAND = m_offsetsToOR_AND[2 * layer + 1];
        uint32_t offsetToNumUsedFlags = m_offsetsToNumUsedFlags[layer];
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t &ORFlagBin = m_flagBins[offsetToOR + binIdx];
            uint32_t &ANDFlagBin = m_flagBins[offsetToAND + binIdx];
            uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[offsetToNumUsedFlags + binIdx];

            uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
            for (int bit = 0; bit < static_cast<int32_t>(numFlagsInBin); ++bit) {
                uint32_t lBinIdx = 32 * binIdx + bit;
                uint32_t lORFlagBin = m_flagBins[offsetToOR_last + lBinIdx];
                uint32_t lANDFlagBin = m_flagBins[offsetToAND_last + lBinIdx];
                uint32_t lNumFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer - 1] - 32 * lBinIdx);
                if (lORFlagBin != 0)
                    ORFlagBin |= 1 << bit;
                if (popcnt(lANDFlagBin) == lNumFlagsInBin)
                    ANDFlagBin |= 1 << bit;
                numUsedFlagsUnderBin += m_numUsedFlagsUnderBinList[offsetToNumUsedFlags_last + lBinIdx];
            }
        }

        offsetToOR_last = offsetToOR;
        offsetToAND_last = offsetToAND;
        offsetToNumUsedFlags_last = offsetToNumUsedFlags;
    }
}

void SlotFinder::resize(uint32_t numSlots) {
    if (numSlots == m_numFlagsInLayerList[0])
        return;

    SlotFinder newFinder;
    newFinder.initialize(numSlots);

    uint32_t numLowestFlagBins = std::min(m_numLowestFlagBins, newFinder.m_numLowestFlagBins);
    for (int binIdx = 0; binIdx < static_cast<int32_t>(numLowestFlagBins); ++binIdx) {
        uint32_t numFlagsInBin = std::min(32u, numSlots - 32 * binIdx);
        uint32_t mask = numFlagsInBin >= 32 ? 0xFFFFFFFF : ((1 << numFlagsInBin) - 1);
        uint32_t value = m_flagBins[0 + binIdx] & mask;
        newFinder.m_flagBins[0 + binIdx] = value;
        newFinder.m_numUsedFlagsUnderBinList[0 + binIdx] = popcnt(value);
    }

    newFinder.aggregate();

    *this = std::move(newFinder);
}

void SlotFinder::setInUse(uint32_t slotIdx) {
    if (getUsage(slotIdx))
        return;

    bool setANDFlag = false;
    uint32_t flagIdxInLayer = slotIdx;
    for (int layer = 0; layer < static_cast<int32_t>(m_numLayers); ++layer) {
        uint32_t binIdx = flagIdxInLayer / 32;
        uint32_t flagIdxInBin = flagIdxInLayer % 32;

        // JP: 最下層ではOR/ANDは同じ実体だがsetANDFlagが初期値falseであるので設定は1回きり。
        uint32_t &ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
        uint32_t &ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
        uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
        ORFlagBin |= (1 << flagIdxInBin);
        if (setANDFlag)
            ANDFlagBin |= (1 << flagIdxInBin);
        ++numUsedFlagsUnderBin;

        // JP: このビンに利用可能なスロットが無くなった場合は次のANDレイヤーもフラグを立てる。
        uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
        setANDFlag = popcnt(ANDFlagBin) == numFlagsInBin;

        flagIdxInLayer = binIdx;
    }
}

void SlotFinder::setNotInUse(uint32_t slotIdx) {
    if (!getUsage(slotIdx))
        return;

    bool resetORFlag = false;
    uint32_t flagIdxInLayer = slotIdx;
    for (int layer = 0; layer < static_cast<int32_t>(m_numLayers); ++layer) {
        uint32_t binIdx = flagIdxInLayer / 32;
        uint32_t flagIdxInBin = flagIdxInLayer % 32;

        // JP: 最下層ではOR/ANDは同じ実体だがresetORFlagが初期値falseであるので設定は1回きり。
        uint32_t &ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
        uint32_t &ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
        uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
        if (resetORFlag)
            ORFlagBin &= ~(1 << flagIdxInBin);
        ANDFlagBin &= ~(1 << flagIdxInBin);
        --numUsedFlagsUnderBin;

        // JP: このビンに使用中スロットが無くなった場合は次のORレイヤーのフラグを下げる。
        uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
        resetORFlag = ORFlagBin == 0;

        flagIdxInLayer = binIdx;
    }
}

uint32_t SlotFinder::getFirstAvailableSlot() const {
    uint32_t binIdx = 0;
    for (int layer = m_numLayers - 1; layer >= 0; --layer) {
        uint32_t ANDFlagBinOffset = m_offsetsToOR_AND[2 * layer + 1];
        uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
        uint32_t ANDFlagBin = m_flagBins[ANDFlagBinOffset + binIdx];

        if (popcnt(ANDFlagBin) != numFlagsInBin) {
            // JP: このビンに利用可能なスロットを発見。
            binIdx = tzcnt(~ANDFlagBin) + 32 * binIdx;
        }
        else {
            // JP: 利用可能なスロットが見つからなかった。
            return 0xFFFFFFFF;
        }
    }

    Assert(binIdx < m_numFlagsInLayerList[0], "Invalid value.");
    return binIdx;
}

uint32_t SlotFinder::getFirstUsedSlot() const {
    uint32_t binIdx = 0;
    for (int layer = m_numLayers - 1; layer >= 0; --layer) {
        uint32_t ORFlagBinOffset = m_offsetsToOR_AND[2 * layer + 0];
        uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
        uint32_t ORFlagBin = m_flagBins[ORFlagBinOffset + binIdx];

        if (ORFlagBin != 0) {
            // JP: このビンに使用中のスロットを発見。
            binIdx = tzcnt(ORFlagBin) + 32 * binIdx;
        }
        else {
            // JP: 使用中スロットが見つからなかった。
            return 0xFFFFFFFF;
        }
    }

    Assert(binIdx < m_numFlagsInLayerList[0], "Invalid value.");
    return binIdx;
}

uint32_t SlotFinder::find_nthUsedSlot(uint32_t n) const {
    if (n >= getNumUsed())
        return 0xFFFFFFFF;

    uint32_t startBinIdx = 0;
    uint32_t accNumUsed = 0;
    for (int layer = m_numLayers - 1; layer >= 0; --layer) {
        uint32_t numUsedFlagsOffset = m_offsetsToNumUsedFlags[layer];
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
        for (int binIdx = startBinIdx; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[numUsedFlagsOffset + binIdx];

            // JP: 現在のビンの配下にインデックスnの使用中スロットがある。
            if (accNumUsed + numUsedFlagsUnderBin > n) {
                startBinIdx = 32 * binIdx;
                if (layer == 0) {
                    uint32_t flagBin = m_flagBins[binIdx];
                    startBinIdx += nthSetBit(flagBin, n - accNumUsed);
                }
                break;
            }

            accNumUsed += numUsedFlagsUnderBin;
        }
    }

    Assert(startBinIdx < m_numFlagsInLayerList[0], "Invalid value.");
    return startBinIdx;
}

void SlotFinder::debugPrint() const {
    uint32_t numLowestFlagBins = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 5);
    hpprintf("----");
    for (int binIdx = 0; binIdx < static_cast<int32_t>(numLowestFlagBins); ++binIdx) {
        hpprintf("------------------------------------");
    }
    hpprintf("\n");
    for (int layer = m_numLayers - 1; layer > 0; --layer) {
        hpprintf("layer %u (%u):\n", layer, m_numFlagsInLayerList[layer]);
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
        hpprintf(" OR:");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
            for (int i = 0; i < 32; ++i) {
                if (i % 8 == 0)
                    hpprintf(" ");

                bool valid = binIdx * 32 + i < static_cast<int32_t>(m_numFlagsInLayerList[layer]);
                if (!valid)
                    continue;

                bool b = (ORFlagBin >> i) & 0x1;
                hpprintf("%c", b ? '|' : '_');
            }
        }
        hpprintf("\n");
        hpprintf("AND:");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
            for (int i = 0; i < 32; ++i) {
                if (i % 8 == 0)
                    hpprintf(" ");

                bool valid = binIdx * 32 + i < static_cast<int32_t>(m_numFlagsInLayerList[layer]);
                if (!valid)
                    continue;

                bool b = (ANDFlagBin >> i) & 0x1;
                hpprintf("%c", b ? '|' : '_');
            }
        }
        hpprintf("\n");
        hpprintf("    ");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
            hpprintf("                            %8u", numUsedFlagsUnderBin);
        }
        hpprintf("\n");
    }
    {
        hpprintf("layer 0 (%u):\n", m_numFlagsInLayerList[0]);
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 5);
        hpprintf("   :");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t ORFlagBin = m_flagBins[binIdx];
            for (int i = 0; i < 32; ++i) {
                if (i % 8 == 0)
                    hpprintf(" ");

                bool valid = binIdx * 32 + i < static_cast<int32_t>(m_numFlagsInLayerList[0]);
                if (!valid)
                    continue;

                bool b = (ORFlagBin >> i) & 0x1;
                hpprintf("%c", b ? '|' : '_');
            }
        }
        hpprintf("\n");
        hpprintf("    ");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[binIdx];
            hpprintf("                            %8u", numUsedFlagsUnderBin);
        }
        hpprintf("\n");
    }
}



struct FlattenedNode {
    Matrix4x4 transform;
    std::vector<uint32_t> meshIndices;
};

static void computeFlattenedNodes(const aiScene* scene, const Matrix4x4 &parentXfm, const aiNode* curNode,
                                  std::vector<FlattenedNode> &flattenedNodes) {
    aiMatrix4x4 curAiXfm = curNode->mTransformation;
    Matrix4x4 curXfm = Matrix4x4(float4(curAiXfm.a1, curAiXfm.a2, curAiXfm.a3, curAiXfm.a4),
                                 float4(curAiXfm.b1, curAiXfm.b2, curAiXfm.b3, curAiXfm.b4),
                                 float4(curAiXfm.c1, curAiXfm.c2, curAiXfm.c3, curAiXfm.c4),
                                 float4(curAiXfm.d1, curAiXfm.d2, curAiXfm.d3, curAiXfm.d4));
    FlattenedNode flattenedNode;
    flattenedNode.transform = parentXfm * transpose(curXfm);
    flattenedNode.meshIndices.resize(curNode->mNumMeshes);
    if (curNode->mNumMeshes > 0) {
        std::copy_n(curNode->mMeshes, curNode->mNumMeshes, flattenedNode.meshIndices.data());
        flattenedNodes.push_back(flattenedNode);
    }

    for (uint32_t cIdx = 0; cIdx < curNode->mNumChildren; ++cIdx)
        computeFlattenedNodes(scene, flattenedNode.transform, curNode->mChildren[cIdx], flattenedNodes);
}

static void translate(dds::Format ddsFormat, cudau::ArrayElementType* cudaType, bool* needsDegamma) {
    *needsDegamma = false;
    switch (ddsFormat) {
    case dds::Format::BC1_UNorm:
        *cudaType = cudau::ArrayElementType::BC1_UNorm;
        break;
    case dds::Format::BC1_UNorm_sRGB:
        *cudaType = cudau::ArrayElementType::BC1_UNorm;
        *needsDegamma = true;
        break;
    case dds::Format::BC2_UNorm:
        *cudaType = cudau::ArrayElementType::BC2_UNorm;
        break;
    case dds::Format::BC2_UNorm_sRGB:
        *cudaType = cudau::ArrayElementType::BC2_UNorm;
        *needsDegamma = true;
        break;
    case dds::Format::BC3_UNorm:
        *cudaType = cudau::ArrayElementType::BC3_UNorm;
        break;
    case dds::Format::BC3_UNorm_sRGB:
        *cudaType = cudau::ArrayElementType::BC3_UNorm;
        *needsDegamma = true;
        break;
    case dds::Format::BC4_UNorm:
        *cudaType = cudau::ArrayElementType::BC4_UNorm;
        break;
    case dds::Format::BC4_SNorm:
        *cudaType = cudau::ArrayElementType::BC4_SNorm;
        break;
    case dds::Format::BC5_UNorm:
        *cudaType = cudau::ArrayElementType::BC5_UNorm;
        break;
    case dds::Format::BC5_SNorm:
        *cudaType = cudau::ArrayElementType::BC5_SNorm;
        break;
    case dds::Format::BC6H_UF16:
        *cudaType = cudau::ArrayElementType::BC6H_UF16;
        break;
    case dds::Format::BC6H_SF16:
        *cudaType = cudau::ArrayElementType::BC6H_SF16;
        break;
    case dds::Format::BC7_UNorm:
        *cudaType = cudau::ArrayElementType::BC7_UNorm;
        break;
    case dds::Format::BC7_UNorm_sRGB:
        *cudaType = cudau::ArrayElementType::BC7_UNorm;
        *needsDegamma = true;
        break;
    default:
        break;
    }
};

enum class BumpMapTextureType {
    NormalMap = 0,
    NormalMap_BC,
    NormalMap_BC_2ch,
    HeightMap,
    HeightMap_BC,
};

struct TextureCacheKey {
    std::filesystem::path filePath;
    CUcontext cuContext;

    bool operator<(const TextureCacheKey &rKey) const {
        if (filePath < rKey.filePath)
            return true;
        else if (filePath > rKey.filePath)
            return false;
        if (cuContext < rKey.cuContext)
            return true;
        else if (cuContext > rKey.cuContext)
            return false;
        return false;
    }
};

struct Fx1ImmTextureCacheKey {
    float immValue;
    CUcontext cuContext;

    bool operator<(const Fx1ImmTextureCacheKey &rKey) const {
        if (immValue < rKey.immValue)
            return true;
        else if (immValue > rKey.immValue)
            return false;
        if (cuContext < rKey.cuContext)
            return true;
        else if (cuContext > rKey.cuContext)
            return false;
        return false;
    }
};

struct Fx3ImmTextureCacheKey {
    float3 immValue;
    CUcontext cuContext;

    bool operator<(const Fx3ImmTextureCacheKey &rKey) const {
        if (immValue.z < rKey.immValue.z)
            return true;
        else if (immValue.z > rKey.immValue.z)
            return false;
        if (immValue.y < rKey.immValue.y)
            return true;
        else if (immValue.y > rKey.immValue.y)
            return false;
        if (immValue.x < rKey.immValue.x)
            return true;
        else if (immValue.x > rKey.immValue.x)
            return false;
        if (cuContext < rKey.cuContext)
            return true;
        else if (cuContext > rKey.cuContext)
            return false;
        return false;
    }
};

struct Fx4ImmTextureCacheKey {
    float4 immValue;
    CUcontext cuContext;

    bool operator<(const Fx4ImmTextureCacheKey &rKey) const {
        if (immValue.w < rKey.immValue.w)
            return true;
        else if (immValue.w > rKey.immValue.w)
            return false;
        if (immValue.z < rKey.immValue.z)
            return true;
        else if (immValue.z > rKey.immValue.z)
            return false;
        if (immValue.y < rKey.immValue.y)
            return true;
        else if (immValue.y > rKey.immValue.y)
            return false;
        if (immValue.x < rKey.immValue.x)
            return true;
        else if (immValue.x > rKey.immValue.x)
            return false;
        if (cuContext < rKey.cuContext)
            return true;
        else if (cuContext > rKey.cuContext)
            return false;
        return false;
    }
};

struct TextureCacheValue {
    cudau::Array texture;
    bool needsDegamma;
    bool isHDR;
    BumpMapTextureType bumpMapType;
};

static std::map<TextureCacheKey, TextureCacheValue> s_textureCache;
static std::map<Fx1ImmTextureCacheKey, TextureCacheValue> s_Fx1ImmTextureCache;
static std::map<Fx3ImmTextureCacheKey, TextureCacheValue> s_Fx3ImmTextureCache;
static std::map<Fx4ImmTextureCacheKey, TextureCacheValue> s_Fx4ImmTextureCache;

void finalizeTextureCaches() {
    for (auto &it : s_textureCache)
        it.second.texture.finalize();
    for (auto &it : s_Fx1ImmTextureCache)
        it.second.texture.finalize();
    for (auto &it : s_Fx3ImmTextureCache)
        it.second.texture.finalize();
    for (auto &it : s_Fx4ImmTextureCache)
        it.second.texture.finalize();
}

static void createFx1ImmTexture(
    CUcontext cuContext,
    float immValue,
    bool isNormalized,
    const cudau::Array** texture) {
    Fx1ImmTextureCacheKey cacheKey;
    cacheKey.immValue = immValue;
    cacheKey.cuContext = cuContext;
    if (s_Fx1ImmTextureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_Fx1ImmTextureCache.at(cacheKey);
        *texture = &value.texture;
        return;
    }

    TextureCacheValue cacheValue;
    cacheValue.isHDR = !isNormalized;
    if (isNormalized) {
        uint8_t data = std::min<uint32_t>(255 * immValue, 255);
        cacheValue.texture.initialize2D(
            cuContext, cudau::ArrayElementType::UInt8, 1,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write<uint8_t>(reinterpret_cast<uint8_t*>(&data), 1);
    }
    else {
        cacheValue.texture.initialize2D(
            cuContext, cudau::ArrayElementType::Float32, 1,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write(&immValue, 1);
    }

    s_Fx1ImmTextureCache[cacheKey] = std::move(cacheValue);

    *texture = &s_Fx1ImmTextureCache.at(cacheKey).texture;
}

static void createFx3ImmTexture(
    CUcontext cuContext,
    const float3 &immValue,
    bool isNormalized,
    const cudau::Array** texture) {
    Fx3ImmTextureCacheKey cacheKey;
    cacheKey.immValue = immValue;
    cacheKey.cuContext = cuContext;
    if (s_Fx3ImmTextureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_Fx3ImmTextureCache.at(cacheKey);
        *texture = &value.texture;
        return;
    }

    TextureCacheValue cacheValue;
    cacheValue.isHDR = !isNormalized;
    if (isNormalized) {
        uint32_t data = ((std::min<uint32_t>(255 * immValue.x, 255) << 0) |
                         (std::min<uint32_t>(255 * immValue.y, 255) << 8) |
                         (std::min<uint32_t>(255 * immValue.z, 255) << 16) |
                         255 << 24);
        cacheValue.texture.initialize2D(
            cuContext, cudau::ArrayElementType::UInt8, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write<uint8_t>(reinterpret_cast<uint8_t*>(&data), 4);
    }
    else {
        float data[4] = {
            immValue.x, immValue.y, immValue.z, 1.0f
        };
        cacheValue.texture.initialize2D(
            cuContext, cudau::ArrayElementType::Float32, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write(data, 4);
    }

    s_Fx3ImmTextureCache[cacheKey] = std::move(cacheValue);

    *texture = &s_Fx3ImmTextureCache.at(cacheKey).texture;
}

static void createFx4ImmTexture(
    CUcontext cuContext,
    const float4 &immValue,
    bool isNormalized,
    const cudau::Array** texture) {
    Fx4ImmTextureCacheKey cacheKey;
    cacheKey.immValue = immValue;
    cacheKey.cuContext = cuContext;
    if (s_Fx4ImmTextureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_Fx4ImmTextureCache.at(cacheKey);
        *texture = &value.texture;
        return;
    }

    TextureCacheValue cacheValue;
    cacheValue.isHDR = !isNormalized;
    if (isNormalized) {
        uint32_t data = ((std::min<uint32_t>(255 * immValue.x, 255) << 0) |
                         (std::min<uint32_t>(255 * immValue.y, 255) << 8) |
                         (std::min<uint32_t>(255 * immValue.z, 255) << 16) |
                         (std::min<uint32_t>(255 * immValue.w, 255) << 24));
        cacheValue.texture.initialize2D(
            cuContext, cudau::ArrayElementType::UInt8, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write<uint8_t>(reinterpret_cast<uint8_t*>(&data), 4);
    }
    else {
        float data[4] = {
            immValue.x, immValue.y, immValue.z, immValue.w
        };
        cacheValue.texture.initialize2D(
            cuContext, cudau::ArrayElementType::Float32, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write(data, 4);
    }

    s_Fx4ImmTextureCache[cacheKey] = std::move(cacheValue);

    *texture = &s_Fx4ImmTextureCache.at(cacheKey).texture;
}

static bool loadTexture(
    const std::filesystem::path &filePath, const float4 &fallbackValue,
    CUcontext cuContext,
    const cudau::Array** texture,
    bool* needsDegamma,
    bool* isHDR = nullptr) {
    TextureCacheKey cacheKey;
    cacheKey.filePath = filePath;
    cacheKey.cuContext = cuContext;
    if (s_textureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_textureCache.at(cacheKey);
        *texture = &value.texture;
        *needsDegamma = value.needsDegamma;
        if (isHDR)
            *isHDR = value.isHDR;
        return true;
    }

    bool success = true;
    TextureCacheValue cacheValue;
    if (filePath.extension() == ".dds" ||
        filePath.extension() == ".DDS") {
        int32_t width, height, mipCount;
        dds::Format ddsFormat;
        size_t* sizes;
        uint8_t** imageData = dds::load(filePath.string().c_str(),
                                        &width, &height, &mipCount, &sizes, &ddsFormat);
        if (imageData) {
            cudau::ArrayElementType elemType;
            translate(ddsFormat, &elemType, &cacheValue.needsDegamma);
            cacheValue.isHDR =
                elemType == cudau::ArrayElementType::BC6H_SF16 ||
                elemType == cudau::ArrayElementType::BC6H_UF16;
            cacheValue.texture.initialize2D(
                cuContext, elemType, 1,
                cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                width, height, 1);
            cacheValue.texture.write<uint8_t>(imageData[0], static_cast<uint32_t>(sizes[0]));
            dds::free(imageData, mipCount, sizes);
        }
        else {
            success = false;
        }
    }
    else {
        int32_t width, height, n;
        uint8_t* linearImageData = stbi_load(filePath.string().c_str(),
                                             &width, &height, &n, 4);
        if (linearImageData) {
            cacheValue.texture.initialize2D(
                cuContext, cudau::ArrayElementType::UInt8, 4,
                cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                width, height, 1);
            cacheValue.texture.write<uint8_t>(linearImageData, width * height * 4);
            stbi_image_free(linearImageData);
            cacheValue.needsDegamma = true;
        }
        else {
            success = false;
        }
    }

    if (success) {
        s_textureCache[cacheKey] = std::move(cacheValue);

        *texture = &s_textureCache.at(cacheKey).texture;
        *needsDegamma = s_textureCache.at(cacheKey).needsDegamma;
        if (isHDR)
            *isHDR = s_textureCache.at(cacheKey).isHDR;
    }
    else {
        createFx4ImmTexture(cuContext, fallbackValue, true, texture);
        cacheValue.needsDegamma = true;
        cacheValue.isHDR = false;
    }

    return success;
}

static bool loadNormalTexture(
    const std::filesystem::path &filePath,
    CUcontext cuContext,
    const cudau::Array** texture,
    BumpMapTextureType* bumpMapType) {
    TextureCacheKey cacheKey;
    cacheKey.filePath = filePath;
    cacheKey.cuContext = cuContext;
    if (s_textureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_textureCache.at(cacheKey);
        *texture = &value.texture;
        *bumpMapType = value.bumpMapType;
        return true;
    }

    bool success = true;
    TextureCacheValue cacheValue;
    if (filePath.extension() == ".dds" ||
        filePath.extension() == ".DDS") {
        int32_t width, height, mipCount;
        dds::Format ddsFormat;
        size_t* sizes;
        uint8_t** imageData = dds::load(filePath.string().c_str(),
                                        &width, &height, &mipCount, &sizes, &ddsFormat);
        if (imageData) {
            cudau::ArrayElementType elemType;
            translate(ddsFormat, &elemType, &cacheValue.needsDegamma);
            if (elemType == cudau::ArrayElementType::BC1_UNorm ||
                elemType == cudau::ArrayElementType::BC2_UNorm ||
                elemType == cudau::ArrayElementType::BC3_UNorm ||
                elemType == cudau::ArrayElementType::BC7_UNorm)
                cacheValue.bumpMapType = BumpMapTextureType::NormalMap_BC;
            else if (elemType == cudau::ArrayElementType::BC4_SNorm ||
                     elemType == cudau::ArrayElementType::BC4_UNorm)
                cacheValue.bumpMapType = BumpMapTextureType::HeightMap_BC;
            else if (elemType == cudau::ArrayElementType::BC5_UNorm)
                cacheValue.bumpMapType = BumpMapTextureType::NormalMap_BC_2ch;
            else
                Assert_NotImplemented();
            auto textureGather = cacheValue.bumpMapType == BumpMapTextureType::HeightMap_BC ?
                cudau::ArrayTextureGather::Enable :
                cudau::ArrayTextureGather::Disable;
            cacheValue.texture.initialize2D(
                cuContext, elemType, 1,
                cudau::ArraySurface::Disable,
                textureGather,
                width, height, 1);
            cacheValue.texture.write<uint8_t>(imageData[0], static_cast<uint32_t>(sizes[0]));
            dds::free(imageData, mipCount, sizes);
        }
        else {
            success = false;
        }
    }
    else {
        int32_t width, height, n;
        uint8_t* linearImageData = stbi_load(filePath.string().c_str(),
                                             &width, &height, &n, 4);
        std::string filename = filePath.filename().string();
        if (n > 1 &&
            filename != "spnza_bricks_a_bump.png") // Dedicated fix for crytek sponza model.
            cacheValue.bumpMapType = BumpMapTextureType::NormalMap;
        else
            cacheValue.bumpMapType = BumpMapTextureType::HeightMap;
        if (linearImageData) {
            auto textureGather = cacheValue.bumpMapType == BumpMapTextureType::HeightMap ?
                cudau::ArrayTextureGather::Enable :
                cudau::ArrayTextureGather::Disable;
            cacheValue.texture.initialize2D(
                cuContext, cudau::ArrayElementType::UInt8, 4,
                cudau::ArraySurface::Disable, textureGather,
                width, height, 1);
            cacheValue.texture.write<uint8_t>(linearImageData, width * height * 4);
            stbi_image_free(linearImageData);
        }
        else {
            success = false;
        }
    }

    if (success) {
        s_textureCache[cacheKey] = std::move(cacheValue);
        *texture = &s_textureCache.at(cacheKey).texture;
        *bumpMapType = s_textureCache.at(cacheKey).bumpMapType;
    }
    else {
        createFx3ImmTexture(cuContext, float3(0.5f, 0.5f, 1.0f), true, texture);
        *bumpMapType = BumpMapTextureType::NormalMap;
    }

    return success;
}

static void createNormalTexture(
    CUcontext cuContext,
    const std::filesystem::path &normalPath,
    Material* mat, BumpMapTextureType* bumpMapType) {
    if (normalPath.empty()) {
        createFx3ImmTexture(cuContext, float3(0.5f, 0.5f, 1.0f), true, &mat->normal);
        *bumpMapType = BumpMapTextureType::NormalMap;
    }
    else {
        hpprintf("  Reading: %s ... ", normalPath.string().c_str());
        if (loadNormalTexture(normalPath, cuContext, &mat->normal, bumpMapType))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
}

static void createEmittanceTexture(
    CUcontext cuContext,
    const std::filesystem::path &emittancePath, const float3 &immEmittance,
    Material* mat,
    bool* needsDegamma, bool* isHDR) {
    *needsDegamma = false;
    *isHDR = false;
    if (emittancePath.empty()) {
        mat->texEmittance = 0;
        if (immEmittance != float3(0.0f, 0.0f, 0.0f))
            createFx3ImmTexture(cuContext, immEmittance, false, &mat->emittance);
    }
    else {
        hpprintf("  Reading: %s ... ", emittancePath.string().c_str());
        if (loadTexture(emittancePath, float4(immEmittance, 1.0f), cuContext,
                        &mat->emittance, needsDegamma, isHDR))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
}

static Material* createLambertMaterial(
    CUcontext cuContext, Scene* scene,
    const std::filesystem::path &reflectancePath, const float3 &immReflectance,
    const std::filesystem::path &normalPath,
    const std::filesystem::path &emittancePath, const float3 &immEmittance) {
    shared::MaterialData* matDataOnHost = scene->materialDataBuffer.getMappedPointer();

    cudau::TextureSampler sampler_sRGB;
    sampler_sRGB.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_sRGB.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_sRGB.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

    cudau::TextureSampler sampler_float;
    sampler_float.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_float.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_float.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_float.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_float.setReadMode(cudau::TextureReadMode::ElementType);

    cudau::TextureSampler sampler_normFloat;
    sampler_normFloat.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_normFloat.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_normFloat.setReadMode(cudau::TextureReadMode::NormalizedFloat);

    Material* mat = new Material();
    bool needsDegamma;

    mat->body = Material::Lambert();
    auto &body = std::get<Material::Lambert>(mat->body);
    if (!reflectancePath.empty()) {
        hpprintf("  Reading: %s ... ", reflectancePath.string().c_str());
        if (loadTexture(reflectancePath, float4(immReflectance, 1.0f), cuContext,
                        &body.reflectance, &needsDegamma))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
    if (!body.reflectance) {
        createFx3ImmTexture(cuContext, immReflectance, true, &body.reflectance);
        needsDegamma = true;
    }
    if (needsDegamma)
        body.texReflectance = sampler_sRGB.createTextureObject(*body.reflectance);
    else
        body.texReflectance = sampler_normFloat.createTextureObject(*body.reflectance);

    BumpMapTextureType bumpMapType;
    createNormalTexture(cuContext, normalPath, mat, &bumpMapType);
    mat->texNormal = sampler_normFloat.createTextureObject(*mat->normal);
    CallableProgram dcReadModifiedNormal;
    if (bumpMapType == BumpMapTextureType::NormalMap ||
        bumpMapType == BumpMapTextureType::NormalMap_BC)
        dcReadModifiedNormal = CallableProgram_readModifiedNormalFromNormalMap;
    else if (bumpMapType == BumpMapTextureType::NormalMap_BC)
        dcReadModifiedNormal = CallableProgram_readModifiedNormalFromNormalMap2ch;
    else
        dcReadModifiedNormal = CallableProgram_readModifiedNormalFromHeightMap;

    bool isHDR;
    createEmittanceTexture(cuContext, emittancePath, immEmittance,
                           mat, &needsDegamma, &isHDR);
    if (mat->emittance) {
        if (needsDegamma)
            mat->texEmittance = sampler_sRGB.createTextureObject(*mat->emittance);
        else if (isHDR)
            mat->texEmittance = sampler_float.createTextureObject(*mat->emittance);
        else
            mat->texEmittance = sampler_normFloat.createTextureObject(*mat->emittance);
    }

    mat->materialSlot = scene->materialSlotFinder.getFirstAvailableSlot();
    scene->materialSlotFinder.setInUse(mat->materialSlot);

    shared::MaterialData matData = {};
    matData.asLambert.reflectance = body.texReflectance;
    matData.normal = mat->texNormal;
    matData.emittance = mat->texEmittance;
    matData.normalWidth = mat->normal->getWidth();
    matData.normalHeight = mat->normal->getHeight();
    matData.readModifiedNormal = shared::ReadModifiedNormal(dcReadModifiedNormal);
    matData.setupBSDFBody = shared::SetupBSDFBody(CallableProgram_setupLambertBRDF);
    matData.bsdfGetSurfaceParameters = shared::BSDFGetSurfaceParameters(CallableProgram_LambertBRDF_getSurfaceParameters);
    matData.bsdfSampleThroughput = shared::BSDFSampleThroughput(CallableProgram_LambertBRDF_sampleThroughput);
    matData.bsdfEvaluate = shared::BSDFEvaluate(CallableProgram_LambertBRDF_evaluate);
    matData.bsdfEvaluatePDF = shared::BSDFEvaluatePDF(CallableProgram_LambertBRDF_evaluatePDF);
    matData.bsdfEvaluateDHReflectanceEstimate = shared::BSDFEvaluateDHReflectanceEstimate(CallableProgram_LambertBRDF_evaluateDHReflectanceEstimate);
    matDataOnHost[mat->materialSlot] = matData;

    return mat;
}

static Material* createDiffuseAndSpecularMaterial(
    CUcontext cuContext, Scene* scene,
    const std::filesystem::path &diffuseColorPath, const float3 &immDiffuseColor,
    const std::filesystem::path &specularColorPath, const float3 &immSpecularColor,
    float immSmoothness,
    const std::filesystem::path &normalPath,
    const std::filesystem::path &emittancePath, const float3 &immEmittance) {
    shared::MaterialData* matDataOnHost = scene->materialDataBuffer.getMappedPointer();

    cudau::TextureSampler sampler_sRGB;
    sampler_sRGB.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_sRGB.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_sRGB.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

    cudau::TextureSampler sampler_float;
    sampler_float.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_float.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_float.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_float.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_float.setReadMode(cudau::TextureReadMode::ElementType);

    cudau::TextureSampler sampler_normFloat;
    sampler_normFloat.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_normFloat.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_normFloat.setReadMode(cudau::TextureReadMode::NormalizedFloat);

    Material* mat = new Material();
    bool needsDegamma = false;

    mat->body = Material::DiffuseAndSpecular();
    auto &body = std::get<Material::DiffuseAndSpecular>(mat->body);

    if (!diffuseColorPath.empty()) {
        hpprintf("  Reading: %s ... ", diffuseColorPath.string().c_str());
        if (loadTexture(diffuseColorPath, float4(immDiffuseColor, 1.0f), cuContext,
                        &body.diffuse, &needsDegamma))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
    if (!body.diffuse) {
        createFx3ImmTexture(cuContext, immDiffuseColor, true, &body.diffuse);
        needsDegamma = true;
    }
    if (needsDegamma)
        body.texDiffuse = sampler_sRGB.createTextureObject(*body.diffuse);
    else
        body.texDiffuse = sampler_normFloat.createTextureObject(*body.diffuse);

    if (!specularColorPath.empty()) {
        hpprintf("  Reading: %s ... ", specularColorPath.string().c_str());
        if (loadTexture(specularColorPath, float4(immSpecularColor, 1.0f), cuContext,
                        &body.specular, &needsDegamma))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
    if (!body.specular) {
        createFx3ImmTexture(cuContext, immSpecularColor, true, &body.specular);
        needsDegamma = true;
    }
    if (needsDegamma)
        body.texSpecular = sampler_sRGB.createTextureObject(*body.specular);
    else
        body.texSpecular = sampler_normFloat.createTextureObject(*body.specular);

    createFx1ImmTexture(cuContext, immSmoothness, true, &body.smoothness);
    body.texSmoothness = sampler_normFloat.createTextureObject(*body.smoothness);

    BumpMapTextureType bumpMapType;
    createNormalTexture(cuContext, normalPath, mat, &bumpMapType);
    mat->texNormal = sampler_normFloat.createTextureObject(*mat->normal);
    CallableProgram dcReadModifiedNormal;
    if (bumpMapType == BumpMapTextureType::NormalMap ||
        bumpMapType == BumpMapTextureType::NormalMap_BC)
        dcReadModifiedNormal = CallableProgram_readModifiedNormalFromNormalMap;
    else if (bumpMapType == BumpMapTextureType::NormalMap_BC)
        dcReadModifiedNormal = CallableProgram_readModifiedNormalFromNormalMap2ch;
    else
        dcReadModifiedNormal = CallableProgram_readModifiedNormalFromHeightMap;

    bool isHDR;
    createEmittanceTexture(cuContext, emittancePath, immEmittance,
                           mat, &needsDegamma, &isHDR);
    if (mat->emittance) {
        if (needsDegamma)
            mat->texEmittance = sampler_sRGB.createTextureObject(*mat->emittance);
        else if (isHDR)
            mat->texEmittance = sampler_float.createTextureObject(*mat->emittance);
        else
            mat->texEmittance = sampler_normFloat.createTextureObject(*mat->emittance);
    }

    mat->materialSlot = scene->materialSlotFinder.getFirstAvailableSlot();
    scene->materialSlotFinder.setInUse(mat->materialSlot);

    shared::MaterialData matData = {};
    matData.asDiffuseAndSpecular.diffuse = body.texDiffuse;
    matData.asDiffuseAndSpecular.specular = body.texSpecular;
    matData.asDiffuseAndSpecular.smoothness = body.texSmoothness;
    matData.normal = mat->texNormal;
    matData.emittance = mat->texEmittance;
    matData.normalWidth = mat->normal->getWidth();
    matData.normalHeight = mat->normal->getHeight();
    matData.readModifiedNormal = shared::ReadModifiedNormal(dcReadModifiedNormal);
    matData.setupBSDFBody = shared::SetupBSDFBody(CallableProgram_setupDiffuseAndSpecularBRDF);
    matData.bsdfGetSurfaceParameters =
        shared::BSDFGetSurfaceParameters(CallableProgram_DiffuseAndSpecularBRDF_getSurfaceParameters);
    matData.bsdfSampleThroughput =
        shared::BSDFSampleThroughput(CallableProgram_DiffuseAndSpecularBRDF_sampleThroughput);
    matData.bsdfEvaluate = shared::BSDFEvaluate(CallableProgram_DiffuseAndSpecularBRDF_evaluate);
    matData.bsdfEvaluatePDF = shared::BSDFEvaluatePDF(CallableProgram_DiffuseAndSpecularBRDF_evaluatePDF);
    matData.bsdfEvaluateDHReflectanceEstimate =
        shared::BSDFEvaluateDHReflectanceEstimate(CallableProgram_DiffuseAndSpecularBRDF_evaluateDHReflectanceEstimate);
    matDataOnHost[mat->materialSlot] = matData;

    return mat;
}

static Material* createSimplePBRMaterial(
    CUcontext cuContext, Scene* scene,
    const std::filesystem::path &baseColor_opacityPath, const float4 &immBaseColor_opacity,
    const std::filesystem::path &occlusion_roughness_metallicPath,
    const float3 &immOcclusion_roughness_metallic,
    const std::filesystem::path &normalPath,
    const std::filesystem::path &emittancePath, const float3 &immEmittance) {
    shared::MaterialData* matDataOnHost = scene->materialDataBuffer.getMappedPointer();

    cudau::TextureSampler sampler_sRGB;
    sampler_sRGB.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_sRGB.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_sRGB.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

    cudau::TextureSampler sampler_float;
    sampler_float.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_float.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_float.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_float.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_float.setReadMode(cudau::TextureReadMode::ElementType);

    cudau::TextureSampler sampler_normFloat;
    sampler_normFloat.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_normFloat.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_normFloat.setReadMode(cudau::TextureReadMode::NormalizedFloat);

    Material* mat = new Material();
    bool needsDegamma = false;

    mat->body = Material::SimplePBR();
    auto &body = std::get<Material::SimplePBR>(mat->body);

    if (!baseColor_opacityPath.empty()) {
        hpprintf("  Reading: %s ... ", baseColor_opacityPath.string().c_str());
        if (loadTexture(baseColor_opacityPath, immBaseColor_opacity, cuContext,
                        &body.baseColor_opacity, &needsDegamma))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
    if (!body.baseColor_opacity) {
        createFx4ImmTexture(cuContext, immBaseColor_opacity, true,
                            &body.baseColor_opacity);
        needsDegamma = true;
    }
    if (needsDegamma)
        body.texBaseColor_opacity = sampler_sRGB.createTextureObject(*body.baseColor_opacity);
    else
        body.texBaseColor_opacity = sampler_normFloat.createTextureObject(*body.baseColor_opacity);

    if (!occlusion_roughness_metallicPath.empty()) {
        hpprintf("  Reading: %s ... ", occlusion_roughness_metallicPath.string().c_str());
        if (loadTexture(occlusion_roughness_metallicPath, float4(immOcclusion_roughness_metallic, 0.0f),
                        cuContext,
                        &body.occlusion_roughness_metallic, &needsDegamma))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
    if (!body.occlusion_roughness_metallic) {
        createFx3ImmTexture(cuContext, immOcclusion_roughness_metallic, true,
                            &body.occlusion_roughness_metallic);
    }
    body.texOcclusion_roughness_metallic =
        sampler_normFloat.createTextureObject(*body.occlusion_roughness_metallic);

    BumpMapTextureType bumpMapType;
    createNormalTexture(cuContext, normalPath, mat, &bumpMapType);
    mat->texNormal = sampler_normFloat.createTextureObject(*mat->normal);
    CallableProgram dcReadModifiedNormal;
    if (bumpMapType == BumpMapTextureType::NormalMap ||
        bumpMapType == BumpMapTextureType::NormalMap_BC)
        dcReadModifiedNormal = CallableProgram_readModifiedNormalFromNormalMap;
    else if (bumpMapType == BumpMapTextureType::NormalMap_BC)
        dcReadModifiedNormal = CallableProgram_readModifiedNormalFromNormalMap2ch;
    else
        dcReadModifiedNormal = CallableProgram_readModifiedNormalFromHeightMap;

    bool isHDR;
    createEmittanceTexture(cuContext, emittancePath, immEmittance,
                           mat, &needsDegamma, &isHDR);
    if (mat->emittance) {
        if (needsDegamma)
            mat->texEmittance = sampler_sRGB.createTextureObject(*mat->emittance);
        else if (isHDR)
            mat->texEmittance = sampler_float.createTextureObject(*mat->emittance);
        else
            mat->texEmittance = sampler_normFloat.createTextureObject(*mat->emittance);
    }

    mat->materialSlot = scene->materialSlotFinder.getFirstAvailableSlot();
    scene->materialSlotFinder.setInUse(mat->materialSlot);

    shared::MaterialData matData = {};
    matData.asSimplePBR.baseColor_opacity = body.texBaseColor_opacity;
    matData.asSimplePBR.occlusion_roughness_metallic = body.texOcclusion_roughness_metallic;
    matData.normal = mat->texNormal;
    matData.emittance = mat->texEmittance;
    matData.normalWidth = mat->normal->getWidth();
    matData.normalHeight = mat->normal->getHeight();
    matData.readModifiedNormal = shared::ReadModifiedNormal(dcReadModifiedNormal);
    matData.setupBSDFBody = shared::SetupBSDFBody(CallableProgram_setupSimplePBR_BRDF);
    matData.bsdfGetSurfaceParameters =
        shared::BSDFGetSurfaceParameters(CallableProgram_DiffuseAndSpecularBRDF_getSurfaceParameters);
    matData.bsdfSampleThroughput =
        shared::BSDFSampleThroughput(CallableProgram_DiffuseAndSpecularBRDF_sampleThroughput);
    matData.bsdfEvaluate = shared::BSDFEvaluate(CallableProgram_DiffuseAndSpecularBRDF_evaluate);
    matData.bsdfEvaluatePDF = shared::BSDFEvaluatePDF(CallableProgram_DiffuseAndSpecularBRDF_evaluatePDF);
    matData.bsdfEvaluateDHReflectanceEstimate =
        shared::BSDFEvaluateDHReflectanceEstimate(CallableProgram_DiffuseAndSpecularBRDF_evaluateDHReflectanceEstimate);
    matDataOnHost[mat->materialSlot] = matData;

    return mat;
}

static GeometryInstance* createGeometryInstance(
    CUcontext cuContext, Scene* scene,
    const std::vector<shared::Vertex> &vertices,
    const std::vector<shared::Triangle> &triangles,
    const Material* mat) {
    shared::GeometryInstanceData* geomInstDataOnHost = scene->geomInstDataBuffer.getMappedPointer();

    GeometryInstance* geomInst = new GeometryInstance();

    for (int triIdx = 0; triIdx < triangles.size(); ++triIdx) {
        const shared::Triangle &tri = triangles[triIdx];
        const shared::Vertex (&vs)[3] = {
            vertices[tri.index0],
            vertices[tri.index1],
            vertices[tri.index2],
        };
        geomInst->aabb
            .unify(vertices[0].position)
            .unify(vertices[1].position)
            .unify(vertices[2].position);
    }

    geomInst->mat = mat;
    geomInst->vertexBuffer.initialize(cuContext, Scene::bufferType, vertices);
    geomInst->triangleBuffer.initialize(cuContext, Scene::bufferType, triangles);
    if (mat->emittance) {
#if USE_PROBABILITY_TEXTURE
        geomInst->emitterPrimDist.initialize(cuContext, triangles.size());
#else
        geomInst->emitterPrimDist.initialize(cuContext, Scene::bufferType, nullptr, triangles.size());
#endif
    }
    geomInst->geomInstSlot = scene->geomInstSlotFinder.getFirstAvailableSlot();
    scene->geomInstSlotFinder.setInUse(geomInst->geomInstSlot);

    shared::GeometryInstanceData geomInstData = {};
    geomInstData.vertexBuffer = geomInst->vertexBuffer.getDevicePointer();
    geomInstData.triangleBuffer = geomInst->triangleBuffer.getDevicePointer();
    geomInst->emitterPrimDist.getDeviceType(&geomInstData.emitterPrimDist);
    geomInstData.materialSlot = mat->materialSlot;
    geomInstData.geomInstSlot = geomInst->geomInstSlot;
    geomInstDataOnHost[geomInst->geomInstSlot] = geomInstData;

    geomInst->optixGeomInst = scene->optixScene.createGeometryInstance();
    geomInst->optixGeomInst.setVertexBuffer(geomInst->vertexBuffer);
    geomInst->optixGeomInst.setTriangleBuffer(geomInst->triangleBuffer);
    geomInst->optixGeomInst.setNumMaterials(1, optixu::BufferView());
    geomInst->optixGeomInst.setMaterial(0, 0, scene->optixDefaultMaterial);
    geomInst->optixGeomInst.setUserData(geomInst->geomInstSlot);

    return geomInst;
}

static GeometryGroup* createGeometryGroup(
    Scene* scene,
    const std::set<const GeometryInstance*> &geomInsts) {
    GeometryGroup* geomGroup = new GeometryGroup();
    geomGroup->geomInsts = geomInsts;
    geomGroup->numEmitterPrimitives = 0;

    geomGroup->optixGas = scene->optixScene.createGeometryAccelerationStructure();
    for (auto it = geomInsts.cbegin(); it != geomInsts.cend(); ++it) {
        const GeometryInstance* geomInst = *it;
        geomGroup->optixGas.addChild(geomInst->optixGeomInst);
        if (geomInst->mat->emittance)
            geomGroup->numEmitterPrimitives += geomInst->triangleBuffer.numElements();
        geomGroup->aabb.unify(geomInst->aabb);
    }
    geomGroup->optixGas.setNumMaterialSets(1);
    geomGroup->optixGas.setNumRayTypes(0, scene->numRayTypes);

    return geomGroup;
}

constexpr bool useLambertMaterial = false;

void createTriangleMeshes(
    const std::string &meshName,
    const std::filesystem::path &filePath,
    MaterialConvention matConv,
    const Matrix4x4 &preTransform,
    CUcontext cuContext, Scene* scene) {
    hpprintf("Reading: %s ... ", filePath.string().c_str());
    fflush(stdout);
    Assimp::Importer importer;
    const aiScene* aiscene = importer.ReadFile(
        filePath.string(),
        aiProcess_Triangulate |
        aiProcess_GenNormals |
        aiProcess_CalcTangentSpace |
        aiProcess_FlipUVs);
    if (!aiscene) {
        hpprintf("Failed to load %s.\n", filePath.string().c_str());
        return;
    }
    hpprintf("done.\n");

    std::filesystem::path dirPath = filePath;
    dirPath.remove_filename();

    uint32_t baseMatIndex = static_cast<uint32_t>(scene->materials.size());
    shared::MaterialData* matDataOnHost = scene->materialDataBuffer.getMappedPointer();
    for (uint32_t matIdx = 0; matIdx < aiscene->mNumMaterials; ++matIdx) {
        std::filesystem::path emittancePath;
        float3 immEmittance = float3(0.0f);

        const aiMaterial* aiMat = aiscene->mMaterials[matIdx];
        aiString strValue;
        float color[3];

        std::string matName;
        if (aiMat->Get(AI_MATKEY_NAME, strValue) == aiReturn_SUCCESS)
            matName = strValue.C_Str();
        hpprintf("%s:\n", matName.c_str());

        std::filesystem::path reflectancePath;
        float3 immReflectance;
        std::filesystem::path diffuseColorPath;
        float3 immDiffuseColor;
        std::filesystem::path specularColorPath;
        float3 immSpecularColor;
        float immSmoothness;
        if constexpr (useLambertMaterial) {
            if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
                reflectancePath = dirPath / strValue.C_Str();
            }
            else {
                if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) != aiReturn_SUCCESS) {
                    color[0] = 1.0f;
                    color[1] = 0.0f;
                    color[2] = 1.0f;
                }
                immReflectance = float3(color[0], color[1], color[2]);
            }
            (void)diffuseColorPath;
            (void)immDiffuseColor;
            (void)specularColorPath;
            (void)immSpecularColor;
            (void)immSmoothness;
        }
        else {
            if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
                diffuseColorPath = dirPath / strValue.C_Str();
            }
            else {
                if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) != aiReturn_SUCCESS) {
                    color[0] = 0.0f;
                    color[1] = 0.0f;
                    color[2] = 0.0f;
                }
                immDiffuseColor = float3(color[0], color[1], color[2]);
            }

            if (aiMat->Get(AI_MATKEY_TEXTURE_SPECULAR(0), strValue) == aiReturn_SUCCESS) {
                specularColorPath = dirPath / strValue.C_Str();
            }
            else {
                if (aiMat->Get(AI_MATKEY_COLOR_SPECULAR, color, nullptr) != aiReturn_SUCCESS) {
                    color[0] = 0.0f;
                    color[1] = 0.0f;
                    color[2] = 0.0f;
                }
                immSpecularColor = float3(color[0], color[1], color[2]);
            }

            // JP: 極端に鋭いスペキュラーにするとNEEで寄与が一切サンプルできなくなってしまう。
            // EN: Exteremely sharp specular makes it impossible to sample a contribution with NEE.
            if (aiMat->Get(AI_MATKEY_SHININESS, &immSmoothness, nullptr) != aiReturn_SUCCESS)
                immSmoothness = 0.0f;
            immSmoothness = std::sqrt(immSmoothness);
            immSmoothness = immSmoothness / 11.0f/*30.0f*/;

            (void)reflectancePath;
            (void)immReflectance;
        }

        std::filesystem::path normalPath;
        if (aiMat->Get(AI_MATKEY_TEXTURE_HEIGHT(0), strValue) == aiReturn_SUCCESS)
            normalPath = dirPath / strValue.C_Str();
        else if (aiMat->Get(AI_MATKEY_TEXTURE_NORMALS(0), strValue) == aiReturn_SUCCESS)
            normalPath = dirPath / strValue.C_Str();

        if (matName == "Pavement_Cobblestone_Big_BLENDSHADER") {
            immSmoothness = 0.2f;
        }
        else if (matName == "Pavement_Cobblestone_Small_BLENDSHADER") {
            immSmoothness = 0.2f;
        }
        else if (matName == "Pavement_Brick_BLENDSHADER") {
            immSmoothness = 0.2f;
        }
        else if (matName == "Pavement_Cobblestone_Wet_BLENDSHADER") {
            immSmoothness = 0.2f;
        }

        if (aiMat->Get(AI_MATKEY_TEXTURE_EMISSIVE(0), strValue) == aiReturn_SUCCESS)
            emittancePath = dirPath / strValue.C_Str();
        else if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, color, nullptr) == aiReturn_SUCCESS)
            immEmittance = float3(color[0], color[1], color[2]);

        Material* mat;
        if (matConv == MaterialConvention::Traditional) {
            if constexpr (useLambertMaterial) {
                mat = createLambertMaterial(
                    cuContext, scene,
                    reflectancePath, immReflectance,
                    normalPath,
                    emittancePath, immEmittance);
            }
            else {
                mat = createDiffuseAndSpecularMaterial(
                    cuContext, scene,
                    diffuseColorPath, immDiffuseColor,
                    specularColorPath, immSpecularColor,
                    immSmoothness,
                    normalPath,
                    emittancePath, immEmittance);
            }
        }
        else {
            // JP: diffuseテクスチャーとしてベースカラー + 不透明度
            //     specularテクスチャーとしてオクルージョン、ラフネス、メタリック
            //     が格納されていると仮定している。
            // EN: We assume diffuse texture as base color + opacity,
            //     specular texture as occlusion, roughness, metallic.
            mat = createSimplePBRMaterial(
                cuContext, scene,
                diffuseColorPath, float4(immDiffuseColor, 1.0f),
                specularColorPath, immSpecularColor,
                normalPath,
                emittancePath, immEmittance);
        }

        scene->materials.push_back(mat);
    }

    uint32_t baseGeomInstIndex = static_cast<uint32_t>(scene->geomInsts.size());
    Matrix3x3 preNormalTransform = transpose(inverse(preTransform.getUpperLeftMatrix()));
    for (uint32_t meshIdx = 0; meshIdx < aiscene->mNumMeshes; ++meshIdx) {
        const aiMesh* aiMesh = aiscene->mMeshes[meshIdx];

        std::vector<shared::Vertex> vertices(aiMesh->mNumVertices);
        for (int vIdx = 0; vIdx < vertices.size(); ++vIdx) {
            const aiVector3D &aip = aiMesh->mVertices[vIdx];
            const aiVector3D &ain = aiMesh->mNormals[vIdx];
            aiVector3D aitc0dir;
            if (aiMesh->mTangents)
                aitc0dir = aiMesh->mTangents[vIdx];
            if (!aiMesh->mTangents || !std::isfinite(aitc0dir.x)) {
                const auto makeCoordinateSystem = []
                (const float3 &normal, float3* tangent, float3* bitangent) {
                    float sign = normal.z >= 0 ? 1.0f : -1.0f;
                    const float a = -1 / (sign + normal.z);
                    const float b = normal.x * normal.y * a;
                    *tangent = make_float3(1 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
                    *bitangent = make_float3(b, sign + normal.y * normal.y * a, -normal.y);
                };
                float3 tangent, bitangent;
                makeCoordinateSystem(float3(ain.x, ain.y, ain.z), &tangent, &bitangent);
                aitc0dir = aiVector3D(tangent.x, tangent.y, tangent.z);
            }
            const aiVector3D ait = aiMesh->mTextureCoords[0] ?
                aiMesh->mTextureCoords[0][vIdx] :
                aiVector3D(0.0f, 0.0f, 0.0f);

            shared::Vertex v;
            v.position = preTransform * float3(aip.x, aip.y, aip.z);
            v.normal = normalize(preNormalTransform * float3(ain.x, ain.y, ain.z));
            v.texCoord0Dir = normalize(preTransform * float3(aitc0dir.x, aitc0dir.y, aitc0dir.z));
            v.texCoord = float2(ait.x, ait.y);
            vertices[vIdx] = v;
        }

        std::vector<shared::Triangle> triangles(aiMesh->mNumFaces);
        for (int fIdx = 0; fIdx < triangles.size(); ++fIdx) {
            const aiFace &aif = aiMesh->mFaces[fIdx];
            Assert(aif.mNumIndices == 3, "Number of face vertices must be 3 here.");
            shared::Triangle tri;
            tri.index0 = aif.mIndices[0];
            tri.index1 = aif.mIndices[1];
            tri.index2 = aif.mIndices[2];
            triangles[fIdx] = tri;
        }

        scene->geomInsts.push_back(createGeometryInstance(
            cuContext, scene, vertices, triangles, scene->materials[baseMatIndex + aiMesh->mMaterialIndex]));
    }

    std::vector<FlattenedNode> flattenedNodes;
    computeFlattenedNodes(aiscene, Matrix4x4(), aiscene->mRootNode, flattenedNodes);
    //for (int i = 0; i < flattenedNodes.size(); ++i) {
    //    const Matrix4x4 &mat = flattenedNodes[i].transform;
    //    hpprintf("%8.5f, %8.5f, %8.5f, %8.5f\n", mat.m00, mat.m01, mat.m02, mat.m03);
    //    hpprintf("%8.5f, %8.5f, %8.5f, %8.5f\n", mat.m10, mat.m11, mat.m12, mat.m13);
    //    hpprintf("%8.5f, %8.5f, %8.5f, %8.5f\n", mat.m20, mat.m21, mat.m22, mat.m23);
    //    hpprintf("%8.5f, %8.5f, %8.5f, %8.5f\n", mat.m30, mat.m31, mat.m32, mat.m33);
    //    hpprintf("\n");
    //}

    auto mesh = new Mesh();
    shared::InstanceData* instDataOnHost = scene->instDataBuffer[0].getMappedPointer();
    std::map<std::set<const GeometryInstance*>, GeometryGroup*> geomGroupMap;
    for (int nodeIdx = 0; nodeIdx < flattenedNodes.size(); ++nodeIdx) {
        const FlattenedNode &node = flattenedNodes[nodeIdx];
        if (node.meshIndices.size() == 0)
            continue;

        std::set<const GeometryInstance*> srcGeomInsts;
        for (int i = 0; i < node.meshIndices.size(); ++i)
            srcGeomInsts.insert(scene->geomInsts[baseGeomInstIndex + node.meshIndices[i]]);
        GeometryGroup* geomGroup;
        if (geomGroupMap.count(srcGeomInsts) > 0) {
            geomGroup = geomGroupMap.at(srcGeomInsts);
        }
        else {
            geomGroup = createGeometryGroup(scene, srcGeomInsts);
            scene->geomGroups.push_back(geomGroup);
        }

        Mesh::Group g = {};
        g.geomGroup = geomGroup;
        g.transform = node.transform;
        mesh->groups.push_back(g);
    }

    scene->meshes[meshName] = mesh;
}

void createRectangleLight(
    const std::string &meshName,
    float width, float depth,
    const float3 &reflectance,
    const std::filesystem::path &emittancePath,
    const float3 &immEmittance,
    const Matrix4x4 &transform,
    CUcontext cuContext, Scene* scene) {
    Material* material;
    if constexpr (useLambertMaterial)
        material = createLambertMaterial(cuContext, scene, "", reflectance, "", emittancePath, immEmittance);
    else
        material = createDiffuseAndSpecularMaterial(
            cuContext, scene, "", reflectance, "", float3(0.0f), 0.3f,
            "",
            emittancePath, immEmittance);
    scene->materials.push_back(material);

    std::vector<shared::Vertex> vertices = {
        shared::Vertex{float3(-0.5f * width, 0.0f, -0.5f * depth), float3(0, -1, 0), float3(1, 0, 0), float2(0.0f, 1.0f)},
        shared::Vertex{float3(0.5f * width, 0.0f, -0.5f * depth), float3(0, -1, 0), float3(1, 0, 0), float2(1.0f, 1.0f)},
        shared::Vertex{float3(0.5f * width, 0.0f, 0.5f * depth), float3(0, -1, 0), float3(1, 0, 0), float2(1.0f, 0.0f)},
        shared::Vertex{float3(-0.5f * width, 0.0f, 0.5f * depth), float3(0, -1, 0), float3(1, 0, 0), float2(0.0f, 0.0f)},
    };
    std::vector<shared::Triangle> triangles = {
        shared::Triangle{0, 1, 2},
        shared::Triangle{0, 2, 3},
    };
    GeometryInstance* geomInst = createGeometryInstance(cuContext, scene, vertices, triangles, material);
    scene->geomInsts.push_back(geomInst);

    std::set<const GeometryInstance*> srcGeomInsts = { geomInst };
    GeometryGroup* geomGroup = createGeometryGroup(scene, srcGeomInsts);
    scene->geomGroups.push_back(geomGroup);

    auto mesh = new Mesh();
    Mesh::Group g = {};
    g.geomGroup = geomGroup;
    g.transform = transform;
    mesh->groups.clear();
    mesh->groups.push_back(g);
    scene->meshes[meshName] = mesh;
}

void createSphereLight(
    const std::string &meshName,
    float radius,
    const float3 &reflectance,
    const std::filesystem::path &emittancePath,
    const float3 &immEmittance,
    const float3 &position,
    CUcontext cuContext, Scene* scene) {
    Material* material;
    if constexpr (useLambertMaterial)
        material = createLambertMaterial(cuContext, scene, "", reflectance, "", emittancePath, immEmittance);
    else
        material = createDiffuseAndSpecularMaterial(
            cuContext, scene, "", reflectance, "", float3(0.0f), 0.3f,
            "",
            emittancePath, immEmittance);
    scene->materials.push_back(material);

    constexpr uint32_t numZenithSegments = 8;
    constexpr uint32_t numAzimuthSegments = 16;
    constexpr uint32_t numVertices = 2 + (numZenithSegments - 1) * numAzimuthSegments;
    constexpr uint32_t numTriangles = (2 + 2 * (numZenithSegments - 2)) * numAzimuthSegments;
    constexpr float zenithDelta = pi_v<float> / numZenithSegments;
    constexpr float azimushDelta = 2 * pi_v<float> / numAzimuthSegments;
    std::vector<shared::Vertex> vertices(numVertices);
    std::vector<shared::Triangle> triangles(numTriangles);
    uint32_t vIdx = 0;
    uint32_t triIdx = 0;
    vertices[vIdx++] = shared::Vertex{ float3(0, radius, 0), float3(0, 1, 0), float3(1, 0, 0), float2(0, 0) };
    {
        float zenith = zenithDelta;
        float2 texCoord = float2(0, zenith / pi_v<float>);
        for (int aIdx = 0; aIdx < numAzimuthSegments; ++aIdx) {
            float azimuth = aIdx * azimushDelta;
            float3 n = float3(std::cos(azimuth) * std::sin(zenith),
                              std::cos(zenith),
                              std::sin(azimuth) * std::sin(zenith));
            float3 tc0Dir = float3(-std::sin(azimuth), 0, std::cos(azimuth));
            uint32_t lrIdx = 1 + aIdx;
            uint32_t llIdx = 1 + (aIdx + 1) % numAzimuthSegments;
            uint32_t uIdx = 0;
            texCoord.x = azimuth / (2 * pi_v<float>);
            vertices[vIdx++] = shared::Vertex{ radius * n, n, tc0Dir, texCoord };
            triangles[triIdx++] = shared::Triangle{ llIdx, lrIdx, uIdx };
        }
    }
    for (int zIdx = 1; zIdx < numZenithSegments - 1; ++zIdx) {
        float zenith = (zIdx + 1) * zenithDelta;
        float2 texCoord = float2(0, zenith / pi_v<float>);
        uint32_t baseVIdx = vIdx;
        for (int aIdx = 0; aIdx < numAzimuthSegments; ++aIdx) {
            float azimuth = aIdx * azimushDelta;
            float3 n = float3(std::cos(azimuth) * std::sin(zenith),
                              std::cos(zenith),
                              std::sin(azimuth) * std::sin(zenith));
            float3 tc0Dir = float3(-std::sin(azimuth), 0, std::cos(azimuth));
            texCoord.x = azimuth / (2 * pi_v<float>);
            vertices[vIdx++] = shared::Vertex{ radius * n, n, tc0Dir, texCoord };
            uint32_t lrIdx = baseVIdx + aIdx;
            uint32_t llIdx = baseVIdx + (aIdx + 1) % numAzimuthSegments;
            uint32_t ulIdx = baseVIdx - numAzimuthSegments + (aIdx + 1) % numAzimuthSegments;
            uint32_t urIdx = baseVIdx - numAzimuthSegments + aIdx;
            triangles[triIdx++] = shared::Triangle{ llIdx, lrIdx, urIdx };
            triangles[triIdx++] = shared::Triangle{ llIdx, urIdx, ulIdx };
        }
    }
    vertices[vIdx++] = shared::Vertex{ float3(0, -radius, 0), float3(0, -1, 0), float3(1, 0, 0), float2(0, 1) };
    {
        for (int aIdx = 0; aIdx < numAzimuthSegments; ++aIdx) {
            uint32_t lIdx = numVertices - 1;
            uint32_t ulIdx = numVertices - 1 - numAzimuthSegments + (aIdx + 1) % numAzimuthSegments;
            uint32_t urIdx = numVertices - 1 - numAzimuthSegments + aIdx;
            triangles[triIdx++] = shared::Triangle{ lIdx, urIdx, ulIdx };
        }
    }
    GeometryInstance* geomInst = createGeometryInstance(cuContext, scene, vertices, triangles, material);
    scene->geomInsts.push_back(geomInst);

    std::set<const GeometryInstance*> srcGeomInsts = { geomInst };
    GeometryGroup* geomGroup = createGeometryGroup(scene, srcGeomInsts);
    scene->geomGroups.push_back(geomGroup);

    auto mesh = new Mesh();
    Mesh::Group g = {};
    g.geomGroup = geomGroup;
    g.transform = Matrix4x4();
    mesh->groups.clear();
    mesh->groups.push_back(g);
    scene->meshes[meshName] = mesh;
}

Instance* createInstance(
    CUcontext cuContext, Scene* scene,
    const GeometryGroup* geomGroup,
    const Matrix4x4 &transform) {
    shared::InstanceData* instDataOnHost = scene->instDataBuffer[0].getMappedPointer();

    float3 scale;
    transform.decompose(&scale, nullptr, nullptr);
    float uniformScale = scale.x;

    // JP: 各ジオメトリインスタンスの光源サンプリングに関わるインポータンスは
    //     プリミティブのインポータンスの合計値とする。
    // EN: Use the sum of importance values of primitives as each geometry instances's importance
    //     for sampling a light source
    std::vector<uint32_t> geomInstSlots;
    bool hasEmitterGeomInsts = false;
    for (auto it = geomGroup->geomInsts.cbegin(); it != geomGroup->geomInsts.cend(); ++it) {
        const GeometryInstance* geomInst = *it;
        geomInstSlots.push_back(geomInst->geomInstSlot);
        if (geomInst->mat->emittance)
            hasEmitterGeomInsts = true;
    }

    if (hasEmitterGeomInsts &&
        (std::fabs(scale.y - uniformScale) / uniformScale >= 0.001f ||
         std::fabs(scale.z - uniformScale) / uniformScale >= 0.001f ||
         uniformScale <= 0.0f)) {
        hpprintf("Non-uniform scaling (%g, %g, %g) is not recommended for a light source instance.\n",
                 scale.x, scale.y, scale.z);
    }

    Instance* inst = new Instance();
    inst->geomGroup = geomGroup;
    inst->geomInstSlots.initialize(cuContext, Scene::bufferType, geomInstSlots);
    if (hasEmitterGeomInsts) {
#if USE_PROBABILITY_TEXTURE
        inst->lightGeomInstDist.initialize(cuContext, geomInstSlots.size());
#else
        inst->lightGeomInstDist.initialize(cuContext, Scene::bufferType, nullptr, geomInstSlots.size());
#endif
    }
    inst->instSlot = scene->instSlotFinder.getFirstAvailableSlot();
    scene->instSlotFinder.setInUse(inst->instSlot);

    shared::InstanceData instData = {};
    instData.transform = transform;
    instData.prevTransform = transform;
    instData.normalMatrix = transpose(inverse(transform.getUpperLeftMatrix()));
    instData.uniformScale = uniformScale;
    instData.geomInstSlots = inst->geomInstSlots.getDevicePointer();
    inst->lightGeomInstDist.getDeviceType(&instData.lightGeomInstDist);
    instDataOnHost[inst->instSlot] = instData;

    inst->optixInst = scene->optixScene.createInstance();
    inst->optixInst.setID(inst->instSlot);
    inst->optixInst.setChild(geomGroup->optixGas);
    float xfm[12] = {
        transform.m00, transform.m01, transform.m02, transform.m03,
        transform.m10, transform.m11, transform.m12, transform.m13,
        transform.m20, transform.m21, transform.m22, transform.m23,
    };
    inst->optixInst.setTransform(xfm);

    return inst;
}

void loadEnvironmentalTexture(
    const std::filesystem::path &filePath,
    CUcontext cuContext,
    cudau::Array* envLightArray, CUtexObject* envLightTexture,
    RegularConstantContinuousDistribution2D* envLightImportanceMap) {
    cudau::TextureSampler sampler_float;
    sampler_float.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_float.setWrapMode(0, cudau::TextureWrapMode::Clamp);
    sampler_float.setWrapMode(1, cudau::TextureWrapMode::Clamp);
    sampler_float.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_float.setReadMode(cudau::TextureReadMode::ElementType);

    int32_t width, height;
    float* textureData;
    const char* errMsg = nullptr;
    int ret = LoadEXR(&textureData, &width, &height, filePath.string().c_str(), &errMsg);
    if (ret == TINYEXR_SUCCESS) {
        float* importanceData = new float[width * height];
        for (int y = 0; y < height; ++y) {
            float theta = pi_v<float> * (y + 0.5f) / height;
            float sinTheta = std::sin(theta);
            for (int x = 0; x < width; ++x) {
                uint32_t idx = 4 * (y * width + x);
                textureData[idx + 0] = std::max(textureData[idx + 0], 0.0f);
                textureData[idx + 1] = std::max(textureData[idx + 1], 0.0f);
                textureData[idx + 2] = std::max(textureData[idx + 2], 0.0f);
                float3 value(textureData[idx + 0],
                             textureData[idx + 1],
                             textureData[idx + 2]);
                importanceData[y * width + x] = sRGB_calcLuminance(value) * sinTheta;
            }
        }

        envLightArray->initialize2D(
            cuContext, cudau::ArrayElementType::Float32, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            width, height, 1);
        envLightArray->write(textureData, width * height * 4);

        free(textureData);

        envLightImportanceMap->initialize(
            cuContext, Scene::bufferType, importanceData, width, height);
        delete[] importanceData;

        *envLightTexture = sampler_float.createTextureObject(*envLightArray);
    }
    else {
        hpprintf("Failed to read %s\n", filePath.string().c_str());
        hpprintf("%s\n", errMsg);
        FreeEXRErrorMessage(errMsg);
    }
}



void saveImage(const std::filesystem::path &filepath, uint32_t width, uint32_t height, const uint32_t* data) {
    if (filepath.extension() == ".png")
        stbi_write_png(filepath.string().c_str(), width, height, 4, data,
                       width * sizeof(uint32_t));
    else if (filepath.extension() == ".bmp")
        stbi_write_bmp(filepath.string().c_str(), width, height, 4, data);
    else
        Assert_ShouldNotBeCalled();
}

void saveImageHDR(const std::filesystem::path &filepath, uint32_t width, uint32_t height,
                  float brightnessScale,
                  const float4* data) {
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 4;

    std::vector<float> images[4];
    images[0].resize(width * height);
    images[1].resize(width * height);
    images[2].resize(width * height);
    images[3].resize(width * height);

    for (int i = 0; i < width * height; i++) {
        images[0][i] = brightnessScale * data[i].x;
        images[1][i] = brightnessScale * data[i].y;
        images[2][i] = brightnessScale * data[i].z;
        images[3][i] = brightnessScale * data[i].w;
    }

    float* image_ptr[4];
    image_ptr[0] = &(images[3].at(0)); // A
    image_ptr[1] = &(images[2].at(0)); // B
    image_ptr[2] = &(images[1].at(0)); // G
    image_ptr[3] = &(images[0].at(0)); // R

    image.images = (unsigned char**)image_ptr;
    image.width = width;
    image.height = height;

    header.num_channels = 4;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "A", 255); header.channels[0].name[strlen("A")] = '\0';
    strncpy(header.channels[1].name, "B", 255); header.channels[1].name[strlen("B")] = '\0';
    strncpy(header.channels[2].name, "G", 255); header.channels[2].name[strlen("G")] = '\0';
    strncpy(header.channels[3].name, "R", 255); header.channels[3].name[strlen("R")] = '\0';

    header.pixel_types = (int32_t*)malloc(sizeof(int32_t) * header.num_channels);
    header.requested_pixel_types = (int32_t*)malloc(sizeof(int32_t) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }

    const char* err = nullptr;
    int32_t ret = SaveEXRImageToFile(&image, &header, filepath.string().c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "Save EXR err: %s\n", err);
        FreeEXRErrorMessage(err);
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}

void saveImage(const std::filesystem::path &filepath, uint32_t width, uint32_t height, const float4* data,
               float brightnessScale,
               bool applyToneMap, bool apply_sRGB_gammaCorrection) {
    auto image = new uint32_t[width * height];
    for (int y = 0; y < static_cast<int32_t>(height); ++y) {
        for (int x = 0; x < static_cast<int32_t>(width); ++x) {
            float4 src = data[y * width + x];
            if (applyToneMap) {
                float3 rgb = make_float3(src);
                float lum = sRGB_calcLuminance(rgb);
                float lumT = simpleToneMap_s(brightnessScale * lum);
                float s = lumT / lum;
                src.x = rgb.x * s;
                src.y = rgb.y * s;
                src.z = rgb.z * s;
            }
            if (apply_sRGB_gammaCorrection) {
                src.x = sRGB_gamma_s(src.x);
                src.y = sRGB_gamma_s(src.y);
                src.z = sRGB_gamma_s(src.z);
            }
            uint32_t &dst = image[y * width + x];
            dst = ((std::min<uint32_t>(static_cast<uint32_t>(src.x * 255), 255) << 0) |
                   (std::min<uint32_t>(static_cast<uint32_t>(src.y * 255), 255) << 8) |
                   (std::min<uint32_t>(static_cast<uint32_t>(src.z * 255), 255) << 16) |
                   (std::min<uint32_t>(static_cast<uint32_t>(src.w * 255), 255) << 24));
        }
    }

    saveImage(filepath, width, height, image);

    delete[] image;
}

void saveImage(const std::filesystem::path &filepath,
               uint32_t width, cudau::TypedBuffer<float4> &buffer,
               float brightnessScale,
               bool applyToneMap, bool apply_sRGB_gammaCorrection) {
    Assert(buffer.numElements() % width == 0, "Buffer's length is not divisible by the width.");
    uint32_t height = buffer.numElements() / width;
    auto data = buffer.map();
    saveImage(filepath, width, height, data,
              brightnessScale,
              applyToneMap, apply_sRGB_gammaCorrection);
    buffer.unmap();
}

void saveImage(const std::filesystem::path &filepath,
               cudau::Array &array,
               float brightnessScale,
               bool applyToneMap, bool apply_sRGB_gammaCorrection) {
    auto data = array.map<float4>();
    saveImage(filepath, array.getWidth(), array.getHeight(), data,
              brightnessScale,
              applyToneMap, apply_sRGB_gammaCorrection);
    array.unmap();
}
