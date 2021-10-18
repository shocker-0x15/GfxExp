#include "common_host.h"

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
    m_numValues = static_cast<uint32_t>(numValues);
#if defined(USE_WALKER_ALIAS_METHOD)
    m_PMF.initialize(cuContext, type, m_numValues);
    m_aliasTable.initialize(cuContext, type, m_numValues);
    m_valueMaps.initialize(cuContext, type, m_numValues);

    RealType* PMF = m_PMF.map();
    std::memcpy(PMF, values, sizeof(RealType) * m_numValues);

    CompensatedSum<RealType> sum(0);
    for (uint32_t i = 0; i < m_numValues; ++i)
        sum += values[i];
    RealType avgWeight = sum / m_numValues;
    m_integral = sum;

    for (uint32_t i = 0; i < m_numValues; ++i)
        PMF[i] /= m_integral;
    m_PMF.unmap();

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
    m_PMF.initialize(cuContext, type, m_numValues);
    m_CDF.initialize(cuContext, type, m_numValues + 1);

    RealType* PMF = m_PMF.map();
    RealType* CDF = m_CDF.map();
    std::memcpy(PMF, values, sizeof(RealType) * m_numValues);

    CompensatedSum<RealType> sum(0);
    for (uint32_t i = 0; i < m_numValues; ++i) {
        CDF[i] = sum;
        sum += PMF[i];
    }
    m_integral = sum;
    for (uint32_t i = 0; i < m_numValues; ++i) {
        PMF[i] /= m_integral;
        CDF[i] /= m_integral;
    }
    CDF[m_numValues] = 1.0f;

    m_CDF.unmap();
    m_PMF.unmap();
#endif
}

template class DiscreteDistribution1DTemplate<float>;



template <typename RealType>
void RegularConstantContinuousDistribution1DTemplate<RealType>::
initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues) {
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
}

template class RegularConstantContinuousDistribution1DTemplate<float>;



template <typename RealType>
void RegularConstantContinuousDistribution2DTemplate<RealType>::
initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numD1, size_t numD2) {
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
}

template class RegularConstantContinuousDistribution2DTemplate<float>;



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



void saveImage(const std::filesystem::path &filepath, uint32_t width, uint32_t height, const uint32_t* data) {
    if (filepath.extension() == ".png")
        stbi_write_png(filepath.string().c_str(), width, height, 4, data,
                       width * sizeof(uint32_t));
    else if (filepath.extension() == ".bmp")
        stbi_write_bmp(filepath.string().c_str(), width, height, 4, data);
    else
        Assert_ShouldNotBeCalled();
}

void saveImage(const std::filesystem::path &filepath, uint32_t width, uint32_t height, const float4* data,
               bool applyToneMap, bool apply_sRGB_gammaCorrection) {
    auto image = new uint32_t[width * height];
    for (int y = 0; y < static_cast<int32_t>(height); ++y) {
        for (int x = 0; x < static_cast<int32_t>(width); ++x) {
            float4 src = data[y * width + x];
            if (applyToneMap) {
                src.x = simpleToneMap_s(src.x);
                src.y = simpleToneMap_s(src.y);
                src.z = simpleToneMap_s(src.z);
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
               bool applyToneMap, bool apply_sRGB_gammaCorrection) {
    Assert(buffer.numElements() % width == 0, "Buffer's length is not divisible by the width.");
    uint32_t height = buffer.numElements() / width;
    auto data = buffer.map();
    saveImage(filepath, width, height, data, applyToneMap, apply_sRGB_gammaCorrection);
    buffer.unmap();
}

void saveImage(const std::filesystem::path &filepath,
               cudau::Array &array,
               bool applyToneMap, bool apply_sRGB_gammaCorrection) {
    auto data = array.map<float4>();
    saveImage(filepath, array.getWidth(), array.getHeight(), data, applyToneMap, apply_sRGB_gammaCorrection);
    array.unmap();
}
