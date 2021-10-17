#pragma once

#include <chrono>
#include <vector>
#include <stdint.h>

using namespace std::chrono;

enum class StopWatchDurationType : uint32_t {
    Nanoseconds,
    Microseconds,
    Milliseconds,
    Seconds,
};

template <typename res>
class StopWatchTemplate {
    std::vector<typename res::duration> m_measurements;
    std::vector<typename res::time_point> m_startTPStack;

public:
    uint64_t durationCast(const typename res::duration &duration, StopWatchDurationType dt) const {
        switch (dt) {
        case StopWatchDurationType::Nanoseconds:
            return duration_cast<nanoseconds>(duration).count();
        case StopWatchDurationType::Microseconds:
            return duration_cast<microseconds>(duration).count();
        case StopWatchDurationType::Milliseconds:
            return duration_cast<milliseconds>(duration).count();
        case StopWatchDurationType::Seconds:
            return duration_cast<seconds>(duration).count();
        default:
            break;
        }
        return UINT64_MAX;
    }

    void start() {
        m_startTPStack.push_back(res::now());
    }

    uint32_t stop() {
        uint32_t mIdx = 0xFFFFFFFF;
        mIdx = static_cast<uint32_t>(m_measurements.size());
        m_measurements.push_back(res::now() - m_startTPStack.back());
        m_startTPStack.pop_back();
        return mIdx;
    }

    uint64_t getMeasurement(uint32_t index, StopWatchDurationType dt = StopWatchDurationType::Milliseconds) const {
        if (index >= m_measurements.size())
            return UINT64_MAX;
        return durationCast(m_measurements[index], dt);
    }

    uint64_t getElapsed(StopWatchDurationType dt = StopWatchDurationType::Milliseconds) {
        typename res::duration duration = res::now() - m_startTPStack.back();
        return durationCast(duration, dt);
    }

    uint64_t getElapsedFromRoot(StopWatchDurationType dt = StopWatchDurationType::Milliseconds) {
        typename res::duration duration = res::now() - m_startTPStack.front();
        return durationCast(duration, dt);
    }

    void clearAllMeasurements() {
        m_measurements.clear();
    }

    void reset() {
        m_startTPStack.clear();
        m_measurements.clear();
    }
};

using StopWatch = StopWatchTemplate<system_clock>;
using StopWatchHiRes = StopWatchTemplate<high_resolution_clock>;
