#pragma once

#include "../common/common_shared.h"
#if !defined(__CUDA_ARCH__)
#include <cfenv>
#endif

namespace shared {

template <typename T>
concept Number = std::integral<T> || std::floating_point<T>;



template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE void swap(T &a, T &b) {
    T temp = a;
    a = b;
    b = temp;
}



enum class RoundingMode {
    N = 0,
    U,
    D,
    Z
};

template <RoundingMode roundingMode, std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE FloatType add(FloatType a, FloatType b) {
#if defined(__CUDA_ARCH__)
    FloatType ret;
    if constexpr (roundingMode == RoundingMode::N)
        ret = __fadd_rn(a, b);
    else if constexpr (roundingMode == RoundingMode::U)
        ret = __fadd_ru(a, b);
    else if constexpr (roundingMode == RoundingMode::D)
        ret = __fadd_rd(a, b);
    else
        ret = __fadd_rz(a, b);
#else
    if constexpr (roundingMode == RoundingMode::N)
        std::fesetround(FE_TONEAREST);
    else if constexpr (roundingMode == RoundingMode::U)
        std::fesetround(FE_UPWARD);
    else if constexpr (roundingMode == RoundingMode::D)
        std::fesetround(FE_DOWNWARD);
    else
        std::fesetround(FE_TOWARDZERO);
    FloatType ret = a + b;
#endif
    return ret;
}

template <RoundingMode roundingMode, std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE FloatType sub(FloatType a, FloatType b) {
#if defined(__CUDA_ARCH__)
    FloatType ret;
    if constexpr (roundingMode == RoundingMode::N)
        ret = __fsub_rn(a, b);
    else if constexpr (roundingMode == RoundingMode::U)
        ret = __fsub_ru(a, b);
    else if constexpr (roundingMode == RoundingMode::D)
        ret = __fsub_rd(a, b);
    else
        ret = __fsub_rz(a, b);
#else
    if constexpr (roundingMode == RoundingMode::N)
        std::fesetround(FE_TONEAREST);
    else if constexpr (roundingMode == RoundingMode::U)
        std::fesetround(FE_UPWARD);
    else if constexpr (roundingMode == RoundingMode::D)
        std::fesetround(FE_DOWNWARD);
    else
        std::fesetround(FE_TOWARDZERO);
    FloatType ret = a - b;
#endif
    return ret;
}

template <RoundingMode roundingMode, std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE FloatType mul(FloatType a, FloatType b) {
#if defined(__CUDA_ARCH__)
    FloatType ret;
    if constexpr (roundingMode == RoundingMode::N)
        ret = __fmul_rn(a, b);
    else if constexpr (roundingMode == RoundingMode::U)
        ret = __fmul_ru(a, b);
    else if constexpr (roundingMode == RoundingMode::D)
        ret = __fmul_rd(a, b);
    else
        ret = __fmul_rz(a, b);
#else
    if constexpr (roundingMode == RoundingMode::N)
        std::fesetround(FE_TONEAREST);
    else if constexpr (roundingMode == RoundingMode::U)
        std::fesetround(FE_UPWARD);
    else if constexpr (roundingMode == RoundingMode::D)
        std::fesetround(FE_DOWNWARD);
    else
        std::fesetround(FE_TOWARDZERO);
    FloatType ret = a * b;
#endif
    return ret;
}

template <RoundingMode roundingMode, std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE FloatType div(FloatType a, FloatType b) {
#if defined(__CUDA_ARCH__)
    FloatType ret;
    if constexpr (roundingMode == RoundingMode::N)
        ret = __fdiv_rn(a, b);
    else if constexpr (roundingMode == RoundingMode::U)
        ret = __fdiv_ru(a, b);
    else if constexpr (roundingMode == RoundingMode::D)
        ret = __fdiv_rd(a, b);
    else
        ret = __fdiv_rz(a, b);
#else
    if constexpr (roundingMode == RoundingMode::N)
        std::fesetround(FE_TONEAREST);
    else if constexpr (roundingMode == RoundingMode::U)
        std::fesetround(FE_UPWARD);
    else if constexpr (roundingMode == RoundingMode::D)
        std::fesetround(FE_DOWNWARD);
    else
        std::fesetround(FE_TOWARDZERO);
    FloatType ret = a / b;
#endif
    return ret;
}

template <RoundingMode roundingMode, std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE FloatType sqrt(FloatType x) {
#if defined(__CUDA_ARCH__)
    FloatType ret;
    if constexpr (roundingMode == RoundingMode::N)
        ret = __fsqrt_rn(x);
    else if constexpr (roundingMode == RoundingMode::U)
        ret = __fsqrt_ru(x);
    else if constexpr (roundingMode == RoundingMode::D)
        ret = __fsqrt_rd(x);
    else
        ret = __fsqrt_rz(x);
#else
    if constexpr (roundingMode == RoundingMode::N)
        std::fesetround(FE_TONEAREST);
    else if constexpr (roundingMode == RoundingMode::U)
        std::fesetround(FE_UPWARD);
    else if constexpr (roundingMode == RoundingMode::D)
        std::fesetround(FE_DOWNWARD);
    else
        std::fesetround(FE_TOWARDZERO);
    FloatType ret = std::sqrt(x);
#endif
    return ret;
}

template <RoundingMode roundingMode, std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE FloatType &addAssign(FloatType &a, FloatType b) {
#if defined(__CUDA_ARCH__)
    if constexpr (roundingMode == RoundingMode::N)
        a = __fadd_rn(a, b);
    else if constexpr (roundingMode == RoundingMode::U)
        a = __fadd_ru(a, b);
    else if constexpr (roundingMode == RoundingMode::D)
        a = __fadd_rd(a, b);
    else
        a = __fadd_rz(a, b);
#else
    if constexpr (roundingMode == RoundingMode::N)
        std::fesetround(FE_TONEAREST);
    else if constexpr (roundingMode == RoundingMode::U)
        std::fesetround(FE_UPWARD);
    else if constexpr (roundingMode == RoundingMode::D)
        std::fesetround(FE_DOWNWARD);
    else
        std::fesetround(FE_TOWARDZERO);
    a += b;
#endif
    return a;
}

template <RoundingMode roundingMode, std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE FloatType &subAssign(FloatType &a, FloatType b) {
#if defined(__CUDA_ARCH__)
    if constexpr (roundingMode == RoundingMode::N)
        a = __fsub_rn(a, b);
    else if constexpr (roundingMode == RoundingMode::U)
        a = __fsub_ru(a, b);
    else if constexpr (roundingMode == RoundingMode::D)
        a = __fsub_rd(a, b);
    else
        a = __fsub_rz(a, b);
#else
    if constexpr (roundingMode == RoundingMode::N)
        std::fesetround(FE_TONEAREST);
    else if constexpr (roundingMode == RoundingMode::U)
        std::fesetround(FE_UPWARD);
    else if constexpr (roundingMode == RoundingMode::D)
        std::fesetround(FE_DOWNWARD);
    else
        std::fesetround(FE_TOWARDZERO);
    a -= b;
#endif
    return a;
}

template <RoundingMode roundingMode, std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE FloatType &mulAssign(FloatType &a, FloatType b) {
#if defined(__CUDA_ARCH__)
    if constexpr (roundingMode == RoundingMode::N)
        a = __fmul_rn(a, b);
    else if constexpr (roundingMode == RoundingMode::U)
        a = __fmul_ru(a, b);
    else if constexpr (roundingMode == RoundingMode::D)
        a = __fmul_rd(a, b);
    else
        a = __fmul_rz(a, b);
#else
    if constexpr (roundingMode == RoundingMode::N)
        std::fesetround(FE_TONEAREST);
    else if constexpr (roundingMode == RoundingMode::U)
        std::fesetround(FE_UPWARD);
    else if constexpr (roundingMode == RoundingMode::D)
        std::fesetround(FE_DOWNWARD);
    else
        std::fesetround(FE_TOWARDZERO);
    a *= b;
#endif
    return a;
}

template <RoundingMode roundingMode, std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE FloatType &divAssign(FloatType &a, FloatType b) {
#if defined(__CUDA_ARCH__)
    if constexpr (roundingMode == RoundingMode::N)
        a = __fdiv_rn(a, b);
    else if constexpr (roundingMode == RoundingMode::U)
        a = __fdiv_ru(a, b);
    else if constexpr (roundingMode == RoundingMode::D)
        a = __fdiv_rd(a, b);
    else
        a = __fdiv_rz(a, b);
#else
    if constexpr (roundingMode == RoundingMode::N)
        std::fesetround(FE_TONEAREST);
    else if constexpr (roundingMode == RoundingMode::U)
        std::fesetround(FE_UPWARD);
    else if constexpr (roundingMode == RoundingMode::D)
        std::fesetround(FE_DOWNWARD);
    else
        std::fesetround(FE_TOWARDZERO);
    a /= b;
#endif
    return a;
}

template <RoundingMode roundingMode>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setRoundingMode() {
#if !defined(__CUDA_ARCH__)
    if constexpr (roundingMode == RoundingMode::N)
        std::fesetround(FE_TONEAREST);
    else if constexpr (roundingMode == RoundingMode::U)
        std::fesetround(FE_UPWARD);
    else if constexpr (roundingMode == RoundingMode::D)
        std::fesetround(FE_DOWNWARD);
    else
        std::fesetround(FE_TOWARDZERO);
#endif
}





template <std::floating_point FloatType>
class IAFloat {
#if !defined(__CUDA_ARCH__)
    static bool s_truncateImaginaryValue;
#endif

    FloatType m_lo;
    FloatType m_hi;

public:
#if !defined(__CUDA_ARCH__)
    static void setImaginaryValueHandling(bool truncate) {
        s_truncateImaginaryValue = truncate;
    }
#endif

    static bool truncateImaginaryValue() {
#if defined(__CUDA_ARCH__)
        return true;
#else
        return s_truncateImaginaryValue;
#endif
    }

    CUDA_DEVICE_FUNCTION IAFloat(FloatType x = 0) : m_lo(x), m_hi(x) {}
    CUDA_DEVICE_FUNCTION IAFloat(FloatType lo, FloatType hi) : m_lo(lo), m_hi(hi) {
        if (m_hi < m_lo)
            swap(m_lo, m_hi);
    }

    CUDA_DEVICE_FUNCTION FloatType &lo() {
        return m_lo;
    }
    CUDA_DEVICE_FUNCTION FloatType &hi() {
        return m_hi;
    }
    CUDA_DEVICE_FUNCTION const FloatType &lo() const {
        return m_lo;
    }
    CUDA_DEVICE_FUNCTION const FloatType &hi() const {
        return m_hi;
    }
    CUDA_DEVICE_FUNCTION FloatType center() const {
        return (m_lo + m_hi) / 2;
    }
    CUDA_DEVICE_FUNCTION FloatType radius() const {
        FloatType c = center();
        FloatType d = std::max(
            sub<RoundingMode::U>(c - m_lo),
            sub<RoundingMode::U>(m_hi - c));
        setRoundingMode<RoundingMode::N>();
        return d;
    }

    CUDA_DEVICE_FUNCTION IAFloat operator+() const {
        return *this;
    }
    CUDA_DEVICE_FUNCTION IAFloat operator-() const {
        IAFloat ret;
        ret.m_lo = -m_hi;
        ret.m_hi = -m_lo;
        return ret;
    }

    CUDA_DEVICE_FUNCTION IAFloat &operator+=(const IAFloat &r) {
        addAssign<RoundingMode::D>(m_lo, r.m_lo);
        addAssign<RoundingMode::U>(m_hi, r.m_hi);
        setRoundingMode<RoundingMode::N>();

        return *this;
    }
    CUDA_DEVICE_FUNCTION IAFloat &operator+=(FloatType r) {
        return *this += IAFloat(r);
    }
    CUDA_DEVICE_FUNCTION IAFloat &operator-=(const IAFloat &r) {
        subAssign<RoundingMode::D>(m_lo, r.m_hi);
        subAssign<RoundingMode::U>(m_hi, r.m_lo);
        setRoundingMode<RoundingMode::N>();

        return *this;
    }
    CUDA_DEVICE_FUNCTION IAFloat &operator-=(FloatType r) {
        return *this -= IAFloat(r);
    }
    CUDA_DEVICE_FUNCTION IAFloat &operator*=(const IAFloat &r) {
        IAFloat l = *this;
        if (l.m_lo >= 0.0f) {
            if (r.m_lo >= 0.0f) {
                m_lo = mul<RoundingMode::D>(l.m_lo, r.m_lo);
                m_hi = mul<RoundingMode::U>(l.m_hi, r.m_hi);
            }
            else if (r.m_hi <= 0.0f) {
                m_lo = mul<RoundingMode::D>(l.m_hi, r.m_lo);
                m_hi = mul<RoundingMode::U>(l.m_lo, r.m_hi);
            }
            else {
                m_lo = mul<RoundingMode::D>(l.m_hi, r.m_lo);
                m_hi = mul<RoundingMode::U>(l.m_hi, r.m_hi);
            }
        }
        else if (l.m_hi <= 0.0f) {
            if (r.m_lo >= 0.0f) {
                m_lo = mul<RoundingMode::D>(l.m_lo, r.m_hi);
                m_hi = mul<RoundingMode::U>(l.m_hi, r.m_lo);
            }
            else if (r.m_hi <= 0.0f) {
                m_lo = mul<RoundingMode::D>(l.m_hi, r.m_hi);
                m_hi = mul<RoundingMode::U>(l.m_lo, r.m_lo);
            }
            else {
                m_lo = mul<RoundingMode::D>(l.m_lo, r.m_hi);
                m_hi = mul<RoundingMode::U>(l.m_lo, r.m_lo);
            }
        }
        else {
            if (r.m_lo >= 0.0f) {
                m_lo = mul<RoundingMode::D>(l.m_lo, r.m_hi);
                m_hi = mul<RoundingMode::U>(l.m_hi, r.m_hi);
            }
            else if (r.m_hi <= 0.0f) {
                m_lo = mul<RoundingMode::D>(l.m_hi, r.m_lo);
                m_hi = mul<RoundingMode::U>(l.m_lo, r.m_lo);
            }
            else {
                m_lo = std::fmin(mul<RoundingMode::D>(l.m_lo, r.m_hi), mul<RoundingMode::D>(l.m_hi, r.m_lo));
                m_hi = std::fmax(mul<RoundingMode::U>(l.m_lo, r.m_lo), mul<RoundingMode::U>(l.m_hi, r.m_hi));
            }
        }
        setRoundingMode<RoundingMode::N>();

        return *this;
    }
    CUDA_DEVICE_FUNCTION IAFloat &operator*=(FloatType r) {
        IAFloat l = *this;
        if (r >= 0.0f) {
            m_lo = mul<RoundingMode::D>(l.m_lo, r);
            m_hi = mul<RoundingMode::U>(l.m_hi, r);
        }
        else {
            m_lo = mul<RoundingMode::D>(l.m_hi, r);
            m_hi = mul<RoundingMode::U>(l.m_lo, r);
        }
        setRoundingMode<RoundingMode::N>();

        return *this;
    }
    CUDA_DEVICE_FUNCTION IAFloat &operator/=(const IAFloat &r) {
        IAFloat l = *this;
        if (r.m_lo > 0.0f) {
            if (l.m_lo >= 0.0f) {
                m_lo = div<RoundingMode::D>(l.m_lo, r.m_hi);
                m_hi = div<RoundingMode::U>(l.m_hi, r.m_lo);
            }
            else if (l.m_hi <= 0.0f) {
                m_lo = div<RoundingMode::D>(l.m_lo, r.m_lo);
                m_hi = div<RoundingMode::U>(l.m_hi, r.m_hi);
            }
            else {
                m_lo = div<RoundingMode::D>(l.m_lo, r.m_lo);
                m_hi = div<RoundingMode::U>(l.m_hi, r.m_lo);
            }
        }
        else if (r.m_hi < 0.0f) {
            if (l.m_lo >= 0.0f) {
                m_lo = div<RoundingMode::D>(l.m_hi, r.m_hi);
                m_hi = div<RoundingMode::U>(l.m_lo, r.m_lo);
            }
            else if (l.m_hi <= 0.0f) {
                m_lo = div<RoundingMode::D>(l.m_hi, r.m_lo);
                m_hi = div<RoundingMode::U>(l.m_lo, r.m_hi);
            }
            else {
                m_lo = div<RoundingMode::D>(l.m_hi, r.m_hi);
                m_hi = div<RoundingMode::U>(l.m_lo, r.m_hi);
            }
        }
        else {
            setRoundingMode<RoundingMode::N>();
#if defined(__CUDA_ARCH__)
            Assert(false, "IAFloat: division by 0.");
#else
            throw std::domain_error("IAFloat: division by 0.");
#endif
        }
        setRoundingMode<RoundingMode::N>();

        return *this;
    }
    CUDA_DEVICE_FUNCTION IAFloat &operator/=(FloatType r) {
        IAFloat l = *this;
        if (r > 0.0f) {
            m_lo = div<RoundingMode::D>(l.m_lo, r);
            m_hi = div<RoundingMode::U>(l.m_hi, r);
        }
        else if (r < 0.0f) {
            m_lo = div<RoundingMode::D>(l.m_hi, r);
            m_hi = div<RoundingMode::U>(l.m_lo, r);
        }
        else {
            setRoundingMode<RoundingMode::N>();
#if defined(__CUDA_ARCH__)
            Assert(false, "IAFloat: division by 0.");
#else
            throw std::domain_error("IAFloat: division by 0.");
#endif
        }
        setRoundingMode<RoundingMode::N>();

        return *this;
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE friend IAFloat abs(const IAFloat &v) {
        IAFloat ret;
        FloatType absLo = std::fabs(v.m_lo);
        FloatType absHi = std::fabs(v.m_hi);
        if (absLo > absHi)
            swap(absLo, absHi);
        ret.m_lo = absLo;
        ret.m_hi = absHi;

        return ret;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE friend IAFloat pow2(const IAFloat &v) {
        IAFloat ret;
        FloatType absLo = std::fabs(v.m_lo);
        FloatType absHi = std::fabs(v.m_hi);
        if (absLo > absHi)
            swap(absLo, absHi);
        ret.m_lo = mul<RoundingMode::D>(absLo, absLo);
        ret.m_hi = mul<RoundingMode::U>(absHi, absHi);
        setRoundingMode<RoundingMode::N>();

        return ret;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE friend IAFloat sqrt(const IAFloat &v) {
        IAFloat ret;
        IAFloat mv = v;
        if (mv.m_lo < 0.0f) {
            if (truncateImaginaryValue())
                mv.m_lo = 0.0f;
            else
#if defined(__CUDA_ARCH__)
                Assert(false, "IAFloat: sqrt of a negative value.");
#else
                throw std::domain_error("IAFloat: sqrt of a negative value.");
#endif
        }

        ret.m_lo = sqrt<RoundingMode::D>(mv.m_lo);
        ret.m_hi = sqrt<RoundingMode::U>(mv.m_hi);
        setRoundingMode<RoundingMode::N>();

        return ret;
    }
};

#if !defined(__CUDA_ARCH__)
template <std::floating_point FloatType>
bool IAFloat<FloatType>::s_truncateImaginaryValue = true;
#endif



template <std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE IAFloat<FloatType> operator+(
    const IAFloat<FloatType> &a, const IAFloat<FloatType> &b) {
    IAFloat<FloatType> ret = a;
    ret += b;
    return ret;
}

template <std::floating_point FloatType, Number N>
CUDA_DEVICE_FUNCTION CUDA_INLINE IAFloat<FloatType> operator+(
    const IAFloat<FloatType> &a, N b) {
    IAFloat<FloatType> ret = a;
    ret += static_cast<FloatType>(b);
    return ret;
}

template <Number N, std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE IAFloat<FloatType> operator+(
    N a, const IAFloat<FloatType> &b) {
    IAFloat<FloatType> ret = b;
    ret += static_cast<FloatType>(a);
    return ret;
}

template <std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE IAFloat<FloatType> operator-(
    const IAFloat<FloatType> &a, const IAFloat<FloatType> &b) {
    IAFloat<FloatType> ret = a;
    ret -= b;
    return ret;
}

template <std::floating_point FloatType, Number N>
CUDA_DEVICE_FUNCTION CUDA_INLINE IAFloat<FloatType> operator-(
    const IAFloat<FloatType> &a, N b) {
    IAFloat<FloatType> ret = a;
    ret -= static_cast<FloatType>(b);
    return ret;
}

template <Number N, std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE IAFloat<FloatType> operator-(
    N a, const IAFloat<FloatType> &b) {
    IAFloat<FloatType> ret = -b;
    ret += static_cast<FloatType>(a);
    return ret;
}

template <std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE IAFloat<FloatType> operator*(
    const IAFloat<FloatType> &a, const IAFloat<FloatType> &b) {
    IAFloat<FloatType> ret = a;
    ret *= b;
    return ret;
}

template <std::floating_point FloatType, Number N>
CUDA_DEVICE_FUNCTION CUDA_INLINE IAFloat<FloatType> operator*(
    const IAFloat<FloatType> &a, N b) {
    IAFloat<FloatType> ret = a;
    ret *= static_cast<FloatType>(b);
    return ret;
}

template <Number N, std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE IAFloat<FloatType> operator*(
    N a, const IAFloat<FloatType> &b) {
    IAFloat<FloatType> ret = b;
    ret *= static_cast<FloatType>(a);
    return ret;
}

template <std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE IAFloat<FloatType> operator/(
    const IAFloat<FloatType> &a, const IAFloat<FloatType> &b) {
    IAFloat<FloatType> ret = a;
    ret /= b;
    return ret;
}

template <std::floating_point FloatType, Number N>
CUDA_DEVICE_FUNCTION CUDA_INLINE IAFloat<FloatType> operator/(
    const IAFloat<FloatType> &a, N b) {
    IAFloat<FloatType> ret = a;
    ret /= static_cast<FloatType>(b);
    return ret;
}

template <Number N, std::floating_point FloatType>
CUDA_DEVICE_FUNCTION CUDA_INLINE IAFloat<FloatType> operator/(
    N a, const IAFloat<FloatType> &b) {
    IAFloat<FloatType> ret(static_cast<FloatType>(a));
    ret /= b;
    return ret;
}



class AAFloatOn2D {
public:
    using FloatType = float;

private:
    using ia_fp_t = IAFloat<FloatType>;

    FloatType m_centralValue;
    FloatType m_coeffs[2];
    FloatType m_coeffOthers;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static AAFloatOn2D affineApproximation(
        const AAFloatOn2D &v,
        const FloatType alpha, const FloatType beta, FloatType delta) {
        AAFloatOn2D ret(alpha * v.m_centralValue + beta);
        for (int dim = 0; dim < 2; ++dim)
            ret.m_coeffs[dim] = alpha * v.m_coeffs[dim];
        delta += std::fabs(alpha) * v.m_coeffOthers;

        ret.m_coeffOthers = delta;

        return ret;
    }

public:
    CUDA_DEVICE_FUNCTION AAFloatOn2D(
        FloatType centralValue = 0,
        FloatType coeff0 = 0, FloatType coeff1 = 0, FloatType coeffOthers = 0) :
        m_centralValue(centralValue), m_coeffs{ coeff0, coeff1 }, m_coeffOthers(coeffOthers) {}

    CUDA_DEVICE_FUNCTION operator IAFloat<FloatType>() const {
        ia_fp_t ret(m_centralValue);
        for (int dim = 0; dim < 2; ++dim) {
            FloatType d = m_coeffs[dim];
            ret += ia_fp_t(-d, d);
        }
        ret += ia_fp_t(-m_coeffOthers, m_coeffOthers);
        return ret;
    }
    CUDA_DEVICE_FUNCTION IAFloat<FloatType> toIAFloat() const {
        return static_cast<ia_fp_t>(*this);
    }

    CUDA_DEVICE_FUNCTION FloatType getCentralValue() const {
        return m_centralValue;
    }
    template <std::integral I>
    CUDA_DEVICE_FUNCTION FloatType getCoeff(I idx) const {
        return m_coeffs[idx];
    }
    CUDA_DEVICE_FUNCTION FloatType getCoeffOthers() const {
        return m_coeffOthers;
    }

    CUDA_DEVICE_FUNCTION AAFloatOn2D operator+() const {
        return *this;
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D operator-() const {
        AAFloatOn2D ret = *this;
        ret.m_centralValue *= -1;
        for (int dim = 0; dim < 2; ++dim)
            ret.m_coeffs[dim] *= -1;
        return ret;
    }

    CUDA_DEVICE_FUNCTION AAFloatOn2D &operator+=(const AAFloatOn2D &r) {
        m_centralValue += r.m_centralValue;
        for (int dim = 0; dim < 2; ++dim)
            m_coeffs[dim] += r.m_coeffs[dim];
        m_coeffOthers += r.m_coeffOthers;

        return *this;
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D &operator+=(FloatType r) {
        m_centralValue += r;

        return *this;
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D &operator-=(const AAFloatOn2D &r) {
        m_centralValue -= r.m_centralValue;
        for (int dim = 0; dim < 2; ++dim)
            m_coeffs[dim] -= r.m_coeffs[dim];
        m_coeffOthers += r.m_coeffOthers;

        return *this;
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D &operator-=(FloatType r) {
        m_centralValue -= r;

        return *this;
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D &operator*=(const AAFloatOn2D &r) {
        FloatType u = 0;
        FloatType v = 0;
        FloatType c0 = m_centralValue;
        FloatType r_c0 = r.m_centralValue;

        m_centralValue = c0 * r_c0;
        u += std::fabs(m_coeffs[0]);
        v += std::fabs(r.m_coeffs[0]);
        m_coeffs[0] = c0 * r.m_coeffs[0] + r_c0 * m_coeffs[0];
        u += std::fabs(m_coeffs[1]);
        v += std::fabs(r.m_coeffs[1]);
        m_coeffs[1] = c0 * r.m_coeffs[1] + r_c0 * m_coeffs[1];
        u += m_coeffOthers;
        v += r.m_coeffOthers;
        m_coeffOthers = std::fabs(r_c0) * m_coeffOthers + std::fabs(c0) * r.m_coeffOthers;
        m_coeffOthers += u * v;

        return *this;
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D &operator*=(FloatType r) {
        m_centralValue *= r;
        for (int dim = 0; dim < 2; ++dim)
            m_coeffs[dim] *= r;
        m_coeffOthers *= std::fabs(r);

        return *this;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE friend AAFloatOn2D reciprocal(const AAFloatOn2D &v) {
        ia_fp_t interval = static_cast<ia_fp_t>(v);

        FloatType a = interval.lo();
        FloatType b = interval.hi();
        FloatType ab = a * b;
        FloatType sqrtab = (a > 0.0f ? 1 : -1) * std::sqrt(ab);

        FloatType alpha = -1 / (a * b);
        FloatType beta = (a + 2 * sqrtab + b) / (2 * ab);
        FloatType delta = (a - 2 * sqrtab + b) / (2 * ab);

        return affineApproximation(v, alpha, beta, delta);
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D &operator/=(const AAFloatOn2D &r) {
        return *this *= reciprocal(r);
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D &operator/=(FloatType r) {
        if (std::isinf(r)) {
            m_coeffs[0] = 0;
            for (int dim = 0; dim < 2; ++dim)
                m_coeffs[dim] = 0;
            m_coeffOthers = 0;
        }
        else {
            *this *= AAFloatOn2D(1.0f / r);
        }
        return *this;
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE friend AAFloatOn2D recSqrt(const AAFloatOn2D &v) {
        ia_fp_t interval = static_cast<ia_fp_t>(v);

        const auto f = []
        (FloatType x) {
            return 1 / std::sqrt(x);
        };

        FloatType a = interval.lo();
        FloatType b = interval.hi();

        FloatType alpha;
        FloatType beta;
        FloatType delta;
        constexpr bool useMinRange = true;
        if constexpr (useMinRange) {
            alpha = -0.5f * pow3(f(b));
            beta = 0.5f * (f(a) + f(b) - alpha * (a + b));
            delta = 0.5f * std::fabs(f(a) - f(b) - alpha * (a - b));
        }
        else {
            FloatType sqrta = std::sqrt(a);
            FloatType sqrtb = std::sqrt(b);
            FloatType temp = a * sqrtb + b * sqrta;
            FloatType u = std::pow(temp / 2, 2.0f / 3);
            FloatType sqrtu = std::sqrt(u);

            alpha = -1 / temp;
            beta = 3 / (4 * sqrtu) + (a + sqrta * sqrtb + b) / (2 * temp);
            delta = std::fabs(0.5f * (1 / std::sqrt(u) - (a + sqrta * sqrtb + b - u) / temp));
        }

        return affineApproximation(v, alpha, beta, delta);
    }
};

CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D operator+(
    const AAFloatOn2D &a, const AAFloatOn2D &b) {
    AAFloatOn2D ret = a;
    ret += b;
    return ret;
}

template <Number32bit N>
CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D operator+(
    const AAFloatOn2D &a, N b) {
    AAFloatOn2D ret = a;
    ret += static_cast<AAFloatOn2D::FloatType>(b);
    return ret;
}

template <Number32bit N>
CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D operator+(
    N a, const AAFloatOn2D &b) {
    AAFloatOn2D ret = b;
    ret += static_cast<AAFloatOn2D::FloatType>(a);
    return ret;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D operator-(
    const AAFloatOn2D &a, const AAFloatOn2D &b) {
    AAFloatOn2D ret = a;
    ret -= b;
    return ret;
}

template <Number32bit N>
CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D operator-(
    const AAFloatOn2D &a, N b) {
    AAFloatOn2D ret = a;
    ret -= static_cast<AAFloatOn2D::FloatType>(b);
    return ret;
}

template <Number32bit N>
CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D operator-(
    N a, const AAFloatOn2D &b) {
    AAFloatOn2D ret = -b;
    ret += static_cast<AAFloatOn2D::FloatType>(a);
    return ret;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D operator*(
    const AAFloatOn2D &a, const AAFloatOn2D &b) {
    AAFloatOn2D ret = a;
    ret *= b;
    return ret;
}

template <Number32bit N>
CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D operator*(
    const AAFloatOn2D &a, N b) {
    AAFloatOn2D ret = a;
    ret *= static_cast<AAFloatOn2D::FloatType>(b);
    return ret;
}

template <Number32bit N>
CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D operator*(
    N a, const AAFloatOn2D &b) {
    AAFloatOn2D ret = b;
    ret *= static_cast<AAFloatOn2D::FloatType>(a);
    return ret;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D operator/(
    const AAFloatOn2D &a, const AAFloatOn2D &b) {
    AAFloatOn2D ret = a;
    ret /= b;
    return ret;
}

template <Number32bit N>
CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D operator/(
    const AAFloatOn2D &a, N b) {
    AAFloatOn2D ret = a;
    ret /= static_cast<AAFloatOn2D::FloatType>(b);
    return ret;
}

template <Number32bit N>
CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D operator/(
    N a, const AAFloatOn2D &b) {
    AAFloatOn2D ret(static_cast<AAFloatOn2D::FloatType>(a));
    ret /= b;
    return ret;
}



struct AAFloatOn2D_Vector3D {
    AAFloatOn2D x, y, z;

    CUDA_DEVICE_FUNCTION AAFloatOn2D_Vector3D(const AAFloatOn2D &v = 0.0f) :
        x(v), y(v), z(v) {}
    CUDA_DEVICE_FUNCTION AAFloatOn2D_Vector3D(const AAFloatOn2D &xx, const AAFloatOn2D &yy, const AAFloatOn2D &zz) :
        x(xx), y(yy), z(zz) {}
    CUDA_DEVICE_FUNCTION AAFloatOn2D_Vector3D(
        Vector3D centralValue,
        Vector3D coeff0 = 0, Vector3D coeff1 = 0, Vector3D coeffOthers = 0) :
        x(centralValue.x, coeff0.x, coeff1.x, coeffOthers.x),
        y(centralValue.y, coeff0.y, coeff1.y, coeffOthers.y),
        z(centralValue.z, coeff0.z, coeff1.z, coeffOthers.z) {}

    CUDA_DEVICE_FUNCTION Vector3D getCentralValue() const {
        return Vector3D(x.getCentralValue(), y.getCentralValue(), z.getCentralValue());
    }
    template <std::integral I>
    CUDA_DEVICE_FUNCTION Vector3D getCoeff(I idx) const {
        return Vector3D(x.getCoeff(idx), y.getCoeff(idx), z.getCoeff(idx));
    }
    CUDA_DEVICE_FUNCTION Vector3D getCoeffOthers() const {
        return Vector3D(x.getCoeffOthers(), y.getCoeffOthers(), z.getCoeffOthers());
    }

    CUDA_DEVICE_FUNCTION AAFloatOn2D_Vector3D &operator+=(const AAFloatOn2D_Vector3D &r) {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D_Vector3D &operator-=(const AAFloatOn2D_Vector3D &r) {
        x -= r.x;
        y -= r.y;
        z -= r.z;
        return *this;
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D_Vector3D &operator*=(const AAFloatOn2D &r) {
        x *= r;
        y *= r;
        z *= r;
        return *this;
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D_Vector3D &operator*=(const AAFloatOn2D_Vector3D &r) {
        x *= r.x;
        y *= r.y;
        z *= r.z;
        return *this;
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D_Vector3D &operator/=(const AAFloatOn2D &r) {
        AAFloatOn2D rr = 1 / r;
        x *= rr;
        y *= rr;
        z *= rr;
        return *this;
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D_Vector3D &operator/=(const AAFloatOn2D_Vector3D &r) {
        x /= r.x;
        y /= r.y;
        z /= r.z;
        return *this;
    }

    CUDA_DEVICE_FUNCTION AAFloatOn2D sqLength() const {
        Vector3D xc = getCentralValue();
        Vector3D xu = getCoeff(0);
        Vector3D xv = getCoeff(1);
        Vector3D xK = getCoeffOthers();
        AAFloatOn2D::FloatType offset = 0.5f * (abs(xu) + abs(xv) + xK).sqLength();
        return AAFloatOn2D(
            xc.sqLength() + offset,
            2 * dot(xc, xu),
            2 * dot(xc, xv),
            2 * abs(dot(xc, xK)) + offset);
    }
    CUDA_DEVICE_FUNCTION AAFloatOn2D_Vector3D &normalize() {
        AAFloatOn2D l = sqLength();
        *this *= recSqrt(l);
        return *this;
    }
};

CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D_Vector3D operator+(
    const Vector3D &a, const AAFloatOn2D_Vector3D &b) {
    AAFloatOn2D_Vector3D ret(a);
    ret += b;
    return ret;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D_Vector3D operator+(
    const AAFloatOn2D_Vector3D &a, const AAFloatOn2D_Vector3D &b) {
    AAFloatOn2D_Vector3D ret(a);
    ret += b;
    return ret;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D_Vector3D operator*(
    const Vector3D &a, const AAFloatOn2D &b) {
    AAFloatOn2D_Vector3D ret(a);
    ret *= b;
    return ret;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D dot(
    const Vector3D &a, const AAFloatOn2D_Vector3D &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D dot(
    const AAFloatOn2D_Vector3D &a, const AAFloatOn2D_Vector3D &b) {
    Vector3D ac = a.getCentralValue();
    Vector3D bc = a.getCentralValue();
    Vector3D au = a.getCoeff(0);
    Vector3D bu = b.getCoeff(0);
    Vector3D av = a.getCoeff(1);
    Vector3D bv = b.getCoeff(1);
    Vector3D aK = a.getCoeffOthers();
    Vector3D bK = b.getCoeffOthers();
    return AAFloatOn2D(
        dot(ac, bc),
        dot(au, bc) + dot(ac, bu),
        dot(av, bc) + dot(ac, bv),
        abs(dot(aK, bc)) + abs(dot(ac, bK)) + dot(abs(au) + abs(av) + aK, abs(bu) + abs(bv) + bK));
}

CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D_Vector3D operator*(
    const AAFloatOn2D &a, const AAFloatOn2D_Vector3D &b) {
    AAFloatOn2D_Vector3D ret = b;
    ret *= a;
    return ret;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE AAFloatOn2D_Vector3D operator*(
    const Matrix3x3 &a, const AAFloatOn2D_Vector3D &b) {
    return AAFloatOn2D_Vector3D(dot(a.row(0), b), dot(a.row(1), b), dot(a.row(2), b));
}

}
