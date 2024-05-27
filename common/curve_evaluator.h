#pragma once

#include "basic_types.h"

namespace curve {
    template <OptixPrimitiveType curveType>
    constexpr bool always_false_v = false;

    template <OptixPrimitiveType curveType>
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr uint32_t
        getNumControlPoints() {
        static_assert(always_false_v<curveType>, "Invalid curve type.");
        return 0;
    }

    template <>
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr uint32_t
        getNumControlPoints<OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR>() {
        return 2;
    }

    template <>
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr uint32_t
        getNumControlPoints<OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE>() {
        return 3;
    }

    template <>
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr uint32_t
        getNumControlPoints<OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE>() {
        return 3;
    }

    template <>
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr uint32_t
        getNumControlPoints<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE>() {
        return 4;
    }

    template <>
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr uint32_t
        getNumControlPoints<OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM>() {
        return 4;
    }

    template <>
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr uint32_t
        getNumControlPoints<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER>() {
        return 4;
    }



    template <OptixPrimitiveType curveType>
    struct Interpolator;

    /*
    p0, p1: control points
    u: curve parameter
    P(u) =   (-p0 + p1) * u
           + p0
    */
    template <>
    struct Interpolator<OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR> {
        float4 m_p[2];

        CUDA_DEVICE_FUNCTION Interpolator(const float4 cps[2]) {
            m_p[1] = cps[1] - cps[0];
            m_p[0] = cps[0];
        }

        CUDA_DEVICE_FUNCTION float4 position_radius(float u) const {
            return m_p[1] * u + m_p[0];
        }
        CUDA_DEVICE_FUNCTION float3 position(float u) const {
            return make_float3(m_p[1]) * u + make_float3(m_p[0]);
        }
        CUDA_DEVICE_FUNCTION float radius(float u) const {
            return m_p[1].w * u + m_p[0].w;
        }

        CUDA_DEVICE_FUNCTION float minRadius(float uA, float uB) const {
            return fminf(radius(uA), radius(uB));
        }
        CUDA_DEVICE_FUNCTION float maxRadius(float uA, float uB) const {
            return fmaxf(radius(uA), radius(uB));
        }

        CUDA_DEVICE_FUNCTION float4 dPosition_dRadius(float u) const {
            return m_p[1];
        }
        CUDA_DEVICE_FUNCTION float3 dPosition(float u) const {
            return make_float3(m_p[1]);
        }
        CUDA_DEVICE_FUNCTION float dRadius(float u) const {
            return m_p[1].w;
        }

        CUDA_DEVICE_FUNCTION float4 ddPosition_ddRadius(float u) const {
            return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        CUDA_DEVICE_FUNCTION float3 ddPosition(float u) const {
            return make_float3(0.0f, 0.0f, 0.0f);
        }
    };

    /*
    p0, p1, p2: control points
    u: curve parameter
    P(u) =   (0.5 * p0 - p1 + 0.5 * p2) * u^2
           + (-p0 + p1) * u
           + 0.5 * (p0 + p1)
    */
    template <>
    struct Interpolator<OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE> {
        float4 m_p[3];

        CUDA_DEVICE_FUNCTION Interpolator(const float4 cps[3]) {
            m_p[2] = 0.5f * cps[0] - cps[1] + 0.5f * cps[2];
            m_p[1] = -cps[0] + cps[1];
            m_p[0] = 0.5f * (cps[0] + cps[1]);
        }

        CUDA_DEVICE_FUNCTION float4 position_radius(float u) const {
            return m_p[2] * u * u + m_p[1] * u + m_p[0];
        }
        CUDA_DEVICE_FUNCTION float3 position(float u) const {
            return make_float3(m_p[2]) * u * u + make_float3(m_p[1]) * u + make_float3(m_p[0]);
        }
        CUDA_DEVICE_FUNCTION float radius(float u) const {
            return m_p[2].w * u * u + m_p[1].w * u + m_p[0].w;
        }

        CUDA_DEVICE_FUNCTION float minRadius(float uA, float uB) const {
            float root1 = stc::clamp(-0.5f * m_p[1].w / m_p[2].w, uA, uB);
            return fminf(fminf(radius(uA), radius(uB)), radius(root1));
        }
        CUDA_DEVICE_FUNCTION float maxRadius(float uA, float uB) const {
            if (m_p[1].w == 0.0f && m_p[2].w == 0.0f)
                return m_p[0].w; // constant width
            float root1 = stc::clamp(-0.5f * m_p[1].w / m_p[2].w, uA, uB);
            return fmaxf(fmaxf(radius(uA), radius(uB)), radius(root1));
        }

        CUDA_DEVICE_FUNCTION float4 dPosition_dRadius(float u) const {
            return 2 * m_p[2] * u + m_p[1];
        }
        CUDA_DEVICE_FUNCTION float3 dPosition(float u) const {
            return 2 * make_float3(m_p[2]) * u + make_float3(m_p[1]);
        }
        CUDA_DEVICE_FUNCTION float dRadius(float u) const {
            return 2 * m_p[2].w * u + m_p[1].w;
        }

        CUDA_DEVICE_FUNCTION float4 ddPosition_ddRadius(float u) const {
            return 2 * m_p[2];
        }
        CUDA_DEVICE_FUNCTION float3 ddPosition(float u) const {
            return 2 * make_float3(m_p[2]);
        }
    };

    /*
    p0, p1, p2, p3: control points
    u: curve parameter
    P(u) =   (-p0 + 3 * p1 - 3 * p2 + p3) / 6 * u^3
           + (0.5 * p0 + p1 + 0.5 * p2) * u^2
           + (-p0 + p2) / 2 * u
           + (p0 + 4 * p1 + p2) / 6
    */
    template <>
    struct Interpolator<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE> {
        float4 m_p[4];

        CUDA_DEVICE_FUNCTION Interpolator() {}
        CUDA_DEVICE_FUNCTION Interpolator(const float4 cps[4]) {
            m_p[3] = (-cps[0] + 3 * cps[1] - 3 * cps[2] + cps[3]) / 6;
            m_p[2] = (0.5f * cps[0] - cps[1] + 0.5f * cps[2]);
            m_p[1] = (-cps[0] + cps[2]) * 0.5f;
            m_p[0] = (cps[0] + 4 * cps[1] + cps[2]) / 6;
        }

        CUDA_DEVICE_FUNCTION float4 position_radius(float u) const {
            return m_p[3] * u * u * u + m_p[2] * u * u + m_p[1] * u + m_p[0];
        }
        CUDA_DEVICE_FUNCTION float3 position(float u) const {
            return
                make_float3(m_p[3]) * u * u * u
                + make_float3(m_p[2]) * u * u
                + make_float3(m_p[1]) * u
                + make_float3(m_p[0]);
        }
        CUDA_DEVICE_FUNCTION float radius(float u) const {
            return m_p[3].w * u * u * u + m_p[2].w * u * u + m_p[1].w * u + m_p[0].w;
        }

        CUDA_DEVICE_FUNCTION float minRadius(float uA, float uB) const {
            float a = 3 * m_p[3].w;
            float b = 2 * m_p[2].w;
            float c = m_p[1].w;
            float rmin = fminf(radius(uA), radius(uB));
            if (fabsf(a) < 1e-5f) {
                float root1 = stc::clamp(-c / b, uA, uB);
                return fminf(rmin, radius(root1));
            }
            else {
                float det = b * b - 4 * a * c;
                det = det <= 0.0f ? 0.0f : sqrtf(det);
                float root1 = stc::clamp((-b + det) / (2 * a), uA, uB);
                float root2 = stc::clamp((-b - det) / (2 * a), uA, uB);
                return fminf(rmin, fminf(radius(root1), radius(root2)));
            }
        }
        CUDA_DEVICE_FUNCTION float maxRadius(float uA, float uB) const {
            if (m_p[1].w == 0 && m_p[2].w == 0 && m_p[3].w == 0)
                return m_p[0].w; // constant width
            float a = 3 * m_p[3].w;
            float b = 2 * m_p[2].w;
            float c = m_p[1].w;
            float rmax = fmaxf(radius(uA), radius(uB));
            if (fabsf(a) < 1e-5f) {
                float root1 = stc::clamp(-c / b, uA, uB);
                return fmaxf(rmax, radius(root1));
            }
            else {
                float det = b * b - 4 * a * c;
                det = det <= 0.0f ? 0.0f : sqrtf(det);
                float root1 = stc::clamp((-b + det) / (2 * a), uA, uB);
                float root2 = stc::clamp((-b - det) / (2 * a), uA, uB);
                return fmaxf(rmax, fmaxf(radius(root1), radius(root2)));
            }
        }

        CUDA_DEVICE_FUNCTION float4 dPosition_dRadius(float u) const {
            // adjust u to avoid problems with tripple knots.
            if (u == 0)
                u = 0.000001f;
            if (u == 1)
                u = 0.999999f;
            return 3 * m_p[3] * u * u + 2 * m_p[2] * u + m_p[1];
        }
        CUDA_DEVICE_FUNCTION float3 dPosition(float u) const {
            // adjust u to avoid problems with tripple knots.
            if (u == 0)
                u = 0.000001f;
            if (u == 1)
                u = 0.999999f;
            return 3 * make_float3(m_p[3]) * u * u + 2 * make_float3(m_p[2]) * u + make_float3(m_p[1]);
        }
        CUDA_DEVICE_FUNCTION float dRadius(float u) const {
            // adjust u to avoid problems with tripple knots.
            if (u == 0)
                u = 0.000001f;
            if (u == 1)
                u = 0.999999f;
            return 3 * m_p[3].w * u * u + 2 * m_p[2].w * u + m_p[1].w;
        }

        CUDA_DEVICE_FUNCTION float4 ddPosition_ddRadius(float u) const {
            return 6 * m_p[3] * u + 2 * m_p[2];
        }
        CUDA_DEVICE_FUNCTION float3 ddPosition(float u) const {
            return 6 * make_float3(m_p[3]) * u + 2 * make_float3(m_p[2]);
        }
    };

    /*
    p0, p1, p2, p3: control points
    u: curve parameter
    P(u) =   0.5 * (-p0 + 3 * p1 - 3 * p2 + p3) * u^3
           + 0.5 * (2 * p0 - 5 * p1 + 4 * p2 - p3) * u^2
           + 0.5 * (-p0 + p2) * u
           + p1
    */
    template <>
    struct Interpolator<OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM> :
        public Interpolator<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE> {
        CUDA_DEVICE_FUNCTION Interpolator(const float4 cps[4]) {
            m_p[3] = 0.5f * (-cps[0] + 3 * (cps[1] - cps[2]) + cps[3]);
            m_p[2] = 0.5f * (2 * cps[0] - 5 * cps[1] + 4 * cps[2] - cps[3]);
            m_p[1] = 0.5f * (-cps[0] + cps[2]);
            m_p[0] = cps[1];
        }
    };

    /*
    p0, p1, p2, p3: control points
    u: curve parameter
    P(u) =   (-p0 + 3 * p1 - 3 * p2 + p3) * u^3
           + (3 * p0 - 6 * p1 + 3 * p2) * u^2
           + (-3 * p0 + 3 * p1) * u
           + p0
    */
    template <>
    struct Interpolator<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER> :
        public Interpolator<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE> {
        CUDA_DEVICE_FUNCTION Interpolator(const float4 cps[4]) {
            m_p[3] = (-cps[0] + 3 * (cps[1] - cps[2]) + cps[3]);
            m_p[2] = (3 * cps[0] - 6 * cps[1] + 3 * cps[2]);
            m_p[1] = (-3 * cps[0] + 3 * cps[1]);
            m_p[0] = cps[0];
        }
    };



    // Based on OptiX SDK/cuda/curve.h
    template <OptixPrimitiveType curveType>
    class Evaluator {
        Interpolator<curveType> m_interpolator;

    public:
        CUDA_DEVICE_FUNCTION Evaluator(const float4 controlPoints[getNumControlPoints<curveType>()]) :
            m_interpolator(controlPoints) {}

        // type - 0     ~ cylindrical approximation (correct if radius' == 0)
        //        1     ~ conic       approximation (correct if curve'' == 0)
        //        other ~ the bona fide surface normal
        template <uint32_t type = 2>
        CUDA_DEVICE_FUNCTION float3 calcNormal(float u, const float3 &hitPointInObject) const {
            float3 hp = hitPointInObject;

            float3 normal;
            if (u == 0.0f) {
                if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR)
                    normal = hp - make_float3(m_interpolator.m_p[0]); // special handling for round endcaps
                else
                    normal = -m_interpolator.dPosition(0); // special handling for flat endcaps
            }
            else if (u >= 1.0f) {
                if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR) {
                    // reconstruct second control point (Note: the interpolator pre-transforms
                    // the control-points to speed up repeated evaluation.
                    float3 p1 = make_float3(m_interpolator.m_p[1]) + make_float3(m_interpolator.m_p[0]);
                    normal = hp - p1; // special handling for round endcaps
                }
                else {
                    normal = m_interpolator.dPosition(1); // special handling for flat endcaps
                }
            }
            else {
                // hp is a point that is near the curve's offset surface,
                // usually ray.origin + ray.direction * rayt.
                // We will push it exactly to the surface by projecting it to the plane(p,d).
                // The function derivation:
                // we (implicitly) transform the curve into coordinate system
                // {p, o1 = normalize(hp - p), o2 = normalize(curve'(t)), o3 = o1 x o2} in which
                // curve'(t) = (0, length(d), 0); hp = (r, 0, 0);
                float4 p4 = m_interpolator.position_radius(u);
                float3 p = make_float3(p4);
                float  r = p4.w; // == length(hp - p) if hp is already on the surface
                float4 d4 = m_interpolator.dPosition_dRadius(u);
                float3 d = make_float3(d4);
                float  dr = d4.w;
                float  dd = dot(d, d);

                float3 o1 = hp - p;          // dot(modified_o1, d) == 0 by design:
                o1 -= (dot(o1, d) / dd) * d; // first, project hp to the plane(p,d)
                o1 *= r / length(o1);        // and then drop it to the surface
                hp = p + o1;                 // fine-tuning the hit point
                if constexpr (type == 0) {
                    normal = o1; // cylindrical approximation
                }
                else {
                    if constexpr (curveType != OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR && type != 1)
                        dd -= dot(m_interpolator.ddPosition(u), o1);
                    normal = dd * o1 - (dr * r) * d;
                }
            }

            return normal; // non-normalized
        }

        CUDA_DEVICE_FUNCTION float3 calcTangent(float u) const {
            float3 tangent = m_interpolator.dPosition(u);
            return tangent; // non-normalized;
        }
    };
}
