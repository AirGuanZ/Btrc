#pragma once

#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

// w and n must be normalized
// w: from reflection point to 
inline CVec3f reflect(ref<CVec3f> w, ref<CVec3f> n);

/**
 * @brief compute fresnel value of dielectric surface
 *
 * @param eta_i inner IOR
 * @param eta_o outer IOR
 */
inline f32 dielectric_fresnel(f32 eta_i, f32 eta_o, f32 cos_theta_i);

// ========================== impl ==========================

inline CVec3f reflect(ref<CVec3f> w, ref<CVec3f> n)
{
    return 2.0f * dot(w, n) * n - w;
}

inline f32 dielectric_fresnel(f32 eta_i, f32 eta_o, f32 cos_theta_i)
{
    $if(cos_theta_i < 0)
    {
        f32 teta = eta_i;
        eta_i = eta_o;
        eta_o = teta;
        cos_theta_i = -cos_theta_i;
    };
    f32 sin_theta_i = cstd::sqrt((cstd::max)(f32(0), 1.0f - cos_theta_i * cos_theta_i));
    f32 sin_theta_t = eta_o / eta_i * sin_theta_i;
    f32 result;
    $if(sin_theta_t >= 1)
    {
        result = 1;
    }
    $else
    {
        f32 cos_theta_t = cstd::sqrt((cstd::max)(
            f32(0), 1.0f - sin_theta_t * sin_theta_t));
        f32 para = (eta_i * cos_theta_i - eta_o * cos_theta_t)
                 / (eta_i * cos_theta_i + eta_o * cos_theta_t);
        f32 perp = (eta_o * cos_theta_i - eta_i * cos_theta_t)
                 / (eta_o * cos_theta_i + eta_i * cos_theta_t);
        result = 0.5f * (para * para + perp * perp);
    };

    return result;
}

BTRC_END
