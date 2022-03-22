#include <btrc/builtin/material/utils/fresnel.h>

BTRC_BUILTIN_BEGIN

f32 dielectric_fresnel(f32 eta_i, f32 eta_o, f32 cos_theta_i)
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

f32 schlick_approx(f32 R0, f32 cos_theta_i)
{
    var t = 1.0f - cos_theta_i;
    var t2 = t * t;
    var t5 = t2 * t2 * t;
    return R0 + (1.0f - R0) * t5;
}

CSpectrum schlick_approx(ref<CSpectrum> R0, f32 cos_theta_i)
{
    var t = 1.0f - cos_theta_i;
    var t2 = t * t;
    var t5 = t2 * t2 * t;
    return R0 + (CSpectrum::from_rgb(1, 1, 1) - R0) * t5;
}

BTRC_BUILTIN_END
