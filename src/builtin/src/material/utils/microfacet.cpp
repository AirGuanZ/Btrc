#include <btrc/builtin/material/utils/microfacet.h>
#include <btrc/utils/local_angle.h>

BTRC_BUILTIN_BEGIN

namespace microfacet
{
    
    f32 sqr(f32 x)
    {
        return x * x;
    }

    f32 compute_a(f32 sin_phi_h, f32 cos_phi_h, f32 ax, f32 ay)
    {
        return sqr(cos_phi_h / ax) + sqr(sin_phi_h / ay);
    }

    f32 gtr2(f32 cos_theta_h, f32 alpha)
    {
        return sqr(alpha) / (btrc_pi * sqr(1.0f + (sqr(alpha) - 1.0f) * sqr(cos_theta_h)));
    }

    f32 smith_gtr2(f32 tan_theta, f32 alpha)
    {
        var result = 1.0f;
        $if(tan_theta != 0)
        {
            var root = alpha * tan_theta;
            result = 2.0f / (1.0f + cstd::sqrt(1.0f + root * root));
        };
        return result;
    }

    CVec3f sample_gtr2(f32 alpha, ref<CVec2f> sample)
    {
        var phi = 2.0f * btrc_pi * sample.x;
        var cos_theta = cstd::sqrt((1.0f - sample.y) / (1.0f + (sqr(alpha) - 1.0f) * sample.y));
        var sin_theta = local_angle::cos2sin(cos_theta);
        return normalize(CVec3f(sin_theta * cstd::cos(phi), sin_theta * cstd::sin(phi), cos_theta));
    }

    f32 anisotropic_gtr2(f32 sin_phi_h, f32 cos_phi_h, f32 sin_theta_h, f32 cos_theta_h, f32 ax, f32 ay)
    {
        var A = compute_a(sin_phi_h, cos_phi_h, ax, ay);
        var RD = sqr(sin_theta_h) * A + sqr(cos_theta_h);
        return 1.0f / (btrc_pi * ax * ay * sqr(RD));
    }

    f32 smith_anisotropic_gtr2(f32 cos_phi, f32 sin_phi, f32 ax, f32 ay, f32 tan_theta)
    {
        var t = sqr(ax * cos_phi) + sqr(ay * sin_phi);
        var sqr_val = 1.0f + t * sqr(tan_theta);
        var lambda = -0.5f + 0.5f * cstd::sqrt(sqr_val);
        return 1.0f / (1.0f + lambda);
    }

    CVec3f sample_anisotropic_gtr2(f32 ax, f32 ay, ref<CVec2f> sample)
    {
        var sin_phi_h = ay * cstd::sin(2 * btrc_pi * sample.x);
        var cos_phi_h = ax * cstd::cos(2 * btrc_pi * sample.x);
        var nor = 1.0f / cstd::sqrt(sqr(sin_phi_h) + sqr(cos_phi_h));
        sin_phi_h = sin_phi_h * nor;
        cos_phi_h = cos_phi_h * nor;
        var A = compute_a(sin_phi_h, cos_phi_h, ax, ay);
        var cos_theta_h = cstd::sqrt(A * (1.0f - sample.y) / ((1.0f - A) * sample.y + A));
        var sin_theta_h = local_angle::cos2sin(cos_theta_h);
        return normalize(CVec3f(sin_theta_h * cos_phi_h, sin_theta_h * sin_phi_h, cos_theta_h));
    }

    f32 gtr1(f32 sin_theta_h, f32 cos_theta_h, f32 alpha)
    {
        var U = sqr(alpha) - 1;
        var LD = 2.0f * btrc_pi * cstd::log(alpha);
        var RD = sqr(alpha * cos_theta_h) + sqr(sin_theta_h);
        return U / (LD * RD);
    }

    CVec3f sample_gtr1(f32 alpha, ref<CVec2f> sample)
    {
        var phi = 2 * btrc_pi * sample.x;
        var cos_theta = cstd::sqrt((cstd::pow(alpha, 2.0f - 2.0f * sample.y) - 1.0f) / (sqr(alpha) - 1.0f));
        var sin_theta = local_angle::cos2sin(cos_theta);
        return normalize(CVec3f(sin_theta * cstd::cos(phi), sin_theta * cstd::sin(phi), cos_theta));
    }

    CVec3f sample_anisotropic_gtr2_vnor(ref<CVec3f> ve, f32 ax, f32 ay, ref<CVec2f> sam)
    {
        var vh = normalize(CVec3f(ax * ve.x, ay * ve.y, ve.z));
        var lensq = vh.x * vh.x + vh.y * vh.y;

        var t1 = CVec3f(1, 0, 0);
        $if(lensq > 3e-4f)
        {
            t1 = CVec3f(-vh.y, vh.x, 0) / cstd::sqrt(lensq);
        };
        var t2 = cross(vh, t1);

        var r = cstd::sqrt(sam.x);
        var phi = 2 * btrc_pi * sam.y;
        var t_1 = r * cstd::cos(phi);
        var _t_2 = r * cstd::sin(phi);
        var s = 0.5f * (1.0f + vh.z);
        var t_2 = (1.0f - s) * cstd::sqrt(1.0f - t_1 * t_1) + s * _t_2;

        var nh = t_1 * t1 + t_2 * t2 + cstd::sqrt(
            (cstd::max)(f32(0), 1.0f - t_1 * t_1 - t_2 * t_2)) * vh;
        var ne = normalize(CVec3f(ax * nh.x, ay * nh.y, (cstd::max)(f32(0), nh.z)));
        return ne;
    }

} // namespace microfacet

BTRC_BUILTIN_END
