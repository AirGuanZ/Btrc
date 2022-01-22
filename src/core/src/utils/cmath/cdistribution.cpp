#include <btrc/core/utils/cmath/cdistribution.h>

BTRC_CORE_BEGIN

CVec3f sample_sphere_uniform(f32 u1, f32 u2)
{
    var z = 1.0f - (u1 + u1);
    var phi = 2.0f * btrc_pi * u2;
    var r = cstd::sqrt((cstd::max)(f32(0), 1.0f - z * z));
    var x = r * cstd::cos(phi);
    var y = r * cstd::sin(phi);
    return CVec3f(x, y, z);
}

CVec3f sample_sphere_uniform(ref<CVec2f> sam)
{
    return sample_sphere_uniform(sam.x, sam.y);
}

CVec3f sample_sphere_uniform(ref<cstd::LCG> rng)
{
    return sample_sphere_uniform(CVec2f(rng));
}

f32 pdf_sample_sphere_uniform()
{
    return 1 / (4 * btrc_pi);
}

CVec3f sample_hemisphere_zweighted(f32 u1, f32 u2)
{
    CVec2f sam;
    u1 = 2.0f * u1 - 1.0f;
    u2 = 2.0f * u2 - 1.0f;
    $if(u1 != 0.0f | u2 != 0.0f)
    {
        f32 theta, r;
        $if(cstd::abs(u1) > cstd::abs(u2))
        {
            r = u1;
            theta = 0.25f * btrc_pi * (u2 / u1);
        }
        $else
        {
            r = u2;
            theta = 0.5f * btrc_pi - 0.5f * btrc_pi * (u1 / u2);
        };
        sam = r * CVec2f(cstd::cos(theta), cstd::sin(theta));
    };
    f32 z = cstd::sqrt((cstd::max)(f32(0), 1.0f - length_square(sam)));
    return CVec3f(sam.x, sam.y, z);
}

CVec3f sample_hemisphere_zweighted(ref<CVec2f> sam)
{
    return sample_hemisphere_zweighted(sam.x, sam.y);
}

CVec3f sample_hemisphere_zweighted(ref<cstd::LCG> rng)
{
    return sample_hemisphere_zweighted(CVec2f(rng));
}

f32 pdf_sample_hemisphere_zweighted(ref<CVec3f> v)
{
    return cstd::select(v.z >= 0, v.z / btrc_pi, f32(0));
}

BTRC_CORE_END
