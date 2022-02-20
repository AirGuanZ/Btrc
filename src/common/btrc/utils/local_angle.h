#pragma once

#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

namespace local_angle
{

    // w must be normalized
    inline f32 cos_theta(ref<CVec3f> w)
    {
        return w.z;
    }

    inline f32 cos2sin(f32 cos)
    {
        return cstd::sqrt(cstd::max(f32(0), 1.0f - cos * cos));
    }

    // w must be normalized
    inline f32 tan_theta(ref<CVec3f> w)
    {
        var t = 1.0f - w.z * w.z;
        var result = 0.0f;
        $if(t > 0)
        {
            result = cstd::sqrt(t) / w.z;
        };
        return result;
    }

    inline f32 tan_theta_2(ref<CVec3f> w)
    {
        var z2 = w.z * w.z;
        var t = 1.0f - z2;
        var result = 0.0f;
        $if(t > 0)
        {
            result = t / z2;
        };
        return result;
    }

    // w must be normalized
    inline f32 theta(ref<CVec3f> w)
    {
        return cstd::acos(cstd::clamp(cos_theta(w), -1.0f, 1.0f));
    }

    inline f32 phi(ref<CVec3f> w)
    {
        var result = 0.0f;
        $if(w.x != 0 | w.y != 0)
        {
            result = cstd::atan2(w.y, w.x);
            result = cstd::select(result < 0, result + 2 * btrc_pi, f32(result));
        };
        return result;
    }

    // w and n must be normalized
    // w: from reflection point to 
    inline CVec3f reflect(ref<CVec3f> w, ref<CVec3f> n)
    {
        return 2.0f * dot(w, n) * n - w;
    }

} // namespace local_angle

BTRC_END
