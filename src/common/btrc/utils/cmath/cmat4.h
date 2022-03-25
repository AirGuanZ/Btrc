#pragma once

#include <btrc/utils/cmath/cscalar.h>
#include <btrc/utils/cmath/cvec4.h>
#include <btrc/utils/math/mat4.h>

BTRC_BEGIN

CUJ_PROXY_CLASS_EX(CMat4, Mat4, data)
{
    CUJ_BASE_CONSTRUCTORS

    CMat4(const Mat4 &m)
    {
        data[0] = CVec4f(m.data[0]);
        data[1] = CVec4f(m.data[1]);
        data[2] = CVec4f(m.data[2]);
        data[3] = CVec4f(m.data[3]);
    }

    ref<f32> at(i32 r, i32 c) const
    {
        return data[c][r];
    }
};

BTRC_END
