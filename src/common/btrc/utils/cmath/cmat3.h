#pragma once

#include <btrc/utils/cmath/cscalar.h>
#include <btrc/utils/cmath/cvec3.h>
#include <btrc/utils/math/mat3.h>

BTRC_BEGIN

CUJ_PROXY_CLASS_EX(CMat3, Mat3, data)
{
    CUJ_BASE_CONSTRUCTORS

    CMat3(const Mat3 &m)
    {
        data[0] = CVec3f(m.data[0]);
        data[1] = CVec3f(m.data[1]);
        data[2] = CVec3f(m.data[2]);
    }

    ref<f32> at(i32 r, i32 c) const
    {
        return data[c][r];
    }
};

BTRC_END
