#pragma once

#include <btrc/utils/cmath/cvec3.h>
#include <btrc/utils/math/ray.h>

BTRC_BEGIN

CUJ_PROXY_CLASS_EX(CRay, Ray, o, d, t)
{
    CUJ_BASE_CONSTRUCTORS

    CRay()
        : CRay(CVec3f(0), CVec3f(1, 0, 0))
    {
        
    }

    CRay(const CVec3f &o, const CVec3f &d, f32 t = btrc_max_float)
    {
        this->o = o;
        this->d = d;
        this->t = t;
    }
};

BTRC_END
