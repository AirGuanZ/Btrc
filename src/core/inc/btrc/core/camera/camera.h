#pragma once

#include <btrc/core/common/context.h>
#include <btrc/core/spectrum/spectrum.h>

BTRC_CORE_BEGIN

class Camera : public Object
{
public:

    CUJ_CLASS_BEGIN(SampleWeResult)
        CUJ_MEMBER_VARIABLE(CVec3f,    pos)
        CUJ_MEMBER_VARIABLE(CVec3f,    dir)
        CUJ_MEMBER_VARIABLE(CSpectrum, throughput)
        CUJ_MEMBER_VARIABLE(f32,       time)
    CUJ_CLASS_END

    virtual ~Camera() = default;

    virtual SampleWeResult generate_ray_inline(
        ref<CVec2f> film_coord, f32 time_sample) const = 0;

    SampleWeResult generate_ray(
        ref<CVec2f> film_record, f32 time_sample) const
    {
        return record(
            &Camera::generate_ray_inline, "generate_ray",
            film_record, time_sample);
    }
};

BTRC_CORE_END
