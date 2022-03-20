#pragma once

#include <btrc/core/context.h>
#include <btrc/core/spectrum.h>

BTRC_BEGIN

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

    virtual void set_w_over_h(float ratio) = 0;

    virtual AABB3f get_bounding_box() const = 0;

    virtual SampleWeResult generate_ray_inline(
        CompileContext &cc, ref<CVec2f> film_coord, f32 time_sample) const = 0;

    SampleWeResult generate_ray(CompileContext &cc, ref<CVec2f> film_record, f32 time_sample) const
    {
        return record(cc, &Camera::generate_ray_inline, "generate_ray", film_record, time_sample);
    }
};

BTRC_END
