#pragma once

#include <btrc/core/context.h>
#include <btrc/core/geometry.h>
#include <btrc/core/spectrum.h>
#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

class AreaLight;
class EnvirLight;

class Light : public Object
{
public:

    virtual bool is_area() const noexcept = 0;

    virtual const AreaLight *as_area() const { return nullptr; }

    virtual const EnvirLight *as_envir() const { return nullptr; }
};

class AreaLight : public Light
{
public:

    CUJ_CLASS_BEGIN(SampleLiResult)
        CUJ_MEMBER_VARIABLE(CVec3f,    position)
        CUJ_MEMBER_VARIABLE(CVec3f,    normal)
        CUJ_MEMBER_VARIABLE(CSpectrum, radiance)
        CUJ_MEMBER_VARIABLE(f32,       pdf)
    CUJ_CLASS_END

    bool is_area() const noexcept final { return true; }

    const AreaLight *as_area() const final { return this; }

    virtual void set_geometry(
        RC<Geometry> geometry, const Transform3D &local_to_world) = 0;

    virtual CSpectrum eval_le_inline(
        CompileContext &cc,
        ref<CVec3f>     pos,
        ref<CVec3f>     nor,
        ref<CVec2f>     uv,
        ref<CVec2f>     tex_coord,
        ref<CVec3f>     wr) const = 0;

    virtual SampleLiResult sample_li_inline(
        CompileContext &cc,
        ref<CVec3f>     ref_pos,
        ref<CVec3f>     sam) const = 0;

    virtual f32 pdf_li_inline(
        CompileContext &cc,
        ref<CVec3f>     ref_pos,
        ref<CVec3f>     pos,
        ref<CVec3f>     nor) const = 0;

    CSpectrum eval_le(
        CompileContext &cc,
        ref<CVec3f> pos,
        ref<CVec3f> nor,
        ref<CVec2f> uv,
        ref<CVec2f> tex_coord,
        ref<CVec3f> wr) const
    {
        return record(
            cc, &AreaLight::eval_le_inline, "eval_le",
            pos, nor, uv, tex_coord, wr);
    }

    SampleLiResult sample_li(
        CompileContext &cc,
        ref<CVec3f> ref_pos,
        ref<CVec3f> sam) const
    {
        return record(
            cc, &AreaLight::sample_li_inline, "sample_li",
            ref_pos, sam);
    }

    f32 pdf_li(
        CompileContext &cc,
        ref<CVec3f> ref_pos,
        ref<CVec3f> pos,
        ref<CVec3f> nor) const
    {
        return record(
            cc, &AreaLight::pdf_li_inline, "pdf_li",
            ref_pos, pos, nor);
    }
};

class EnvirLight : public Light
{
public:

    CUJ_CLASS_BEGIN(SampleLiResult)
        CUJ_MEMBER_VARIABLE(CVec3f,    direction_to_light)
        CUJ_MEMBER_VARIABLE(CSpectrum, radiance)
        CUJ_MEMBER_VARIABLE(f32,       pdf)
    CUJ_CLASS_END

    bool is_area() const noexcept final { return false; }

    const EnvirLight *as_envir() const final { return this; }

    virtual CSpectrum eval_le_inline(CompileContext &cc, ref<CVec3f> to_light) const = 0;

    virtual SampleLiResult sample_li_inline(CompileContext &cc, ref<CVec3f> sam) const = 0;

    virtual f32 pdf_li_inline(CompileContext &cc, ref<CVec3f> to_light) const = 0;

    CSpectrum eval_le(CompileContext &cc, const CVec3f &to_light) const
    {
        return record(cc, &EnvirLight::eval_le_inline, "eval_le", to_light);
    }

    SampleLiResult sample_li(CompileContext &cc, ref<CVec3f> sam) const
    {
        return record(cc, &EnvirLight::sample_li_inline, "sample_li", sam);
    }

    f32 pdf_li(CompileContext &cc, ref<CVec3f> to_light) const
    {
        return record(cc, &EnvirLight::pdf_li_inline, "pdf_li", to_light);
    }
};

BTRC_END
