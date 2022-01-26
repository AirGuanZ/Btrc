#pragma once

#include <btrc/core/spectrum/spectrum.h>

BTRC_CORE_BEGIN

class AreaLight;
class EnvirLight;

class Light
{
public:

    virtual ~Light() = default;

    virtual bool is_area() const noexcept = 0;

    virtual const AreaLight *as_area() const { return nullptr; }

    virtual const EnvirLight *as_envir() const { return nullptr; }
};

class AreaLight : public Light
{
public:

    struct SampleLiResult
    {
        CVec3f    position;
        CSpectrum radiance;
        f32       pdf;
    };

    bool is_area() const noexcept final { return true; }

    const AreaLight *as_area() const final { return this; }

    virtual CSpectrum eval_le(
        const CVec3f &pos,
        const CVec3f &nor,
        const CVec2f &uv,
        const CVec2f &tex_coord,
        const CVec3f &wr) const = 0;

    virtual SampleLiResult sample_li(
        const CVec3f &ref,
        const CVec3f &sam) const = 0;

    virtual f32 pdf_li(
        const CVec3f &ref,
        const CVec3f &pos,
        const CVec3f &nor) const = 0;
};

class EnvirLight : public Light
{
public:

    struct SampleLiResult
    {
        CVec3f    direction_to_light;
        CSpectrum radiance;
        f32       pdf;
    };

    bool is_area() const noexcept final { return false; }

    const EnvirLight *as_envir() const final { return this; }

    virtual CSpectrum eval_le(const CVec3f &to_light) const = 0;

    virtual SampleLiResult sample_li(const CVec3f &sam) const = 0;

    virtual f32 pdf_li(const CVec3f &to_light) const = 0;
};

BTRC_CORE_END
