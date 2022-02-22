#pragma once

#include <btrc/core/context.h>
#include <btrc/core/surface_point.h>
#include <btrc/core/spectrum.h>
#include <btrc/utils/cmath/cmath.h>
#include <btrc/utils/variant.h>

BTRC_BEGIN

class Texture2D : public Object
{
public:

    virtual CSpectrum sample_spectrum_inline(CompileContext &cc, ref<CVec2f> uv) const = 0;

    virtual f32 sample_float_inline(CompileContext &cc, ref<CVec2f> uv) const = 0;

    virtual CSpectrum sample_spectrum_inline(CompileContext &cc, ref<SurfacePoint> spt) const
    {
        return sample_spectrum_inline(cc, spt.tex_coord);
    }

    virtual f32 sample_float_inline(CompileContext &cc, ref<SurfacePoint> spt) const
    {
        return sample_float_inline(cc, spt.tex_coord);
    }

    CSpectrum sample_spectrum(CompileContext &cc, ref<CVec2f> uv) const
    {
        using T = CSpectrum(Texture2D::*)(CompileContext &, ref<CVec2f>)const;
        return record(cc, T(&Texture2D::sample_spectrum_inline), "sample_spectrum_uv", uv);
    }

    f32 sample_float(CompileContext &cc, ref<CVec2f> uv) const
    {
        using T = f32(Texture2D::*)(CompileContext &, ref<CVec2f>)const;
        return record(cc, T(&Texture2D::sample_float_inline), "sample_float_uv", uv);
    }

    CSpectrum sample_spectrum(CompileContext &cc, ref<SurfacePoint> spt) const
    {
        using T = CSpectrum(Texture2D::*)(CompileContext &, ref<SurfacePoint>)const;
        return record(cc, T(&Texture2D::sample_spectrum_inline), "sample_spectrum_spt", spt);
    }
    
    f32 sample_float(CompileContext &cc, ref<SurfacePoint> spt) const
    {
        using T = f32(Texture2D::*)(CompileContext&, ref<SurfacePoint>)const;
        return record(cc, T(&Texture2D::sample_float_inline), "sample_float_spt", spt);
    }
};

class Constant2D : public Texture2D
{
    BTRC_PROPERTY(Spectrum, value_);

public:

    Constant2D();

    void set_value(float value);

    void set_value(const Spectrum &value);

    f32 sample_float_inline(CompileContext &cc, ref<CVec2f> uv) const override;

    CSpectrum sample_spectrum_inline(CompileContext &cc, ref<CVec2f> uv) const override;
};

BTRC_END
