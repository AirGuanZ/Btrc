#pragma once

#include <btrc/core/context.h>
#include <btrc/core/surface_point.h>
#include <btrc/core/spectrum.h>
#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

class Texture3D : public Object
{
public:

    virtual CSpectrum sample_spectrum_inline(CompileContext &cc, ref<CVec3f> uvw) const = 0;

    virtual f32 sample_float_inline(CompileContext &cc, ref<CVec3f> uvw) const = 0;

    virtual CSpectrum sample_spectrum_inline(CompileContext &cc, ref<SurfacePoint> spt) const
    {
        return sample_spectrum_inline(cc, spt.position);
    }

    virtual f32 sample_float_inline(CompileContext &cc, ref<SurfacePoint> spt) const
    {
        return sample_float_inline(cc, spt.position);
    }

    CSpectrum sample_spectrum(CompileContext &cc, ref<CVec3f> uv) const
    {
        using T = CSpectrum(Texture3D:: *)(CompileContext &, ref<CVec3f>)const;
        return record(cc, T(&Texture3D::sample_spectrum_inline), "sample_spectrum_uv", uv);
    }

    f32 sample_float(CompileContext &cc, ref<CVec3f> uv) const
    {
        using T = f32(Texture3D:: *)(CompileContext &, ref<CVec3f>)const;
        return record(cc, T(&Texture3D::sample_float_inline), "sample_float_uv", uv);
    }

    CSpectrum sample_spectrum(CompileContext &cc, ref<SurfacePoint> spt) const
    {
        using T = CSpectrum(Texture3D:: *)(CompileContext &, ref<SurfacePoint>)const;
        return record(cc, T(&Texture3D::sample_spectrum_inline), "sample_spectrum_spt", spt);
    }

    f32 sample_float(CompileContext &cc, ref<SurfacePoint> spt) const
    {
        using T = f32(Texture3D:: *)(CompileContext &, ref<SurfacePoint>)const;
        return record(cc, T(&Texture3D::sample_float_inline), "sample_float_spt", spt);
    }

    virtual CSpectrum get_max_spectrum(CompileContext &cc) const = 0;

    virtual f32 get_max_float(CompileContext &cc) const
    {
        return get_max_spectrum(cc).r;
    }
};

class Constant3D : public Texture3D
{
    BTRC_PROPERTY(Spectrum, value_);

public:

    Constant3D();

    void set_value(float value);

    void set_value(const Spectrum &value);

    f32 sample_float_inline(CompileContext &cc, ref<CVec3f> uvw) const override;

    CSpectrum sample_spectrum_inline(CompileContext &cc, ref<CVec3f> uvw) const override;

    CSpectrum get_max_spectrum(CompileContext &cc) const override;
};

BTRC_END
