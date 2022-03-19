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

    CSpectrum sample_spectrum(CompileContext &cc, ref<CVec3f> uvw) const
    {
        using T = CSpectrum(Texture3D:: *)(CompileContext &, ref<CVec3f>)const;
        return record(cc, T(&Texture3D::sample_spectrum_inline), "sample_spectrum_uvw", uvw);
    }

    f32 sample_float(CompileContext &cc, ref<CVec3f> uvw) const
    {
        using T = f32(Texture3D:: *)(CompileContext &, ref<CVec3f>)const;
        return record(cc, T(&Texture3D::sample_float_inline), "sample_float_uvw", uvw);
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

    virtual Spectrum get_max_spectrum() const = 0;

    virtual float get_max_float() const
    {
        return get_max_spectrum().r;
    }

    virtual Spectrum get_min_spectrum() const = 0;

    virtual float get_min_float() const
    {
        return get_min_spectrum().r;
    }
};

class Constant3D : public Texture3D
{
    Spectrum value_;

public:

    Constant3D();

    void set_value(float value);

    void set_value(const Spectrum &value);

    f32 sample_float_inline(CompileContext &cc, ref<CVec3f> uvw) const override;

    CSpectrum sample_spectrum_inline(CompileContext &cc, ref<CVec3f> uvw) const override;

    Spectrum get_max_spectrum() const override;

    Spectrum get_min_spectrum() const override;
};

BTRC_END
