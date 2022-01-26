#pragma once

#include <btrc/core/spectrum/spectrum.h>
#include <btrc/core/utils/uncopyable.h>

BTRC_CORE_BEGIN

class RGBSpectrumImpl : public SpectrumImpl
{
public:

    float r, g, b;

    static Spectrum from_rgb(float r, float g, float b);

    Spectrum add(const SpectrumImpl &other) const override;

    Spectrum mul(const SpectrumImpl &other) const override;

    Spectrum add(float v) const override;

    Spectrum mul(float v) const override;

    void assign(const SpectrumImpl &other) override;

    Vec3f to_rgb() const override;

    bool is_zero() const override;

    float get_lum() const override;

    Box<SpectrumImpl> clone() const override;

    Box<CSpectrumImpl> to_cspectrum() const override;

    const SpectrumType *get_type() const override;
};

class CRGBSpectrumImpl : public CSpectrumImpl
{
public:

    f32 r, g, b;

    static CSpectrum from_rgb(f32 r, f32 g, f32 b);

    CSpectrum add(const CSpectrumImpl &other) const override;

    CSpectrum mul(const CSpectrumImpl &other) const override;

    CSpectrum add(f32 v) const override;

    CSpectrum mul(f32 v) const override;

    void assign(const CSpectrumImpl &other) override;

    CVec3f to_rgb() const override;

    boolean is_zero() const override;

    f32 get_lum() const override;

    Box<CSpectrumImpl> clone() const override;

    const SpectrumType *get_type() const override;
};

class RGBSpectrumType : public Uncopyable, public SpectrumType
{
    RGBSpectrumType() = default;

public:

    static const RGBSpectrumType *get_instance();

    int get_word_count() const override;

    Spectrum create_one() const override;

    Spectrum create_zero() const override;

    Spectrum create_from_rgb(float r, float g, float b) const override;

    CSpectrum create_c_from_rgb(f32 r, f32 g, f32 b) const override;

    CSpectrum load(ptr<f32> beta) const override;

    void save(ptr<f32> beta, const CSpectrum &spec) const override;
};

BTRC_CORE_END
