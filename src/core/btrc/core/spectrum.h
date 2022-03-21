#pragma once

#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

class Spectrum
{
public:

    float r, g, b;
    float additional_data;

    static Spectrum from_rgb(float r, float g, float b);

    static Spectrum one();

    static Spectrum zero();

    Spectrum();

    bool is_zero() const;

    float get_lum() const;

    Vec3f to_rgb() const;
};

CUJ_PROXY_CLASS_EX(CSpectrum, Spectrum, r, g, b, additional_data)
{
    CUJ_BASE_CONSTRUCTORS

    static CSpectrum from_rgb(f32 _r, f32 _g, f32 _b);

    static CSpectrum one();

    static CSpectrum zero();
        
    CSpectrum(const Spectrum &s = {});

    CSpectrum(f32 r, f32 g, f32 b, f32 w);

    boolean is_zero() const;

    f32 get_lum() const;

    CVec3f to_rgb() const;
};

CSpectrum load_aligned(ptr<CSpectrum> addr);

void save_aligned(ref<CSpectrum> spec, ptr<CSpectrum> addr);

Spectrum operator+(const Spectrum &a, const Spectrum &b);
Spectrum operator-(const Spectrum &a, const Spectrum &b);
Spectrum operator*(const Spectrum &a, const Spectrum &b);
Spectrum operator/(const Spectrum &a, const Spectrum &b);

Spectrum operator+(const Spectrum &a, float b);
Spectrum operator*(const Spectrum &a, float b);
Spectrum operator/(const Spectrum &a, float b);

Spectrum operator+(float a, const Spectrum &b);
Spectrum operator*(float a, const Spectrum &b);

Spectrum (max)(const Spectrum &a, const Spectrum &b);
Spectrum (min)(const Spectrum &a, const Spectrum &b);

CSpectrum operator+(const CSpectrum &a, const CSpectrum &b);
CSpectrum operator-(const CSpectrum &a, const CSpectrum &b);
CSpectrum operator*(const CSpectrum &a, const CSpectrum &b);
CSpectrum operator/(const CSpectrum &a, const CSpectrum &b);

CSpectrum operator+(const CSpectrum &a, f32 b);
CSpectrum operator*(const CSpectrum &a, f32 b);
CSpectrum operator/(const CSpectrum &a, f32 b);

CSpectrum operator+(f32 a, const CSpectrum &b);
CSpectrum operator*(f32 a, const CSpectrum &b);

CSpectrum (max)(const CSpectrum &a, const CSpectrum &b);
CSpectrum (min)(const CSpectrum &a, const CSpectrum &b);

BTRC_END
