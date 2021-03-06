#include <btrc/core/spectrum.h>

BTRC_BEGIN

Spectrum Spectrum::from_rgb(float r, float g, float b)
{
    Spectrum ret;
    ret.r = r;
    ret.g = g;
    ret.b = b;
    return ret;
}

Spectrum Spectrum::one()
{
    return from_rgb(1, 1, 1);
}

Spectrum Spectrum::zero()
{
    return from_rgb(0, 0, 0);
}

Spectrum::Spectrum()
    : r(0), g(0), b(0)
{
    
}

bool Spectrum::is_zero() const
{
    return r == 0 && g == 0 && b == 0;
}

float Spectrum::get_lum() const
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

Vec3f Spectrum::to_rgb() const
{
    return Vec3f(r, g, b);
}

CSpectrum CSpectrum::from_rgb(f32 _r, f32 _g, f32 _b)
{
    CSpectrum ret;
    ret.r = _r;
    ret.g = _g;
    ret.b = _b;
    return ret;
}

CSpectrum CSpectrum::one()
{
    return from_rgb(1, 1, 1);
}

CSpectrum CSpectrum::zero()
{
    return from_rgb(0, 0, 0);
}

CSpectrum::CSpectrum(const Spectrum &s)
{
    r = s.r;
    g = s.g;
    b = s.b;
}

CSpectrum::CSpectrum(f32 _r, f32 _g, f32 _b, f32 w)
{
    r = _r;
    g = _g;
    b = _b;
    additional_data = w;
}

boolean CSpectrum::is_zero() const
{
    return r == f32(0) & g == f32(0) & b == f32(0);
}

f32 CSpectrum::get_lum() const
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

CVec3f CSpectrum::to_rgb() const
{
    return CVec3f(r, g, b);
}

CSpectrum load_aligned(ptr<CSpectrum> addr)
{
    f32 r, g, b, u;
    cstd::load_f32x4(cuj::bitcast<ptr<f32>>(addr), r, g, b, u);
    var ret = CSpectrum::from_rgb(r, g, b);
    ret.additional_data = u;
    return ret;
}

void save_aligned(ref<CSpectrum> spec, ptr<CSpectrum> addr)
{
    cstd::store_f32x4(cuj::bitcast<ptr<f32>>(addr), spec.r, spec.g, spec.b, spec.additional_data);
}

Spectrum operator+(const Spectrum &a, const Spectrum &b)
{
    return Spectrum::from_rgb(a.r + b.r, a.g + b.g, a.b + b.b);
}

Spectrum operator-(const Spectrum &a, const Spectrum &b)
{
    return Spectrum::from_rgb(a.r - b.r, a.g - b.g, a.b - b.b);
}

Spectrum operator*(const Spectrum &a, const Spectrum &b)
{
    return Spectrum::from_rgb(a.r * b.r, a.g * b.g, a.b * b.b);
}

Spectrum operator/(const Spectrum &a, const Spectrum &b)
{
    return Spectrum::from_rgb(a.r / b.r, a.g / b.g, a.b / b.b);
}

Spectrum operator+(const Spectrum &a, float b)
{
    return Spectrum::from_rgb(a.r + b, a.g + b, a.b + b);
}

Spectrum operator*(const Spectrum &a, float b)
{
    return Spectrum::from_rgb(a.r * b, a.g * b, a.b * b);
}

Spectrum operator/(const Spectrum &a, float b)
{
    return Spectrum::from_rgb(a.r / b, a.g / b, a.b / b);
}

Spectrum operator+(float a, const Spectrum &b)
{
    return b + a;
}

Spectrum operator*(float a, const Spectrum &b)
{
    return b * a;
}

Spectrum (max)(const Spectrum &a, const Spectrum &b)
{
    return Spectrum::from_rgb(std::max(a.r, b.r), std::max(a.g, b.g), std::max(a.b, b.b));
}

Spectrum (min)(const Spectrum &a, const Spectrum &b)
{
    return Spectrum::from_rgb(std::min(a.r, b.r), std::min(a.g, b.g), std::min(a.b, b.b));
}

CSpectrum operator+(const CSpectrum &a, const CSpectrum &b)
{
    return CSpectrum::from_rgb(a.r + b.r, a.g + b.g, a.b + b.b);
}

CSpectrum operator-(const CSpectrum &a, const CSpectrum &b)
{
    return CSpectrum::from_rgb(a.r - b.r, a.g - b.g, a.b - b.b);
}

CSpectrum operator*(const CSpectrum &a, const CSpectrum &b)
{
    return CSpectrum::from_rgb(a.r * b.r, a.g * b.g, a.b * b.b);
}

CSpectrum operator/(const CSpectrum &a, const CSpectrum &b)
{
    return CSpectrum::from_rgb(a.r / b.r, a.g / b.g, a.b / b.b);
}

CSpectrum operator+(const CSpectrum &a, f32 b)
{
    return CSpectrum::from_rgb(a.r + b, a.g + b, a.b + b);
}

CSpectrum operator*(const CSpectrum &a, f32 b)
{
    return CSpectrum::from_rgb(a.r * b, a.g * b, a.b * b);
}

CSpectrum operator/(const CSpectrum &a, f32 b)
{
    return CSpectrum::from_rgb(a.r / b, a.g / b, a.b / b);
}

CSpectrum operator+(f32 a, const CSpectrum &b)
{
    return b + a;
}

CSpectrum operator*(f32 a, const CSpectrum &b)
{
    return b * a;
}

CSpectrum (max)(const CSpectrum &a, const CSpectrum &b)
{
    return CSpectrum::from_rgb(cstd::max(a.r, b.r), cstd::max(a.g, b.g), cstd::max(a.b, b.b));
}

CSpectrum (min)(const CSpectrum &a, const CSpectrum &b)
{
    return CSpectrum::from_rgb(cstd::min(a.r, b.r), cstd::min(a.g, b.g), cstd::min(a.b, b.b));
}

CSpectrum lerp(const CSpectrum &a, const CSpectrum &b, f32 t)
{
    return a * (1.0f - t) + b * t;
}

BTRC_END
