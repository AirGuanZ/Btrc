#include <btrc/core/spectrum/rgb.h>

BTRC_CORE_BEGIN

#define TO_RGB(dst, src) \
    auto &dst = dynamic_cast<const RGBSpectrumImpl&>(src)

#define TO_CRGB(dst, src) \
    auto &dst = dynamic_cast<const CRGBSpectrumImpl &>(src)

Spectrum RGBSpectrumImpl::from_rgb(float r, float g, float b)
{
    auto impl = newBox<RGBSpectrumImpl>();
    impl->r = r;
    impl->g = g;
    impl->b = b;
    return Spectrum(std::move(impl));
}

Spectrum RGBSpectrumImpl::add(const SpectrumImpl &other) const
{
    TO_RGB(rhs, other);
    return from_rgb(r + rhs.r, g + rhs.g, b + rhs.b);
}

Spectrum RGBSpectrumImpl::mul(const SpectrumImpl &other) const
{
    TO_RGB(rhs, other);
    return from_rgb(r * rhs.r, g * rhs.g, b * rhs.b);
}

Spectrum RGBSpectrumImpl::add(float v) const
{
    return from_rgb(r + v, g + v, b + v);
}

Spectrum RGBSpectrumImpl::mul(float v) const
{
    return from_rgb(r * v, g * v, b * v);
}

void RGBSpectrumImpl::assign(const SpectrumImpl &other)
{
    TO_RGB(rhs, other);
    r = rhs.r;
    g = rhs.g;
    b = rhs.b;
}

Vec3f RGBSpectrumImpl::to_rgb() const
{
    return Vec3f(r, g, b);
}

bool RGBSpectrumImpl::is_zero() const
{
    return r == 0.0f && g == 0.0f && b == 0.0f;
}

float RGBSpectrumImpl::get_lum() const
{
    return (std::max)((std::max)(r, g), b);
}

Box<SpectrumImpl> RGBSpectrumImpl::clone() const
{
    auto ret = newBox<RGBSpectrumImpl>();
    ret->r = r;
    ret->g = g;
    ret->b = b;
    return Box<SpectrumImpl>(ret.release());
}

Box<CSpectrumImpl> RGBSpectrumImpl::to_cspectrum() const
{
    auto ret = newBox<CRGBSpectrumImpl>();
    ret->r = r;
    ret->g = g;
    ret->b = b;
    return Box<CSpectrumImpl>(ret.release());
}

const SpectrumType *RGBSpectrumImpl::get_type() const
{
    return RGBSpectrumType::get_instance();
}

CSpectrum CRGBSpectrumImpl::from_rgb(f32 r, f32 g, f32 b)
{
    auto impl = newBox<CRGBSpectrumImpl>();
    impl->r = r;
    impl->g = g;
    impl->b = b;
    return CSpectrum(std::move(impl));
}

CSpectrum CRGBSpectrumImpl::add(const CSpectrumImpl &other) const
{
    TO_CRGB(rhs, other);
    return from_rgb(r + rhs.r, g + rhs.g, b + rhs.b);
}

CSpectrum CRGBSpectrumImpl::mul(const CSpectrumImpl &other) const
{
    TO_CRGB(rhs, other);
    return from_rgb(r * rhs.r, g * rhs.g, b * rhs.b);
}

CSpectrum CRGBSpectrumImpl::add(f32 v) const
{
    return from_rgb(r + v, g + v, b + v);
}

CSpectrum CRGBSpectrumImpl::mul(f32 v) const
{
    return from_rgb(r * v, g * v, b * v);
}

void CRGBSpectrumImpl::assign(const CSpectrumImpl &other)
{
    TO_CRGB(rhs, other);
    r = rhs.r;
    g = rhs.g;
    b = rhs.b;
}

CVec3f CRGBSpectrumImpl::to_rgb() const
{
    return CVec3f(r, g, b);
}

boolean CRGBSpectrumImpl::is_zero() const
{
    return r == 0.0f & g == 0.0f & b == 0.0f;
}

f32 CRGBSpectrumImpl::get_lum() const
{
    return cstd::max(cstd::max(r, g), b);
}

Box<CSpectrumImpl> CRGBSpectrumImpl::clone() const
{
    auto ret = newBox<CRGBSpectrumImpl>();
    ret->r = r;
    ret->g = g;
    ret->b = b;
    return Box<CSpectrumImpl>(ret.release());
}

const SpectrumType *CRGBSpectrumImpl::get_type() const
{
    return RGBSpectrumType::get_instance();
}

const RGBSpectrumType *RGBSpectrumType::get_instance()
{
    static const RGBSpectrumType ret;
    return &ret;
}

int RGBSpectrumType::get_word_count() const
{
    return 4;
}

Spectrum RGBSpectrumType::create_one() const
{
    return RGBSpectrumImpl::from_rgb(1, 1, 1);
}

Spectrum RGBSpectrumType::create_zero() const
{
    return RGBSpectrumImpl::from_rgb(0, 0, 0);
}

Spectrum RGBSpectrumType::create_from_rgb(float r, float g, float b) const
{
    return RGBSpectrumImpl::from_rgb(r, g, b);
}

CSpectrum RGBSpectrumType::create_c_from_rgb(f32 r, f32 g, f32 b) const
{
    return CRGBSpectrumImpl::from_rgb(r, g, b);
}

CSpectrum RGBSpectrumType::load(ptr<f32> beta) const
{
    f32 a, b, c, d;
    cstd::load_f32x4(beta, a, b, c, d);
    return CRGBSpectrumImpl::from_rgb(a, b, c);
}

void RGBSpectrumType::save(ptr<f32> beta, const CSpectrum &spec) const
{
    auto &impl = spec.impl();
    TO_CRGB(rgb, impl);
    cstd::store_f32x4(beta, rgb.r, rgb.g, rgb.b, 0.0f);
}

BTRC_CORE_END
