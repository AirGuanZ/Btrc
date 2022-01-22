#include <btrc/core/spectrum/spectrum.h>

BTRC_CORE_BEGIN

Spectrum::Spectrum(Box<SpectrumImpl> impl)
    : impl_(std::move(impl))
{
    
}

Spectrum::Spectrum(const Spectrum &other)
{
    *this = other;
}

Spectrum &Spectrum::operator=(const Spectrum &other)
{
    if(!impl_)
    {
        if(other.impl_)
            impl_ = other.impl_->clone();
    }
    else
        impl_->assign(other.impl());
    return *this;
}

SpectrumImpl &Spectrum::impl()
{
    return *impl_;
}

const SpectrumImpl &Spectrum::impl() const
{
    return *impl_;
}

Vec3f Spectrum::to_rgb() const
{
    return impl_->to_rgb();
}

bool Spectrum::is_zero() const
{
    return impl_->is_zero();
}

float Spectrum::get_lum() const
{
    return impl_->get_lum();
}

CSpectrum Spectrum::to_cspectrum() const
{
    return CSpectrum(impl_->to_cspectrum());
}

const SpectrumType *Spectrum::get_type() const
{
    return impl_->get_type();
}

CSpectrum::CSpectrum(Box<CSpectrumImpl> impl)
    : impl_(std::move(impl))
{
    
}

CSpectrum::CSpectrum(const Spectrum &s)
    : CSpectrum(s.to_cspectrum())
{
    
}

CSpectrum::CSpectrum(const CSpectrum &other)
{
    *this = other;
}

CSpectrum &CSpectrum::operator=(const CSpectrum &other)
{
    if(!impl_)
    {
        if(other.impl_)
            impl_ = other.impl_->clone();
    }
    else
        impl_->assign(other.impl());
    return *this;
}

CSpectrumImpl &CSpectrum::impl()
{
    return *impl_;
}
const CSpectrumImpl &CSpectrum::impl() const
{
    return *impl_;
}

CVec3f CSpectrum::to_rgb() const
{
    return impl_->to_rgb();
}

boolean CSpectrum::is_zero() const
{
    return impl_->is_zero();
}

f32 CSpectrum::get_lum() const
{
    return impl_->get_lum();
}

const SpectrumType *CSpectrum::get_type() const
{
    return impl_->get_type();
}

CSpectrum SpectrumType::create_cone() const
{
    return create_one().to_cspectrum();
}

CSpectrum SpectrumType::create_czero() const
{
    return create_zero().to_cspectrum();
}

CSpectrum SpectrumType::load_soa(ptr<f32> beta, i32 soa_index) const
{
    return load(beta + get_word_count() * soa_index);
}

void SpectrumType::save_soa(
    ptr<f32> beta, const CSpectrum &spec, i32 soa_index) const
{
    save(beta + get_word_count() * soa_index, spec);
}

BTRC_CORE_END
