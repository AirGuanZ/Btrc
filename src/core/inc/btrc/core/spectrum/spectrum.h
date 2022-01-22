#pragma once

#include <btrc/core/utils/cmath/cmath.h>

BTRC_CORE_BEGIN

class CSpectrumImpl;
class CSpectrum;
class SpectrumImpl;
class Spectrum;
class SpectrumType;

class SpectrumImpl
{
public:

    virtual ~SpectrumImpl() = default;

    virtual Spectrum add(const SpectrumImpl &other) const = 0;

    virtual Spectrum mul(const SpectrumImpl &other) const = 0;

    virtual Spectrum add(float v) const = 0;

    virtual Spectrum mul(float v) const = 0;

    virtual void assign(const SpectrumImpl &other) = 0;

    virtual Vec3f to_rgb() const = 0;

    virtual bool is_zero() const = 0;

    virtual float get_lum() const = 0;

    virtual Box<SpectrumImpl> clone() const = 0;

    virtual Box<CSpectrumImpl> to_cspectrum() const = 0;

    virtual const SpectrumType *get_type() const = 0;
};

class CSpectrumImpl
{
public:

    virtual ~CSpectrumImpl() = default;

    virtual CSpectrum add(const CSpectrumImpl &other) const = 0;

    virtual CSpectrum mul(const CSpectrumImpl &other) const = 0;

    virtual CSpectrum add(f32 v) const = 0;

    virtual CSpectrum mul(f32 v) const = 0;

    virtual void assign(const CSpectrumImpl &other) = 0;

    virtual CVec3f to_rgb() const = 0;

    virtual boolean is_zero() const = 0;

    virtual f32 get_lum() const = 0;

    virtual Box<CSpectrumImpl> clone() const = 0;

    virtual const SpectrumType *get_type() const = 0;
};

class Spectrum
{
    Box<SpectrumImpl> impl_;

public:

    explicit Spectrum(Box<SpectrumImpl> impl = {});

    Spectrum(const Spectrum &other);

    Spectrum &operator=(const Spectrum &other);

          SpectrumImpl &impl();
    const SpectrumImpl &impl() const;

    Vec3f to_rgb() const;

    bool is_zero() const;

    float get_lum() const;

    CSpectrum to_cspectrum() const;

    const SpectrumType *get_type() const;
};

class CSpectrum
{
    Box<CSpectrumImpl> impl_;

public:

    explicit CSpectrum(Box<CSpectrumImpl> impl = {});

    CSpectrum(const Spectrum &s);

    CSpectrum(const CSpectrum &other);

    CSpectrum &operator=(const CSpectrum &other);

          CSpectrumImpl &impl();
    const CSpectrumImpl &impl() const;

    CVec3f to_rgb() const;

    boolean is_zero() const;

    f32 get_lum() const;

    const SpectrumType *get_type() const;
};

class SpectrumType
{
public:

    virtual ~SpectrumType() = default;

    virtual int get_word_count() const = 0;

    virtual Spectrum create_one() const = 0;

    virtual Spectrum create_zero() const = 0;

    virtual CSpectrum load(ptr<f32> beta) const = 0;

    virtual void save(ptr<f32> beta, const CSpectrum &spec) const = 0;

    CSpectrum create_cone() const;

    CSpectrum create_czero() const;

    CSpectrum load_soa(ptr<f32> beta, i32 soa_index) const;

    void save_soa(ptr<f32> beta, const CSpectrum &spec, i32 soa_index) const;
};

inline Spectrum operator+(const Spectrum &a, const Spectrum &b) { return a.impl().add(b.impl()); }
inline Spectrum operator*(const Spectrum &a, const Spectrum &b) { return a.impl().mul(b.impl()); }

inline Spectrum operator+(const Spectrum &a, float b) { return a.impl().add(b); }
inline Spectrum operator*(const Spectrum &a, float b) { return a.impl().mul(b); }
inline Spectrum operator/(const Spectrum &a, float b) { return a.impl().mul(1.0f / b); }

inline Spectrum operator+(float a, const Spectrum &b) { return b.impl().add(a); }
inline Spectrum operator*(float a, const Spectrum &b) { return b.impl().mul(a); }

inline CSpectrum operator+(const CSpectrum &a, const CSpectrum &b) { return a.impl().add(b.impl()); }
inline CSpectrum operator*(const CSpectrum &a, const CSpectrum &b) { return a.impl().mul(b.impl()); }

inline CSpectrum operator+(const CSpectrum &a, f32 b) { return a.impl().add(b); }
inline CSpectrum operator*(const CSpectrum &a, f32 b) { return a.impl().mul(b); }
inline CSpectrum operator/(const CSpectrum &a, f32 b) { return a.impl().mul(1.0f / b); }

inline CSpectrum operator+(f32 a, const CSpectrum &b) { return b.impl().add(a); }
inline CSpectrum operator*(f32 a, const CSpectrum &b) { return b.impl().mul(a); }

BTRC_CORE_END
