#pragma once

#include <btrc/core/material/material.h>

BTRC_CORE_BEGIN

class Diffuse : public Material
{
public:

    void set_albedo(const Spectrum &albedo);

    Box<BSDF> create_bsdf(const CIntersection &inct) const override;

private:

    Spectrum albedo_;
};

BTRC_CORE_END
