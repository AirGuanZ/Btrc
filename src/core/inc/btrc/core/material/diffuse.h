#pragma once

#include <btrc/core/material/material.h>

BTRC_CORE_BEGIN

class Diffuse : public Material
{
public:

    void set_albedo(const Spectrum &albedo);
    
    RC<Shader> create_shader(const CIntersection &inct) const override;

private:

    Spectrum albedo_;
};

BTRC_CORE_END
