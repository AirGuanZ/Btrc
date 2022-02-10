#pragma once

#include <btrc/core/material/material.h>

BTRC_CORE_BEGIN

class Diffuse : public Material
{
public:

    void set_albedo(const Spectrum &albedo);
    
    RC<Shader> create_shader(const SurfacePoint &inct) const override;

private:

    Spectrum albedo_ = Spectrum::from_rgb(0.8f, 0.8f, 0.8f);
};

BTRC_CORE_END
