#pragma once

#include <btrc/core/material/material.h>

BTRC_CORE_BEGIN

class Glass : public Material
{
public:

    void set_color(const Spectrum &color);

    void set_ior(float ior);

    RC<Shader> create_shader(const CIntersection &inct) const override;

private:

    Spectrum color_ = Spectrum::from_rgb(0.8f, 0.8f, 0.8f);
    float ior_ = 1.5f;
};

BTRC_CORE_END
