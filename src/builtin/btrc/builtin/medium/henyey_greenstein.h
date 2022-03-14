#pragma once

#include <btrc/core/medium.h>

BTRC_BUILTIN_BEGIN

class HenyeyGreensteinPhaseShader : public PhaseShader
{
public:

    void set_g(f32 g);

    void set_color(ref<CSpectrum> color);

    SampleResult sample(CompileContext &cc, ref<CVec3f> wo, ref<CVec3f> sam) const override;

    CSpectrum eval(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo) const override;

    f32 pdf(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo) const override;

private:

    f32 g_;
    CSpectrum color_;
};

BTRC_BUILTIN_END
