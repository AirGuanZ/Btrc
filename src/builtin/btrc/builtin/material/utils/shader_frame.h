#pragma once

#include <btrc/core/material.h>

BTRC_BEGIN

CUJ_CLASS_BEGIN(ShaderFrame)

    CUJ_MEMBER_VARIABLE(CFrame, geometry)
    CUJ_MEMBER_VARIABLE(CFrame, shading)

    boolean is_black_fringes(ref<CVec3f> w) const;

    boolean is_black_fringes(ref<CVec3f> w1, ref<CVec3f> w2) const;

    Shader::SampleResult sample_black_fringes(
        ref<CVec3f> wo, ref<CVec3f> sam, ref<CSpectrum> albedo) const;

    CSpectrum eval_black_fringes(
        ref<CVec3f> wi, ref<CVec3f> wo, ref<CSpectrum> albedo) const;

    f32 pdf_black_fringes(ref<CVec3f> wi, ref<CVec3f> wo) const;

    f32 correct_shading_energy(ref<CVec3f> wi) const;

    ShaderFrame flip_for_black_fringes(ref<CVec3f> wo) const;

CUJ_CLASS_END

BTRC_END
