#pragma once

#include <btrc/builtin/material/utils/aggregate.h>

BTRC_BUILTIN_BEGIN

CUJ_CLASS_BEGIN(DiffuseComponentImpl)

    CUJ_MEMBER_VARIABLE(CSpectrum, albedo_value)

    BSDFComponent::SampleResult sample(ref<CVec3f> lwo, ref<Sam3> sam, TransportMode mode) const;

    BSDFComponent::SampleBidirResult sample_bidir(ref<CVec3f> lwo, ref<Sam3> sam, TransportMode mode) const;

    CSpectrum eval(ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const;

    f32 pdf(ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const;

    CSpectrum albedo() const;

CUJ_CLASS_END

BTRC_BUILTIN_END
