#include <btrc/core/material.h>

BTRC_BEGIN

Shader::SampleResult Shader::discard_pdf_rev(const SampleBidirResult &result)
{
    SampleResult ret;
    ret.bsdf = result.bsdf;
    ret.dir = result.dir;
    ret.pdf = result.pdf;
    ret.is_delta = result.is_delta;
    return ret;
}

BTRC_END
