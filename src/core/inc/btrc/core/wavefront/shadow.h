#pragma once

#include <btrc/core/film/film.h>
#include <btrc/core/utils/uncopyable.h>

BTRC_WAVEFRONT_BEGIN

class ShadowPipeline : public Uncopyable
{
public:

    ShadowPipeline() = default;

    explicit ShadowPipeline(Film &film);
};

BTRC_WAVEFRONT_END
