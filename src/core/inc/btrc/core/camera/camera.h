#pragma once

#include <btrc/core/spectrum/spectrum.h>

BTRC_CORE_BEGIN

class Camera
{
public:

    struct SampleWeResult
    {
        CVec3f    pos;
        CVec3f    dir;
        CSpectrum throughput;
        f32       time;
    };

    virtual ~Camera() = default;

    virtual SampleWeResult generate_ray(
        const CVec2f &film_coord, f32 time_sample) const = 0;
};

BTRC_CORE_END
