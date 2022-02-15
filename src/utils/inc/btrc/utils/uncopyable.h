#pragma once

#include <btrc/common.h>

BTRC_BEGIN

class Uncopyable
{
public:

    BTRC_CPU Uncopyable()                              = default;
    BTRC_CPU Uncopyable(const Uncopyable &)            = delete;
    BTRC_CPU Uncopyable &operator=(const Uncopyable &) = delete;
};

BTRC_END
