#pragma once

#include <btrc/common.h>

BTRC_BEGIN

class Uncopyable
{
public:

    Uncopyable()                              = default;
    Uncopyable(const Uncopyable &)            = delete;
    Uncopyable &operator=(const Uncopyable &) = delete;
};

BTRC_END
