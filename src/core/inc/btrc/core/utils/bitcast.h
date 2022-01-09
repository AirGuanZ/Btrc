#pragma once

#include <cstring>

#include <btrc/core/common.h>

BTRC_CORE_BEGIN

template<typename To, typename From>
BTRC_XPU To bitcast(From from)
{
    static_assert(sizeof(To) == sizeof(From));
    To result;
    std::memcpy(&result, &from, sizeof(From));
    return result;
}

BTRC_CORE_END
