#pragma once

#include <btrc/core/common.h>

BTRC_CORE_BEGIN

[[noreturn]] inline void unreachable()
{
#if defined(_MSC_VER)
    __assume(0);
#elif defined(__clang__) || defined(__GNUC__)
    __builtin_unreachable();
#else
    std::terminate();
#endif
}

BTRC_CORE_END
