#pragma once

#include <exception>
#include <string>

#include <btrc/common.h>

BTRC_BEGIN

template<typename TIt>
void extract_hierarchy_exceptions(const std::exception &e, TIt out_it)
{
    *out_it++ = e.what();
    try
    {
        std::rethrow_if_nested(e);
    }
    catch(const std::exception &e2)
    {
        extract_hierarchy_exceptions(e2, out_it);
    }
    catch(...)
    {
        *out_it++ = "an unknown exception was thrown";
    }
}

#define BTRC_HI_TRY try {

#define BTRC_HI_WRAP(MSG) \
    } \
    catch(...) \
    { \
        std::throw_with_nested(std::runtime_error(MSG)); \
    }

inline std::string extract_exception_ptr(const std::exception_ptr &ptr)
{
    try
    {
        std::rethrow_exception(ptr);
    }
    catch(const std::exception &err)
    {
        return err.what();
    }
    catch(...)
    {
        return "unknown exception";
    }
}

BTRC_END
