#pragma once

#include <btrc/core/utils/bitflag.h>

BTRC_CORE_BEGIN

class ObjectBase
{
public:

    enum class CompileOption
    {
        Auto,
        Separate,
        Inlined
    };

    virtual ~ObjectBase() = default;

    virtual CompileOption get_compile_option() const;
};

// ========================== impl ==========================

inline ObjectBase::CompileOption ObjectBase::get_compile_option() const
{
    return CompileOption::Auto;
}

BTRC_CORE_END
