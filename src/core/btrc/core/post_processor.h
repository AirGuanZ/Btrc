#pragma once

#include <btrc/core/context.h>
#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

class PostProcessor : public Object
{
public:

    enum class ExecutionPolicy
    {
        Always,
        AfterComplete
    };

    virtual ExecutionPolicy get_execution_policy() const = 0;

    virtual void process(
        Vec4f *color,
        Vec4f *albedo,
        Vec4f *normal,
        int    width,
        int    height) = 0;
};

BTRC_END
