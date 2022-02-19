#pragma once

#include <optix.h>

#include <btrc/utils/uncopyable.h>

BTRC_OPTIX_BEGIN

class Module : public Uncopyable
{
public:

    Module();

    explicit Module(OptixModule optix_module);

    Module(Module &&other) noexcept;

    Module &operator=(Module &&other) noexcept;

    ~Module();

    operator bool() const;

    operator OptixModule() const;

    OptixModule get_handle() const;
};

BTRC_OPTIX_END
