#pragma once

#include <any>

#include <btrc/core/common.h>

BTRC_CORE_BEGIN

class Any : public std::any
{
public:

    using any::any;

    template<typename T>
    bool is() const { return as_if<T>() != nullptr; }

    template<typename T>
    T as() const { return std::any_cast<T>(*this); }

    template<typename T>
    T as() { return std::any_cast<T>(*this); }

    template<typename T>
    T *as_if() { return std::any_cast<T>(this); }

    template<typename T>
    const T *as_if() const { return std::any_cast<T>(this); }
};

BTRC_CORE_END
