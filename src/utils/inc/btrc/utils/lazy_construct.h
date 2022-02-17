#pragma once

#include <btrc/common.h>

BTRC_BEGIN

template<typename Constructor>
class LazyConstructor
{
    Constructor constructor_;

public:

    template<typename F>
        requires !std::is_same_v<std::remove_cvref_t<F>, LazyConstructor>
    explicit LazyConstructor(F &&f)
        : constructor_{ std::forward<F>(f) }
    {

    }

    LazyConstructor(const LazyConstructor &) = default;
    LazyConstructor(LazyConstructor &&) noexcept = default;

    LazyConstructor &operator=(const LazyConstructor &) = default;
    LazyConstructor &operator=(LazyConstructor &&) noexcept = default;

    [[nodiscard]] operator auto() noexcept(noexcept(constructor_()))
    {
        return constructor_();
    }
};

template<typename F>
auto lazy_construct(F &&f) noexcept
{
    return LazyConstructor<std::remove_cvref_t<F>>(std::forward<F>(f));
}

#define BTRC_LAZY_CONSTRUCT(EXPR) \
    (::btrc::core::lazy_construct([&]{ return (EXPR); }))

BTRC_END
