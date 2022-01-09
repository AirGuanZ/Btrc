#pragma once

#include <exception>
#include <type_traits>

#include <btrc/core/utils/anonymous_name.h>
#include <btrc/core/utils/uncopyable.h>

BTRC_CORE_BEGIN

template<typename T, typename = std::enable_if_t<std::is_invocable_v<T>>>
class scope_guard_t : public Uncopyable
{
    bool call_ = true;

    T func_;

public:

    BTRC_XPU explicit scope_guard_t(const T &func)
        : func_(func)
    {

    }

    BTRC_XPU explicit scope_guard_t(T &&func)
        : func_(std::move(func))
    {

    }

    BTRC_XPU ~scope_guard_t()
    {
        if(call_)
            func_();
    }

    BTRC_XPU void dismiss()
    {
        call_ = false;
    }
};

template<typename F, bool ExecuteOnException>
class exception_scope_guard_t : public Uncopyable
{
    F func_;

    int exceptions_;

public:

    BTRC_CPU explicit exception_scope_guard_t(const F &func)
        : func_(func), exceptions_(std::uncaught_exceptions())
    {

    }

    BTRC_CPU explicit exception_scope_guard_t(F &&func)
        : func_(std::move(func)),
        exceptions_(std::uncaught_exceptions())
    {

    }

    BTRC_CPU ~exception_scope_guard_t()
    {
        const int now_exceptions = std::uncaught_exceptions();
        if((now_exceptions > exceptions_) == ExecuteOnException)
            func_();
    }
};

struct scope_guard_builder_t            {};
struct scope_guard_on_fail_builder_t    {};
struct scope_guard_on_success_builder_t {};

template<typename Func>
BTRC_CPU auto operator+(scope_guard_builder_t, Func &&f)
{
    return scope_guard_t<std::decay_t<Func>>(std::forward<Func>(f));
}

template<typename Func>
BTRC_CPU auto operator+(scope_guard_on_fail_builder_t, Func &&f)
{
    return exception_scope_guard_t<std::decay_t<Func>, true>(
        std::forward<Func>(f));
}

template<typename Func>
BTRC_CPU auto operator+(scope_guard_on_success_builder_t, Func &&f)
{
    return exception_scope_guard_t<std::decay_t<Func>, false>(
        std::forward<Func>(f));
}

#define BTRC_SCOPE_EXIT                                                         \
    auto BTRC_ANONYMOUS_NAME(_btrc_scope_exit) =                                \
    ::btrc::core::scope_guard_builder_t{} + [&]

#define BTRC_SCOPE_FAIL                                                         \
    auto BTRC_ANONYMOUS_NAME(_btrc_scope_fail) =                                \
    ::btrc::core::scope_guard_on_fail_builder_t{} + [&]

#define BTRC_SCOPE_SUCCESS                                                      \
    auto BTRC_ANONYMOUS_NAME(_btrc_scope_success) =                             \
    ::btrc::core::scope_guard_on_success_builder_t{} + [&]

BTRC_CORE_END
