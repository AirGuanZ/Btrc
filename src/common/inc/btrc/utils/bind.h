#pragma once

#include <btrc/common.h>

BTRC_BEGIN

namespace bind_detail
{

    template<typename F>
    struct BindThisAux { };

    template<typename Class, typename Ret, typename...Args>
    struct BindThisAux<Ret(Class::*)(Args...)>
    {
        Class *class_ptr;
        Ret(Class::*mem_func_ptr)(Args...);

        auto operator()(Args...args)
        {
            return ((*class_ptr).*mem_func_ptr)(args...);
        }
    };

    template<typename Class, typename Ret, typename...Args>
    struct BindThisAux<Ret(Class::*)(Args...)const>
    {
        const Class *class_ptr;
        Ret(Class::*mem_func_ptr)(Args...)const;

        auto operator()(Args...args) const
        {
            return ((*class_ptr).*mem_func_ptr)(args...);
        }
    };

} // namespace bind_detail

template<typename F, typename Class>
    requires std::is_member_function_pointer_v<F>
auto bind_this(F f, Class *class_ptr)
{
    bind_detail::BindThisAux<F> ret;
    ret.class_ptr = class_ptr;
    ret.mem_func_ptr = f;
    return ret;
}

template<typename F, typename Class>
auto bind_this(F f, const Class *class_ptr)
{
    bind_detail::BindThisAux<F> ret;
    ret.class_ptr = class_ptr;
    ret.mem_func_ptr = f;
    return ret;
}

BTRC_END
