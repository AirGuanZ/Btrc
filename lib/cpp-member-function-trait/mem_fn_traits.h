/*
 * MIT License
 * 
 * Copyright (c) 2022 Zhuang Guan
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*
 * Usage:
 *
 *     member_function_pointer_trait<decltype(&Class::MemberFunction)>::
 *         class_type
 *         return_type
 *         arg_types // tuple of arguments, exists when has_va_args is false
 *         arg<I>    // argument type
 *         has_va_args // is declared with ...
 *         has_const_qualifier
 *         has_volatile_qualifier
 *         has_lvalue_ref_qualifier
 *         has_rvalue_ref_qualifier
 *
 * Example:
 * 
 *     class A
 *     {
 *     public:
 *
 *         void f1(const int &z) const;
 *     
 *         void f2(...) volatile &;
 *     };
 *
 *     member_function_pointer_trait<decltype(&A::f1)>::arg<0>                   -> const int &
 *     member_function_pointer_trait<decltype(&A::f1)>::has_const_qualifier      -> true
 *     member_function_pointer_trait<decltype(&A::f2)>::has_va_args              -> true
 *     member_function_pointer_trait<decltype(&A::f2)>::has_volatile_qualifier   -> true
 *     member_function_pointer_trait<decltype(&A::f2)>::has_rvalue_ref_qualifier -> true
 */

#ifndef MEM_FN_TRAIT_H
#define MEM_FN_TRAIT_H

#include <tuple>

template<typename T>
struct member_function_pointer_trait;

template<typename C, typename R, typename...As>
struct member_function_pointer_trait<R(C::*)(As...)>
{
    using class_type  = C;
    using return_type = R;
    using arg_types   = std::tuple<As...>;

    static constexpr int n_args = std::tuple_size<arg_types>::value;
    template<int I>
    using arg = typename std::tuple_element<I, arg_types>::type;

    static constexpr bool has_va_args              = false;
    static constexpr bool has_const_qualifier      = false;
    static constexpr bool has_volatile_qualifier   = false;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R, typename...As>
struct member_function_pointer_trait<R(C::*)(As...) &&>
{
    using class_type  = C;
    using return_type = R;
    using arg_types   = std::tuple<As...>;

    static constexpr int n_args = std::tuple_size<arg_types>::value;
    template<int I>
    using arg = typename std::tuple_element<I, arg_types>::type;
    
    static constexpr bool has_va_args              = false;
    static constexpr bool has_const_qualifier      = false;
    static constexpr bool has_volatile_qualifier   = false;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = true;
};

template<typename C, typename R, typename...As>
struct member_function_pointer_trait<R(C::*)(As...) &>
{
    using class_type  = C;
    using return_type = R;
    using arg_types   = std::tuple<As...>;

    static constexpr int n_args = std::tuple_size<arg_types>::value;
    template<int I>
    using arg = typename std::tuple_element<I, arg_types>::type;
    
    static constexpr bool has_va_args              = false;
    static constexpr bool has_const_qualifier      = false;
    static constexpr bool has_volatile_qualifier   = false;
    static constexpr bool has_lvalue_ref_qualifier = true;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R, typename...As>
struct member_function_pointer_trait<R(C::*)(As...) volatile>
{
    using class_type  = C;
    using return_type = R;
    using arg_types   = std::tuple<As...>;

    static constexpr int n_args = std::tuple_size<arg_types>::value;
    template<int I>
    using arg = typename std::tuple_element<I, arg_types>::type;
    
    static constexpr bool has_va_args              = false;
    static constexpr bool has_const_qualifier      = false;
    static constexpr bool has_volatile_qualifier   = true;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R, typename...As>
struct member_function_pointer_trait<R(C::*)(As...) volatile &&>
{
    using class_type  = C;
    using return_type = R;
    using arg_types   = std::tuple<As...>;

    static constexpr int n_args = std::tuple_size<arg_types>::value;
    template<int I>
    using arg = typename std::tuple_element<I, arg_types>::type;
    
    static constexpr bool has_va_args              = false;
    static constexpr bool has_const_qualifier      = false;
    static constexpr bool has_volatile_qualifier   = true;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = true;
};

template<typename C, typename R, typename...As>
struct member_function_pointer_trait<R(C::*)(As...) volatile &>
{
    using class_type  = C;
    using return_type = R;
    using arg_types   = std::tuple<As...>;

    static constexpr int n_args = std::tuple_size<arg_types>::value;
    template<int I>
    using arg = typename std::tuple_element<I, arg_types>::type;
    
    static constexpr bool has_va_args              = false;
    static constexpr bool has_const_qualifier      = false;
    static constexpr bool has_volatile_qualifier   = true;
    static constexpr bool has_lvalue_ref_qualifier = true;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R, typename...As>
struct member_function_pointer_trait<R(C::*)(As...) const>
{
    using class_type  = C;
    using return_type = R;
    using arg_types   = std::tuple<As...>;

    static constexpr int n_args = std::tuple_size<arg_types>::value;
    template<int I>
    using arg = typename std::tuple_element<I, arg_types>::type;
    
    static constexpr bool has_va_args              = false;
    static constexpr bool has_const_qualifier      = true;
    static constexpr bool has_volatile_qualifier   = false;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R, typename...As>
struct member_function_pointer_trait<R(C::*)(As...) const &&>
{
    using class_type  = C;
    using return_type = R;
    using arg_types   = std::tuple<As...>;

    static constexpr int n_args = std::tuple_size<arg_types>::value;
    template<int I>
    using arg = typename std::tuple_element<I, arg_types>::type;
    
    static constexpr bool has_va_args              = false;
    static constexpr bool has_const_qualifier      = true;
    static constexpr bool has_volatile_qualifier   = false;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = true;
};

template<typename C, typename R, typename...As>
struct member_function_pointer_trait<R(C::*)(As...) const &>
{
    using class_type  = C;
    using return_type = R;
    using arg_types   = std::tuple<As...>;

    static constexpr int n_args = std::tuple_size<arg_types>::value;
    template<int I>
    using arg = typename std::tuple_element<I, arg_types>::type;
    
    static constexpr bool has_va_args              = false;
    static constexpr bool has_const_qualifier      = true;
    static constexpr bool has_volatile_qualifier   = false;
    static constexpr bool has_lvalue_ref_qualifier = true;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R, typename...As>
struct member_function_pointer_trait<R(C::*)(As...) const volatile>
{
    using class_type  = C;
    using return_type = R;
    using arg_types   = std::tuple<As...>;

    static constexpr int n_args = std::tuple_size<arg_types>::value;
    template<int I>
    using arg = typename std::tuple_element<I, arg_types>::type;
    
    static constexpr bool has_va_args              = false;
    static constexpr bool has_const_qualifier      = true;
    static constexpr bool has_volatile_qualifier   = true;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R, typename...As>
struct member_function_pointer_trait<R(C::*)(As...) const volatile &&>
{
    using class_type  = C;
    using return_type = R;
    using arg_types   = std::tuple<As...>;

    static constexpr int n_args = std::tuple_size<arg_types>::value;
    template<int I>
    using arg = typename std::tuple_element<I, arg_types>::type;
    
    static constexpr bool has_va_args              = false;
    static constexpr bool has_const_qualifier      = true;
    static constexpr bool has_volatile_qualifier   = true;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = true;
};

template<typename C, typename R, typename...As>
struct member_function_pointer_trait<R(C::*)(As...) const volatile &>
{
    using class_type  = C;
    using return_type = R;
    using arg_types   = std::tuple<As...>;

    static constexpr int n_args = std::tuple_size<arg_types>::value;
    template<int I>
    using arg = typename std::tuple_element<I, arg_types>::type;
    
    static constexpr bool has_va_args              = false;
    static constexpr bool has_const_qualifier      = true;
    static constexpr bool has_volatile_qualifier   = true;
    static constexpr bool has_lvalue_ref_qualifier = true;
    static constexpr bool has_rvalue_ref_qualifier = false;
};
template<typename C, typename R>
struct member_function_pointer_trait<R(C::*)(...)>
{
    using class_type  = C;
    using return_type = R;
    
    static constexpr bool has_va_args              = true;
    static constexpr bool has_const_qualifier      = false;
    static constexpr bool has_volatile_qualifier   = false;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R>
struct member_function_pointer_trait<R(C::*)(...) &&>
{
    using class_type  = C;
    using return_type = R;
    
    static constexpr bool has_va_args              = true;
    static constexpr bool has_const_qualifier      = false;
    static constexpr bool has_volatile_qualifier   = false;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = true;
};

template<typename C, typename R>
struct member_function_pointer_trait<R(C::*)(...) &>
{
    using class_type  = C;
    using return_type = R;
    
    static constexpr bool has_va_args              = true;
    static constexpr bool has_const_qualifier      = false;
    static constexpr bool has_volatile_qualifier   = false;
    static constexpr bool has_lvalue_ref_qualifier = true;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R>
struct member_function_pointer_trait<R(C::*)(...) volatile>
{
    using class_type  = C;
    using return_type = R;
    
    static constexpr bool has_va_args              = true;
    static constexpr bool has_const_qualifier      = false;
    static constexpr bool has_volatile_qualifier   = true;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R>
struct member_function_pointer_trait<R(C::*)(...) volatile &&>
{
    using class_type  = C;
    using return_type = R;

    static constexpr bool has_const_qualifier      = false;
    static constexpr bool has_volatile_qualifier   = true;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = true;
};

template<typename C, typename R>
struct member_function_pointer_trait<R(C::*)(...) volatile &>
{
    using class_type  = C;
    using return_type = R;
    
    static constexpr bool has_va_args              = true;
    static constexpr bool has_const_qualifier      = false;
    static constexpr bool has_volatile_qualifier   = true;
    static constexpr bool has_lvalue_ref_qualifier = true;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R>
struct member_function_pointer_trait<R(C::*)(...) const>
{
    using class_type  = C;
    using return_type = R;
    
    static constexpr bool has_va_args              = true;
    static constexpr bool has_const_qualifier      = true;
    static constexpr bool has_volatile_qualifier   = false;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R>
struct member_function_pointer_trait<R(C::*)(...) const &&>
{
    using class_type  = C;
    using return_type = R;
    
    static constexpr bool has_va_args              = true;
    static constexpr bool has_const_qualifier      = true;
    static constexpr bool has_volatile_qualifier   = false;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = true;
};

template<typename C, typename R>
struct member_function_pointer_trait<R(C::*)(...) const &>
{
    using class_type  = C;
    using return_type = R;
    
    static constexpr bool has_va_args              = true;
    static constexpr bool has_const_qualifier      = true;
    static constexpr bool has_volatile_qualifier   = false;
    static constexpr bool has_lvalue_ref_qualifier = true;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R>
struct member_function_pointer_trait<R(C::*)(...) const volatile >
{
    using class_type  = C;
    using return_type = R;
    
    static constexpr bool has_va_args              = true;
    static constexpr bool has_const_qualifier      = true;
    static constexpr bool has_volatile_qualifier   = true;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

template<typename C, typename R>
struct member_function_pointer_trait<R(C::*)(...) const volatile &&>
{
    using class_type  = C;
    using return_type = R;
    
    static constexpr bool has_va_args              = true;
    static constexpr bool has_const_qualifier      = true;
    static constexpr bool has_volatile_qualifier   = true;
    static constexpr bool has_lvalue_ref_qualifier = false;
    static constexpr bool has_rvalue_ref_qualifier = true;
};

template<typename C, typename R>
struct member_function_pointer_trait<R(C::*)(...) const volatile &>
{
    using class_type  = C;
    using return_type = R;
    
    static constexpr bool has_va_args              = true;
    static constexpr bool has_const_qualifier      = true;
    static constexpr bool has_volatile_qualifier   = true;
    static constexpr bool has_lvalue_ref_qualifier = true;
    static constexpr bool has_rvalue_ref_qualifier = false;
};

#endif // #ifndef MEM_FN_TRAIT_H
