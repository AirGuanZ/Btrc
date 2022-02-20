#pragma once

#include <format>
#include <map>
#include <string>

#include <cuj.h>

#include <btrc/utils/any.h>
#include <btrc/utils/bind.h>
#include <btrc/utils/scope_guard.h>

BTRC_BEGIN

class Object : public std::enable_shared_from_this<Object>
{
public:

    virtual ~Object() = default;

    RC<Object> as_shared() { return this->shared_from_this(); }

    RC<const Object> as_shared() const { return this->shared_from_this(); }

protected:

    template<typename MemberFuncPtr, typename...Args>
        requires std::is_member_function_pointer_v<MemberFuncPtr>
    auto record(MemberFuncPtr ptr, std::string_view action_name, Args...args) const;
};

class CompileContext
{
public:

    static CompileContext *get_current_context();

    static void push_context(CompileContext *context);

    static void pop_context();

    template<typename ObjectAction, typename...Args>
    auto record_object_action(
        RC<const Object>    object,
        const std::string  &action_name,
        const ObjectAction &action,
        Args             ...args);
    
    template<typename ObjectAction, typename...Args>
    auto record_object_action(
        RC<Object>          object,
        const std::string  &action_name,
        const ObjectAction &action,
        Args             ...args);

private:

    struct ActionRecord
    {
        Any cuj_func;
    };

    struct ObjectRecord
    {
        std::map<std::string, ActionRecord, std::less<>> actions;
    };

    std::map<RC<const Object>, ObjectRecord> object_records_;
};

// ========================== impl ==========================

namespace object_detail
{

    template<typename F>
    struct MemberFuncionToClassTypeAux;

    template<typename C, typename F>
    struct MemberFuncionToClassTypeAux<F C::*>
    {
        using Type = C;
    };

} // namespace object_detail

template<typename MemberFuncPtr, typename...Args>
    requires std::is_member_function_pointer_v<MemberFuncPtr>
auto Object::record(MemberFuncPtr ptr, std::string_view action_name, Args...args) const
{
    using Class = typename object_detail::MemberFuncionToClassTypeAux<MemberFuncPtr>::Type;
    static_assert(std::is_base_of_v<Object, Class>);
    auto this_ptr = dynamic_cast<const Class *>(this);
    return CompileContext::get_current_context()->record_object_action(
        this->as_shared(), std::string(action_name), bind_this(ptr, this_ptr), args...);
}

template<typename ObjectAction, typename...Args>
auto CompileContext::record_object_action(
    RC<const Object>    object,
    const std::string  &action_name,
    const ObjectAction &action,
    Args             ...args)
{
    using StdFunction = decltype(std::function{ action });
    using CujFunction = decltype(cuj::Function{ std::declval<StdFunction>() });

    auto &object_record = object_records_[object];

    if(auto it = object_record.actions.find(action_name);
       it != object_record.actions.end())
    {
        Any &untyped_func = it->second.cuj_func;
        auto func = untyped_func.as<CujFunction>();
        return func(args...);
    }

    const auto func_symbol_name = std::format(
        "btrc_{}_of_object_{}", action_name, static_cast<const void*>(object.get()));

    auto func = cuj::function(func_symbol_name, action);
    auto &action_record = object_record.actions[action_name];
    action_record.cuj_func = func;
    return func(args...);
}

template<typename ObjectAction, typename...Args>
auto CompileContext::record_object_action(
    RC<Object>          object,
    const std::string  &action_name,
    const ObjectAction &action,
    Args             ...args)
{
    return this->record_object_action(
        RC<const Object>(object), action_name, action, args...);
}

BTRC_END
