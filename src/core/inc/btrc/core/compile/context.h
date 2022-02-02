#pragma once

#include <format>
#include <map>

#include <cuj.h>

#include <btrc/core/compile/object_base.h>
#include <btrc/core/utils/any.h>
#include <btrc/core/utils/scope_guard.h>

BTRC_CORE_BEGIN

class CompileContext
{
public:

    static CompileContext *get_current_context();

    static void push_context(CompileContext *context);

    static void pop_context();

    explicit CompileContext(bool offline);

    bool is_offline() const;

    bool should_inline(const RC<const ObjectBase> &object) const;

    template<typename DerivedObject,
             typename ObjectAction,
             typename...Args>
        requires std::is_base_of_v<ObjectBase, DerivedObject>
    auto record_object_action(
        RC<const DerivedObject> object,
        const std::string      &action_name,
        const ObjectAction     &action,
        Args                 ...args);
    
    template<typename DerivedObject,
             typename ObjectAction,
             typename...Args>
        requires std::is_base_of_v<ObjectBase, DerivedObject>
    auto record_object_action(
        RC<DerivedObject>   object,
        const std::string  &action_name,
        const ObjectAction &action,
        Args             ...args);

    std::vector<std::string_view> generate_separate_codes() const;

private:

    struct ActionRecord
    {
        Any cuj_func;
        Any cuj_decl;
    };

    struct ObjectRecord
    {
        mutable std::string cached_code;
        cuj::Module cuj_module;
        std::map<std::string, ActionRecord, std::less<>> actions;
    };

    bool offline_;

    std::map<RC<const ObjectBase>, ObjectRecord> object_records_;
};

// ========================== impl ==========================

template<typename DerivedObject, typename ObjectAction, typename...Args>
    requires std::is_base_of_v<ObjectBase, DerivedObject>
auto CompileContext::record_object_action(
    RC<const DerivedObject> object,
    const std::string      &action_name,
    const ObjectAction     &action,
    Args                ... args)
{
    using StdFunction = decltype(std::function{ action });
    using CujFunction = decltype(cuj::Function{ std::declval<StdFunction>() });

    auto &object_record = object_records_[object];
    
    if(auto it = object_record.actions.find(action_name);
       it != object_record.actions.end())
    {
        Any &untyped_func = it->second.cuj_func;
        Any &untyped_decl = it->second.cuj_decl;
        auto func = untyped_func.as<CujFunction>();
        auto decl = untyped_decl.as<CujFunction>();
        assert(func.get_module() == &object_record.cuj_module);
        assert(decl.get_module() == nullptr);
        return decl(args...);
    }

    const auto func_symbol_name = std::format(
        "btrc_{}_of_object_{}", action_name, static_cast<const void*>(object.get()));

    auto old_cuj_module = cuj::Module::get_current_module();
    BTRC_SCOPE_EXIT{ cuj::Module::set_current_module(old_cuj_module); };

    if(!this->should_inline(object))
        cuj::Module::set_current_module(&object_record.cuj_module);
    auto func = cuj::function(func_symbol_name, action);

    cuj::Module::set_current_module(nullptr);
    CujFunction decl;
    decl.set_name(func_symbol_name);

    auto &action_record = object_record.actions[action_name];
    action_record.cuj_func = func;
    action_record.cuj_decl = decl;

    cuj::Module::set_current_module(old_cuj_module);
    return decl(args...);
}

template<typename DerivedObject, typename ObjectAction, typename ... Args>
    requires std::is_base_of_v<ObjectBase, DerivedObject>
auto CompileContext::record_object_action(
    RC<DerivedObject>   object,
    const std::string  &action_name,
    const ObjectAction &action,
    Args             ...args)
{
    return this->record_object_action(
        RC<const DerivedObject>(object), action_name, action, args...);
}

BTRC_CORE_END
