#pragma once

#include <map>
#include <string>

#include <cuj.h>
#include <fmt/format.h>
#include <mem_fn_traits.h>

#include <btrc/core/object_proxy.h>
#include <btrc/utils/any.h>
#include <btrc/utils/bind.h>
#include <btrc/utils/cuda/error.h>

BTRC_BEGIN

class CompileContext;
class Object;

class ObjectReferenceCommon
{
public:

    virtual ~ObjectReferenceCommon() = default;

    virtual RC<Object> get_object() = 0;
};

template<typename T>
class ObjectReference : public ObjectReferenceCommon
{
public:

    RC<T> object;

    RC<Object> get_object() override { return object; }
};

template<typename T>
class ObjectSlot
{
public:

    RC<ObjectReference<T>> reference;

    void set(RC<T> object);

    const RC<T> &get() const;

    ObjectSlot &operator=(RC<T> object);

    auto operator->() const;
};

class Object : public std::enable_shared_from_this<Object>
{
public:

    virtual ~Object() = default;

    RC<Object> as_shared();

    RC<const Object> as_shared() const;

    virtual bool should_compile_separately() const { return false; }

    virtual void commit() { }

    virtual std::vector<RC<Object>> get_dependent_objects();

    virtual void create_proxy(ObjectProxy &proxy) { }

protected:

    template<typename MemberFuncPtr, typename...Args>
        requires std::is_member_function_pointer_v<MemberFuncPtr>
    auto record(CompileContext &cc, MemberFuncPtr ptr, std::string_view action_name, Args...args) const;

    template<typename T>
    ObjectSlot<T> new_object();

private:

    std::vector<ObjectReferenceCommon *> dependent_objects_;
};

#define BTRC_OBJECT(TYPE, NAME) ObjectSlot<TYPE> NAME = new_object<TYPE>()

class CompileContext
{
public:

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

    void set_allow_separate_compile(bool allow);

    std::vector<const cuj::Module *> get_separate_modules() const;

private:

    struct ActionRecord
    {
        Any cuj_func;
    };

    struct ObjectRecord
    {
        Box<cuj::Module> separate_module;
        std::map<std::pair<const cuj::Module*, std::string>, ActionRecord, std::less<>> actions;
    };

    bool allow_separate_compile_ = false;

    std::map<RC<const Object>, ObjectRecord> object_records_;
};

// ========================== impl ==========================

template<typename T>
void ObjectSlot<T>::set(RC<T> object)
{
    reference->object = std::move(object);
}

template<typename T>
const RC<T> &ObjectSlot<T>::get() const
{
    return reference->object;
}

template<typename T>
ObjectSlot<T> &ObjectSlot<T>::operator=(RC<T> object)
{
    this->set(std::move(object));
    return *this;
}

template<typename T>
auto ObjectSlot<T>::operator->() const
{
    return get().get();
}

namespace object_detail
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
    
    template<typename F>
    struct BindThisAndCCAux { };
    
    template<typename Class, typename Ret, typename...Args>
    struct BindThisAndCCAux<Ret(Class::*)(CompileContext&, Args...)>
    {
        Class *class_ptr;
        CompileContext *cc;
        Ret(Class::*mem_func_ptr)(CompileContext&, Args...);
    
        auto operator()(Args...args)
        {
            return ((*class_ptr).*mem_func_ptr)(*cc, args...);
        }
    };
    
    template<typename Class, typename Ret, typename...Args>
    struct BindThisAndCCAux<Ret(Class::*)(CompileContext&, Args...)const>
    {
        const Class *class_ptr;
        CompileContext *cc;
        Ret(Class::*mem_func_ptr)(CompileContext&, Args...)const;
    
        auto operator()(Args...args) const
        {
            return ((*class_ptr).*mem_func_ptr)(*cc, args...);
        }
    };

    template<typename F, typename Class>
        requires std::is_member_function_pointer_v<F>
    auto bind_this(F f, Class *class_ptr)
    {
        BindThisAux<F> ret;
        ret.class_ptr = class_ptr;
        ret.mem_func_ptr = f;
        return ret;
    }

    template<typename F, typename Class>
    auto bind_this(F f, const Class *class_ptr)
    {
        BindThisAux<F> ret;
        ret.class_ptr = class_ptr;
        ret.mem_func_ptr = f;
        return ret;
    }

    template<typename F, typename Class>
        requires std::is_member_function_pointer_v<F>
    auto bind_this(F f, Class *class_ptr, CompileContext &cc)
    {
        BindThisAndCCAux<F> ret;
        ret.class_ptr = class_ptr;
        ret.cc = &cc;
        ret.mem_func_ptr = f;
        return ret;
    }

    template<typename F, typename Class>
    auto bind_this(F f, const Class *class_ptr, CompileContext &cc)
    {
        BindThisAndCCAux<F> ret;
        ret.class_ptr = class_ptr;
        ret.cc = &cc;
        ret.mem_func_ptr = f;
        return ret;
    }

} // namespace object_detail

inline RC<Object> Object::as_shared()
{
    return this->shared_from_this();
}

inline RC<const Object> Object::as_shared() const
{
    return this->shared_from_this();
}

inline std::vector<RC<Object>> Object::get_dependent_objects()
{
    std::vector<RC<Object>> result;
    for(auto &o : dependent_objects_)
    {
        auto obj = o->get_object();
        assert(obj);
        result.push_back(std::move(obj));
    }
    return result;
}

template<typename MemberFuncPtr, typename...Args>
    requires std::is_member_function_pointer_v<MemberFuncPtr>
auto Object::record(CompileContext &cc, MemberFuncPtr ptr, std::string_view action_name, Args...args) const
{
    using MemFnTrait = member_function_pointer_trait<MemberFuncPtr>;
    using Class = typename MemFnTrait::class_type;
    static_assert(std::is_base_of_v<Object, Class>);
    auto this_ptr = dynamic_cast<const Class *>(this);

    if constexpr(MemFnTrait::n_args > 0)
    {
        using Arg0 = typename MemFnTrait::template arg<0>;
        if constexpr(std::is_same_v<std::remove_cvref_t<Arg0>, CompileContext>)
        {
            return cc.record_object_action(
                this->as_shared(), std::string(action_name),
                object_detail::bind_this(ptr, this_ptr, cc), args...);
        }
        else
        {
            return cc.record_object_action(
                this->as_shared(), std::string(action_name),
                object_detail::bind_this(ptr, this_ptr), args...);
        }
    }
    else
    {
        return cc.record_object_action(
            this->as_shared(), std::string(action_name),
            object_detail::bind_this(ptr, this_ptr), args...);
    }
}

template<typename T>
ObjectSlot<T> Object::new_object()
{
    ObjectSlot<T> result;
    result.reference = newRC<ObjectReference<T>>();
    dependent_objects_.push_back(result.reference.get());
    return result;
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

    auto old_module = cuj::Module::get_current_module();
    auto &object_record = object_records_[object];

    if(auto it = object_record.actions.find({ old_module, action_name }); it != object_record.actions.end())
    {
        Any &untyped_func = it->second.cuj_func;
        auto func = untyped_func.as<CujFunction>();
        return func(args...);
    }

    if(!(allow_separate_compile_ && object->should_compile_separately()))
    {
        // separately compiled function are always defined in current bound module.
        // thus it may duplicate in modules. we encode module ptr in its name to avoid symbol conflict.

        const auto func_symbol_name = fmt::format(
            "btrc_{}_of_object_{}_in_{}", action_name,
            static_cast<const void *>(object.get()),
            static_cast<const void*>(old_module));

        auto func = cuj::function(func_symbol_name, action);
        auto &action_record = object_record.actions[{ old_module, action_name }];
        action_record.cuj_func = func;
        return func(args...);
    }

    const auto func_symbol_name = fmt::format(
        "btrc_{}_of_object_{}", action_name, static_cast<const void *>(object.get()));

    if(!object_record.separate_module)
        object_record.separate_module = newBox<cuj::Module>();
    auto separate_module = object_record.separate_module.get();

    // define func in its own module, and...

    cuj::Module::set_current_module(separate_module);
    if(object_record.actions.find({ separate_module, action_name }) == object_record.actions.end())
    {
        auto func = cuj::function(func_symbol_name, action);
        auto &separate_action_record = object_record.actions[{ separate_module, action_name }];
        separate_action_record.cuj_func = func;
    }
    cuj::Module::set_current_module(old_module);

    // declare it in current module

    CujFunction decl;
    decl.set_name(func_symbol_name);
    auto &action_record = object_record.actions[{ old_module, action_name }];
    action_record.cuj_func = decl;
    return decl(args...);
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

inline void CompileContext::set_allow_separate_compile(bool allow)
{
    allow_separate_compile_ = allow;
}

inline std::vector<const cuj::Module*> CompileContext::get_separate_modules() const
{
    std::vector<const cuj::Module *> ret;
    for(auto &[_, record] : object_records_)
    {
        if(record.separate_module)
            ret.push_back(record.separate_module.get());
    }
    return ret;
}

BTRC_END
