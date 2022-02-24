#pragma once

#include <format>
#include <map>
#include <string>

#include <cuj.h>
#include <mem_fn_traits.h>

#include <cuda_runtime.h>

#include <btrc/utils/any.h>
#include <btrc/utils/bind.h>
#include <btrc/utils/cuda/error.h>
#include <btrc/utils/scope_guard.h>

BTRC_BEGIN

class CompileContext;
class Object;

class PropertyCommon
{
public:

    virtual ~PropertyCommon() = default;

    virtual void update() = 0;

    virtual bool is_dirty() const = 0;
};

template<typename T>
class Property : public PropertyCommon, public Uncopyable
{
public:

    Property();

    template<typename...Args>
    explicit Property(T *device_pointer, Args&&...args);

    Property(Property &&other) noexcept;

    Property &operator=(Property &&other) noexcept;

    ~Property() override;

    void swap(Property &other) noexcept;

    operator bool() const;

    void set(const T &value);

    const T &get() const;

    cuj::cxx<T> read(CompileContext &cc) const;

    void update() override;

    bool is_dirty() const override;

private:

    bool is_dirty_;
    T value_;
    T *device_pointer_;
};

template<typename T>
class PropertySlot
{
public:

    RC<Property<T>> prop;

    void set(const T &value) { prop->set(value); }

    auto &operator=(const T &value) { this->set(value); return *this; }

    const T &get() const { return prop->get(); }

    cuj::cxx<T> read(CompileContext &cc) const { return prop->read(cc); }

    void update() { prop->update(); }

    bool is_dirty() const { return prop->is_dirty(); }
};

class PropertyPool : public Uncopyable
{
    PropertyPool() = default;

public:

    ~PropertyPool();

    static void initialize_instance();

    static void destroy_instance();

    static PropertyPool &get_instance();

    template<typename T, typename...Args>
    Property<T> allocate(Args&&...args);

    template<typename T>
    void release(T *device_pointer);

private:

    static void new_chunk(std::vector<void *> &output, size_t size, size_t align);

    void *allocate_impl(std::type_index type_index, size_t size, size_t align);

    void release_impl(std::type_index type_index, void *device_pointer);

    std::vector<void *> chunks_;
    std::map<std::type_index, std::vector<void *>> free_properties_;
};

class ScopedPropertyPool : public Uncopyable
{
public:

    ScopedPropertyPool();

    ~ScopedPropertyPool();
};

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

    bool need_recompile() const;

    bool need_commit() const;

    void set_recompile(bool recompile = true);

    std::vector<PropertyCommon *> get_properties();

    virtual std::vector<RC<Object>> get_dependent_objects();

    virtual void commit() { }

protected:

    template<typename MemberFuncPtr, typename...Args>
        requires std::is_member_function_pointer_v<MemberFuncPtr>
    auto record(CompileContext &cc, MemberFuncPtr ptr, std::string_view action_name, Args...args) const;

    template<typename T, typename...Args>
    PropertySlot<T> new_property(Args&&...args);

    template<typename T>
    ObjectSlot<T> new_object();

private:

    bool need_recompile_ = true;

    std::vector<PropertyCommon *> properties_;
    std::vector<ObjectReferenceCommon *> dependent_objects_;
};

#define BTRC_OBJECT(TYPE, NAME) ObjectSlot<TYPE> NAME = new_object<TYPE>()
#define BTRC_PROPERTY(TYPE, NAME, ...) PropertySlot<TYPE> NAME = new_property<TYPE>(__VA_ARGS__)

class CompileContext
{
public:

    explicit CompileContext(bool offline = true);

    bool is_offline_mode() const;

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

    bool offline_mode_;
    std::map<RC<const Object>, ObjectRecord> object_records_;
};

// ========================== impl ==========================

template<typename T>
Property<T>::Property()
    : Property(nullptr)
{

}

template<typename T>
template<typename...Args>
Property<T>::Property(T *device_pointer, Args&&...args)
    : is_dirty_(true), value_(std::forward<Args>(args)...), device_pointer_(device_pointer)
{

}

template<typename T>
Property<T>::Property(Property &&other) noexcept
    : Property()
{
    this->swap(other);
}

template<typename T>
Property<T> &Property<T>::operator=(Property &&other) noexcept
{
    this->swap(other);
    return *this;
}

template<typename T>
Property<T>::~Property()
{
    if(device_pointer_)
        PropertyPool::get_instance().release(device_pointer_);
}

template<typename T>
void Property<T>::swap(Property &other) noexcept
{
    std::swap(value_, other.value_);
    std::swap(device_pointer_, other.device_pointer_);
}

template<typename T>
Property<T>::operator bool() const
{
    return device_pointer_ != nullptr;
}

template<typename T>
void Property<T>::set(const T &value)
{
    assert(*this);
    value_ = value;
    is_dirty_ = true;
}

template<typename T>
const T &Property<T>::get() const
{
    assert(*this);
    return value_;
}

template<typename T>
cuj::cxx<T> Property<T>::read(CompileContext &cc) const
{
    assert(*this);
    if(cc.is_offline_mode())
        return cuj::cxx<T>(value_);
    return *cuj::import_pointer(device_pointer_);
}

template<typename T>
void Property<T>::update()
{
    assert(*this);
    if(is_dirty_)
    {
        throw_on_error(cudaMemcpy(
            device_pointer_, &value_, sizeof(T), cudaMemcpyHostToDevice));
        is_dirty_ = false;
    }
}

template<typename T>
bool Property<T>::is_dirty() const
{
    assert(*this);
    return is_dirty_;
}

template<typename T, typename...Args>
Property<T> PropertyPool::allocate(Args&&...args)
{
    auto device_pointer = allocate_impl(std::type_index(typeid(T)), sizeof(T), alignof(T));
    return Property<T>(static_cast<T*>(device_pointer), std::forward<Args>(args)...);
}

template<typename T>
void PropertyPool::release(T *device_pointer)
{
    this->release_impl(std::type_index(typeid(T)), device_pointer);
}

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

inline bool Object::need_recompile() const
{
    return need_recompile_;
}

inline bool Object::need_commit() const
{
    if(need_recompile_)
        return true;
    for(auto p : properties_)
    {
        if(p->is_dirty())
            return true;
    }
    return false;
}

inline void Object::set_recompile(bool recompile)
{
    need_recompile_ = recompile;
}

inline std::vector<PropertyCommon*> Object::get_properties()
{
    return properties_;
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

template<typename T, typename...Args>
PropertySlot<T> Object::new_property(Args&&...args)
{
    auto prop = newRC<Property<T>>(PropertyPool::get_instance().allocate<T>(std::forward<Args>(args)...));
    properties_.push_back(prop.get());
    PropertySlot<T> result;
    result.prop = std::move(prop);
    return result;
}

template<typename T>
ObjectSlot<T> Object::new_object()
{
    ObjectSlot<T> result;
    result.reference = newRC<ObjectReference<T>>();
    dependent_objects_.push_back(result.reference.get());
    return result;
}

inline CompileContext::CompileContext(bool offline)
    : offline_mode_(offline)
{
    
}

inline bool CompileContext::is_offline_mode() const
{
    return offline_mode_;
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
