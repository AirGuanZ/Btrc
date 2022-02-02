#pragma once

#include <btrc/core/compile/context.h>
#include <btrc/core/compile/property_manager.h>
#include <btrc/core/utils/bind.h>

BTRC_CORE_BEGIN

template<typename T>
class Object : public ObjectBase, public std::enable_shared_from_this<T>
{
public:

    Object();

    explicit Object(PropertyManager &property_manager);

    template<typename Derived = T>
    RC<Derived> as_shared();

    template<typename Derived = const T>
    RC<const Derived> as_shared() const;

protected:

    template<typename MemFuncPtr, typename...Args>
    auto record(MemFuncPtr mem_func, std::string_view name, Args...args);

    template<typename MemFuncPtr, typename...Args>
    auto record(MemFuncPtr mem_func, std::string_view name, Args...args) const;

    template<typename P>
    Property<P> new_property(const P &initial_value = {});

private:

    PropertyManager &property_manager_;
};

// ========================== impl ==========================

template<typename T>
Object<T>::Object()
    : Object(*PropertyManager::get_current_manager())
{

}

template<typename T>
Object<T>::Object(PropertyManager &property_manager)
    : property_manager_(property_manager)
{

}

template<typename T>
template<typename Derived>
RC<Derived> Object<T>::as_shared()
{
    return std::dynamic_pointer_cast<Derived>(this->shared_from_this());
}

template<typename T>
template<typename Derived>
RC<const Derived> Object<T>::as_shared() const
{
    return std::dynamic_pointer_cast<const Derived>(this->shared_from_this());
}

template<typename T>
template<typename MemFuncPtr, typename...Args>
auto Object<T>::record(MemFuncPtr mem_func, std::string_view name, Args...args)
{
    auto derived = this->as_shared<T>();
    return CompileContext::get_current_context()->record_object_action(
        derived, std::string(name), bind_this(mem_func, derived.get()), args...);
}

template<typename T>
template<typename MemFuncPtr, typename...Args>
auto Object<T>::record(MemFuncPtr mem_func, std::string_view name, Args...args) const
{
    auto derived = this->as_shared<T>();
    return CompileContext::get_current_context()->record_object_action(
        derived, std::string(name), bind_this(mem_func, derived.get()), args...);
}

template<typename T>
template<typename P>
Property<P> Object<T>::new_property(const P &initial_value)
{
    auto prop = property_manager_.allocate_property<P>();
    prop.set(initial_value);
    return prop;
}

BTRC_CORE_END
