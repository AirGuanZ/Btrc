#pragma once

#include <btrc/core/compile/property_pool.h>

BTRC_CORE_BEGIN

class PropertyManager : public Uncopyable
{
public:

    static PropertyManager *get_current_manager();

    static void push_manager(PropertyManager *manager);

    static void pop_manager();

    PropertyManager();

    PropertyManager(PropertyManager &&other) noexcept;

    PropertyManager &operator=(PropertyManager &&other) noexcept;

    void swap(PropertyManager &other) noexcept;

    template<typename T>
    Property<T> allocate_property();

private:

    template<typename...Args>
    using Pools = std::tuple<RC<PropertyPool<Args>>...>;

#define BTRC_PROPERTY_TYPE(X) X
#define BTRC_PROPERTY_TYPE_SEPERATOR ,

    Pools<
#include <btrc/core/compile/property_type_list.txt>
    > pools_;
};

template<typename T>
Property<T> PropertyManager::allocate_property()
{
    return std::get<RC<PropertyPool<T>>>(pools_)->allocate_property();
}

BTRC_CORE_END
