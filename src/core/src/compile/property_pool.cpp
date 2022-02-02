#include <btrc/core/compile/property_pool.h>

BTRC_CORE_BEGIN

template<typename T>
Property<T> PropertyPool<T>::allocate_property()
{
    if(free_device_values_.empty())
        allocate_chunk();
    assert(!free_device_values_.empty());
    CUJ_SCOPE_SUCCESS{ free_device_values_.pop_back(); };
    return Property<T>(this, free_device_values_.back());
}

template<typename T>
void PropertyPool<T>::free_device_value(T *device_ptr)
{
    free_device_values_.push_back(device_ptr);
}

template<typename T>
void PropertyPool<T>::allocate_chunk()
{
    constexpr int CHUNK_VALUE_COUNT = 512;
    chunks_.push_back(CUDABuffer<T>(CHUNK_VALUE_COUNT));
    auto last_chunk = chunks_.back().get();
    for(int i = 0; i < CHUNK_VALUE_COUNT; ++i)
        free_device_values_.push_back(last_chunk + i);
}

#define BTRC_PROPERTY_TYPE(X) template<> class PropertyPool<X>;
#include <btrc/core/compile/property_type_list.txt>

BTRC_CORE_END
