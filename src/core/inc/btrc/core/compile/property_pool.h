#pragma once

#include <btrc/core/compile/property.h>

BTRC_CORE_BEGIN

template<typename T>
class PropertyPool : public Uncopyable
{
public:

    Property<T> allocate_property();

    void free_device_value(T *device_ptr);

private:

    void allocate_chunk();

    std::vector<CUDABuffer<T>> chunks_;
    std::vector<T *> free_device_values_;
};

BTRC_CORE_END
