#pragma once

#include <cuj.h>

#include <btrc/core/utils/cuda/buffer.h>
#include <btrc/core/utils/math/math.h>

BTRC_CORE_BEGIN

class CompileContext;

template<typename T>
class PropertyPool;

template<typename T>
class Property
{
#define BTRC_PROPERTY_TYPE(X) std::is_same_v<T, X>
#define BTRC_PROPERTY_TYPE_SEPERATOR ||

    static_assert(
#include <btrc/core/compile/property_type_list.txt>
        );

public:

    using Pool = PropertyPool<T>;

    Property() = default;

    Property(Pool *pool, T *device_value);

    void swap(Property &other) noexcept;

    operator bool() const;

    const T &get() const;

    void set(const T &value);

    bool is_dirty() const;

    void update_device_value();

    cuj::cxx<T> read(const CompileContext &ctx) const;

private:

    struct Impl : Uncopyable
    {
        ~Impl();

        Pool *pool;
        bool is_dirty;
        T value;
        T *device_value;
    };

    RC<Impl> impl_;
};

BTRC_CORE_END
