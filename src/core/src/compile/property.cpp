#include <btrc/core/compile/context.h>
#include <btrc/core/compile/property_pool.h>
#include <btrc/core/utils/cmath/cmath.h>
#include <btrc/core/utils/math/math.h>

BTRC_CORE_BEGIN

template<typename T>
Property<T>::Property(Pool *pool, T *device_value)
{
    impl_ = newRC<Impl>();
    impl_->pool = pool;
    impl_->is_dirty = true;
    impl_->value = T{};
    impl_->device_value = device_value;
    assert(impl_->pool && impl_->device_value);
}

template<typename T>
void Property<T>::swap(Property &other) noexcept
{
    impl_.swap(other.impl_);
}

template<typename T>
Property<T>::operator bool() const
{
    return impl_;
}

template<typename T>
const T &Property<T>::get() const
{
    return impl_->value;
}

template<typename T>
void Property<T>::set(const T &value)
{
    impl_->value = value;
    impl_->is_dirty = true;
}

template<typename T>
bool Property<T>::is_dirty() const
{
    return impl_->is_dirty;
}

template<typename T>
void Property<T>::update_device_value()
{
    assert(*this);
    if(impl_->is_dirty)
    {
        throw_on_error(cudaMemcpy(
            impl_->device_value, &impl_->value,
            sizeof(T), cudaMemcpyHostToDevice));
        impl_->is_dirty = false;
    }
}

template<typename T>
cuj::cxx<T> Property<T>::read(const CompileContext &ctx) const
{
    if(ctx.is_offline())
    {
        if constexpr(std::is_arithmetic_v<T>)
            return cuj::cxx<T>(impl_->value);
        else if constexpr(std::is_same_v<T, Vec2f>)
        {
            return CVec2f(
                impl_->value.x, impl_->value.y);
        }
        else if constexpr(std::is_same_v<T, Vec3f>)
        {
            return CVec3f(
                impl_->value.x, impl_->value.y, impl_->value.z);
        }
        else if constexpr(std::is_same_v<T, Vec4f>)
        {
            return CVec4f(
                impl_->value.x, impl_->value.y, impl_->value.z, impl_->value.w);
        }
        else
        {
            static_assert(std::is_same_v<T, Quaterion>);
            return CQuaterion(
                impl_->value.w, impl_->value.x, impl_->value.y, impl_->value.z);
        }
    }
    return *cuj::import_pointer(impl_->device_value);
}

template<typename T>
Property<T>::Impl::~Impl()
{
    assert(pool && device_value);
    pool->free_device_value(device_value);
}

#define BTRC_PROPERTY_TYPE(X) template<> class Property<X>;
#include <btrc/core/compile/property_type_list.txt>

BTRC_CORE_END
