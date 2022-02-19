#include <mutex>

#include <btrc/utils/cuda/context.h>
#include <btrc/utils/cuda/error.h>
#include <btrc/utils/scope_guard.h>

BTRC_CUDA_BEGIN

Context::Context()
    : device_(0), context_(nullptr)
{
    
}

Context::Context(int device_index)
    : device_(device_index), context_(nullptr)
{
    static std::once_flag cuda_init_flag;
    std::call_once(cuda_init_flag, [&]
    {
        throw_on_error(cuInit(0));
    });

    CUdevice cuda_device;
    throw_on_error(cuDeviceGet(&cuda_device, device_index));

    throw_on_error(cuCtxCreate(&context_, 0, cuda_device));
    BTRC_SCOPE_FAIL{ cuCtxDestroy(context_); };
}

Context::Context(Context &&other) noexcept
    : Context()
{
    swap(other);
}

Context &Context::operator=(Context &&other) noexcept
{
    swap(other);
    return *this;
}

Context::~Context()
{
    if(context_)
        cuCtxDestroy(context_);
}

void Context::swap(Context &other) noexcept
{
    std::swap(device_, other.device_);
    std::swap(context_, other.context_);
}

int Context::get_device() const
{
    return device_;
}

Context::operator bool() const
{
    return context_ != nullptr;
}

Context::operator CUcontext() const
{
    return context_;
}

BTRC_CUDA_END
