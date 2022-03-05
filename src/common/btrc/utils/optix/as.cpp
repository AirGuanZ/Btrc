#include <btrc/utils/optix/as.h>

BTRC_OPTIX_BEGIN

SingleBufferAS::SingleBufferAS()
    : SingleBufferAS(0, {})
{
    
}

SingleBufferAS::SingleBufferAS(OptixTraversableHandle handle, cuda::Buffer<> buffer)
    : handle_(handle), buffer_(std::move(buffer))
{

}

SingleBufferAS::SingleBufferAS(SingleBufferAS &&other) noexcept
    : SingleBufferAS()
{
    swap(other);
}

SingleBufferAS &SingleBufferAS::operator=(SingleBufferAS &&other) noexcept
{
    swap(other);
    return *this;
}

void SingleBufferAS::swap(SingleBufferAS &other) noexcept
{
    std::swap(handle_, other.handle_);
    buffer_.swap(other.buffer_);
}

SingleBufferAS::operator bool() const
{
    return !buffer_.is_empty();
}

OptixTraversableHandle SingleBufferAS::get_handle() const
{
    return handle_;
}

BTRC_OPTIX_END
