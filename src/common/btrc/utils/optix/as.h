#pragma once

#include <btrc/utils/cuda/buffer.h>
#include <btrc/utils/uncopyable.h>

BTRC_OPTIX_BEGIN

class AS
{
public:

    virtual ~AS() = default;

    virtual OptixTraversableHandle get_handle() const = 0;

    operator OptixTraversableHandle() const { return get_handle(); }
};

class SingleBufferAS : public Uncopyable, public AS
{
public:

    SingleBufferAS();

    SingleBufferAS(OptixTraversableHandle handle, cuda::Buffer<> buffer);

    SingleBufferAS(SingleBufferAS &&other) noexcept;

    SingleBufferAS &operator=(SingleBufferAS &&other) noexcept;

    void swap(SingleBufferAS &other) noexcept;

    operator bool() const;

    OptixTraversableHandle get_handle() const override;

private:

    OptixTraversableHandle handle_;
    cuda::Buffer<>         buffer_;
};

using TriangleAS = SingleBufferAS;
using InstanceAS = SingleBufferAS;

BTRC_OPTIX_END
