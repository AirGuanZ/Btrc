#pragma once

#include <optix.h>

#include <btrc/core/utils/cuda/buffer.h>
#include <btrc/core/utils/optix/as.h>
#include <btrc/core/utils/uncopyable.h>

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

    SingleBufferAS(
        OptixTraversableHandle handle,
        CUDABuffer<>           buffer);

    SingleBufferAS(SingleBufferAS &&other) noexcept;

    SingleBufferAS &operator=(SingleBufferAS &&other) noexcept;

    void swap(SingleBufferAS &other) noexcept;

    operator bool() const;

    OptixTraversableHandle get_handle() const override;

private:

    OptixTraversableHandle handle_;
    CUDABuffer<>           buffer_;
};

using TriangleAS = SingleBufferAS;
using InstanceAS = SingleBufferAS;

BTRC_OPTIX_END
