#pragma once

#include <cuda.h>

#include <btrc/utils/uncopyable.h>

BTRC_CUDA_BEGIN

class Context : public Uncopyable
{
public:

    Context();

    explicit Context(int device_index);

    Context(Context &&other) noexcept;

    Context &operator=(Context &&other) noexcept;

    ~Context();

    void swap(Context &other) noexcept;

    int get_device() const;

    operator bool() const;

    operator CUcontext() const;

protected:

    int       device_;
    CUcontext context_;
};

BTRC_CUDA_END
