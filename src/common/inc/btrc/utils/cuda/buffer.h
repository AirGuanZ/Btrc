#pragma once

#include <cassert>
#include <span>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <btrc/utils/cuda/error.h>
#include <btrc/utils/scope_guard.h>
#include <btrc/utils/uncopyable.h>

BTRC_CUDA_BEGIN

template<typename T = char>
class CUDABuffer : public Uncopyable
{
public:

    CUDABuffer();

    explicit CUDABuffer(size_t elem_count, const T *cpu_data = nullptr);

    explicit CUDABuffer(std::span<const T> data);

    CUDABuffer(CUDABuffer &&other) noexcept;

    CUDABuffer &operator=(CUDABuffer &&other) noexcept;

    ~CUDABuffer();

    void initialize(size_t elem_count, const T *cpu_data = nullptr);

    void destroy();

    void swap(CUDABuffer &other) noexcept;

    operator bool() const;

    bool is_empty() const;

    size_t get_size() const;

    size_t get_size_in_bytes() const;

    operator T *();

    operator const T *() const;

    operator CUdeviceptr();

    T *get();

    const T *get() const;

    CUdeviceptr get_device_ptr();

    template<typename U>
    U *as();

    template<typename U>
    const U *as() const;

    void clear(const T &val);

    void clear_bytes(uint8_t byte);

    void to_cpu(T *output, size_t beg = 0, size_t end = 0) const;

    void from_cpu(const T *cpu_data, size_t beg = 0, size_t end = 0);

private:

    size_t elem_count_;
    T     *buffer_;
};

// ========================== impl ==========================

template<typename T>
CUDABuffer<T>::CUDABuffer()
    : elem_count_(0), buffer_(nullptr)
{
    
}

template<typename T>
CUDABuffer<T>::CUDABuffer(size_t elem_count, const T *cpu_data)
    : CUDABuffer()
{
    if(elem_count)
        initialize(elem_count, cpu_data);
}

template<typename T>
CUDABuffer<T>::CUDABuffer(std::span<const T> data)
    : CUDABuffer(data.size(), data.data())
{
    
}

template<typename T>
CUDABuffer<T>::CUDABuffer(CUDABuffer &&other) noexcept
    : CUDABuffer()
{
    swap(other);
}

template<typename T>
CUDABuffer<T> &CUDABuffer<T>::operator=(CUDABuffer &&other) noexcept
{
    swap(other);
    return *this;
}

template<typename T>
CUDABuffer<T>::~CUDABuffer()
{
    cudaFree(buffer_);
}

template<typename T>
void CUDABuffer<T>::initialize(size_t elem_count, const T *cpu_data)
{
    destroy();
    assert(elem_count);
    elem_count_ = elem_count;
    throw_on_error(cudaMalloc(&buffer_, sizeof(T) * elem_count));
    if(cpu_data)
    {
        BTRC_SCOPE_FAIL{ cudaFree(buffer_); };
        this->from_cpu(cpu_data);
    }
}

template<typename T>
void CUDABuffer<T>::destroy()
{
    cudaFree(buffer_);
    elem_count_ = 0;
    buffer_ = nullptr;
}

template<typename T>
void CUDABuffer<T>::swap(CUDABuffer &other) noexcept
{
    std::swap(elem_count_, other.elem_count_);
    std::swap(buffer_, other.buffer_);
}

template<typename T>
CUDABuffer<T>::operator bool() const
{
    return buffer_ != nullptr;
}

template<typename T>
bool CUDABuffer<T>::is_empty() const
{
    return buffer_ == nullptr;
}

template<typename T>
size_t CUDABuffer<T>::get_size() const
{
    return elem_count_;
}

template<typename T>
size_t CUDABuffer<T>::get_size_in_bytes() const
{
    return elem_count_ * sizeof(T);
}

template<typename T>
CUDABuffer<T>::operator T*()
{
    return buffer_;
}

template<typename T>
CUDABuffer<T>::operator const T*() const
{
    return buffer_;
}

template<typename T>
CUDABuffer<T>::operator CUdeviceptr()
{
    return get_device_ptr();
}

template<typename T>
T *CUDABuffer<T>::get()
{
    return buffer_;
}

template<typename T>
const T *CUDABuffer<T>::get() const
{
    return buffer_;
}

template<typename T>
CUdeviceptr CUDABuffer<T>::get_device_ptr()
{
    return reinterpret_cast<CUdeviceptr>(buffer_);
}

template<typename T>
template<typename U>
U *CUDABuffer<T>::as()
{
    return reinterpret_cast<U *>(buffer_);
}

template<typename T>
template<typename U>
const U *CUDABuffer<T>::as() const
{
    return reinterpret_cast<const U *>(buffer_);
}

template<typename T>
void CUDABuffer<T>::clear(const T &val)
{
    assert(!is_empty());
    std::vector<T> vals(elem_count_, val);
    this->from_cpu(vals.data());
}

template<typename T>
void CUDABuffer<T>::clear_bytes(uint8_t byte)
{
    assert(!is_empty());
    throw_on_error(cudaMemset(buffer_, byte, get_size_in_bytes()));
}

template<typename T>
void CUDABuffer<T>::from_cpu(const T *cpu_data, size_t beg, size_t end)
{
    if(end <= beg)
        end = elem_count_;
    assert(beg < end);
    const size_t bytes = sizeof(T) * (end - beg);
    throw_on_error(cudaMemcpy(
        buffer_ + beg, cpu_data, bytes, cudaMemcpyHostToDevice));
}

template<typename T>
void CUDABuffer<T>::to_cpu(T *output, size_t beg, size_t end) const
{
    if(end <= beg)
        end = elem_count_;
    assert(beg < end);
    const size_t bytes = sizeof(T) * (end - beg);
    throw_on_error(cudaMemcpy(
        output, buffer_ + beg, bytes, cudaMemcpyDeviceToHost));
}

BTRC_CUDA_END
