#pragma once

#include <memory>
#include <string>

#include <btrc/core/utils/uncopyable.h>

BTRC_CORE_BEGIN

class CUDAModule : public Uncopyable
{
public:

    struct Dim3 { int x = 1, y = 1, z = 1; };

    CUDAModule();

    CUDAModule(CUDAModule &&other) noexcept;

    CUDAModule &operator=(CUDAModule &&other) noexcept;

    ~CUDAModule();

    void swap(CUDAModule &other) noexcept;

    bool is_linked() const;

    void load_ptx_from_memory(const void *data, size_t bytes);

    void load_ptx_from_file(const std::string &filename);

    void link();
    
    template<typename...Args>
    void launch(
        const std::string &entry_name,
        const Dim3        &block_cnt,
        const Dim3        &block_size,
        Args            ...kernel_args) const;

private:

    template<typename Arg0>
    void take_kernel_arg_ptrs(void **arg_ptrs, Arg0 &arg0) const;

    template<typename Arg0, typename Arg1, typename...Args>
    void take_kernel_arg_ptrs(
        void **arg_ptrs, Arg0 &arg0, Arg1 &arg1, Args &...args) const;

    void launch_impl(
        const std::string &entry_name,
        const Dim3        &block_cnt,
        const Dim3        &block_size,
        void             **kernel_args) const;

    struct Impl;

    std::unique_ptr<Impl> impl_;
};

// ========================== impl ==========================

template<typename ... Args>
void CUDAModule::launch(
    const std::string &entry_name,
    const Dim3        &block_cnt,
    const Dim3        &block_size,
    Args            ...kernel_args) const
{
    if constexpr(sizeof...(kernel_args) > 0)
    {
        void *kernel_arg_ptrs[sizeof...(kernel_args)];
        this->take_kernel_arg_ptrs(kernel_arg_ptrs, kernel_args...);
        this->launch_impl(entry_name, block_cnt, block_size, kernel_arg_ptrs);
    }
    else
        this->launch_impl(entry_name, block_cnt, block_size, nullptr);
}

template<typename Arg0>
void CUDAModule::take_kernel_arg_ptrs(void **arg_ptrs, Arg0 &arg0) const
{
    *arg_ptrs = &arg0;
}

template<typename Arg0, typename Arg1, typename...Args>
void CUDAModule::take_kernel_arg_ptrs(
    void **arg_ptrs, Arg0 &arg0, Arg1 &arg1, Args &...args) const
{
    *arg_ptrs = &arg0;
    this->take_kernel_arg_ptrs(arg_ptrs + 1, arg1, args...);
}

BTRC_CORE_END
