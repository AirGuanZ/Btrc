#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>

#include <cuda.h>

#include <btrc/core/utils/cuda/error.h>
#include <btrc/core/utils/cuda/module.h>
#include <btrc/core/utils/scope_guard.h>

BTRC_CORE_BEGIN

struct CUDAModule::Impl
{
    std::vector<std::string> ptx_data;

    CUmodule cu_module = nullptr;
};

CUDAModule::CUDAModule()
{
    impl_ = std::make_unique<Impl>();
}

CUDAModule::CUDAModule(CUDAModule &&other) noexcept
    : CUDAModule()
{
    std::swap(impl_, other.impl_);
}

CUDAModule &CUDAModule::operator=(CUDAModule &&other) noexcept
{
    std::swap(impl_, other.impl_);
    return *this;
}

CUDAModule::~CUDAModule()
{
    if(impl_->cu_module)
        cuModuleUnload(impl_->cu_module);
}

void CUDAModule::load_ptx_from_memory(const void *data, size_t bytes)
{
    std::string new_data;
    new_data.resize(bytes);
    std::memcpy(new_data.data(), data, bytes);
    impl_->ptx_data.push_back(new_data);
}

void CUDAModule::load_ptx_from_file(const std::string &filename)
{
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    if(!fin)
        throw std::runtime_error("failed to open file: " + filename);

    fin.seekg(0, std::ios::end);
    const auto len = fin.tellg();
    fin.seekg(0, std::ios::beg);

    if(!len)
        throw std::runtime_error("empty ptx file: " + filename);

    std::vector<char> data(static_cast<size_t>(len) + 1, '\0');
    fin.read(data.data(), len);
    if(!fin)
        throw std::runtime_error("failed to read context from " + filename);

    load_ptx_from_memory(data.data(), data.size());
}

void CUDAModule::link()
{
    assert(!impl_->cu_module && !impl_->ptx_data.empty());

    CUlinkState link_state = nullptr;
    BTRC_SCOPE_EXIT{ if(link_state) cuLinkDestroy(link_state); };

    std::array<char, 4096> err_buffer;
    std::array<CUjit_option, 2> options = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    };
    std::array<void *, 2> option_values = {
        err_buffer.data(),
        reinterpret_cast<void *>(err_buffer.size())
    };

    throw_on_error(cuLinkCreate(
        options.size(), options.data(), option_values.data(), &link_state));

    auto report_error = [&](CUresult result)
    {
        if(result == CUDA_SUCCESS)
            return;
        const char *result_str;
        cuGetErrorName(result, &result_str);
        throw std::runtime_error(result_str + std::string(". ") + err_buffer.data());
    };

    for(auto &ptx : impl_->ptx_data)
    {
        std::vector<char> data(
            ptx.data(), ptx.data() + ptx.size() + 1);

        const auto result = cuLinkAddData(
            link_state, CU_JIT_INPUT_PTX, data.data(),
            ptx.size(), nullptr, 0, nullptr, nullptr);

        report_error(result);
    }

    void *cubin = nullptr; size_t cubin_size = 0;
    auto result = cuLinkComplete(link_state, &cubin, &cubin_size);
    report_error(result);

    result = cuModuleLoadDataEx(
        &impl_->cu_module, cubin, 2, options.data(), option_values.data());
    report_error(result);
}

void CUDAModule::launch_impl(
    const std::string &entry_name,
    const Dim3        &block_cnt,
    const Dim3        &block_size,
    void             **kernel_args)
{
    if(!impl_->cu_module)
        link();
    assert(impl_->cu_module);

    CUfunction entry_func = nullptr;
    throw_on_error(cuModuleGetFunction(
        &entry_func, impl_->cu_module, entry_name.c_str()));
    assert(entry_func);

    throw_on_error(cuLaunchKernel(
        entry_func,
        block_cnt.x, block_cnt.y, block_cnt.z,
        block_size.x, block_size.y, block_size.z,
        0, nullptr, kernel_args, nullptr));
}

BTRC_CORE_END
