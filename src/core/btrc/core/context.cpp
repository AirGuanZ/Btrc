#include <cuda_runtime.h>

#include <btrc/core/context.h>
#include <btrc/utils/cuda/error.h>

BTRC_BEGIN

namespace
{

    Box<PropertyPool> pool_instance;

} // namespace anonymous

PropertyPool::~PropertyPool()
{
    for(auto c : chunks_)
        cudaFree(c);
}

void PropertyPool::initialize_instance()
{
    assert(!pool_instance);
    auto instance = new PropertyPool;
    pool_instance.reset(instance);
}

void PropertyPool::destroy_instance()
{
    pool_instance.reset();
}

PropertyPool &PropertyPool::get_instance()
{
    return *pool_instance;
}

void PropertyPool::new_chunk(std::vector<void *> &output, size_t size, size_t align)
{
    assert(size % align == 0);
    if(align > 256)
    {
        throw BtrcException(std::format(
            "Property: alignment value {} is too large!", align));
    }

    unsigned char *chunk = nullptr;
    constexpr size_t CHUNK_ELEM_COUNT = 256;
    throw_on_error(cudaMalloc(&chunk, size * CHUNK_ELEM_COUNT));

    for(size_t i = 0; i < CHUNK_ELEM_COUNT; ++i)
        output.push_back(chunk + i * size);
}

void *PropertyPool::allocate_impl(std::type_index type_index, size_t size, size_t align)
{
    auto &properties = free_properties_[type_index];
    if(properties.empty())
        new_chunk(properties, size, align);
    assert(!properties.empty());
    auto device_pointer = properties.back();
    properties.pop_back();
    return device_pointer;
}

void PropertyPool::release_impl(std::type_index type_index, void *device_pointer)
{
    free_properties_[type_index].push_back(device_pointer);
}

ScopedPropertyPool::ScopedPropertyPool()
{
    PropertyPool::initialize_instance();
}

ScopedPropertyPool::~ScopedPropertyPool()
{
    PropertyPool::destroy_instance();
}

BTRC_END
