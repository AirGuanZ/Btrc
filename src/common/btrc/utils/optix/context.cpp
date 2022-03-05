#include <cassert>
#include <iostream>
#include <mutex>

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <btrc/utils/cuda/error.h>
#include <btrc/utils/optix/context.h>
#include <btrc/utils/scope_guard.h>

BTRC_OPTIX_BEGIN

Context::Context()
    : context_(nullptr)
{
    
}

Context::Context(CUcontext cuda_context)
    : context_(nullptr)
{
    if(cuda_context)
        throw_on_error(cuCtxGetCurrent(&cuda_context));

    static std::once_flag init_optix_flag;
    std::call_once(init_optix_flag, []
    {
        throw_on_error(optixInit());
    });

    OptixDeviceContextOptions options;
    options.logCallbackFunction = &log_callback;
    options.logCallbackData = this;
    options.logCallbackLevel = 4;
#if BTRC_IS_DEBUG
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif

    throw_on_error(optixDeviceContextCreate(cuda_context, &options, &context_));
    throw_on_error(optixDeviceContextSetCacheEnabled(context_, false));

    message_callback_ = [](unsigned int level, const char *tag, const char *msg)
    {
        if(level <= 3)
            std::cerr << fmt::format("[{}][{:12}]: {}", level, tag, msg);
    };
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
        optixDeviceContextDestroy(context_);
}

void Context::swap(Context &other) noexcept
{
    std::swap(context_, other.context_);
}

Context::operator bool() const
{
    return context_ != nullptr;
}

Context::operator OptixDeviceContext_t*() const
{
    return context_;
}

void Context::set_message_callback(MessageCallback callback)
{
    message_callback_ = std::move(callback);
}

TriangleAS Context::create_triangle_as(
    std::span<const Vec3f>   vertices,
    std::span<const int16_t> indices)
{
    return create_triangle_as_impl(vertices, indices);
}

TriangleAS Context::create_triangle_as(
    std::span<const Vec3f>   vertices,
    std::span<const int32_t> indices)
{
    return create_triangle_as_impl(vertices, indices);
}

InstanceAS Context::create_instance_as(std::span<const Instance> instances)
{
    // upload instance data

    std::vector<OptixInstance> optix_instances;
    optix_instances.reserve(instances.size());
    for(auto &i : instances)
    {
        optix_instances.push_back(OptixInstance{
            .instanceId        = i.id,
            .sbtOffset         = 0,
            .visibilityMask    = i.mask,
            .flags             = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT,
            .traversableHandle = i.handle
        });
        std::memcpy(
            optix_instances.back().transform,
            i.local_to_world.data(),
            sizeof(i.local_to_world));
    }

    OptixInstance *device_instances = nullptr;
    throw_on_error(cudaMalloc(
        &device_instances, sizeof(OptixInstance) * optix_instances.size()));
    BTRC_SCOPE_EXIT{ cudaFree(device_instances); };
    throw_on_error(cudaMemcpy(
        device_instances, optix_instances.data(),
        sizeof(OptixInstance) * optix_instances.size(), cudaMemcpyHostToDevice));

    // build input

    OptixBuildInput build_input = {
        .type          = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        .instanceArray = OptixBuildInputInstanceArray{
            .instances    = reinterpret_cast<CUdeviceptr>(device_instances),
            .numInstances = static_cast<unsigned int>(optix_instances.size())
        }
    };

    auto [handle, as_buffer] = build_accel(build_input);
    return InstanceAS(handle, std::move(as_buffer));
}

void Context::log_callback(
    unsigned level, const char *tag, const char *msg, void *data)
{
    assert(data);
    Context *context = static_cast<Context *>(data);
    if(context->message_callback_)
        context->message_callback_(level, tag, msg);
}

std::pair<OptixTraversableHandle, cuda::Buffer<>>
    Context::build_accel(const OptixBuildInput &build_input)
{
    // memory usage

    OptixAccelBuildOptions built_options = {
        .buildFlags    = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                         OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
        .operation     = OPTIX_BUILD_OPERATION_BUILD,
        .motionOptions = {}
    };

    OptixAccelBufferSizes uncompacted_sizes;
    throw_on_error(optixAccelComputeMemoryUsage(
        context_, &built_options, &build_input, 1, &uncompacted_sizes));
    const size_t uncompacted_size = uncompacted_sizes.outputSizeInBytes;

    // non-compacted buffers

    const size_t compacted_size_offset = up_align(uncompacted_size, 8);

    cuda::Buffer as_temp_buffer(uncompacted_sizes.tempSizeInBytes);
    cuda::Buffer as_buffer(compacted_size_offset + sizeof(size_t));

    // build

    OptixTraversableHandle handle = 0;
    OptixAccelEmitDesc emit_desc = {
        .result = reinterpret_cast<CUdeviceptr>(
            as_buffer.get() + compacted_size_offset),
        .type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE
    };
    throw_on_error(optixAccelBuild(
        context_, nullptr, &built_options, &build_input, 1,
        as_temp_buffer,
        uncompacted_sizes.tempSizeInBytes,
        as_buffer,
        uncompacted_size, &handle, &emit_desc, 1));

    // compact

    size_t compacted_size;
    throw_on_error(cudaMemcpy(
        &compacted_size,
        as_buffer.get() + compacted_size_offset,
        sizeof(size_t), cudaMemcpyDeviceToHost));
    if(compacted_size < uncompacted_size)
    {
        cuda::Buffer compacted_as_buffer(compacted_size);
        throw_on_error(optixAccelCompact(
            context_, nullptr, handle,
            compacted_as_buffer, compacted_size, &handle));
        cudaStreamSynchronize(nullptr);
        as_buffer.swap(compacted_as_buffer);
    }

    return { handle, std::move(as_buffer) };
}

template<typename Index>
TriangleAS Context::create_triangle_as_impl(
    std::span<const Vec3f> vertices,
    std::span<const Index> indices)
{
    // device vertex buffer

    Vec3f *device_vertices = nullptr;
    throw_on_error(cudaMalloc(
        &device_vertices, sizeof(Vec3f) * vertices.size()));
    BTRC_SCOPE_EXIT{ cudaFree(device_vertices); };
    throw_on_error(cudaMemcpy(
        device_vertices, vertices.data(),
        sizeof(Vec3f) * vertices.size(), cudaMemcpyHostToDevice));

    // device index buffer

    Index *device_indices = nullptr;
    if(!indices.empty())
    {
        assert(indices.size() % 3 == 0);
        throw_on_error(cudaMalloc(
            &device_indices, sizeof(Index) * indices.size()));
    }
    BTRC_SCOPE_EXIT{ cudaFree(device_indices); };
    if(device_indices)
    {
        assert(device_indices);
        throw_on_error(cudaMemcpy(
            device_indices, indices.data(),
            sizeof(Index) * indices.size(), cudaMemcpyHostToDevice));
    }

    // build input

    OptixIndicesFormat index_format;
    unsigned int index_stride;
    if(!device_indices)
    {
        index_format = OPTIX_INDICES_FORMAT_NONE;
        index_stride = 0;
    }
    else if constexpr(std::is_same_v<Index, int16_t>)
    {
        index_format = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
        index_stride = sizeof(int16_t) * 3;
    }
    else
    {
        static_assert(std::is_same_v<Index, int32_t>);
        index_format = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        index_stride = sizeof(int32_t) * 3;
    }
    unsigned int triangle_flags[1] = {
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    };
    auto cu_device_vertices = reinterpret_cast<CUdeviceptr>(device_vertices);
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers               = &cu_device_vertices;
    build_input.triangleArray.numVertices                 = static_cast<unsigned int>(vertices.size());
    build_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes         = sizeof(Vec3f);
    build_input.triangleArray.indexBuffer                 = reinterpret_cast<CUdeviceptr>(device_indices);
    build_input.triangleArray.numIndexTriplets            = static_cast<unsigned int>(indices.size()) / 3;
    build_input.triangleArray.indexFormat                 = index_format;
    build_input.triangleArray.indexStrideInBytes          = index_stride;
    build_input.triangleArray.preTransform                = 0;
    build_input.triangleArray.flags                       = triangle_flags;
    build_input.triangleArray.numSbtRecords               = 1;
    build_input.triangleArray.sbtIndexOffsetBuffer        = 0;
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;
    build_input.triangleArray.primitiveIndexOffset        = 0;
    build_input.triangleArray.transformFormat             = OPTIX_TRANSFORM_FORMAT_NONE;

    auto [handle, as_buffer] = build_accel(build_input);
    return TriangleAS(handle, std::move(as_buffer));
}

BTRC_OPTIX_END
