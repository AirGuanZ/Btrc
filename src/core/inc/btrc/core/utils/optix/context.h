#pragma once

#include <array>
#include <functional>
#include <span>

#include <btrc/core/utils/math/math.h>
#include <btrc/core/utils/optix/as.h>
#include <btrc/core/utils/uncopyable.h>

BTRC_OPTIX_BEGIN

class Context : public Uncopyable
{
public:

    struct Instance
    {
        std::array<float, 12>  local_to_world;
        uint32_t               id;
        uint32_t               mask;
        OptixTraversableHandle handle;
    };

    using MessageCallback = std::function<
        void(unsigned int, const char *, const char *)>;

    explicit Context(CUcontext cuda_context = nullptr);

    Context(Context &&other) noexcept;

    Context &operator=(Context &&other) noexcept;

    ~Context();

    void swap(Context &other) noexcept;

    operator bool() const;

    operator OptixDeviceContext() const;

    void set_message_callback(MessageCallback callback);

    TriangleAS create_triangle_as(
        std::span<const Vec3f>   vertices,
        std::span<const int16_t> indices = {});
    
    TriangleAS create_triangle_as(
        std::span<const Vec3f>   vertices,
        std::span<const int32_t> indices);

    InstanceAS create_instance_as(std::span<const Instance> instances);

private:

    static void log_callback(
        unsigned int level, const char *tag, const char *msg, void *data);

    std::pair<OptixTraversableHandle, CUDABuffer<>>
        build_accel(const OptixBuildInput &build_input);

    template<typename Index>
    TriangleAS create_triangle_as_impl(
        std::span<const Vec3f> vertices,
        std::span<const Index> indices);

    OptixDeviceContext context_;
    MessageCallback    message_callback_;
};

BTRC_OPTIX_END
