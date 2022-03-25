#include <btrc/builtin/denoise/optix_denoiser.h>
#include <btrc/utils/cuda/module.h>

BTRC_BUILTIN_BEGIN

struct OptixAIDenoiser::Impl
{
    ~Impl()
    {
        if(denoiser)
            optixDenoiserDestroy(denoiser);
    }

    OptixDeviceContext context = nullptr;
    OptixDenoiser denoiser = nullptr;

    int width = 0;
    int height = 0;

    cuda::Buffer<Vec4f> output;

    cuda::Buffer<float> intensity;
    cuda::Buffer<float> scratch_buffer;
    cuda::Buffer<float> state_buffer;
};

OptixAIDenoiser::OptixAIDenoiser(
    optix::Context &context,
    bool            albedo,
    bool            normal,
    int             width,
    int             height)
{
    impl_ = new Impl;
    BTRC_SCOPE_FAIL{ delete impl_; impl_ = nullptr; };
    impl_->context = context;

    const OptixDenoiserOptions options = {
        .guideAlbedo = albedo ? 1u : 0u,
        .guideNormal = normal ? 1u : 0u
    };
    throw_on_error(optixDenoiserCreate(
        context, OPTIX_DENOISER_MODEL_KIND_HDR, &options, &impl_->denoiser));

    impl_->width = width;
    impl_->height = height;
    impl_->output.initialize(width * height);
    impl_->output.clear({ 0, 0, 0, 1 });

    OptixDenoiserSizes sizes;
    throw_on_error(optixDenoiserComputeMemoryResources(
        impl_->denoiser, width, height, &sizes));

    impl_->intensity.initialize(1);
    impl_->scratch_buffer.initialize(sizes.withoutOverlapScratchSizeInBytes);
    impl_->state_buffer.initialize(sizes.stateSizeInBytes);

    throw_on_error(optixDenoiserSetup(
        impl_->denoiser, nullptr, width, height,
        impl_->state_buffer.get_device_ptr(),
        impl_->state_buffer.get_size_in_bytes(),
        impl_->scratch_buffer.get_device_ptr(),
        impl_->scratch_buffer.get_size_in_bytes()));
}

OptixAIDenoiser::OptixAIDenoiser(OptixAIDenoiser &&other) noexcept
    : OptixAIDenoiser()
{
    swap(other);
}

OptixAIDenoiser &OptixAIDenoiser::operator=(OptixAIDenoiser &&other) noexcept
{
    swap(other);
    return *this;
}

OptixAIDenoiser::~OptixAIDenoiser()
{
    delete impl_;
}

void OptixAIDenoiser::swap(OptixAIDenoiser &other) noexcept
{
    std::swap(impl_, other.impl_);
}

void OptixAIDenoiser::denoise(const Vec4f *color, const Vec4f *albedo, const Vec4f *normal) const
{
    OptixDenoiserLayer layer{};
    layer.input = OptixImage2D{
        .data = reinterpret_cast<CUdeviceptr>(color),
        .width = static_cast<unsigned>(impl_->width),
        .height = static_cast<unsigned>(impl_->height),
        .rowStrideInBytes = static_cast<unsigned>(sizeof(Vec4f) * impl_->width),
        .pixelStrideInBytes = sizeof(Vec4f),
        .format = OPTIX_PIXEL_FORMAT_FLOAT4
    };
    layer.output = OptixImage2D{
        .data = impl_->output.get_device_ptr(),
        .width = static_cast<unsigned>(impl_->width),
        .height = static_cast<unsigned>(impl_->height),
        .rowStrideInBytes = static_cast<unsigned>(sizeof(Vec4f) * impl_->width),
        .pixelStrideInBytes = sizeof(Vec4f),
        .format = OPTIX_PIXEL_FORMAT_FLOAT4
    };

    OptixDenoiserGuideLayer guide_layer{};
    guide_layer.albedo = OptixImage2D{
        .data = reinterpret_cast<CUdeviceptr>(albedo),
        .width = static_cast<unsigned>(impl_->width),
        .height = static_cast<unsigned>(impl_->height),
        .rowStrideInBytes = static_cast<unsigned>(sizeof(Vec4f) * impl_->width),
        .pixelStrideInBytes = sizeof(Vec4f),
        .format = OPTIX_PIXEL_FORMAT_FLOAT4
    };
    guide_layer.normal = OptixImage2D{
        .data = reinterpret_cast<CUdeviceptr>(normal),
        .width = static_cast<unsigned>(impl_->width),
        .height = static_cast<unsigned>(impl_->height),
        .rowStrideInBytes = static_cast<unsigned>(sizeof(Vec4f) * impl_->width),
        .pixelStrideInBytes = sizeof(Vec4f),
        .format = OPTIX_PIXEL_FORMAT_FLOAT4
    };

    throw_on_error(optixDenoiserComputeIntensity(
        impl_->denoiser, nullptr, &layer.input,
        impl_->intensity.get_device_ptr(),
        impl_->scratch_buffer.get_device_ptr(),
        impl_->scratch_buffer.get_size_in_bytes()));

    const OptixDenoiserParams params = {
        .denoiseAlpha = false,
        .hdrIntensity = impl_->intensity.get_device_ptr(),
        .blendFactor = 0,
        .hdrAverageColor = 0
    };
    throw_on_error(optixDenoiserInvoke(
        impl_->denoiser, nullptr, &params,
        impl_->state_buffer.get_device_ptr(),
        impl_->state_buffer.get_size_in_bytes(),
        &guide_layer, &layer, 1, 0, 0,
        impl_->scratch_buffer.get_device_ptr(),
        impl_->scratch_buffer.get_size_in_bytes()));
}

const Vec4f *OptixAIDenoiser::get_output_buffer() const
{
    return impl_->output.get();
}

BTRC_BUILTIN_END
