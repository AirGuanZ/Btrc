#include <optix_denoiser_tiling.h>

#include <btrc/builtin/postprocess/optix_denoiser.h>
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

    bool albedo = false;
    bool normal = false;

    int width = 0;
    int height = 0;
    int tile_width = 0;
    int tile_height = 0;
    int overlap_size = 0;

    cuda::Buffer<Vec4f> output;

    cuda::Buffer<float> intensity;
    cuda::Buffer<float> scratch_buffer;
    cuda::Buffer<float> state_buffer;
};

OptixAIDenoiser::OptixAIDenoiser(optix::Context &context)
{
    impl_ = new Impl;
    impl_->context = context;
}

OptixAIDenoiser::~OptixAIDenoiser()
{
    delete impl_;
}

PostProcessor::ExecutionPolicy OptixAIDenoiser::get_execution_policy() const
{
    return ExecutionPolicy::Always;
}

void OptixAIDenoiser::process(Vec4f *color, Vec4f *albedo, Vec4f *normal, int width, int height)
{
    setup(albedo != nullptr, normal != nullptr, width, height);

    OptixDenoiserLayer layer{};
    layer.input = OptixImage2D{
        .data = reinterpret_cast<CUdeviceptr>(color),
        .width = static_cast<unsigned>(impl_->width),
        .height = static_cast<unsigned>(impl_->height),
        .rowStrideInBytes = static_cast<unsigned>(sizeof(Vec4f) * impl_->width),
        .pixelStrideInBytes = sizeof(Vec4f),
        .format = OPTIX_PIXEL_FORMAT_FLOAT3
    };
    layer.output = OptixImage2D{
        .data = impl_->output.get_device_ptr(),
        .width = static_cast<unsigned>(impl_->width),
        .height = static_cast<unsigned>(impl_->height),
        .rowStrideInBytes = static_cast<unsigned>(sizeof(Vec4f) * impl_->width),
        .pixelStrideInBytes = sizeof(Vec4f),
        .format = OPTIX_PIXEL_FORMAT_FLOAT3
    };

    OptixDenoiserGuideLayer guide_layer{};

    if(albedo)
    {
        guide_layer.albedo = OptixImage2D{
            .data = reinterpret_cast<CUdeviceptr>(albedo),
            .width = static_cast<unsigned>(impl_->width),
            .height = static_cast<unsigned>(impl_->height),
            .rowStrideInBytes = static_cast<unsigned>(sizeof(Vec4f) * impl_->width),
            .pixelStrideInBytes = sizeof(Vec4f),
            .format = OPTIX_PIXEL_FORMAT_FLOAT3
        };
    }

    if(normal)
    {
        guide_layer.normal = OptixImage2D{
            .data = reinterpret_cast<CUdeviceptr>(normal),
            .width = static_cast<unsigned>(impl_->width),
            .height = static_cast<unsigned>(impl_->height),
            .rowStrideInBytes = static_cast<unsigned>(sizeof(Vec4f) * impl_->width),
            .pixelStrideInBytes = sizeof(Vec4f),
            .format = OPTIX_PIXEL_FORMAT_FLOAT3
        };
    }

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

    throw_on_error(optixUtilDenoiserInvokeTiled(
        impl_->denoiser, nullptr, &params,
        impl_->state_buffer.get_device_ptr(),
        impl_->state_buffer.get_size_in_bytes(),
        &guide_layer, &layer, 1,
        impl_->scratch_buffer.get_device_ptr(),
        impl_->scratch_buffer.get_size_in_bytes(),
        impl_->overlap_size,
        impl_->tile_width, impl_->tile_width));

    throw_on_error(cudaMemcpy(
        color, impl_->output.get(),
        sizeof(Vec4f) * width * height,
        cudaMemcpyDeviceToDevice));
}

void OptixAIDenoiser::setup(bool albedo, bool normal, int width, int height)
{
    if(impl_->width >= width &&
       impl_->height >= height &&
       impl_->albedo == albedo && 
       impl_->normal == normal)
        return;

    if(impl_->denoiser)
    {
        throw_on_error(optixDenoiserDestroy(impl_->denoiser));
        impl_->denoiser = nullptr;
    }
    
    const OptixDenoiserOptions options = {
        .guideAlbedo = albedo ? 1u : 0u,
        .guideNormal = normal ? 1u : 0u
    };

    impl_->albedo = albedo;
    impl_->normal = normal;

    throw_on_error(optixDenoiserCreate(
        impl_->context, OPTIX_DENOISER_MODEL_KIND_HDR, &options, &impl_->denoiser));

    impl_->width = width;
    impl_->height = height;
    impl_->output.initialize(width * height);
    impl_->output.clear({ 0, 0, 0, 1 });

    const int tile_width = width > 1024 ? 1024 : 0;
    const int tile_height = height > 1024 ? 1024 : 0;

    impl_->tile_width = tile_width > 0 ? tile_width : width;
    impl_->tile_height = tile_height > 0 ? tile_height : height;

    if(impl_->intensity.is_empty())
        impl_->intensity.initialize(1);

    OptixDenoiserSizes sizes;
    {
        throw_on_error(optixDenoiserComputeMemoryResources(
            impl_->denoiser, impl_->tile_width, impl_->tile_height, &sizes));
    }

    if(tile_width == 0)
    {
        impl_->scratch_buffer.initialize(sizes.withoutOverlapScratchSizeInBytes);
        impl_->overlap_size = 0;
    }
    else
    {
        impl_->scratch_buffer.initialize(sizes.withOverlapScratchSizeInBytes);
        impl_->overlap_size = static_cast<int>(sizes.overlapWindowSizeInPixels);
    }
    impl_->state_buffer.initialize(sizes.stateSizeInBytes);

    throw_on_error(optixDenoiserSetup(
        impl_->denoiser, nullptr,
        impl_->tile_width + 2 * impl_->overlap_size,
        impl_->tile_height + 2 * impl_->overlap_size,
        impl_->state_buffer.get_device_ptr(),
        impl_->state_buffer.get_size_in_bytes(),
        impl_->scratch_buffer.get_device_ptr(),
        impl_->scratch_buffer.get_size_in_bytes()));
}

RC<PostProcessor> OptixAIDenoiserCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    return newRC<OptixAIDenoiser>(context.get_optix_context());
}

BTRC_BUILTIN_END
