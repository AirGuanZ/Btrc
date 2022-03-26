#include <btrc/builtin/postprocess/save_to_image.h>

BTRC_BUILTIN_BEGIN

void SaveToImage::set_gamma(float value)
{
    gamma_ = value;
}

void SaveToImage::set_color_filename(std::string filename)
{
    color_filename_ = std::move(filename);
}

void SaveToImage::set_albedo_filename(std::string filename)
{
    albedo_filename_ = std::move(filename);
}

void SaveToImage::set_normal_filename(std::string filename)
{
    normal_filename_ = std::move(filename);
}

PostProcessor::ExecutionPolicy SaveToImage::get_execution_policy() const
{
    return ExecutionPolicy::AfterComplete;
}

void SaveToImage::process(Vec4f *color, Vec4f *albedo, Vec4f *normal, int width, int height)
{
    const int texel_count = width * height;
    Image<Vec4f> image(width, height);

    if(color && !color_filename_.empty())
    {
        throw_on_error(cudaMemcpy(
            image.data(), color,
            sizeof(Vec4f) * texel_count,
            cudaMemcpyDeviceToHost));
        image.pow_(1 / gamma_);
        image.save(color_filename_);
    }

    if(albedo && !albedo_filename_.empty())
    {
        throw_on_error(cudaMemcpy(
            image.data(), albedo,
            sizeof(Vec4f) * texel_count,
            cudaMemcpyDeviceToHost));
        image.save(albedo_filename_);
    }

    if(normal && !normal_filename_.empty())
    {
        throw_on_error(cudaMemcpy(
            image.data(), albedo,
            sizeof(Vec4f) * texel_count,
            cudaMemcpyDeviceToHost));
        image.save(normal_filename_);
    }
}

RC<PostProcessor> SaveToImageCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    const float gamma = node->parse_child_or("gamma", 1.0f);
    std::string color = context.resolve_path(node->parse_child_or("color", std::string())).string();
    std::string albedo = context.resolve_path(node->parse_child_or("albedo", std::string())).string();
    std::string normal = context.resolve_path(node->parse_child_or("normal", std::string())).string();
    auto ret = newRC<SaveToImage>();
    ret->set_gamma(gamma);
    ret->set_color_filename(std::move(color));
    ret->set_albedo_filename(std::move(albedo));
    ret->set_normal_filename(std::move(normal));
    return ret;
}

BTRC_BUILTIN_END
