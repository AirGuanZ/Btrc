#pragma once

#include <btrc/core/post_processor.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class SaveToImage : public PostProcessor
{
public:

    void set_gamma(float value);

    void set_color_filename(std::string filename);

    void set_albedo_filename(std::string filename);

    void set_normal_filename(std::string filename);

    ExecutionPolicy get_execution_policy() const override;

    void process(Vec4f *color, Vec4f *albedo, Vec4f *normal, int width, int height) override;

private:

    float gamma_ = 1.0f;
    std::string color_filename_;
    std::string albedo_filename_;
    std::string normal_filename_;
};

class SaveToImageCreator : public factory::Creator<PostProcessor>
{
public:

    std::string get_name() const override { return "save"; }

    RC<PostProcessor> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
