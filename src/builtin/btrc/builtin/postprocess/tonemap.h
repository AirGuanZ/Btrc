#pragma once

#include <btrc/core/post_processor.h>
#include <btrc/factory/context.h>
#include <btrc/utils/cuda/module.h>

BTRC_BUILTIN_BEGIN

class ACESToneMap : public PostProcessor
{
public:

    ACESToneMap();

    void set_exposure(float exposure);

    ExecutionPolicy get_execution_policy() const override;

    void process(
        Vec4f *color,
        Vec4f *albedo,
        Vec4f *normal,
        int    width,
        int    height) override;

private:

    float exposure_ = 1;
    cuda::Module module_;
};

class ACESToneMapCreator : public factory::Creator<PostProcessor>
{
public:

    std::string get_name() const override { return "tonemap"; };

    RC<PostProcessor> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
