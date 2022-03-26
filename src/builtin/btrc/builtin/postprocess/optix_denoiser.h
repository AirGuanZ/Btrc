#pragma once

#include <btrc/core/post_processor.h>
#include <btrc/factory/context.h>
#include <btrc/utils/optix/context.h>

BTRC_BUILTIN_BEGIN

class OptixAIDenoiser : public PostProcessor
{
public:

    explicit OptixAIDenoiser(optix::Context &context);

    ~OptixAIDenoiser();

    ExecutionPolicy get_execution_policy() const override;

    void process(Vec4f *color, Vec4f *albedo, Vec4f *normal, int width, int height) override;

private:

    void setup(bool albedo, bool normal, int width, int height);

    struct Impl;

    Impl *impl_ = nullptr;
};

class OptixAIDenoiserCreator : public factory::Creator<PostProcessor>
{
public:

    std::string get_name() const override { return "denoise"; }

    RC<PostProcessor> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
