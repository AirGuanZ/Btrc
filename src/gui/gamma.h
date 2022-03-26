#pragma once

#include <btrc/core/post_processor.h>
#include <btrc/factory/context.h>
#include <btrc/utils/cuda/module.h>

class Gamma : public btrc::PostProcessor
{
public:

    Gamma();

    ExecutionPolicy get_execution_policy() const override;

    void process(
        btrc::Vec4f *color,
        btrc::Vec4f *albedo,
        btrc::Vec4f *normal,
        int          width,
        int          height) override;

private:

    btrc::cuda::Module module_;
};
