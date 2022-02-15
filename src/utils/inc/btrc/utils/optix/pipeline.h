#pragma once

#include <btrc/utils/optix/sbt.h>

BTRC_OPTIX_BEGIN

class SimpleOptixPipeline : public Uncopyable
{
public:

    struct Program
    {
        std::string ptx;
        std::string launch_params_name;
        std::string raygen_name;
        std::string miss_name;
        std::string closesthit_name;
    };

    struct Config
    {
        int  payload_count;
        int  traversable_depth;
        bool motion_blur;
        bool triangle_only;
    };

    SimpleOptixPipeline() = default;

    SimpleOptixPipeline(
        OptixDeviceContext context,
        const Program &prog,
        const Config &config);

    SimpleOptixPipeline(SimpleOptixPipeline &&other) noexcept;

    SimpleOptixPipeline &operator=(SimpleOptixPipeline &&other) noexcept;

    ~SimpleOptixPipeline();

    void swap(SimpleOptixPipeline &other) noexcept;

    operator bool() const;

    operator OptixPipeline() const;

    const OptixShaderBindingTable &get_sbt() const;

private:

    OptixModule   module_   = nullptr;
    OptixPipeline pipeline_ = nullptr;

    OptixProgramGroup raygen_group_ = nullptr;
    OptixProgramGroup miss_group_   = nullptr;
    OptixProgramGroup hit_group_    = nullptr;

    SBT sbt_;
};

BTRC_OPTIX_END
