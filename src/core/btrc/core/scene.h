#pragma once

#include <btrc/core/light.h>
#include <btrc/core/light_sampler.h>
#include <btrc/core/geometry.h>
#include <btrc/core/material.h>
#include <btrc/core/medium.h>
#include <btrc/utils/optix/as.h>

BTRC_BEGIN

struct InstanceInfo
{
    int32_t   geometry_id = 0;
    int32_t   material_id = 0;
    int32_t   light_id = 0;
    Transform transform;
    MediumID  inner_medium_id = MEDIUM_ID_VOID;
    MediumID  outer_medium_id = MEDIUM_ID_VOID;
};

CUJ_PROXY_CLASS(
    CInstanceInfo,
    InstanceInfo,
    geometry_id,
    material_id,
    light_id,
    transform,
    inner_medium_id,
    outer_medium_id);

class Scene
{
public:

    struct Instance
    {
        RC<Geometry>  geometry;
        RC<Material>  material;
        RC<AreaLight> light;
        Transform     transform;
        RC<Medium>    inner_medium;
        RC<Medium>    outer_medium;
    };

    explicit Scene(optix::Context &optix_ctx);

    Scene(const Scene &other) noexcept = delete;

    Scene &operator=(const Scene &other) noexcept = delete;

    Scene(Scene &&other) noexcept = default;

    Scene &operator=(Scene &&other) noexcept = default;

    void add_instance(const Instance &inst);

    void set_envir_light(RC<EnvirLight> env);

    void set_light_sampler(RC<LightSampler> light_sampler);

    void precommit();

    void postcommit();

    void clear_device_data();
    
    OptixTraversableHandle get_tlas() const;

    void collect_objects(std::set<RC<Object>> &output) const;

    const GeometryInfo *get_device_geometry_info() const;

    const InstanceInfo *get_device_instance_info() const;

    const LightSampler *get_light_sampler() const;

    int get_material_count() const;

    const Material *get_material(int id) const;

    int get_medium_count() const;

    const Medium *get_medium(int id) const;

    bool has_motion_blur() const;

    bool is_triangle_only() const;

private:

    optix::Context *optix_ctx_;

    std::vector<Instance> instances_;
    RC<EnvirLight>        env_light_;

    optix::InstanceAS          tlas_;
    std::vector<RC<Material>>  materials_;
    std::vector<RC<Medium>>    mediums_;
    RC<LightSampler>           light_sampler_;
    cuda::Buffer<InstanceInfo> instance_info_;
    cuda::Buffer<GeometryInfo> geometry_info_;
};

BTRC_END
