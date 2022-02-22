#pragma once

#include <btrc/core/camera.h>
#include <btrc/core/light.h>
#include <btrc/core/light_sampler.h>
#include <btrc/core/geometry.h>
#include <btrc/core/material.h>
#include <btrc/utils/optix/as.h>

BTRC_BEGIN

struct InstanceInfo
{
    int32_t   geometry_id = 0;
    int32_t   material_id = 0;
    int32_t   light_id = 0;
    Transform transform;
};

CUJ_PROXY_CLASS(
    CInstanceInfo,
    InstanceInfo,
    geometry_id,
    material_id,
    light_id,
    transform);

class Scene
{
public:

    struct Instance
    {
        RC<Geometry>  geometry;
        RC<Material>  material;
        RC<AreaLight> light;
        Transform     transform;
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
    
    OptixTraversableHandle get_tlas() const;

    void collect_objects(std::set<RC<Object>> &output) const;

    const GeometryInfo *get_device_geometry_info() const;

    const InstanceInfo *get_device_instance_info() const;

    const int32_t *get_device_instance_to_material() const;

    const LightSampler *get_light_sampler() const;

    int get_material_count() const;

    const Material *get_material(int id) const;

    bool has_motion_blur() const;

    bool is_triangle_only() const;

private:

    optix::Context *optix_ctx_;

    std::vector<Instance> instances_;
    RC<EnvirLight>        env_light_;

    optix::InstanceAS              tlas_;
    std::vector<RC<Material>>      materials_;
    RC<LightSampler>               light_sampler_;
    cuda::CUDABuffer<InstanceInfo> instance_info_;
    cuda::CUDABuffer<GeometryInfo> geometry_info_;
    cuda::CUDABuffer<int32_t>      instance_to_material_;
};

BTRC_END
