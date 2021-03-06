#pragma once

#include <btrc/core/light.h>
#include <btrc/core/light_sampler.h>
#include <btrc/core/geometry.h>
#include <btrc/core/material.h>
#include <btrc/core/medium.h>
#include <btrc/core/volume.h>
#include <btrc/utils/optix/as.h>

BTRC_BEGIN

class Scene;

using InstanceFlag = uint32_t;
using CInstanceFlag = cuj::cxx<InstanceFlag>;

constexpr InstanceFlag INSTANCE_FLAG_CAUSTICS_PROJECTOR = 1u << 0;
constexpr InstanceFlag INSTANCE_FLAG_CAUSTICS_RECRIVER  = 1u << 1;

struct InstanceInfo
{
    int32_t      geometry_id = 0;
    int32_t      material_id = 0;
    int32_t      light_id = 0;
    Transform3D  transform;
    MediumID     inner_medium_id = 0;
    MediumID     outer_medium_id = 0;
    InstanceFlag flag = 0;
};

CUJ_PROXY_CLASS(
    CInstanceInfo,
    InstanceInfo,
    geometry_id,
    material_id,
    light_id,
    transform,
    inner_medium_id,
    outer_medium_id,
    flag);

SurfacePoint get_hitinfo(
    const CVec3f        &o,
    const CVec3f        &d,
    const CInstanceInfo &instance,
    const CGeometryInfo &geometry,
    f32                  t,
    u32                  prim_id,
    const CVec2f        &uv);

class Scene : public Object
{
public:

    struct Instance
    {
        RC<Geometry>  geometry;
        RC<Material>  material;
        RC<AreaLight> light;
        Transform3D   transform;
        RC<Medium>    inner_medium;
        RC<Medium>    outer_medium;
        InstanceFlag  flag;
    };

    explicit Scene(optix::Context &optix_ctx);

    Scene(const Scene &other) noexcept = delete;

    Scene &operator=(const Scene &other) noexcept = delete;

    Scene(Scene &&other) noexcept = default;

    Scene &operator=(Scene &&other) noexcept = default;

    void add_instance(const Instance &inst);

    void add_volume(RC<VolumePrimitive> vol);

    void set_envir_light(RC<EnvirLight> env);

    void set_light_sampler(RC<LightSampler> light_sampler);

    void commit() override;

    std::vector<RC<Object>> get_dependent_objects() override;

    OptixTraversableHandle get_tlas() const;

    int get_geometry_count() const;

    const GeometryInfo *get_host_geometry_info() const;

    const GeometryInfo *get_device_geometry_info() const;

    int get_instance_count() const;

    const InstanceInfo *get_host_instance_info() const;

    const InstanceInfo *get_device_instance_info() const;

    const LightSampler *get_light_sampler() const;

    int get_material_count() const;

    const Material *get_material(int id) const;

    int get_medium_count() const;

    const Medium *get_medium(int id) const;

    MediumID get_volume_primitive_medium_id() const;

    const Medium *get_volume_primitive_medium() const;

    bool has_motion_blur() const;

    bool is_triangle_only() const;

    const AABB3f &get_bbox() const;

    bool has_medium() const;

    void access_material(i32 idx, const std::function<void(const Material *)> &func) const;

    void access_medium(i32 idx, const std::function<void(const Medium *)> &func) const;

private:

    optix::Context *optix_ctx_;

    std::vector<Instance> instances_;
    RC<EnvirLight>        env_light_;

    RC<VolumePrimitiveMedium> vol_prim_medium_;

    optix::InstanceAS          tlas_;
    std::vector<RC<Material>>  materials_;
    std::vector<RC<Medium>>    mediums_;
    RC<LightSampler>           light_sampler_;
    std::vector<InstanceInfo>  host_instance_info_;
    cuda::Buffer<InstanceInfo> device_instance_info_;
    std::vector<GeometryInfo>  host_geometry_info_;
    cuda::Buffer<GeometryInfo> device_geometry_info_;

    AABB3f bbox_;
};

BTRC_END
