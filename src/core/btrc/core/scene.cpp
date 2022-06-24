#include <algorithm>

#include <btrc/core/camera.h>
#include <btrc/core/scene.h>
#include <btrc/utils/optix/device_funcs.h>

BTRC_BEGIN

SurfacePoint get_hitinfo(
    const CVec3f        &o,
    const CVec3f        &d,
    const CInstanceInfo &instance,
    const CGeometryInfo &geometry,
    f32                  t,
    u32                  prim_id,
    const CVec2f        &uv)
{
    ref local_to_world = instance.transform;

    // position

    var position = o + t * d;

    // geometry frame

    var gx_ua = load_aligned(geometry.geometry_ex_tex_coord_u_a + prim_id);
    var gy_uba = load_aligned(geometry.geometry_ey_tex_coord_u_ba + prim_id);
    var gz_uca = load_aligned(geometry.geometry_ez_tex_coord_u_ca + prim_id);

    var sn_v_a = load_aligned(geometry.shading_normal_tex_coord_v_a + prim_id);
    var sn_v_ba = load_aligned(geometry.shading_normal_tex_coord_v_ba + prim_id);
    var sn_v_ca = load_aligned(geometry.shading_normal_tex_coord_v_ca + prim_id);

    CFrame geometry_frame = CFrame(gx_ua.xyz(), gy_uba.xyz(), gz_uca.xyz());

    geometry_frame.x = local_to_world.apply_to_vector(geometry_frame.x);
    geometry_frame.y = local_to_world.apply_to_vector(geometry_frame.y);
    geometry_frame.z = local_to_world.apply_to_normal(geometry_frame.z);
    geometry_frame.x = normalize(geometry_frame.x);
    geometry_frame.y = normalize(geometry_frame.y);
    geometry_frame.z = normalize(geometry_frame.z);

    // interpolated normal

    var interp_normal = sn_v_a.xyz() + sn_v_ba.xyz() * uv.x + sn_v_ca.xyz() * uv.y;
    interp_normal = normalize(local_to_world.apply_to_normal(interp_normal));

    // tex coord

    var tex_coord_u = gx_ua.w + gy_uba.w * uv.x + gz_uca.w * uv.y;
    var tex_coord_v = sn_v_a.w + sn_v_ba.w * uv.x + sn_v_ca.w * uv.y;
    var tex_coord = CVec2f(tex_coord_u, tex_coord_v);

    // intersection

    SurfacePoint material_inct;
    material_inct.position = position;
    material_inct.frame = geometry_frame;
    material_inct.interp_z = interp_normal;
    material_inct.uv = uv;
    material_inct.tex_coord = tex_coord;

    return material_inct;
}

Scene::Scene(optix::Context &optix_ctx)
    : optix_ctx_(&optix_ctx)
{
    vol_prim_medium_ = newRC<VolumePrimitiveMedium>();
}

void Scene::add_instance(const Instance &inst)
{
    instances_.push_back(inst);
}

void Scene::add_volume(RC<VolumePrimitive> vol)
{
    vol_prim_medium_->add_volume(std::move(vol));
}

void Scene::set_envir_light(RC<EnvirLight> env)
{
    env_light_ = std::move(env);
}

void Scene::set_light_sampler(RC<LightSampler> light_sampler)
{
    light_sampler_ = std::move(light_sampler);
}

void Scene::commit()
{
    light_sampler_->clear();
    for(auto &inst : instances_)
    {
        if(inst.light)
            light_sampler_->add_light(inst.light);
    }
    if(env_light_)
        light_sampler_->add_light(env_light_);

    tlas_ = {};
    materials_ = {};
    mediums_ = {};

    std::vector<optix::Context::Instance> blas_instances;

    std::vector<InstanceInfo> instance_info;
    std::vector<GeometryInfo> geometry_info;

    std::map<RC<Geometry>, int> geometry_indices;
    {
        for(auto &inst : instances_)
            geometry_indices.insert({ inst.geometry, 0 });
        int i = 0;
        for(auto &p : geometry_indices)
        {
            p.second = i++;
            geometry_info.push_back(p.first->get_geometry_info());
        }
    }

    std::map<RC<Material>, int> material_indices;
    {
        for(auto &inst : instances_)
            material_indices.insert({ inst.material, 0 });
        int i = 0;
        for(auto &p : material_indices)
        {
            p.second = i++;
            materials_.push_back(p.first);
        }
    }

    std::map<RC<Medium>, MediumID> medium_indices;
    {
        for(auto &inst : instances_)
        {
            medium_indices[inst.inner_medium] = 0;
            medium_indices[inst.outer_medium] = 0;
        }
        std::vector<RC<Medium>> ids = { vol_prim_medium_ };
        for(auto &p : medium_indices)
        {
            if(p.first)
                ids.push_back(p.first);
        }
        std::sort(ids.begin(), ids.end(), [](const auto &a, const auto &b)
        {
            return a->get_priority() > b->get_priority();
        });
        int i = 0;
        for(auto &med : ids)
        {
            medium_indices[med] = i++;
            mediums_.push_back(med);
        }
        assert(ids.back() == vol_prim_medium_);
        medium_indices[nullptr] = static_cast<MediumID>(medium_indices.size() - 1);
    }

    int next_light_idx = 0;

    for(auto &inst : instances_)
    {
        // geometry id

        const int geo_id = geometry_indices.at(inst.geometry);

        // material id

        const int mat_id = material_indices.at(inst.material);

        // medium id

        const MediumID inner_med_id = medium_indices.at(inst.inner_medium);
        const MediumID outer_med_id = medium_indices.at(inst.outer_medium);

        // light id

        int light_idx = -1;
        if(inst.light)
            light_idx = next_light_idx++;

        instance_info.push_back(InstanceInfo{
            .geometry_id     = geo_id,
            .material_id     = mat_id,
            .light_id        = light_idx,
            .transform       = inst.transform,
            .inner_medium_id = inner_med_id,
            .outer_medium_id = outer_med_id,
            .flag            = inst.flag
        });

        blas_instances.push_back(optix::Context::Instance{
            .local_to_world = std::array
            {
                inst.transform.mat.at(0, 0), inst.transform.mat.at(0, 1), inst.transform.mat.at(0, 2), inst.transform.mat.at(0, 3),
                inst.transform.mat.at(1, 0), inst.transform.mat.at(1, 1), inst.transform.mat.at(1, 2), inst.transform.mat.at(1, 3),
                inst.transform.mat.at(2, 0), inst.transform.mat.at(2, 1), inst.transform.mat.at(2, 2), inst.transform.mat.at(2, 3)
            },
            .id     = static_cast<uint32_t>(blas_instances.size()),
            .mask   = optix::RAY_MASK_ALL,
            .handle = inst.geometry->get_blas()
        });
    }

    tlas_ = optix_ctx_->create_instance_as(blas_instances);

    if(!instance_info.empty())
    {
        device_instance_info_.initialize(instance_info.size());
        device_instance_info_.from_cpu(instance_info.data());
        host_instance_info_ = std::move(instance_info);
    }

    if(!geometry_info.empty())
    {
        device_geometry_info_.initialize(geometry_info.size());
        device_geometry_info_.from_cpu(geometry_info.data());
        host_geometry_info_ = std::move(geometry_info);
    }

    bbox_ = {};
    for(auto &inst : instances_)
    {
        auto inst_bbox = inst.geometry->get_bounding_box();
        inst_bbox = inst.transform.apply_to_aabb(inst_bbox);
        bbox_ = union_aabb(bbox_, inst_bbox);
    }

    for(auto &vol : vol_prim_medium_->get_prims())
    {
        auto vol_bbox = vol->get_bounding_box();
        bbox_ = union_aabb(bbox_, vol_bbox);
    }
}

std::vector<RC<Object>> Scene::get_dependent_objects()
{
    std::vector<RC<Object>> output;
    for(auto &inst : instances_)
    {
        output.push_back(inst.geometry);
        output.push_back(inst.material);
        if(inst.light)
            output.push_back(inst.light);
        if(inst.inner_medium)
            output.push_back(inst.inner_medium);
        if(inst.outer_medium)
            output.push_back(inst.outer_medium);
    }
    if(env_light_)
        output.push_back(env_light_);
    output.push_back(vol_prim_medium_);
    return output;
}

OptixTraversableHandle Scene::get_tlas() const
{
    return tlas_;
}

int Scene::get_instance_count() const
{
    return static_cast<int>(instances_.size());
}

int Scene::get_geometry_count() const
{
    return static_cast<int>(host_geometry_info_.size());
}

const GeometryInfo *Scene::get_host_geometry_info() const
{
    return host_geometry_info_.data();
}

const InstanceInfo *Scene::get_host_instance_info() const
{
    return host_instance_info_.data();
}

const GeometryInfo *Scene::get_device_geometry_info() const
{
    return device_geometry_info_;
}

const InstanceInfo *Scene::get_device_instance_info() const
{
    return device_instance_info_;
}

const LightSampler *Scene::get_light_sampler() const
{
    return light_sampler_.get();
}

int Scene::get_material_count() const
{
    return static_cast<int>(materials_.size());
}

const Material *Scene::get_material(int id) const
{
    return materials_[id].get();
}

int Scene::get_medium_count() const
{
    return static_cast<int>(mediums_.size());
}

const Medium *Scene::get_medium(int id) const
{
    return mediums_[id].get();
}

MediumID Scene::get_volume_primitive_medium_id() const
{
    assert(mediums_.back() == vol_prim_medium_);
    return static_cast<MediumID>(mediums_.size() - 1);
}

const Medium *Scene::get_volume_primitive_medium() const
{
    return vol_prim_medium_.get();
}

bool Scene::has_motion_blur() const
{
    return false;
}

bool Scene::is_triangle_only() const
{
    return true;
}

const AABB3f &Scene::get_bbox() const
{
    return bbox_;
}

bool Scene::has_medium() const
{
    if(!vol_prim_medium_->get_prims().empty())
        return true;
    for(auto &inst : instances_)
    {
        if(inst.inner_medium || inst.outer_medium)
            return true;
    }
    return false;
}

void Scene::access_material(i32 idx, const std::function<void(const Material *)> &func) const
{
    $switch(idx)
    {
        for(int i = 0; i < get_material_count(); ++i)
        {
            $case(i)
            {
                func(get_material(i));
            };
        }
        $default
        {
            cstd::unreachable();
        };
    };
}

void Scene::access_medium(i32 idx, const std::function<void(const Medium *)> &func) const
{
    $switch(idx)
    {
        for(int i = 0; i < get_medium_count(); ++i)
        {
            $case(i)
            {
                func(get_medium(i));
            };
        }
        $default
        {
            cstd::unreachable();
        };
    };
}

BTRC_END
