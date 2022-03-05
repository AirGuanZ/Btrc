#include <algorithm>

#include <btrc/core/scene.h>
#include <btrc/utils/optix/device_funcs.h>

BTRC_BEGIN

Scene::Scene(optix::Context &optix_ctx)
    : optix_ctx_(&optix_ctx)
{

}

void Scene::add_instance(const Instance &inst)
{
    instances_.push_back(inst);
}

void Scene::set_envir_light(RC<EnvirLight> env)
{
    env_light_ = std::move(env);
}

void Scene::set_light_sampler(RC<LightSampler> light_sampler)
{
    light_sampler_ = std::move(light_sampler);
}

void Scene::precommit()
{
    light_sampler_->clear();
    for(auto &inst : instances_)
    {
        if(inst.light)
            light_sampler_->add_light(inst.light);
    }
    if(env_light_)
        light_sampler_->add_light(env_light_);
}

void Scene::postcommit()
{
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
            medium_indices[inst.inner_medium] = MEDIUM_ID_VOID;
            medium_indices[inst.outer_medium] = MEDIUM_ID_VOID;
        }
        std::vector<RC<Medium>> ids;
        for(auto &p : medium_indices)
        {
            if(p.first)
                ids.push_back(p.first);
        }
        std::sort(ids.begin(), ids.end(), [](const auto &a, const auto &b)
        {
            return a->get_priority() < b->get_priority();
        });
        int i = 0;
        for(auto &med : ids)
        {
            medium_indices.at(med) = i++;
            mediums_.push_back(med);
        }
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
            .outer_medium_id = outer_med_id
        });

        blas_instances.push_back(optix::Context::Instance{
            .local_to_world = inst.transform.to_transform_matrix(),
            .id             = static_cast<uint32_t>(blas_instances.size()),
            .mask           = optix::RAY_MASK_ALL,
            .handle         = inst.geometry->get_blas()
        });
    }

    tlas_ = optix_ctx_->create_instance_as(blas_instances);

    if(!instance_info_)
        instance_info_.initialize(instance_info.size());
    assert(instance_info_.get_size() == instance_info.size());
    instance_info_.from_cpu(instance_info.data());

    if(!geometry_info_)
        geometry_info_.initialize(geometry_info.size());
    assert(geometry_info_.get_size() == geometry_info.size());
    geometry_info_.from_cpu(geometry_info.data());
}

void Scene::clear_device_data()
{
    instance_info_ = {};
    geometry_info_ = {};
}

OptixTraversableHandle Scene::get_tlas() const
{
    return tlas_;
}

void Scene::collect_objects(std::set<RC<Object>> &output) const
{
    for(auto &inst : instances_)
    {
        output.insert(inst.geometry);
        output.insert(inst.material);
        if(inst.light)
            output.insert(inst.light);
        if(inst.inner_medium)
            output.insert(inst.inner_medium);
        if(inst.outer_medium)
            output.insert(inst.outer_medium);
    }
    if(env_light_)
        output.insert(env_light_);
}

const GeometryInfo *Scene::get_device_geometry_info() const
{
    return geometry_info_;
}

const InstanceInfo *Scene::get_device_instance_info() const
{
    return instance_info_;
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

bool Scene::has_motion_blur() const
{
    return false;
}

bool Scene::is_triangle_only() const
{
    return true;
}

BTRC_END
