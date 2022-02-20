#include <btrc/core/scene.h>
#include <btrc/utils/optix/device_funcs.h>

BTRC_BEGIN

void Scene::add_instance(const Instance &inst)
{
    instances_.push_back(inst);
}

void Scene::set_envir_light(RC<const EnvirLight> env)
{
    env_light_ = std::move(env);
}

void Scene::set_light_sampler(RC<LightSampler> light_sampler)
{
    light_sampler_ = std::move(light_sampler);
}

void Scene::preprocess(optix::Context &optix_ctx)
{
    std::vector<optix::Context::Instance> blas_instances;

    std::vector<InstanceInfo>          instance_info;
    std::vector<GeometryInfo>          geometry_info;
    std::map<RC<const Geometry>, int>  geometry_info_indices;
    std::map<RC<const Material>, int>  material_indices;

    int next_light_idx = 0;
    light_sampler_->clear();

    for(auto &inst : instances_)
    {
        // geometry id

        auto geo_it = geometry_info_indices.find(inst.geometry);
        if(geo_it == geometry_info_indices.end())
        {
            const int id = static_cast<int>(geometry_info.size());
            geo_it = geometry_info_indices.insert({ inst.geometry, id }).first;
            geometry_info.push_back(inst.geometry->get_geometry_info());
        }
        const int geo_id = geo_it->second;

        // material id

        auto mat_it = material_indices.find(inst.material);
        if(mat_it == material_indices.end())
        {
            const int id = static_cast<int>(materials_.size());
            mat_it = material_indices.insert({ inst.material, id }).first;
            materials_.push_back(inst.material);
        }
        const int mat_id = mat_it->second;

        // light id

        int light_idx = -1;
        if(inst.light)
        {
            light_sampler_->add_light(inst.light);
            light_idx = next_light_idx++;
        }

        instance_info.push_back(InstanceInfo{
            .geometry_id = geo_id,
            .material_id = mat_id,
            .light_id    = light_idx,
            .transform   = inst.transform
        });

        blas_instances.push_back(optix::Context::Instance{
            .local_to_world = inst.transform.to_transform_matrix(),
            .id             = static_cast<uint32_t>(blas_instances.size()),
            .mask           = optix::RAY_MASK_ALL,
            .handle         = inst.geometry->get_blas()
        });
    }

    if(env_light_)
        light_sampler_->add_light(env_light_);

    std::vector<int32_t> instance_to_material(instance_info.size());
    for(size_t i = 0; i < instances_.size(); ++i)
        instance_to_material[i] = instance_info[i].material_id;

    tlas_                 = optix_ctx.create_instance_as(blas_instances);
    instance_info_        = cuda::CUDABuffer<InstanceInfo>(instance_info);
    geometry_info_        = cuda::CUDABuffer<GeometryInfo>(geometry_info);
    instance_to_material_ = cuda::CUDABuffer<int32_t>(instance_to_material);
}

OptixTraversableHandle Scene::get_tlas() const
{
    return tlas_;
}

const GeometryInfo *Scene::get_device_geometry_info() const
{
    return geometry_info_;
}

const InstanceInfo *Scene::get_device_instance_info() const
{
    return instance_info_;
}

const int32_t *Scene::get_device_instance_to_material() const
{
    return instance_to_material_;
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

bool Scene::has_motion_blur() const
{
    return false;
}

bool Scene::is_triangle_only() const
{
    return true;
}

BTRC_END
