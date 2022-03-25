#include <btrc/factory/scene.h>

BTRC_FACTORY_BEGIN

RC<Scene> create_scene(const RC<const Node> &scene_root, Context &context)
{
    auto result = newRC<Scene>(context.get_optix_context());

    auto entity_array = scene_root->child_node("entities")->as_array();
    if(!entity_array)
        throw BtrcException("entity array expected");
    for(size_t i = 0; i < entity_array->get_size(); ++i)
    {
        auto entity = entity_array->get_element(i)->as_group();
        if(!entity)
            throw BtrcException("entity group expected");

        auto geometry = context.create<Geometry>(entity->child_node("geometry"));
        auto material = context.create<Material>(entity->child_node("material"));
        auto transform = entity->parse_child_or("local_to_world", Transform3D{});

        RC<Medium> inner_medium;
        if(auto node = entity->find_child_node("inner_medium"))
            inner_medium = context.create<Medium>(node);

        RC<Medium> outer_medium;
        if(auto node = entity->find_child_node("outer_medium"))
            outer_medium = context.create<Medium>(node);

        RC<AreaLight> area_light;
        if(auto light_node = entity->find_child_node("light"))
        {
            auto light = context.create<Light>(light_node);
            area_light = std::dynamic_pointer_cast<AreaLight>(light);
            if(!area_light)
                throw BtrcException("area light expected");
            area_light->set_geometry(geometry, transform);
        }

        result->add_instance(Scene::Instance{
            .geometry     = std::move(geometry),
            .material     = std::move(material),
            .light        = std::move(area_light),
            .transform    = transform,
            .inner_medium = std::move(inner_medium),
            .outer_medium = std::move(outer_medium)
        });
    }

    if(auto volarr_node = scene_root->find_child_node("volumes"))
    {
        auto volume_array = volarr_node->as_array();
        if(!volume_array)
            throw BtrcException("'volumes' is not an array node");

        for(size_t i = 0; i < volume_array->get_size(); ++i)
        {
            auto volume = volume_array->get_element(i);
            auto o = volume->parse_child<Vec3f>("o");
            auto x = volume->parse_child<Vec3f>("x");
            auto y = volume->parse_child<Vec3f>("y");
            auto z = volume->parse_child<Vec3f>("z");
            auto sigma_t = context.create<Texture3D>(volume->child_node("sigma_t"));
            auto albedo = context.create<Texture3D>(volume->child_node("albedo"));

            auto vol = newRC<VolumePrimitive>();
            vol->set_geometry(o, x, y, z);
            vol->set_sigma_t(std::move(sigma_t));
            vol->set_albedo(std::move(albedo));

            result->add_volume(std::move(vol));
        }
    }

    if(auto env_node = scene_root->find_child_node("envir_light"))
    {
        auto light = context.create<Light>(env_node);
        auto env_light = std::dynamic_pointer_cast<EnvirLight>(light);
        if(!env_light)
            throw BtrcException("'envir_light' is not an environment light");
        result->set_envir_light(std::move(env_light));
    }

    result->set_light_sampler(newRC<UniformLightSampler>());

    return result;
}

BTRC_FACTORY_END
