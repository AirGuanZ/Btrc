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
        auto transform = entity->parse_child_or("local_to_world", Transform{});

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
            .geometry = std::move(geometry),
            .material = std::move(material),
            .light = std::move(area_light),
            .transform = transform
        });
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
