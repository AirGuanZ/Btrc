#include <chrono>
#include <iostream>

#include <btrc/builtin/register.h>
#include <btrc/builtin/reporter/console.h>
#include <btrc/core/scene.h>
#include <btrc/factory/context.h>
#include <btrc/factory/node/parser.h>
#include <btrc/factory/scene.h>
#include <btrc/utils/cuda/context.h>
#include <btrc/utils/optix/context.h>
#include <btrc/utils/exception.h>
#include <btrc/utils/file.h>

// commit
// recompile
// update
// render

/*
 * scene.preprocess()
 * commit()
 * recompile()
 * update_parameters()
 * render()
 */

void process_object(
    btrc::RC<btrc::Object>               object,
    std::set<btrc::RC<btrc::Object>>    &processed_objects,
    std::vector<btrc::RC<btrc::Object>> &output)
{
    if(processed_objects.contains(object))
        return;
    for(auto &d : object->get_dependent_objects())
        process_object(d, processed_objects, output);
    assert(!processed_objects.contains(object));
    processed_objects.insert(object);
    output.push_back(object);
}

std::vector<btrc::RC<btrc::Object>> topology_sort_objects(const std::set<btrc::RC<btrc::Object>> &entry_objects)
{
    using namespace btrc;
    std::set<RC<Object>> processed_objects;
    std::vector<RC<Object>> output;
    for(auto &e : entry_objects)
        process_object(e, processed_objects, output);
    return output;
}

void run(const std::string &scene_filename)
{
    using namespace btrc;

    const auto scene_dir = std::filesystem::path(scene_filename).parent_path();

    cuda::Context cuda_context(0);
    optix::Context optix_context(cuda_context);

    PropertyPool::initialize_instance();
    BTRC_SCOPE_EXIT{ PropertyPool::destroy_instance(); };

    factory::Context btrc_context(optix_context);
    builtin::register_builtin_creators(btrc_context);
    btrc_context.add_path_mapping("scene_directory", scene_dir.string());

    factory::JSONParser parser;
    std::string json_source = read_txt_file(scene_filename);
    parser.set_source(std::move(json_source));
    parser.add_include_directory(scene_dir);
    parser.parse();
    auto root_node = parser.get_result();

    auto scene_node = root_node->child_node("scene");
    auto scene = create_scene(scene_node, btrc_context);

    const int width = root_node->parse_child<int>("width");
    const int height = root_node->parse_child<int>("height");

    auto camera = btrc_context.create<Camera>(root_node->child_node("camera"));
    camera->set_w_over_h(static_cast<float>(width) / height);

    auto renderer = btrc_context.create<Renderer>(root_node->child_node("renderer"));
    renderer->set_camera(camera);
    renderer->set_film(width, height);
    renderer->set_scene(scene);
    renderer->set_reporter(newRC<builtin::ConsoleReporter>());

    auto entry_objects = renderer->get_dependent_objects();
    auto sorted_objects = topology_sort_objects(std::set(entry_objects.begin(), entry_objects.end()));

    std::cout << "commit objects..." << std::endl;

    scene->precommit();
    for(auto &obj : sorted_objects)
        obj->commit();
    scene->postcommit();

    std::cout << "compile kernel..." << std::endl;

    renderer->recompile();

    std::cout << "render..." << std::endl;

    auto result = renderer->render();

    result.value.save(root_node->parse_child_or<std::string>("value_filename", "output.exr"));
    if(result.albedo)
        result.albedo.save(root_node->parse_child_or<std::string>("albedo_filename", "output_albedo.png"));
    if(result.normal)
        result.normal.save(root_node->parse_child_or<std::string>("normal_filename", "output_normal.png"));
}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        std::cout << "usage: BtrcCLI config.json" << std::endl;
        return 0;
    }

    try
    {
        run(argv[1]);
    }
    catch(const std::exception &err)
    {
        std::vector<std::string> err_msgs;
        btrc::extract_hierarchy_exceptions(err, std::back_inserter(err_msgs));
        for(auto &s : err_msgs)
            std::cerr << s << std::endl;
        return -1;
    }
}
