#include <filesystem>
#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <imgui.h>
#include <portable-file-dialogs.h>

#include <btrc/builtin/register.h>
#include <btrc/core/object_dag.h>
#include <btrc/core/scene.h>
#include <btrc/factory/context.h>
#include <btrc/factory/node/parser.h>
#include <btrc/factory/post_processor.h>
#include <btrc/factory/scene.h>
#include <btrc/utils/cuda/context.h>
#include <btrc/utils/optix/context.h>
#include <btrc/utils/exception.h>
#include <btrc/utils/file.h>

#include "camera_controller.h"
#include "message_box.h"
#include "reporter.h"
#include "window.h"

using namespace btrc;
using namespace gui;

struct BtrcScene
{
    Box<factory::Context> object_context;

    int width  = 0;
    int height = 0;

    RC<factory::Node> root;

    RC<Scene>                      scene;
    RC<Camera>                     camera;
    RC<Renderer>                   renderer;
    std::vector<RC<PostProcessor>> post_processors;
};

struct Rect2D
{
    Vec2f lower;
    Vec2f upper;
};

void prepare_scene(const RC<Renderer> &renderer, const RC<Scene> &scene)
{
    ObjectDAG dag(renderer);

    scene->precommit();
    dag.commit();
    scene->postcommit();

    renderer->recompile();
}

bool select_config_filename(std::string &output_filename)
{
    auto open = pfd::open_file(
        "Select Scene Configuration FIle",
        std::filesystem::current_path().string(),
        { ".json" });
    const auto result = open.result();
    if(result.size() != 1)
        return false;
    output_filename = result.at(0);
    return true;
}

BtrcScene initialize_btrc_scene(const std::string &filename, optix::Context &optix_context)
{
    BtrcScene result;

    std::cout << "create object context" << std::endl;

    result.object_context = newBox<factory::Context>(optix_context);
    builtin::register_builtin_creators(*result.object_context);

    const auto scene_dir = std::filesystem::path(filename).parent_path();
    result.object_context->add_path_mapping("scene_directory", scene_dir.string());

    std::cout << "parse scene" << std::endl;

    factory::JSONParser parser;
    parser.set_source(read_txt_file(filename));
    parser.add_include_directory(scene_dir);
    parser.parse();
    result.root = parser.get_result();

    std::cout << "create scene" << std::endl;

    result.scene = create_scene(result.root->child_node("scene"), *result.object_context);

    std::cout << "create camera" << std::endl;

    result.width = result.root->parse_child<int>("width");
    result.height = result.root->parse_child<int>("height");

    result.camera = result.object_context->create<Camera>(result.root->child_node("camera"));
    result.camera->set_w_over_h(static_cast<float>(result.width) / result.height);

    std::cout << "create renderer" << std::endl;

    result.renderer = result.object_context->create<Renderer>(result.root->child_node("renderer"));
    result.renderer->set_camera(result.camera);
    result.renderer->set_film(result.width, result.height);
    result.renderer->set_scene(result.scene);

    std::cout << "create post processors" << std::endl;

    result.post_processors = parse_post_processors(
        result.root->find_child_node("post_processors"), *result.object_context);

    std::cout << "compile kernel" << std::endl;

    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    prepare_scene(result.renderer, result.scene);
    const auto compile_time = Clock::now() - start;

    std::cout << "compile time: " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(compile_time).count() << "ms" << std::endl;

    return result;
}

Rect2D compute_preview_image_rect(const Vec2f &window_size, const Vec2f &scene_size)
{
    if(window_size.x < 50 || window_size.y < 50)
        return Rect2D{ { 0, 0 }, { 0, 0 } };

    const float avail_width  = (std::max)(window_size.x * 0.9f, window_size.x - 50);
    const float avail_height = (std::max)(window_size.y * 0.9f, window_size.y - 50);

    const float window_ratio = avail_width / avail_height;
    const float image_ratio = static_cast<float>(scene_size.x) / scene_size.y;

    float display_size_x, display_size_y;
    if(window_ratio > image_ratio)
    {
        display_size_y = avail_height;
        display_size_x = image_ratio * display_size_y;
    }
    else
    {
        display_size_x = avail_width;
        display_size_y = display_size_x / image_ratio;
    }

    Rect2D result;
    result.lower = {
            0.5f * (window_size.x - display_size_x),
            0.5f * (window_size.y - display_size_y)
    };
    result.upper = result.lower + Vec2f(display_size_x, display_size_y);
    return result;
}

void display_image(GLuint tex_handle, const Rect2D &rect)
{
    if(rect.lower.x == rect.upper.x || rect.lower.y == rect.upper.y)
        return;
    const auto im_tex = reinterpret_cast<ImTextureID>(static_cast<size_t>(tex_handle));
    ImGui::SetCursorPos({ rect.lower.x, rect.lower.y });
    ImGui::Image(im_tex, ImVec2({ rect.upper.x - rect.lower.x, rect.upper.y - rect.lower.y }));
}

void execute_post_processors(
    Renderer::RenderResult &result,
    const std::vector<RC<PostProcessor>> &post_processors,
    int width, int height)
{
    for(auto &p : post_processors)
        p->process(result.color, result.albedo, result.normal, width, height);
}

// returns next config filename
std::string run(Window &window, const std::string &config_filename)
{
    cuda::Context cuda_context(0);
    optix::Context optix_context(nullptr);

    auto scene = initialize_btrc_scene(config_filename, optix_context);

    auto reporter = newRC<GUIPreviewer>();
    reporter->set_preview_interval(0);
    reporter->set_fast_preview(true);
    reporter->set_post_processors(scene.post_processors);

    GLuint tex_handle = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &tex_handle);
    if(!tex_handle)
        throw std::runtime_error("failed to create gl texture");
    glTextureStorage2D(tex_handle, 1, GL_RGBA32F, scene.width, scene.height);

    CameraController camera_controller(std::dynamic_pointer_cast<builtin::PinholeCamera>(scene.camera));
    CameraController::ControlParams controller_params;

    constexpr int MIN_UPDATED_IMAGE_COUNT = 2;
    int updated_image_count = 0;

    auto update_image = [&](const Image<Vec4f> &image)
    {
        if(image && ++updated_image_count >= MIN_UPDATED_IMAGE_COUNT)
        {
            glTextureSubImage2D(
                tex_handle, 0, 0, 0, scene.width, scene.height,
                GL_RGBA, GL_FLOAT, image.data());
        }
    };

    scene.renderer->set_reporter(reporter);
    scene.renderer->render_async();

    using Clock = std::chrono::steady_clock;
    auto start_render_time = Clock::now();
    bool print_render_time = false;

    auto restart_render = [&]
    {
        reporter->progress(0);
        reporter->set_preview_interval(0);
        reporter->set_fast_preview(true);

        updated_image_count = 0;
        scene.renderer->render_async();
        start_render_time = Clock::now();
        print_render_time = false;
    };

    bool exit_render_session = false;
    std::string next_config_name;

    while(!window.should_close() && !exit_render_session)
    {
        window.begin_frame();

        if(ImGui::IsKeyDown(ImGuiKey_Escape))
            window.set_close(true);

        if(scene.renderer->is_rendering())
            reporter->access_dirty_image(update_image);
        else
        {
            updated_image_count = MIN_UPDATED_IMAGE_COUNT;
            reporter->access_image(update_image);
        }

        bool open_still_rendering_window = false;
        if(ImGui::BeginMainMenuBar())
        {
            if(ImGui::MenuItem("[Execute Post Processors]"))
            {
                if(!scene.renderer->is_rendering())
                {
                    if(scene.renderer->is_waitable())
                    {
                        auto result = scene.renderer->wait_async();
                        execute_post_processors(
                            result, scene.post_processors, scene.width, scene.height);
                    }
                }
                else
                    open_still_rendering_window = true;
            }

            if(ImGui::MenuItem("[Load New]"))
            {
                if(select_config_filename(next_config_name))
                    exit_render_session = true;
            }

            if(ImGui::MenuItem("[Reload]"))
            {
                next_config_name = config_filename;
                exit_render_session = true;
            }

            if(ImGui::BeginMenu("[Camera]"))
            {
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));
                ImGui::SliderFloat("RotateAxisX Speed", &controller_params.rotate_speed_hori, 0.1f, 3.0f);
                ImGui::SliderFloat("RotateAxisY Speed", &controller_params.rotate_speed_vert, 0.1f, 3.0f);
                ImGui::SliderFloat("Distance Speed", &controller_params.dist_adjust_speed, 0.05f, 0.3f);
                ImGui::PopStyleColor();
                ImGui::EndMenu();
            }
        }
        ImGui::EndMainMenuBar();

        if(open_still_rendering_window)
            ImGui::EnableMessageBox("StillRenderingWindow");
        ImGui::DisplayMessageBox("StillRenderingWindow", "rendering is not done");

        const auto imgui_viewport = ImGui::GetMainViewport();
        Rect2D display_rect = compute_preview_image_rect(
            { imgui_viewport->WorkSize.x, imgui_viewport->WorkSize.y },
            { static_cast<float>(scene.width), static_cast<float>(scene.height) });
        if(updated_image_count >= MIN_UPDATED_IMAGE_COUNT)
        {
            auto mouse_pos = ImGui::GetMousePos();
            mouse_pos.x -= imgui_viewport->WorkPos.x;
            mouse_pos.y -= imgui_viewport->WorkPos.y;

            const auto relative_mouse_pos = Vec2f(
                (mouse_pos.x - display_rect.lower.x) / (display_rect.upper.x - display_rect.lower.x),
                (mouse_pos.y - display_rect.lower.y) / (display_rect.upper.y - display_rect.lower.y));

            const bool update = camera_controller.update(CameraController::InputParams{
                .cursor_pos = relative_mouse_pos,
                .wheel_offset = ImGui::GetIO().MouseWheel,
                .button_down = {
                    ImGui::IsMouseDown(ImGuiMouseButton_Left),
                    ImGui::IsMouseDown(ImGuiMouseButton_Middle),
                    ImGui::IsMouseDown(ImGuiMouseButton_Right)
                }
            }, controller_params);

            if(update)
            {
                if(scene.renderer->is_waitable())
                    scene.renderer->stop_async();
                scene.camera->commit();
                restart_render();
            }
            else if(updated_image_count == MIN_UPDATED_IMAGE_COUNT)
            {
                reporter->set_preview_interval(100);
                reporter->set_fast_preview(false);
            }
        }

        if(reporter->get_percentage() > 10 || Clock::now() - start_render_time > std::chrono::seconds(3))
            reporter->set_preview_interval(1000);

        {
            auto viewport = ImGui::GetMainViewport();

            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

            ImGui::SetNextWindowPos(viewport->WorkPos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(viewport->WorkSize, ImGuiCond_Always);

            constexpr ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_NoTitleBar |
                ImGuiWindowFlags_NoCollapse |
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoDocking |
                ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoNavFocus;
            ImGui::Begin("display", nullptr, window_flags);
            ImGui::PopStyleVar(3);
            {
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.1f, 0.8f, 0.1f, 1));
                ImGui::ProgressBar(reporter->get_percentage() / 100.0f);
                ImGui::PopStyleColor();
                display_image(tex_handle, display_rect);
            }
            ImGui::End();
        }

        if(!print_render_time && reporter->get_percentage() >= 100)
        {
            auto end_render_time = Clock::now();
            auto render_time = end_render_time - start_render_time;
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(render_time);
            std::cout << "render time: " << ms.count() / 1000.0f << "s" << std::endl;
            print_render_time = true;
        }

        window.end_frame();
    }

    glDeleteTextures(1, &tex_handle);

    if(scene.renderer->is_waitable())
        scene.renderer->stop_async();

    return next_config_name;
}

int main(int argc, char *argv[])
{
    std::cout << ">>> Btrc Renderer <<<" << std::endl;

    if(argc > 2)
    {
        std::cout << "usage: BtrcGUI (optional config.json)" << std::endl;
        return 0;
    }

    std::string filename;
    if(argc == 2)
        filename = argv[1];
    else if(!select_config_filename(filename))
        return 0;

    try
    {
        Window window("BtrcGUI", 1024, 768);
        while(!filename.empty())
            filename = run(window, filename);
    }
    catch(const std::exception &err)
    {
        std::vector<std::string> err_msgs;
        extract_hierarchy_exceptions(err, std::back_inserter(err_msgs));
        for(auto &s : err_msgs)
            std::cerr << s << std::endl;
        return -1;
    }
}
