#include <iostream>
#include <vector>

#ifdef WIN32
#include <Windows.h>
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <btrc/builtin/register.h>
#include <btrc/builtin/reporter/console.h>
#include <btrc/core/scene.h>
#include <btrc/core/traversal.h>
#include <btrc/factory/context.h>
#include <btrc/factory/node/parser.h>
#include <btrc/factory/scene.h>
#include <btrc/utils/cuda/context.h>
#include <btrc/utils/optix/context.h>
#include <btrc/utils/exception.h>
#include <btrc/utils/file.h>

#include "reporter.h"

using namespace btrc;

struct OpenGLContext
{
    GLFWwindow *window;
};

struct BtrcScene
{
    Box<cuda::Context>      cuda_context;
    Box<optix::Context>     optix_context;
    Box<ScopedPropertyPool> property_pool;
    Box<factory::Context>   object_context;

    int width  = 0;
    int height = 0;

    RC<Scene>    scene;
    RC<Camera>   camera;
    RC<Reporter> reporter;
    RC<Renderer> renderer;
};

OpenGLContext initialize_opengl()
{
    if(!glfwInit())
        throw std::runtime_error("failed to initialize glfw");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    auto glfw_window = glfwCreateWindow(640, 480, "BtrcGUI", nullptr, nullptr);
    if(!glfw_window)
        throw std::runtime_error("failed to create glfw window");

    glfwMakeContextCurrent(glfw_window);

    if(glewInit() != GLEW_OK)
        throw std::runtime_error("failed to iniailize glew");

    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    ImGui::GetIO().IniFilename = nullptr;

    ImGui_ImplGlfw_InitForOpenGL(glfw_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    ImGui::StyleColorsLight();

    return { glfw_window };
}

void destroy_opengl(OpenGLContext &context)
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(context.window);
    glfwTerminate();
}

BtrcScene initialize_btrc_scene(const std::string &filename)
{
    BtrcScene result;

    result.cuda_context = newBox<cuda::Context>(0);
    result.optix_context = newBox<optix::Context>(*result.cuda_context);
    result.property_pool = newBox<ScopedPropertyPool>();

    result.object_context = newBox<factory::Context>(*result.optix_context);
    builtin::register_builtin_creators(*result.object_context);

    const auto scene_dir = std::filesystem::path(filename).parent_path();
    result.object_context->add_path_mapping("scene_directory", scene_dir.string());

    factory::JSONParser parser;
    parser.set_source(read_txt_file(filename));
    parser.add_include_directory(scene_dir);
    parser.parse();
    auto root = parser.get_result();

    result.scene = create_scene(root->child_node("scene"), *result.object_context);

    result.width = root->parse_child<int>("width");
    result.height = root->parse_child<int>("height");

    result.camera = result.object_context->create<Camera>(root->child_node("camera"));
    result.camera->set_w_over_h(static_cast<float>(result.width) / result.height);

    result.renderer = result.object_context->create<Renderer>(root->child_node("renderer"));
    result.renderer->set_camera(result.camera);
    result.renderer->set_film(result.width, result.height);
    result.renderer->set_scene(result.scene);

    auto sorted_objects = topology_sort_object_tree(result.renderer->get_dependent_objects());

    result.scene->precommit();
    for(auto &obj : sorted_objects)
        obj->commit();
    result.scene->postcommit();

    result.renderer->recompile(true);

    //for(auto &obj : sorted_objects)
    //{
    //    for(auto &p : obj->get_properties())
    //        p->update();
    //}

    return result;
}

void display_image(GLuint tex_handle, int scene_width, int scene_height)
{
    const auto [window_width, window_height] = ImGui::GetWindowSize();
    if(window_width < 50 || window_height < 50)
        return;

    const float avail_width = (std::max)(window_width * 0.9f, window_width - 50);
    const float avail_height = (std::max)(window_height * 0.9f, window_height - 50);

    const float window_ratio = avail_width / avail_height;
    const float image_ratio = static_cast<float>(scene_width) / scene_height;

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

    ImGui::SetCursorPosX(0.5f * (window_width - display_size_x));
    ImGui::SetCursorPosY(0.5f * (window_height - display_size_y));
    ImGui::Image(
        reinterpret_cast<ImTextureID>(static_cast<size_t>(tex_handle)),
        ImVec2(display_size_x, display_size_y));
}

void run(const std::string &config_filename)
{
    auto scene = initialize_btrc_scene(config_filename);

    auto opengl = initialize_opengl();
    BTRC_SCOPE_EXIT{ destroy_opengl(opengl); };

    auto reporter = newRC<GUIPreviewer>();
    scene.renderer->set_reporter(reporter);
    scene.renderer->set_preview_interval(200);
    scene.renderer->render_async();

    GLuint tex_handle = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &tex_handle);
    if(!tex_handle)
        throw std::runtime_error("failed to create gl texture");
    glTextureStorage2D(tex_handle, 1, GL_RGBA32F, scene.width, scene.height);

    auto update_image = [&](const Image<Vec4f> &image)
    {
        if(image)
        {
            glTextureSubImage2D(
                tex_handle, 0, 0, 0, scene.width, scene.height,
                GL_RGBA, GL_FLOAT, image.data());
        }
    };

    while(!glfwWindowShouldClose(opengl.window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        glClearColor(0, 1, 1, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        if(reporter->get_dirty_flag())
            reporter->access_image(update_image);

        {
            auto viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(viewport->WorkPos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(viewport->WorkSize, ImGuiCond_Always);
            ImGuiWindowFlags window_flags = 0;
            window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking;
            window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

            if(ImGui::Begin("display", nullptr, window_flags))
            {
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.1f, 0.8f, 0.1f, 1));
                ImGui::ProgressBar(reporter->get_percentage() / 100.0f);
                ImGui::PopStyleColor();
                display_image(tex_handle, scene.width, scene.height);
            }
            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow *backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(opengl.window);
    }
    
    glDeleteTextures(1, &tex_handle);

    if(scene.renderer->is_rendering())
        scene.renderer->stop_async();
    else
        scene.renderer->wait_async();
}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        std::cout << "usage: BtrcGUI config.json" << std::endl;
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
