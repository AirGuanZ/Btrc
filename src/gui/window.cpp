#ifdef WIN32
#include <Windows.h>
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <btrc/utils/scope_guard.h>

#include "window.h"

struct Window::Impl
{
    GLFWwindow *glfw_window;
};

Window::Window(const std::string &title, int width, int height)
{
    impl_ = newBox<Impl>();

    if(!glfwInit())
        throw std::runtime_error("failed to initialize glfw");
    BTRC_SCOPE_FAIL{ glfwTerminate(); };

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    impl_->glfw_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if(!impl_->glfw_window)
        throw std::runtime_error("failed to create glfw window");
    BTRC_SCOPE_FAIL{ glfwDestroyWindow(impl_->glfw_window); };

    glfwMakeContextCurrent(impl_->glfw_window);
    glfwSwapInterval(1);

    if(glewInit() != GLEW_OK)
        throw std::runtime_error("failed to iniailize glew");

    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    ImGui::GetIO().IniFilename = nullptr;

    ImFontConfig font_config = {};
    font_config.SizePixels = 16;
    ImGui::GetIO().Fonts->AddFontDefault(&font_config);

    ImGui_ImplGlfw_InitForOpenGL(impl_->glfw_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    ImGui::StyleColorsLight();
}

Window ::~Window()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(impl_->glfw_window);
    glfwTerminate();
}

void Window::begin_frame()
{
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    glClearColor(0, 1, 1, 0);
    glClear(GL_COLOR_BUFFER_BIT);
}

void Window::end_frame()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    if(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow *backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
    glfwSwapBuffers(impl_->glfw_window);
}

void Window::set_close(bool close)
{
    glfwSetWindowShouldClose(impl_->glfw_window, close);
}

bool Window::should_close() const
{
    return glfwWindowShouldClose(impl_->glfw_window);
}
