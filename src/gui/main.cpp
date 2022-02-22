#include <iostream>
#include <vector>

#include <btrc/utils/exception.h>
#include <btrc/utils/scope_guard.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

void run()
{
    if(!glfwInit())
        throw std::runtime_error("failed to initialize glfw");
    BTRC_SCOPE_EXIT{ glfwTerminate(); };

    auto glfw_window = glfwCreateWindow(640, 480, "BtrcGUI", nullptr, nullptr);
    if(!glfw_window)
        throw std::runtime_error("failed to create glfw window");
    BTRC_SCOPE_EXIT{ glfwDestroyWindow(glfw_window); };

    glfwMakeContextCurrent(glfw_window);

    if(glewInit() != GLEW_OK)
        throw std::runtime_error("failed to iniailize glew");

    ImGui::CreateContext();
    BTRC_SCOPE_EXIT{ ImGui::DestroyContext(); };
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    ImGui::GetIO().IniFilename = nullptr;

    ImGui_ImplGlfw_InitForOpenGL(glfw_window, true);
    BTRC_SCOPE_EXIT{ ImGui_ImplGlfw_Shutdown(); };

    ImGui_ImplOpenGL3_Init("#version 330");
    BTRC_SCOPE_EXIT{ ImGui_ImplOpenGL3_Shutdown(); };

    ImGui::StyleColorsLight();

    while(!glfwWindowShouldClose(glfw_window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        glClearColor(0, 1, 1, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui::DockSpaceOverViewport(nullptr, ImGuiDockNodeFlags_PassthruCentralNode);

        ImGui::SetNextWindowSize(ImVec2(400, 400));
        if(ImGui::Begin("hello, imgui!"))
        {
            if(ImGui::Button("exit"))
                glfwSetWindowShouldClose(glfw_window, 1);
        }
        ImGui::End();

        ImGui::SetNextWindowSize(ImVec2(400, 400));
        if(ImGui::Begin("hello, imgui docking!", nullptr))
        {
            if(ImGui::Button("exit"))
                glfwSetWindowShouldClose(glfw_window, 1);
        }
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow *backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(glfw_window);
    }
}

int main()
{
    try
    {
        run();
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
