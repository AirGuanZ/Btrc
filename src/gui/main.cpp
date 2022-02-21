#include <iostream>
#include <vector>

#include <btrc/utils/exception.h>
#include <btrc/utils/scope_guard.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

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

    while(!glfwWindowShouldClose(glfw_window))
    {
        glfwPollEvents();
        glClearColor(0, 1, 1, 0);
        glClear(GL_COLOR_BUFFER_BIT);
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
