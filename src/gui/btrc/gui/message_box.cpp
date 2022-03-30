#include <imgui.h>

#include "message_box.h"

namespace ImGui
{

    void EnableMessageBox(const char *title)
    {
        OpenPopup(title);
    }

    void DisplayMessageBox(const char *title, const char *text)
    {
        ImVec2 center = GetMainViewport()->GetCenter();
        SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        if(BeginPopup(title, ImGuiWindowFlags_AlwaysAutoResize))
        {
            TextUnformatted(text);
            EndPopup();
        }
    }

} // namespace ImGui
