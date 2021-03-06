#pragma once

#include <btrc/builtin/camera/pinhole.h>
#include <btrc/gui/common.h>

BTRC_GUI_BEGIN

class CameraController
{
public:

    struct ControlParams
    {
        float rotate_speed_hori = 2;
        float rotate_speed_vert = 2;
        float dist_adjust_speed = 0.1f;
    };

    struct InputParams
    {
        Vec2f cursor_pos = { 0.0f, 0.0f }; // normalized to [0, 1]^2
        float wheel_offset = 0;
        std::array<bool, 3> button_down = { false, false, false };
    };

    explicit CameraController(RC<builtin::PinholeCamera> camera);

    bool update(const InputParams &params, const ControlParams &control);

private:

    void translate(const Vec2f &old_cursor, const Vec2f &new_cursor);

    void rotate(const Vec2f &old_cursor, const Vec2f &new_cursor, const Vec2f &speed);

    bool adjust_distance(float wheel_offset, float speed);

    RC<builtin::PinholeCamera> camera_;

    Vec2f last_cursor_pos_;
};

BTRC_GUI_END
