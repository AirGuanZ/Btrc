#pragma once

#include <btrc/builtin/camera/pinhole.h>

#include "common.h"

class CameraController
{
public:

    struct InputParams
    {
        Vec2f cursor_pos = { 0.0f, 0.0f }; // normalized to [0, 1]^2
        float wheel_offset = 0;
        std::array<bool, 3> button_down = { false, false, false };
    };

    explicit CameraController(RC<builtin::PinholeCamera> camera);

    bool update(const InputParams &params);

private:

    void translate(const Vec2f &old_cursor, const Vec2f &new_cursor);

    void rotate(const Vec2f &old_cursor, const Vec2f &new_cursor);

    bool adjust_distance(float wheel_offset);

    RC<builtin::PinholeCamera> camera_;

    Vec2f last_cursor_pos_;
};
