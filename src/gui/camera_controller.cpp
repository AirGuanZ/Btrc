#include "camera_controller.h"

CameraController::CameraController(RC<builtin::PinholeCamera> camera)
    : camera_(std::move(camera))
{

}

bool CameraController::update(const InputParams &params)
{
    bool result = false;

    // translate

    if(params.button_down[1])
    {
        if(params.cursor_pos.x != last_cursor_pos_.x || params.cursor_pos.y != last_cursor_pos_.y)
        {
            translate_camera(last_cursor_pos_, params.cursor_pos);
            result = true;
        }
    }
    last_cursor_pos_ = params.cursor_pos;

    // TODO: rotate, dist

    return result;
}

void CameraController::translate_camera(const Vec2f &old_cursor, const Vec2f &new_cursor)
{
    const Vec3f eye = camera_->get_eye();
    const Vec3f dst = camera_->get_dst();
    const Vec3f up = camera_->get_up();

    const Vec3f forward = normalize(dst - eye);
    const Vec3f ex = normalize(cross(forward, up));
    const Vec3f ey = normalize(cross(forward, ex));

    const float dist = length(dst - eye);

    const float y_len = 2 * dist * std::tan(0.5f * camera_->get_fov_y_deg() * btrc_pi / 180);
    const float x_len = camera_->get_w_over_h() * y_len;
    const Vec3f film_x = x_len * ex;
    const Vec3f film_y = y_len * ey;

    // old_world_pos = dst     - 0.5f * (x_len * ex + y_len * ey) + old_cursor.x * film_x + old_cursor.y * film_y;
    // new_world_pos = new_dst - 0.5f * (x_len * ex + y_len * ey) + new_cursor.x * film_x + new_cursor.y * film_y
    // let new_world_pos == old_world_pos
    // dst     + old_cursor.x * film_x + old_cursor.y * film_y ==
    // new_dst + new_cursor.x * film_x + new_cursor.y * film_y

    const Vec3f new_dst = dst + (old_cursor.x - new_cursor.x) * film_x + (old_cursor.y - new_cursor.y) * film_y;
    const Vec3f new_eye = eye + (new_dst - dst);

    camera_->set_dst(new_dst);
    camera_->set_eye(new_eye);
}
