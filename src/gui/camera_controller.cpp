#include <btrc/utils/local_angle.h>

#include "camera_controller.h"

CameraController::CameraController(RC<builtin::PinholeCamera> camera)
    : camera_(std::move(camera))
{

}

bool CameraController::update(const InputParams &params, const ControlParams &control)
{
    if(params.cursor_pos.x < 0 || params.cursor_pos.x > 1 ||
       params.cursor_pos.y < 0 || params.cursor_pos.y > 1)
    {
        last_cursor_pos_ = params.cursor_pos;
        return false;
    }

    bool result = false;

    // translate

    if(params.button_down[1])
    {
        if(params.cursor_pos.x != last_cursor_pos_.x || params.cursor_pos.y != last_cursor_pos_.y)
        {
            translate(last_cursor_pos_, params.cursor_pos);
            result = true;
        }
    }

    // rotate

    if(params.button_down[2])
    {
        if(params.cursor_pos.x != last_cursor_pos_.x || params.cursor_pos.y != last_cursor_pos_.y)
        {
            rotate(last_cursor_pos_, params.cursor_pos, { control.rotate_speed_hori, control.rotate_speed_vert });
            result = true;
        }
    }

    // distance

    if(params.wheel_offset != 0)
        result |= adjust_distance(params.wheel_offset, control.dist_adjust_speed);

    last_cursor_pos_ = params.cursor_pos;

    return result;
}

void CameraController::translate(const Vec2f &old_cursor, const Vec2f &new_cursor)
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

    const Vec3f new_dst = dst + (old_cursor.x - new_cursor.x) * film_x + (old_cursor.y - new_cursor.y) * film_y;
    const Vec3f new_eye = eye + (new_dst - dst);

    camera_->set_dst(new_dst);
    camera_->set_eye(new_eye);
}

bool CameraController::adjust_distance(float wheel_offset, float speed)
{
    const Vec3f eye = camera_->get_eye();
    const Vec3f dst = camera_->get_dst();
    const Vec3f dir = normalize(dst - eye);
    float dist = length(eye - dst);
    if(wheel_offset > 0)
        dist /= 1 + speed;
    else
        dist *= 1 + speed;
    camera_->set_eye(dst - dist * dir);
    return true;
}

void CameraController::rotate(const Vec2f &old_cursor, const Vec2f &new_cursor, const Vec2f &speed)
{
    const Vec3f eye = camera_->get_eye();
    const Vec3f dst = camera_->get_dst();
    const Vec3f up = camera_->get_up();
    const Frame frame = Frame::from_z(up);

    const Vec3f dir = normalize(frame.global_to_local(dst - eye));
    float hori = local_angle::phi(dir);
    float vert = std::asin(std::clamp(dir.z, -1.0f, 1.0f));

    hori += (new_cursor.x - old_cursor.x) * speed.x;
    vert -= (new_cursor.y - old_cursor.y) * speed.y;

    while(hori > 2 * btrc_pi)
        hori -= 2 * btrc_pi;
    while(hori < 0)
        hori += 2 * btrc_pi;

    vert = std::clamp(vert, -0.5f * btrc_pi + 0.01f, 0.5f * btrc_pi - 0.01f);

    const Vec3f new_dir = frame.local_to_global(Vec3f(
        std::cos(hori) * std::cos(vert), std::sin(hori) * std::cos(vert), std::sin(vert)));
    const float dist = length(dst - eye);

    camera_->set_eye(dst - dist * new_dir);
}
