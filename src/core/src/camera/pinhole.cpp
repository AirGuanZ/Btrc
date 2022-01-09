#include <btrc/core/camera/pinhole.h>

BTRC_CORE_BEGIN

void PinholeCamera::set_eye(const Vec3f &eye)
{
    eye_ = eye;
}

void PinholeCamera::set_dst(const Vec3f &dst)
{
    dst_ = dst;
}

void PinholeCamera::set_up(const Vec3f &up)
{
    up_ = up;
}

void PinholeCamera::set_fov_y_deg(float deg)
{
    fov_y_deg_ = deg;
}

void PinholeCamera::set_w_over_h(float ratio)
{
    w_over_h_ = ratio;
}

void PinholeCamera::set_duration(float beg, float end)
{
    assert(beg <= end);
    beg_time_ = beg;
    end_time_ = end;
}

void PinholeCamera::preprocess()
{
    const Vec3f forward = normalize(dst_ - eye_);
    const Vec3f ex = normalize(cross(forward, up_));
    const Vec3f ey = normalize(cross(forward, ex));

    const float y_len = 2 * std::tan(0.5f * fov_y_deg_ * btrc_pi / 180);
    const float x_len = w_over_h_ * y_len;

    left_bottom_corner_ = eye_ + forward - 0.5f * (x_len * ex + y_len * ey);
    film_x_ = x_len * ex;
    film_y_ = y_len * ey;
}

Camera::SampleWeResult PinholeCamera::generate_ray(
    const CVec2f &film_coord, f32 time_sample) const
{
    CVec3f x = film_coord.x * film_x_;
    CVec3f y = film_coord.y * film_y_;
    CVec3f dst = left_bottom_corner_ + x + y;
    CVec3f dir = normalize(dst - eye_);

    SampleWeResult result;
    result.pos = eye_;
    result.dir = dir;
    result.throughput = CSpectrum::ones();
    result.time = beg_time_ < end_time_ ?
        lerp(beg_time_, end_time_, time_sample) : f32(beg_time_);

    return result;
}

BTRC_CORE_END
