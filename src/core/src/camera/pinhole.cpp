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

Camera::SampleWeResult PinholeCamera::generate_ray_inline(
    ref<CVec2f> film_coord, f32 time_sample) const
{
    const auto generate_data = preprocess();

    var x = film_coord.x * generate_data.film_x;
    var y = film_coord.y * generate_data.film_y;
    var dst = generate_data.left_bottom_corner + x + y;
    var dir = normalize(dst - eye_);
    var time = beg_time_ < end_time_ ?
        lerp(beg_time_, end_time_, time_sample) : f32(beg_time_);

    SampleWeResult result;
    result.throughput = CSpectrum::one();
    result.pos = CVec3f(eye_);
    result.dir = dir;
    result.time = time;
    return result;
}

PinholeCamera::GenerateData PinholeCamera::preprocess() const
{
    const Vec3f forward = normalize(dst_ - eye_);
    const Vec3f ex = normalize(cross(forward, up_));
    const Vec3f ey = normalize(cross(forward, ex));

    const float y_len = 2 * std::tan(0.5f * fov_y_deg_ * btrc_pi / 180);
    const float x_len = w_over_h_ * y_len;

    GenerateData ret;
    ret.left_bottom_corner = eye_ + forward - 0.5f * (x_len * ex + y_len * ey);
    ret.film_x = x_len * ex;
    ret.film_y = y_len * ey;
    return ret;
}

BTRC_CORE_END
