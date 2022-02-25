#include <btrc/builtin/camera/pinhole.h>

BTRC_BUILTIN_BEGIN

void PinholeCamera::set_eye(const Vec3f &eye)
{
    eye_.set(eye);
}

void PinholeCamera::set_dst(const Vec3f &dst)
{
    dst_.set(dst);
}

void PinholeCamera::set_up(const Vec3f &up)
{
    up_.set(up);
}

void PinholeCamera::set_fov_y_deg(float deg)
{
    fov_y_deg_.set(deg);
}

void PinholeCamera::set_w_over_h(float ratio)
{
    w_over_h_.set(ratio);
}

void PinholeCamera::set_duration(float beg, float end)
{
    assert(beg <= end);
    beg_time_.set(beg);
    end_time_.set(end);
}

const Vec3f &PinholeCamera::get_eye() const
{
    return eye_.get();
}

const Vec3f &PinholeCamera::get_dst() const
{
    return dst_.get();
}

const Vec3f &PinholeCamera::get_up() const
{
    return up_.get();
}

float PinholeCamera::get_fov_y_deg() const
{
    return fov_y_deg_.get();
}

float PinholeCamera::get_w_over_h() const
{
    return w_over_h_.get();
}

void PinholeCamera::commit()
{
    const Vec3f forward = normalize(dst_.get() - eye_.get());
    const Vec3f ex = normalize(cross(forward, up_.get()));
    const Vec3f ey = normalize(cross(forward, ex));

    const float y_len = 2 * std::tan(0.5f * fov_y_deg_.get() * btrc_pi / 180);
    const float x_len = w_over_h_.get() * y_len;

    left_bottom_corner_ = eye_.get() + forward - 0.5f * (x_len * ex + y_len * ey);
    film_x_ = x_len * ex;
    film_y_ = y_len * ey;
}

Camera::SampleWeResult PinholeCamera::generate_ray_inline(
    CompileContext &cc, ref<CVec2f> film_coord, f32 time_sample) const
{
    var x = film_coord.x * film_x_.read(cc);
    var y = film_coord.y * film_y_.read(cc);
    var dst = left_bottom_corner_.read(cc) + x + y;
    var dir = normalize(dst - eye_.read(cc));

    var time = beg_time_.read(cc);
    $if(beg_time_.read(cc) < end_time_.read(cc))
    {
        time = lerp(beg_time_.read(cc), end_time_.read(cc), time_sample);
    };

    SampleWeResult result;
    result.throughput = CSpectrum::one();
    result.pos = eye_.read(cc);
    result.dir = dir;
    result.time = time;
    return result;
}

RC<Camera> PinholeCameraCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    const Vec3f eye = node->parse_child<Vec3f>("eye");
    const Vec3f dst = node->parse_child<Vec3f>("dst");
    const Vec3f up  = node->parse_child<Vec3f>("up");

    const float fov_y_deg = node->parse_child<factory::Degree>("fov_y").value;
    const float beg_time = node->parse_child_or<float>("beg_time", 0);
    const float end_time = node->parse_child_or<float>("end_time", 0);

    auto camera = newRC<PinholeCamera>();
    camera->set_eye(eye);
    camera->set_dst(dst);
    camera->set_up(up);
    camera->set_fov_y_deg(fov_y_deg);
    camera->set_duration(beg_time, end_time);

    return camera;
}

BTRC_BUILTIN_END
