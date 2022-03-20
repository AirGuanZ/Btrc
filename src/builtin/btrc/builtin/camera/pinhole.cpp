#include <btrc/builtin/camera/pinhole.h>

BTRC_BUILTIN_BEGIN

void PinholeCamera::set_eye(const Vec3f &eye)
{
    eye_ = eye;
    set_need_commit();
}

void PinholeCamera::set_dst(const Vec3f &dst)
{
    dst_ = dst;
    set_need_commit();
}

void PinholeCamera::set_up(const Vec3f &up)
{
    up_ = up;
    set_need_commit();
}

void PinholeCamera::set_fov_y_deg(float deg)
{
    fov_y_deg_ = deg;
    set_need_commit();
}

void PinholeCamera::set_w_over_h(float ratio)
{
    w_over_h_ = ratio;
    set_need_commit();
}

void PinholeCamera::set_duration(float beg, float end)
{
    assert(beg <= end);
    beg_time_ = beg;
    end_time_ = end;
    set_need_commit();
}

const Vec3f &PinholeCamera::get_eye() const
{
    return eye_;
}

const Vec3f &PinholeCamera::get_dst() const
{
    return dst_;
}

const Vec3f &PinholeCamera::get_up() const
{
    return up_;
}

float PinholeCamera::get_fov_y_deg() const
{
    return fov_y_deg_;
}

float PinholeCamera::get_w_over_h() const
{
    return w_over_h_;
}

void PinholeCamera::commit()
{
    const Vec3f forward = normalize(dst_ - eye_);
    const Vec3f ex = normalize(cross(forward, up_));
    const Vec3f ey = normalize(cross(forward, ex));

    const float y_len = 2 * std::tan(0.5f * fov_y_deg_ * btrc_pi / 180);
    const float x_len = w_over_h_ * y_len;

    const Vec3f left_bottom_corner = eye_ + forward - 0.5f * (x_len * ex + y_len * ey);
    const Vec3f film_x = x_len * ex;
    const Vec3f film_y = y_len * ey;

    const DeviceProperties device_properties = {
        .eye = eye_,
        .beg_time = beg_time_,
        .end_time = end_time_,
        .left_bottom_corner = left_bottom_corner,
        .film_x = film_x,
        .film_y = film_y
    };
    if(device_properties_.is_empty())
        device_properties_.initialize(1);
    device_properties_.from_cpu(&device_properties);
}

AABB3f PinholeCamera::get_bounding_box() const
{
    return AABB3f(eye_, eye_);
}

Camera::SampleWeResult PinholeCamera::generate_ray_inline(
    CompileContext &cc, ref<CVec2f> film_coord, f32 time_sample) const
{
    ref device_properties = *cuj::import_pointer(device_properties_.get());

    var x = film_coord.x * device_properties.film_x;
    var y = film_coord.y * device_properties.film_y;
    var dst = device_properties.left_bottom_corner + x + y;
    var dir = normalize(dst - device_properties.eye);

    var time = lerp(device_properties.beg_time, device_properties.end_time, time_sample);

    SampleWeResult result;
    result.throughput = CSpectrum::one();
    result.pos = device_properties.eye;
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
