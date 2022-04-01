#pragma once

#include <btrc/core/camera.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

namespace pinhole_detail
{

    struct DeviceProperties
    {
        Vec3f eye;
        Vec3f left_bottom_corner;
        Vec3f film_x;
        Vec3f film_y;
    };

    CUJ_PROXY_CLASS(
        CDeviceProperties, DeviceProperties,
        eye, left_bottom_corner, film_x, film_y);

} // namespace pinhole_detail

class PinholeCamera : public Camera
{
public:

    void set_eye(const Vec3f &eye);

    void set_dst(const Vec3f &dst);

    void set_up(const Vec3f &up);

    void set_fov_y_deg(float deg);

    void set_w_over_h(float ratio) override;

    const Vec3f &get_eye() const;

    const Vec3f &get_dst() const;

    const Vec3f &get_up() const;

    float get_fov_y_deg() const;

    float get_w_over_h() const;

    void commit() override;

    AABB3f get_bounding_box() const override;

    SampleWeResult generate_ray_inline(
        CompileContext &cc,
        ref<CVec2f>     film_coord,
        f32             time_sample) const override;

private:

    using DeviceProperties = pinhole_detail::DeviceProperties;
    using CDeviceProperties = pinhole_detail::CDeviceProperties;

    Vec3f eye_;
    Vec3f dst_;
    Vec3f up_;

    float fov_y_deg_ = 60.0f;
    float w_over_h_ = 1.0f;

    cuda::Buffer<DeviceProperties> device_properties_;
};

class PinholeCameraCreator : public factory::Creator<Camera>
{
public:

    std::string get_name() const override { return "pinhole"; }

    RC<Camera> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
