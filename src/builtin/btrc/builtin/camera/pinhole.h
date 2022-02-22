#pragma once

#include <btrc/core/camera.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class PinholeCamera : public Camera
{
public:

    void set_eye(const Vec3f &eye);

    void set_dst(const Vec3f &dst);

    void set_up(const Vec3f &up);

    void set_fov_y_deg(float deg);

    void set_w_over_h(float ratio) override;

    void set_duration(float beg, float end);

    void commit() override;

    SampleWeResult generate_ray_inline(
        CompileContext &cc,
        ref<CVec2f>     film_coord,
        f32             time_sample) const override;

private:

    BTRC_PROPERTY(Vec3f, eye_);
    BTRC_PROPERTY(Vec3f, dst_);
    BTRC_PROPERTY(Vec3f, up_);

    BTRC_PROPERTY(float, beg_time_, 0.0f);
    BTRC_PROPERTY(float, end_time_, 0.0f);

    BTRC_PROPERTY(float, fov_y_deg_, 60.0f);
    BTRC_PROPERTY(float, w_over_h_, 1.0f);

    BTRC_PROPERTY(Vec3f, left_bottom_corner_);
    BTRC_PROPERTY(Vec3f, film_x_);
    BTRC_PROPERTY(Vec3f, film_y_);
};

class PinholeCameraCreator : public factory::Creator<Camera>
{
public:

    std::string get_name() const override { return "pinhole"; }

    RC<Camera> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
