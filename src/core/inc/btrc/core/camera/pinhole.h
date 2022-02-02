#pragma once

#include <btrc/core/camera/camera.h>

BTRC_CORE_BEGIN

class PinholeCamera : public Camera
{
public:

    void set_eye(const Vec3f &eye);

    void set_dst(const Vec3f &dst);

    void set_up(const Vec3f &up);

    void set_fov_y_deg(float deg);

    void set_w_over_h(float ratio);

    void set_duration(float beg, float end);

    SampleWeResult generate_ray(
        const CVec2f &film_coord, f32 time_sample) const override;

private:

    struct GenerateData
    {
        Vec3f left_bottom_corner;
        Vec3f film_x;
        Vec3f film_y;
    };

    GenerateData preprocess() const;

    Vec3f eye_;
    Vec3f dst_;
    Vec3f up_;
    float beg_time_ = 0;
    float end_time_ = 0;

    float fov_y_deg_ = 60;
    float w_over_h_  = 1;
};

BTRC_CORE_END
