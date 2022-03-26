#pragma once

#include <mutex>

#include <btrc/core/post_processor.h>
#include <btrc/core/reporter.h>

#include "./gamma.h"

class GUIPreviewer : public btrc::Reporter
{
public:

    using Image = btrc::Image<btrc::Vec4f>;

    void new_stage(std::string_view name) override;

    void complete_stage() override;

    void progress(float percentage) override;

    bool need_preview() const override;

    void new_preview(
        btrc::Vec4f *device_preview,
        btrc::Vec4f *device_albedo,
        btrc::Vec4f *device_normal,
        int width, int height) override;

    void set_preview_interval(int ms);

    void set_post_processors(std::vector<btrc::RC<btrc::PostProcessor>> post_processors);

    template<typename F>
    void access_image(const F &f);

    template<typename F>
    void access_dirty_image(const F &f);

    float get_percentage() const;

private:

    using PreviewClock = std::chrono::steady_clock;

    std::vector<btrc::RC<btrc::PostProcessor>> post_processors_;
    btrc::RC<Gamma> gamma_;

    // update image

    std::mutex image_lock_;
    Image      image_;

    bool dirty_flag_ = true;
    std::atomic<float> progress_ = 0.0f;

    std::atomic<int> preview_interval_ms_ = 1000;
    PreviewClock::time_point last_preview_time_ = PreviewClock::now();
};

template<typename F>
void GUIPreviewer::access_image(const F &f)
{
    std::lock_guard lock(image_lock_);
    f(image_);
    dirty_flag_ = false;
}

template<typename F>
void GUIPreviewer::access_dirty_image(const F &f)
{
    std::lock_guard lock(image_lock_);
    if(dirty_flag_)
    {
        f(image_);
        dirty_flag_ = false;
    }
}
