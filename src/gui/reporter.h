#pragma once

#include <mutex>

#include <btrc/core/reporter.h>

class GUIPreviewer : public btrc::Reporter
{
public:

    using Image = btrc::Image<btrc::Vec4f>;

    void new_stage(std::string_view name) override;

    void complete_stage() override;

    void progress(float percentage) override;

    bool need_preview() const override;

    void new_preview(const Image &preview) override;

    void set_preview_interval(int ms);

    template<typename F>
    void access_dirty_image(const F &f);

    float get_percentage() const;

private:

    using PreviewClock = std::chrono::steady_clock;

    std::mutex image_lock_;
    Image      image_;

    bool dirty_flag_ = true;
    std::atomic<float> progress_ = 0.0f;

    std::atomic<int> preview_interval_ms_ = 1000;
    PreviewClock::time_point last_preview_time_ = PreviewClock::now();
};

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
