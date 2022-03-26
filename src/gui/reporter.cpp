#include <cuda_runtime.h>

#include <btrc/utils/cuda/error.h>

#include "reporter.h"

void GUIPreviewer::new_stage(std::string_view name)
{
    progress_ = 0.0f;
}

void GUIPreviewer::complete_stage()
{
    progress_ = 100;
}

void GUIPreviewer::progress(float percentage)
{
    progress_ = percentage;
}

bool GUIPreviewer::need_preview() const
{
    const auto interval = std::chrono::milliseconds(preview_interval_ms_);
    return PreviewClock::now() - last_preview_time_ >= interval;
}

void GUIPreviewer::new_preview(
    btrc::Vec4f *device_preview,
    btrc::Vec4f *device_albedo,
    btrc::Vec4f *device_normal,
    int width, int height)
{
    if(!device_preview)
        return;

    for(auto &p : post_processors_)
    {
        if(p->get_execution_policy() == btrc::PostProcessor::ExecutionPolicy::Always)
            p->process(device_preview, device_albedo, device_normal, width, height);
    }

    if(!gamma_)
        gamma_ = btrc::newRC<Gamma>();
    gamma_->process(device_preview, device_albedo, device_normal, width, height);

    {
        std::lock_guard lock(image_lock_);
        if(image_.width() != width || image_.height() != height)
            image_ = Image(width, height);
        btrc::throw_on_error(cudaMemcpy(
            image_.data(), device_preview,
            sizeof(btrc::Vec4f) * width * height,
            cudaMemcpyDeviceToHost));
        dirty_flag_ = true;
    }

    last_preview_time_ = PreviewClock::now();
}

void GUIPreviewer::set_preview_interval(int ms)
{
    preview_interval_ms_ = ms;
}

void GUIPreviewer::set_post_processors(std::vector<btrc::RC<btrc::PostProcessor>> post_processors)
{
    post_processors_ = std::move(post_processors);
}

float GUIPreviewer::get_percentage() const
{
    return progress_;
}
