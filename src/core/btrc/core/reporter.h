#pragma once

#include <btrc/utils/image.h>

BTRC_BEGIN

class Reporter
{
public:

    virtual ~Reporter() = default;

    // will only be called from one thread

    virtual void new_stage(std::string_view name = {}) { }

    virtual void complete_stage() { }

    virtual void progress(float percentage) { }

    virtual bool need_preview() const { return false; }

    virtual void new_preview(const Vec4f *device_preview, int width, int height) { }

    void set_fast_preview(bool enable_fast_preview) { fast_preview_ = enable_fast_preview; }

    bool need_fast_preview() const { return fast_preview_; }

    void set_denoise(bool denoise) { denoise_ = denoise; }

    bool need_denoise() const { return denoise_; }

private:

    std::atomic<bool> fast_preview_ = false;
    std::atomic<bool> denoise_ = false;
};

BTRC_END
