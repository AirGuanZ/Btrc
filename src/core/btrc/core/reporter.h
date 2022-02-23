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

    virtual void new_preview(const Image<Vec4f> &preview) { }
};

BTRC_END
