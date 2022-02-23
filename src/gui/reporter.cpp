#include <iostream>

#include "reporter.h"

void GUIPreviewer::new_stage(std::string_view name)
{
    if(!name.empty())
        std::cout << name << std::endl;
    progress_ = 0.0f;
}

void GUIPreviewer::complete_stage()
{
    
}

void GUIPreviewer::progress(float percentage)
{
    progress_ = percentage;
    std::cout << percentage << "%" << std::endl;
}

bool GUIPreviewer::need_preview() const
{
    return true;
}

void GUIPreviewer::new_preview(const Image &preview)
{
    std::lock_guard lock(image_lock_);
    image_ = preview;
    dirty_flag_ = true;
}

bool GUIPreviewer::get_dirty_flag() const
{
    return true;
}

float GUIPreviewer::get_percentage() const
{
    return progress_;
}
