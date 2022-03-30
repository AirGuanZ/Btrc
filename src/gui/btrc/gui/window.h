#pragma once

#include <btrc/utils/uncopyable.h>

#include "common.h"

BTRC_GUI_BEGIN

class Window : public Uncopyable
{
public:

    Window(const std::string &title, int width, int height);

    ~Window();

    void begin_frame();

    void end_frame();

    void set_close(bool close);

    bool should_close() const;

private:

    struct Impl;

    Box<Impl> impl_;
};

BTRC_GUI_END
