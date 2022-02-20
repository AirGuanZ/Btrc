#pragma once

#include <iostream>
#include <chrono>

#include <btrc/common.h>

BTRC_BEGIN

class ConsoleProgressBar
{
    float percent_;
    int   width_;
    char  complete_;
    char  incomplete_;

    std::chrono::steady_clock::time_point start_ =
        std::chrono::steady_clock::now();

public:

    explicit ConsoleProgressBar(int width, char complete = '#', char incomplete = ' ')
        : percent_(0),
          width_(width),
          complete_(complete),
          incomplete_(incomplete)
    {

    }

    void set_percent(float percent)
    {
        percent_ = percent;
    }

    float get_percent() const
    {
        return percent_;
    }

    void reset_time()
    {
        start_ = std::chrono::steady_clock::now();
    }

    auto get_time_ms() const
    {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_).count();
    }

    void display() const
    {
        const int pos = static_cast<int>(width_ * percent_ / 100);
        const float time_elapsed = static_cast<float>(get_time_ms());

        std::cout << "[";

        for(int i = 0; i < width_; ++i)
        {
            if(i < pos)
                std::cout << complete_;
            else if(i == pos)
                std::cout << ">";
            else
                std::cout << incomplete_;
        }
        std::cout << "] " << int(percent_) << "% "
                  << time_elapsed / 1000.0f << "s   \r";
        std::cout.flush();
    }

    void done()
    {
        set_percent(100);
        display();
        std::cout << std::endl;
    }
};

BTRC_END
