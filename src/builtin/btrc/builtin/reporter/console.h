#pragma once

#include <btrc/core/reporter.h>
#include <btrc/utils/pbar.h>

BTRC_BUILTIN_BEGIN

class ConsoleReporter : public Reporter
{
public:

    ConsoleReporter();

    void new_stage(std::string_view name) override;

    void complete_stage() override;

    void progress(float percentage) override;

private:

    ConsoleProgressBar pbar_;
};

BTRC_BUILTIN_END
