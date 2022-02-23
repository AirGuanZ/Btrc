#include <btrc/builtin/reporter/console.h>

BTRC_BUILTIN_BEGIN

ConsoleReporter::ConsoleReporter()
    : pbar_(80, '=')
{
    
}

void ConsoleReporter::new_stage(std::string_view name)
{
    if(!name.empty())
        std::cout << name << std::endl;
    pbar_.reset_time();
    pbar_.set_percent(0);
    pbar_.display();
}

void ConsoleReporter::complete_stage()
{
    pbar_.done();
}

void ConsoleReporter::progress(float percentage)
{
    if(percentage - pbar_.get_percent() >= 1)
    {
        pbar_.set_percent(percentage);
        pbar_.display();
    }
}

BTRC_BUILTIN_END
