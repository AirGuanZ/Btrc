#pragma once

#include <filesystem>
#include <map>

#include <btrc/common.h>

BTRC_FACTORY_BEGIN

class PathResolver
{
public:

    void add_env_value(std::string_view name, std::string value);

    std::filesystem::path resolve(std::string_view path) const;

private:

    std::map<std::string, std::string> replacers_;
};

BTRC_FACTORY_END
