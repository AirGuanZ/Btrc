#include <btrc/factory/path_resolver.h>
#include <btrc/utils/string.h>

BTRC_FACTORY_BEGIN

void PathResolver::add_env_value(std::string_view name, std::string value)
{
    replacers_.insert({ "$env{" + std::string(name) + "}", std::move(value)});
}

std::filesystem::path PathResolver::resolve(std::string_view path) const
{
    std::string result(path);
    for(auto &[key, value] : replacers_)
        replace_(result, key, value);
    return absolute(std::filesystem::path(result)).lexically_normal();
}

BTRC_FACTORY_END
