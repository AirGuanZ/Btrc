#include <filesystem>
#include <fstream>

#include <btrc/utils/file.h>
#include <btrc/utils/ptx_cache.h>

BTRC_BEGIN

std::string load_kernel_cache(const std::string &cache_filename)
{
    const auto parent = get_executable_filename().parent_path();
    try
    {
        return read_txt_file(cache_filename);
    }
    catch(...)
    {
        return {};
    }
}

void create_kernel_cache(const std::string &filename, const std::string &ptx)
{
    create_directories(std::filesystem::path(filename).parent_path());
    std::ofstream fout(filename, std::ofstream::out | std::ofstream::trunc);
    if(!fout)
        throw BtrcException("failed to open file: " + filename);
    fout << ptx;
}

BTRC_END
