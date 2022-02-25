#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <Windows.h>
#elif
#include <unistd.h>
#endif

#include <btrc/utils/file.h>

BTRC_BEGIN

std::string read_txt_file(const std::string &filename)
{
    std::ifstream fin(filename, std::ios_base::in);
    if(!fin)
        throw BtrcException("failed to open text file: " + filename);
    std::stringstream sst;
    sst << fin.rdbuf();
    return sst.str();
}

std::filesystem::path get_executable_filename()
{
#ifdef _WIN32
    wchar_t sz_path[MAX_PATH];
    GetModuleFileNameW(nullptr, sz_path, MAX_PATH);
#else
    char sz_path[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", sz_path, PATH_MAX);
    if(count < 0 || count >= PATH_MAX)
        throw BtrcException("failed to get executable path");
    sz_path[count] = '\0';
#endif
    return absolute(std::filesystem::path(sz_path)).lexically_normal();
}

BTRC_END
