#include <fstream>
#include <sstream>

#include <btrc/utils/file.h>

BTRC_BEGIN

std::string read_txt_file(const std::string &filename)
{
    std::ifstream fin(filename, std::ios_base::in);
    if(!fin)
        throw std::runtime_error("failed to open text file: " + filename);
    std::stringstream sst;
    sst << fin.rdbuf();
    return sst.str();
}

BTRC_END
