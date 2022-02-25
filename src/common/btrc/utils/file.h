#pragma once

#include <filesystem>

#include <btrc/common.h>

BTRC_BEGIN

std::string read_txt_file(const std::string &filename);

std::filesystem::path get_executable_filename();

BTRC_END
