#pragma once

#include <btrc/common.h>

BTRC_BEGIN

std::string load_kernel_cache(const std::string &cache_filename);

void create_kernel_cache(const std::string &filename, const std::string &ptx);

BTRC_END
