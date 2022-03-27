#pragma once

#include <array>

#include <btrc/common.h>

BTRC_BEGIN

constexpr size_t PRIME_TABLE_SIZE = 1000;

extern const std::array<int, PRIME_TABLE_SIZE> PRIME_TABLE;

BTRC_END
