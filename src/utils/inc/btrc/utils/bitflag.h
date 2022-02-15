#pragma once

#include <btrc/core/common.h>

BTRC_BEGIN

template<typename T, typename FlagTag = void>
class BitFlag
{
public:

    using UnderlyingType = T;

    constexpr BitFlag(T value = {});

    constexpr BitFlag operator|(BitFlag other) const;

    constexpr operator bool() const;

    constexpr bool operator==(BitFlag rhs) const;

    constexpr std::strong_ordering operator<=>(BitFlag rhs) const;

private:

    static_assert(std::is_integral_v<T>);

    T value_;
};

// ========================== impl ==========================

template<typename T, typename FlagTag>
constexpr BitFlag<T, FlagTag>::BitFlag(T value)
    : value_(value)
{
    
}

template<typename T, typename FlagTag>
constexpr BitFlag<T, FlagTag> BitFlag<T, FlagTag>::operator|(BitFlag other) const
{
    return BitFlag(value_ | other.value_);
}

template<typename T, typename FlagTag>
constexpr BitFlag<T, FlagTag>::operator bool() const
{
    return value_ != 0;
}

template<typename T, typename FlagTag>
constexpr bool BitFlag<T, FlagTag>::operator==(BitFlag rhs) const
{
    return value_ != rhs.value_;
}

template<typename T, typename FlagTag>
constexpr std::strong_ordering BitFlag<T, FlagTag>::operator<=>(BitFlag rhs) const
{
    return value_ <=> rhs.value_;
}

BTRC_END
