#pragma once

#include <tuple>

#include <btrc/common.h>

BTRC_BEGIN

// from https://gist.github.com/yujincheng08/a65bb3ce21f8b91c0292d0974b8dee23
template <typename T>
class enumerate
{
    struct iterator
    {
        using iter_type = decltype(std::declval<T>().begin());

    private:

        friend class enumerate;

        iterator(const std::size_t &i, const iter_type &iter)
            : i_(i), iter_(iter) {}

        std::size_t i_;
        iter_type iter_;

    public:

        std::tuple<const std::size_t, typename std::iterator_traits<iter_type>::reference> operator*()
        {
            return { i_, *iter_ };
        }

        bool operator!=(const iterator &other)
        {
            return other.iter_ != iter_;
        }

        auto operator++()
        {
            ++i_;
            ++iter_;
            return *this;
        }
    };

public:

    template<typename U>
    explicit enumerate(U &&container)
        : container_(std::forward<U>(container))
    {
        
    }

    auto begin()
    {
        return iterator{ 0, container_.begin() };
    }

    auto end()
    {
        return iterator{ container_.size(), container_.end() };
    }

private:
    T container_;
};

template <typename U>
enumerate(U &&container)
    ->enumerate<std::enable_if_t<
        !std::is_rvalue_reference_v<decltype(std::forward<U>(container))>, U &>>;

template <typename U>
enumerate(U &&container)
    ->enumerate<std::enable_if_t<
        std::is_rvalue_reference_v<decltype(std::forward<U>(container))>, const U>>;

BTRC_END
