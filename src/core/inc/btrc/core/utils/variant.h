#pragma once

#include <variant>

#include <cuj/utils/variant.h>

#include <btrc/core/common.h>

BTRC_CORE_BEGIN

template<typename...Types>
class Variant : public std::variant<Types...>
{
public:

    using std::variant<Types...>::variant;
    using std_variant_t = std::variant<Types...>;

    Variant(const std_variant_t &other)
        : std_variant_t(other)
    {
        
    }

    template<typename T>
    bool is() const noexcept
    {
        return std::holds_alternative<T>(*this);
    }

    template<typename T>
    auto &as() const noexcept
    {
        return std::get<T>(*this);
    }

    template<typename T>
    auto &as() noexcept
    {
        return std::get<T>(*this);
    }

    template<typename T>
    Variant &operator=(T &&rhs)
    {
        *static_cast<std_variant_t*>(this) = std::forward<T>(rhs);
        return *this;
    }

    template<typename T>
    auto as_if() noexcept
    {
        return std::get_if<T>(this);
    }

    template<typename T>
    auto as_if() const noexcept
    {
        return std::get_if<T>(this);
    }

    template<typename...Vs>
    auto match(Vs...vs);

    template<typename...Vs>
    auto match(Vs...vs) const;

    auto operator<=>(const Variant &rhs) const;

    bool operator==(const Variant &) const;
};

namespace variant_impl
{

    template<typename T>
    decltype(auto) to_std_variant(T &&var) { return std::forward<T>(var); }

    template<typename...Ts>
    std::variant<Ts...> &to_std_variant(Variant<Ts...> &var)
    {
        return static_cast<std::variant<Ts...>&>(var);
    }

    template<typename...Ts>
    const std::variant<Ts...> &to_std_variant(const Variant<Ts...> &var)
    {
        return static_cast<const std::variant<Ts...>&>(var);
    }

    template<typename...Ts>
    std::variant<Ts...> to_std_variant(Variant<Ts...> &&var)
    {
        return static_cast<std::variant<Ts...>>(var);
    }

} // namespace variant_impl

template<typename E, typename...Vs>
auto match_variant(E &&e, Vs...vs)
{
    struct overloaded : Vs...
    {
        explicit overloaded(Vs...vss) : Vs(vss)... { }
        using Vs::operator()...;
    };
    return std::visit(
        overloaded(vs...), variant_impl::to_std_variant(std::forward<E>(e)));
}

template<typename...Types>
template<typename...Vs>
auto Variant<Types...>::match(Vs ...vs)
{
    return cuj::match_variant(*this, std::move(vs)...);
}

template<typename...Types>
template<typename...Vs>
auto Variant<Types...>::match(Vs ...vs) const
{
    return cuj::match_variant(*this, std::move(vs)...);
}

template<typename ... Types>
auto Variant<Types...>::operator<=>(const Variant &rhs) const
{
    return variant_impl::to_std_variant(*this) <=>
           variant_impl::to_std_variant(rhs);
}

template<typename ... Types>
bool Variant<Types...>::operator==(const Variant &rhs) const
{
    return variant_impl::to_std_variant(*this) ==
           variant_impl::to_std_variant(rhs);
}

BTRC_CORE_END
