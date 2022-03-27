#pragma once

#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

// reference: pbrt-v4/utils/hash.h
// https://github/mmp/pbrt-v4
namespace hash
{

    inline u64 murmur_hash64A(ptr<u8> key, u64 len, u64 seed)
    {
        constexpr uint64_t m = 0xc6a4a7935bd1e995ull;
        constexpr int r = 47;
        var h = seed ^ (len * m);
        var end = key + u64(8) * (len / 8);
        $while(key != end)
        {
            u64 k;
            cstd::memcpy(k.address(), key, sizeof(uint64_t));
            key = key + 8;
            k = k *  m;
            k = k ^ (k >> r);
            k = k * m;
            h = h ^ k;
            h = h * m;
        };
        $switch(len & 7)
        {
            $case(7)
            {
                h = h ^ u64(key[6]) << 48;
                $fallthrough;
            };
            $case(6)
            {
                h = h ^ u64(key[5]) << 40;
                $fallthrough;
            };
            $case(5)
            {
                h = h ^ u64(key[4]) << 32;
                $fallthrough;
            };
            $case(4)
            {
                h = h ^ u64(key[3]) << 24;
                $fallthrough;
            };
            $case(3)
            {
                h = h ^ u64(key[2]) << 16;
                $fallthrough;
            };
            $case(2)
            {
                h = h ^ u64(key[1]) << 8;
                $fallthrough;
            };
            $case(1)
            {
                h = h ^ u64(key[0]);
                h = h * m;
                $fallthrough;
            };
        };
        h = h ^ (h >> r);
        h = h * m;
        h = h ^ (h >> r);
        return h;
    }

    template<typename... Args>
    void hash_recursive_copy(ptr<u8> buf, Args...);

    template<>
    inline void hash_recursive_copy(ptr<u8> buf)
    {
        
    }

    template<typename T, typename...Args>
    void hash_recursive_copy(ptr<u8> buf, T v, Args...args)
    {
        constexpr size_t size = sizeof(cuj::dsl::cuj_to_cxx_t<cuj::dsl::remove_reference_t<T>>);
        cstd::memcpy(buf, v.address(), size);
        hash_recursive_copy(buf + size, args...);
    }

    template<typename...Args>
    u64 hash(Args...args)
    {
        constexpr size_t sz = (sizeof(cuj::dsl::cuj_to_cxx_t<cuj::dsl::remove_reference_t<Args>>) + ... + 0);
        constexpr size_t n = (sz + 7) / 8;
        cuj::arr<u64, n> buf;
        hash_recursive_copy(cuj::bitcast<ptr<u8>>(buf[0].address()), args...);
        return murmur_hash64A(cuj::bitcast<ptr<u8>>(buf[0].address()), sz, 0);
    }

} // namespace anonymous

BTRC_END
