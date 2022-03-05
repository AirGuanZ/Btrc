#pragma once

#include <btrc/utils/cmath/cscalar.h>
#include <btrc/utils/cuda/buffer.h>
#include <btrc/utils/math/alias.h>

BTRC_BEGIN

class CAliasTable
{
public:

    CUJ_CLASS_BEGIN(Unit)
        CUJ_MEMBER_VARIABLE(f32, accept_prob)
        CUJ_MEMBER_VARIABLE(u32, another_idx)
    CUJ_CLASS_END

    CAliasTable() = default;

    explicit CAliasTable(const AliasTable &table);

    u32 sample(f32 _u) const;

private:

    cuda::Buffer<AliasTable::Unit> units_;
};

BTRC_END

namespace cuj::dsl
{

    template<>
    struct CXXClassToCujClass<btrc::AliasTable::Unit>
    {
        using Type = btrc::CAliasTable::Unit;
    };

} // namespace cuj::dsl
