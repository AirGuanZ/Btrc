#pragma once

#include <btrc/core/utils/cmath/cscalar.h>
#include <btrc/core/utils/cuda/buffer.h>
#include <btrc/core/utils/math/alias.h>

BTRC_CORE_BEGIN

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

    CUDABuffer<AliasTable::Unit> units_;
};

BTRC_CORE_END

namespace cuj::dsl
{

    template<>
    struct CXXClassToCujClass<btrc::core::AliasTable::Unit>
    {
        using Type = btrc::core::CAliasTable::Unit;
    };

} // namespace cuj::dsl
