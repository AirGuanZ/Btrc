#include <btrc/utils/cmath/calias.h>

BTRC_BEGIN

CAliasTable::CAliasTable(const AliasTable &table)
{
    units_ = cuda::CUDABuffer(table.get_table());
}

u32 CAliasTable::sample(f32 _u) const
{
    static auto func = cuj::function(
        "btrc_sample_alias_table",
        [](ptr<Unit> table, u32 n, f32 u)
    {
        f32 nu = f32(n) * u;
        u32 i = cstd::min(u32(nu), n - 1);
        f32 s = nu - f32(i);
        return cstd::select(s <= table[i].accept_prob, i, u32(table[i].another_idx));
    });
    var table_ptr = cuj::import_pointer(const_cast<AliasTable::Unit *>(units_.get()));
    return func(table_ptr, static_cast<uint32_t>(units_.get_size()), _u);
}

BTRC_END
