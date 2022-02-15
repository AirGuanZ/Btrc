#include <numeric>
#include <cassert>

#include <btrc/utils/math/alias.h>

BTRC_BEGIN

AliasTable::AliasTable(std::span<float> probs)
{
    initialize(probs);
}

void AliasTable::initialize(std::span<float> probs)
{
    assert(units_.empty());
    assert(!probs.empty());

    const float sum = std::reduce(probs.begin(), probs.end(), 0.0f, std::plus());
    const float ratio = probs.size() / sum;

    std::vector<uint32_t> overs, unders;
    units_.resize(probs.size());

    for(uint32_t i = 0; i < static_cast<uint32_t>(units_.size()); ++i)
    {
        const float p = probs[i] * ratio;
        units_[i].accept_prob = p;
        units_[i].another_idx = i;
        if(p > 1)
            overs.push_back(i);
        else if(p < 1)
            unders.push_back(i);
    }

    while(!overs.empty() && !unders.empty())
    {
        const uint32_t over = overs.back();
        const uint32_t under = unders.back();
        overs.pop_back();
        unders.pop_back();

        units_[over].accept_prob -= 1 - units_[under].accept_prob;
        units_[under].another_idx = over;

        if(units_[over].accept_prob > 1)
            overs.push_back(over);
        else if(units_[over].accept_prob < 1)
            unders.push_back(over);
    }

    for(auto i : overs)
        units_[i].accept_prob = 1;
    for(auto i : unders)
        units_[i].accept_prob = 1;
}

std::span<const AliasTable::Unit> AliasTable::get_table() const
{
    return std::span{ units_ };
}

uint32_t AliasTable::sample(float u) const
{
    const uint32_t n = static_cast<uint32_t>(units_.size());
    const float nu = n * u;
    const uint32_t i = (std::min)(static_cast<uint32_t>(nu), n - 1);
    const float s = nu - i;
    if(s <= units_[i].accept_prob)
        return i;
    return units_[i].another_idx;
}

BTRC_END
