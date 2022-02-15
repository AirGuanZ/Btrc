#pragma once

#include <span>
#include <vector>

#include <btrc/common.h>

BTRC_BEGIN

class AliasTable
{
public:

    struct Unit
    {
        float    accept_prob;
        uint32_t another_idx;
    };

    AliasTable() = default;

    explicit AliasTable(std::span<float> probs);

    void initialize(std::span<float> probs);

    std::span<const Unit> get_table() const;

    uint32_t sample(float u) const;

private:

    std::vector<Unit> units_;
};

BTRC_END
