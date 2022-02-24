#pragma once

#include <btrc/core/context.h>

BTRC_BEGIN

class ObjectDAG
{
public:

    template<typename It>
    ObjectDAG(It begin, It end);

    explicit ObjectDAG(const std::vector<RC<Object>> &objects);

    explicit ObjectDAG(const RC<Object> &object);

    const std::vector<RC<Object>> &get_sorted_objects() const;

    void commit();

    bool need_recompile() const;

    void clear_recompile_flag();

    void update_properties() const;

private:

    void add(const RC<Object> &object, std::set<RC<Object>> &processed);

    std::set<RC<Object>>    entries_;
    std::vector<RC<Object>> sorted_;
};

template<typename It>
ObjectDAG::ObjectDAG(It begin, It end)
{
    std::set<RC<Object>> processed;
    while(begin != end)
    {
        auto entry = *begin++;
        this->add(entry, processed);
        entries_.insert(entry);
    }
}

BTRC_END
