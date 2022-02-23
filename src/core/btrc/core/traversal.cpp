#include <btrc/core/traversal.h>

BTRC_BEGIN

namespace
{
    
    void process(RC<Object> object, std::set<RC<Object>> &processed_objects, std::vector<RC<Object>> &output)
    {
        if(processed_objects.contains(object))
            return;
        for(auto &d : object->get_dependent_objects())
            process(d, processed_objects, output);
        assert(!processed_objects.contains(object));
        processed_objects.insert(object);
        output.push_back(object);
    }

} // namespace anonymous

std::vector<RC<Object>> topology_sort_object_tree(const std::set<RC<Object>> &entries)
{
    std::set<RC<Object>> processed_objects;
    std::vector<RC<Object>> output;
    for(auto &e : entries)
        process(e, processed_objects, output);
    return output;
}

BTRC_END
