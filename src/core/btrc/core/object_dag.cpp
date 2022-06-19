#include <btrc/core/object_dag.h>

BTRC_BEGIN

ObjectDAG::ObjectDAG(const std::vector<RC<Object>> &objects)
    : ObjectDAG(objects.begin(), objects.end())
{
    
}

ObjectDAG::ObjectDAG(const RC<Object> &object)
    : ObjectDAG(&object, &object + 1)
{
    
}

const std::vector<RC<Object>> &ObjectDAG::get_sorted_objects() const
{
    return sorted_;
}

void ObjectDAG::commit()
{
    for(auto &obj : sorted_)
        obj->commit();
}

void ObjectDAG::add(const RC<Object> &object, std::set<RC<Object>> &processed)
{
    if(processed.contains(object))
        return;
    for(auto &d : object->get_dependent_objects())
        add(d, processed);
    assert(!processed.contains(object));
    processed.insert(object);
    sorted_.push_back(object);
}

BTRC_END
