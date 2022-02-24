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
    std::map<RC<Object>, bool> need_commit;
    for(auto &obj : sorted_)
    {
        bool need = false;
        if(obj->need_commit())
            need = true;
        else
        {
            for(auto &d : obj->get_dependent_objects())
            {
                if(need_commit.at(d))
                {
                    need = true;
                    break;
                }
            }
        }
        if(need)
            obj->commit();
        need_commit.insert({ obj, need });
    }
}

bool ObjectDAG::need_recompile() const
{
    for(auto &obj : sorted_)
    {
        if(obj->need_recompile())
            return true;
    }
    return false;
}

void ObjectDAG::clear_recompile_flag()
{
    for(auto &obj : sorted_)
        obj->set_recompile(false);
}

void ObjectDAG::update_properties() const
{
    for(auto &obj : sorted_)
    {
        for(auto p : obj->get_properties())
            p->update();
    }
}

void ObjectDAG::add(const RC<Object> &object, std::set<RC<Object>> &processed)
{
    if(processed.contains(object))
        return;
    for(auto &d : object->get_dependent_objects())
        add(d, processed);
    assert(!processed_objects.contains(object));
    processed.insert(object);
    sorted_.push_back(object);
}

BTRC_END
