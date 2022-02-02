#include <stack>

#include <btrc/core/compile/property_manager.h>

BTRC_CORE_BEGIN

namespace
{

    std::stack<PropertyManager*> &global_prop_mgr()
    {
        thread_local static std::stack<PropertyManager*> result;
        return result;
    }

} // namespace anonymous

PropertyManager *PropertyManager::get_current_manager()
{
    return global_prop_mgr().top();
}

void PropertyManager::push_manager(PropertyManager *manager)
{
    global_prop_mgr().push(manager);
}

void PropertyManager::pop_manager()
{
    global_prop_mgr().pop();
}

#define BTRC_PROPERTY_TYPE(X) newRC<PropertyPool<X>>()
#define BTRC_PROPERTY_TYPE_SEPERATOR ,

PropertyManager::PropertyManager()
    : pools_{
#include <btrc/core/compile/property_type_list.txt>
    }
{

}

PropertyManager::PropertyManager(PropertyManager &&other) noexcept
    : PropertyManager()
{
    swap(other);
}

PropertyManager &PropertyManager::operator=(PropertyManager &&other) noexcept
{
    swap(other);
    return *this;
}

void PropertyManager::swap(PropertyManager &other) noexcept
{
    pools_.swap(other.pools_);
}

BTRC_CORE_END
