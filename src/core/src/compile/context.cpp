#include <stack>

#include <btrc/core/compile/context.h>
#include <btrc/core/utils/unreachable.h>

BTRC_CORE_BEGIN

namespace
{

    std::stack<CompileContext *> &global_ccs()
    {
        thread_local static std::stack<CompileContext *> ccs;
        return ccs;
    }

} // namespace anonymous

CompileContext *CompileContext::get_current_context()
{
    return global_ccs().top();
}

void CompileContext::push_context(CompileContext *context)
{
    global_ccs().push(context);
}

void CompileContext::pop_context()
{
    global_ccs().pop();
}

CompileContext::CompileContext(bool offline)
    : offline_(offline)
{
    
}

bool CompileContext::is_offline() const
{
    return offline_;
}

bool CompileContext::should_inline(const RC<const ObjectBase> &object) const
{
    switch(object->get_compile_option())
    {
    case ObjectBase::CompileOption::Auto:
        return offline_;
    case ObjectBase::CompileOption::Separate:
        return false;
    case ObjectBase::CompileOption::Inlined:
        return true;
    }
    unreachable();
}

std::vector<std::string_view> CompileContext::generate_separate_codes() const
{
    std::vector<std::string_view> result;
    for(auto &[_, object_record] : object_records_)
    {
        if(object_record.cached_code.empty())
        {
            cuj::PTXGenerator gen;
            gen.set_options(cuj::Options{
                .opt_level        = cuj::OptimizationLevel::O3,
                .fast_math        = true,
                .approx_math_func = true
            });
            gen.generate(object_record.cuj_module);
            object_record.cached_code = gen.get_ptx();
        }
        result.push_back(object_record.cached_code);
    }
    return result;
}

BTRC_CORE_END
