#include <stack>

#include <btrc/core/context.h>
#include <btrc/utils/unreachable.h>

BTRC_BEGIN

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

BTRC_END
