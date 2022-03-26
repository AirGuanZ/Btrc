#include <btrc/factory/post_processor.h>

BTRC_FACTORY_BEGIN

std::vector<RC<PostProcessor>> parse_post_processors(const RC<const Node> &node, Context &context)
{
    if(!node)
        return {};

    std::vector<RC<PostProcessor>> ret;
    if(auto grp = node->as_group())
    {
        ret.push_back(context.create<PostProcessor>(node));
        return ret;
    }

    if(auto arr = node->as_array())
    {
        for(size_t i = 0; i < arr->get_size(); ++i)
            ret.push_back(context.create<PostProcessor>(arr->get_element(i)));
        return ret;
    }

    throw BtrcException("parse_post_processors: array or group is expected");
}

BTRC_FACTORY_END
