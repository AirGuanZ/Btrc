#pragma once

#include <btrc/core/post_processor.h>
#include <btrc/factory/context.h>

BTRC_FACTORY_BEGIN

std::vector<RC<PostProcessor>> parse_post_processors(const RC<const Node> &node, Context &context);

BTRC_FACTORY_END
