#pragma once

#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

void register_builtin_creators(factory::Factory<Camera>       &factory);
void register_builtin_creators(factory::Factory<Geometry>     &factory);
void register_builtin_creators(factory::Factory<Light>        &factory);
void register_builtin_creators(factory::Factory<Material>     &factory);
void register_builtin_creators(factory::Factory<Medium>       &factory);
void register_builtin_creators(factory::Factory<Renderer>     &factory);
void register_builtin_creators(factory::Factory<Texture2D>    &factory);

void register_builtin_creators(factory::Context &context);

BTRC_BUILTIN_END
