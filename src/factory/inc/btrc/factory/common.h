#pragma once

#include <btrc/core/common.h>

#define BTRC_FACTORY_BEGIN namespace btrc::factory {
#define BTRC_FACTORY_END   }

BTRC_FACTORY_BEGIN

using core::RC;
using core::newRC;

using core::Box;
using core::newBox;

class BtrcFactoryException : public core::BtrcException
{
public:

    using BtrcException::BtrcException;
};

BTRC_FACTORY_END
