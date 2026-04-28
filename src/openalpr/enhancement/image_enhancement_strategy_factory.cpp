/*
 * Copyright (c) 2026 OpenALPR contributors.
 */

#include "image_enhancement_strategy_factory.h"

#include "config.h"
#include "garage_enhancement_strategy.h"
#include "no_op_image_enhancement_strategy.h"

namespace alpr
{

std::unique_ptr<IImageEnhancementStrategy> ImageEnhancementStrategyFactory::createForConfig(const Config* config)
{
  if (config != nullptr && config->scenario == "garagem" && config->garagePlateEnhancement)
    return std::make_unique<GarageEnhancementStrategy>();
  return std::make_unique<NoOpImageEnhancementStrategy>();
}

} // namespace alpr
