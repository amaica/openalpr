/*
 * Copyright (c) 2026 OpenALPR contributors.
 * Factory: selects enhancement strategy from runtime config (garagem vs default).
 */

#ifndef OPENALPR_IMAGE_ENHANCEMENT_STRATEGY_FACTORY_H
#define OPENALPR_IMAGE_ENHANCEMENT_STRATEGY_FACTORY_H

#include <memory>

namespace alpr
{

class Config;
class IImageEnhancementStrategy;

class ImageEnhancementStrategyFactory
{
public:
  /** Returns Garage strategy when scenario is garagem and enhancement is enabled; otherwise No-op. */
  static std::unique_ptr<IImageEnhancementStrategy> createForConfig(const Config* config);
};

} // namespace alpr

#endif
