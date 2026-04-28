/*
 * Copyright (c) 2026 OpenALPR contributors.
 */

#include "plate_image_enhancer.h"

#include "image_enhancement_strategy_factory.h"
#include "i_image_enhancement_strategy.h"

namespace alpr
{

PlateImageEnhancer::PlateImageEnhancer(std::unique_ptr<IImageEnhancementStrategy> strategy)
    : strategy_(std::move(strategy))
{
}

PlateImageEnhancer PlateImageEnhancer::fromConfig(const Config* config)
{
  return PlateImageEnhancer(ImageEnhancementStrategyFactory::createForConfig(config));
}

void PlateImageEnhancer::applyBeforeOcr(cv::Mat& plateGray)
{
  if (!strategy_ || plateGray.empty())
    return;
  plateGray = strategy_->enhance(plateGray);
}

} // namespace alpr
