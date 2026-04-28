/*
 * Copyright (c) 2026 OpenALPR contributors.
 *
 * Encapsulates plate-crop enhancement: delegates to an IImageEnhancementStrategy
 * chosen by ImageEnhancementStrategyFactory. The recognition pipeline only calls
 * applyBeforeOcr — no OpenCV pipeline logic here.
 *
 * Example:
 *   PlateImageEnhancer enhancer(PlateImageEnhancer::fromConfig(config));
 *   enhancer.applyBeforeOcr(pipeline_data.crop_gray);
 */

#ifndef OPENALPR_PLATE_IMAGE_ENHANCER_H
#define OPENALPR_PLATE_IMAGE_ENHANCER_H

#include <memory>
#include <opencv2/core.hpp>

#include "i_image_enhancement_strategy.h"

namespace alpr
{

class Config;

class PlateImageEnhancer
{
public:
  /** Takes ownership of the strategy (typically from ImageEnhancementStrategyFactory). */
  explicit PlateImageEnhancer(std::unique_ptr<IImageEnhancementStrategy> strategy);

  /** Builds enhancer with the correct strategy for the current config. */
  static PlateImageEnhancer fromConfig(const Config* config);

  /**
   * Replaces \p plateGray with the strategy output when garagem enhancement is active;
   * with No-op strategy this is a cheap assignment of the same matrix header.
   */
  void applyBeforeOcr(cv::Mat& plateGray);

private:
  std::unique_ptr<IImageEnhancementStrategy> strategy_;
};

} // namespace alpr

#endif
