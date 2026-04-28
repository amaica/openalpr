/*
 * Copyright (c) 2026 OpenALPR contributors.
 * Garagem-mode plate crop enhancement: CLAHE + bilateral + sharpen (OpenCV only).
 */

#ifndef OPENALPR_GARAGE_ENHANCEMENT_STRATEGY_H
#define OPENALPR_GARAGE_ENHANCEMENT_STRATEGY_H

#include "i_image_enhancement_strategy.h"

namespace alpr
{

class GarageEnhancementStrategy : public IImageEnhancementStrategy
{
public:
  cv::Mat enhance(const cv::Mat& input) override;

private:
  static constexpr double kClaheClipLimit = 2.0;
  static constexpr int kClaheTile = 8;
  static constexpr int kBilateralD = 5;
  static constexpr double kBilateralSigmaColor = 45.0;
  static constexpr double kBilateralSigmaSpace = 45.0;
};

} // namespace alpr

#endif
