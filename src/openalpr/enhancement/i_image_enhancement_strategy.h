/*
 * Copyright (c) 2026 OpenALPR contributors.
 * Open source Automated License Plate Recognition
 *
 * Strategy interface for optional plate crop enhancement (e.g. garagem mode).
 */

#ifndef OPENALPR_I_IMAGE_ENHANCEMENT_STRATEGY_H
#define OPENALPR_I_IMAGE_ENHANCEMENT_STRATEGY_H

#include <opencv2/core.hpp>

namespace alpr
{

class IImageEnhancementStrategy
{
public:
  virtual cv::Mat enhance(const cv::Mat& input) = 0;
  virtual ~IImageEnhancementStrategy() = default;
};

} // namespace alpr

#endif
