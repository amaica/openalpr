/*
 * Copyright (c) 2026 OpenALPR contributors.
 * Highway / default path: no processing, zero overhead (shares input data).
 */

#ifndef OPENALPR_NO_OP_IMAGE_ENHANCEMENT_STRATEGY_H
#define OPENALPR_NO_OP_IMAGE_ENHANCEMENT_STRATEGY_H

#include "i_image_enhancement_strategy.h"

namespace alpr
{

class NoOpImageEnhancementStrategy : public IImageEnhancementStrategy
{
public:
  cv::Mat enhance(const cv::Mat& input) override { return input; }
};

} // namespace alpr

#endif
