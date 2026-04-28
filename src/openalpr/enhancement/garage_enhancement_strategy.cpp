/*
 * Copyright (c) 2026 OpenALPR contributors.
 */

#include "garage_enhancement_strategy.h"

#include <opencv2/imgproc.hpp>

namespace alpr
{

cv::Mat GarageEnhancementStrategy::enhance(const cv::Mat& input)
{
  if (input.empty())
    return input;

  cv::Mat gray;
  if (input.channels() == 1)
    gray = input;
  else if (input.channels() == 3)
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  else if (input.channels() == 4)
    cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
  else
    return input.clone();

  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(kClaheClipLimit, cv::Size(kClaheTile, kClaheTile));
  cv::Mat afterClahe;
  clahe->apply(gray, afterClahe);

  cv::Mat denoised;
  cv::bilateralFilter(afterClahe, denoised, kBilateralD, kBilateralSigmaColor, kBilateralSigmaSpace);

  cv::Mat tmp16;
  denoised.convertTo(tmp16, CV_16S);
  cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0.f, -1.f, 0.f, -1.f, 5.f, -1.f, 0.f, -1.f, 0.f);
  cv::filter2D(tmp16, tmp16, CV_16S, kernel);

  cv::Mat sharpened;
  cv::convertScaleAbs(tmp16, sharpened);
  return sharpened;
}

} // namespace alpr
