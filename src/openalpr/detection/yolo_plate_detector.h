/*
 * YOLO-based plate detector using OpenCV DNN.
 * Provides a plugable alternative to the classic detectors.
 */

#pragma once

#include <opencv2/dnn.hpp>
#include "detector.h"

namespace alpr
{
  class YoloPlateDetector : public Detector
  {
  public:
    YoloPlateDetector(Config* config, PreWarp* prewarp, Detector* fallbackDetector);
    virtual ~YoloPlateDetector();

    std::vector<cv::Rect> find_plates(cv::Mat frame, cv::Size min_plate_size, cv::Size max_plate_size) override;

  private:
    cv::dnn::Net net;
    bool netLoaded;
    Detector* fallback;
    std::string backendName;

    std::vector<cv::Rect> parseOutput(const cv::Mat& output, int imgWidth, int imgHeight);
  };
}

