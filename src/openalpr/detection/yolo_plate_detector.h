/*
 * YOLO-based plate detector using OpenCV DNN.
 * Provides a plugable alternative to the classic detectors.
 */

#pragma once

#include <opencv2/dnn.hpp>
#include <memory>
#ifdef ENABLE_ORT_YOLO
#include <onnxruntime_cxx_api.h>
#endif
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
    bool ortLoaded;
    Detector* fallback;
    std::string backendName;

#ifdef ENABLE_ORT_YOLO
    std::unique_ptr<Ort::Env> ortEnv;
    std::unique_ptr<Ort::SessionOptions> ortSessionOpts;
    std::unique_ptr<Ort::Session> ortSession;
    std::vector<std::string> ortInputNameStorage;
    std::vector<std::string> ortOutputNameStorage;
    std::vector<const char*> ortInputNames;
    std::vector<const char*> ortOutputNames;
    int ortInputWidth = 640;
    int ortInputHeight = 640;
#endif

    std::vector<cv::Rect> parseOutput(const cv::Mat& output, int imgWidth, int imgHeight);
#ifdef ENABLE_ORT_YOLO
    bool initOrtSession(const std::string& modelPath);
    std::vector<cv::Rect> inferOrt(const cv::Mat& frame);
    std::vector<cv::Rect> decodeOrtOutput(const float* data, const std::vector<int64_t>& shape,
                                          int origW, int origH);
#endif
  };
}

