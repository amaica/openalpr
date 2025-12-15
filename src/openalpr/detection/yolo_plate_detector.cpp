#include "yolo_plate_detector.h"
#include "detectorfactory.h"
#include "utility.h"
#include "support/filesystem.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR >= 3
#include <opencv2/core/cuda.hpp>
#endif

namespace alpr
{

  YoloPlateDetector::YoloPlateDetector(Config* config, PreWarp* prewarp, Detector* fallbackDetector)
    : Detector(config, prewarp), netLoaded(false), fallback(fallbackDetector)
  {
    try
    {
      if (config->yoloModelPath.length() > 0 && fileExists(config->yoloModelPath.c_str()))
      {
        net = cv::dnn::readNetFromONNX(config->yoloModelPath);

        bool cudaAvailable = false;
#if CV_VERSION_MAJOR >= 3
        try
        {
          int count = cv::cuda::getCudaEnabledDeviceCount();
          cudaAvailable = (count > 0);
        }
        catch (...) { cudaAvailable = false; }
#endif
        try
        {
          if (cudaAvailable)
          {
            #if defined(DNN_BACKEND_CUDA) && defined(DNN_TARGET_CUDA_FP16)
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
            backendName = "CUDA_FP16";
            #else
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            backendName = "CPU";
            #endif
          }
          else
          {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            backendName = "CPU";
          }
        }
        catch (...)
        {
          net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
          net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
          backendName = "CPU";
        }

        netLoaded = true;
        std::cout << "[detector] YOLO loaded (" << config->yoloModelPath << "), backend=" << backendName << std::endl;
      }
      else
      {
        if (config->debugDetector)
          std::cerr << "[detector] YOLO model path missing. Falling back to classic detector." << std::endl;
      }
    }
    catch (const std::exception& ex)
    {
      std::cerr << "[detector] Error loading YOLO model: " << ex.what() << std::endl;
      netLoaded = false;
    }
  }

  YoloPlateDetector::~YoloPlateDetector()
  {
    if (fallback != NULL)
      delete fallback;
  }

  std::vector<cv::Rect> YoloPlateDetector::parseOutput(const cv::Mat& output, int imgWidth, int imgHeight)
  {
    std::vector<cv::Rect> boxes;
    // Expect shape [N, 5 + classes]; tolerate 2-D or 3-D output
    int rows = output.rows;
    int cols = output.cols;
    if (output.dims == 3)
    {
      rows = output.size[1];
      cols = output.size[2];
    }
    const float* data = (float*)output.data;
    for (int i = 0; i < rows; i++)
    {
      int idx = i * cols;
      float cx = data[idx + 0];
      float cy = data[idx + 1];
      float w = data[idx + 2];
      float h = data[idx + 3];
      float obj = data[idx + 4];

      float cls = 1.0f;
      if (cols > 5)
      {
        cls = 0.0f;
        for (int c = 5; c < cols; c++)
        {
          if (data[idx + c] > cls)
            cls = data[idx + c];
        }
      }
      float score = obj * cls;
      if (score < config->yoloConfThreshold)
        continue;

      int x = (int)((cx - w / 2.0f) * imgWidth);
      int y = (int)((cy - h / 2.0f) * imgHeight);
      int width = (int)(w * imgWidth);
      int height = (int)(h * imgHeight);

      cv::Rect rect(x, y, width, height);
      boxes.push_back(rect);
    }
    return boxes;
  }

  std::vector<cv::Rect> YoloPlateDetector::find_plates(cv::Mat frame, cv::Size min_plate_size, cv::Size max_plate_size)
  {
    std::vector<cv::Rect> detections;

    if (netLoaded)
    {
      try
      {
        cv::Mat inputBlob;
        cv::Mat bgr;
        if (frame.channels() == 1)
          cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
        else
          bgr = frame;

        cv::dnn::blobFromImage(bgr, inputBlob, 1.0/255.0, cv::Size(config->yoloInputWidth, config->yoloInputHeight), cv::Scalar(), true, false);
        net.setInput(inputBlob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs);
        for (size_t i = 0; i < outputs.size(); i++)
        {
          std::vector<cv::Rect> parsed = parseOutput(outputs[i], frame.cols, frame.rows);
          detections.insert(detections.end(), parsed.begin(), parsed.end());
        }

        // NMS
        std::vector<cv::Rect> nmsBoxes;
        std::vector<float> scores;
        for (size_t i = 0; i < detections.size(); i++)
          scores.push_back(1.0f);
        std::vector<int> indices;
        if (detections.size() > 0)
        {
          cv::dnn::NMSBoxes(detections, scores, config->yoloConfThreshold, config->yoloNmsThreshold, indices);
          for (size_t i = 0; i < indices.size(); i++)
          {
            cv::Rect r = detections[indices[i]];
            r = expandRect(r, 0, 0, frame.cols, frame.rows);
            if (r.width >= min_plate_size.width && r.height >= min_plate_size.height &&
                r.width <= max_plate_size.width && r.height <= max_plate_size.height)
              nmsBoxes.push_back(r);
          }
        }
        detections = nmsBoxes;
      }
      catch (const std::exception& ex)
      {
        std::cerr << "[detector] YOLO inference error: " << ex.what() << ". Falling back to classic." << std::endl;
        netLoaded = false;
      }
    }

    if ((int)detections.size() < config->yoloMinDetections && fallback != NULL && config->detectorFallbackClassic)
    {
      if (config->debugDetector)
        std::cout << "[detector] YOLO detections below threshold, using classic fallback" << std::endl;
      std::vector<cv::Rect> fallbackRects = fallback->find_plates(frame, min_plate_size, max_plate_size);
      detections.insert(detections.end(), fallbackRects.begin(), fallbackRects.end());
    }

    return detections;
  }

}

