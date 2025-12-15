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
    : Detector(config, prewarp), netLoaded(false), ortLoaded(false), fallback(fallbackDetector)
  {
#ifdef ENABLE_ORT_YOLO
    if (config->yoloModelPath.length() > 0 && fileExists(config->yoloModelPath.c_str()))
    {
      ortInputWidth = config->yoloInputWidth;
      ortInputHeight = config->yoloInputHeight;
      ortLoaded = initOrtSession(config->yoloModelPath);
    }
#endif

    if (!ortLoaded)
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
  }

  YoloPlateDetector::~YoloPlateDetector()
  {
    if (fallback != NULL)
      delete fallback;
  }

#ifdef ENABLE_ORT_YOLO
  bool YoloPlateDetector::initOrtSession(const std::string& modelPath)
  {
    try
    {
      ortEnv = std::unique_ptr<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "alpr-yolo"));

      ortSessionOpts.reset(new Ort::SessionOptions());
      ortSessionOpts->SetIntraOpNumThreads(1);
      ortSessionOpts->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

      ortSession.reset(new Ort::Session(*ortEnv, modelPath.c_str(), *ortSessionOpts));

      size_t inputCount = ortSession->GetInputCount();
      size_t outputCount = ortSession->GetOutputCount();
      Ort::AllocatorWithDefaultOptions allocator;
      for (size_t i = 0; i < inputCount; i++)
      {
        auto nameAlloc = ortSession->GetInputNameAllocated(i, allocator);
        ortInputNameStorage.emplace_back(nameAlloc.get());
        ortInputNames.push_back(ortInputNameStorage.back().c_str());
      }
      for (size_t i = 0; i < outputCount; i++)
      {
        auto nameAlloc = ortSession->GetOutputNameAllocated(i, allocator);
        ortOutputNameStorage.emplace_back(nameAlloc.get());
        ortOutputNames.push_back(ortOutputNameStorage.back().c_str());
      }

      const char* ver = OrtGetApiBase()->GetVersionString();
      std::vector<std::string> providers;
      for (auto& p : Ort::GetAvailableProviders())
        providers.push_back(p);

      std::cout << "[yolo][ort] version=" << (ver ? ver : "unknown")
                << " providers=";
      for (size_t i = 0; i < providers.size(); i++)
      {
        if (i) std::cout << ",";
        std::cout << providers[i];
      }
      std::cout << " model_path=" << modelPath << " load_ok=1" << std::endl;
      backendName = "ort_yolo";
      return true;
    }
    catch (const Ort::Exception& ex)
    {
      std::cerr << "[yolo][ort] load_ok=0 error=" << ex.what() << std::endl;
      return false;
    }
  }

  std::vector<cv::Rect> YoloPlateDetector::decodeOrtOutput(const float* data, const std::vector<int64_t>& shape,
                                          int origW, int origH)
  {
    std::vector<cv::Rect> boxes;
    if (shape.size() < 2)
      return boxes;

    int64_t dim0 = shape[0];
    int64_t dim1 = shape[1];
    int64_t dim2 = shape.size() > 2 ? shape[2] : 0;

    int64_t numBoxes = 0;
    int64_t attrs = 0;
    bool transposed = false;
    if (shape.size() == 3)
    {
      // Handle [1, 84, 8400] or [1, 8400, 84]
      if (dim1 < dim2) { attrs = dim1; numBoxes = dim2; transposed = true; }
      else { attrs = dim2; numBoxes = dim1; transposed = false; }
    }
    else
    {
      numBoxes = dim0;
      attrs = dim1;
    }

    // Letterbox scale reverse
    float scale = std::min((float)ortInputWidth / (float)origW, (float)ortInputHeight / (float)origH);
    float padX = ((float)ortInputWidth - (float)origW * scale) / 2.0f;
    float padY = ((float)ortInputHeight - (float)origH * scale) / 2.0f;

    auto clamp = [](float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); };

    auto getVal = [&](int64_t boxIdx, int64_t attrIdx) -> float {
      if (transposed)
        return data[attrIdx * numBoxes + boxIdx];
      return data[boxIdx * attrs + attrIdx];
    };

    for (int64_t i = 0; i < numBoxes; i++)
    {
      float cx = getVal(i, 0);
      float cy = getVal(i, 1);
      float w  = getVal(i, 2);
      float h  = getVal(i, 3);
      float obj = (attrs > 4) ? getVal(i, 4) : 1.0f;

      float bestClsScore = 0.0f;
      if (attrs > 5)
      {
        for (int64_t c = 5; c < attrs; c++)
        {
          float s = getVal(i, c);
          if (s > bestClsScore) bestClsScore = s;
        }
      }
      else
      {
        bestClsScore = 1.0f;
      }

      float score = obj * bestClsScore;
      if (score < config->yoloConfThreshold)
        continue;

      float x0 = (cx - w / 2.0f - padX) / scale;
      float y0 = (cy - h / 2.0f - padY) / scale;
      float ww = w / scale;
      float hh = h / scale;

      int x = (int)clamp(x0, 0.0f, (float)(origW - 1));
      int y = (int)clamp(y0, 0.0f, (float)(origH - 1));
      int width = (int)clamp(ww, 1.0f, (float)origW - x);
      int height = (int)clamp(hh, 1.0f, (float)origH - y);

      boxes.push_back(cv::Rect(x, y, width, height));
    }
    return boxes;
  }

  std::vector<cv::Rect> YoloPlateDetector::inferOrt(const cv::Mat& frame)
  {
    std::vector<cv::Rect> detections;
    try
    {
      static bool loggedBackend = false;
      if (!loggedBackend)
      {
        std::cout << "[detector] using=ort_yolo" << std::endl;
        loggedBackend = true;
      }

      cv::Mat bgr;
      if (frame.channels() == 1)
        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
      else
        bgr = frame;

      // Letterbox resize
      float scale = std::min((float)ortInputWidth / (float)bgr.cols, (float)ortInputHeight / (float)bgr.rows);
      int newW = (int)(bgr.cols * scale);
      int newH = (int)(bgr.rows * scale);
      cv::Mat resized;
      cv::resize(bgr, resized, cv::Size(newW, newH));

      cv::Mat canvas = cv::Mat::zeros(ortInputHeight, ortInputWidth, CV_8UC3);
      int dx = (ortInputWidth - newW) / 2;
      int dy = (ortInputHeight - newH) / 2;
      resized.copyTo(canvas(cv::Rect(dx, dy, newW, newH)));

      cv::Mat blob;
      canvas.convertTo(blob, CV_32F, 1.0 / 255.0);
      std::vector<int64_t> inputShape {1, 3, ortInputHeight, ortInputWidth};
      std::vector<float> inputData(inputShape[1]*inputShape[2]*inputShape[3]);

      // HWC to NCHW
      size_t idx = 0;
      for (int c = 0; c < 3; c++)
      {
        for (int y = 0; y < ortInputHeight; y++)
        {
          for (int x = 0; x < ortInputWidth; x++)
          {
            inputData[idx++] = blob.at<cv::Vec3f>(y,x)[c];
          }
        }
      }

      Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
      Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memInfo, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());

      auto output = ortSession->Run(Ort::RunOptions{nullptr}, ortInputNames.data(), &inputTensor, 1, ortOutputNames.data(), ortOutputNames.size());
      for (auto& out : output)
      {
        float* outData = out.GetTensorMutableData<float>();
        auto outShape = out.GetTensorTypeAndShapeInfo().GetShape();
        std::vector<cv::Rect> parsed = decodeOrtOutput(outData, outShape, frame.cols, frame.rows);
        detections.insert(detections.end(), parsed.begin(), parsed.end());
      }

      // NMS (using OpenCV for convenience)
      std::vector<int> indices;
      std::vector<float> scores(detections.size(), 1.0f);
      std::vector<cv::Rect> finalBoxes;
      if (!detections.empty())
      {
        cv::dnn::NMSBoxes(detections, scores, config->yoloConfThreshold, config->yoloNmsThreshold, indices);
        for (size_t i = 0; i < indices.size(); i++)
        {
          cv::Rect r = expandRect(detections[indices[i]], 0, 0, frame.cols, frame.rows);
          if (r.width >= 1 && r.height >= 1)
            finalBoxes.push_back(r);
        }
      }

      if (config->debugDetector)
        std::cout << "[detector] using=ort_yolo parsed=" << finalBoxes.size() << std::endl;

      // Log per-bbox when ORT is active
      for (const auto& r : finalBoxes)
      {
        std::cout << "yolo_bbox=" << r.x << "," << r.y << "," << r.width << "," << r.height << std::endl;
      }

      return finalBoxes;
    }
    catch (const Ort::Exception& ex)
    {
      std::cerr << "[yolo][ort] inference error=" << ex.what() << ". Falling back to OpenCV/Classic." << std::endl;
      ortLoaded = false;
      return detections;
    }
  }
#endif

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

    if (ortLoaded)
    {
      std::vector<cv::Rect> ortBoxes = inferOrt(frame);
      detections.insert(detections.end(), ortBoxes.begin(), ortBoxes.end());
      if (config->debugDetector)
        std::cout << "[detector] using=ort_yolo boxes=" << detections.size() << std::endl;
    }

    if (netLoaded && detections.empty())
    {
      try
      {
        static bool loggedCvBackend = false;
        if (!loggedCvBackend)
        {
          std::cout << "[detector] using=opencv_yolo backend=" << backendName << std::endl;
          loggedCvBackend = true;
        }
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

