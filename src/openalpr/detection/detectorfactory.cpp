#include "detectorfactory.h"
#include "detectormorph.h"
#include "detectorocl.h"

namespace alpr
{
  namespace {
    Detector* createClassicDetector(Config* config, PreWarp* prewarp)
    {
      if (config->detector == DETECTOR_LBP_CPU)
      {
        return new DetectorCPU(config, prewarp);
      }
      else if (config->detector == DETECTOR_LBP_GPU)
      {
        #ifdef COMPILE_GPU
        return new DetectorCUDA(config, prewarp);
        #else
        std::cerr << "Error: GPU detector requested, but GPU extensions are not compiled.  "
                  << "Add COMPILE_GPU=1 to the compiler definitions to enable GPU compilation."
                  << std::endl;
        return new DetectorCPU(config, prewarp);
        #endif
      }
      else if (config->detector == DETECTOR_LBP_OPENCL)
      {
        #if OPENCV_MAJOR_VERSION == 3
        return new DetectorOCL(config, prewarp);
        #else
        std::cerr << "Error: OpenCL acceleration requires OpenCV 3.0.  " << std::endl;
        return new DetectorCPU(config, prewarp);
        #endif
      }
      else if (config->detector == DETECTOR_MORPH_CPU)
      {
        return new DetectorMorph(config, prewarp);
      }
      else
      {
        std::cerr << "Unknown detector requested.  Using LBP CPU" << std::endl;
        return new DetectorCPU(config, prewarp);
      }
    }
  }

  Detector* createDetector(Config* config, PreWarp* prewarp)
  {
    // auto mode: prefer yolo if model path set, else classic
    if (config->detectorType == "auto")
    {
      if (config->yoloModelPath.length() > 0)
      {
        Detector* classicFallback = NULL;
        if (config->detectorFallbackClassic)
          classicFallback = createClassicDetector(config, prewarp);

        Detector* yolo = new YoloPlateDetector(config, prewarp, classicFallback);
        if (yolo != NULL)
          return yolo;
        if (config->debugDetector)
          std::cout << "[detector] auto mode: YOLO init failed, using classic" << std::endl;
        return classicFallback != NULL ? classicFallback : createClassicDetector(config, prewarp);
      }
      if (config->debugDetector)
        std::cout << "[detector] auto mode: no YOLO model, using classic" << std::endl;
      return createClassicDetector(config, prewarp);
    }

    if (config->detectorType == "yolo")
    {
      Detector* classicFallback = NULL;
      if (config->detectorFallbackClassic)
        classicFallback = createClassicDetector(config, prewarp);

      if (config->yoloModelPath.length() == 0)
      {
        std::cerr << "YOLO detector selected but yolo_model_path is empty. Falling back to classic." << std::endl;
        if (classicFallback != NULL)
          return classicFallback;
      }
      if (config->debugDetector)
        std::cout << "Using YOLO detector" << std::endl;
      return new YoloPlateDetector(config, prewarp, classicFallback);
    }

    if (config->debugDetector)
      std::cout << "Using classic detector" << std::endl;
    return createClassicDetector(config, prewarp);
  }

}

