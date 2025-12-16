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
    // Classic detectors only; YOLO/ORT removed
    if (config->debugDetector)
      std::cout << "[detector] using classic detector" << std::endl;
    return createClassicDetector(config, prewarp);
  }

}

