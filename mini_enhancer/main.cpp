#include "image_processor.h"
#include "thread_pool.h"

#include <opencv2/core.hpp>

#include <iostream>
#include <mutex>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  cv::setNumThreads(1);

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [--alpr | --alpr-max] image1.jpg [image2.png ...]\n";
    std::cerr << "  --alpr      ALPR-safe: bilateral leve (recomendado por defeito).\n";
    std::cerr << "  --alpr-max  CLAHE + bilateral + unsharp leve (só se --alpr não bastar).\n";
    std::cerr << "  Saídas: *_enhanced_alpr.ext ou *_enhanced_alpr_max.ext\n";
    return 1;
  }

  EnhanceTarget target = EnhanceTarget::General;
  std::vector<std::string> paths;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--alpr-max")
      target = EnhanceTarget::AlprOcrMax;
    else if (a == "--alpr")
      target = EnhanceTarget::AlprOcr;
    else
      paths.push_back(a);
  }

  if (paths.empty()) {
    std::cerr << "Error: no image paths given.\n";
    return 1;
  }

  ThreadPool pool(2);
  std::mutex print_mtx;

  for (const std::string& path : paths) {
    pool.enqueue([path, target, &print_mtx]() {
      {
        std::lock_guard<std::mutex> lock(print_mtx);
        std::cout << "Processing: " << path << "\n";
      }

      const std::string out = enhanceImage(path, target);
      {
        std::lock_guard<std::mutex> lock(print_mtx);
        if (out.empty()) {
          std::cerr << "Error: failed to load or write: " << path << "\n";
          return;
        }
        std::cout << "Done: " << out << "\n";
      }
    });
  }

  pool.wait_all();
  return 0;
}
