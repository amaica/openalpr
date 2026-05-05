#include "image_processor.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>

namespace fs = std::filesystem;

static std::string writeOutput(const fs::path& in, const std::string& stem_suffix,
                               const cv::Mat& out) {
  const fs::path parent = in.parent_path().empty() ? fs::path(".") : in.parent_path();
  const fs::path output =
      parent / (in.stem().string() + stem_suffix + in.extension().string());
  if (!cv::imwrite(output.string(), out))
    return "";
  return fs::absolute(output).string();
}

std::string enhanceImage(const std::string& inputPath, EnhanceTarget target) {
  cv::setNumThreads(1);

  cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
  if (src.empty())
    return "";

  const fs::path in(inputPath);
  cv::Mat out;

  if (target == EnhanceTarget::AlprOcr) {
    // Conservador: só suavização preservando borda — evita “colar” dígitos (ex.: 1.png).
    cv::bilateralFilter(src, out, 5, 28, 28);
    return writeOutput(in, "_enhanced_alpr", out);
  }

  if (target == EnhanceTarget::AlprOcrMax) {
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch(3);
    cv::split(lab, ch);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(ch[0], ch[0]);
    cv::merge(ch, lab);
    cv::Mat after_l;
    cv::cvtColor(lab, after_l, cv::COLOR_Lab2BGR);

    cv::Mat smooth;
    cv::bilateralFilter(after_l, smooth, 5, 40, 40);

    cv::Mat blur;
    cv::GaussianBlur(smooth, blur, cv::Size(0, 0), 0.45);
    cv::addWeighted(smooth, 1.06, blur, -0.06, 0.0, out);

    return writeOutput(in, "_enhanced_alpr_max", out);
  }

  cv::Mat blurred3;
  cv::blur(src, blurred3, cv::Size(3, 3));
  cv::Mat gauss;
  cv::GaussianBlur(blurred3, gauss, cv::Size(0, 0), 0.8);
  cv::addWeighted(src, 1.2, gauss, -0.2, 0.0, out);

  return writeOutput(in, "_enhanced", out);
}
