#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "openalpr/alpr.h"

using namespace std;
using namespace alpr;

static bool fileExists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static std::string trim(const std::string& s) {
  size_t a = 0;
  while (a < s.size() && isspace(static_cast<unsigned char>(s[a]))) a++;
  size_t b = s.size();
  while (b > a && isspace(static_cast<unsigned char>(s[b-1]))) b--;
  return s.substr(a, b - a);
}

static std::string jsonEscape(const std::string& s) {
  std::ostringstream o;
  for (char c : s) {
    switch (c) {
      case '\\': o << "\\\\"; break;
      case '"':  o << "\\\""; break;
      case '\n': o << "\\n"; break;
      case '\r': o << "\\r"; break;
      case '\t': o << "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) o << "\\u" << std::hex << (int)c;
        else o << c;
    }
  }
  return o.str();
}

static void printErrJson(const std::string& err, int code, const std::string& processedPath) {
  // same schema as garage_alpr.sh --json errors
  std::cout.flush();
  std::cerr
      << "{\"error\":\"" << jsonEscape(err) << "\","
      << "\"code\":" << code << ","
      << "\"processed_path\":\"" << jsonEscape(processedPath) << "\"}"
      << std::endl;
  std::cerr.flush();
}

struct ScopedSilence {
  std::ofstream nullOut;
  std::streambuf* coutBuf = nullptr;
  std::streambuf* cerrBuf = nullptr;
  bool active = false;

  ScopedSilence(bool enable) {
    if (!enable) return;
    nullOut.open("/dev/null");
    if (!nullOut.is_open()) return;
    coutBuf = std::cout.rdbuf(nullOut.rdbuf());
    cerrBuf = std::cerr.rdbuf(nullOut.rdbuf());
    active = true;
  }

  ~ScopedSilence() {
    if (!active) return;
    std::cout.rdbuf(coutBuf);
    std::cerr.rdbuf(cerrBuf);
  }
};

static void printOkJson(const std::string& plate, double conf, const std::string& used,
                        const std::string& processedPath, const std::string& sourcePath) {
  // same schema as garage_alpr.sh --json success
  std::cout
      << "{\"plate\":\"" << jsonEscape(plate) << "\","
      << "\"confidence\":" << conf << ","
      << "\"used\":\"" << jsonEscape(used) << "\","
      << "\"processed_path\":\"" << jsonEscape(processedPath) << "\","
      << "\"source_path\":\"" << jsonEscape(sourcePath) << "\"}"
      << std::endl;
  std::cout.flush();
}

static void printCmdOkJson(const std::string& cmd) {
  std::cout << "{\"ok\":true,\"cmd\":\"" << jsonEscape(cmd) << "\"}" << std::endl;
  std::cout.flush();
}

static void printStatsJson(time_t startedAt, uint64_t requests, uint64_t okCount, uint64_t errCount) {
  time_t now = time(nullptr);
  long uptime = (now >= startedAt) ? static_cast<long>(now - startedAt) : 0;
  std::cout
      << "{\"ok\":true,"
      << "\"cmd\":\"stats\","
      << "\"uptime_sec\":" << uptime << ","
      << "\"requests\":" << requests << ","
      << "\"ok_count\":" << okCount << ","
      << "\"error_count\":" << errCount << "}"
      << std::endl;
  std::cout.flush();
}

static bool extractBest(const AlprResults& results, std::string& plateOut, double& confOut) {
  if (results.plates.size() == 0) return false;
  plateOut = results.plates[0].bestPlate.characters;
  confOut = results.plates[0].bestPlate.overall_confidence;
  return !plateOut.empty();
}

static std::string runEnhancerAlprMaxExternal(const std::string& enhBin, const std::string& inputPath) {
  // Executes: enhBin --alpr-max inputPath
  // Parses stdout line: "Done: <path>"
  std::string cmd = "\"" + enhBin + "\" --alpr-max \"" + inputPath + "\"";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return "";
  char buf[4096];
  std::string out;
  while (fgets(buf, sizeof(buf), pipe)) out += buf;
  int rc = pclose(pipe);
  (void)rc;

  std::istringstream iss(out);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.rfind("Done: ", 0) == 0) {
      std::string path = trim(line.substr(strlen("Done: ")));
      return path;
    }
  }
  return "";
}

static bool enhanceAlprMaxInternal(const std::string& inputPath, const std::string& outputPath) {
  cv::Mat bgr = cv::imread(inputPath, cv::IMREAD_COLOR);
  if (bgr.empty()) return false;

  // CLAHE on L channel (LAB)
  cv::Mat lab;
  cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
  std::vector<cv::Mat> ch;
  cv::split(lab, ch);
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
  clahe->apply(ch[0], ch[0]);
  cv::merge(ch, lab);
  cv::Mat claheBgr;
  cv::cvtColor(lab, claheBgr, cv::COLOR_Lab2BGR);

  // Bilateral filter (preserve edges)
  cv::Mat smooth;
  cv::bilateralFilter(claheBgr, smooth, /*d*/ 7, /*sigmaColor*/ 50, /*sigmaSpace*/ 50);

  // Mild unsharp mask
  cv::Mat blur;
  cv::GaussianBlur(smooth, blur, cv::Size(0, 0), 1.0);
  cv::Mat sharp;
  cv::addWeighted(smooth, 1.25, blur, -0.25, 0.0, sharp);

  return cv::imwrite(outputPath, sharp);
}

static std::string mkTempDir() {
  std::string tmpl = "/tmp/garage_alprd.XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  char* res = mkdtemp(buf.data());
  if (!res) return "";
  return std::string(res);
}

static bool copyFile(const std::string& src, const std::string& dst) {
  FILE* in = fopen(src.c_str(), "rb");
  if (!in) return false;
  FILE* out = fopen(dst.c_str(), "wb");
  if (!out) { fclose(in); return false; }
  char buf[8192];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
    if (fwrite(buf, 1, n, out) != n) { fclose(in); fclose(out); return false; }
  }
  fclose(in);
  fclose(out);
  return true;
}

static off_t fileSize(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) return -1;
  return st.st_size;
}

static void waitForStableFile(const std::string& path, int tries = 10, int sleepMs = 20) {
  off_t last = -2;
  for (int i = 0; i < tries; i++) {
    off_t sz = fileSize(path);
    if (sz > 0 && sz == last) return;
    last = sz;
    usleep(sleepMs * 1000);
  }
}

static std::string basenameOnly(const std::string& path) {
  size_t pos = path.find_last_of("/\\");
  return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

static void rmrf(const std::string& dir) {
  if (dir.empty()) return;
  std::string cmd = "rm -rf -- \"" + dir + "\"";
  system(cmd.c_str());
}

static void usage() {
  std::cerr <<
    "Usage:\n"
    "  garage_alpr_daemon [--country br] [--config <openalpr.conf>] [--enh-bin <mini_enhancer>]\\n"
    "                    [--keep-temp-on-fail|--no-keep-temp-on-fail] [--debug] [--no-enhance]\\n"
    "                    [--external-enhancer] [--no-quiet]\\n"
    "\\n"
    "Protocol (stdin): one image path per line; output is one JSON line per request on stdout.\\n"
    "Errors are one JSON line on stderr (same schema as garage_alpr.sh --json).\\n"
    "Commands: __ping, __stats, __quit\\n"
    "Send EOF (Ctrl+D) to exit.\\n";
}

int main(int argc, char** argv) {
  std::string country = "br";
  std::string configFile = "./config/openalpr.conf.defaults";
  std::string enhBin = "./mini_enhancer/build/mini_enhancer";
  bool keepTempOnFail = true;
  bool debug = false;
  bool disableEnhance = false;
  bool useExternalEnhancer = false;
  bool quietProtocol = true;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto eat = [&](std::string& target) {
      if (i + 1 >= argc) { usage(); return false; }
      target = argv[++i];
      return true;
    };
    if (a == "-h" || a == "--help") { usage(); return 0; }
    if (a == "--country" || a == "-c") { if (!eat(country)) return 1; continue; }
    if (a == "--config") { if (!eat(configFile)) return 1; continue; }
    if (a == "--enh-bin") { if (!eat(enhBin)) return 1; continue; }
    if (a == "--keep-temp-on-fail") { keepTempOnFail = true; continue; }
    if (a == "--no-keep-temp-on-fail") { keepTempOnFail = false; continue; }
    if (a == "--debug") { debug = true; continue; }
    if (a == "--no-enhance") { disableEnhance = true; continue; }
    if (a == "--external-enhancer") { useExternalEnhancer = true; continue; }
    if (a == "--no-quiet") { quietProtocol = false; continue; }
    usage();
    return 1;
  }

  if (!fileExists(configFile)) {
    printErrJson("CONFIG_NOT_FOUND", 2, configFile);
    return 2;
  }
  if (useExternalEnhancer && access(enhBin.c_str(), X_OK) != 0) {
    printErrJson("ENH_BIN_NOT_EXECUTABLE", 5, enhBin);
    return 5;
  }

  // The OpenALPR config loader prints to stdout/stderr in this fork.
  // Silence library noise so our protocol stays 1 JSON per line.
  Alpr* alprOrigPtr = nullptr;
  Alpr* alprEnhPtr = nullptr;
  {
    ScopedSilence sil(true);
    alprOrigPtr = new Alpr(country, configFile);
    alprEnhPtr = new Alpr(country, configFile);
  }
  if (alprOrigPtr == nullptr || alprEnhPtr == nullptr || !alprOrigPtr->isLoaded() || !alprEnhPtr->isLoaded()) {
    printErrJson("ALPR_LOAD_FAILED", 5, configFile);
    delete alprOrigPtr;
    delete alprEnhPtr;
    return 5;
  }
  Alpr& alprOrig = *alprOrigPtr;
  Alpr& alprEnh = *alprEnhPtr;

  const std::regex legacyRe("^[A-Z]{3}[0-9]{4}$");

  time_t startedAt = time(nullptr);
  uint64_t requests = 0;
  uint64_t okCount = 0;
  uint64_t errCount = 0;

  std::string line;
  while (std::getline(std::cin, line)) {
    std::string srcPath = trim(line);
    if (srcPath.empty()) continue;

    // Command protocol
    if (srcPath == "__ping") { printCmdOkJson("ping"); continue; }
    if (srcPath == "__stats") { printStatsJson(startedAt, requests, okCount, errCount); continue; }
    if (srcPath == "__quit") { printCmdOkJson("quit"); break; }

    requests++;

    if (!fileExists(srcPath)) {
      printErrJson("FILE_NOT_FOUND", 2, srcPath);
      errCount++;
      continue;
    }

    std::string bestPlate;
    double bestConf = 0.0;
    std::string usedMode = "original";
    std::string processedPath = srcPath;

    size_t origPlates = 0;
    {
      ScopedSilence sil(quietProtocol);
      AlprResults r = alprOrig.recognize(srcPath);
      origPlates = r.plates.size();
      extractBest(r, bestPlate, bestConf);
    }
    if (debug) std::cerr << "[dbg] original plates=" << origPlates << " plate=" << bestPlate << " conf=" << bestConf << std::endl;

    bool needEnhance = true;
    if (!bestPlate.empty() && std::regex_match(bestPlate, legacyRe)) {
      needEnhance = false;
    }
    if (disableEnhance) needEnhance = false;

    if (needEnhance) {
      std::string tmp = mkTempDir();
      if (tmp.empty()) {
        printErrJson("TMPDIR_FAILED", 3, srcPath);
        errCount++;
        continue;
      }
      std::string tmpIn = tmp + "/" + basenameOnly(srcPath);
      if (!copyFile(srcPath, tmpIn)) {
        rmrf(tmp);
        printErrJson("COPY_FAILED", 3, srcPath);
        errCount++;
        continue;
      }
      std::string enhOut;
      if (useExternalEnhancer) {
        enhOut = runEnhancerAlprMaxExternal(enhBin, tmpIn);
      } else {
        enhOut = tmp + "/enhanced_alpr_max.png";
        if (!enhanceAlprMaxInternal(tmpIn, enhOut)) enhOut.clear();
      }
      if (enhOut.empty() || !fileExists(enhOut)) {
        rmrf(tmp);
        printErrJson("ENHANCEMENT_FAILED", 3, srcPath);
        errCount++;
        continue;
      }

      // Copy to a stable filename before recognition (avoids any lingering write/rename edge cases).
      std::string stableEnh = tmp + "/enhanced_stable.png";
      if (!copyFile(enhOut, stableEnh)) {
        rmrf(tmp);
        printErrJson("COPY_FAILED", 3, enhOut);
        errCount++;
        continue;
      }

      processedPath = stableEnh;
      usedMode = "enhanced";
      bestPlate.clear();
      bestConf = 0.0;

      // Ensure the enhanced file is fully written before OpenCV reads it.
      waitForStableFile(stableEnh);

      size_t enhPlates = 0;
      for (int attempt = 0; attempt < 3; attempt++) {
        ScopedSilence sil(quietProtocol);
        AlprResults r = alprEnh.recognize(stableEnh);
        enhPlates = r.plates.size();
        extractBest(r, bestPlate, bestConf);
        if (!bestPlate.empty()) break;
        usleep(30 * 1000);
      }
      if (debug) std::cerr << "[dbg] enhanced plates=" << enhPlates << " plate=" << bestPlate << " conf=" << bestConf << " path=" << stableEnh << std::endl;
      if (!bestPlate.empty()) {
        rmrf(tmp);
      } else if (!keepTempOnFail) {
        rmrf(tmp);
      }
    }

    if (bestPlate.empty()) {
      printErrJson("NO_PLATE", 4, processedPath);
      errCount++;
      continue;
    }

    printOkJson(bestPlate, bestConf, usedMode, processedPath, srcPath);
    okCount++;
  }

  delete alprOrigPtr;
  delete alprEnhPtr;
  return 0;
}

