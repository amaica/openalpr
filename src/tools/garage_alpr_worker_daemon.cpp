#include <iostream>
#include <sstream>
#include <string>
#include <regex>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>

#include "cli/recognition_worker_process.h"
#include "openalpr/cjson.h"

using namespace std;

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
      default: o << c;
    }
  }
  return o.str();
}

static void printErrJson(const std::string& err, int code, const std::string& processedPath) {
  std::cout.flush();
  std::cerr
      << "{\"error\":\"" << jsonEscape(err) << "\","
      << "\"code\":" << code << ","
      << "\"processed_path\":\"" << jsonEscape(processedPath) << "\"}"
      << std::endl;
  std::cerr.flush();
}

static void printOkJson(const std::string& plate, double conf, const std::string& used,
                        const std::string& processedPath, const std::string& sourcePath) {
  std::cout
      << "{\"plate\":\"" << jsonEscape(plate) << "\","
      << "\"confidence\":" << conf << ","
      << "\"used\":\"" << jsonEscape(used) << "\","
      << "\"processed_path\":\"" << jsonEscape(processedPath) << "\","
      << "\"source_path\":\"" << jsonEscape(sourcePath) << "\"}"
      << std::endl;
  std::cout.flush();
}

static bool extractBestPlateAndConf(const std::string& json, std::string& plate, double& conf) {
  plate.clear();
  conf = 0.0;
  cJSON* root = cJSON_Parse(json.c_str());
  if (!root) return false;
  cJSON* results = cJSON_GetObjectItem(root, "results");
  if (!results || results->type != cJSON_Array || cJSON_GetArraySize(results) == 0) {
    cJSON_Delete(root);
    return false;
  }
  cJSON* r0 = cJSON_GetArrayItem(results, 0);
  if (!r0) { cJSON_Delete(root); return false; }
  cJSON* plateObj = cJSON_GetObjectItem(r0, "plate");
  cJSON* confObj = cJSON_GetObjectItem(r0, "confidence");
  if (plateObj && plateObj->type == cJSON_String && plateObj->valuestring) plate = plateObj->valuestring;
  if (confObj && confObj->type == cJSON_Number) conf = confObj->valuedouble;
  cJSON_Delete(root);
  return !plate.empty();
}

static std::string mkTempDir() {
  std::string tmpl = "/tmp/garage_alprw.XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  char* res = mkdtemp(buf.data());
  if (!res) return "";
  return std::string(res);
}

static std::string basenameOnly(const std::string& path) {
  size_t pos = path.find_last_of("/\\");
  return (pos == std::string::npos) ? path : path.substr(pos + 1);
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

static std::string runEnhancerAlprMaxExternal(const std::string& enhBin, const std::string& inputPath) {
  std::string cmd = "\"" + enhBin + "\" --alpr-max \"" + inputPath + "\"";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return "";
  char buf[4096];
  std::string out;
  while (fgets(buf, sizeof(buf), pipe)) out += buf;
  pclose(pipe);

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

static void rmrf(const std::string& dir) {
  if (dir.empty()) return;
  std::string cmd = "rm -rf -- \"" + dir + "\"";
  system(cmd.c_str());
}

static void usage() {
  std::cerr <<
    "Usage:\n"
    "  garage_alpr_worker_daemon [--country br] [--config <openalpr.conf>] [--enh-bin <mini_enhancer>] [--debug]\n"
    "\n"
    "Reads image paths from stdin (one per line). Writes 1 JSON line per request to stdout.\n"
    "Errors are JSON lines on stderr (same schema as garage_alpr.sh --json).\n";
}

int main(int argc, char** argv) {
  std::string country = "br";
  std::string configFile = "./config/openalpr.conf.defaults";
  std::string enhBin = "./mini_enhancer/build/mini_enhancer";
  bool debug = false;

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
    if (a == "--debug") { debug = true; continue; }
    usage();
    return 1;
  }

  if (!fileExists(configFile)) {
    printErrJson("CONFIG_NOT_FOUND", 2, configFile);
    return 2;
  }
  if (access(enhBin.c_str(), X_OK) != 0) {
    printErrJson("ENH_BIN_NOT_EXECUTABLE", 5, enhBin);
    return 5;
  }

  RecognitionWorkerProcess::Params params;
  params.country = country;
  params.configFile = configFile;
  params.topn = 10;
  params.detectRegion = false;
  params.skipDetection = false;
  params.debug = false;

  RecognitionWorkerProcess worker(params);
  if (!worker.start()) {
    printErrJson("WORKER_START_FAILED", 5, "");
    return 5;
  }

  const std::regex legacyRe("^[A-Z]{3}[0-9]{4}$");
  std::string line;
  while (std::getline(std::cin, line)) {
    std::string srcPath = trim(line);
    if (srcPath.empty()) continue;

    if (!fileExists(srcPath)) {
      printErrJson("FILE_NOT_FOUND", 2, srcPath);
      continue;
    }

    std::string imgPath;
    std::string json;
    if (!worker.sendJob(srcPath) || !worker.readResult(imgPath, json)) {
      printErrJson("WORKER_IO_FAILED", 5, srcPath);
      continue;
    }

    std::string bestPlate;
    double bestConf = 0.0;
    extractBestPlateAndConf(json, bestPlate, bestConf);
    if (debug) std::cerr << "[dbg] original plate=" << bestPlate << " conf=" << bestConf << std::endl;

    bool needEnhance = true;
    if (!bestPlate.empty() && std::regex_match(bestPlate, legacyRe)) needEnhance = false;

    std::string used = "original";
    std::string processed = srcPath;

    if (needEnhance) {
      std::string tmp = mkTempDir();
      if (tmp.empty()) { printErrJson("TMPDIR_FAILED", 3, srcPath); continue; }
      std::string tmpIn = tmp + "/" + basenameOnly(srcPath);
      if (!copyFile(srcPath, tmpIn)) { rmrf(tmp); printErrJson("COPY_FAILED", 3, srcPath); continue; }

      std::string enhOut = runEnhancerAlprMaxExternal(enhBin, tmpIn);
      if (enhOut.empty() || !fileExists(enhOut)) { rmrf(tmp); printErrJson("ENHANCEMENT_FAILED", 3, srcPath); continue; }

      used = "enhanced";
      processed = enhOut;

      std::string imgPath2;
      std::string json2;
      if (!worker.sendJob(enhOut) || !worker.readResult(imgPath2, json2)) {
        rmrf(tmp);
        printErrJson("WORKER_IO_FAILED", 5, enhOut);
        continue;
      }
      bestPlate.clear(); bestConf = 0.0;
      extractBestPlateAndConf(json2, bestPlate, bestConf);
      if (debug) std::cerr << "[dbg] enhanced plate=" << bestPlate << " conf=" << bestConf << std::endl;

      if (bestPlate.empty()) {
        // keep tmp dir for inspection on failure
        printErrJson("NO_PLATE", 4, enhOut);
        continue;
      }

      // success: cleanup tmp dir
      rmrf(tmp);
    }

    if (bestPlate.empty()) {
      printErrJson("NO_PLATE", 4, processed);
      continue;
    }
    printOkJson(bestPlate, bestConf, used, processed, srcPath);
  }

  worker.stop();
  return 0;
}

