/**
 * OpenALPR Web Configurator — Crow backend
 * Serves static UI (Tailwind) and API for config/presets.
 * Build with Crow: see web/CMakeLists.txt
 */
#include "crow.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <cstdio>
#include <memory>
#ifdef __linux__
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// Base path: executable dir (e.g. build/bin) or cwd for static + config roots
static std::string g_staticRoot;
static std::string g_projectRoot; // openalpr repo root (parent of web/)

static std::string readFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) return "";
  std::ostringstream os;
  os << f.rdbuf();
  return os.str();
}

static bool writeFile(const std::string& path, const std::string& content) {
  std::ofstream f(path);
  if (!f) return false;
  f << content;
  return true;
}

static std::string getMime(const std::string& ext) {
  if (ext == ".html") return "text/html";
  if (ext == ".js") return "application/javascript";
  if (ext == ".css") return "text/css";
  if (ext == ".json") return "application/json";
  if (ext == ".ico") return "image/x-icon";
  return "application/octet-stream";
}

// Parse .conf: get value for key (key = value), trim, ignore ; comments
static std::string confGet(const std::string& content, const std::string& key) {
  std::istringstream is(content);
  std::string line;
  while (std::getline(is, line)) {
    size_t sem = line.find(';');
    if (sem != std::string::npos) line = line.substr(0, sem);
    size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string k = line.substr(0, eq);
    std::string v = line.substr(eq + 1);
    auto trim = [](std::string& s) {
      while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
      size_t i = 0; while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) i++;
      s = s.substr(i);
    };
    trim(k); trim(v);
    if (k == key) return v;
  }
  return "";
}

// List country codes from runtime_data/region/*.xml
static std::vector<std::string> listCountries(const std::string& runtimeDir) {
  std::vector<std::string> out;
  std::string regionDir = runtimeDir + "/region";
  if (!fs::exists(regionDir)) return out;
  for (const auto& e : fs::directory_iterator(regionDir)) {
    if (e.path().extension() == ".xml")
      out.push_back(e.path().stem().string());
  }
  std::sort(out.begin(), out.end());
  return out;
}

// Auto-select country: br2 if exists, else br, else first
static std::string selectCountry(const std::vector<std::string>& countries) {
  for (const std::string& c : {"br2", "br"}) {
    if (std::find(countries.begin(), countries.end(), c) != countries.end())
      return c;
  }
  return countries.empty() ? "" : countries[0];
}

int main(int argc, char* argv[]) {
  crow::SimpleApp app;

  // Resolve paths: prefer executable location so alpr-tool is found regardless of cwd
  char buf[4096];
  {
    std::string exeDir;
#ifdef __linux__
    char exePath[4096];
    ssize_t n = readlink("/proc/self/exe", exePath, sizeof(exePath)-1);
    if (n > 0) { exePath[n] = '\0'; exeDir = exePath; }
#endif
    if (!exeDir.empty()) {
      fs::path p(exeDir);
      p = p.parent_path();  // bin/
      if (p.filename() == "bin") {
        p = p.parent_path();  // build/
        if (p.filename() == "build") {
          g_staticRoot = p.string();
          p = p.parent_path();  // web/
          if (p.filename() == "web")
            g_projectRoot = p.parent_path().string();
        }
      }
    }
    if (g_projectRoot.empty()) {
      if (getcwd(buf, sizeof(buf))) {
        g_projectRoot = buf;
        if (g_projectRoot.find("build") != std::string::npos) {
          g_staticRoot = g_projectRoot;
          size_t pos = g_projectRoot.rfind("build");
          if (pos != std::string::npos) g_projectRoot = g_projectRoot.substr(0, pos);
          if (g_projectRoot.size() >= 4 && g_projectRoot.substr(g_projectRoot.size()-4) == "/web")
            g_projectRoot = g_projectRoot.substr(0, g_projectRoot.size()-4);
        } else {
          g_staticRoot = g_projectRoot + "/web/static";
        }
      }
    } else if (g_staticRoot.empty()) {
      g_staticRoot = g_projectRoot + "/web/static";
    }
  }
  if (g_staticRoot.empty()) g_staticRoot = ".";
  if (g_projectRoot.empty() && getcwd(buf, sizeof(buf))) g_projectRoot = buf;
  if (g_projectRoot.empty()) g_projectRoot = ".";

  // Static files
  CROW_ROUTE(app, "/")
  ([] {
    std::string path = g_staticRoot + "/index.html";
    if (path.find("/web/") != std::string::npos) { /* from repo */ }
    std::string body = readFile(path);
    if (body.empty()) body = "<!DOCTYPE html><html><body><h1>Static not found</h1><p>Run from repo root or build/bin with static/ copied.</p></body></html>";
    return crow::response(200, body);
  });

  CROW_ROUTE(app, "/static/<string>")
  ([](const std::string& name) {
    std::string path = g_staticRoot + "/" + name;
    std::string body = readFile(path);
    if (body.empty()) return crow::response(404);
    size_t dot = name.rfind('.');
    std::string ext = dot != std::string::npos ? name.substr(dot) : "";
    crow::response r(200, body);
    r.set_header("Content-Type", getMime(ext));
    return r;
  });

  // API: list presets (artifacts/configs/*.conf)
  CROW_ROUTE(app, "/api/presets")
  ([&] {
    crow::json::wvalue out;
    std::vector<std::string> list;
    std::string dir = g_projectRoot + "/artifacts/configs";
    if (!fs::exists(dir)) {
      out["presets"] = list;
      return crow::response(200, out.dump());
    }
    for (const auto& e : fs::directory_iterator(dir)) {
      if (e.path().extension() == ".conf")
        list.push_back(e.path().filename().string());
    }
    std::sort(list.begin(), list.end());
    out["presets"] = list;
    out["configs_dir"] = dir;
    return crow::response(200, out.dump());
  });

  // API: get config file body
  CROW_ROUTE(app, "/api/config")
  ([](const crow::request& req) {
    std::string path = req.url_params.get("path") ? req.url_params.get("path") : "";
    if (path.empty()) path = g_projectRoot + "/config/openalpr.conf.defaults";
    else if (path[0] != '/') path = g_projectRoot + "/" + path;
    std::string body = readFile(path);
    if (body.empty()) return crow::response(404, "File not found or empty");
    crow::json::wvalue j;
    j["path"] = path;
    j["content"] = body;
    return crow::response(200, j.dump());
  });

  // API: save config (POST body = raw .conf content or JSON { "path": "...", "content": "..." }
  CROW_ROUTE(app, "/api/config")
  .methods("POST"_method)
  ([](const crow::request& req) {
    std::string path = g_projectRoot + "/config/openalpr.conf.user";
    std::string content = req.body;
    if (req.get_header_value("Content-Type").find("application/json") != std::string::npos) {
      auto j = crow::json::load(content);
      if (j && j.has("path")) path = j["path"].s();
      if (j && j.has("content")) content = j["content"].s();
    }
    if (path.empty() || path.find("..") != std::string::npos) return crow::response(400, "Invalid path");
    if (!writeFile(path, content)) return crow::response(500, "Write failed");
    crow::json::wvalue out;
    out["saved"] = path;
    return crow::response(200, out.dump());
  });

  // Stream MJPEG: run alpr-tool preview --output-mjpeg 1 and pipe to response
  CROW_ROUTE(app, "/stream")
  ([](const crow::request& req) {
    const char* src = req.url_params.get("source");
    if (!src || !src[0]) {
      crow::response r(400, "source= required (path to video)");
      r.set_header("Content-Type", "text/plain");
      return r;
    }
    std::string source(src);
    std::string country = req.url_params.get("country") ? req.url_params.get("country") : "br";
    std::string toolPath = g_projectRoot + "/build/src/alpr-tool";
    if (!fs::exists(toolPath)) {
      crow::response r(503, "alpr-tool not found at " + toolPath + " (run build first)");
      r.set_header("Content-Type", "text/plain");
      return r;
    }
    std::string safeSource;
    for (char c : source) {
      if (c == '\'') { safeSource += "'\\''"; continue; }
      safeSource += c;
    }
    std::string cmd = "cd '" + g_projectRoot + "' && '" + toolPath + "' preview --source '" + safeSource + "' --country " + country + " --output-mjpeg 1 --max-seconds 30 2>/dev/null";
    std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
      crow::response r(500, "Failed to run alpr-tool");
      r.set_header("Content-Type", "text/plain");
      return r;
    }
    std::ostringstream body;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), pipe.get())) > 0)
      body.write(buf, static_cast<std::streamsize>(n));
    crow::response r(200, body.str());
    r.set_header("Content-Type", "multipart/x-mixed-replace; boundary=--frame");
    return r;
  });

  // API: debug paths (to troubleshoot stream errors)
  CROW_ROUTE(app, "/api/debug-paths")
  ([&] {
    std::string toolPath = g_projectRoot + "/build/src/alpr-tool";
    crow::json::wvalue j;
    j["project_root"] = g_projectRoot;
    j["static_root"] = g_staticRoot;
    j["alpr_tool_path"] = toolPath;
    j["alpr_tool_exists"] = fs::exists(toolPath);
    return crow::response(200, j.dump());
  });

  // API: bootstrap — auto-detect runtime, countries, validate, generate config if missing
  CROW_ROUTE(app, "/api/bootstrap")
  ([&] {
    crow::json::wvalue j;
    std::string configPath = g_projectRoot + "/config/openalpr.conf.user";
    std::string defaultsPath = g_projectRoot + "/config/openalpr.conf.defaults";
    std::string content = readFile(configPath);
    if (content.empty()) content = readFile(defaultsPath);
    j["config_path"] = content.empty() ? "" : configPath;
    j["config_exists"] = !content.empty();

    std::string runtimeDir = confGet(content, "runtime_dir");
    if (runtimeDir.empty()) runtimeDir = "./runtime_data";
    if (runtimeDir[0] != '/') {
      if (runtimeDir.substr(0, 2) == "./") runtimeDir = g_projectRoot + "/" + runtimeDir.substr(2);
      else runtimeDir = g_projectRoot + "/" + runtimeDir;
    }
    if (!fs::exists(runtimeDir)) {
      runtimeDir = g_projectRoot + "/runtime_data";
    }
    j["runtime_data_path"] = runtimeDir;
    bool runtimeOk = fs::exists(runtimeDir) && fs::is_directory(runtimeDir);
    j["runtime_data_ok"] = runtimeOk;

    std::vector<std::string> countries = listCountries(runtimeDir);
    j["countries"] = countries;
    std::string selected = selectCountry(countries);
    j["selected_country"] = selected;

    std::string cascadePath = runtimeDir + "/region/" + selected + ".xml";
    bool cascadeOk = !selected.empty() && fs::exists(cascadePath);
    j["cascade_ok"] = cascadeOk;
    j["cascade_path"] = cascadePath;

    std::string tessDir = runtimeDir + "/tessdata";
    if (!fs::exists(tessDir)) tessDir = runtimeDir + "/tessdata";
    bool tessOk = fs::exists(tessDir) && fs::is_directory(tessDir);
    if (!tessOk) tessOk = fs::exists(runtimeDir + "/tessdata");
    j["tessdata_ok"] = tessOk;
    j["tessdata_path"] = tessDir;

    bool configExists = !content.empty();
    if (!configExists && fs::exists(defaultsPath)) {
      std::string def = readFile(defaultsPath);
      if (!def.empty()) {
        writeFile(configPath, def);
        j["config_generated"] = true;
        j["config_exists"] = true;
        j["config_path"] = configPath;
      } else {
        j["config_generated"] = false;
      }
    } else {
      j["config_generated"] = false;
    }
    return crow::response(200, j.dump());
  });

  // API: list countries (from runtime_data/region/*.xml)
  CROW_ROUTE(app, "/api/countries")
  ([&](const crow::request& req) {
    std::string rt = req.url_params.get("runtime_dir") ? req.url_params.get("runtime_dir") : "";
    if (rt.empty()) {
      std::string cfg = readFile(g_projectRoot + "/config/openalpr.conf.user");
      if (cfg.empty()) cfg = readFile(g_projectRoot + "/config/openalpr.conf.defaults");
      rt = confGet(cfg, "runtime_dir");
      if (rt.empty()) rt = "./runtime_data";
      if (rt[0] != '/') rt = g_projectRoot + "/" + (rt.substr(0,2)=="./" ? rt.substr(2) : rt);
    }
    if (!fs::exists(rt)) rt = g_projectRoot + "/runtime_data";
    std::vector<std::string> list = listCountries(rt);
    crow::json::wvalue j;
    j["countries"] = list;
    j["runtime_data_path"] = rt;
    return crow::response(200, j.dump());
  });

  // API: validate runtime (same checks as bootstrap, returns per-item status)
  CROW_ROUTE(app, "/api/validate-runtime")
  ([&](const crow::request& req) {
    std::string rt = req.url_params.get("runtime_dir") ? req.url_params.get("runtime_dir") : "";
    std::string country = req.url_params.get("country") ? req.url_params.get("country") : "br";
    if (rt.empty()) {
      std::string cfg = readFile(g_projectRoot + "/config/openalpr.conf.user");
      if (cfg.empty()) cfg = readFile(g_projectRoot + "/config/openalpr.conf.defaults");
      rt = confGet(cfg, "runtime_dir");
      if (rt.empty()) rt = "./runtime_data";
      if (rt[0] != '/') rt = g_projectRoot + "/" + (rt.substr(0,2)=="./" ? rt.substr(2) : rt);
    }
    if (!fs::exists(rt)) rt = g_projectRoot + "/runtime_data";
    crow::json::wvalue j;
    j["runtime_data_ok"] = fs::exists(rt) && fs::is_directory(rt);
    j["cascade_ok"] = fs::exists(rt + "/region/" + country + ".xml");
    j["tessdata_ok"] = fs::exists(rt + "/tessdata") && fs::is_directory(rt + "/tessdata");
    j["runtime_data_path"] = rt;
    j["cascade_path"] = rt + "/region/" + country + ".xml";
    return crow::response(200, j.dump());
  });

  // API: build preview command (suggested CLI)
  CROW_ROUTE(app, "/api/preview-command")
  ([](const crow::request& req) {
    std::string conf = req.url_params.get("conf") ? req.url_params.get("conf") : "";
    std::string source = req.url_params.get("source") ? req.url_params.get("source") : "";
    std::string country = req.url_params.get("country") ? req.url_params.get("country") : "br";
    std::string plates_only = req.url_params.get("plates_only_past_line") ? req.url_params.get("plates_only_past_line") : "0";
    std::string line_pct = req.url_params.get("crossing_line_pct") ? req.url_params.get("crossing_line_pct") : "50";
    if (conf.empty()) conf = g_projectRoot + "/config/openalpr.conf.defaults";
    else if (conf.find("/") == std::string::npos) conf = g_projectRoot + "/artifacts/configs/" + conf;
    std::ostringstream cmd;
    cmd << "./build/src/alpr-tool preview --country " << country << " --conf " << conf;
    if (!source.empty()) cmd << " --source " << source;
    if (plates_only == "1") cmd << " --plates-only-past-line 1 --crossing-line-pct " << line_pct;
    crow::json::wvalue out;
    out["command"] = cmd.str();
    return crow::response(200, out.dump());
  });

  int port = 18080;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--port" && i + 1 < argc) { port = std::stoi(argv[++i]); break; }
  }
  app.port(port).multithreaded().run();
  return 0;
}
