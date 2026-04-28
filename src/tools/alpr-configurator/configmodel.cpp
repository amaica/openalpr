#include "configmodel.h"
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

static string trim(const string& s) {
  auto b = s.find_first_not_of(" \t\r\n");
  if (b == string::npos) return "";
  auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

bool ConfigModel::load(const std::string& path) {
  kv_.clear();
  ifstream in(path);
  if (!in.good()) return false;
  string line;
  while (getline(in, line)) {
    auto t = trim(line);
    if (t.empty() || t[0]=='#' || t[0]==';') continue;
    auto pos = t.find('=');
    if (pos == string::npos) continue;
    string key = trim(t.substr(0,pos));
    string val = trim(t.substr(pos+1));
    kv_[key]=val;
  }
  return true;
}

bool ConfigModel::save(const std::string& path) const {
  ofstream out(path);
  if (!out.good()) return false;
  for (auto& kv : kv_) {
    out << kv.first << " = " << kv.second << "\n";
  }
  return true;
}

std::string ConfigModel::get(const std::string& key, const std::string& def) const {
  auto it = kv_.find(key);
  if (it == kv_.end()) return def;
  return it->second;
}

void ConfigModel::set(const std::string& key, const std::string& val) {
  kv_[key]=val;
}

void ConfigModel::remove(const std::string& key) {
  kv_.erase(key);
}

std::vector<std::pair<std::string,std::string>> ConfigModel::items() const {
  std::vector<std::pair<std::string,std::string>> v(kv_.begin(), kv_.end());
  return v;
}

void ConfigModel::replaceAll(const std::map<std::string,std::string>& kv) {
  kv_ = kv;
}

