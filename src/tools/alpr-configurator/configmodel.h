// Simple key-value config model for openalpr.conf
#pragma once
#include <string>
#include <map>
#include <vector>

class ConfigModel {
public:
  bool load(const std::string& path);
  bool save(const std::string& path) const;

  std::string get(const std::string& key, const std::string& def="") const;
  void set(const std::string& key, const std::string& val);
  void remove(const std::string& key);

  std::vector<std::pair<std::string,std::string>> items() const;
  void replaceAll(const std::map<std::string,std::string>& kv);

private:
  std::map<std::string,std::string> kv_;
};

