// Project model for multi-source ALPR setups (.alprproj.json)
#pragma once
#include <QString>
#include <QJsonObject>
#include <QJsonArray>
#include <vector>

struct SourceEntry {
  QString id;
  QString type;
  QString uri;
  QString country;
  QString confPath;
  QString runtimeData; // may be "auto"
  QJsonObject roi;     // {x,y,w,h}
  QJsonObject crossing; // line/dir thresholds
  QJsonObject prewarp; // points or enabled
  QJsonObject previewParams;
};

class ProjectModel {
public:
  bool load(const QString& path);
  bool save(const QString& path) const;

  void setRuntimeData(const QString& rt) { runtimeData_ = rt; }
  QString runtimeData() const { return runtimeData_; }

  std::vector<SourceEntry>& sources() { return sources_; }
  const std::vector<SourceEntry>& sources() const { return sources_; }

  void clear();

private:
  QJsonObject toJson() const;
  void fromJson(const QJsonObject& obj);

  QString runtimeData_ = "auto";
  std::vector<SourceEntry> sources_;
};

