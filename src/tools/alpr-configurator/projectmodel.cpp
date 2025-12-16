#include "projectmodel.h"
#include <QFile>
#include <QJsonDocument>

bool ProjectModel::load(const QString& path) {
  QFile f(path);
  if (!f.open(QIODevice::ReadOnly)) return false;
  auto doc = QJsonDocument::fromJson(f.readAll());
  if (!doc.isObject()) return false;
  fromJson(doc.object());
  return true;
}

bool ProjectModel::save(const QString& path) const {
  QFile f(path);
  if (!f.open(QIODevice::WriteOnly)) return false;
  QJsonDocument doc(toJson());
  f.write(doc.toJson(QJsonDocument::Indented));
  return true;
}

void ProjectModel::clear() {
  sources_.clear();
  runtimeData_ = "auto";
}

QJsonObject ProjectModel::toJson() const {
  QJsonObject obj;
  obj["version"] = 1;
  obj["runtime_data"] = runtimeData_;
  QJsonArray arr;
  for (const auto& s : sources_) {
    QJsonObject o;
    o["id"] = s.id;
    o["type"] = s.type;
    o["uri"] = s.uri;
    o["country"] = s.country;
    o["conf_path"] = s.confPath;
    o["runtime_data"] = s.runtimeData.isEmpty() ? runtimeData_ : s.runtimeData;
    o["roi"] = s.roi;
    o["crossing"] = s.crossing;
    o["prewarp"] = s.prewarp;
    o["preview"] = s.previewParams;
    arr.append(o);
  }
  obj["sources"] = arr;
  return obj;
}

void ProjectModel::fromJson(const QJsonObject& obj) {
  clear();
  runtimeData_ = obj.value("runtime_data").toString("auto");
  auto arr = obj.value("sources").toArray();
  for (auto v : arr) {
    if (!v.isObject()) continue;
    auto o = v.toObject();
    SourceEntry s;
    s.id = o.value("id").toString();
    s.type = o.value("type").toString("rtsp");
    s.uri = o.value("uri").toString();
    s.country = o.value("country").toString("br");
    s.confPath = o.value("conf_path").toString();
    s.runtimeData = o.value("runtime_data").toString();
    s.roi = o.value("roi").toObject();
    s.crossing = o.value("crossing").toObject();
    s.prewarp = o.value("prewarp").toObject();
    s.previewParams = o.value("preview").toObject();
    sources_.push_back(s);
  }
}

