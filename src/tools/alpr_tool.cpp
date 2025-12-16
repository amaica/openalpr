#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <tclap/CmdLine.h>
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <limits>
#include <climits>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>
#include <algorithm>
#include <regex>
#include "openalpr/alpr.h"
#include "openalpr/config.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <unistd.h>
#include "support/filesystem.h"
#include <stdexcept>

using namespace std;
using namespace cv;
using namespace alpr;

struct ConfigWriter {
  string path;
  vector<string> lines;
  map<string,size_t> keyIndex;
  string lastWritePath;

  static string trim(const string& s) {
    const char* ws = " \t";
    size_t b = s.find_first_not_of(ws);
    if (b == string::npos) return "";
    size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
  }

  bool load(const string& p) {
    path = p;
    lines.clear();
    keyIndex.clear();
    ifstream in(p);
    if (!in.good()) return false;
    string line;
    size_t idx = 0;
    while (getline(in, line)) {
      lines.push_back(line);
      auto pos = line.find('=');
      if (pos != string::npos) {
        string key = trim(line.substr(0, pos));
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        if (!key.empty() && key[0] != ';' && key[0] != '#')
          keyIndex[key] = idx;
      }
      idx++;
    }
    lastWritePath = path;
    return true;
  }

  string get(const string& key, const string& def="") const {
    string k = key;
    std::transform(k.begin(), k.end(), k.begin(), ::tolower);
    auto it = keyIndex.find(k);
    if (it == keyIndex.end()) return def;
    auto pos = lines[it->second].find('=');
    if (pos == string::npos) return def;
    return trim(lines[it->second].substr(pos+1));
  }

  void set(const string& key, const string& value) {
    string k = key;
    std::transform(k.begin(), k.end(), k.begin(), ::tolower);
    string newline = k + " = " + value;
    auto it = keyIndex.find(k);
    if (it != keyIndex.end()) {
      lines[it->second] = newline;
    } else {
      keyIndex[k] = lines.size();
      lines.push_back(newline);
    }
  }

  bool save() {
    lastWritePath = path;
    ofstream out(path);
    if (out.good()) {
      for (auto& l : lines) out << l << "\n";
      return true;
    }
    // fallback
    string alt = path + ".new";
    ofstream out2(alt);
    if (!out2.good()) return false;
    for (auto& l : lines) out2 << l << "\n";
    lastWritePath = alt;
    cerr << "Could not write " << path << ", wrote " << alt << " instead\n";
    return true;
  }
};

struct RoiState {
  bool drawing=false;
  Point start;
  Rect draft;     // in original coords
  Rect applied;   // in original coords (source of truth)
  bool dirty=false;
  bool defaultUsed=false;
};

struct PrewarpState {
  bool enabled=false;
  bool editing=false;
  bool dirty=false;
  bool valid=false;
  std::vector<Point2f> ptsOrig; // size 4, in original coords
  Mat homography;
};

static RoiState g_roiState;
static PrewarpState g_prewarpState;
static bool g_saveRequested = false;
static bool g_saveAndExitRequested = false;
static bool g_quitRequested = false;

struct DisplayMapper {
  int origW=1, origH=1;
  int dispW=1, dispH=1;
  int offX=0, offY=0;
  double scale=1.0;

  void setOriginal(int w, int h) {
    origW = std::max(1, w);
    origH = std::max(1, h);
    const double maxW = 1280.0;
    const double maxH = 720.0;
    scale = std::min(1.0, std::min(maxW / origW, maxH / origH));
    dispW = static_cast<int>(std::round(origW * scale));
    dispH = static_cast<int>(std::round(origH * scale));
    offX = 0;
    offY = 0;
  }

  Point origToDisp(const Point& p) const {
    int x = static_cast<int>(std::round(p.x * scale)) + offX;
    int y = static_cast<int>(std::round(p.y * scale)) + offY;
    return Point(x,y);
  }

  Point dispToOrig(const Point& p) const {
    double x = (p.x - offX) / scale;
    double y = (p.y - offY) / scale;
    int xi = static_cast<int>(std::round(x));
    int yi = static_cast<int>(std::round(y));
    xi = std::min(std::max(0, xi), origW-1);
    yi = std::min(std::max(0, yi), origH-1);
    return Point(xi, yi);
  }

  Rect origToDisp(const Rect& r) const {
    Point p1 = origToDisp(Point(r.x, r.y));
    Point p2 = origToDisp(Point(r.x + r.width, r.y + r.height));
    return Rect(Point(std::min(p1.x,p2.x), std::min(p1.y,p2.y)),
                Point(std::max(p1.x,p2.x), std::max(p1.y,p2.y)));
  }

  Rect dispToOrig(const Rect& r) const {
    Point p1 = dispToOrig(Point(r.x, r.y));
    Point p2 = dispToOrig(Point(r.x + r.width, r.y + r.height));
    return Rect(Point(std::min(p1.x,p2.x), std::min(p1.y,p2.y)),
                Point(std::max(p1.x,p2.x), std::max(p1.y,p2.y)));
  }
};

static Rect normalizedRect(const Rect& r, const Mat& frame) {
  int x = std::max(0, std::min(r.x, frame.cols-1));
  int y = std::max(0, std::min(r.y, frame.rows-1));
  int w = std::min(frame.cols - x, std::abs(r.width));
  int h = std::min(frame.rows - y, std::abs(r.height));
  return Rect(x,y,w,h);
}

static Rect defaultRoi(const Mat& frame) {
  return Rect(0, frame.rows/2, frame.cols, frame.rows/2);
}

static vector<Point2f> defaultPrewarpPts(const Mat& frame) {
  return {
    Point2f(0.0f, 0.0f),
    Point2f(static_cast<float>(frame.cols - 1), 0.0f),
    Point2f(static_cast<float>(frame.cols - 1), static_cast<float>(frame.rows - 1)),
    Point2f(0.0f, static_cast<float>(frame.rows - 1))
  };
}

static void ensurePrewarpHomography(PrewarpState& st, const Size& sz) {
  if (st.ptsOrig.size() != 4) { st.valid = false; return; }
  vector<Point2f> dst = defaultPrewarpPts(Mat(sz, CV_8UC1));
  st.homography = getPerspectiveTransform(st.ptsOrig, dst);
  st.valid = !st.homography.empty();
}

static string derivePlanarStringFromHomography(const Mat& H, const Size& sz) {
  if (H.empty()) return "";
  try {
    Mat K = Mat::eye(3,3,CV_64F);
    vector<Mat> Rs, ts, ns;
    int solutions = decomposeHomographyMat(H, K, Rs, ts, ns);
    if (solutions <= 0) return "";
    Mat R = Rs[0];
    Mat t = ts[0];
    double rx = atan2(R.at<double>(2,1), R.at<double>(2,2));
    double ry = atan2(-R.at<double>(2,0), sqrt(R.at<double>(2,1)*R.at<double>(2,1) + R.at<double>(2,2)*R.at<double>(2,2)));
    double rz = atan2(R.at<double>(1,0), R.at<double>(0,0));
    double panX = t.at<double>(0);
    double panY = t.at<double>(1);
    double dist = t.at<double>(2);
    double stretchX = 1.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "planar," << sz.width << "," << sz.height << ","
        << rx << "," << ry << "," << rz << ","
        << stretchX << "," << dist << "," << panX << "," << panY;
    return oss.str();
  } catch (...) {
    return "";
  }
}

static void ensureParentDir(const string& path) {
  auto pos = path.find_last_of('/');
  if (pos == string::npos) return;
  string dir = path.substr(0, pos);
  if (dir.empty()) return;
  mkdir(dir.c_str(), 0755);
}

enum PlayState { STATE_PLAYING, STATE_PAUSED, STATE_STOPPED };

struct Button {
  string label;
  Rect area;
};

static vector<Button> buildButtons(int width) {
  const int btnW = 128;
  const int btnH = 32;
  const int pad = 6;
  vector<string> labels = {"PLAY","PAUSE","STOP","SAVE ROI","SAVE & EXIT","RESET","PREWARP ON/OFF","EDIT PREWARP","QUIT"};
  vector<Button> btns;
  int x = pad;
  int y = pad;
  for (auto& l : labels) {
    btns.push_back({l, Rect(x,y,btnW,btnH)});
    x += btnW + pad;
    if (x + btnW > width) { // wrap if window too small
      x = pad;
      y += btnH + pad;
    }
  }
  return btns;
}

static void drawButtons(Mat& frame, const vector<Button>& btns, PlayState st) {
  for (const auto& b : btns) {
    Scalar bg(30,30,30);
    if ((st == STATE_PLAYING && b.label == "PLAY") ||
        (st == STATE_PAUSED && b.label == "PAUSE") ||
        (st == STATE_STOPPED && b.label == "STOP")) {
      bg = Scalar(40,80,160);
    }
    rectangle(frame, b.area, bg, FILLED);
    rectangle(frame, b.area, Scalar(200,200,200), 1);
    putText(frame, b.label, Point(b.area.x+8, b.area.y+b.area.height-10),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1, LINE_AA);
  }
}

static bool pointInRect(const Point& p, const Rect& r) {
  return p.x >= r.x && p.x < r.x + r.width && p.y >= r.y && p.y < r.y + r.height;
}

struct MouseCtx {
  vector<Button>* buttons;
  PlayState* state;
  DisplayMapper* mapper;
  PrewarpState* prewarp;
};

struct SpeedConfig {
  bool enabled=false;
  string mode="lines";
  double yA=0.0, yB=0.0;
  double dist_m=0.0;
  string timeSource="fps";
  double minKmh=0.0;
  double maxKmh=300.0;
  string smoothing="none";
  double emaAlpha=0.2;
  bool log=false;
  bool requirePlate=true;
};

struct Track {
  int id=0;
  Rect last_bbox;
  double last_centerY_ema=-1.0;
  double last_seen_t=0.0;
  bool crossedA=false;
  bool crossedB=false;
  bool fired=false;
  double tA=0.0;
  double tB=0.0;
  double last_speed_kmh=0.0;
  std::string best_plate_text;
  double best_plate_conf=-1.0;
  std::string last_logged_plate_text;
  double last_logged_t=-1.0;
};

struct PreviewRuntimeOptions {
  bool logPlates=false;
  bool logEvents=true;
  bool logOcrMetrics=false;
  bool ocrOnlyAfterCrossing=false;
  bool logCrossingMetrics=false;
  std::string crossingMode="off"; // off|motion
  bool crossingRoiProvided=false;
  Rect crossingRoi;
  bool alprRoiProvided=false;
  Rect alprRoi;
  Point crossingP1;
  Point crossingP2;
  int motionThresh=25;
  int motionMinArea=1500;
  int crossingDebounce=3;
  double motionMinRatio=0.01;
  bool motionDirectionFilter=true;
  int crossingArmMinFrames=10;
  double crossingArmMinRatio=0.01;
  int logThrottleMs=400;
  int logPlatesEveryN=10;
  int maxTracks=32;
  int trackTtlMs=1000;
  std::string logPlatesFile;
  int maxSeconds=0;
  bool gateAfterCrossing=false;
  std::string reportJsonPath;
  double crossingLinePct=50.0; // percent of frame/ROI height
};

struct RuntimeResolveResult {
  bool ok=false;
  std::string path;
  std::string reason;
  std::vector<std::string> tested;
  bool preferredInvalid=false;
  std::string preferredReason;
};

struct DoctorResult {
  bool ok=false;
  std::string confPath;
  std::string runtimePath;
};

static RuntimeResolveResult resolveRuntimeData(const std::string& country, const std::string& preferred);

static double iouRect(const Rect& a, const Rect& b) {
  int x1 = std::max(a.x, b.x);
  int y1 = std::max(a.y, b.y);
  int x2 = std::min(a.x + a.width,  b.x + b.width);
  int y2 = std::min(a.y + a.height, b.y + b.height);
  int iw = std::max(0, x2 - x1);
  int ih = std::max(0, y2 - y1);
  int inter = iw * ih;
  int ua = a.width * a.height + b.width * b.height - inter;
  if (ua <= 0) return 0.0;
  return static_cast<double>(inter) / static_cast<double>(ua);
}

static void mouseCb(int event, int x, int y, int, void* userdata) {
  auto* payload = reinterpret_cast<MouseCtx*>(userdata);
  if (!payload) return;
  auto& buttons = *payload->buttons;
  PlayState& state = *payload->state;
  DisplayMapper& mapper = *payload->mapper;
  PrewarpState& prewarp = *payload->prewarp;

  if (event == EVENT_LBUTTONDOWN) {
    // Check buttons first
    for (const auto& b : buttons) {
      if (pointInRect(Point(x,y), b.area)) {
        if (b.label == "PLAY") state = STATE_PLAYING;
        else if (b.label == "PAUSE") state = STATE_PAUSED;
        else if (b.label == "STOP") state = STATE_STOPPED;
        else if (b.label == "SAVE ROI") { g_roiState.dirty = true; g_saveRequested = true; }
        else if (b.label == "SAVE & EXIT") { g_roiState.dirty = true; g_saveRequested = true; g_saveAndExitRequested = true; }
        else if (b.label == "RESET") { g_roiState.applied = Rect(); g_roiState.draft = Rect(); g_roiState.dirty = false; g_roiState.defaultUsed=false; prewarp.enabled=false; prewarp.dirty=true; prewarp.ptsOrig.clear(); prewarp.valid=false; }
        else if (b.label == "PREWARP ON/OFF") { prewarp.enabled = !prewarp.enabled; prewarp.dirty = true; }
        else if (b.label == "EDIT PREWARP") { prewarp.editing = true; }
        else if (b.label == "QUIT") { g_quitRequested = true; state = STATE_STOPPED; cout << "QUIT CLICKED\n"; }
        return;
      }
    }
    if (prewarp.editing) {
      Point p = mapper.dispToOrig(Point(x,y));
      if (prewarp.ptsOrig.size() < 4) prewarp.ptsOrig.push_back(p);
      else {
        // move nearest point
        int idx = 0;
        double best = std::numeric_limits<double>::max();
        for (size_t i=0;i<prewarp.ptsOrig.size();++i) {
          double d = norm(prewarp.ptsOrig[i] - Point2f(p));
          if (d < best) { best = d; idx = static_cast<int>(i); }
        }
        prewarp.ptsOrig[idx] = p;
      }
      prewarp.dirty = true;
      return;
    }
    // Start drawing ROI in display coords, convert to original
    g_roiState.drawing = true;
    Point p = mapper.dispToOrig(Point(x,y));
    g_roiState.start = p;
    g_roiState.draft = Rect();
  } else if (event == EVENT_MOUSEMOVE && g_roiState.drawing) {
    Point p = mapper.dispToOrig(Point(x,y));
    g_roiState.draft = Rect(g_roiState.start, p);
  } else if (event == EVENT_LBUTTONUP) {
    g_roiState.drawing = false;
    Point p = mapper.dispToOrig(Point(x,y));
    g_roiState.draft = Rect(g_roiState.start, p);
    g_roiState.dirty = true;
  }
}

static void overlayInfo(Mat& frame, const Rect& roiDisp, const Rect& roiOrig, const DisplayMapper& mapper, const string& confPath, bool dirty, bool defaultUsed, const PrewarpState& prewarp) {
  if (roiDisp.area() > 0) rectangle(frame, roiDisp, Scalar(0,255,0), 2);

  std::ostringstream oss;
  if (roiOrig.area() > 0) {
    float rx = roiOrig.x / static_cast<float>(mapper.origW);
    float ry = roiOrig.y / static_cast<float>(mapper.origH);
    float rw = roiOrig.width / static_cast<float>(mapper.origW);
    float rh = roiOrig.height / static_cast<float>(mapper.origH);
    oss << "ROI % x=" << rx << " y=" << ry << " w=" << rw << " h=" << rh;
  } else {
    oss << "ROI disabled";
  }
  if (defaultUsed) oss << " [DEFAULT]";
  if (dirty) oss << " [DIRTY]";
  putText(frame, oss.str(), Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,255,255),1,LINE_AA);
  std::ostringstream ps;
  ps << "Prewarp: " << (prewarp.enabled ? "ON" : "OFF");
  if (prewarp.editing) ps << " [EDIT]";
  if (prewarp.dirty) ps << " [DIRTY]";
  putText(frame, ps.str(), Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,0),1,LINE_AA);
  putText(frame, confPath, Point(10, 110), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200,200,200),1,LINE_AA);
}

static bool openCapture(const string& src, VideoCapture& cap) {
  if (src.empty()) return false;
  if (isdigit(src[0]) && src.find_first_not_of("0123456789") == string::npos) {
    int idx = stoi(src);
    return cap.open(idx);
  }
  return cap.open(src);
}

static Mat applyPrewarpDisplay(const Mat& frame, const PrewarpState& st) {
  if (!st.enabled || !st.valid || st.homography.empty()) return frame;
  Mat warped;
  warpPerspective(frame, warped, st.homography, frame.size(), INTER_LINEAR, BORDER_REPLICATE);
  return warped;
}

static void saveRoiToConfig(const Rect& roi, const Mat& frame, ConfigWriter& cfg) {
  float rx = static_cast<float>(roi.x) / frame.cols;
  float ry = static_cast<float>(roi.y) / frame.rows;
  float rw = static_cast<float>(roi.width) / frame.cols;
  float rh = static_cast<float>(roi.height) / frame.rows;
  cfg.set("enable_roi", "1");
  cfg.set("roi_x", to_string(rx));
  cfg.set("roi_y", to_string(ry));
  cfg.set("roi_width", to_string(rw));
  cfg.set("roi_height", to_string(rh));
}

static void disableRoi(ConfigWriter& cfg) {
  cfg.set("enable_roi", "0");
  cfg.set("roi_x", "0.0");
  cfg.set("roi_y", "0.0");
  cfg.set("roi_width", "1.0");
  cfg.set("roi_height", "1.0");
}

static Rect roiFromConfig(const ConfigWriter& cfg, const Mat& frame) {
  if (cfg.get("enable_roi","0") != "1") return Rect();
  float rx = stof(cfg.get("roi_x","0"));
  float ry = stof(cfg.get("roi_y","0"));
  float rw = stof(cfg.get("roi_width","1"));
  float rh = stof(cfg.get("roi_height","1"));
  int x = static_cast<int>(rx * frame.cols);
  int y = static_cast<int>(ry * frame.rows);
  int w = static_cast<int>(rw * frame.cols);
  int h = static_cast<int>(rh * frame.rows);
  return normalizedRect(Rect(x,y,w,h), frame);
}

static PrewarpState prewarpFromConfig(const ConfigWriter& cfg, const Mat& frame) {
  PrewarpState st;
  st.enabled = cfg.get("prewarp_enabled","0") == "1";
  vector<Point2f> pts(4, Point2f(0,0));
  bool hasPts = true;
  for (int i=0;i<4;i++) {
    std::ostringstream kx, ky;
    kx << "prewarp_p" << (i+1) << "x";
    ky << "prewarp_p" << (i+1) << "y";
    string sx = cfg.get(kx.str(),"");
    string sy = cfg.get(ky.str(),"");
    if (sx.empty() || sy.empty()) { hasPts = false; break; }
    float px = stof(sx);
    float py = stof(sy);
    pts[i].x = px * frame.cols;
    pts[i].y = py * frame.rows;
  }
  if (!hasPts) {
    st.ptsOrig = defaultPrewarpPts(frame);
  } else {
    st.ptsOrig = pts;
  }
  ensurePrewarpHomography(st, frame.size());
  return st;
}

static void savePrewarpToConfig(const PrewarpState& st, const Mat& frame, ConfigWriter& cfg) {
  cfg.set("prewarp_enabled", st.enabled ? "1" : "0");
  if (st.ptsOrig.size() == 4) {
    for (int i=0;i<4;i++) {
      std::ostringstream kx, ky;
      kx << "prewarp_p" << (i+1) << "x";
      ky << "prewarp_p" << (i+1) << "y";
      float px = st.ptsOrig[i].x / frame.cols;
      float py = st.ptsOrig[i].y / frame.rows;
      cfg.set(kx.str(), to_string(px));
      cfg.set(ky.str(), to_string(py));
    }
  }
  if (st.valid && !st.homography.empty()) {
    string planar = derivePlanarStringFromHomography(st.homography, frame.size());
    if (!planar.empty()) cfg.set("prewarp", planar);
  }
}

static SpeedConfig loadSpeedConfig(const ConfigWriter& cfg) {
  SpeedConfig s;
  s.enabled = cfg.get("speed_enabled","0") == "1";
  s.mode = cfg.get("speed_mode","lines");
  s.yA = atof(cfg.get("speed_line_a_y_percent","40").c_str())/100.0;
  s.yB = atof(cfg.get("speed_line_b_y_percent","70").c_str())/100.0;
  s.dist_m = atof(cfg.get("speed_dist_m","10").c_str());
  s.timeSource = cfg.get("speed_time_source","timestamp");
  s.minKmh = atof(cfg.get("speed_min_kmh","5").c_str());
  s.maxKmh = atof(cfg.get("speed_max_kmh","250").c_str());
  s.smoothing = cfg.get("speed_smoothing","ema");
  s.emaAlpha = atof(cfg.get("speed_ema_alpha","0.25").c_str());
  s.log = cfg.get("speed_log","1") == "1";
  s.requirePlate = cfg.get("speed_require_plate","1") == "1";
  return s;
}

static void cmdRoi(const string& source, const string& confPath, bool autoDemo=false, bool autoDemoNoPrewarp=false) {
  ConfigWriter cfg;
  cfg.load(confPath);
  g_saveRequested = false;
  g_saveAndExitRequested = false;
  g_quitRequested = false;
  if (cfg.lastWritePath.empty()) cfg.lastWritePath = confPath;
  string src = source;
  if (src.empty()) src = cfg.get("video_source", "");
  if (src.empty()) {
    cout << "Enter video source (rtsp/device/video path): ";
    getline(cin, src);
  }
  VideoCapture cap;
  if (!openCapture(src, cap)) {
    cerr << "Could not open source: " << src << endl;
    return;
  }
  namedWindow("alpr-tool roi", WINDOW_AUTOSIZE);
  auto buttons = buildButtons(1280);
  PlayState playState = STATE_PAUSED;
  DisplayMapper mapper;
  MouseCtx mctx{&buttons, &playState, &mapper, &g_prewarpState};
  setMouseCallback("alpr-tool roi", mouseCb, &mctx);

  Rect roiOrig;
  bool defaultUsed = false;
  int frameIndex = 0;
  Mat frame, display, canvas;
  // Start paused: grab first frame
  if (!cap.read(frame) || frame.empty()) {
    cerr << "No frames available\n";
    destroyWindow("alpr-tool roi");
    return;
  }
  mapper.setOriginal(frame.cols, frame.rows);
  roiOrig = roiFromConfig(cfg, frame);
  if (roiOrig.area() == 0) { roiOrig = defaultRoi(frame); defaultUsed = true; }
  g_roiState.applied = roiOrig;
  g_roiState.draft = Rect();
  g_roiState.dirty = false;
  g_roiState.defaultUsed = defaultUsed;
  g_prewarpState = prewarpFromConfig(cfg, frame);
  if (autoDemo) {
    g_roiState.applied = normalizedRect(Rect(frame.cols/4, frame.rows/3, frame.cols/2, frame.rows/2), frame);
    g_roiState.dirty = true;
    if (!autoDemoNoPrewarp) {
      g_prewarpState.enabled = true;
      g_prewarpState.ptsOrig = {
        Point2f(frame.cols*0.1f, frame.rows*0.2f),
        Point2f(frame.cols*0.9f, frame.rows*0.15f),
        Point2f(frame.cols*0.8f, frame.rows*0.85f),
        Point2f(frame.cols*0.2f, frame.rows*0.9f)
      };
      g_prewarpState.dirty = true;
    } else {
      g_prewarpState.enabled = false;
      g_prewarpState.dirty = true;
    }
    g_saveRequested = true;
    g_saveAndExitRequested = true;
    defaultUsed = false;
  }

  while (true) {
    if (playState == STATE_PLAYING) {
      if (!cap.read(frame) || frame.empty()) { playState = STATE_STOPPED; continue; }
      frameIndex++;
    } else if (playState == STATE_STOPPED) {
      cap.set(CAP_PROP_POS_FRAMES, 0);
      cap.read(frame);
      frameIndex = 0;
      playState = STATE_PAUSED;
    }

    mapper.setOriginal(frame.cols, frame.rows);

    if (g_prewarpState.dirty && !g_prewarpState.editing) {
      if (g_prewarpState.ptsOrig.empty()) g_prewarpState.ptsOrig = defaultPrewarpPts(frame);
      ensurePrewarpHomography(g_prewarpState, frame.size());
      g_prewarpState.dirty = false;
    }

    Rect currentOrig = g_roiState.draft.area() > 0 ? g_roiState.draft : g_roiState.applied;
    currentOrig = normalizedRect(currentOrig, frame);
    if (currentOrig.area() == 0) { currentOrig = defaultRoi(frame); defaultUsed = true; }

    Mat shown = g_prewarpState.enabled ? applyPrewarpDisplay(frame, g_prewarpState) : frame;
    resize(shown, canvas, Size(mapper.dispW, mapper.dispH));
    Rect dispR = mapper.origToDisp(currentOrig);
    display = canvas.clone();
    // draw prewarp points overlay
    if (g_prewarpState.ptsOrig.size() == 4) {
      for (size_t i=0;i<g_prewarpState.ptsOrig.size();++i) {
        Point pd = mapper.origToDisp(Point(static_cast<int>(g_prewarpState.ptsOrig[i].x), static_cast<int>(g_prewarpState.ptsOrig[i].y)));
        circle(display, pd, 5, Scalar(0,255,255), FILLED);
        putText(display, to_string(static_cast<int>(i+1)), Point(pd.x+6, pd.y-6), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255),1,LINE_AA);
      }
    }
    drawButtons(display, buttons, playState);
    overlayInfo(display, dispR, currentOrig, mapper, cfg.path.empty() ? confPath : cfg.path, g_roiState.dirty, defaultUsed, g_prewarpState);
    imshow("alpr-tool roi", display);
    int key = waitKey(30);
    if (key == 'q' || key == 27) break;
    if (key == ' ') playState = (playState == STATE_PLAYING ? STATE_PAUSED : STATE_PLAYING);
    if (key == 's') { g_roiState.dirty = true; g_saveRequested = true; }
    if (key == 'x') { g_roiState.dirty = true; g_saveRequested = true; g_saveAndExitRequested = true; }
    if (key == 'p') { g_prewarpState.enabled = !g_prewarpState.enabled; g_prewarpState.dirty = true; }
    if (key == 'e') { g_prewarpState.editing = !g_prewarpState.editing; }
    if (key == 'r') { g_roiState.draft = Rect(); g_roiState.applied = Rect(); g_roiState.dirty = false; g_prewarpState = prewarpFromConfig(cfg, frame); defaultUsed = false; }
    if (key == '1') { g_roiState.applied = defaultRoi(frame); g_roiState.draft = Rect(); g_roiState.dirty = true; defaultUsed = true; }
    if (g_quitRequested) { cout << "EXITING ROI TOOL\n"; break; }

    if (g_prewarpState.editing && g_prewarpState.ptsOrig.size() == 4) {
      g_prewarpState.editing = false;
      g_prewarpState.dirty = true;
    }

    if ((g_roiState.dirty || g_prewarpState.dirty) && g_saveRequested) {
      Rect saveR = g_roiState.draft.area() > 0 ? normalizedRect(g_roiState.draft, frame) : currentOrig;
      saveRoiToConfig(saveR, frame, cfg);
      ensurePrewarpHomography(g_prewarpState, frame.size());
      savePrewarpToConfig(g_prewarpState, frame, cfg);
      cfg.save();
      g_roiState.applied = saveR;
      g_roiState.draft = Rect();
      g_roiState.dirty = false;
      g_saveRequested = false;
      defaultUsed = false;
      cout << "ROI saved to " << cfg.lastWritePath << " (percent)"
           << " px=(" << saveR.x << "," << saveR.y << "," << saveR.width << "," << saveR.height << ")"
           << " perc=("
           << static_cast<float>(saveR.x)/frame.cols << ","
           << static_cast<float>(saveR.y)/frame.rows << ","
           << static_cast<float>(saveR.width)/frame.cols << ","
           << static_cast<float>(saveR.height)/frame.rows << ")\n";
      cout << "orig=" << frame.cols << "x" << frame.rows
           << " disp=" << mapper.dispW << "x" << mapper.dispH
           << " scale=" << mapper.scale << " off_x=" << mapper.offX << " off_y=" << mapper.offY << "\n";
      if (g_prewarpState.valid && g_prewarpState.ptsOrig.size() == 4) {
        cout << "PREWARP pts(px)="
             << " (" << g_prewarpState.ptsOrig[0].x << "," << g_prewarpState.ptsOrig[0].y << ")"
             << " (" << g_prewarpState.ptsOrig[1].x << "," << g_prewarpState.ptsOrig[1].y << ")"
             << " (" << g_prewarpState.ptsOrig[2].x << "," << g_prewarpState.ptsOrig[2].y << ")"
             << " (" << g_prewarpState.ptsOrig[3].x << "," << g_prewarpState.ptsOrig[3].y << ")" << "\n";
        cout << "PREWARP pts(%)="
             << " (" << g_prewarpState.ptsOrig[0].x/frame.cols << "," << g_prewarpState.ptsOrig[0].y/frame.rows << ")"
             << " (" << g_prewarpState.ptsOrig[1].x/frame.cols << "," << g_prewarpState.ptsOrig[1].y/frame.rows << ")"
             << " (" << g_prewarpState.ptsOrig[2].x/frame.cols << "," << g_prewarpState.ptsOrig[2].y/frame.rows << ")"
             << " (" << g_prewarpState.ptsOrig[3].x/frame.cols << "," << g_prewarpState.ptsOrig[3].y/frame.rows << ")" << "\n";
        cout << "PREWARP enabled=" << (g_prewarpState.enabled ? "1" : "0") << "\n";
        string planar = derivePlanarStringFromHomography(g_prewarpState.homography, frame.size());
        if (!planar.empty()) cout << "PREWARP planar=" << planar << "\n";
      }
      if (g_saveAndExitRequested) {
        cout << "SAVE & EXIT requested\n";
        break;
      }
    }

    if (key == 'p') playState = STATE_PAUSED;
    if (g_saveAndExitRequested && !g_saveRequested && !g_roiState.dirty && !g_prewarpState.dirty) break;
  }
  try { destroyWindow("alpr-tool roi"); } catch (...) { cv::destroyAllWindows(); }
}

struct PreprocParams {
  int brightness=0;      // slider -100..100
  int contrast=100;      // slider 0..200 -> 0..2
  int gamma=100;         // slider 10..300 -> 0.1..3
  int claheEnable=0;
  int claheClip=200;     // 0.1..4
  int sharpen=0;         // 0..100 -> 0..1
  int denoise=0;         // 0..50
};

static float sliderVal(int v, float scale, float offset=0.0f) { return v * scale + offset; }

static void applyPreprocFrame(const PreprocParams& p, Mat& color, Mat& gray, const Rect& roi) {
  Rect r = roi.area() > 0 ? roi : Rect(0,0,color.cols,color.rows);
  Mat c = color(r);
  Mat g = gray(r);
  float alpha = p.contrast / 100.0f;
  float beta = p.brightness;
  c.convertTo(c, -1, alpha, beta);
  g.convertTo(g, -1, alpha, beta);
  if (p.gamma != 100) {
    float gamma = p.gamma / 100.0f;
    Mat lut(1,256,CV_8U);
    for (int i=0;i<256;i++) lut.at<uchar>(i) = saturate_cast<uchar>(pow(i/255.0f, 1.0f/gamma)*255.0f);
    LUT(c, lut, c);
    LUT(g, lut, g);
  }
  if (p.claheEnable) {
    Ptr<CLAHE> clahe = createCLAHE(std::max(0.1f, p.claheClip/100.0f), Size(8,8));
    clahe->apply(g, g);
  }
  if (p.sharpen > 0) {
    float k = p.sharpen/100.0f;
    Mat b;
    GaussianBlur(g, b, Size(0,0), 1.0);
    addWeighted(g, 1.0+k, b, -k, 0, g);
  }
  if (p.denoise > 0) {
    Mat tmp;
    cv::fastNlMeansDenoising(g, tmp, p.denoise);
    tmp.copyTo(g);
  }
}

static void cmdTune(const string& source, const string& confPath) {
  ConfigWriter cfg;
  cfg.load(confPath);
  string src = source;
  if (src.empty()) src = cfg.get("video_source", "");
  if (src.empty()) {
    cout << "Enter video source (rtsp/device/video path): ";
    getline(cin, src);
  }
  VideoCapture cap;
  if (!openCapture(src, cap)) {
    cerr << "Could not open source: " << src << endl;
    return;
  }
  namedWindow("alpr-tool tune", WINDOW_NORMAL);
  PreprocParams p;
  createTrackbar("brightness", "alpr-tool tune", &p.brightness, 200); // center 0..200
  setTrackbarPos("brightness", "alpr-tool tune", 100);
  createTrackbar("contrast", "alpr-tool tune", &p.contrast, 200);
  createTrackbar("gamma", "alpr-tool tune", &p.gamma, 300);
  setTrackbarPos("gamma", "alpr-tool tune", 100);
  createTrackbar("clahe_enable", "alpr-tool tune", &p.claheEnable, 1);
  createTrackbar("clahe_clipx100", "alpr-tool tune", &p.claheClip, 400);
  createTrackbar("sharpen", "alpr-tool tune", &p.sharpen, 100);
  createTrackbar("denoise", "alpr-tool tune", &p.denoise, 50);

  Rect roi;
  bool showProcessed = true;
  while (true) {
    Mat frame;
    if (!cap.read(frame) || frame.empty()) break;
    if (roi.area() == 0) roi = roiFromConfig(cfg, frame);
    Mat gray;
    if (frame.channels() > 1) cvtColor(frame, gray, COLOR_BGR2GRAY); else gray = frame.clone();
    Mat processed = frame.clone();
    Mat processedGray = gray.clone();

    PreprocParams cur = p;
    cur.brightness -= 100; // center at 0
    applyPreprocFrame(cur, processed, processedGray, roi);

    Mat display = showProcessed ? processed : frame;
    if (roi.area() > 0) rectangle(display, roi, Scalar(0,255,0), 2);
    putText(display, "[SPACE] toggle original/processed | S save | C disable | Q quit", Point(10,20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255),1,LINE_AA);
    imshow("alpr-tool tune", display);
    char key = (char)waitKey(10);
    if (key == 'q' || key == 27) break;
    if (key == ' ') showProcessed = !showProcessed;
    if (key == 'c') {
      cfg.set("preproc_enable","0");
      cfg.save();
      cout << "Preproc disabled (preproc_enable=0)\n";
    }
    if (key == 's') {
      cfg.set("preproc_enable","1");
      cfg.set("preproc_brightness", to_string(cur.brightness));
      cfg.set("preproc_contrast", to_string(cur.contrast/100.0f));
      cfg.set("preproc_gamma", to_string(cur.gamma/100.0f));
      cfg.set("preproc_clahe_enable", to_string(cur.claheEnable));
      cfg.set("preproc_clahe_clip", to_string(cur.claheClip/100.0f));
      cfg.set("preproc_sharpen", to_string(cur.sharpen/100.0f));
      cfg.set("preproc_denoise", to_string(cur.denoise));
      cfg.save();
      cout << "Preproc saved to " << confPath << endl;
    }
  }
  destroyWindow("alpr-tool tune");
}

static void drawResults(Mat& frame, const AlprResults& results) {
  for (const auto& plate : results.plates) {
    for (int i=0;i<4;i++) {
      Point p1(plate.plate_points[i].x, plate.plate_points[i].y);
      Point p2(plate.plate_points[(i+1)%4].x, plate.plate_points[(i+1)%4].y);
      line(frame, p1, p2, Scalar(0,0,255), 2);
    }
    std::ostringstream oss;
    oss << plate.bestPlate.characters << " (" << plate.bestPlate.overall_confidence << ")";
    putText(frame, oss.str(), Point(plate.plate_points[0].x, plate.plate_points[0].y-5), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0),2,LINE_AA);
  }
}

static double bboxCenterY(const AlprPlateResult& p) {
  double cy = 0.0;
  for (int i=0;i<4;i++) cy += p.plate_points[i].y;
  return cy / 4.0;
}

static Rect plateRect(const AlprPlateResult& p) {
  int minx=INT_MAX, miny=INT_MAX, maxx=0, maxy=0;
  for (int i=0;i<4;i++) {
    minx = std::min(minx, p.plate_points[i].x);
    miny = std::min(miny, p.plate_points[i].y);
    maxx = std::max(maxx, p.plate_points[i].x);
    maxy = std::max(maxy, p.plate_points[i].y);
  }
  return Rect(minx, miny, maxx-minx, maxy-miny);
}

static std::string cwdPath() {
  char buf[4096];
  if (getcwd(buf, sizeof(buf))) return std::string(buf);
  return ".";
}

static std::string joinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

static bool cascadeLoadable(const std::string& cascadePath) {
  cv::CascadeClassifier cc;
  return cc.load(cascadePath);
}

static bool ensureDir(const std::string& path) {
  if (path.empty()) return false;
  if (DirectoryExists(path.c_str())) return true;
  return makePath(path.c_str(), 0755);
}

static bool writeConfigFile(const std::string& path, const std::string& runtimeDir, const std::string& country) {
  std::ofstream out(path);
  if (!out.good()) return false;
  out << "; Auto-generated by alpr-tool doctor\n";
  out << "runtime_dir = " << runtimeDir << "\n";
  out << "country = " << country << "\n";
  out << "detector = lbpcpu\n";
  out << "skip_detection = 0 ; set to 1 to disable detection and use provided ROIs\n";
  out << "debug_general = 0\n";
  out << "debug_detector = 0\n";
  out << "debug_ocr = 0\n";
  return true;
}

static bool writePerformanceConfig(const std::string& path, const std::string& runtimeDir, const std::string& country) {
  std::ofstream out(path);
  if (!out.good()) return false;
  out << "; Auto-generated by alpr-tool doctor (performance preset)\n";
  out << "; Focused on minimal logging and classic detector\n";
  out << "runtime_dir = " << runtimeDir << "\n";
  out << "country = " << country << "\n";
  out << "detector = lbpcpu\n";
  out << "skip_detection = 0 ; set to 1 to disable detection and use provided ROIs\n";
  out << "debug_general = 0\n";
  out << "debug_detector = 0\n";
  out << "debug_ocr = 0\n";
  out << "debug_postprocess = 0\n";
  out << "debug_show_images = 0\n";
  out << "debug_timing = 0\n";
  return true;
}

static DoctorResult runDoctor(const std::string& country, const std::string& outDir) {
  DoctorResult dr;
  RuntimeResolveResult rt = resolveRuntimeData(country, "");
  if (rt.preferredInvalid) {
    std::cerr << "[warn] runtime_data from config invalid for country=" << country << ": " << rt.preferredReason << "; trying fallbacks...\n";
  }
  if (!rt.ok) {
    std::cerr << "[error] Could not resolve runtime_data for country=" << country << std::endl;
    if (!rt.reason.empty()) std::cerr << " reason: " << rt.reason << std::endl;
    std::cerr << " tried: ";
    for (size_t i=0;i<rt.tested.size();i++) {
      if (i>0) std::cerr << ", ";
      std::cerr << rt.tested[i];
    }
    std::cerr << "\nPlease install openalpr runtime_data containing region/*.xml and ocr/.\n";
    return dr;
  }
  ensureDir(outDir);
  ensureDir("artifacts");
  ensureDir("artifacts/logs");
  std::string baseCountryCfg = outDir + "/openalpr." + country + ".conf";
  std::string baseDefaultCfg = outDir + "/openalpr.default.conf";
  std::string basePerfCfg = outDir + "/openalpr.performance.conf";
  if (!writeConfigFile(baseCountryCfg, rt.path, country)) {
    std::cerr << "[error] Failed to write config " << baseCountryCfg << std::endl;
    return dr;
  }
  if (!writeConfigFile(baseDefaultCfg, rt.path, country)) {
    std::cerr << "[error] Failed to write config " << baseDefaultCfg << std::endl;
    return dr;
  }
  if (!writePerformanceConfig(basePerfCfg, rt.path, country)) {
    std::cerr << "[error] Failed to write config " << basePerfCfg << std::endl;
    return dr;
  }
  // optional br2
  std::string br2Cascade = joinPath(joinPath(rt.path, "region"), "br2.xml");
  if (fileExists(br2Cascade.c_str())) {
    std::string br2Cfg = outDir + "/openalpr.br2.conf";
    if (!writeConfigFile(br2Cfg, rt.path, "br2")) {
      std::cerr << "[error] Failed to write config " << br2Cfg << std::endl;
      return dr;
    }
  }
  // list available countries (first 20)
  std::vector<std::string> regionFiles = getFilesInDir((rt.path + "/region").c_str());
  size_t total = regionFiles.size();
  std::cout << "[doctor] available countries (first 20): ";
  for (size_t i=0;i<regionFiles.size() && i<20;i++) {
    if (i>0) std::cout << ", ";
    std::string f = regionFiles[i];
    if (f.find(".xml") != std::string::npos) f = f.substr(0, f.find(".xml"));
    std::cout << f;
  }
  if (total > 20) std::cout << " ... (" << total << " total)";
  std::cout << std::endl;
  bool countryAvailable = false;
  for (const auto& f : regionFiles) {
    if (f.find(country + ".xml") != std::string::npos) { countryAvailable = true; break; }
  }
  if (!countryAvailable && !regionFiles.empty()) {
    std::string suggestion = regionFiles[0];
    if (suggestion.find(".xml") != std::string::npos) suggestion = suggestion.substr(0, suggestion.find(".xml"));
    std::cerr << "[warn] requested country '" << country << "' not found; try --country " << suggestion << std::endl;
    return dr;
  }
  dr.ok = true;
  dr.runtimePath = rt.path;
  if (countryAvailable && fileExists(baseCountryCfg.c_str()))
    dr.confPath = baseCountryCfg;
  else
    dr.confPath = baseDefaultCfg;
  std::cout << "[doctor] configs written to " << outDir << std::endl;
  std::cout << "[doctor] runtime_data_path_resolved=" << rt.path << " (auto selected)" << std::endl;
  return dr;
}

static RuntimeResolveResult resolveRuntimeData(const std::string& country, const std::string& preferred) {
  RuntimeResolveResult rr;
  std::vector<std::string> candidates;
  auto pushIfUnique = [&](const std::string& p){
    if (p.empty()) return;
    if (std::find(candidates.begin(), candidates.end(), p) == candidates.end())
      candidates.push_back(p);
  };
  bool hasPreferred = !preferred.empty();
  pushIfUnique(preferred);
  const char* envRt = getenv("OPENALPR_RUNTIME_DATA");
  if (envRt) pushIfUnique(std::string(envRt));
  pushIfUnique("/usr/share/openalpr/runtime_data");
  pushIfUnique("/usr/local/share/openalpr/runtime_data");
  pushIfUnique("./runtime_data");
  pushIfUnique(joinPath(cwdPath(), "runtime_data"));
  // try repo root heuristic: if we are inside build/* go up one
  std::string up = joinPath(cwdPath(), "../runtime_data");
  pushIfUnique(up);

  bool firstTried = true;
  for (const auto& base : candidates) {
    rr.tested.push_back(base);
    std::string regionDir = joinPath(base, "region");
    std::string cascade = joinPath(regionDir, country + ".xml");
    if (!DirectoryExists(base.c_str())) { rr.reason = "runtime_data path missing"; goto next; }
    if (!DirectoryExists(regionDir.c_str())) { rr.reason = "region dir missing"; goto next; }
    if (!fileExists(cascade.c_str())) { rr.reason = "cascade file missing: " + cascade; goto next; }
    if (!cascadeLoadable(cascade)) { rr.reason = "cascade not loadable: " + cascade; goto next; }
    rr.ok = true;
    rr.path = base;
    return rr;
next:
    if (firstTried && hasPreferred) {
      rr.preferredInvalid = true;
      rr.preferredReason = rr.reason;
    }
    firstTried = false;
  }
  return rr;
}

static bool getTimeSeconds(VideoCapture& cap, int frameIdx, double fpsReported, bool fpsValid, double& tOut) {
  double tsMs = cap.get(CAP_PROP_POS_MSEC);
  if (tsMs > 0) {
    tOut = tsMs / 1000.0;
    return true;
  }
  if (fpsValid && fpsReported > 0) {
    tOut = frameIdx / fpsReported;
    return true;
  }
  return false;
}

static bool isValidMercosul(const string& plate) {
  static const std::regex merc("^([A-Z]{3}[0-9][A-Z][0-9]{2})$");
  return std::regex_match(plate, merc);
}
static bool isValidOldBr(const string& plate) {
  static const std::regex old("^([A-Z]{3}[0-9]{4})$");
  return std::regex_match(plate, old);
}

static void cmdPreview(const string& source, const string& confPath, const string& logPath, bool selfTest, PreviewRuntimeOptions opts, const string& countryArg, bool doctorMode, bool doctorAlreadyRan) {
  ConfigWriter cfg;
  cfg.load(confPath);
  SpeedConfig speedCfg = loadSpeedConfig(cfg);
  string src = source.empty() ? cfg.get("video_source","") : source;
  string countryCfg = cfg.get("country","br2");
  string country = countryArg.empty() ? countryCfg : countryArg;
  if (src.empty()) {
    cout << "Enter video source (rtsp/device/video path): ";
    getline(cin, src);
  }
  cout << "[config] conf_path=" << confPath << "\n";
  cout << "[config] runtime_data_path=" << cfg.get("runtime_dir","") << "\n";
  cout << "[config] country=" << country << "\n";
  string skipDet = cfg.get("skip_detection","0");
  cout << "[config] skip_detection=" << skipDet << "\n";
  if (cfg.get("ocr_only_after_crossing","0") == "1") opts.ocrOnlyAfterCrossing = true;
  if (cfg.get("log_ocr_metrics","0") == "1") opts.logOcrMetrics = true;
  if (cfg.get("log_crossing_metrics","0") == "1") opts.logCrossingMetrics = true;
  if (opts.crossingMode != "off" && opts.crossingMode != "motion") {
    cerr << "Invalid --crossing-mode (expected off|motion)\n";
    return;
  }
  RuntimeResolveResult rt = resolveRuntimeData(country, cfg.get("runtime_dir",""));
  if (rt.preferredInvalid) {
    cerr << "[warn] runtime_data from config invalid for country=" << country << ": " << rt.preferredReason << "; trying fallbacks...\n";
  }
  if (!rt.ok) {
    cerr << "[error] Could not resolve runtime_data for country=" << country << endl;
    if (!rt.reason.empty()) cerr << " reason: " << rt.reason << endl;
    cerr << " tried: ";
    for (size_t i=0;i<rt.tested.size();i++) {
      if (i>0) cerr << ", ";
      cerr << rt.tested[i];
    }
    cerr << "\nPlease install openalpr runtime_data or point --conf runtime_dir to a valid path containing region/*.xml and ocr/.\n";
    return;
  }
  cout << "[config] runtime_data_path_resolved=" << rt.path << " (auto selected)\n";
  cout << "[config] runtime_data_path_resolved=" << rt.path << " (auto selected)\n";
  VideoCapture cap;
  if (!openCapture(src, cap)) {
    cerr << "Could not open source: " << src << endl;
    return;
  }
  Alpr alpr(country, confPath, rt.path);
  if (!alpr.isLoaded()) {
    cerr << "Could not load ALPR with config: " << confPath << endl;
    if (doctorMode && !doctorAlreadyRan) {
      cerr << "[doctor] detector not loaded, running auto-setup...\n";
      DoctorResult dr = runDoctor(country, "artifacts/configs");
      if (!dr.ok) {
        cerr << "[doctor] auto-setup failed; aborting.\n";
        return;
      }
      std::string newConf = dr.confPath.empty() ? confPath : dr.confPath;
      cout << "[doctor] using generated config: " << newConf << "\n";
      cmdPreview(source, newConf, logPath, selfTest, opts, countryArg, doctorMode, true);
      return;
    }
    return;
  }
  if (alpr.getConfig()) {
    cout << "[config] runtime_data_path_resolved=" << alpr.getConfig()->getRuntimeBaseDir() << "\n";
  }
    if (selfTest) speedCfg.enabled = true;
  ofstream logFile;
  if (!logPath.empty()) {
    ensureParentDir(logPath);
    logFile.open(logPath, ios::out | ios::app);
    if (!logFile.good()) cerr << "Could not open log file: " << logPath << endl;
  }
  ofstream plateLogFile;
  if (!opts.logPlatesFile.empty()) {
    ensureParentDir(opts.logPlatesFile);
    plateLogFile.open(opts.logPlatesFile, ios::out | ios::trunc);
    if (!plateLogFile.good()) cerr << "Could not open plate log file: " << opts.logPlatesFile << endl;
  }
  auto logLine = [&](const std::string& s){
    cout << s << endl;
    if (logFile.good()) logFile << s << "\n";
  };
  auto logPlateLine = [&](const std::string& s){
    if (plateLogFile.good()) plateLogFile << s << "\n";
    else cout << s << endl;
  };
  auto plateLog = [&](const std::string& s){
    logLine(s);
    logPlateLine(s);
  };
  namedWindow("alpr-tool preview", WINDOW_NORMAL);
  Rect roi;
  bool defaultUsed=false;
  int64 lastTick = getTickCount();
  int frameIdx = 0;
  double fpsReported = cap.get(CAP_PROP_FPS);
  bool fpsValid = fpsReported > 1.0 && fpsReported < 300.0;
  vector<Track> tracks;
  int nextTrackId = 1;
  const double iouThreshold = 0.3;
  const double throttleSec = std::max(0, opts.logThrottleMs) / 1000.0;
  const double trackTtlSec = std::max(1, opts.trackTtlMs) / 1000.0;
  const double tickFreq = getTickFrequency();
  auto wallSeconds = [&](){ return static_cast<double>(getTickCount()) / tickFreq; };
  const bool trackingActive = speedCfg.enabled || opts.logPlates;
  const std::string detectorLabel = (cfg.get("skip_detection","0") == "1") ? "skip" : "classic";
  bool detectorLogged = false;
  double startWall = wallSeconds();
  auto wallClockStart = std::chrono::system_clock::now();
  int framesTotal = 0;
  int ocrCallsTotal = 0;
  int ocrCallsPostCrossing = 0;
  int platesFound = 0;
  int platesNone = 0;
  int platesFoundPostCrossing = 0;
  int platesNonePostCrossing = 0;
  bool crossingEnabled = (opts.crossingMode == "motion");
  if (crossingEnabled && (opts.crossingP1 == opts.crossingP2)) {
    cerr << "crossing-mode=motion requires --line x1,y1,x2,y2\n";
    return;
  }
  Mat prevGray;
  int lastStableSide = 0;
  int sideStreakSide = 0;
  int sideStreakCount = 0;
  int crossingFrame = -1;
  while (true) {
    Mat frame;
    if (!cap.read(frame) || frame.empty()) break;
    frameIdx++;
    framesTotal++;
    if (opts.maxSeconds > 0) {
      double elapsed = wallSeconds() - startWall;
      if (elapsed >= opts.maxSeconds) break;
    }
    if (roi.area() == 0) {
      roi = roiFromConfig(cfg, frame);
      if (roi.area() == 0) { roi = defaultRoi(frame); defaultUsed=true; }
    }
    auto clampRect = [&](Rect r)->Rect{
      int x = std::max(0, std::min(r.x, frame.cols-1));
      int y = std::max(0, std::min(r.y, frame.rows-1));
      int w = std::min(frame.cols - x, std::max(0, r.width));
      int h = std::min(frame.rows - y, std::max(0, r.height));
      return Rect(x,y,w,h);
    };
    Rect crossingRoi = opts.crossingRoiProvided ? clampRect(opts.crossingRoi) : Rect(0,0,frame.cols, frame.rows);
    Rect alprRoi = opts.alprRoiProvided ? clampRect(opts.alprRoi) : Rect(0,0,frame.cols, frame.rows);
    vector<AlprRegionOfInterest> rois;
    if (alprRoi.area() > 0) rois.push_back(AlprRegionOfInterest(alprRoi.x, alprRoi.y, alprRoi.width, alprRoi.height));
    Mat bgr;
    if (frame.channels() == 1) cvtColor(frame, bgr, COLOR_GRAY2BGR); else bgr = frame;
    if (!bgr.isContinuous()) bgr = bgr.clone();
    double lineA = (roi.area() > 0 ? roi.y + speedCfg.yA * roi.height : speedCfg.yA * frame.rows);
    double lineB = (roi.area() > 0 ? roi.y + speedCfg.yB * roi.height : speedCfg.yB * frame.rows);

    bool motionDetected = false;
    bool crossingEvent = false;
    int currentSide = 0;
    bool gatedByCrossing = opts.ocrOnlyAfterCrossing && crossingFrame < 0;
    bool ocrRan = !gatedByCrossing;
    if (crossingEnabled) {
      Mat gray;
      cvtColor(frame, gray, COLOR_BGR2GRAY);
      GaussianBlur(gray, gray, Size(5,5), 0);
      Rect r = clampRect(crossingRoi);
      if (r.area() > 0) gray = gray(r);
      if (!prevGray.empty()) {
        Mat diff, thresh;
        absdiff(gray, prevGray, diff);
        threshold(diff, thresh, opts.motionThresh, 255, THRESH_BINARY);
        dilate(thresh, thresh, Mat(), Point(-1,-1), 2);
        vector<vector<Point>> contours;
        findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        double maxArea = 0.0;
        vector<Point> best;
        for (const auto& c : contours) {
          double a = contourArea(c);
          if (a > maxArea) { maxArea = a; best = c; }
        }
        int motionPixels = !thresh.empty() ? countNonZero(thresh) : 0;
        double roiArea = static_cast<double>(r.area());
        double motionRatio = (roiArea > 0) ? (motionPixels / roiArea) : 0.0;
        if (maxArea >= opts.motionMinArea && !best.empty() && motionRatio >= opts.motionMinRatio) {
          motionDetected = true;
          Moments m = moments(best);
          if (m.m00 != 0) {
            Point2f c(static_cast<float>(m.m10/m.m00), static_cast<float>(m.m01/m.m00));
            Point2f cGlobal(c.x + r.x, c.y + r.y);
            Point2f p1Local(static_cast<float>(opts.crossingP1.x - r.x), static_cast<float>(opts.crossingP1.y - r.y));
            Point2f p2Local(static_cast<float>(opts.crossingP2.x - r.x), static_cast<float>(opts.crossingP2.y - r.y));
            double cross = (p2Local.x - p1Local.x)*(c.y - p1Local.y) - (p2Local.y - p1Local.y)*(c.x - p1Local.x);
            if (cross > 0) currentSide = 1;
            else if (cross < 0) currentSide = -1;
            bool dirOk = true;
            static Point2f prevCentroid(0,0);
            static bool hasPrev = false;
            if (opts.motionDirectionFilter) {
              if (hasPrev) {
                double dx = cGlobal.x - prevCentroid.x;
                double dy = cGlobal.y - prevCentroid.y;
                double ldx = opts.crossingP2.x - opts.crossingP1.x;
                double ldy = opts.crossingP2.y - opts.crossingP1.y;
                double nx = -ldy;
                double ny = ldx;
                double projNormal = dx*nx + dy*ny;
                double projLine = dx*ldx + dy*ldy;
                dirOk = std::abs(projNormal) > std::abs(projLine);
              }
            }
            prevCentroid = cGlobal;
            hasPrev = true;

            if (dirOk && motionRatio >= opts.crossingArmMinRatio) {
              static int armCount = 0;
              armCount++;
              if (armCount >= opts.crossingArmMinFrames) {
                if (motionDetected && currentSide != 0) {
                  if (currentSide == sideStreakSide) sideStreakCount++;
                  else { sideStreakSide = currentSide; sideStreakCount = 1; }
                  if (lastStableSide == 0 && sideStreakCount >= opts.crossingDebounce) {
                    lastStableSide = sideStreakSide;
                  } else if (lastStableSide != 0 && sideStreakSide != lastStableSide && sideStreakCount >= opts.crossingDebounce) {
                    crossingEvent = true;
                    lastStableSide = sideStreakSide;
                    if (crossingFrame < 0) {
                      std::ostringstream coss;
                      coss << "[crossing] frame=" << frameIdx
                           << " ratio=" << motionRatio
                           << " area=" << maxArea
                           << " dir_ok=" << (dirOk ? 1 : 0);
                      logLine(coss.str());
                    }
                  }
                }
              }
            }
          }
        }
      }
      gray.copyTo(prevGray);
      if (crossingEvent && crossingFrame < 0) {
        crossingFrame = frameIdx;
      }
      gatedByCrossing = opts.ocrOnlyAfterCrossing && crossingFrame < 0;
      ocrRan = !gatedByCrossing;
    }

    AlprResults results;
    if (opts.maxSeconds > 0 && (wallSeconds() - startWall) >= opts.maxSeconds) break;

    if (!ocrRan) {
      results.total_processing_time_ms = 0;
      results.img_width = frame.cols;
      results.img_height = frame.rows;
    } else if (selfTest) {
      results.total_processing_time_ms = 0;
      results.img_width = frame.cols;
      results.img_height = frame.rows;
      int startY = static_cast<int>(std::max(5.0, lineA - 80.0));
      int endY   = static_cast<int>(std::min((roi.area() > 0 ? roi.y + roi.height - 5.0 : frame.rows - 5.0), lineB + 120.0));
      if (endY <= startY) endY = startY + 50;
      int cycle = 90;
      for (int i=0;i<3;i++) {
        AlprPlateResult pr;
        double prog = ((frameIdx + i*15) % cycle) / static_cast<double>(cycle-1);
        int cy = startY + static_cast<int>((endY - startY) * prog);
        int cx = frame.cols/2 + (i-1)*80;
        int bw = 100, bh = 50;
        pr.plate_points[0].x = cx - bw/2; pr.plate_points[0].y = cy - bh/2;
        pr.plate_points[1].x = cx + bw/2; pr.plate_points[1].y = cy - bh/2;
        pr.plate_points[2].x = cx + bw/2; pr.plate_points[2].y = cy + bh/2;
        pr.plate_points[3].x = cx - bw/2; pr.plate_points[3].y = cy + bh/2;
        std::ostringstream t; t << "SELF" << (i+1);
        pr.bestPlate.characters = t.str();
        pr.bestPlate.overall_confidence = 99.0;
        results.plates.push_back(pr);
      }
    } else {
      results = alpr.recognize(bgr.data, bgr.elemSize(), bgr.cols, bgr.rows, rois);
    }
    double tNow = 0.0;
    bool tOk = getTimeSeconds(cap, frameIdx, fpsReported, fpsValid, tNow);
    if (!tOk) {
      if (fpsValid && fpsReported > 0) {
        tNow = frameIdx / fpsReported;
        tOk = true;
      } else {
        tNow = wallSeconds();
        tOk = true;
      }
    }

    if (ocrRan) {
      ocrCallsTotal++;
    }
    bool anyDetections = !results.plates.empty();
    bool plateFoundThisFrame = false;
    if (!results.plates.empty()) {
      for (const auto& plate : results.plates) {
        if (!plate.bestPlate.characters.empty()) { plateFoundThisFrame = true; break; }
      }
    }
    bool isPostCrossing = (crossingFrame >= 0 && frameIdx >= crossingFrame);
    bool crossedNow = crossingEvent;
    if (ocrRan) {
      if (plateFoundThisFrame) platesFound++; else platesNone++;
    }
    if (trackingActive) {
      for (const auto& plate : results.plates) {
        Rect pr = plateRect(plate);
        Point c((pr.x+pr.width/2), (pr.y+pr.height/2));
        double cy = c.y;
        int bestIdx = -1;
        double bestIou = iouThreshold;
        for (size_t i=0;i<tracks.size();++i) {
          double iou = iouRect(tracks[i].last_bbox, pr);
          if (iou > bestIou) { bestIou = iou; bestIdx = static_cast<int>(i); }
        }
        if (bestIdx == -1) {
          if (static_cast<int>(tracks.size()) >= opts.maxTracks) continue;
          Track nt;
          nt.id = nextTrackId++;
          nt.last_bbox = pr;
          nt.last_centerY_ema = cy;
          nt.last_seen_t = tNow;
          nt.best_plate_text = plate.bestPlate.characters;
          nt.best_plate_conf = plate.bestPlate.overall_confidence;
          tracks.push_back(nt);
          bestIdx = static_cast<int>(tracks.size()) - 1;
        }
        Track& tr = tracks[bestIdx];
        double prevEma = (tr.last_centerY_ema < 0 ? cy : tr.last_centerY_ema);
        double newEma = cy;
        if (speedCfg.smoothing == "ema") newEma = speedCfg.emaAlpha * cy + (1.0 - speedCfg.emaAlpha) * prevEma;
        tr.last_centerY_ema = newEma;
        tr.last_bbox = pr;
        tr.last_seen_t = tNow;
        if (!plate.bestPlate.characters.empty() && plate.bestPlate.overall_confidence >= tr.best_plate_conf) {
          tr.best_plate_text = plate.bestPlate.characters;
          tr.best_plate_conf = plate.bestPlate.overall_confidence;
        }

        bool inRoi = roi.area() == 0 || roi.contains(c);
        string plateText = !plate.bestPlate.characters.empty() ? plate.bestPlate.characters :
                           (!tr.best_plate_text.empty() ? tr.best_plate_text : string("<none>"));
        double plateConf = plate.bestPlate.overall_confidence >= 0 ? plate.bestPlate.overall_confidence : tr.best_plate_conf;
        string reason = "ok";
        if (plate.bestPlate.characters.empty()) {
          if (plate.topNPlates.size() == 0) reason = "no_candidates";
          else reason = "ocr_empty";
          if (plate.bestPlate.overall_confidence > 0 && plate.bestPlate.overall_confidence < 50) reason = "low_confidence";
        }
        int candidates = static_cast<int>(plate.topNPlates.size());
        string country = plate.region.empty() ? cfg.get("country","") : plate.region;

        if (opts.logPlates && inRoi) {
          bool foundText = !plate.bestPlate.characters.empty();
          bool everyN = (opts.logPlatesEveryN <= 1) || (frameIdx % opts.logPlatesEveryN == 0);
          bool shouldLog = (foundText || everyN);
          if (foundText && tr.last_logged_t >= 0 && (tNow - tr.last_logged_t) < throttleSec && plateText == tr.last_logged_plate_text) {
            shouldLog = false;
          }
          if (shouldLog) {
            std::ostringstream oss;
            oss << "frame=" << frameIdx
                << " track=" << tr.id
                << " plate=" << plateText
                << " conf=" << plateConf
                << " bbox=" << pr.x << "," << pr.y << "," << pr.width << "," << pr.height
                << " country=" << country
                << " candidates=" << candidates
                << " detector=" << detectorLabel;
            if (reason != "ok") oss << " reason=" << reason;
            plateLog(oss.str());
            tr.last_logged_plate_text = plateText;
            tr.last_logged_t = tNow;
          }
        }
        if ((country == "br" || country == "br2") && !plateText.empty() && plateText != "<none>") {
          string norm = plateText;
          std::transform(norm.begin(), norm.end(), norm.begin(), ::toupper);
          string match = "invalid";
          if (isValidMercosul(norm)) match = "mercosul";
          else if (isValidOldBr(norm)) match = "old";
          std::ostringstream oss;
          oss << "[br] plate_candidate=" << norm << " match=" << match;
          logLine(oss.str());
        }

        if (speedCfg.enabled && inRoi && tOk) {
          if (!tr.crossedA && prevEma < lineA && newEma >= lineA) {
            tr.crossedA = true;
            tr.tA = tNow;
            if (opts.logEvents && speedCfg.log) {
              std::ostringstream oss; oss << "frame=" << frameIdx << " track=" << tr.id << " arm=A crossed";
              logLine(oss.str());
            }
          }
          if (tr.crossedA && !tr.fired && prevEma < lineB && newEma >= lineB) {
            double dt = tNow - tr.tA;
            if (dt > 0 && speedCfg.dist_m > 0) {
              double mps = speedCfg.dist_m / dt;
              double kmh = mps * 3.6;
              if (kmh >= speedCfg.minKmh && kmh <= speedCfg.maxKmh) {
                bool plateOk = !speedCfg.requirePlate || !tr.best_plate_text.empty() || !plateText.empty();
                if (plateOk) {
                  tr.fired = true;
                  tr.crossedB = true;
                  tr.tB = tNow;
                  tr.last_speed_kmh = kmh;
                  if (opts.logEvents && speedCfg.log) {
                    std::ostringstream oss;
                    oss << "frame=" << frameIdx
                        << " track=" << tr.id
                        << " plate=" << plateText
                        << " conf=" << plateConf
                        << " speed_kmh=" << kmh
                        << " dt=" << dt
                        << " mode=lines crossed=A->B";
                    logLine(oss.str());
                  }
                }
              }
            }
          }
          if (tr.crossedA || tr.fired) crossedNow = true;
        }
      }

      // Expire old tracks
      tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [&](const Track& t){
        return (tNow - t.last_seen_t) > trackTtlSec;
      }), tracks.end());
      if (opts.logPlates && !anyDetections) {
        bool everyN = (opts.logPlatesEveryN <= 1) || (frameIdx % opts.logPlatesEveryN == 0);
        if (everyN) {
          std::ostringstream oss;
          oss << "frame=" << frameIdx
              << " plate=<none>"
              << " conf=0"
              << " bbox=0,0,0,0"
              << " detector=" << detectorLabel
              << " reason=no_candidates";
          plateLog(oss.str());
        }
      }
    } else {
      drawResults(frame, results);
    }

    bool crossedFrame = isPostCrossing;
    if (ocrRan) {
      if (isPostCrossing) {
        ocrCallsPostCrossing++;
        if (plateFoundThisFrame) platesFoundPostCrossing++; else platesNonePostCrossing++;
      }
    }
    if (opts.logOcrMetrics || opts.logCrossingMetrics) {
      std::ostringstream m;
      std::string plateText = (ocrRan && plateFoundThisFrame && !results.plates.empty()) ? results.plates.front().bestPlate.characters : std::string("<none>");
      std::string reason = plateFoundThisFrame ? "ok" : (gatedByCrossing ? "gated_by_crossing" : "no_candidates");
      m << "frame=" << frameIdx
        << " crossed=" << (crossedFrame ? 1 : 0)
        << " ocr_ran=" << (ocrRan ? 1 : 0)
        << " gated_by_crossing=" << (gatedByCrossing ? 1 : 0)
        << " motion=" << (crossingEvent || motionDetected ? 1 : 0);
      if (opts.logOcrMetrics) {
        m << " plate=" << plateText
          << " reason=" << reason;
      }
      logLine(m.str());
    }

    if (roi.area() > 0) rectangle(frame, roi, Scalar(0,255,0), 2);
    if (speedCfg.enabled) {
      int yApx = static_cast<int>(lineA);
      int yBpx = static_cast<int>(lineB);
      line(frame, Point(0, yApx), Point(frame.cols-1, yApx), Scalar(255,255,0), 1, LINE_AA);
      line(frame, Point(0, yBpx), Point(frame.cols-1, yBpx), Scalar(255,255,0), 1, LINE_AA);
      std::ostringstream so;
      so << "speed lines A=" << speedCfg.yA*100 << "% B=" << speedCfg.yB*100 << "% dist=" << speedCfg.dist_m << "m";
      putText(frame, so.str(), Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255),1,LINE_AA);
    }
    if (trackingActive) {
      for (const auto& tr : tracks) {
        Scalar color = tr.fired ? Scalar(0,255,255) : Scalar(0,0,255);
        rectangle(frame, tr.last_bbox, color, 2);
        std::ostringstream tid; tid << "T" << tr.id;
        putText(frame, tid.str(), Point(tr.last_bbox.x, std::max(0, tr.last_bbox.y-5)), FONT_HERSHEY_SIMPLEX, 0.5, color, 1, LINE_AA);
        if (tr.fired) {
          std::ostringstream ss; ss << std::fixed << setprecision(1) << tr.last_speed_kmh << " km/h";
          putText(frame, ss.str(), Point(tr.last_bbox.x, tr.last_bbox.y + tr.last_bbox.height + 15), FONT_HERSHEY_SIMPLEX, 0.5, color, 2, LINE_AA);
        }
      }
    }
    if (defaultUsed) putText(frame, "ROI DEFAULT (lower half)", Point(10,40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255),1,LINE_AA);
    int64 now = getTickCount();
    double fps = getTickFrequency() / (now - lastTick + 1);
    lastTick = now;
    std::ostringstream oss;
    oss << "FPS: " << fps;
    putText(frame, oss.str(), Point(10,20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255),1,LINE_AA);
    imshow("alpr-tool preview", frame);
    char key = (char)waitKey(1);
    if (key == 'q' || key == 27) break;
  }
  auto wallClockEnd = std::chrono::system_clock::now();
  double wallTimeSeconds = std::chrono::duration<double>(wallClockEnd - wallClockStart).count();
  double fpsReport = (wallTimeSeconds > 0.0) ? (framesTotal / wallTimeSeconds) : 0.0;
  cout << "[report]\n";
  cout << "frames=" << framesTotal << "\n";
  cout << "ocr_calls=" << ocrCallsTotal << "\n";
  cout << "ocr_calls_post_crossing=" << ocrCallsPostCrossing << "\n";
  cout << "plates_found=" << platesFound << "\n";
  cout << "plates_none=" << platesNone << "\n";
  cout << "plates_found_post_crossing=" << platesFoundPostCrossing << "\n";
  cout << "plates_none_post_crossing=" << platesNonePostCrossing << "\n";
  cout << "crossing_frame=" << crossingFrame << "\n";
  int framesAfterCrossing = (crossingFrame >= 0) ? (framesTotal - crossingFrame + 1) : 0;
  cout << "frames_after_crossing=" << framesAfterCrossing << "\n";
  cout << "wall_time_s=" << wallTimeSeconds << "\n";
  cout << "fps=" << fpsReport << "\n";
  if (logFile.good()) logFile.close();
  destroyWindow("alpr-tool preview");
}

static int runCmd(const string& cmd) {
  return system(cmd.c_str());
}

static void cmdExportYolo(const string& model, const string& out, int imgsz, const string& confPath, bool updateConf) {
  if (model.empty() || out.empty()) {
    cerr << "model and out are required\n";
    return;
  }
  std::ostringstream oss;
  oss << "python3 tools/export_yolo.py --model \"" << model << "\" --out \"" << out << "\" --imgsz " << imgsz;
  int rc = runCmd(oss.str());
  if (rc != 0) {
    cerr << "export failed (rc=" << rc << ")\n";
    return;
  }
  cout << "export completed: " << out << endl;
  if (updateConf) {
    if (confPath.empty()) {
      cerr << "--update-conf requires --conf\n";
      return;
    }
    ConfigWriter cfg;
    if (!cfg.load(confPath)) {
      cerr << "Could not load conf for update: " << confPath << endl;
      return;
    }
    cfg.set("detector_type", "yolo");
    cfg.set("yolo_model_path", out);
    cfg.set("yolo_input_width", to_string(imgsz));
    cfg.set("yolo_input_height", to_string(imgsz));
    cfg.save();
    cout << "Config updated with new model path\n";
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cout << "Usage: alpr-tool <roi|tune|preview|export-yolo> [options]\n";
    return 1;
  }
  string sub = argv[1];
  vector<string> subArgs;
  for (int i=2;i<argc;i++) subArgs.push_back(argv[i]);

  try {
    if (sub == "roi") {
      string conf = "./config/openalpr.conf.defaults";
      string src;
      bool autoDemo = false;
      bool autoDemoNoPrewarp = false;
      for (size_t i=0;i<subArgs.size();++i) {
        string a = subArgs[i];
        auto eatValue = [&](string& target){
          if (a.find('=') != string::npos) {
            target = a.substr(a.find('=')+1);
          } else if (i+1 < subArgs.size()) {
            target = subArgs[++i];
          } else {
            throw std::runtime_error("missing value after " + a);
          }
        };
        if (a.rfind("--conf",0)==0) { eatValue(conf); continue; }
        if (a.rfind("--source",0)==0) { eatValue(src); continue; }
        if (a == "--auto-demo") { autoDemo = true; continue; }
        if (a == "--auto-demo-no-prewarp") { autoDemo = true; autoDemoNoPrewarp = true; continue; }
        if (a=="-h"||a=="--help") {
          cout << "alpr-tool roi --source <video|device> [--conf <path>] [--auto-demo] [--auto-demo-no-prewarp]\n";
          return 0;
        }
        throw std::runtime_error(string("Unknown arg: ")+a);
      }
      cmdRoi(src, conf, autoDemo, autoDemoNoPrewarp);
      return 0;
    }
    if (sub == "tune") {
      TCLAP::CmdLine cmd("alpr-tool tune", ' ', "1.0");
      TCLAP::ValueArg<string> confArg("","conf","Path to config",false,"./config/openalpr.conf.defaults","string");
      TCLAP::ValueArg<string> srcArg("","source","Video source (rtsp/device/path)",false,"","string");
      cmd.add(confArg); cmd.add(srcArg);
      cmd.parse(subArgs);
      cmdTune(srcArg.getValue(), confArg.getValue());
      return 0;
    }
    if (sub == "doctor") {
      std::string outDir = "artifacts/configs";
      std::string country = "br";
      for (size_t i=0;i<subArgs.size();++i) {
        std::string a = subArgs[i];
        auto eatValue = [&](std::string& target){
          if (a.find('=') != std::string::npos) target = a.substr(a.find('=')+1);
          else if (i+1 < subArgs.size()) target = subArgs[++i];
          else throw std::runtime_error("missing value after " + a);
        };
        if (a.rfind("--country",0)==0) { eatValue(country); continue; }
        if (a.rfind("--out",0)==0) { eatValue(outDir); continue; }
        if (a=="-h"||a=="--help") {
          std::cout << "alpr-tool doctor --country <code> [--out <dir>]\n";
          return 0;
        }
        throw std::runtime_error(std::string("Unknown arg: ")+a);
      }
      RuntimeResolveResult rt = resolveRuntimeData(country, "");
      if (rt.preferredInvalid) {
        std::cerr << "[warn] runtime_data from config invalid for country=" << country << ": " << rt.preferredReason << "; trying fallbacks...\n";
      }
      if (!rt.ok) {
        std::cerr << "[error] Could not resolve runtime_data for country=" << country << std::endl;
        if (!rt.reason.empty()) std::cerr << " reason: " << rt.reason << std::endl;
        std::cerr << " tried: ";
        for (size_t i=0;i<rt.tested.size();i++) {
          if (i>0) std::cerr << ", ";
          std::cerr << rt.tested[i];
        }
        std::cerr << "\nPlease install openalpr runtime_data containing region/*.xml and ocr/.\n";
        return 1;
      }
      ensureDir(outDir);
      ensureDir("artifacts");
      ensureDir("artifacts/logs");
      std::string baseCountryCfg = outDir + "/openalpr." + country + ".conf";
      std::string baseDefaultCfg = outDir + "/openalpr.default.conf";
      std::string basePerfCfg = outDir + "/openalpr.performance.conf";
      if (!writeConfigFile(baseCountryCfg, rt.path, country)) {
        std::cerr << "[error] Failed to write config " << baseCountryCfg << std::endl;
        return 1;
      }
      if (!writeConfigFile(baseDefaultCfg, rt.path, country)) {
        std::cerr << "[error] Failed to write config " << baseDefaultCfg << std::endl;
        return 1;
      }
      if (!writePerformanceConfig(basePerfCfg, rt.path, country)) {
        std::cerr << "[error] Failed to write config " << basePerfCfg << std::endl;
        return 1;
      }
      // optional br2
      std::string br2Cascade = joinPath(joinPath(rt.path, "region"), "br2.xml");
      if (fileExists(br2Cascade.c_str())) {
        std::string br2Cfg = outDir + "/openalpr.br2.conf";
        if (!writeConfigFile(br2Cfg, rt.path, "br2")) {
          std::cerr << "[error] Failed to write config " << br2Cfg << std::endl;
          return 1;
        }
      }
      // list available countries (first 20)
      std::vector<std::string> regionFiles = getFilesInDir((rt.path + "/region").c_str());
      size_t total = regionFiles.size();
      std::cout << "[doctor] available countries (first 20): ";
      for (size_t i=0;i<regionFiles.size() && i<20;i++) {
        if (i>0) std::cout << ", ";
        std::string f = regionFiles[i];
        if (f.find(".xml") != std::string::npos) f = f.substr(0, f.find(".xml"));
        std::cout << f;
      }
      if (total > 20) std::cout << " ... (" << total << " total)";
      std::cout << std::endl;
      bool countryAvailable = false;
      for (const auto& f : regionFiles) {
        if (f.find(country + ".xml") != std::string::npos) { countryAvailable = true; break; }
      }
      if (!countryAvailable && !regionFiles.empty()) {
        std::string suggestion = regionFiles[0];
        if (suggestion.find(".xml") != std::string::npos) suggestion = suggestion.substr(0, suggestion.find(".xml"));
        std::cerr << "[warn] requested country '" << country << "' not found; try --country " << suggestion << std::endl;
      }
      std::cout << "[doctor] configs written to " << outDir << std::endl;
      std::cout << "[doctor] runtime_data_path_resolved=" << rt.path << " (auto selected)" << std::endl;
      std::cout << "[doctor] run preview with:\n";
      std::cout << "./build/src/alpr-tool preview --conf " << baseCountryCfg << " --source <video> --country " << country << "\n";
      return 0;
    }
    if (sub == "preview") {
      string conf = "./config/openalpr.conf.defaults";
      string src;
      string log = "artifacts/logs/preview.log";
      bool selfTest = false;
      PreviewRuntimeOptions opts;
      string countryArg;
      bool previewDoctor=false;
      for (size_t i=0;i<subArgs.size();++i) {
        string a = subArgs[i];
        auto eatValue = [&](string& target){
          if (a.find('=') != string::npos) target = a.substr(a.find('=')+1);
          else if (i+1<subArgs.size()) target = subArgs[++i];
          else throw std::runtime_error("missing value after "+a);
        };
        if (a.rfind("--conf",0)==0) { eatValue(conf); continue; }
        if (a.rfind("--source",0)==0) { eatValue(src); continue; }
        if (a.rfind("--log-file",0)==0) { eatValue(log); continue; }
        if (a.rfind("--speed-selftest",0)==0) { selfTest = true; continue; }
        if (a.rfind("--country",0)==0) { eatValue(countryArg); continue; }
        if (a.rfind("--doctor",0)==0) { previewDoctor = true; continue; }
        if (a.rfind("--crossing-mode",0)==0) { eatValue(opts.crossingMode); continue; }
        if (a.rfind("--crossing-roi",0)==0) {
          string v; eatValue(v);
          size_t p1=v.find(','); size_t p2=v.find(',', p1==string::npos?0:p1+1); size_t p3=v.find(',', p2==string::npos?0:p2+1);
          if (p1==string::npos||p2==string::npos||p3==string::npos) throw std::runtime_error("Invalid --crossing-roi format, expected x,y,w,h");
          opts.crossingRoi.x = stoi(v.substr(0,p1));
          opts.crossingRoi.y = stoi(v.substr(p1+1,p2-p1-1));
          opts.crossingRoi.width = stoi(v.substr(p2+1,p3-p2-1));
          opts.crossingRoi.height = stoi(v.substr(p3+1));
          if (opts.crossingRoi.width <=0 || opts.crossingRoi.height<=0) throw std::runtime_error("Invalid --crossing-roi dimensions");
          opts.crossingRoiProvided = true;
          continue;
        }
        if (a.rfind("--alpr-roi",0)==0) {
          string v; eatValue(v);
          size_t p1=v.find(','); size_t p2=v.find(',', p1==string::npos?0:p1+1); size_t p3=v.find(',', p2==string::npos?0:p2+1);
          if (p1==string::npos||p2==string::npos||p3==string::npos) throw std::runtime_error("Invalid --alpr-roi format, expected x,y,w,h");
          opts.alprRoi.x = stoi(v.substr(0,p1));
          opts.alprRoi.y = stoi(v.substr(p1+1,p2-p1-1));
          opts.alprRoi.width = stoi(v.substr(p2+1,p3-p2-1));
          opts.alprRoi.height = stoi(v.substr(p3+1));
          if (opts.alprRoi.width <=0 || opts.alprRoi.height<=0) throw std::runtime_error("Invalid --alpr-roi dimensions");
          opts.alprRoiProvided = true;
          continue;
        }
        if (a.rfind("--line",0)==0) {
          string v; eatValue(v);
          vector<int> parts;
          size_t start=0; for (int k=0;k<3;k++) {
            size_t comma = v.find(',', start);
            if (comma==string::npos) throw std::runtime_error("Invalid --line format, expected x1,y1,x2,y2");
            parts.push_back(stoi(v.substr(start, comma-start)));
            start = comma+1;
          }
          parts.push_back(stoi(v.substr(start)));
          opts.crossingP1 = Point(parts[0], parts[1]);
          opts.crossingP2 = Point(parts[2], parts[3]);
          continue;
        }
        if (a.rfind("--motion-thresh",0)==0) { string v; eatValue(v); opts.motionThresh = stoi(v); continue; }
        if (a.rfind("--motion-min-area",0)==0) { string v; eatValue(v); opts.motionMinArea = stoi(v); continue; }
        if (a.rfind("--motion-min-ratio",0)==0) { string v; eatValue(v); opts.motionMinRatio = stod(v); continue; }
        if (a.rfind("--motion-direction-filter",0)==0) { string v; eatValue(v); opts.motionDirectionFilter = (v!="0"&&v!="false"); continue; }
        if (a.rfind("--crossing-debounce",0)==0) { string v; eatValue(v); opts.crossingDebounce = std::max(1, stoi(v)); continue; }
        if (a.rfind("--crossing-arm-min-frames",0)==0) { string v; eatValue(v); opts.crossingArmMinFrames = std::max(1, stoi(v)); continue; }
        if (a.rfind("--crossing-arm-min-ratio",0)==0) { string v; eatValue(v); opts.crossingArmMinRatio = stod(v); continue; }
        if (a.rfind("--ocr-only-after-crossing",0)==0) { string v; eatValue(v); opts.ocrOnlyAfterCrossing = (v=="1"||v=="true"); continue; }
        if (a.rfind("--log-crossing-metrics",0)==0) { string v; eatValue(v); opts.logCrossingMetrics = (v=="1"||v=="true"); continue; }
        if (a.rfind("--ocr-only-after-crossing",0)==0) { string v; eatValue(v); opts.ocrOnlyAfterCrossing = (v=="1"||v=="true"); continue; }
        if (a.rfind("--log-ocr-metrics",0)==0) { string v; eatValue(v); opts.logOcrMetrics = (v=="1"||v=="true"); continue; }
        if (a.rfind("--log-plates-every-n",0)==0) { string v; eatValue(v); opts.logPlatesEveryN = std::max(1, stoi(v)); continue; }
        if (a.rfind("--log-plates-file",0)==0) { eatValue(opts.logPlatesFile); continue; }
        if (a.rfind("--log-plates",0)==0) { string v; eatValue(v); opts.logPlates = (v=="1"||v=="true"); continue; }
        if (a.rfind("--max-seconds",0)==0) { string v; eatValue(v); opts.maxSeconds = std::max(0, stoi(v)); continue; }
        if (a.rfind("--log-events",0)==0) { string v; eatValue(v); opts.logEvents = (v!="0"&&v!="false"); continue; }
        if (a.rfind("--gate-after-crossing",0)==0) { string v; eatValue(v); opts.gateAfterCrossing = (v=="1"||v=="true"); continue; }
        if (a.rfind("--report-json",0)==0) { eatValue(opts.reportJsonPath); continue; }
        if (a.rfind("--crossing-line-pct",0)==0) { string v; eatValue(v); opts.crossingLinePct = std::max(1.0, std::min(99.0, stod(v))); continue; }
        if (a.rfind("--log-throttle-ms",0)==0) { string v; eatValue(v); opts.logThrottleMs = stoi(v); continue; }
        if (a.rfind("--max-tracks",0)==0) { string v; eatValue(v); opts.maxTracks = stoi(v); continue; }
        if (a.rfind("--track-ttl-ms",0)==0) { string v; eatValue(v); opts.trackTtlMs = stoi(v); continue; }
        if (a=="-h"||a=="--help") {
          cout << "alpr-tool preview --source <video|device> [--conf <path>] [--log-file <path>] [--country <country>] [--speed-selftest]"
               << " [--log-plates 0|1] [--log-plates-every-n <int>] [--log-plates-file <path>] [--log-events 0|1] [--log-throttle-ms <int>]"
               << " [--max-tracks <int>] [--track-ttl-ms <int>] [--max-seconds <int>] [--gate-after-crossing 0|1] [--report-json <path>] [--crossing-line-pct <0-100>]"
               << " [--ocr-only-after-crossing 0|1] [--log-ocr-metrics 0|1] [--doctor]"
               << " [--crossing-mode off|motion] [--crossing-roi x,y,w,h] [--alpr-roi x,y,w,h] [--line x1,y1,x2,y2]"
               << " [--motion-thresh N] [--motion-min-area N] [--motion-min-ratio R] [--motion-direction-filter 0|1]"
               << " [--crossing-debounce N] [--crossing-arm-min-frames N] [--crossing-arm-min-ratio R]"
               << " [--log-crossing-metrics 0|1]\n";
          return 0;
        }
        throw std::runtime_error(string("Unknown arg: ")+a);
      }
      cmdPreview(src, conf, log, selfTest, opts, countryArg, previewDoctor, false);
      return 0;
    }
    if (sub == "export-yolo") {
      TCLAP::CmdLine cmd("alpr-tool export-yolo", ' ', "1.0");
      TCLAP::ValueArg<string> modelArg("","model","Path to .pt",true,"","string");
      TCLAP::ValueArg<string> outArg("","out","Path to output .onnx",true,"","string");
      TCLAP::ValueArg<int> imgszArg("","imgsz","Image size",false,640,"int");
      TCLAP::ValueArg<string> confArg("","conf","Path to config",false,"","string");
      TCLAP::SwitchArg updateArg("","update-conf","Update config with model path", false);
      cmd.add(modelArg); cmd.add(outArg); cmd.add(imgszArg); cmd.add(confArg); cmd.add(updateArg);
      cmd.parse(subArgs);
      if (updateArg.getValue() && confArg.getValue().empty()) {
        cerr << "--update-conf requires --conf\n";
        return 1;
      }
      cmdExportYolo(modelArg.getValue(), outArg.getValue(), imgszArg.getValue(), confArg.getValue(), updateArg.getValue());
      return 0;
    }
  } catch (TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for arg " << e.argId() << endl;
    return 1;
  } catch (const std::exception& e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  cerr << "Unknown subcommand: " << sub << endl;
  return 1;
}

