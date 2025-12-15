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
#include <limits>
#include <climits>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>
#include <algorithm>
#include "openalpr/alpr.h"
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

struct Track {
  int id=0;
  Rect bbox;
  double centerY=0.0;
  double emaCenterY=0.0;
  bool hasEma=false;
  bool crossedA=false;
  double tA=0.0;
  bool fired=false;
  double speedKmh=0.0;
  string reason;
  string plate;
  double plateConf=0.0;
  int lastFrame=0;
  double lastTsMs=0.0;
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

static void cmdPreview(const string& source, const string& confPath, const string& logPath, bool selfTest) {
  ConfigWriter cfg;
  cfg.load(confPath);
  SpeedConfig speedCfg = loadSpeedConfig(cfg);
  string src = source.empty() ? cfg.get("video_source","") : source;
  if (src.empty()) {
    cout << "Enter video source (rtsp/device/video path): ";
    getline(cin, src);
  }
  VideoCapture cap;
  if (!openCapture(src, cap)) {
    cerr << "Could not open source: " << src << endl;
    return;
  }
  string country = cfg.get("country","us");
  Alpr alpr(country, confPath, "");
  if (!alpr.isLoaded()) {
    cerr << "Could not load ALPR with config: " << confPath << endl;
    return;
  }
  if (selfTest) speedCfg.enabled = true;
  ofstream logFile;
  if (!logPath.empty()) {
    ensureParentDir(logPath);
    logFile.open(logPath, ios::out | ios::app);
    if (!logFile.good()) cerr << "Could not open log file: " << logPath << endl;
  }
  namedWindow("alpr-tool preview", WINDOW_NORMAL);
  Rect roi;
  bool defaultUsed=false;
  int64 lastTick = getTickCount();
  int frameIdx = 0;
  double emaY = -1.0;
  bool crossedA = false;
  bool fired = false;
  double tA = 0.0;
  double lastSpeed = 0.0;
  double fpsReported = cap.get(CAP_PROP_FPS);
  bool fpsValid = fpsReported > 1.0 && fpsReported < 300.0;
  while (true) {
    Mat frame;
    if (!cap.read(frame) || frame.empty()) break;
    frameIdx++;
    if (roi.area() == 0) {
      roi = roiFromConfig(cfg, frame);
      if (roi.area() == 0) { roi = defaultRoi(frame); defaultUsed=true; }
    }
    vector<AlprRegionOfInterest> rois;
    if (roi.area() > 0) rois.push_back(AlprRegionOfInterest(roi.x, roi.y, roi.width, roi.height));
    Mat bgr;
    if (frame.channels() == 1) cvtColor(frame, bgr, COLOR_GRAY2BGR); else bgr = frame;
    if (!bgr.isContinuous()) bgr = bgr.clone();
    AlprResults results;
    if (selfTest) {
      results.total_processing_time_ms = 0;
      results.img_width = frame.cols;
      results.img_height = frame.rows;
      AlprPlateResult pr;
      int startY = roi.area() > 0 ? roi.y + 10 : static_cast<int>(speedCfg.yA * frame.rows);
      int endY   = roi.area() > 0 ? roi.y + roi.height - 10 : static_cast<int>(speedCfg.yB * frame.rows);
      if (endY <= startY) endY = startY + 50;
      int cycle = 90;
      double prog = (frameIdx % cycle) / static_cast<double>(cycle-1);
      int cy = startY + static_cast<int>((endY - startY) * prog);
      int cx = frame.cols/2;
      int bw = 100, bh = 50;
      pr.plate_points[0].x = cx - bw/2; pr.plate_points[0].y = cy - bh/2;
      pr.plate_points[1].x = cx + bw/2; pr.plate_points[1].y = cy - bh/2;
      pr.plate_points[2].x = cx + bw/2; pr.plate_points[2].y = cy + bh/2;
      pr.plate_points[3].x = cx - bw/2; pr.plate_points[3].y = cy + bh/2;
      pr.bestPlate.characters = "SELFTEST";
      pr.bestPlate.overall_confidence = 99.0;
      results.plates.push_back(pr);
    } else {
      results = alpr.recognize(bgr.data, bgr.elemSize(), bgr.cols, bgr.rows, rois);
    }
    double lineA = speedCfg.yA * frame.rows;
    double lineB = speedCfg.yB * frame.rows;
    if (speedCfg.enabled) {
      for (const auto& plate : results.plates) {
        Rect pr = plateRect(plate);
        Point c((pr.x+pr.width/2), (pr.y+pr.height/2));
        if (roi.area() > 0 && !roi.contains(c)) continue; // ROI gate
        double cy = c.y;
        if (emaY < 0) emaY = cy;
        else if (speedCfg.smoothing == "ema") emaY = speedCfg.emaAlpha * cy + (1.0 - speedCfg.emaAlpha) * emaY;
        else emaY = cy;

        double tNow = 0.0;
        bool tOk = getTimeSeconds(cap, frameIdx, fpsReported, fpsValid, tNow);

        if (!crossedA && emaY >= lineA && tOk) {
          crossedA = true;
          tA = tNow;
          if (speedCfg.log) {
            std::ostringstream oss; oss << "frame=" << frameIdx << " speed_arm A crossed";
            cout << oss.str() << endl;
            if (logFile.good()) logFile << oss.str() << "\n";
          }
        }
        if (crossedA && !fired && emaY >= lineB && tOk) {
          double dt = tNow - tA;
          if (dt > 0 && speedCfg.dist_m > 0) {
            double mps = speedCfg.dist_m / dt;
            double kmh = mps * 3.6;
            if (kmh >= speedCfg.minKmh && kmh <= speedCfg.maxKmh) {
              fired = true;
              lastSpeed = kmh;
              std::ostringstream oss;
              oss << "frame=" << frameIdx
                  << " plate=" << plate.bestPlate.characters
                  << " conf=" << plate.bestPlate.overall_confidence
                  << " speed_kmh=" << kmh
                  << " dt=" << dt
                  << " mode=lines crossed=A->B";
              cout << oss.str() << endl;
              if (logFile.good()) logFile << oss.str() << "\n";
            }
          }
        }
      }
    } else {
      drawResults(frame, results);
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
      if (fired) {
        std::ostringstream ss; ss << lastSpeed << " km/h";
        putText(frame, ss.str(), Point(10,80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,255),2,LINE_AA);
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
    if (sub == "preview") {
      string conf = "./config/openalpr.conf.defaults";
      string src;
      string log = "artifacts/logs/preview.log";
      bool selfTest = false;
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
        if (a=="-h"||a=="--help") {
          cout << "alpr-tool preview --source <video|device> [--conf <path>] [--log-file <path>] [--speed-selftest]\n";
          return 0;
        }
        throw std::runtime_error(string("Unknown arg: ")+a);
      }
      cmdPreview(src, conf, log, selfTest);
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

