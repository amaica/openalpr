#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>
#include <tclap/CmdLine.h>
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
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
  Point start, end;
  Rect draft;
  Rect applied;
  bool dirty=false;
};

static RoiState g_roiState;
static bool g_saveRequested = false;

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
  const int btnW = 110;
  const int btnH = 32;
  const int pad = 8;
  vector<string> labels = {"PLAY","PAUSE","STOP","SAVE ROI","RESET ROI","QUIT"};
  vector<Button> btns;
  int x = pad;
  int y = pad;
  for (auto& l : labels) {
    btns.push_back({l, Rect(x,y,btnW,btnH)});
    x += btnW + pad;
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

static void mouseCb(int event, int x, int y, int, void* userdata) {
  auto* payload = reinterpret_cast<pair<vector<Button>*, PlayState*>*>(userdata);
  if (!payload) return;
  auto& buttons = *payload->first;
  PlayState& state = *payload->second;

  if (event == EVENT_LBUTTONDOWN) {
    // Check buttons first
    for (const auto& b : buttons) {
      if (pointInRect(Point(x,y), b.area)) {
        if (b.label == "PLAY") state = STATE_PLAYING;
        else if (b.label == "PAUSE") state = STATE_PAUSED;
        else if (b.label == "STOP") state = STATE_STOPPED;
        else if (b.label == "SAVE ROI") { g_roiState.dirty = true; g_saveRequested = true; }
        else if (b.label == "RESET ROI") { g_roiState.applied = Rect(); g_roiState.draft = Rect(); g_roiState.dirty = false; }
        else if (b.label == "QUIT") state = STATE_STOPPED;
        return;
      }
    }
    // Start drawing ROI
    g_roiState.drawing = true;
    g_roiState.start = Point(x,y);
    g_roiState.end = g_roiState.start;
  } else if (event == EVENT_MOUSEMOVE && g_roiState.drawing) {
    g_roiState.end = Point(x,y);
    g_roiState.draft = Rect(g_roiState.start, g_roiState.end);
  } else if (event == EVENT_LBUTTONUP) {
    g_roiState.drawing = false;
    g_roiState.end = Point(x,y);
    g_roiState.draft = Rect(g_roiState.start, g_roiState.end);
    g_roiState.dirty = true;
  }
}

static void overlayInfo(Mat& frame, const Rect& roiPx, const string& confPath, bool dirty, bool defaultUsed) {
  if (roiPx.area() > 0) rectangle(frame, roiPx, Scalar(0,255,0), 2);

  std::ostringstream oss;
  if (roiPx.area() > 0) {
    float rx = roiPx.x / static_cast<float>(frame.cols);
    float ry = roiPx.y / static_cast<float>(frame.rows);
    float rw = roiPx.width / static_cast<float>(frame.cols);
    float rh = roiPx.height / static_cast<float>(frame.rows);
    oss << "ROI % x=" << rx << " y=" << ry << " w=" << rw << " h=" << rh;
  } else {
    oss << "ROI disabled";
  }
  if (defaultUsed) oss << " [DEFAULT]";
  if (dirty) oss << " [DIRTY]";
  putText(frame, oss.str(), Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,255,255),1,LINE_AA);
  putText(frame, confPath, Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200,200,200),1,LINE_AA);
}

static bool openCapture(const string& src, VideoCapture& cap) {
  if (src.empty()) return false;
  if (isdigit(src[0]) && src.find_first_not_of("0123456789") == string::npos) {
    int idx = stoi(src);
    return cap.open(idx);
  }
  return cap.open(src);
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

static void cmdRoi(const string& source, const string& confPath) {
  ConfigWriter cfg;
  cfg.load(confPath);
  g_saveRequested = false;
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
  namedWindow("alpr-tool roi", WINDOW_NORMAL);
  auto buttons = buildButtons(800);
  PlayState playState = STATE_PAUSED;
  pair<vector<Button>*, PlayState*> userdata{&buttons, &playState};
  setMouseCallback("alpr-tool roi", mouseCb, &userdata);

  Rect roiPx;
  bool defaultUsed = false;
  int frameIndex = 0;
  Mat frame, display;
  // Start paused: grab first frame
  if (!cap.read(frame) || frame.empty()) {
    cerr << "No frames available\n";
    destroyWindow("alpr-tool roi");
    return;
  }
  roiPx = roiFromConfig(cfg, frame);
  if (roiPx.area() == 0) { roiPx = defaultRoi(frame); defaultUsed = true; }
  g_roiState.applied = roiPx;
  g_roiState.dirty = true;
  g_saveRequested = true;

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

    Rect drawR = g_roiState.draft.area() > 0 ? normalizedRect(g_roiState.draft, frame) : g_roiState.applied;
    if (drawR.area() == 0) { drawR = defaultRoi(frame); defaultUsed = true; }
    display = frame.clone();
    drawButtons(display, buttons, playState);
    overlayInfo(display, drawR, cfg.path.empty() ? confPath : cfg.path, g_roiState.dirty, defaultUsed);
    imshow("alpr-tool roi", display);
    int key = waitKey(30);
    if (key == 'q' || key == 27) break;
    if (key == ' ') playState = (playState == STATE_PLAYING ? STATE_PAUSED : STATE_PLAYING);
    if (key == 's') { g_roiState.dirty = true; g_saveRequested = true; }
    if (key == 'r') { g_roiState.draft = Rect(); g_roiState.applied = Rect(); g_roiState.dirty = false; }
    if (key == '1') { g_roiState.applied = defaultRoi(frame); g_roiState.draft = Rect(); g_roiState.dirty = true; defaultUsed = true; }
    if (g_roiState.dirty && g_saveRequested) {
      Rect saveR = g_roiState.draft.area() > 0 ? normalizedRect(g_roiState.draft, frame) : drawR;
      saveRoiToConfig(saveR, frame, cfg);
      cfg.save();
      g_roiState.applied = saveR;
      g_roiState.draft = Rect();
      g_roiState.dirty = false;
      g_saveRequested = false;
      cout << "ROI saved to " << cfg.lastWritePath << " (percent)\n";
    }

    if (key == 'p') playState = STATE_PAUSED;
  }
  destroyWindow("alpr-tool roi");
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

static void cmdPreview(const string& source, const string& confPath, const string& logPath) {
  ConfigWriter cfg;
  cfg.load(confPath);
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
  std::map<string,bool> wasBelow;
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
    AlprResults results = alpr.recognize(bgr.data, bgr.elemSize(), bgr.cols, bgr.rows, rois);
    double triggerY = frame.rows * 0.5;
    for (const auto& plate : results.plates) {
      drawResults(frame, results);
      double cy = bboxCenterY(plate);
      bool below = cy < triggerY;
      bool prevBelow = wasBelow[plate.bestPlate.characters];
      bool crossed = (prevBelow && !below);
      wasBelow[plate.bestPlate.characters] = below;
      if (crossed) {
        Rect pr = plateRect(plate);
        std::ostringstream oss;
        oss << "frame=" << frameIdx
            << " plate=" << plate.bestPlate.characters
            << " conf=" << plate.bestPlate.overall_confidence
            << " bbox=" << pr.x << "," << pr.y << "," << pr.width << "," << pr.height
            << " crossed_line=true";
        cout << oss.str() << endl;
        if (logFile.good()) logFile << oss.str() << "\n";
      }
    }
    if (roi.area() > 0) rectangle(frame, roi, Scalar(0,255,0), 2);
    line(frame, Point(0, (int)triggerY), Point(frame.cols-1, (int)triggerY), Scalar(255,255,0), 1, LINE_AA);
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
        if (a=="-h"||a=="--help") {
          cout << "alpr-tool roi --source <video|device> [--conf <path>]\n";
          return 0;
        }
        throw std::runtime_error(string("Unknown arg: ")+a);
      }
      cmdRoi(src, conf);
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
        if (a=="-h"||a=="--help") {
          cout << "alpr-tool preview --source <video|device> [--conf <path>] [--log-file <path>]\n";
          return 0;
        }
        throw std::runtime_error(string("Unknown arg: ")+a);
      }
      cmdPreview(src, conf, log);
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

