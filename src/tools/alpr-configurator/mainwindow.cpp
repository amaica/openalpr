#include "mainwindow.h"
#include <QApplication>
#include <QAction>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
#include <QLabel>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QTableWidget>
#include <QHeaderView>
#include <QFileDialog>
#include <QMessageBox>
#include <QClipboard>
#include <QStatusBar>
#include <QListWidget>
#include <QDockWidget>
#include <QToolBar>
#include <QStyle>
#include <QProcess>
#include <QSplitter>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QTimer>

#include <filesystem>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;

namespace {
static bool cascadeLoadable(const std::string& cascadePath) {
  cv::CascadeClassifier cc;
  return cc.load(cascadePath);
}

static QString colorize(const QString& text, const QString& color) {
  return QString("<span style='color:%1;'>%2</span>").arg(color, text);
}
}

MainWindow::MainWindow(const QString& projectArg, QWidget* parent)
  : QMainWindow(parent) {
  applyDarkTheme();
  buildMenus();
  buildToolBar();
  buildDocks();
  createCentralPreview();
  createTabsConfigPanel();
  buildStatusBar();
  setWindowTitle("ALPR Configurator");
  resize(1280, 820);

  if (!projectArg.isEmpty()) {
    currentProjectPath_ = projectArg;
    openProject();
  } else {
    newProjectWizard();
  }
  updateStatusIndicators();
}

void MainWindow::applyDarkTheme() {
  QPalette dark;
  dark.setColor(QPalette::Window, QColor(30,30,30));
  dark.setColor(QPalette::WindowText, Qt::white);
  dark.setColor(QPalette::Base, QColor(45,45,45));
  dark.setColor(QPalette::AlternateBase, QColor(60,60,60));
  dark.setColor(QPalette::ToolTipBase, Qt::white);
  dark.setColor(QPalette::ToolTipText, Qt::white);
  dark.setColor(QPalette::Text, Qt::white);
  dark.setColor(QPalette::Button, QColor(45,45,45));
  dark.setColor(QPalette::ButtonText, Qt::white);
  dark.setColor(QPalette::BrightText, Qt::red);
  dark.setColor(QPalette::Highlight, QColor(70, 120, 200));
  dark.setColor(QPalette::HighlightedText, Qt::black);
  qApp->setPalette(dark);
  qApp->setStyleSheet("QToolBar { border: none; spacing: 6px; } "
                      "QTabBar::tab { height: 26px; padding: 6px; } "
                      "QDockWidget { color: white; } "
                      "QStatusBar { color: white; }");
}

void MainWindow::buildMenus() {
  auto fileMenu = menuBar()->addMenu("&File");
  actOpenProject_ = fileMenu->addAction("Open Project");
  actOpenProject_->setShortcut(QKeySequence("Ctrl+O"));
  connect(actOpenProject_, &QAction::triggered, this, &MainWindow::openProject);

  actSaveProject_ = fileMenu->addAction("Save Project");
  actSaveProject_->setShortcut(QKeySequence("Ctrl+S"));
  connect(actSaveProject_, &QAction::triggered, this, &MainWindow::saveProject);

  actSaveProjectAs_ = fileMenu->addAction("Save Project As...");
  connect(actSaveProjectAs_, &QAction::triggered, this, &MainWindow::saveProjectAs);

  auto newProj = fileMenu->addAction("New Project Wizard");
  connect(newProj, &QAction::triggered, this, &MainWindow::newProjectWizard);

  fileMenu->addSeparator();
  auto exitAct = fileMenu->addAction("Exit");
  connect(exitAct, &QAction::triggered, this, &QWidget::close);

  auto sourceMenu = menuBar()->addMenu("&Source");
  actAddSource_ = sourceMenu->addAction("Add Source");
  connect(actAddSource_, &QAction::triggered, this, &MainWindow::addSource);
  actDupSource_ = sourceMenu->addAction("Duplicate Source");
  connect(actDupSource_, &QAction::triggered, this, &MainWindow::duplicateSource);
  actRemoveSource_ = sourceMenu->addAction("Remove Source");
  connect(actRemoveSource_, &QAction::triggered, this, &MainWindow::removeSource);

  auto configMenu = menuBar()->addMenu("&Config");
  actExportConf_ = configMenu->addAction("Export Config for Source");
  connect(actExportConf_, &QAction::triggered, this, &MainWindow::exportConfig);

  auto toolsMenu = menuBar()->addMenu("&Tools");
  actPreview_ = toolsMenu->addAction("Preview");
  actPreview_->setShortcut(QKeySequence(Qt::Key_Space));
  connect(actPreview_, &QAction::triggered, this, &MainWindow::togglePreview);
  actRoi_ = toolsMenu->addAction("ROI Editor");
  actRoi_->setShortcut(QKeySequence("R"));
  connect(actRoi_, &QAction::triggered, this, &MainWindow::openRoiEditor);
  actPrewarp_ = toolsMenu->addAction("Prewarp Editor");
  connect(actPrewarp_, &QAction::triggered, this, &MainWindow::openPrewarpEditor);
  actDoctor_ = toolsMenu->addAction("Doctor");
  connect(actDoctor_, &QAction::triggered, this, &MainWindow::runDoctor);

  menuBar()->addMenu("&View");
  menuBar()->addMenu("&Help");
}

void MainWindow::buildToolBar() {
  auto tb = addToolBar("Main");
  tb->setMovable(false);
  tb->addAction(style()->standardIcon(QStyle::SP_DirOpenIcon), "Open Project", this, &MainWindow::openProject);
  tb->addAction(style()->standardIcon(QStyle::SP_DialogSaveButton), "Save", this, &MainWindow::saveProject);
  tb->addSeparator();
  tb->addAction(style()->standardIcon(QStyle::SP_FileIcon), "Add Source", this, &MainWindow::addSource);
  tb->addAction(style()->standardIcon(QStyle::SP_FileDialogNewFolder), "Duplicate Source", this, &MainWindow::duplicateSource);
  tb->addAction(style()->standardIcon(QStyle::SP_TrashIcon), "Remove Source", this, &MainWindow::removeSource);
  tb->addSeparator();
  tb->addAction("Preview", this, &MainWindow::togglePreview);
  tb->addAction("ROI", this, &MainWindow::openRoiEditor);
  tb->addAction("Prewarp", this, &MainWindow::openPrewarpEditor);
  tb->addAction("Doctor", this, &MainWindow::runDoctor);
}

void MainWindow::buildDocks() {
  auto dockSources = new QDockWidget("Sources", this);
  sourceList_ = new QListWidget(dockSources);
  sourceList_->setSelectionMode(QAbstractItemView::SingleSelection);
  connect(sourceList_, &QListWidget::currentRowChanged, this, &MainWindow::onSourceSelectionChanged);
  dockSources->setWidget(sourceList_);
  addDockWidget(Qt::LeftDockWidgetArea, dockSources);
}

void MainWindow::createCentralPreview() {
  previewPanel_ = new QWidget(this);
  auto v = new QVBoxLayout();
  previewStatusLabel_ = new QLabel("Preview not started");
  previewStatusLabel_->setAlignment(Qt::AlignCenter);
  previewStatusLabel_->setMinimumHeight(260);
  previewStatusLabel_->setStyleSheet("border: 1px solid #555; background:#222; color:#ccc;");
  v->addWidget(previewStatusLabel_);
  previewPanel_->setLayout(v);
  setCentralWidget(previewPanel_);
}

void MainWindow::createTabsConfigPanel() {
  configTabs_ = new QTabWidget(this);

  // Source tab
  QWidget* sourceTab = new QWidget();
  auto srcLayout = new QFormLayout();
  typeCombo_ = new QComboBox();
  typeCombo_->addItems({"rtsp","video","file","camera"});
  uriEdit_ = new QLineEdit();
  fpsEdit_ = new QLineEdit();
  frameSkipEdit_ = new QLineEdit();
  bufferEdit_ = new QLineEdit();
  confPathEdit_ = new QLineEdit();
  srcLayout->addRow("Type", typeCombo_);
  srcLayout->addRow("URI", uriEdit_);
  srcLayout->addRow("Target FPS", fpsEdit_);
  srcLayout->addRow("Frame Skip", frameSkipEdit_);
  srcLayout->addRow("Buffer", bufferEdit_);
  srcLayout->addRow("Config path", confPathEdit_);
  sourceTab->setLayout(srcLayout);

  // Runtime tab
  QWidget* rtTab = new QWidget();
  auto rtLayout = new QFormLayout();
  countryEdit_ = new QComboBox();
  countryEdit_->setEditable(true);
  runtimeEdit_ = new QLineEdit();
  rtLayout->addRow("Country", countryEdit_);
  rtLayout->addRow("Runtime data", runtimeEdit_);
  rtTab->setLayout(rtLayout);

  // Detection
  QWidget* detTab = new QWidget();
  auto detLayout = new QFormLayout();
  skipDetectionCheck_ = new QCheckBox("Skip detection (use ROI)");
  detectorTypeEdit_ = new QLineEdit();
  detectorConfigEdit_ = new QLineEdit();
  detLayout->addRow(skipDetectionCheck_);
  detLayout->addRow("detector_type", detectorTypeEdit_);
  detLayout->addRow("detector", detectorConfigEdit_);
  detTab->setLayout(detLayout);

  // OCR
  QWidget* ocrTab = new QWidget();
  auto ocrLayout = new QFormLayout();
  vehicleCombo_ = new QComboBox(); vehicleCombo_->addItems({"car","moto"});
  scenarioCombo_ = new QComboBox(); scenarioCombo_->addItems({"default","garagem"});
  burstEdit_ = new QLineEdit();
  voteWindowEdit_ = new QLineEdit();
  minVotesEdit_ = new QLineEdit();
  fallbackCheck_ = new QCheckBox("fallback_ocr_enabled");
  ocrLayout->addRow("vehicle", vehicleCombo_);
  ocrLayout->addRow("scenario", scenarioCombo_);
  ocrLayout->addRow("ocr_burst_frames", burstEdit_);
  ocrLayout->addRow("vote_window", voteWindowEdit_);
  ocrLayout->addRow("min_votes", minVotesEdit_);
  ocrLayout->addRow(fallbackCheck_);
  ocrTab->setLayout(ocrLayout);

  // ROI / Crossing
  QWidget* roiTab = new QWidget();
  auto roiLayout = new QFormLayout();
  roiEdit_ = new QPlainTextEdit();
  roiEdit_->setPlaceholderText("{ \"x\":0, \"y\":0.5, \"w\":1, \"h\":0.5 }");
  lineEdit_ = new QLineEdit();
  motionThreshEdit_ = new QLineEdit();
  motionAreaEdit_ = new QLineEdit();
  motionRatioEdit_ = new QLineEdit();
  debounceEdit_ = new QLineEdit();
  armFramesEdit_ = new QLineEdit();
  ocrAfterCrossCheck_ = new QCheckBox("ocr_only_after_crossing");
  roiLayout->addRow("ROI (json)", roiEdit_);
  roiLayout->addRow("Line (x1,y1,x2,y2 norm)", lineEdit_);
  roiLayout->addRow("motion_thresh", motionThreshEdit_);
  roiLayout->addRow("motion_min_area", motionAreaEdit_);
  roiLayout->addRow("motion_min_ratio", motionRatioEdit_);
  roiLayout->addRow("crossing_debounce", debounceEdit_);
  roiLayout->addRow("crossing_arm_min_frames", armFramesEdit_);
  roiLayout->addRow(ocrAfterCrossCheck_);
  auto roiButtons = new QHBoxLayout();
  auto roiBtn = new QPushButton("Open ROI Editor");
  connect(roiBtn, &QPushButton::clicked, this, &MainWindow::openRoiEditor);
  roiButtons->addWidget(roiBtn);
  roiLayout->addRow(roiButtons);
  roiTab->setLayout(roiLayout);

  // Prewarp
  QWidget* prewarpTab = new QWidget();
  auto pwLayout = new QFormLayout();
  prewarpEnableCheck_ = new QCheckBox("Enable prewarp");
  prewarpPointsEdit_ = new QPlainTextEdit();
  prewarpPointsEdit_->setPlaceholderText("[[0,0],[1,0],[1,1],[0,1]]");
  pwLayout->addRow(prewarpEnableCheck_);
  pwLayout->addRow("Points", prewarpPointsEdit_);
  auto pwBtn = new QPushButton("Open Prewarp Editor");
  connect(pwBtn, &QPushButton::clicked, this, &MainWindow::openPrewarpEditor);
  pwLayout->addRow(pwBtn);
  prewarpTab->setLayout(pwLayout);

  // Logging & Metrics
  QWidget* logTab = new QWidget();
  auto logLayout = new QFormLayout();
  logPlatesCheck_ = new QCheckBox("log_plates");
  logOcrCheck_ = new QCheckBox("log_ocr_metrics");
  logEveryNEdit_ = new QLineEdit();
  logFileEdit_ = new QLineEdit();
  reportJsonEdit_ = new QLineEdit();
  logLayout->addRow(logPlatesCheck_);
  logLayout->addRow(logOcrCheck_);
  logLayout->addRow("log_plates_every_n", logEveryNEdit_);
  logLayout->addRow("log_file", logFileEdit_);
  logLayout->addRow("report_json", reportJsonEdit_);
  logTab->setLayout(logLayout);

  // Advanced
  QWidget* advTab = new QWidget();
  auto advLayout = new QVBoxLayout();
  auto filterRow = new QHBoxLayout();
  advancedFilter_ = new QLineEdit();
  filterRow->addWidget(new QLabel("Filter:"));
  filterRow->addWidget(advancedFilter_);
  connect(advancedFilter_, &QLineEdit::textChanged, this, &MainWindow::onAdvancedFilter);
  advancedTable_ = new QTableWidget();
  advancedTable_->setColumnCount(2);
  QStringList headers; headers << "Key" << "Value";
  advancedTable_->setHorizontalHeaderLabels(headers);
  advancedTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  auto advBtnRow = new QHBoxLayout();
  auto addRowBtn = new QPushButton("Add");
  auto delRowBtn = new QPushButton("Remove");
  connect(addRowBtn, &QPushButton::clicked, this, &MainWindow::onAdvancedAddRow);
  connect(delRowBtn, &QPushButton::clicked, this, &MainWindow::onAdvancedRemoveRow);
  advBtnRow->addWidget(addRowBtn);
  advBtnRow->addWidget(delRowBtn);
  advLayout->addLayout(filterRow);
  advLayout->addWidget(advancedTable_);
  advLayout->addLayout(advBtnRow);
  advTab->setLayout(advLayout);

  // Raw
  QWidget* rawTab = new QWidget();
  auto rawLayout = new QVBoxLayout();
  rawEdit_ = new QPlainTextEdit();
  auto rawBtns = new QHBoxLayout();
  auto applyRawBtn = new QPushButton("Apply raw → UI");
  auto fromUiBtn = new QPushButton("From UI → raw");
  connect(applyRawBtn, &QPushButton::clicked, this, &MainWindow::applyRawToModel);
  connect(fromUiBtn, &QPushButton::clicked, this, &MainWindow::fromModelToRaw);
  rawBtns->addWidget(applyRawBtn);
  rawBtns->addWidget(fromUiBtn);
  rawLayout->addWidget(rawEdit_);
  rawLayout->addLayout(rawBtns);
  rawTab->setLayout(rawLayout);

  configTabs_->addTab(sourceTab, "Source");
  configTabs_->addTab(rtTab, "Runtime");
  configTabs_->addTab(detTab, "Detection");
  configTabs_->addTab(ocrTab, "OCR");
  configTabs_->addTab(roiTab, "ROI / Crossing");
  configTabs_->addTab(prewarpTab, "Prewarp");
  configTabs_->addTab(logTab, "Logging & Metrics");
  configTabs_->addTab(advTab, "Advanced (All keys)");
  configTabs_->addTab(rawTab, "Raw Config");

  auto markDirty = [this](){
    setWindowModified(true);
    updateStatusIndicators();
  };
  for (auto le : {uriEdit_, fpsEdit_, frameSkipEdit_, bufferEdit_, confPathEdit_, runtimeEdit_, detectorTypeEdit_, detectorConfigEdit_,
                  burstEdit_, voteWindowEdit_, minVotesEdit_, lineEdit_, motionThreshEdit_, motionAreaEdit_, motionRatioEdit_,
                  debounceEdit_, armFramesEdit_, logEveryNEdit_, logFileEdit_, reportJsonEdit_}) {
    if (le) connect(le, &QLineEdit::textChanged, this, markDirty);
  }
  for (auto cb : {typeCombo_, countryEdit_, vehicleCombo_, scenarioCombo_}) {
    if (cb) connect(cb, &QComboBox::currentTextChanged, this, markDirty);
  }
  for (auto chk : {skipDetectionCheck_, fallbackCheck_, prewarpEnableCheck_, ocrAfterCrossCheck_, logPlatesCheck_, logOcrCheck_}) {
    if (chk) connect(chk, &QCheckBox::toggled, this, markDirty);
  }
  connect(roiEdit_, &QPlainTextEdit::textChanged, this, markDirty);
  connect(prewarpPointsEdit_, &QPlainTextEdit::textChanged, this, markDirty);
  connect(rawEdit_, &QPlainTextEdit::textChanged, this, [this](){ setWindowModified(true); });

  auto dockCfg = new QDockWidget("Config", this);
  dockCfg->setWidget(configTabs_);
  addDockWidget(Qt::RightDockWidgetArea, dockCfg);
}

void MainWindow::buildStatusBar() {
  runtimeIndicator_ = new QLabel();
  cascadeIndicator_ = new QLabel();
  tessIndicator_ = new QLabel();
  sourceIndicator_ = new QLabel();
  statusBar()->addPermanentWidget(runtimeIndicator_);
  statusBar()->addPermanentWidget(cascadeIndicator_);
  statusBar()->addPermanentWidget(tessIndicator_);
  statusBar()->addPermanentWidget(sourceIndicator_);
}

void MainWindow::refreshSourceList() {
  sourceList_->clear();
  int idx = 0;
  for (const auto& s : project_.sources()) {
    QListWidgetItem* it = new QListWidgetItem(s.id.isEmpty() ? QString("source_%1").arg(idx+1) : s.id);
    it->setToolTip(s.uri);
    sourceList_->addItem(it);
    idx++;
  }
}

void MainWindow::updateStatusIndicators() {
  auto setLamp = [](QLabel* lbl, bool ok, const QString& label){
    lbl->setText(ok ? colorize(label+" OK","lightgreen") : colorize(label+" FAIL","red"));
  };
  namespace fs = std::filesystem;
  QString runtime = runtimeEdit_ ? runtimeEdit_->text() : "";
  QString country = countryEdit_ ? countryEdit_->currentText() : "";
  bool rtOk = (!runtime.isEmpty() && fs::exists(runtime.toStdString()));
  bool cascadeOk = false;
  if (rtOk && !country.isEmpty()) {
    fs::path cascade = fs::path(runtime.toStdString()) / "region" / (country.toStdString() + ".xml");
    cascadeOk = fs::exists(cascade) && cascadeLoadable(cascade.string());
  }
  bool tessOk = false;
  if (rtOk) {
    fs::path tess = fs::path(runtime.toStdString()) / "ocr" / "tessdata";
    tessOk = fs::exists(tess);
  }
  bool srcOk = uriEdit_ && !uriEdit_->text().isEmpty();
  setLamp(runtimeIndicator_, rtOk, "runtime_data");
  setLamp(cascadeIndicator_, cascadeOk, "cascade");
  setLamp(tessIndicator_, tessOk, "tessdata");
  setLamp(sourceIndicator_, srcOk, "source");
}

void MainWindow::openProject() {
  if (currentProjectPath_.isEmpty()) {
    QString p = QFileDialog::getOpenFileName(this, "Open Project", "", "ALPR Project (*.alprproj.json)");
    if (p.isEmpty()) return;
    currentProjectPath_ = p;
  }
  if (!project_.load(currentProjectPath_)) {
    QMessageBox::critical(this, "Open Project", "Failed to load project");
    return;
  }
  refreshSourceList();
  if (!project_.sources().empty()) {
    sourceList_->setCurrentRow(0);
    loadSourceIntoUi(0);
  }
  setWindowModified(false);
}

void MainWindow::saveProject() {
  persistCurrentSource();
  if (currentProjectPath_.isEmpty()) {
    saveProjectAs();
    return;
  }
  if (!project_.save(currentProjectPath_)) {
    QMessageBox::critical(this, "Save Project", "Failed to save project");
    return;
  }
  setWindowModified(false);
}

void MainWindow::saveProjectAs() {
  persistCurrentSource();
  QString p = QFileDialog::getSaveFileName(this, "Save Project As", currentProjectPath_, "ALPR Project (*.alprproj.json)");
  if (p.isEmpty()) return;
  currentProjectPath_ = p;
  saveProject();
}

void MainWindow::newProjectWizard() {
  QDialog dlg(this);
  dlg.setWindowTitle("New Project");
  QFormLayout form(&dlg);
  QLineEdit pathEdit, firstSourceId;
  pathEdit.setPlaceholderText("artifacts/projects/demo.alprproj.json");
  QDialogButtonBox buttons(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
  form.addRow("Project path (.alprproj.json):", &pathEdit);
  form.addRow("Initial source id (optional):", &firstSourceId);
  form.addRow(&buttons);
  connect(&buttons, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
  connect(&buttons, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
  if (dlg.exec() == QDialog::Accepted) {
    project_.clear();
    if (!firstSourceId.text().isEmpty()) {
      SourceEntry s;
      s.id = firstSourceId.text();
      s.type = "rtsp";
      s.country = "br";
      project_.sources().push_back(s);
    }
    currentProjectPath_ = pathEdit.text();
    refreshSourceList();
    if (!project_.sources().empty()) {
      sourceList_->setCurrentRow(0);
      loadSourceIntoUi(0);
    }
    setWindowModified(true);
  }
}

void MainWindow::addSource() {
  SourceEntry s;
  s.id = QString("source_%1").arg(project_.sources().size()+1);
  s.type = "rtsp";
  s.country = "br";
  project_.sources().push_back(s);
  refreshSourceList();
  sourceList_->setCurrentRow((int)project_.sources().size()-1);
  loadSourceIntoUi((int)project_.sources().size()-1);
  setWindowModified(true);
}

void MainWindow::duplicateSource() {
  int idx = sourceList_->currentRow();
  if (idx < 0 || idx >= (int)project_.sources().size()) return;
  SourceEntry copy = project_.sources()[idx];
  copy.id += "_copy";
  project_.sources().push_back(copy);
  refreshSourceList();
  sourceList_->setCurrentRow((int)project_.sources().size()-1);
  loadSourceIntoUi((int)project_.sources().size()-1);
  setWindowModified(true);
}

void MainWindow::removeSource() {
  int idx = sourceList_->currentRow();
  if (idx < 0 || idx >= (int)project_.sources().size()) return;
  project_.sources().erase(project_.sources().begin()+idx);
  refreshSourceList();
  if (!project_.sources().empty()) {
    sourceList_->setCurrentRow(0);
    loadSourceIntoUi(0);
  } else {
    currentSourceIndex_ = -1;
  }
  setWindowModified(true);
}

void MainWindow::onSourceSelectionChanged() {
  int idx = sourceList_->currentRow();
  if (idx == currentSourceIndex_) return;
  persistCurrentSource();
  loadSourceIntoUi(idx);
}

void MainWindow::persistCurrentSource() {
  if (currentSourceIndex_ < 0 || currentSourceIndex_ >= (int)project_.sources().size()) return;
  applyUiToConfig();
  SourceEntry& s = project_.sources()[currentSourceIndex_];
  s.type = typeCombo_->currentText();
  s.uri = uriEdit_->text();
  s.country = countryEdit_->currentText();
  s.confPath = confPathEdit_->text();
  s.runtimeData = runtimeEdit_->text();
  if (!runtimeEdit_->text().isEmpty())
    project_.setRuntimeData(runtimeEdit_->text());
  s.roi = QJsonDocument::fromJson(roiEdit_->toPlainText().toUtf8()).object();
  s.crossing = QJsonObject{{"line", lineEdit_->text()}};
  s.prewarp = QJsonObject{{"enabled", prewarpEnableCheck_->isChecked()}, {"points", QJsonDocument::fromJson(prewarpPointsEdit_->toPlainText().toUtf8()).array()}};
}

void MainWindow::loadSourceIntoUi(int index) {
  if (index < 0 || index >= (int)project_.sources().size()) { currentSourceIndex_ = -1; return; }
  currentSourceIndex_ = index;
  const SourceEntry& s = project_.sources()[index];
  typeCombo_->setCurrentText(s.type);
  uriEdit_->setText(s.uri);
  countryEdit_->setEditText(s.country);
  runtimeEdit_->setText(s.runtimeData.isEmpty() ? project_.runtimeData() : s.runtimeData);
  confPathEdit_->setText(s.confPath);
  roiEdit_->setPlainText(QJsonDocument(s.roi).toJson(QJsonDocument::Compact));
  prewarpPointsEdit_->setPlainText(QJsonDocument(s.prewarp.value("points").toArray()).toJson(QJsonDocument::Compact));
  prewarpEnableCheck_->setChecked(s.prewarp.value("enabled").toBool(false));
  lineEdit_->setText(s.crossing.value("line").toString());

  configModel_.load(s.confPath.toStdString());
  reloadConfigIntoUi();
  setWindowModified(true);
  updateStatusIndicators();
}

QString MainWindow::currentConfPath() const {
  return confPathEdit_ ? confPathEdit_->text() : QString();
}

void MainWindow::applyUiToConfig() {
  if (!confPathEdit_) return;
  configModel_.set("video_source", uriEdit_->text().toStdString());
  configModel_.set("country", countryEdit_->currentText().toStdString());
  configModel_.set("runtime_dir", runtimeEdit_->text().toStdString());
  configModel_.set("skip_detection", skipDetectionCheck_->isChecked() ? "1" : "0");
  if (!detectorTypeEdit_->text().isEmpty()) configModel_.set("detector_type", detectorTypeEdit_->text().toStdString());
  if (!detectorConfigEdit_->text().isEmpty()) configModel_.set("detector", detectorConfigEdit_->text().toStdString());
  configModel_.set("vehicle", vehicleCombo_->currentText().toStdString());
  configModel_.set("scenario", scenarioCombo_->currentText().toStdString());
  if (!burstEdit_->text().isEmpty()) configModel_.set("ocr_burst_frames", burstEdit_->text().toStdString());
  if (!voteWindowEdit_->text().isEmpty()) configModel_.set("vote_window", voteWindowEdit_->text().toStdString());
  if (!minVotesEdit_->text().isEmpty()) configModel_.set("min_votes", minVotesEdit_->text().toStdString());
  configModel_.set("fallback_ocr_enabled", fallbackCheck_->isChecked() ? "1" : "0");
  configModel_.set("log_plates", logPlatesCheck_->isChecked() ? "1" : "0");
  configModel_.set("log_ocr_metrics", logOcrCheck_->isChecked() ? "1" : "0");
  if (!logEveryNEdit_->text().isEmpty()) configModel_.set("log_plates_every_n", logEveryNEdit_->text().toStdString());
  if (!logFileEdit_->text().isEmpty()) configModel_.set("log_file", logFileEdit_->text().toStdString());
  if (!reportJsonEdit_->text().isEmpty()) configModel_.set("report_json", reportJsonEdit_->text().toStdString());
  if (!lineEdit_->text().isEmpty()) configModel_.set("crossing_line", lineEdit_->text().toStdString());
  configModel_.set("ocr_only_after_crossing", ocrAfterCrossCheck_->isChecked() ? "1" : "0");
  if (!motionThreshEdit_->text().isEmpty()) configModel_.set("motion_thresh", motionThreshEdit_->text().toStdString());
  if (!motionAreaEdit_->text().isEmpty()) configModel_.set("motion_min_area", motionAreaEdit_->text().toStdString());
  if (!motionRatioEdit_->text().isEmpty()) configModel_.set("motion_min_ratio", motionRatioEdit_->text().toStdString());
  if (!debounceEdit_->text().isEmpty()) configModel_.set("crossing_debounce", debounceEdit_->text().toStdString());
  if (!armFramesEdit_->text().isEmpty()) configModel_.set("crossing_arm_min_frames", armFramesEdit_->text().toStdString());

  std::map<std::string,std::string> updated;
  for (int r=0;r<advancedTable_->rowCount();++r) {
    auto kItem = advancedTable_->item(r,0);
    auto vItem = advancedTable_->item(r,1);
    if (!kItem || kItem->text().trimmed().isEmpty()) continue;
    updated[kItem->text().toStdString()] = vItem ? vItem->text().toStdString() : "";
  }
  auto all = configModel_.items();
  for (auto& kv : all) {
    if (updated.find(kv.first)==updated.end()) updated[kv.first]=kv.second;
  }
  configModel_.replaceAll(updated);
  if (!currentConfPath().isEmpty())
    configModel_.save(currentConfPath().toStdString());
}

void MainWindow::reloadConfigIntoUi() {
  auto get = [&](const std::string& key, const std::string& def=""){ return QString::fromStdString(configModel_.get(key, def)); };
  vehicleCombo_->setCurrentText(get("vehicle","car"));
  scenarioCombo_->setCurrentText(get("scenario","default"));
  burstEdit_->setText(get("ocr_burst_frames","1"));
  voteWindowEdit_->setText(get("vote_window","1"));
  minVotesEdit_->setText(get("min_votes","1"));
  fallbackCheck_->setChecked(configModel_.get("fallback_ocr_enabled","0")=="1");
  skipDetectionCheck_->setChecked(configModel_.get("skip_detection","0")=="1");
  detectorTypeEdit_->setText(get("detector_type",""));
  detectorConfigEdit_->setText(get("detector",""));
  logPlatesCheck_->setChecked(configModel_.get("log_plates","0")=="1");
  logOcrCheck_->setChecked(configModel_.get("log_ocr_metrics","0")=="1");
  logEveryNEdit_->setText(get("log_plates_every_n",""));
  logFileEdit_->setText(get("log_file",""));
  reportJsonEdit_->setText(get("report_json",""));
  lineEdit_->setText(get("crossing_line",""));
  ocrAfterCrossCheck_->setChecked(configModel_.get("ocr_only_after_crossing","0")=="1");
  motionThreshEdit_->setText(get("motion_thresh",""));
  motionAreaEdit_->setText(get("motion_min_area",""));
  motionRatioEdit_->setText(get("motion_min_ratio",""));
  debounceEdit_->setText(get("crossing_debounce",""));
  armFramesEdit_->setText(get("crossing_arm_min_frames",""));

  refreshAdvancedTable();
  fromModelToRaw();
}

void MainWindow::refreshAdvancedTable() {
  auto items = configModel_.items();
  advancedTable_->setRowCount((int)items.size());
  int row=0;
  for (auto& kv : items) {
    advancedTable_->setItem(row,0,new QTableWidgetItem(QString::fromStdString(kv.first)));
    advancedTable_->setItem(row,1,new QTableWidgetItem(QString::fromStdString(kv.second)));
    row++;
  }
}

void MainWindow::onAdvancedAddRow() {
  int row = advancedTable_->rowCount();
  advancedTable_->insertRow(row);
}
void MainWindow::onAdvancedRemoveRow() {
  auto rows = advancedTable_->selectionModel()->selectedRows();
  for (auto idx : rows) advancedTable_->removeRow(idx.row());
}
void MainWindow::onAdvancedFilter(const QString& text) {
  for (int r=0;r<advancedTable_->rowCount();++r) {
    bool match = advancedTable_->item(r,0)->text().contains(text, Qt::CaseInsensitive)
              || advancedTable_->item(r,1)->text().contains(text, Qt::CaseInsensitive);
    advancedTable_->setRowHidden(r, !match);
  }
}

void MainWindow::applyRawToModel() {
  std::map<std::string,std::string> kv;
  std::istringstream iss(rawEdit_->toPlainText().toStdString());
  std::string line;
  while (std::getline(iss,line)) {
    auto pos = line.find('=');
    if (pos==std::string::npos) continue;
    auto key = QString::fromStdString(line.substr(0,pos)).trimmed().toStdString();
    auto val = QString::fromStdString(line.substr(pos+1)).trimmed().toStdString();
    kv[key]=val;
  }
  configModel_.replaceAll(kv);
  reloadConfigIntoUi();
  setWindowModified(true);
}

void MainWindow::fromModelToRaw() {
  std::ostringstream oss;
  for (auto& kv : configModel_.items()) {
    oss << kv.first << " = " << kv.second << "\n";
  }
  rawEdit_->setPlainText(QString::fromStdString(oss.str()));
}

bool MainWindow::validatePaths(QString& message, const QString& runtime, const QString& country) {
  namespace fs = std::filesystem;
  if (runtime.isEmpty() || country.isEmpty()) { message = "runtime_data or country empty"; return false; }
  fs::path base(runtime.toStdString());
  if (!fs::exists(base)) { message = "runtime_data does not exist"; return false; }
  fs::path region = base / "region";
  if (!fs::exists(region)) { message = "region dir missing"; return false; }
  fs::path cascade = region / (country.toStdString() + ".xml");
  if (!fs::exists(cascade)) { message = QString("cascade missing: %1").arg(QString::fromStdString(cascade.string())); return false; }
  if (!cascadeLoadable(cascade.string())) { message = "cascade cannot be loaded"; return false; }
  fs::path tessdir = base / "ocr" / "tessdata";
  if (!fs::exists(tessdir)) { message = "tessdata missing"; return false; }
  message = "OK";
  return true;
}

void MainWindow::runDoctor() {
  QString country = countryEdit_->currentText();
  if (country.isEmpty()) country = "br";
  QProcess proc;
  QString cmd = QString("./build/src/alpr-tool");
  QStringList args; args << "doctor" << "--country" << country;
  proc.start(cmd, args);
  proc.waitForFinished(-1);
  QString out = proc.readAllStandardOutput() + proc.readAllStandardError();
  QMessageBox::information(this, "Doctor", out);
  updateStatusIndicators();
}

void MainWindow::openRoiEditor() {
  QMessageBox::information(this, "ROI Editor", "ROI editor stub (not implemented)");
}

void MainWindow::openPrewarpEditor() {
  QMessageBox::information(this, "Prewarp Editor", "Prewarp editor stub (not implemented)");
}

void MainWindow::togglePreview() {
  previewRunning_ = !previewRunning_;
  previewStatusLabel_->setText(previewRunning_ ? "Preview running (stub)" : "Preview stopped");
}

void MainWindow::exportConfig() {
  applyUiToConfig();
  QMessageBox::information(this, "Export Config", "Config saved to " + currentConfPath());
}
#include "mainwindow.h"
#include <QTabWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
#include <QLabel>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QTableWidget>
#include <QHeaderView>
#include <QFileDialog>
#include <QMessageBox>
#include <QClipboard>
#include <QApplication>
#include <QSortFilterProxyModel>
#include <QTableWidgetItem>
#include <QStatusBar>
#include <cstdlib>

#include <filesystem>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
namespace {
struct RtResolve {
  bool ok=false;
  bool preferredInvalid=false;
  std::string preferredReason;
  std::string path;
  std::string reason;
  std::vector<std::string> tried;
};

static bool cascadeLoadable(const std::string& cascadePath) {
  cv::CascadeClassifier cc;
  return cc.load(cascadePath);
}

static RtResolve resolveRuntimeValidated(const std::string& country, const std::string& preferred) {
  RtResolve rr;
  std::vector<std::string> candidates;
  auto push = [&](const std::string& p){ if(!p.empty()) candidates.push_back(p); };
  const char* envRt = getenv("OPENALPR_RUNTIME_DATA");
  if (!preferred.empty()) push(preferred);
  if (envRt) push(envRt);
  push("/usr/share/openalpr/runtime_data");
  push("/usr/local/share/openalpr/runtime_data");
  push("./runtime_data");
  push((std::filesystem::current_path()/ "runtime_data").string());
  push((std::filesystem::current_path().parent_path()/ "runtime_data").string());

  bool first = true;
  for (const auto& base : candidates) {
    rr.tried.push_back(base);
    std::filesystem::path p(base);
    std::filesystem::path region = p / "region";
    std::filesystem::path cascade = region / (country + ".xml");
    if (!std::filesystem::exists(p)) { rr.reason = "runtime_data path missing"; goto next; }
    if (!std::filesystem::exists(region)) { rr.reason = "region dir missing"; goto next; }
    if (!std::filesystem::exists(cascade)) { rr.reason = "cascade file missing: " + cascade.string(); goto next; }
    if (!cascadeLoadable(cascade.string())) { rr.reason = "cascade not loadable: " + cascade.string(); goto next; }
    rr.ok = true; rr.path = base; return rr;
next:
    if (first && !preferred.empty()) { rr.preferredInvalid = true; rr.preferredReason = rr.reason; }
    first = false;
  }
  return rr;
}
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
  buildUi();
}

void MainWindow::buildUi() {
  auto *tabs = new QTabWidget(this);

  // General tab
  QWidget* general = new QWidget(this);
  auto gLayout = new QVBoxLayout();
  auto confRow = new QHBoxLayout();
  confPathEdit_ = new QLineEdit();
  auto browseConf = new QPushButton("Browse...");
  connect(browseConf, &QPushButton::clicked, this, &MainWindow::onBrowseConf);
  confRow->addWidget(new QLabel("Config file:"));
  confRow->addWidget(confPathEdit_);
  confRow->addWidget(browseConf);
  gLayout->addLayout(confRow);

  auto countryRow = new QHBoxLayout();
  countryEdit_ = new QComboBox();
  countryEdit_->setEditable(true);
  countryRow->addWidget(new QLabel("Country:"));
  countryRow->addWidget(countryEdit_);
  gLayout->addLayout(countryRow);

  auto rtRow = new QHBoxLayout();
  runtimeEdit_ = new QLineEdit();
  auto testPathsBtn = new QPushButton("Validate");
  connect(testPathsBtn, &QPushButton::clicked, this, &MainWindow::onTestPaths);
  auto autoSetupBtn = new QPushButton("Auto Setup");
  connect(autoSetupBtn, &QPushButton::clicked, this, &MainWindow::onAutoSetup);
  testStatus_ = new QLabel("-");
  rtRow->addWidget(new QLabel("Runtime data:"));
  rtRow->addWidget(runtimeEdit_);
  rtRow->addWidget(testPathsBtn);
  rtRow->addWidget(autoSetupBtn);
  rtRow->addWidget(testStatus_);
  gLayout->addLayout(rtRow);

  auto btnRow = new QHBoxLayout();
  auto loadBtn = new QPushButton("Load");
  auto saveBtn = new QPushButton("Save");
  auto saveAsBtn = new QPushButton("Save As");
  connect(loadBtn, &QPushButton::clicked, this, &MainWindow::onLoad);
  connect(saveBtn, &QPushButton::clicked, this, &MainWindow::onSave);
  connect(saveAsBtn, &QPushButton::clicked, this, &MainWindow::onSaveAs);
  btnRow->addWidget(loadBtn);
  btnRow->addWidget(saveBtn);
  btnRow->addWidget(saveAsBtn);
  gLayout->addLayout(btnRow);
  general->setLayout(gLayout);

  // Detection tab
  QWidget* det = new QWidget(this);
  auto dLayout = new QVBoxLayout();
  skipDetectionCheck_ = new QCheckBox("skip_detection");
  detectorEdit_ = new QLineEdit();
  auto detRow = new QHBoxLayout();
  detRow->addWidget(new QLabel("detector:"));
  detRow->addWidget(detectorEdit_);
  dLayout->addWidget(skipDetectionCheck_);
  dLayout->addLayout(detRow);
  det->setLayout(dLayout);

  // OCR tab
  QWidget* ocr = new QWidget(this);
  auto oLayout = new QVBoxLayout();
  tessdataEdit_ = new QLineEdit();
  auto tessRow = new QHBoxLayout();
  tessRow->addWidget(new QLabel("tessdata path:"));
  tessRow->addWidget(tessdataEdit_);
  oLayout->addLayout(tessRow);
  ocr->setLayout(oLayout);

  // Logging tab
  QWidget* logTab = new QWidget(this);
  auto lLayout = new QVBoxLayout();
  logPlatesCheck_ = new QCheckBox("log_plates");
  logOcrCheck_ = new QCheckBox("log_ocr_metrics");
  auto everyRow = new QHBoxLayout();
  logEveryNEdit_ = new QLineEdit();
  everyRow->addWidget(new QLabel("log-plates-every-n:"));
  everyRow->addWidget(logEveryNEdit_);
  lLayout->addWidget(logPlatesCheck_);
  lLayout->addWidget(logOcrCheck_);
  lLayout->addLayout(everyRow);
  logTab->setLayout(lLayout);

  // Video/Performance tab
  QWidget* perf = new QWidget(this);
  auto pLayout = new QVBoxLayout();
  ocrAfterCrossCheck_ = new QCheckBox("ocr_only_after_crossing");
  crossingRoiEdit_ = new QLineEdit();
  crossingLineEdit_ = new QLineEdit();
  motionThreshEdit_ = new QLineEdit();
  motionAreaEdit_ = new QLineEdit();
  motionRatioEdit_ = new QLineEdit();
  debounceEdit_ = new QLineEdit();
  armFramesEdit_ = new QLineEdit();

  auto roiRow = new QHBoxLayout(); roiRow->addWidget(new QLabel("crossing ROI (x,y,w,h norm):")); roiRow->addWidget(crossingRoiEdit_);
  auto lineRow = new QHBoxLayout(); lineRow->addWidget(new QLabel("line (x1,y1,x2,y2 norm):")); lineRow->addWidget(crossingLineEdit_);
  auto thrRow = new QHBoxLayout(); thrRow->addWidget(new QLabel("motion_thresh:")); thrRow->addWidget(motionThreshEdit_);
  auto areaRow = new QHBoxLayout(); areaRow->addWidget(new QLabel("motion_min_area:")); areaRow->addWidget(motionAreaEdit_);
  auto ratioRow = new QHBoxLayout(); ratioRow->addWidget(new QLabel("motion_min_ratio:")); ratioRow->addWidget(motionRatioEdit_);
  auto debRow = new QHBoxLayout(); debRow->addWidget(new QLabel("crossing_debounce:")); debRow->addWidget(debounceEdit_);
  auto armRow = new QHBoxLayout(); armRow->addWidget(new QLabel("crossing_arm_min_frames:")); armRow->addWidget(armFramesEdit_);

  auto genCmdBtn = new QPushButton("Generate Preview Command");
  connect(genCmdBtn, &QPushButton::clicked, this, &MainWindow::onGenerateCmd);
  commandOutput_ = new QPlainTextEdit();
  commandOutput_->setReadOnly(true);
  auto copyBtn = new QPushButton("Copy");
  connect(copyBtn, &QPushButton::clicked, [this](){
    QApplication::clipboard()->setText(commandOutput_->toPlainText());
  });

  pLayout->addWidget(ocrAfterCrossCheck_);
  pLayout->addLayout(roiRow);
  pLayout->addLayout(lineRow);
  pLayout->addLayout(thrRow);
  pLayout->addLayout(areaRow);
  pLayout->addLayout(ratioRow);
  pLayout->addLayout(debRow);
  pLayout->addLayout(armRow);
  pLayout->addWidget(genCmdBtn);
  pLayout->addWidget(commandOutput_);
  pLayout->addWidget(copyBtn);
  perf->setLayout(pLayout);

  // Advanced tab
  QWidget* adv = new QWidget(this);
  auto aLayout = new QVBoxLayout();
  auto filterRow = new QHBoxLayout();
  advancedFilter_ = new QLineEdit();
  filterRow->addWidget(new QLabel("Filter:"));
  filterRow->addWidget(advancedFilter_);
  connect(advancedFilter_, &QLineEdit::textChanged, this, &MainWindow::onAdvancedFilter);
  advancedTable_ = new QTableWidget();
  advancedTable_->setColumnCount(2);
  QStringList headers; headers << "Key" << "Value";
  advancedTable_->setHorizontalHeaderLabels(headers);
  advancedTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  auto advBtnRow = new QHBoxLayout();
  auto addRowBtn = new QPushButton("Add");
  auto delRowBtn = new QPushButton("Remove");
  connect(addRowBtn, &QPushButton::clicked, this, &MainWindow::onAdvancedAddRow);
  connect(delRowBtn, &QPushButton::clicked, this, &MainWindow::onAdvancedRemoveRow);
  advBtnRow->addWidget(addRowBtn);
  advBtnRow->addWidget(delRowBtn);
  aLayout->addLayout(filterRow);
  aLayout->addWidget(advancedTable_);
  aLayout->addLayout(advBtnRow);
  adv->setLayout(aLayout);

  // Raw tab
  QWidget* raw = new QWidget(this);
  auto rLayout = new QVBoxLayout();
  rawEdit_ = new QPlainTextEdit();
  auto rawBtnRow = new QHBoxLayout();
  auto applyRawBtn = new QPushButton("Apply raw → UI");
  auto fromUiBtn = new QPushButton("From UI → raw");
  connect(applyRawBtn, &QPushButton::clicked, this, &MainWindow::onApplyRaw);
  connect(fromUiBtn, &QPushButton::clicked, this, &MainWindow::onFromUiToRaw);
  rawBtnRow->addWidget(applyRawBtn);
  rawBtnRow->addWidget(fromUiBtn);
  rLayout->addWidget(rawEdit_);
  rLayout->addLayout(rawBtnRow);
  raw->setLayout(rLayout);

  tabs->addTab(general, "General");
  tabs->addTab(det, "Detection");
  tabs->addTab(ocr, "OCR");
  tabs->addTab(logTab, "Logging");
  tabs->addTab(perf, "Video / Performance");
  tabs->addTab(adv, "Advanced (All keys)");
  tabs->addTab(raw, "Raw Config");

  setCentralWidget(tabs);
  resize(800, 600);
  statusBarLabel_ = new QLabel("Ready");
  statusBar()->addWidget(statusBarLabel_);
  onAutoSetup();
}

void MainWindow::onBrowseConf() {
  QString f = QFileDialog::getOpenFileName(this, "Select openalpr.conf", currentConfPath_, "Config (*.conf);;All (*)");
  if (!f.isEmpty()) confPathEdit_->setText(f);
}

void MainWindow::onLoad() {
  QString path = confPathEdit_->text();
  if (path.isEmpty()) { QMessageBox::warning(this,"Load","Config path is empty"); return; }
  if (!model_.load(path.toStdString())) {
    QMessageBox::critical(this,"Load","Failed to load config");
    return;
  }
  currentConfPath_ = path;
  populateFromModel();
}

void MainWindow::onSave() {
  QString path = confPathEdit_->text();
  if (path.isEmpty()) { QMessageBox::warning(this,"Save","Config path is empty"); return; }
  applyFromUiToModel();
  if (!model_.save(path.toStdString())) {
    QMessageBox::critical(this,"Save","Failed to save config");
    return;
  }
  currentConfPath_ = path;
}

void MainWindow::onSaveAs() {
  QString f = QFileDialog::getSaveFileName(this,"Save config as", currentConfPath_, "Config (*.conf);;All (*)");
  if (f.isEmpty()) return;
  confPathEdit_->setText(f);
  onSave();
}

bool MainWindow::validatePaths(QString& message) {
  namespace fs = std::filesystem;
  QString runtime = runtimeEdit_->text();
  QString country = countryEdit_->currentText();
  if (runtime.isEmpty() || country.isEmpty()) { message = "runtime_data or country empty"; return false; }
  fs::path base(runtime.toStdString());
  if (!fs::exists(base)) { message = "runtime_data does not exist"; return false; }
  fs::path region = base / "region";
  if (!fs::exists(region)) { message = "region dir missing"; return false; }
  fs::path cascade = region / (country.toStdString() + ".xml");
  if (!fs::exists(cascade)) { message = QString("cascade missing: %1").arg(QString::fromStdString(cascade.string())); return false; }
  cv::CascadeClassifier cc;
  if (!cc.load(cascade.string())) { message = "cascade cannot be loaded"; return false; }
  fs::path tessdir = fs::path(runtime.toStdString()) / "ocr" / "tessdata";
  if (!fs::exists(tessdir)) { message = "tessdata missing"; return false; }
  message = "OK";
  return true;
}

void MainWindow::onTestPaths() {
  QString msg;
  bool ok = validatePaths(msg);
  testStatus_->setText(ok ? "OK" : msg);
  testStatus_->setStyleSheet(ok ? "color: green;" : "color: red;");
  statusBarLabel_->setText(ok ? "runtime_data OK" : msg);
}

void MainWindow::populateFromModel() {
  countryEdit_->setEditText(QString::fromStdString(model_.get("country","")));
  runtimeEdit_->setText(QString::fromStdString(model_.get("runtime_dir","")));
  skipDetectionCheck_->setChecked(model_.get("skip_detection","0")=="1");
  detectorEdit_->setText(QString::fromStdString(model_.get("detector","lbpcpu")));
  tessdataEdit_->setText(QString::fromStdString(model_.get("tessdata_prefix","")));
  logPlatesCheck_->setChecked(model_.get("log_plates","0")=="1");
  logOcrCheck_->setChecked(model_.get("log_ocr_metrics","0")=="1");
  logEveryNEdit_->setText(QString::fromStdString(model_.get("log_plates_every_n","10")));

  ocrAfterCrossCheck_->setChecked(model_.get("ocr_only_after_crossing","0")=="1");
  crossingRoiEdit_->setText(QString::fromStdString(model_.get("crossing_roi","")));
  crossingLineEdit_->setText(QString::fromStdString(model_.get("crossing_line","")));
  motionThreshEdit_->setText(QString::fromStdString(model_.get("motion_thresh","")));
  motionAreaEdit_->setText(QString::fromStdString(model_.get("motion_min_area","")));
  motionRatioEdit_->setText(QString::fromStdString(model_.get("motion_min_ratio","")));
  debounceEdit_->setText(QString::fromStdString(model_.get("crossing_debounce","")));
  armFramesEdit_->setText(QString::fromStdString(model_.get("crossing_arm_min_frames","")));

  refreshAdvancedTable();
  // rebuild raw from model
  std::ostringstream oss;
  for (auto& kv : model_.items()) {
    oss << kv.first << " = " << kv.second << "\n";
  }
  rawEdit_->setPlainText(QString::fromStdString(oss.str()));
}

void MainWindow::applyFromUiToModel() {
  model_.set("country", countryEdit_->currentText().toStdString());
  model_.set("runtime_dir", runtimeEdit_->text().toStdString());
  model_.set("skip_detection", skipDetectionCheck_->isChecked() ? "1" : "0");
  model_.set("detector", detectorEdit_->text().toStdString());
  if (!tessdataEdit_->text().isEmpty()) model_.set("tessdata_prefix", tessdataEdit_->text().toStdString());
  model_.set("log_plates", logPlatesCheck_->isChecked() ? "1" : "0");
  model_.set("log_ocr_metrics", logOcrCheck_->isChecked() ? "1" : "0");
  if (!logEveryNEdit_->text().isEmpty()) model_.set("log_plates_every_n", logEveryNEdit_->text().toStdString());
  model_.set("ocr_only_after_crossing", ocrAfterCrossCheck_->isChecked() ? "1" : "0");
  if (!crossingRoiEdit_->text().isEmpty()) model_.set("crossing_roi", crossingRoiEdit_->text().toStdString());
  if (!crossingLineEdit_->text().isEmpty()) model_.set("crossing_line", crossingLineEdit_->text().toStdString());
  if (!motionThreshEdit_->text().isEmpty()) model_.set("motion_thresh", motionThreshEdit_->text().toStdString());
  if (!motionAreaEdit_->text().isEmpty()) model_.set("motion_min_area", motionAreaEdit_->text().toStdString());
  if (!motionRatioEdit_->text().isEmpty()) model_.set("motion_min_ratio", motionRatioEdit_->text().toStdString());
  if (!debounceEdit_->text().isEmpty()) model_.set("crossing_debounce", debounceEdit_->text().toStdString());
  if (!armFramesEdit_->text().isEmpty()) model_.set("crossing_arm_min_frames", armFramesEdit_->text().toStdString());

  // sync from advanced table back to model
  std::map<std::string,std::string> updated;
  for (int r=0;r<advancedTable_->rowCount();++r) {
    auto kItem = advancedTable_->item(r,0);
    auto vItem = advancedTable_->item(r,1);
    if (!kItem || kItem->text().trimmed().isEmpty()) continue;
    updated[kItem->text().toStdString()] = vItem ? vItem->text().toStdString() : "";
  }
  // keep existing keys not in advanced
  for (auto& kv : model_.items()) {
    if (updated.find(kv.first)==updated.end())
      updated[kv.first]=kv.second;
  }
  model_.replaceAll(updated);
}

void MainWindow::refreshAdvancedTable() {
  auto items = model_.items();
  advancedTable_->setRowCount((int)items.size());
  int row=0;
  for (auto& kv : items) {
    advancedTable_->setItem(row,0,new QTableWidgetItem(QString::fromStdString(kv.first)));
    advancedTable_->setItem(row,1,new QTableWidgetItem(QString::fromStdString(kv.second)));
    row++;
  }
}

void MainWindow::onApplyRaw() {
  std::map<std::string,std::string> kv;
  std::istringstream iss(rawEdit_->toPlainText().toStdString());
  std::string line;
  while (std::getline(iss,line)) {
    auto pos = line.find('=');
    if (pos==std::string::npos) continue;
    auto key = line.substr(0,pos);
    auto val = line.substr(pos+1);
    kv[std::string(key.begin(), key.end())] = std::string(val.begin(), val.end());
  }
  model_.replaceAll(kv);
  populateFromModel();
}

void MainWindow::onFromUiToRaw() {
  applyFromUiToModel();
  std::ostringstream oss;
  for (auto& kv : model_.items()) {
    oss << kv.first << " = " << kv.second << "\n";
  }
  rawEdit_->setPlainText(QString::fromStdString(oss.str()));
}

void MainWindow::onGenerateCmd() {
  applyFromUiToModel();
  std::ostringstream cmd;
  QString conf = confPathEdit_->text();
  if (conf.isEmpty()) conf = "openalpr.conf";
  cmd << "./build/src/alpr-tool preview --conf " << conf.toStdString();
  if (!countryEdit_->currentText().isEmpty()) cmd << " --country " << countryEdit_->currentText().toStdString();
  if (ocrAfterCrossCheck_->isChecked()) cmd << " --ocr-only-after-crossing 1";
  if (!crossingRoiEdit_->text().isEmpty()) cmd << " --crossing-roi " << crossingRoiEdit_->text().toStdString();
  if (!crossingLineEdit_->text().isEmpty()) cmd << " --line " << crossingLineEdit_->text().toStdString();
  if (!motionThreshEdit_->text().isEmpty()) cmd << " --motion-thresh " << motionThreshEdit_->text().toStdString();
  if (!motionAreaEdit_->text().isEmpty()) cmd << " --motion-min-area " << motionAreaEdit_->text().toStdString();
  if (!motionRatioEdit_->text().isEmpty()) cmd << " --motion-min-ratio " << motionRatioEdit_->text().toStdString();
  if (!debounceEdit_->text().isEmpty()) cmd << " --crossing-debounce " << debounceEdit_->text().toStdString();
  if (!armFramesEdit_->text().isEmpty()) cmd << " --crossing-arm-min-frames " << armFramesEdit_->text().toStdString();
  commandOutput_->setPlainText(QString::fromStdString(cmd.str()));
}

void MainWindow::onAdvancedAddRow() {
  int row = advancedTable_->rowCount();
  advancedTable_->insertRow(row);
}

void MainWindow::onAdvancedRemoveRow() {
  auto rows = advancedTable_->selectionModel()->selectedRows();
  for (auto it = rows.rbegin(); it != rows.rend(); ++it) {
    advancedTable_->removeRow(it->row());
  }
}

void MainWindow::onAdvancedFilter(const QString& text) {
  for (int r=0;r<advancedTable_->rowCount();++r) {
    bool match = advancedTable_->item(r,0)->text().contains(text, Qt::CaseInsensitive);
    advancedTable_->setRowHidden(r, !text.isEmpty() && !match);
  }
}

void MainWindow::onAutoSetup() {
  namespace fs = std::filesystem;
  RtResolve rr = resolveRuntimeValidated("br", "");
  std::string chosen = rr.path;
  if (rr.preferredInvalid) {
    statusBarLabel_->setText(QString::fromStdString("config runtime invalid; trying fallbacks"));
  }
  if (!rr.ok) {
    statusBarLabel_->setText("runtime_data not found/invalid");
    return;
  }
  runtimeEdit_->setText(QString::fromStdString(chosen));

  // list countries
  countryEdit_->clear();
  std::vector<std::string> countries;
  for (auto& entry : fs::directory_iterator(fs::path(chosen)/"region")) {
    if (!entry.is_regular_file()) continue;
    auto name = entry.path().filename().string();
    if (name.size() > 4 && name.substr(name.size()-4)==".xml") {
      countries.push_back(name.substr(0,name.size()-4));
    }
  }
  std::sort(countries.begin(), countries.end());
  for (auto& c : countries) countryEdit_->addItem(QString::fromStdString(c));
  QString defCountry;
  if (std::find(countries.begin(), countries.end(), "br2") != countries.end()) defCountry = "br2";
  else if (std::find(countries.begin(), countries.end(), "br") != countries.end()) defCountry = "br";
  else if (!countries.empty()) defCountry = QString::fromStdString(countries.front());
  countryEdit_->setEditText(defCountry);

  // generate default configs if missing
  fs::path outDir = fs::path("artifacts")/"configs";
  fs::create_directories(outDir);
  fs::path cfgPath = outDir/("openalpr." + defCountry.toStdString() + ".conf");
  if (!fs::exists(cfgPath)) {
    ConfigModel tmp;
    tmp.set("runtime_dir", chosen);
    tmp.set("country", defCountry.toStdString());
    tmp.set("detector", "lbpcpu");
    tmp.set("skip_detection", "0");
    tmp.save(cfgPath.string());
    ConfigModel defCfg;
    defCfg.set("runtime_dir", chosen);
    defCfg.set("country", defCountry.toStdString());
    defCfg.set("detector", "lbpcpu");
    defCfg.set("skip_detection", "0");
    defCfg.save((outDir/"openalpr.default.conf").string());
  }
  confPathEdit_->setText(QString::fromStdString(cfgPath.string()));
  onLoad();
  QString msg;
  bool ok = validatePaths(msg);
  testStatus_->setText(ok ? "OK" : msg);
  testStatus_->setStyleSheet(ok ? "color: green;" : "color: red;");
  statusBarLabel_->setText(ok ? "runtime_data OK, cascade OK" : msg);
}

