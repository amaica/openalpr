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

#include <filesystem>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;

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
  countryEdit_ = new QLineEdit();
  countryRow->addWidget(new QLabel("Country:"));
  countryRow->addWidget(countryEdit_);
  gLayout->addLayout(countryRow);

  auto rtRow = new QHBoxLayout();
  runtimeEdit_ = new QLineEdit();
  auto testPathsBtn = new QPushButton("Test Paths");
  connect(testPathsBtn, &QPushButton::clicked, this, &MainWindow::onTestPaths);
  testStatus_ = new QLabel("-");
  rtRow->addWidget(new QLabel("Runtime data:"));
  rtRow->addWidget(runtimeEdit_);
  rtRow->addWidget(testPathsBtn);
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
  QString country = countryEdit_->text();
  if (runtime.isEmpty() || country.isEmpty()) { message = "runtime_data or country empty"; return false; }
  fs::path base(runtime.toStdString());
  if (!fs::exists(base)) { message = "runtime_data does not exist"; return false; }
  fs::path region = base / "region";
  if (!fs::exists(region)) { message = "region dir missing"; return false; }
  fs::path cascade = region / (country.toStdString() + ".xml");
  if (!fs::exists(cascade)) { message = QString("cascade missing: %1").arg(QString::fromStdString(cascade.string())); return false; }
  cv::CascadeClassifier cc;
  if (!cc.load(cascade.string())) { message = "cascade cannot be loaded"; return false; }
  message = "OK";
  return true;
}

void MainWindow::onTestPaths() {
  QString msg;
  bool ok = validatePaths(msg);
  testStatus_->setText(ok ? "OK" : msg);
  testStatus_->setStyleSheet(ok ? "color: green;" : "color: red;");
}

void MainWindow::populateFromModel() {
  countryEdit_->setText(QString::fromStdString(model_.get("country","")));
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
  model_.set("country", countryEdit_->text().toStdString());
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
  if (!countryEdit_->text().isEmpty()) cmd << " --country " << countryEdit_->text().toStdString();
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

