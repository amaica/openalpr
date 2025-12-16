#pragma once
#include <QMainWindow>
#include <memory>
#include "configmodel.h"
#include "projectmodel.h"

class QListWidget;
class QStackedWidget;
class QTabWidget;
class QLabel;
class QLineEdit;
class QComboBox;
class QCheckBox;
class QPlainTextEdit;
class QTableWidget;
class QPushButton;
class QAction;

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(const QString& projectArg, QWidget* parent=nullptr);

private slots:
  // File/Project
  void openProject();
  void saveProject();
  void saveProjectAs();
  void newProjectWizard();
  void exportConfig();

  // Sources
  void addSource();
  void duplicateSource();
  void removeSource();
  void onSourceSelectionChanged();

  // Tools
  void runDoctor();
  void openRoiEditor();
  void openPrewarpEditor();
  void togglePreview();

  // Config round-trip
  void applyUiToConfig();
  void reloadConfigIntoUi();
  void onAdvancedAddRow();
  void onAdvancedRemoveRow();
  void onAdvancedFilter(const QString& text);
  void applyRawToModel();
  void fromModelToRaw();

private:
  void buildMenus();
  void buildToolBar();
  void buildStatusBar();
  void buildDocks();
  void applyDarkTheme();

  void refreshSourceList();
  void loadSourceIntoUi(int index);
  void persistCurrentSource();
  bool validatePaths(QString& message, const QString& runtime, const QString& country);
  void updateStatusIndicators();
  QString currentConfPath() const;

  // helper for key/value editing
  void refreshAdvancedTable();

  ProjectModel project_;
  ConfigModel configModel_;
  QString currentProjectPath_;
  int currentSourceIndex_ = -1;
  bool previewRunning_ = false;

  // UI widgets
  QListWidget* sourceList_ = nullptr;
  QWidget* previewPanel_ = nullptr;
  QLabel* previewStatusLabel_ = nullptr;
  QTabWidget* configTabs_ = nullptr;

  // Status indicators
  QLabel* runtimeIndicator_ = nullptr;
  QLabel* cascadeIndicator_ = nullptr;
  QLabel* tessIndicator_ = nullptr;
  QLabel* sourceIndicator_ = nullptr;

  // Shared controls across tabs
  QLineEdit *confPathEdit_, *uriEdit_, *fpsEdit_, *frameSkipEdit_, *bufferEdit_;
  QComboBox *typeCombo_, *countryEdit_;
  QLineEdit *runtimeEdit_;

  // Detection
  QCheckBox* skipDetectionCheck_;
  QLineEdit* detectorTypeEdit_;
  QLineEdit* detectorConfigEdit_;

  // OCR
  QComboBox* vehicleCombo_;
  QComboBox* scenarioCombo_;
  QLineEdit* burstEdit_;
  QLineEdit* voteWindowEdit_;
  QLineEdit* minVotesEdit_;
  QCheckBox* fallbackCheck_;

  // Prewarp
  QCheckBox* prewarpEnableCheck_;
  QPlainTextEdit* prewarpPointsEdit_;

  // ROI / Crossing
  QPlainTextEdit* roiEdit_;
  QLineEdit* lineEdit_;
  QLineEdit* motionThreshEdit_;
  QLineEdit* motionAreaEdit_;
  QLineEdit* motionRatioEdit_;
  QLineEdit* debounceEdit_;
  QLineEdit* armFramesEdit_;
  QCheckBox* ocrAfterCrossCheck_;

  // Logging & metrics
  QCheckBox* logPlatesCheck_;
  QCheckBox* logOcrCheck_;
  QLineEdit* logEveryNEdit_;
  QLineEdit* logFileEdit_;
  QLineEdit* reportJsonEdit_;

  // Advanced / raw
  QTableWidget* advancedTable_;
  QLineEdit* advancedFilter_;
  QPlainTextEdit* rawEdit_;

  // Actions
  QAction *actOpenProject_, *actSaveProject_, *actSaveProjectAs_, *actExportConf_;
  QAction *actAddSource_, *actDupSource_, *actRemoveSource_, *actPreview_, *actRoi_, *actPrewarp_, *actDoctor_;
};

