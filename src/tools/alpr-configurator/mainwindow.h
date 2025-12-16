#pragma once
#include <QMainWindow>
#include "configmodel.h"

class QLineEdit;
class QComboBox;
class QCheckBox;
class QPlainTextEdit;
class QTableWidget;
class QLabel;

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(QWidget* parent=nullptr);
private slots:
  void onBrowseConf();
  void onLoad();
  void onSave();
  void onSaveAs();
  void onTestPaths();
  void onAutoSetup();
  void onApplyRaw();
  void onFromUiToRaw();
  void onGenerateCmd();
  void onAdvancedAddRow();
  void onAdvancedRemoveRow();
  void onAdvancedFilter(const QString& text);
private:
  void buildUi();
  void populateFromModel();
  void applyFromUiToModel();
  void refreshAdvancedTable();
  bool validatePaths(QString& message);

  ConfigModel model_;
  QString currentConfPath_;

  // General
  QLineEdit* confPathEdit_;
  QComboBox* countryEdit_;
  QLineEdit* runtimeEdit_;
  QLabel* testStatus_;
  QLabel* statusBarLabel_;

  // Detection
  QCheckBox* skipDetectionCheck_;
  QLineEdit* detectorEdit_;

  // OCR
  QLineEdit* tessdataEdit_;

  // Logging
  QCheckBox* logPlatesCheck_;
  QCheckBox* logOcrCheck_;
  QLineEdit* logEveryNEdit_;

  // Video/Performance
  QCheckBox* ocrAfterCrossCheck_;
  QLineEdit* crossingRoiEdit_;
  QLineEdit* crossingLineEdit_;
  QLineEdit* motionThreshEdit_;
  QLineEdit* motionAreaEdit_;
  QLineEdit* motionRatioEdit_;
  QLineEdit* debounceEdit_;
  QLineEdit* armFramesEdit_;

  // Raw/Advanced
  QPlainTextEdit* rawEdit_;
  QTableWidget* advancedTable_;
  QLineEdit* advancedFilter_;

  QPlainTextEdit* commandOutput_;
};

