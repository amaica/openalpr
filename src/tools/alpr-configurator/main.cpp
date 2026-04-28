#include <QApplication>
#include "mainwindow.h"
#include "build_info.h"
#include <iostream>
#include <QCoreApplication>

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
  QString projectArg;
  for (int i=1;i<argc;i++) {
    QString arg = QString::fromLocal8Bit(argv[i]);
    if (arg == "--version") {
      QString exe = QCoreApplication::applicationFilePath();
      std::cout << "alpr-configurator --version\n";
      std::cout << "git_hash=" << ALPR_GIT_HASH << "\n";
      std::cout << "build_time=" << ALPR_BUILD_TIME << "\n";
      std::cout << "build_type=" << ALPR_BUILD_TYPE << "\n";
      std::cout << "exe_path=" << exe.toStdString() << "\n";
      return 0;
    } else {
      projectArg = arg;
    }
  }
  MainWindow w(projectArg);
  w.show();
  return a.exec();
}

