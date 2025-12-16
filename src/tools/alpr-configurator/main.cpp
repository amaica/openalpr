#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
  QString projectArg;
  if (argc > 1) projectArg = QString::fromLocal8Bit(argv[1]);
  MainWindow w(projectArg);
  w.show();
  return a.exec();
}

