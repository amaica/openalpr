cmake_minimum_required(VERSION 3.5)
project(alpr-configurator)

find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

add_executable(alpr-configurator
  main.cpp
  configmodel.cpp
  mainwindow.cpp
)

target_include_directories(alpr-configurator PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(alpr-configurator
  Qt5::Widgets
  ${OpenCV_LIBS}
)

install(TARGETS alpr-configurator RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

