cmake_minimum_required(VERSION 3.5)
project(alpr-configurator)

find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Build info (git hash / build time / build type)
execute_process(
  COMMAND git -C ${CMAKE_SOURCE_DIR} rev-parse HEAD
  OUTPUT_VARIABLE ALPR_GIT_HASH
  RESULT_VARIABLE GIT_HASH_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (NOT GIT_HASH_RESULT EQUAL 0)
  set(ALPR_GIT_HASH "unknown")
endif()
string(TIMESTAMP ALPR_BUILD_TIME "%Y-%m-%d %H:%M" UTC)
if (CMAKE_BUILD_TYPE)
  set(ALPR_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
else()
  set(ALPR_BUILD_TYPE "RelWithDebInfo")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/build_info.h.in ${CMAKE_CURRENT_BINARY_DIR}/build_info.h @ONLY)

add_executable(alpr-configurator
  main.cpp
  configmodel.cpp
  projectmodel.cpp
  mainwindow.cpp
)

target_include_directories(alpr-configurator PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(alpr-configurator PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(alpr-configurator
  Qt5::Widgets
  ${OpenCV_LIBS}
)

install(TARGETS alpr-configurator RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

