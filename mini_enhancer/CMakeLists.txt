cmake_minimum_required(VERSION 3.10)
project(mini_enhancer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(enhancer
  main.cpp
  image_processor.cpp
)
set_target_properties(enhancer PROPERTIES OUTPUT_NAME mini_enhancer)

target_include_directories(enhancer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(enhancer PRIVATE ${OpenCV_LIBS})

include(GNUInstallDirs)
install(TARGETS enhancer RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(MSVC)
  target_compile_options(enhancer PRIVATE /W4)
else()
  target_compile_options(enhancer PRIVATE -Wall -Wextra)
endif()
