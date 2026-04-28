cmake_minimum_required(VERSION 3.5)
project(openalpr-superbuild)

# Ensure module path points to src cmake modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/src/cmake_modules/")
# Align source dir expectations for downstream CMake that assumes build from src/
set(CMAKE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src" CACHE INTERNAL "" FORCE)

# Delegate to the main project under src/
add_subdirectory(src)

