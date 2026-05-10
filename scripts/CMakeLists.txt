cmake_minimum_required(VERSION 3.20)
project(cam_to_world CXX)
set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options($<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-march=native>)

# Eigen3 (header-only)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Intel RealSense SDK 2
find_package(realsense2 REQUIRED)

# Flexiv RDK  (set -DCMAKE_PREFIX_PATH=~/rdk_install)
find_package(flexiv_rdk REQUIRED)

add_executable(cam_to_world cam_to_world.cpp)

target_link_libraries(cam_to_world PRIVATE
  Eigen3::Eigen
  realsense2::realsense2
  flexiv_rdk::flexiv_rdk
  pthread
)
