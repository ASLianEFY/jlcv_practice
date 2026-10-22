cmake_minimum_required(VERSION 3.10)
project(green_ball_tracking CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC 需要显式指定源码为 UTF-8（源码里有中文注释）
if (MSVC)
    add_compile_options(/utf-8)
endif()

# 找 OpenCV。Ubuntu 上装好 libopencv-dev 后会自动找到；
# Windows 上需要手动指定 OpenCV_DIR，例如：
#   cmake -DOpenCV_DIR=D:/opencv/build/x64/vc16/lib ..
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")

add_executable(green_ball_tracking main.cpp)
target_include_directories(green_ball_tracking PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(green_ball_tracking PRIVATE ${OpenCV_LIBS})
