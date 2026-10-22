cmake_minimum_required(VERSION 3.10)
project(green_ball_tracking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(green_ball_tracking green_ball_tracking.cpp)
target_include_directories(green_ball_tracking PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(green_ball_tracking PRIVATE ${OpenCV_LIBS})
