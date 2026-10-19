cmake_minimum_required(VERSION 3.10)
project(liuyunxi_green_ball_tracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(green_ball_tracker main.cpp)
target_include_directories(green_ball_tracker PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(green_ball_tracker PRIVATE ${OpenCV_LIBS})