cmake_minimum_required(VERSION 3.10)
project(first_homework_ball_tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 只链接实际使用的模块，避免引入 viz 等无关模块及其 VTK 依赖。
# OpenCV 5 将部分轮廓几何接口移到了 geometry 模块；OpenCV 4 不需要它。
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio OPTIONAL_COMPONENTS geometry)

add_executable(ball_tracker src/main.cpp)
target_include_directories(ball_tracker PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ball_tracker PRIVATE ${OpenCV_LIBS})
