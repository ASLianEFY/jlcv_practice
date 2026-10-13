# =============================================================================
#  RoboMaster 视觉组进阶作业 —— 绿球识别与运动矢量(箭头)标注
#
#  单文件工程: 全部实现都在 ball_tracker.cpp 里（约 2270 行）
#
#      homework/YaoMuXi/
#        ├── CMakeLists.txt    <- 本文件
#        ├── ball_tracker.cpp  <- 全部代码（数据结构/检测/卡尔曼/追踪/标注/主程序/自检）
#        ├── params.yaml       <- 可选参数文件（不带也能跑，默认值已内置）
#        └── build/            <- 构建目录（cmake 生成）
#
#  Ubuntu 编译:
#     sudo apt update
#     sudo apt install -y build-essential cmake libopencv-dev
#     cd homework/YaoMuXi
#     mkdir -p build && cd build
#     cmake ..
#     make -j$(nproc)
#
#  如果 build/ 下没有生成 Makefile，说明上面的 cmake 没成功（请看它的报错）。
#  也可以直接用与生成器无关的写法（不用 cd 来 cd 去）：
#     cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#     cmake --build build -j$(nproc)
#
#  运行（build/ 目录下）:
#     ./rm_ball_tracker --video ~/Desktop/homework.mp4 --output out.avi --csv track.csv
#     ./rm_ball_tracker --selftest      # 卡尔曼滤波自检，不需要视频
#     ./rm_ball_tracker --help
#
#  无显示器的机器人本体（不装 Qt/GTK 运行库）:
#     cmake .. -DRMV_ENABLE_HIGHGUI=OFF && make -j$(nproc)
# =============================================================================
cmake_minimum_required(VERSION 3.10)

project(rm_green_ball_tracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release/Debug)" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# -----------------------------------------------------------------------------
# OpenCV (Ubuntu: sudo apt install libopencv-dev)
# -----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version : ${OpenCV_VERSION}")
message(STATUS "OpenCV include : ${OpenCV_INCLUDE_DIRS}")

# -----------------------------------------------------------------------------
# highgui 可选:
#   ON  : 支持 --show 实时预览（运行时需要 Qt/GTK 等 GUI 运行库）
#   OFF : headless，只能 --output 存盘（机器人本体推荐）
# -----------------------------------------------------------------------------
option(RMV_ENABLE_HIGHGUI "启用 highgui 实时预览(--show)" ON)

# -----------------------------------------------------------------------------
# 唯一的目标程序（全部代码都在 ball_tracker.cpp）
# -----------------------------------------------------------------------------
add_executable(rm_ball_tracker ball_tracker.cpp)

target_include_directories(rm_ball_tracker PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(rm_ball_tracker PRIVATE ${OpenCV_LIBS})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(rm_ball_tracker PRIVATE -Wall -Wextra)
endif()

if(RMV_ENABLE_HIGHGUI)
  list(FIND OpenCV_LIBS opencv_highgui RMV_HIGHGUI_INDEX)
  if(RMV_HIGHGUI_INDEX GREATER -1)
    target_compile_definitions(rm_ball_tracker PRIVATE RMV_HAVE_HIGHGUI=1)
    message(STATUS "highgui        : ON  (支持 --show 实时预览)")
  else()
    message(STATUS "highgui        : OFF (OpenCV 未提供 highgui 模块, 仅支持 --output)")
  endif()
else()
  message(STATUS "highgui        : OFF (已用 -DRMV_ENABLE_HIGHGUI=OFF 关闭, 仅支持 --output)")
endif()

# -----------------------------------------------------------------------------
# 自检：程序内置的卡尔曼滤波单元测试（不需要视频）
#   ctest --output-on-failure   或   ./rm_ball_tracker --selftest
# -----------------------------------------------------------------------------
enable_testing()
add_test(NAME kalman_selftest COMMAND rm_ball_tracker --selftest)
