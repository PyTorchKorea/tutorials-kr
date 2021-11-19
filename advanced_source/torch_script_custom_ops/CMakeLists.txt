cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(warp_perspective)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# 라이브러리 타겟 정의
add_library(warp_perspective SHARED op.cpp)
# C++14 사용
target_compile_features(warp_perspective PRIVATE cxx_std_14)
# LibTorch 링크
target_link_libraries(warp_perspective "${TORCH_LIBRARIES}")
# OpenCV 링크
target_link_libraries(warp_perspective opencv_core opencv_imgproc)
