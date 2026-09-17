cmake_minimum_required(VERSION 3.16)
project(overlay-detector LANGUAGES CXX)
find_package(SimaNeat CONFIG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
add_executable(overlay-detector
  main.cpp
  support/object_detection/obj_detection_utils.cpp)
target_compile_features(overlay-detector PRIVATE cxx_std_20)
target_include_directories(overlay-detector PRIVATE .)
target_link_libraries(overlay-detector PRIVATE SimaNeat::sima_neat ${OpenCV_LIBS})
