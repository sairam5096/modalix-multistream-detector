cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(shared-multi-model-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")
sima_neat_apps_module("shared-multi-model-detector" OUTPUT_TARGET_VAR _smm_target)

# std::thread needs pthread; opencv + neat + nlohmann come transitively.
find_package(Threads REQUIRED)
target_link_libraries(${_smm_target} PRIVATE Threads::Threads)

# YAML config parsing (yaml-cpp is in the Modalix sysroot).
find_package(yaml-cpp QUIET)
if (TARGET yaml-cpp)
  target_link_libraries(${_smm_target} PRIVATE yaml-cpp)
else()
  target_link_libraries(${_smm_target} PRIVATE yaml-cpp)  # link by name from sysroot
endif()

# httplib.h ships as a compiled split library in the Modalix sysroot
# (libcpp-httplib.so); link it for the HTTP control server.
target_link_libraries(${_smm_target} PRIVATE cpp-httplib)
