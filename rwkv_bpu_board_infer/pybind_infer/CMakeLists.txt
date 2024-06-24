cmake_minimum_required(VERSION 3.5)
project(main)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(pybind11 REQUIRED)
pybind11_add_module(bpu_infer_lib main.cpp)

target_link_directories( bpu_infer_lib
 PRIVATE
  /usr/lib/hbbpu/
)

include_directories(
  include
  /usr/include
)

# Consistently use keyword signature for target_link_libraries
target_link_libraries(bpu_infer_lib
  PRIVATE
    opencv_world
    hlog
    easy_dnn
    dnn
    # hbdk_sim_x86
)
