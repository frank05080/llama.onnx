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

# message("build platform X3")
# add_definitions(-DPLATFORM_X3)
# set(PREFIX_PATH x3)

link_directories(
#   /usr/lib/
  /usr/lib/hbbpu/
#   /usr/lib/hbmedia/
#   ${HOBOTCV_LIB_INSTALL_PATH}
)

include_directories(
  include
  /usr/include
)

add_executable(${PROJECT_NAME}
  main.cpp
)

target_link_libraries(${PROJECT_NAME}
    opencv_world
    hlog
    easy_dnn
    dnn
    # hbdk_sim_x86
  )