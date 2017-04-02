cmake_minimum_required(VERSION 3.0)

project(voronoi CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Weverything")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-exit-time-destructors")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-global-constructors")
# XXX gflags warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-variable-declarations")

find_package(OpenCV 3 REQUIRED
  COMPONENTS
    core
    highgui
    imgproc
    imgcodecs
)

find_package(gflags REQUIRED)

#option(BUILD_SHARED_LIBS "Shared libs" ON)

add_executable(voronoi_example example.cc)
target_include_directories(voronoi_example
  SYSTEM PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${gflags_INCLUDE_DIRS}
)
target_link_libraries(voronoi_example
  PRIVATE
    ${OpenCV_LIBRARIES}
    ${gflags_LIBRARIES}
)
