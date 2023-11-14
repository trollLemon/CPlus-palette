cmake_minimum_required(VERSION 3.10)

# set the project name
project(cpluspalette VERSION 1.2.1 DESCRIPTION "generate color palettes from images")



set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE release)
add_executable(cpluspalette 
    src/common/color.cpp 
    src/kmean/cluster.cpp 
    src/kmean/k_mean.cpp 
    src/mediancut/median_cut.cpp 
    src/mediancut/median_cut_helpers.cpp 
    src/main.cpp)
set_source_files_properties( 
    src/common/color.cpp
    src/kmean/cluster.cpp
    src/kmean/k_mean.cpp 
    src/mediancut/median_cut.cpp
    src/mediancut/median_cut_helpers.cpp
    src/main.cpp

    PROPERTIES
    COMPILE_FLAGS "-O2  -march=native -flto -Wall -Wextra -s")
include_directories("src/includes/")



install(TARGETS cpluspalette )
