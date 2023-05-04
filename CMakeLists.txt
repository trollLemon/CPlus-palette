cmake_minimum_required(VERSION 3.10)

# set the project name
project(cpluspalette VERSION 1.1.0)



set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE release)
add_executable(cpluspalette 
    src/COMMON/color.cpp 
    src/COMMON/LoadAndSelect.cpp 
    src/K_MEAN/cluster.cpp 
    src/K_MEAN/k_mean.cpp 
    src/K_MEAN/adv_color.cpp 
    src/MEDIAN_CUT/median_cut.cpp 
    src/MEDIAN_CUT/median_cut_helpers.cpp 
    src/main.cpp)
set_source_files_properties( 
    src/COMMON/color.cpp
    src/COMMON/LoadAndSelect.cpp 
    src/K_MEAN/cluster.cpp
    src/K_MEAN/k_mean.cpp 
    src/K_MEAN/adv_color.cpp
    src/MEDIAN_CUT/median_cut.cpp
    src/MEDIAN_CUT/median_cut_helpers.cpp
    src/main.cpp

    PROPERTIES
    COMPILE_FLAGS "-O5")


find_package(X11 REQUIRED)
include_directories(${X11_INCLUDE_DIR})
include_directories("HEADERS/")

target_link_libraries(cpluspalette ${X11_LIBRARIES})


install(TARGETS cpluspalette )
