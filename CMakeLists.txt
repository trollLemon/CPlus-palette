cmake_minimum_required(VERSION 3.10)

# set the project name
project(cpluspalette VERSION 1.0.3)



set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE release)
add_executable(cpluspalette src/colorGen.cpp src/median_cut.cpp src/color.cpp src/main.cpp)
set_source_files_properties( 
    src/main.cpp
	src/color.cpp
    src/median_cut.cpp
    src/colorGen.cpp
    PROPERTIES
    COMPILE_FLAGS "-O3"
    )


find_package(X11 REQUIRED)
include_directories(${X11_INCLUDE_DIR})
target_link_libraries(cpluspalette ${X11_LIBRARIES})


install(TARGETS cpluspalette )
