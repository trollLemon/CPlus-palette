cmake_minimum_required(VERSION 3.10)
project(cpluspalette VERSION 1.3.0 DESCRIPTION "generate color palettes from images" LANGUAGES CXX)

option(USE_CUDA "Enable CUDA support" OFF)

if (USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    set(CMAKE_CUDA_FLAGS "--cudart shared -gencode arch=compute_75,code=sm_75")  # Replace 75 with your GPU's architecture
endif()

set(CPP_SOURCE_FILES
    src/common/color.cpp
    src/common/LoadAndSelect.cpp
    src/kmean/cluster.cpp
    src/kmean/k_mean.cpp
    src/kmean/adv_color.cpp
    src/mediancut/median_cut.cpp
    src/mediancut/median_cut_helpers.cpp
    src/main.cpp
)

if (USE_CUDA)
    set(CUDA_SOURCE_FILES
        src/kmean/k_mean.cu
    )
endif()

set_source_files_properties(
    ${CPP_SOURCE_FILES} ${CUDA_SOURCE_FILES}
    PROPERTIES
    COMPILE_FLAGS "-O3 -finline-functions -march=native -Wall -Wextra -s"
)

find_package(X11 REQUIRED)
include_directories(${X11_INCLUDE_DIR})
include_directories("src/includes/")

# Define the executable
add_executable(cpluspalette ${CPP_SOURCE_FILES} ${CUDA_SOURCE_FILES})

# Link CUDA libraries if CUDA is enabled
if (USE_CUDA)
    target_include_directories(cpluspalette PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(cpluspalette PRIVATE ${CUDA_LIBRARIES})
    target_link_libraries(cpluspalette PRIVATE ${CUDA_cudart_LIBRARY})
endif()

target_link_libraries(cpluspalette PRIVATE ${X11_LIBRARIES})

