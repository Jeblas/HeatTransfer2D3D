cmake_minimum_required(VERSION 3.9)
project(heat2D3D)

find_package(CUDA QUIET REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11;)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS})

 
include_directories(
    src
)

set(SOURCE_FILES
    src/cuda-heat-diffusion.cu
)

cuda_add_executable(${PROJECT_NAME} ${SOURCE_FILES} )