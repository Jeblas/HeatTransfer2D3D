cmake_minimum_required(VERSION 3.8)
project(heat2D3D)# LANGUAGES CXX CUDA)

#find_package()

include_directories(src)

set(SOURCE_FILES)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} src/main.cc)

