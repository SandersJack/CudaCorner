cmake_minimum_required(VERSION 3.18)
project(fluid_testing LANGUAGES C)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# Add your CUDA source file
add_executable(fluid_testing main.c)

target_link_libraries(fluid_testing ${SDL2_LIBRARIES})