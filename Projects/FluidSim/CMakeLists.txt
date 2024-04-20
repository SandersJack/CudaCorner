cmake_minimum_required(VERSION 3.18)
project(fluid_testing LANGUAGES CUDA C)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

include_directories(include)

file(GLOB SOURCES "src/*.c" "src/*.cu")

# Add your CUDA source file
add_executable(fluid_sim main.c ${SOURCES})

target_link_libraries(fluid_sim ${SDL2_LIBRARIES} m)

set_target_properties(fluid_sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)