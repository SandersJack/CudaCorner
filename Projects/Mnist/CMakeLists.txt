cmake_minimum_required(VERSION 3.18)
project(mnist LANGUAGES CUDA C)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

include_directories(include)

file(GLOB SOURCES "src/*.c" "src/*.cu")

# Add your CUDA source file
add_executable(mnist main.c ${SOURCES})

target_link_libraries(mnist ${SDL2_LIBRARIES} m)

set_target_properties(mnist PROPERTIES CUDA_SEPARABLE_COMPILATION ON)