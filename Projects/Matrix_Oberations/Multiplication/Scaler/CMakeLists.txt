cmake_minimum_required(VERSION 3.18)
project(cuda_example LANGUAGES CUDA)

# Add your CUDA source file
add_executable(scalerMult scalerMult.cu)

# CUDA-specific settings
set_target_properties(scalerMult PROPERTIES CUDA_SEPARABLE_COMPILATION ON)