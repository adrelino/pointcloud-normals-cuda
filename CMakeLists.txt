project(point-pair-features-cuda)


cmake_minimum_required(VERSION 2.8)
#--- These settings will be inherited by CUDA
#set(CMAKE_CXX_COMPILER "/usr/local/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#--- Setup
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS}) 
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
# If you have some C++11 stuff you don't want to propagate
set(CUDA_PROPAGATE_HOST_FLAGS False)

include_directories(include/)

#-- Compile
#cuda_add_library(simple STATIC src/simple.cu src/cov.cu)
#cuda_add_library(eig STATIC src/eig.cu)
#cuda_add_library(cov STATIC src/cov.cu)

cuda_add_executable(main
    src/main.cpp
    src/simple.cu
    src/eig.cu
)

#target_link_libraries(simple eig)
#target_link_libraries(main eig cov simple ${CUDA_LIBRARIES})

#add_custom_command(TARGET main PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/bunny $<TARGET_FILE_DIR:main>/bunny)


###### compile the matlab kernels ptx files
# this simple command line: nvcc -ptx common_kernels.cu -I ../include/ -DCUDA_MATLAB
# is much harder in Cmake, see my stackoverflow question here:
# http://stackoverflow.com/questions/26198294/nvcc-compile-to-ptx-using-cmakes-cuda-compile-ptx/26200673#26200673
cuda_compile_ptx(
  cuda_ptx_files
  src/eig.cu
)
#we need to rename, since this does not work: set(generated_file_basename "${basename}.ptx")
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/eig.ptx COMMAND ${CMAKE_COMMAND} -E rename ${cuda_ptx_files} ${CMAKE_BINARY_DIR}/eig.ptx DEPENDS ${cuda_ptx_files}) 
add_custom_target(ptx ALL
    DEPENDS ${CMAKE_BINARY_DIR}/eig.ptx src/eig.cu
    SOURCES src/eig.cu
)

cuda_compile_ptx(
  cuda_ptx_files
  src/cov.cu
)
#we need to rename, since this does not work: set(generated_file_basename "${basename}.ptx")
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/cov.ptx COMMAND ${CMAKE_COMMAND} -E rename ${cuda_ptx_files} ${CMAKE_BINARY_DIR}/cov.ptx DEPENDS ${cuda_ptx_files}) 
add_custom_target(ptx2 ALL
    DEPENDS ${CMAKE_BINARY_DIR}/cov.ptx src/cov.cu
    SOURCES src/cov.cu
)
