cmake_minimum_required(VERSION 2.6)
project(uncertainty_double)

# Add the source files
file(GLOB SOURCES
    src/*.h
    src/*.cpp
)
add_executable(uncertainty_double ${SOURCES})

# ------------------------------------------
# Add eigen headers
set(EIGEN_MAIN_DIR "lib/eigen" PATH)
include_directories( ${EIGEN_MAIN_DIR} )


# ------------------------------------------
# Add magma headers
set(MAGMA_MAIN_DIR "lib/magma-2.2.0" PATH)
include_directories( ${MAGMA_MAIN_DIR}/include )
include_directories( ${MAGMA_MAIN_DIR}/control )
include_directories( ${MAGMA_MAIN_DIR}/sparse-iter/include )
include_directories( ${MAGMA_MAIN_DIR}/sparse-iter/control )
include_directories( ${MAGMA_MAIN_DIR}/testing )

# Add magma library
set(MAGMA_LIBRARY_DIR "lib/magma-2.2.0/build/lib/Release" PATH)
target_link_libraries(uncertainty_double ${MAGMA_LIBRARY_DIR}/tester.lib ${MAGMA_LIBRARY_DIR}/lapacktest.lib ${MAGMA_LIBRARY_DIR}/magma.lib)

# ------------------------------------------
# Add Matlab headers
set(MATLAB_EXTERN_INCLUDE "C:/Program Files/MATLAB/R2016b/extern/include" PATH)
include_directories( ${MATLAB_EXTERN_INCLUDE} )

set(MATLAB_LIBRARY_DIR "C:/Program Files/MATLAB/R2016a/extern/lib/win64/microsoft" PATH)
target_link_libraries(uncertainty_double ${MATLAB_LIBRARY_DIR}/libmx.lib ${MATLAB_LIBRARY_DIR}/libmat.lib ${MATLAB_LIBRARY_DIR}/libeng.lib ${MATLAB_LIBRARY_DIR}/libmex.lib )


# ------------------------------------------
# Add Lapack and Blass headers ( Intel MKL were used originally )
set(LAPACK_BLASS_INCLUDE "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2016.3.207/windows/mkl/include" PATH)
include_directories( ${LAPACK_BLASS_INCLUDE} )

# Add Lapack and Blass library
set(LAPACK_BLASS_LIBRARIES_PATH "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2016.3.207/windows/mkl/lib/intel64_win/mkl_intel_lp64_dll.lib; C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2016.3.207/windows/mkl/lib/intel64_win/mkl_intel_thread_dll.lib; C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2016.3.207/windows/mkl/lib/intel64_win/mkl_core_dll.lib; C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2016.4.246/windows/compiler/lib/intel64_win/libiomp5md.lib" STRING)
target_link_libraries(uncertainty_double ${LAPACK_BLASS_LIBRARIES_PATH})


# ------------------------------------------
# Cuda support 
set(CUDA_INCLUDE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/include" PATH)
include_directories( ${CUDA_INCLUDE} )
 
set(CUDA_LIBRARY_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/lib/x64" PATH)
target_link_libraries(uncertainty_double ${CUDA_LIBRARY_DIR}cublas.lib ${CUDA_LIBRARY_DIR}/cusparse.lib ${CUDA_LIBRARY_DIR}/cublas_device.lib ${CUDA_LIBRARY_DIR}/cudadevrt.lib ${CUDA_LIBRARY_DIR}/cudart_static.lib)

