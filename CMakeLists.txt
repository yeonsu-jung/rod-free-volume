cmake_minimum_required(VERSION 3.16)
project(rod_free_volume LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24 AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build" FORCE)
endif()

option(ROD_FREE_VOLUME_ENABLE_CUDA "Build CUDA acceleration when a toolkit is available" ON)

# ── OpenMP ──
# On macOS with Apple Clang, OpenMP requires Homebrew libomp.
# Set hints so find_package can locate it.
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE BREW_RESULT
    )
    if(BREW_RESULT EQUAL 0 AND EXISTS "${HOMEBREW_LIBOMP_PREFIX}")
        message(STATUS "Found Homebrew libomp at ${HOMEBREW_LIBOMP_PREFIX}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found — enabling multithreaded path")
else()
    message(WARNING "OpenMP not found — running in serial mode")
endif()

# ── CUDA (optional) ──
set(CUDA_SOURCES "")
set(CUDA_LIBRARIES "")

if(ROD_FREE_VOLUME_ENABLE_CUDA)
    include(CheckLanguage)

    if(NOT CMAKE_CUDA_COMPILER)
        find_program(ROD_FREE_VOLUME_NVCC
            NAMES nvcc
            HINTS
                ENV CUDA_HOME
                ENV CUDA_PATH
                /usr/local/cuda
                /usr/local/cuda-12.9
                /usr/local/cuda-12.7
                /usr/local/cuda-12.0
                /usr/local/cuda-12
            PATH_SUFFIXES bin)

        if(ROD_FREE_VOLUME_NVCC)
            set(CMAKE_CUDA_COMPILER "${ROD_FREE_VOLUME_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
        endif()
    endif()

    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        message(STATUS "CUDA found — enabling GPU path with ${CMAKE_CUDA_COMPILER}")
        set(CUDA_SOURCES src/free_volume_cuda.cu)
        set(CUDA_LIBRARIES CUDA::cudart)
    else()
        message(STATUS "CUDA not found — GPU path disabled")
    endif()
else()
    message(STATUS "CUDA explicitly disabled")
endif()

# ── Library ──
add_library(fvol_core
    src/free_volume.cpp
    ${CUDA_SOURCES}
)

target_include_directories(fvol_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(fvol_core PUBLIC cxx_std_17)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fvol_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(fvol_core PUBLIC USE_OPENMP)
endif()

if(CUDA_SOURCES)
    target_link_libraries(fvol_core PUBLIC ${CUDA_LIBRARIES})
    target_compile_definitions(fvol_core PUBLIC USE_CUDA)
endif()

# ── Main executable ──
add_executable(rod_free_volume src/main.cpp)
target_link_libraries(rod_free_volume PRIVATE fvol_core)

# ── Test executable ──
add_executable(rod_free_volume_test test/test_basic.cpp)
target_link_libraries(rod_free_volume_test PRIVATE fvol_core)

# ── Install ──
install(TARGETS rod_free_volume DESTINATION bin)
