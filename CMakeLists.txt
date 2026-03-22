cmake_minimum_required(VERSION 3.16)
project(rod_free_volume LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
    add_compile_definitions(USE_OPENMP)
    link_libraries(OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found — running in serial mode")
endif()

# ── CUDA (optional) ──
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA found — enabling GPU path")
    add_compile_definitions(USE_CUDA)
    set(CUDA_SOURCES src/free_volume_cuda.cu)
else()
    message(STATUS "CUDA not found — GPU path disabled")
    set(CUDA_SOURCES "")
endif()

# ── Include paths ──
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── Library ──
add_library(fvol_core
    src/free_volume.cpp
    ${CUDA_SOURCES}
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fvol_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── Main executable ──
add_executable(rod_free_volume src/main.cpp)
target_link_libraries(rod_free_volume PRIVATE fvol_core)

# ── Test executable ──
add_executable(rod_free_volume_test test/test_basic.cpp)
target_link_libraries(rod_free_volume_test PRIVATE fvol_core)

# ── Install ──
install(TARGETS rod_free_volume DESTINATION bin)
