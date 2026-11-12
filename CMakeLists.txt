cmake_minimum_required(VERSION 3.24)

project(FramePipe VERSION 0.2.0 LANGUAGES CXX)

option(FRAMEPIPE_ENABLE_CUDA "Build the custom CUDA backend (explicit request; fails if unavailable)" OFF)
option(FRAMEPIPE_BUILD_TESTS "Build unit and integration tests" ON)
option(FRAMEPIPE_ENABLE_OPENMP "Build the cpu-omp parallel backend when OpenMP is found" ON)
option(FRAMEPIPE_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer (CPU code)" OFF)
option(FRAMEPIPE_OMP_STATIC_SCHEDULE "cpu-omp baseline variant: one static chunk per thread (comparison only)" OFF)
option(FRAMEPIPE_CUDA_DEBUG_SYNC "Synchronize after every CUDA kernel (diagnostics only; never for benchmarks)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# ISO C++ (no GNU extensions). With GCC this also implies -ffp-contract=off,
# so the CPU reference does not silently contract multiply-adds.
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---------------------------------------------------------------- CUDA probe
set(FRAMEPIPE_HAVE_CUDA OFF)
if(FRAMEPIPE_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "FRAMEPIPE_ENABLE_CUDA=ON was requested but no CUDA compiler was found. "
      "Install a CUDA toolkit compatible with your host compiler, or set "
      "CMAKE_CUDA_COMPILER, or configure with FRAMEPIPE_ENABLE_CUDA=OFF.")
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
      "FRAMEPIPE_ENABLE_CUDA=ON requires an explicit CMAKE_CUDA_ARCHITECTURES. "
      "Use 'native' on a machine with a visible supported GPU, or an explicit "
      "list (for example 86) chosen for the intended device. FramePipe does "
      "not guess the target compute capability.")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  find_package(CUDAToolkit REQUIRED)
  set(FRAMEPIPE_HAVE_CUDA ON)
endif()

# ---------------------------------------------------------------- deps
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(nlohmann_json 3.11 REQUIRED)
find_package(Threads REQUIRED)

set(FRAMEPIPE_HAVE_OPENMP OFF)
if(FRAMEPIPE_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    set(FRAMEPIPE_HAVE_OPENMP ON)
  endif()
endif()

# ---------------------------------------------------------------- warnings
add_library(framepipe_warnings INTERFACE)
target_compile_options(framepipe_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion>)

add_library(framepipe_sanitize INTERFACE)
if(FRAMEPIPE_SANITIZE)
  target_compile_options(framepipe_sanitize INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all>)
  target_link_options(framepipe_sanitize INTERFACE -fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------- build info
string(TOUPPER "${CMAKE_BUILD_TYPE}" _fp_bt)
set(FRAMEPIPE_CXX_FLAGS_EFFECTIVE "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_fp_bt}}")
set(FRAMEPIPE_CUDA_FLAGS_EFFECTIVE "")
set(FRAMEPIPE_CUDA_ARCHS "")
if(FRAMEPIPE_HAVE_CUDA)
  set(FRAMEPIPE_CUDA_FLAGS_EFFECTIVE "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${_fp_bt}}")
  set(FRAMEPIPE_CUDA_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")
endif()
foreach(_v FRAMEPIPE_HAVE_CUDA FRAMEPIPE_HAVE_OPENMP FRAMEPIPE_SANITIZE FRAMEPIPE_CUDA_DEBUG_SYNC)
  if(${_v})
    set(${_v}_INT true)
  else()
    set(${_v}_INT false)
  endif()
endforeach()
configure_file(src/core/build_info.hpp.in ${CMAKE_BINARY_DIR}/generated/framepipe/build_info.hpp @ONLY)

# ---------------------------------------------------------------- core lib
add_library(framepipe_core STATIC
  src/core/errors.cpp
  src/core/image_view.cpp
  src/core/config.cpp
  src/core/tables.cpp
  src/core/tracked_alloc.cpp
  src/core/sha256.cpp
  src/core/pipeline.cpp
  src/cpu/cpu_ops.cpp
  src/cpu/cpu_backend.cpp
  src/validation/compare.cpp
)
target_include_directories(framepipe_core
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(framepipe_core
  PUBLIC  Threads::Threads
  PRIVATE nlohmann_json::nlohmann_json framepipe_warnings framepipe_sanitize)
target_compile_definitions(framepipe_core PRIVATE
  FRAMEPIPE_HAVE_CUDA=$<BOOL:${FRAMEPIPE_HAVE_CUDA}>
  FRAMEPIPE_HAVE_OPENMP=$<BOOL:${FRAMEPIPE_HAVE_OPENMP}>
  FRAMEPIPE_OMP_STATIC_SCHEDULE=$<BOOL:${FRAMEPIPE_OMP_STATIC_SCHEDULE}>)
if(FRAMEPIPE_HAVE_OPENMP)
  target_sources(framepipe_core PRIVATE src/cpu/cpu_parallel.cpp)
  target_link_libraries(framepipe_core PRIVATE OpenMP::OpenMP_CXX)
endif()

if(FRAMEPIPE_HAVE_CUDA)
  add_library(framepipe_cuda STATIC
    src/cuda/cuda_backend.cu
    src/cuda/kernels.cu
    src/cuda/cuda_info.cu)
  target_include_directories(framepipe_cuda
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/generated)
  target_link_libraries(framepipe_cuda PRIVATE CUDA::cudart framepipe_warnings)
  # Normal FP32 arithmetic: no --use_fast_math. nvcc's default FMA contraction
  # remains enabled; tolerances in docs/numerical_contract.md account for it.
  target_compile_options(framepipe_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings -lineinfo>)
  target_compile_definitions(framepipe_cuda PRIVATE
    FRAMEPIPE_CUDA_DEBUG_SYNC=$<BOOL:${FRAMEPIPE_CUDA_DEBUG_SYNC}>)
  set_target_properties(framepipe_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(framepipe_core PRIVATE framepipe_cuda)
endif()

# ---------------------------------------------------------------- io / app lib
add_library(framepipe_app_lib STATIC
  src/io/image_io.cpp
  src/io/manifest.cpp
  src/io/json_util.cpp
  src/benchmark/stats.cpp
  src/benchmark/environment.cpp
  src/benchmark/harness.cpp
  src/benchmark/opencv_baseline.cpp
  src/app/cli_args.cpp
  src/app/commands.cpp
  src/app/batch.cpp
)
target_include_directories(framepipe_app_lib
  PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(framepipe_app_lib
  PUBLIC framepipe_core nlohmann_json::nlohmann_json
         opencv_core opencv_imgproc opencv_imgcodecs
  PRIVATE framepipe_warnings framepipe_sanitize)
target_compile_definitions(framepipe_app_lib PRIVATE
  FRAMEPIPE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_executable(framepipe src/app/main.cpp)
target_link_libraries(framepipe PRIVATE framepipe_app_lib framepipe_warnings framepipe_sanitize)

install(TARGETS framepipe RUNTIME DESTINATION bin)

# ---------------------------------------------------------------- tests
if(FRAMEPIPE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "FramePipe: CUDA backend=${FRAMEPIPE_HAVE_CUDA} OpenMP backend=${FRAMEPIPE_HAVE_OPENMP} sanitize=${FRAMEPIPE_SANITIZE}")
