cmake_minimum_required(VERSION 3.16)
project(vla_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VLA_OPENMP "Parallelise matrix products and attention with OpenMP" ON)
option(VLA_BUILD_PYTHON "Build the vla_cpp Python module (needs pybind11)" OFF)
option(VLA_NATIVE "Optimise for the build machine's CPU (-march=native: AVX2/FMA where available)" ON)

add_library(vla_runtime src/policy.cpp)
target_include_directories(vla_runtime PUBLIC include)
target_compile_options(vla_runtime PRIVATE -Wall -Wextra -Wpedantic -O3)
if(VLA_NATIVE)
  target_compile_options(vla_runtime PRIVATE -march=native)
endif()
if(VLA_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(vla_runtime PUBLIC OpenMP::OpenMP_CXX)
  endif()
else()
  target_compile_options(vla_runtime PRIVATE -Wno-unknown-pragmas)
endif()

add_executable(vla_bench src/bench.cpp)
target_link_libraries(vla_bench PRIVATE vla_runtime)

enable_testing()
add_executable(test_runtime tests/test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE vla_runtime)
# A small randomly initialised model keeps the C++ tests self-contained.
add_test(NAME runtime_parity_tiny
         COMMAND test_runtime ${CMAKE_CURRENT_SOURCE_DIR}/tests/tiny_policy.bin
                              ${CMAKE_CURRENT_SOURCE_DIR}/tests/tiny_fixtures.bin)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../checkpoints/policy.bin)
  add_test(NAME runtime_parity_trained
           COMMAND test_runtime ${CMAKE_CURRENT_SOURCE_DIR}/../checkpoints/policy.bin
                                ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures.bin)
endif()

if(VLA_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(vla_cpp src/bindings.cpp)
  target_link_libraries(vla_cpp PRIVATE vla_runtime)
endif()
