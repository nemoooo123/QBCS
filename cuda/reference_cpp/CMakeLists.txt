cmake_minimum_required(VERSION 3.12)
project(qbcs CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

set(QBCS_LIB_SOURCES
  src/encoding.cpp
  src/synth.cpp
  src/algorithms_common.cpp
  src/quantum_algos.cpp
  src/baseline_algos.cpp
  src/data_loader.cpp)

add_library(qbcs_core STATIC ${QBCS_LIB_SOURCES})
target_include_directories(qbcs_core PUBLIC src)
target_link_libraries(qbcs_core PUBLIC Threads::Threads)

add_executable(qbcs src/main.cpp)
target_link_libraries(qbcs PRIVATE qbcs_core)

# Equivalence test against the Python implementation (see tests/README in README.md)
add_executable(replay_test tests/replay_test.cpp)
target_link_libraries(replay_test PRIVATE qbcs_core)

if(MSVC)
  target_compile_options(qbcs_core PRIVATE /W3 /O2)
  target_compile_options(qbcs PRIVATE /W3 /O2)
else()
  target_compile_options(qbcs_core PRIVATE -Wall -Wextra -O2)
  target_compile_options(qbcs PRIVATE -Wall -Wextra -O2)
endif()
