cmake_minimum_required(VERSION 3.0)

project(NeuralNetwork)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=gnu++14" COMPILER_SUPPORTS_CXX14)
if(COMPILER_SUPPORTS_CXX14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")
else()
  message(FATAL_ERROR
    "${CMAKE_CXX_COMPILER} はC++14をサポートしてません。
C++14に対応したコンパイラを指定してください。
EXAMPLE:
    $ cmake . -DCMAKE_CXX_COMPILER=/usr/bin/g++-5.4")
endif()
message("Compiler:\n\t${CMAKE_CXX_COMPILER}")

# Set default build type to Release
if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set compile flags
set(CMAKE_CXX_FLAGS
  #    "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion -Wswitch-default -Wdisabled-optimization -Wformat -Winit-self -Woverloaded-virtual -Wfloat-equal")
  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion -Wdisabled-optimization -Wformat -Winit-self -Woverloaded-virtual -Wfloat-equal")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-old-style-cast -Wno-pragmas")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

# Eigen
set(EIGEN_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/external/Eigen)
include_directories(SYSTEM ${EIGEN_INCLUDE_DIR})
add_definitions(-DEIGEN_NO_DEBUG)
add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO)

# deep-neural-network
include_directories(DEEP_NEURAL_NETWORK ${CMAKE_CURRENT_LIST_DIR}/lib/include/)

# Sources and headers
file(GLOB MAIN_SOURCE ${CMAKE_CURRENT_LIST_DIR} main.cpp)

# Echo compile flags
message("Build type:\n\t${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Compile flags:\n\t${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  message("Compile flags:\n\t${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
else()
  message("Compile flags:\n\t${CMAKE_CXX_FLAGS}")
endif()

# Build main
add_executable(main 
  ${MAIN_SOURCE}
  lib/)
target_link_libraries(main
  #${Boost_LIBRARIES}
  )

# Clean
add_custom_target(cmake-clean
  COMMAND rm -rf `find ${CMAKE_BINARY_DIR} -name \"*[cC][mM]ake*\" -and -not -name \"CMakeLists.txt\"`
  COMMAND rm -rf `find ${CMAKE_BINARY_DIR} -name \"Makefile\"`)

add_custom_target(full-clean
  COMMAND make clean
  COMMAND make cmake-clean)

# Documentation
add_subdirectory(doc)
