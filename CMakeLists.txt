cmake_minimum_required(VERSION 3.0)

project(DeepNeuralNetwork)

# Set compiler
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=gnu++14" COMPILER_SUPPORTS_CXX14)
if(COMPILER_SUPPORTS_CXX14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")  
endif()
message("Compiler:\n\t${CMAKE_CXX_COMPILER}")

# Set default build type to Release
if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen
set(EIGEN_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/external/Eigen)
include_directories(SYSTEM ${EIGEN_INCLUDE_DIR})

# Deep Neural Network
include_directories(LIB_HEADERS ${CMAKE_CURRENT_LIST_DIR}/lib/include/)
#file(GLOB LIB_SOURCES ${CMAKE_CURRENT_LIST_DIR}/lib/src/*.cpp) # TODO

# Build main
file(GLOB MAIN_SOURCE ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
add_executable(main 
  ${MAIN_SOURCE}
  ${LIB_SOURCES})
target_link_libraries(main
  #${Boost_LIBRARIES}
  )

# Documentation
add_subdirectory(doc)
