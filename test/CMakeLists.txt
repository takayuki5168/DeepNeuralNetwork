cmake_minimum_required(VERSION 2.8)

# set runtime directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set(GTEST_ROOT ${CMAKE_CURRENT_DIR}/external/googletest/googletest/)
include_directories(${GTEST_ROOT}/include/)
#link_directories(${GTEST_ROOT}/build/)
add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/dense.cpp)
target_link_libraries(test dense gtest gtest_main pthread)
