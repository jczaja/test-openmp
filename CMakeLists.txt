cmake_minimum_required(VERSION 2.8)
project(test-openmp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CheckCXXCompilerFlag)

find_package(OpenMP)
if(OPENMP_FOUND)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
fatal("OpenMP support missing in compiler")
endif()

if(UNIX)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
add_definitions(-std=c++11)
elseif(COMPILER_SUPPORTS_CXX0X)
add_definitions(-std=c++0x)
endif()
endif()

add_definitions(-I/usr/lib/gcc/x86_64-redhat-linux/4.9.2/include/)


#add_definitions(-fopt-info-vec-missed)
#add_definitions(-fopt-info-vec-optimized)
add_definitions(-march=native)
add_definitions(-Wall)
add_executable(test-openmp main.cpp)
#set_target_properties(test-openmp PROPERTIES COMPILE_FLAGS "-pg" LINK_FLAGS "-pg" )
