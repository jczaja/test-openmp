cmake_minimum_required(VERSION 3.0)
project(test-openmp)

include(CheckCXXCompilerFlag)
include(ExternalProject)

enable_testing()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/xbyak")
add_definitions(-fno-operator-names)

# GFLAGS
find_library(GFLAGS_LIB NAMES gflags)
find_path(GFLAGS_INCLUDES NAMES gflags.h PATHS /usr/include/gflags/)
message("GFLAGS INCLUDES: ${GFLAGS_INCLUDES}")
include_directories(SYSTEM ${GFLAGS_INCLUDES})

find_package(OpenMP)
if(OPENMP_FOUND)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(OpenMP_SIMD_Test_Source
" int main()
{
  const int asize = 8;
  int a[asize] = {1,2,3,4,5,6,7,8};
  int sum = a[0];
# pragma omp simd reduction(+:sum)
  for (int i=1; i<asize; ++i) {
		sum += a[i];
	}
  return 0;
}")
file(WRITE ${CMAKE_BINARY_DIR}/test_openmp.cpp "${OpenMP_SIMD_Test_Source}")
try_compile(COMPILE_SUCCEEDED ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/test_openmp.cpp COMPILE_DEFINITIONS "-fopenmp -Werror -Wunknown-pragmas")

if(COMPILE_SUCCEEDED)
  message("OpenMP SIMD present")
endif()
else()
message("OpenMP support missing in compiler")
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

find_program(PERF NAMES perf)

if(PERF)
message(STATUS "Perf found!")

SET(PERFMON "extern_perfmon")
ExternalProject_Add(
    ${PERFMON}
    GIT_REPOSITORY      "https://git.code.sf.net/p/perfmon2/libpfm4.git"
    GIT_TAG             "8f2208a1ff62e8ef6403d246042e64f941f4f729"   
    SOURCE_DIR          "extern_perfmon"
    BUILD_IN_SOURCE     "True"
    UPDATE_COMMAND      ""
    CONFIGURE_COMMAND   ""
    BUILD_COMMAND       "make"
    INSTALL_COMMAND     ""
)

add_custom_target(
work
COMMAND cmake -DPERFMON=${PERFMON} -E env NUM_REPS=\${NUM_REPS} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_codes.cmake
WORKING_DIRECTORY ${CMAKE_CURRENT_BIN_DIR}
DEPENDS ${PERFMON} test-openmp-gomp
)

endif(PERF)

# roofline needs memtest and cputest
# It is also using gnuplot for generation of script
# it does need memory throughput [GB/sec]
# and cpu throughput [GFLOPS/sec]
add_custom_target(
roofline
COMMAND cmake -DMEMORY_THROUGHPUT="30" -DCPU_THROUGHPUT="100" -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_roofline.cmake
)

add_definitions(-Wall -mavx)
add_definitions(-fopt-info-vec-optimized)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(assembly_dump ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel.cpp)
set_target_properties(assembly_dump PROPERTIES PREFIX "")
set_target_properties(assembly_dump PROPERTIES OUTPUT_NAME "assembly_dump.asm")
set_target_properties(assembly_dump PROPERTIES SUFFIX "")
set_target_properties(assembly_dump PROPERTIES COMPILE_FLAGS "-DGENERATE_ASSEMBLY -S -masm=intel -fopt-info-vec-optimized")

add_executable(test-openmp-gomp ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel.cpp "${CMAKE_CURRENT_SOURCE_DIR}/xbyak/xbyak/xbyak.h") 
target_link_libraries(test-openmp-gomp ${MKL_LIBRARIES} m ${GFLAGS_LIB})

add_test( b1a ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp )
add_test( b1_c50 ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp --batch_size 1 --channel_size 50 )
add_test( b2_c1000 ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp --batch_size 2 --channel_size 1000 --num_reps 1)
add_test( b2_c1007 ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp --batch_size 2 --channel_size 1007 --num_reps 2)
add_test( b4_c50 ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp --batch_size 4 --channel_size 50 )
add_test( b300_c50 ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp --batch_size 300 --channel_size 50 --num_reps 2)
add_test( b300_c500 ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp --batch_size 300 --channel_size 500 --num_reps 1)
