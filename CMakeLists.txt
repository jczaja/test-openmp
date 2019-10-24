cmake_minimum_required(VERSION 3.14)
project(test-openmp)

include(CheckCXXCompilerFlag)
include(ExternalProject)
include(FetchContent)

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

find_package(Git)

if(UNIX)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
add_definitions(-std=c++11)
elseif(COMPILER_SUPPORTS_CXX0X)
add_definitions(-std=c++0x)
endif()
endif()

FetchContent_Declare(
dnnl
GIT_REPOSITORY https://github.com/intel/mkl-dnn
GIT_TAG v1.1.1
GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dnnl)

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
    GIT_SHALLOW         TRUE
)

add_custom_target(
work
COMMAND ${CMAKE_COMMAND} -E echo N=\${N} C=\${C} H=\${H} W=\${W} 
COMMAND ${CMAKE_COMMAND} -DN=\${N} -DC=\${C} -DH=\${H} -DW=\${W} -DPERFMON=${PERFMON} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_codes.cmake | grep -e FLOPS | cut -d ":" -f 2 | tee work.txt
WORKING_DIRECTORY ${CMAKE_CURRENT_BIN_DIR}
DEPENDS ${PERFMON} test-openmp-gomp
)

endif(PERF)


add_definitions(-Wall -mavx)
add_definitions(-fopt-info-vec-optimized)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(assembly_dump ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel.cpp
${CMAKE_CURRENT_SOURCE_DIR}/kernels/dnnl_fp32_conv_kernel.cpp)

set_target_properties(assembly_dump PROPERTIES PREFIX "")
set_target_properties(assembly_dump PROPERTIES OUTPUT_NAME "assembly_dump.asm")
set_target_properties(assembly_dump PROPERTIES SUFFIX "")
set_target_properties(assembly_dump PROPERTIES COMPILE_FLAGS "-DGENERATE_ASSEMBLY -S -masm=intel -fopt-info-vec-optimized")

add_executable(test-openmp-gomp ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel.cpp
"${CMAKE_CURRENT_SOURCE_DIR}/kernels/dnnl_fp32_conv_kernel.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/xbyak/xbyak/xbyak.h") 
set_target_properties(test-openmp-gomp PROPERTIES COMPILE_FLAGS "-DRUNTIME_TEST")
target_link_libraries(test-openmp-gomp m ${GFLAGS_LIB} dnnl)

# Memory traffic counting build
add_executable(test-memory-traffic ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel.cpp 
"${CMAKE_CURRENT_SOURCE_DIR}/xbyak/xbyak/xbyak.h") 
set_target_properties(test-memory-traffic PROPERTIES COMPILE_FLAGS "-DMEMORY_TRAFFIC_COUNT")
target_link_libraries(test-memory-traffic m ${GFLAGS_LIB})


# to compute operational intensity we need to count memory
# traffic and work(FLOPS needed for algorithm execution)
add_custom_target(
traffic 
COMMENT "Measuring memory traffic of kernel (LLC to DRAM)"
COMMAND ${CMAKE_BINARY_DIR}/test-memory-traffic --num_reps 1 -batch_size=\${N} --channel_size=\${C} --height=\${H} --width=\${W} | grep -ie "MemoryTraffic" | cut -d ":" -f 2 | tee traffic.txt
DEPENDS test-memory-traffic 
)

# roofline needs memtest and cputest
# It is also using gnuplot for generation of script
# it does need memory throughput [GB/sec]
# and cpu throughput [GFLOPS/sec]
add_custom_target(
memtest
COMMENT "Performing peak memory test:"
COMMAND ${CMAKE_BINARY_DIR}/test-openmp-gomp --memtest | grep -ie "Tempo" | cut -d " " -f 6 | tee memtest.txt
DEPENDS test-openmp-gomp
VERBATIM
)

add_custom_target(
cputest
COMMENT "Performing peak cpu test:"
COMMAND ${CMAKE_BINARY_DIR}/test-openmp-gomp --cputest | grep -e "Benchmarked" | cut -d " " -f 4 | tee cputest.txt
DEPENDS test-openmp-gomp 
VERBATIM
)

# Optional target trying disable turbo boost 
# for intel pstate driver
add_custom_target(
disable_turbo_boost
COMMENT "Disabling TURBO BOOST using intel pstate"
COMMAND cmake -DVAL="1" -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/turbo_boost.cmake 
)

# Optional target trying disable turbo boost 
# for intel pstate driver
add_custom_target(
enable_turbo_boost
COMMENT "Enabling TURBO BOOST using intel pstate"
COMMAND cmake -DVAL="0" -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/turbo_boost.cmake )

add_custom_target(
runtime
COMMENT "Performing runtime test (Make sure TURBO BOOST is disabled!)"
COMMAND cmake -DN=\${N} -DC=\${C} -DH=\${H} -DW=\${W} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_runtime.cmake
DEPENDS test-openmp-gomp 
)

add_custom_target(
roofline
COMMAND cmake -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_roofline.cmake
DEPENDS memtest cputest traffic work runtime
)

# Unit tests
add_test( b1a ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp )
add_test( b1_c50 ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp --batch_size 1 --channel_size 50 )
add_test( b2_c1000 ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp --batch_size 2 --channel_size 1000 --num_reps 1)
add_test( b2_c1007 ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp --batch_size 2 --channel_size 1007 --num_reps 2)
add_test( b4_c50 ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp --batch_size 4 --channel_size 50 )
add_test( b300_c50 ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp --batch_size 300 --channel_size 50 --num_reps 2)
add_test( b300_c500 ${CMAKE_CURRENT_BIN_DIR}/test-openmp-gomp --batch_size 300 --channel_size 500 --num_reps 1)
