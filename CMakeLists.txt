cmake_minimum_required(VERSION 2.8)
project(test-openmp)

include(CheckCXXCompilerFlag)

find_package(OpenMP)
if(OPENMP_FOUND)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -v")
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

# ---[ Root folders
set(INTEL_ROOT "/opt/intel" CACHE PATH "Folder contains intel libs")
find_path(MKL_ROOT include/mkl.h PATHS $ENV{MKLROOT} $ENV{MKL_ROOT} ${INTEL_ROOT}/mkl
                                   DOC "Folder contains MKL")

# ---[ Find include dir
find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKL_ROOT} PATH_SUFFIXES include)

find_library(MKL_ILP_LIBRARY
        NAMES mkl_intel_ilp64
        PATHS ${MKL_ROOT}/lib/intel64 )
list(APPEND MKL_LIBRARIES ${MKL_ILP_LIBRARY})
find_library(MKL_CORE_LIBRARY
        NAMES libmkl_core mkl_core
        PATHS ${MKL_ROOT}/lib/intel64 )
list(APPEND MKL_LIBRARIES ${MKL_CORE_LIBRARY})

find_library(MKL_INTEL_THREAD_LIBRARY
        NAMES libmkl_gnu_thread mkl_gnu_thread
        PATHS ${MKL_ROOT}/lib/intel64 )
list(APPEND MKL_LIBRARIES ${MKL_INTEL_THREAD_LIBRARY})




if(MKL_ROOT)
  message(STATUS "Found MKL (include: ${MKL_INCLUDE_DIR}, lib: ${MKL_LIBRARIES}")
  include_directories(${MKL_INCLUDE_DIR})
else()
  message(FATAL "Intel MKL not located properly")
endif()

add_definitions(-Wall -mavx)
#add_definitions(-fopt-info-vec-missed)
add_library(assembly_dump main.cpp)
set_target_properties(assembly_dump PROPERTIES PREFIX "")
set_target_properties(assembly_dump PROPERTIES OUTPUT_NAME "assembly_dump.asm")
set_target_properties(assembly_dump PROPERTIES SUFFIX "")
set_target_properties(assembly_dump PROPERTIES COMPILE_FLAGS "-DGENERATE_ASSEMBLY -S -masm=intel -fopt-info-vec-optimized")


add_executable(test-openmp-gomp main.cpp) 
target_link_libraries(test-openmp-gomp ${MKL_LIBRARIES} m )

#add_executable(test-openmp main.cpp trace.cpp)
#add_executable(test-openmp-iomp5 main.cpp) 
#target_link_libraries(test-openmp-iomp5 -Wl,--as-needed iomp5 -L/home/jczaja/test-openmp)
#target_link_libraries(test-openmp-iomp5 -Wl,--as-needed iomp5 )
#set_target_properties(test-openmp-iomp5 PROPERTIES LINK_FLAGS -Wl,--as-needed)




