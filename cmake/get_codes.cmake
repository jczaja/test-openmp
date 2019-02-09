execute_process(
COMMAND examples/showevtinfo
COMMAND grep ": FP_ARITH_INST_RETIRED" -A 10
WORKING_DIRECTORY ${PERFMON}
OUTPUT_VARIABLE SCAN
)

set(SCALAR_SINGLE_CODE "")
execute_process(
COMMAND echo ${SCAN}
COMMAND grep "SCALAR_SINGLE"
COMMAND cut -d : -f 2
COMMAND sed "s: ::g"
COMMAND xargs -I % examples/check_events FP_ARITH_INST_RETIRED:%
COMMAND tail -n 1
COMMAND cut -d : -f 2
COMMAND cut -d x -f 2
WORKING_DIRECTORY ${PERFMON}
OUTPUT_VARIABLE SCALAR_SINGLE_CODE
)

set(128B_PACKED_SINGLE_CODE "")
execute_process(
COMMAND echo ${SCAN}
COMMAND grep "128B_PACKED_SINGLE"
COMMAND cut -d : -f 2
COMMAND sed "s: ::g"
COMMAND xargs -I % examples/check_events FP_ARITH_INST_RETIRED:%
COMMAND tail -n 1
COMMAND cut -d : -f 2
COMMAND cut -d x -f 2
WORKING_DIRECTORY ${PERFMON}
OUTPUT_VARIABLE 128B_PACKED_SINGLE_CODE
)

set(256B_PACKED_SINGLE_CODE "")
execute_process(
COMMAND echo ${SCAN}
COMMAND grep "256B_PACKED_SINGLE"
COMMAND cut -d : -f 2
COMMAND sed "s: ::g"
COMMAND xargs -I % examples/check_events FP_ARITH_INST_RETIRED:%
COMMAND tail -n 1
COMMAND cut -d : -f 2
COMMAND cut -d x -f 2
WORKING_DIRECTORY ${PERFMON}
OUTPUT_VARIABLE 256B_PACKED_SINGLE_CODE
)

set(512B_PACKED_SINGLE_CODE "")
execute_process(
COMMAND echo ${SCAN}
COMMAND grep "512B_PACKED_SINGLE"
COMMAND cut -d : -f 2
COMMAND sed "s: ::g"
COMMAND xargs -I % examples/check_events FP_ARITH_INST_RETIRED:%
COMMAND tail -n 1
COMMAND cut -d : -f 2
COMMAND cut -d x -f 2
WORKING_DIRECTORY ${PERFMON}
OUTPUT_VARIABLE 512B_PACKED_SINGLE_CODE
)

set(CODES perf stat)

if(SCALAR_SINGLE_CODE)
set(CODES ${CODES} -e r${SCALAR_SINGLE_CODE})
message(STATUS "FP_ARITH_INST_RETIRED:SCALAR_SINGLE: ${SCALAR_SINGLE_CODE}")
endif()

if(128B_PACKED_SINGLE_CODE)
set(CODES ${CODES} -e r${128B_PACKED_SINGLE_CODE})
message(STATUS "FP_ARITH_INST_RETIRED:128B_PACKED_SINGLE: ${128B_PACKED_SINGLE_CODE}")
endif()

if(256B_PACKED_SINGLE_CODE)
set(CODES ${CODES} -e r${256B_PACKED_SINGLE_CODE})
message(STATUS "FP_ARITH_INST_RETIRED:256B_PACKED_SINGLE: ${256B_PACKED_SINGLE_CODE}")
endif()

if(512B_PACKED_SINGLE_CODE)
set(CODES ${CODES} -e r${512B_PACKED_SINGLE_CODE})
message(STATUS "FP_ARITH_INST_RETIRED:512B_PACKED_SINGLE: ${512B_PACKED_SINGLE_CODE}")
endif()

set(SEQ_COMMAND ${CODES} ${CMAKE_BINARY_DIR}/test-openmp-gomp --algo=${algo} --impl seq --num_reps $ENV{NUM_REPS})

string(REGEX REPLACE "\n" "" SEQ_COMMAND "${SEQ_COMMAND}")
message(STATUS "${CODES}")
message(STATUS "${EXAMPLE}")
execute_process(
COMMAND ${SEQ_COMMAND}
WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set(SIMD_COMMAND ${CODES} ${CMAKE_BINARY_DIR}/test-openmp-gomp --algo=${algo} --impl simd --num_reps $ENV{NUM_REPS})

string(REGEX REPLACE "\n" "" SIMD_COMMAND "${SIMD_COMMAND}")
message(STATUS "${CODES}")
message(STATUS "${EXAMPLE}")
execute_process(
COMMAND ${SIMD_COMMAND}
WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)


set(JIT_COMMAND ${CODES} ${CMAKE_BINARY_DIR}/test-openmp-gomp --algo=${algo} --impl jit --num_reps $ENV{NUM_REPS})

string(REGEX REPLACE "\n" "" JIT_COMMAND "${JIT_COMMAND}")
message(STATUS "${CODES}")
message(STATUS "${EXAMPLE}")
execute_process(
COMMAND ${JIT_COMMAND}
WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

