function(get_stats ret num_reps mapping)
set(EXPERIMENT_COMMAND ${CODES} ${CMAKE_BINARY_DIR}/test-openmp-gomp --algo=${algo} --num_reps ${num_reps})

list(LENGTH mapping len)
message(STATUS "len list: ${len}" )

message(STATUS "ARGV0:${ARGV0}")
message(STATUS "ARGV1:${num_reps}")
message(STATUS "ARGV2:${mapping}")

string(REGEX REPLACE "\n" "" ${EXPERIMENT_COMMAND} "${EXPERIMENT_COMMAND}")

# Get report of one execution 
execute_process(
COMMAND ${EXPERIMENT_COMMAND}
WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
OUTPUT_VARIABLE OUTPUT_RESULT
ERROR_VARIABLE ANALYSIS_RESULT
)

set(awk_script "{\$1=\$1\;print}")
execute_process(
    COMMAND echo ${ANALYSIS_RESULT}
#    COMMAND grep -e ${SCALAR_SINGLE_CODE} 
    COMMAND grep -e r5302c7  # TODO Events are given by arguments
    COMMAND awk ${awk_script} # Remove trailing white characters
    COMMAND cut -d " " -f 1     # Get value of counter given
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_VARIABLE FLOPS
)
set(${ret} ${FLOPS} PARENT_SCOPE)

endfunction()



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
list(APPEND EventMapping "${SCALAR_SINGLE_CODE} 1") 
endif()

if(128B_PACKED_SINGLE_CODE)
set(CODES ${CODES} -e r${128B_PACKED_SINGLE_CODE})
message(STATUS "FP_ARITH_INST_RETIRED:128B_PACKED_SINGLE: ${128B_PACKED_SINGLE_CODE}")
list(APPEND EventMapping "${128B_PACKED_SINGLE_CODE} 4")
endif()

if(256B_PACKED_SINGLE_CODE)
set(CODES ${CODES} -e r${256B_PACKED_SINGLE_CODE})
message(STATUS "FP_ARITH_INST_RETIRED:256B_PACKED_SINGLE: ${256B_PACKED_SINGLE_CODE}")
list(APPEND EventMapping "${256B_PACKED_SINGLE_CODE} 8")
endif()

if(512B_PACKED_SINGLE_CODE)
set(CODES ${CODES} -e r${512B_PACKED_SINGLE_CODE})
message(STATUS "FP_ARITH_INST_RETIRED:512B_PACKED_SINGLE: ${512B_PACKED_SINGLE_CODE}")
list(APPEND EventMapping "${512B_PACKED_SINGLE_CODE} 16")
endif()

list(LENGTH EventMapping len)
message(STATUS "len list: ${len}" )

set(FLOPS_1 "")
set(FLOPS_2 "")
get_stats(FLOPS_1 1 ${EventMapping})
get_stats(FLOPS_2 2 ${EventMapping})

# Compute diffrence among two iterations of kernel execution and one iteration of kernel
# execution. This diffrence will be number of FLOPS of single instance of kernel
# without FLOPS that comes from rest of program
math(EXPR FLOPS "${FLOPS_2} - ${FLOPS_1}")

message(STATUS "FLOPS: ${FLOPS}")
