# This module executed test-memory-traffic with given defines and produce 
# memory traffic value in traffic.txt
function(floatexpr expr output)
string(REGEX REPLACE "\n" "" ${expr} "${expr}")
#message(STATUS "Expr: ${expr}")
execute_process(COMMAND awk "BEGIN {print ${expr}}" OUTPUT_VARIABLE __output)
set(${output} ${__output} PARENT_SCOPE)
endfunction()

macro(get_data_traffic_count DATA_PATTERN1 DATA_PATTERN2)
set(awk_script "{\$1=\$1\;print}")
set(sed_script "s:,::g")
#message(STATUS "Analysis Result: ${REGULAR_OUTPUT}")
execute_process(
    COMMAND echo ${REGULAR_OUTPUT}
    COMMAND grep -e ${DATA_PATTERN1}
    COMMAND grep -e ${DATA_PATTERN2}
    COMMAND awk ${awk_script} # Remove trailing white characters
    COMMAND sed ${sed_script} # Remove commands from numeric data
    COMMAND cut -d " " -f 2     # Get value of counter given
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_VARIABLE MIB_RAW
)
#message(STATUS "MIB: ${MIB_RAW}")
string(STRIP ${MIB_RAW} MIB)
endmacro()


macro(get_cpu_info)
execute_process(
    COMMAND cat /proc/cpuinfo
    COMMAND grep -m 1 -e "model name"
    COMMAND cut -d ":" -f 2     # Get value of counter given
    OUTPUT_VARIABLE CPU_MODEL
)

if(THREADING STREQUAL "single")
set(THREADING_STR "(single core)")
elseif (THREADING STREQUAL "socket")
set(THREADING_STR "(single socket)")
else()
set(THREADING_STR "(two sockets)")
endif()

file(WRITE ${CMAKE_BINARY_DIR}/cpu_info.txt "${CPU_MODEL} ${THREADING_STR}")
endmacro()

macro(get_algorithm_info)
execute_process(
    COMMAND echo ${REGULAR_OUTPUT}
    COMMAND grep -e "x"
    OUTPUT_VARIABLE ALGO_INFO
)
file(WRITE ${CMAKE_BINARY_DIR}/algo_info.txt ${ALGO_INFO})
endmacro()

function(count_traffic num_reps data_reads data_writes)
execute_process(COMMAND sudo ${CMAKE_BINARY_DIR}/test-memory-traffic --num_reps ${num_reps} --algo=${ALGO} --batch_size=${N} --cold_caches=${COLD_CACHES} --channel_size=${C} --height=${H} --width=${W} --threading=${THREADING}
OUTPUT_VARIABLE REGULAR_OUTPUT
ERROR_VARIABLE ANALYSIS_RESULT)
get_algorithm_info()
#message(STATUS "ANALYSIS(${num_reps}): ${ANALYSIS_RESULT}")
get_data_traffic_count("MiB" "data_reads")
set(${data_reads} ${MIB} PARENT_SCOPE)
get_data_traffic_count("MiB" "data_writes")
set(${data_writes} ${MIB} PARENT_SCOPE)
endfunction()

get_cpu_info()
set(no_execution_reads "0")
set(no_execution_writes "0")
set(execution_reads "0")
set(execution_writes "0")
set(iter "1")
# Warm caches we run 10 repetitions
# and get a mean value as final one
if(COLD_CACHES STREQUAL "false")
set(num_reps "20")
else()
set(num_reps "20")
endif()
math(EXPR minimum_threshold "${N}*${C}*${H}*${W}*4/1024/1024")
message(STATUS "minimum threshold: ${minimum_threshold}")
while(iter LESS num_reps)
set(data_reads "0")
set(data_writes "0")
while(data_reads LESS minimum_threshold)
# Execute program with single execution of evaluated algorithm
count_traffic(1 data_reads data_writes)
message(STATUS "DR: ${data_reads}")
endwhile()
set(equation "(${execution_reads}*(${iter}-1)+${data_reads})/${iter}")
string(STRIP ${equation} equation_stripped)
string(REGEX REPLACE "\n" "" equation_stripped "${equation_stripped}")
#message("EQ_STRIPPED: ${equation_stripped}")
floatexpr("${equation_stripped}" execution_reads)
set(equation "(${execution_writes}*(${iter}-1)+${data_writes})/${iter}")
string(STRIP ${equation} equation_stripped)
string(REGEX REPLACE "\n" "" equation_stripped "${equation_stripped}")
#message("EQ_STRIPPED: ${equation_stripped}")
floatexpr("${equation_stripped}" execution_writes)
math(EXPR iter "${iter}+1")
endwhile()

message(STATUS "execution_reads: ${execution_reads}")
message(STATUS "execution_writes: ${execution_writes}")
# Substract baseline memory usage from total and conver MiB to bytes
set(traffic_equation "1024.0*1024.0*(${execution_reads}+${execution_writes})")
string(STRIP ${traffic_equation} traffic_equation_stripped)
string(REGEX REPLACE "\n" "" traffic_equation_stripped "${traffic_equation_stripped}")
# Combine reads and writes to get total traffic and
# write it into traffic.txt 
floatexpr("${traffic_equation_stripped}" traffic)
message(STATUS "Estimated Memory Traffic: ${traffic}")
file(WRITE ${CMAKE_BINARY_DIR}/traffic.txt ${traffic})
