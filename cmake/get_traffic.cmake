# This module executed test-memory-traffic with given defines and produce 
# memory traffic value in traffic.txt
function(floatexpr expr output)
string(REGEX REPLACE "\n" "" ${expr} "${expr}")
#message(STATUS "Expr: ${expr}")
execute_process(COMMAND awk "BEGIN {print ${expr}}" OUTPUT_VARIABLE __output)
set(${output} ${__output} PARENT_SCOPE)
endfunction()

macro(get_data_traffic_count DATA_PATTERN)
set(awk_script "{\$1=\$1\;print}")
execute_process(
    COMMAND echo ${ANALYSIS_RESULT}
    COMMAND grep -e ${DATA_PATTERN}
    COMMAND awk ${awk_script} # Remove trailing white characters
    COMMAND cut -d " " -f 1     # Get value of counter given
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_VARIABLE MIB_RAW
)
string(STRIP ${MIB_RAW} MIB)
endmacro()


function(count_traffic num_reps data_reads data_writes)
execute_process(COMMAND sudo perf stat -e data_reads,data_writes ${CMAKE_BINARY_DIR}/test-memory-traffic --num_reps ${num_reps} --algo=${ALGO} --batch_size=${N} --channel_size=${C} --height=${H} --width=${W} 
ERROR_VARIABLE ANALYSIS_RESULT)
#message(STATUS "ANALYSIS(${num_reps}): ${ANALYSIS_RESULT}")
set(reads "0")
set(writes "0")
get_data_traffic_count("data_reads")
set(${data_reads} ${MIB} PARENT_SCOPE)
get_data_traffic_count("data_writes")
set(${data_writes} ${MIB} PARENT_SCOPE)
endfunction()

# Execute program with single execution of evaluated algorithm
set(data_reads "0")
set(data_writes "0")
set(num_reps "0")
count_traffic(${num_reps} data_reads data_writes)
set(no_execution_reads ${data_reads})
set(no_execution_writes ${data_writes})
# Execute program without execution of evaluated algorithm 
set(num_reps "1")
count_traffic(${num_reps} data_reads data_writes)
message(STATUS "execution_reads: ${data_reads}")
message(STATUS "execution_writes: ${data_writes}")
message(STATUS "no_execution_reads: ${no_execution_reads}")
message(STATUS "no_execution_writes: ${no_execution_writes}")
# Substract baseline memory usage from total and conver MiB to bytes
set(traffic_equation "1024.0*1024.0*${data_reads}-${no_execution_reads}+${data_writes}-${no_execution_writes}")
string(STRIP ${traffic_equation} traffic_equation_stripped)
# Combine reads and writes to get total traffic and
# write it into traffic.txt 
floatexpr("${traffic_equation_stripped}" traffic)
message(STATUS "Estimated Memory Traffic: ${traffic}")
file(WRITE ${CMAKE_BINARY_DIR}/traffic.txt ${traffic})
