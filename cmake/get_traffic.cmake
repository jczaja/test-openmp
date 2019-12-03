# This module executed test-memory-traffic with given defines and produce 
# memory traffic value in traffic.txt
function(floatexpr expr output)
execute_process(COMMAND awk "BEGIN {print ${expr}}" OUTPUT_VARIABLE __output)
set(${output} ${__output} PARENT_SCOPE)
endfunction()

macro(get_data_traffic_count DATA_PATTERN)
set(awk_script "{\$1=\$1\;print}")
execute_process(
    COMMAND echo ${ANALYSIS_RESULT}
    COMMAND grep -e ${DATA_PATTERN}
    COMMAND sed  "s:,::g" 
    COMMAND awk ${awk_script} # Remove trailing white characters
    COMMAND cut -d " " -f 1     # Get value of counter given
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_VARIABLE MIB
)
endmacro()
endmacro()


function(count_traffic num_reps data_reads data_wrotes)
execute_process(COMMAND sudo perf stat -e data_reads,data_writes ${CMAKE_BINARY_DIR}/test-memory-traffic --num_reps ${num_reps} --algo=${ALGO} --batch_size=${N} --channel_size=${C} --height=${H} --width=${W} 
ERROR_VARIABLE ANALYSIS_RESULT)

endfunction()


