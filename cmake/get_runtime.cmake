function(floatexpr expr output)
execute_process(COMMAND awk "BEGIN {print ${expr}}" OUTPUT_VARIABLE __output)
set(${output} ${__output} PARENT_SCOPE)
endfunction()


function(execute_runtime num_reps total_time)
execute_process(COMMAND ${CMAKE_BINARY_DIR}/test-openmp-gomp --num_reps ${num_reps} 
COMMAND grep -e "RUNTIME"
COMMAND cut -d " " -f 2
OUTPUT_VARIABLE __total_time)
set(${total_time} ${__total_time} PARENT_SCOPE)
endfunction()

set(total_time "0")
set(num_reps "1")
while(total_time LESS "60" AND num_reps LESS "1000000")
execute_runtime(${num_reps} total_time)
math(EXPR num_reps "${num_reps}*2")
endwhile()

# num_reps is twice as big as used for final execution
math(EXPR num_reps "${num_reps}/2")
message(STATUS "Total(num_reps=${num_reps}) execution time: ${total_time}")
string(REGEX REPLACE "\n$" "" total_time_stripped "${total_time}")

floatexpr("${total_time_stripped}/${num_reps}" average_time)

message(STATUS "Average execution time: ${average_time}")
file(WRITE ${CMAKE_BINARY_DIR}/runtime.txt ${average_time})
