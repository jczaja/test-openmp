function(floatexpr expr output)
execute_process(COMMAND awk "BEGIN {print ${expr}}" OUTPUT_VARIABLE __output)
set(${output} ${__output} PARENT_SCOPE)
endfunction()


function(execute_runtime num_reps total_time_c total_time_s)
execute_process(COMMAND ${CMAKE_BINARY_DIR}/test-openmp-gomp --num_reps ${num_reps} --algo=${ALGO} --batch_size=${N} --channel_size=${C} --height=${H} --width=${W} 
COMMAND grep -e "Runtime"
COMMAND cut -d " " -f 2,4
OUTPUT_VARIABLE __times)
separate_arguments(__times)
list(GET __times 0 times_c)
list(GET __times 1 times_s)
set(${total_time_c} ${times_c} PARENT_SCOPE)
set(${total_time_s} ${times_s} PARENT_SCOPE)
endfunction()

set(total_time_c "0")
set(total_time_s "0")
set(num_reps "1")
while(total_time_s LESS "60" AND num_reps LESS "1000000")
execute_runtime(${num_reps} total_time_c total_time_s)
math(EXPR num_reps "${num_reps}*2")
endwhile()

# num_reps is twice as big as used for final execution
math(EXPR num_reps "${num_reps}/2")
message(STATUS "Total(num_reps=${num_reps}) execution time[cycles]: ${total_time_c}")
string(REGEX REPLACE "\n$" "" total_time_stripped "${total_time_s}")

floatexpr("${total_time_stripped}/${num_reps}" average_time)

message(STATUS "Average execution time[cycles]: ${average_time}")
file(WRITE ${CMAKE_BINARY_DIR}/runtime.txt ${average_time})
