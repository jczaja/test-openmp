macro(create_gnuplot_script CPU_THRGHPT MEMORY_THRGHPT OI EXECUTION_TIME RUNTIME_PERFORMANCE HW_INFO ALGO_INFO)
set(script "set terminal pngcairo dashed size 1920, 1080")
set(script "${script}\n set output \"roofline-${ALGO_INFO}.png\"")
set(script "${script}\n set xlabel \"Operational Intensity [FLOPS/Byte]\"")
set(script "${script}\n set ylabel \"Atteinable GFLOPS/s\"")
set(script "${script}\n set title \"${HW_INFO}\"")
set(script "${script}\n LINEROOF=1")
set(script "${script}\n set style line LINEROOF lt 1 lw 6 lc rgb \"black\"")
set(script "${script}\n set logscale xy")
set(script "${script}\n set grid")
set(script "${script}\n max(x,y) = x > y ? x : y")
set(script "${script}\n rigidpoint = ${CPU_THROUGHPUT}/${MEMORY_THRGHPT}")    # Memory roofline
set(script "${script}\n set xrange[0.1:max(rigidpoint,${OI})*10.0]")
set(script "${script}\n set yrange[0.1:${CPU_THRGHPT}*10.0]")
set(script "${script}\n min(x,y) = x < y ? x : y")
set(script "${script}\n memroof(x) = x *${MEMORY_THRGHPT}")    # Memory roofline
set(script "${script}\n cpuroof = ${CPU_THROUGHPUT}")          # cpu_roofline
set(script "${script}\n roofline(x) = min(memroof(x),cpuroof)")
set(script "${script}\n set arrow from ${OI},0.1 to ${OI},roofline(${OI}) nohead dt 2")
set(script "${script}\n set object 3 circle at ${OI},${RUNTIME_PERFORMANCE} size scr 0.004 fc  rgb \"black\" fs solid")
set(script "${script}\n set angles degrees")

if(CHARTS STREQUAL "absolute")
set(script "${script}\n set label \"compute bound (${CPU_THROUGHPUT} GFLOPS)\" at rigidpoint,cpuroof * 1.2 textcolor \"black\"")
set(script "${script}\n set label \"Throughput: ${RUNTIME_PERFORMANCE} GFLOPS\\nExecution time: ${EXECUTION_TIME} ms\" at scr 0.4, 0.35 textcolor \"black\"") 
set(script "${script}\n set arrow from scr 0.5,0.4 to ${OI},${RUNTIME_PERFORMANCE} lw 0.6")
set(script "${script}\n MAXGFLOPS = sprintf(\"%f GFLOPS\", roofline(${OI}))")
else()
set(script "${script}\n set label \"compute bound (Peak Runtime Compute: 100\\%)\" at rigidpoint,cpuroof * 1.2 textcolor \"black\"")
set(script "${script}\n RC = sprintf(\"RC: \\%.2f\\%\", (${RUNTIME_PERFORMANCE}/cpuroof*100))")
set(script "${script}\n set label RC at first 0.52*${OI}, 0.9*${RUNTIME_PERFORMANCE} textcolor \"black\"")
set(script "${script}\n MAXGFLOPS = sprintf(\"Attainable RC: \\%.2f \\%\", roofline(${OI})/cpuroof*100)")
endif()
set(script "${script}\n set label MAXGFLOPS at scr 0.1, first roofline(${OI}) * 1.10")
set(script "${script}\n set label \"${ALGO_INFO}\" at first 0.95*${OI}, scr 0.08 textcolor \"black\" rotate by 90")

set(script "${script}\n set arrow from first 0.1, first roofline(${OI}) to first ${OI}, first roofline(${OI}) nohead dt 3")
set(script "${script}\n ")
set(script "${script}\n plot roofline(x) ls LINEROOF")
file(WRITE "${CMAKE_BINARY_DIR}/roofline.plot" "${script}")
endmacro()

function(floatexpr expr output)
execute_process(COMMAND awk "BEGIN {print ${expr}}" OUTPUT_VARIABLE __output)
set(${output} ${__output} PARENT_SCOPE)
endfunction()

find_package(Gnuplot REQUIRED)

file(STRINGS ${CMAKE_BINARY_DIR}/work.txt WORK)
file(STRINGS ${CMAKE_BINARY_DIR}/traffic.txt MEMORY_TRAFFIC)
file(STRINGS ${CMAKE_BINARY_DIR}/memtest-1.txt MEMORY_THROUGHPUT_1)
file(STRINGS ${CMAKE_BINARY_DIR}/memtest-2.txt MEMORY_THROUGHPUT_2)
file(STRINGS ${CMAKE_BINARY_DIR}/cputest.txt CPU_THROUGHPUT)
file(STRINGS ${CMAKE_BINARY_DIR}/runtime.txt EXECUTION_TIME)
file(STRINGS ${CMAKE_BINARY_DIR}/cpu_info.txt HW_INFO)
file(STRINGS ${CMAKE_BINARY_DIR}/algo_info.txt ALGO_INFO)

# Compute operational intensity
floatexpr("${WORK}/${MEMORY_TRAFFIC}" OI)
message(STATUS "Operational Intensity: ${OI}")

# Compute Actual Runtime performance
floatexpr("${WORK}/${EXECUTION_TIME}/1000000000.0" RUNTIME_PERFORMANCE)
message(STATUS "Runtime performance: ${RUNTIME_PERFORMANCE}")

# convert Execution time to mili seconds
floatexpr("${EXECUTION_TIME}*1000.0" EXECUTION_TIME_MS)
string(REGEX REPLACE "\n$" "" EXECUTION_TIME_MS_STRIPPED "${EXECUTION_TIME_MS}")

string(REGEX REPLACE "\n$" "" OI_STRIPPED "${OI}")
string(REGEX REPLACE "\n$" "" RUNTIME_PERFORMANCE_STRIPPED "${RUNTIME_PERFORMANCE}")

string(REGEX REPLACE "\n$" "" HW_INFO_STRIPPED "${HW_INFO}")

# Combine Memory bandwidth readings from two sockets
floatexpr("${MEMORY_THROUGHPUT_1}+${MEMORY_THROUGHPUT_2}" MEMORY_THROUGHPUT)

create_gnuplot_script("${CPU_THROUGHPUT}" "${MEMORY_THROUGHPUT}" "${OI_STRIPPED}" "${EXECUTION_TIME_MS_STRIPPED}" "${RUNTIME_PERFORMANCE_STRIPPED}" "${HW_INFO_STRIPPED}" "${ALGO_INFO}")
# Execute gnuplot using generated script
execute_process(
    COMMAND ${GNUPLOT_EXECUTABLE} ${CMAKE_BINARY_DIR}/roofline.plot
    ERROR_VARIABLE ERROR_VAR 
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
