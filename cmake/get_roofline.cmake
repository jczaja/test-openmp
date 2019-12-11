macro(create_gnuplot_script CPU_THRGHPT MEMORY_THRGHPT OI RUNTIME_PERFORMANCE HW_INFO ALGO_INFO)
set(script "set terminal pngcairo dashed size 1920, 1080")
set(script "${script}\n set output \"roofline-${ALGO_INFO}.png\"")
set(script "${script}\n set xlabel \"Operational Intensity [FLOPS/Byte]\"")
set(script "${script}\n set ylabel \"Atteinable GFLOPS/s\"")
set(script "${script}\n set title \"${HW_INFO}\"")
set(script "${script}\n LINE_ROOF=1")
set(script "${script}\n set style line LINE_ROOF lt 1 lw 6 lc rgb \"black\"")
set(script "${script}\n set logscale xy")
set(script "${script}\n set grid")
set(script "${script}\n set xrange[0.0001:${OI}*10.0]")
set(script "${script}\n set yrange[0.0001:${CPU_THRGHPT}*10.0]")
set(script "${script}\n min(x,y) = x < y ? x : y")
set(script "${script}\n rigid_point = ${CPU_THROUGHPUT}/${MEMORY_THRGHPT}")    # Memory roofline
set(script "${script}\n mem_roof(x) = x *${MEMORY_THRGHPT}")    # Memory roofline
set(script "${script}\n cpu_roof = ${CPU_THROUGHPUT}")          # cpu_roofline
set(script "${script}\n roofline(x) = min(mem_roof(x),cpu_roof)")
set(script "${script}\n set arrow from ${OI},0.0001 to ${OI},roofline(${OI}) nohead dt 2")
set(script "${script}\n set object 3 circle at ${OI},${RUNTIME_PERFORMANCE} size scr 0.005 fc  rgb \"black\" fs solid")
set(script "${script}\n set label \"compute bound (${CPU_THROUGHPUT} GFLOPS)\" at rigid_point,cpu_roof * 1.2 textcolor \"black\"")
set(script "${script}\n set angles degrees")
set(script "${script}\n set label \"Runtime performance: ${RUNTIME_PERFORMANCE} GFLOPS out of \" at scr 0.4, 0.35 textcolor \"black\"") 
set(script "${script}\n set label \"${ALGO_INFO}\" at first 0.95*${OI}, scr 0.25 textcolor \"black\" rotate by 90") 
set(script "${script}\n set arrow from scr 0.5,0.5 to ${OI},${RUNTIME_PERFORMANCE} lw 0.6")
set(script "${script}\n ")

set(script "${script}\n plot roofline(x) ls LINE_ROOF")
file(WRITE "${CMAKE_BINARY_DIR}/roofline.plot" "${script}")
endmacro()

function(floatexpr expr output)
execute_process(COMMAND awk "BEGIN {print ${expr}}" OUTPUT_VARIABLE __output)
set(${output} ${__output} PARENT_SCOPE)
endfunction()

find_package(Gnuplot REQUIRED)

file(STRINGS ${CMAKE_BINARY_DIR}/work.txt WORK)
file(STRINGS ${CMAKE_BINARY_DIR}/traffic.txt MEMORY_TRAFFIC)
file(STRINGS ${CMAKE_BINARY_DIR}/memtest.txt MEMORY_THROUGHPUT)
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

string(REGEX REPLACE "\n$" "" OI_STRIPPED "${OI}")
string(REGEX REPLACE "\n$" "" RUNTIME_PERFORMANCE_STRIPPED "${RUNTIME_PERFORMANCE}")

create_gnuplot_script("${CPU_THROUGHPUT}" "${MEMORY_THROUGHPUT}" "${OI_STRIPPED}" "${RUNTIME_PERFORMANCE_STRIPPED}" "${HW_INFO}" "${ALGO_INFO}")
# Execute gnuplot using generated script
execute_process(
    COMMAND ${GNUPLOT_EXECUTABLE} ${CMAKE_BINARY_DIR}/roofline.plot
    ERROR_VARIABLE ERROR_VAR 
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
