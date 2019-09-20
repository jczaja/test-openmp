macro(create_gnuplot_script CPU_THRGHPT MEMORY_THRGHPT OI)
set(script "set terminal pngcairo dashed size 1920, 1080")
set(script "${script}\n set output \"roofline.png\"")
set(script "${script}\n set xlabel \"Operational Intensity [FLOPS/Byte]\"")
set(script "${script}\n set ylabel \"Atteinable GFLOPS/s\"")
set(script "${script}\n LINE_ROOF=1")
set(script "${script}\n set style line LINE_ROOF lt 1 lw 6 lc rgb \"blue\"")
set(script "${script}\n set xrange[0:10]")
set(script "${script}\n set yrange[0:${CPU_THRGHPT}*1.1]")
set(script "${script}\n min(x,y) = x < y ? x : y")
set(script "${script}\n rigid_point = ${CPU_THROUGHPUT}/${MEMORY_THRGHPT}")    # Memory roofline
set(script "${script}\n mem_roof(x) = x *${MEMORY_THRGHPT}")    # Memory roofline
set(script "${script}\n cpu_roof = ${CPU_THROUGHPUT}")          # cpu_roofline
set(script "${script}\n set output \"roofline.png\"")
set(script "${script}\n roofline(x) = min(mem_roof(x),cpu_roof)")
set(script "${script}\n set arrow from ${OI},0 to ${OI},roofline(${OI}) nohead dt 2")
set(script "${script}\n set label \"compute bound\" at rigid_point,cpu_roof + 2 textcolor \"blue\"")
set(script "${script}\n set angles degrees")
set(script "${script}\n set label \"memory bound\" at rigid_point/2,mem_roof(rigid_point/2) + 2 textcolor \"blue\" rotate by atan(${MEMORY_THRGHPT}/(${CPU_THRGHPT}*1.1/10.0)/(1920.0/1080.0))") 
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

# Compute operational intensity
floatexpr("${WORK}/${MEMORY_TRAFFIC}" OI)
message(STATUS "Operational Intensity: ${OI}")

string(REGEX REPLACE "\n$" "" OI_STRIPPED "${OI}")

create_gnuplot_script("${CPU_THROUGHPUT}" "${MEMORY_THROUGHPUT}" "${OI_STRIPPED}")
# Execute gnuplot using generated script
execute_process(
    COMMAND ${GNUPLOT_EXECUTABLE} ${CMAKE_BINARY_DIR}/roofline.plot
    ERROR_VARIABLE ERROR_VAR 
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
