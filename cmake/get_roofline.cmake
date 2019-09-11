macro(create_gnuplot_script CPU_THRGHPT MEMORY_THRGHPT)
set(script "set terminal png size 1920, 1080")
set(script "${script}\n set output \"roofline.png\"")
set(script "${script}\n set xlabel \"Operational Intensity [FLOPS/Byte]\"")
set(script "${script}\n set ylabel \"Atteinable GFLOPS/s\"")
set(script "${script}\n LINE_ROOF=1")
set(script "${script}\n set style line LINE_ROOF lt 1 lw 6 lc rgb \"blue\"")
set(script "${script}\n set xrange[0:10]")
set(script "${script}\n set yrange[0:${CPU_THRGHPT}*1.1]")
set(script "${script}\n min(x,y) = x < y ? x : y")
set(script "${script}\n mem_roof(x) = x *${MEMORY_THRGHPT}")    # Memory roofline
set(script "${script}\n cpu_roof = ${CPU_THROUGHPUT}")          # cpu_roofline
set(script "${script}\n roofline(x) = min(mem_roof(x),cpu_roof)")
set(script "${script}\n plot roofline(x) ls LINE_ROOF")
file(WRITE "${CMAKE_BINARY_DIR}/roofline.plot" "${script}")
endmacro()

find_package(Gnuplot REQUIRED)

create_gnuplot_script("${CPU_THROUGHPUT}" "${MEMORY_THROUGHPUT}")
# Execute gnuplot using generated script
execute_process(
    COMMAND ${GNUPLOT_EXECUTABLE} ${CMAKE_BINARY_DIR}/roofline.plot
    ERROR_VARIABLE ERROR_VAR 
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
