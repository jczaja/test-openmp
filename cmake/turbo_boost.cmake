# Enableing/Disabling turbo boost using intel pstate (Require root priviliges)
# check if pstate is present
find_file(NO_TURBO NAMES no_turbo PATHS /sys/devices/system/cpu/intel_pstate/)
if (NO_TURBO) 
  file(READ ${NO_TURBO} NO_TURBO_VAL)
   message(STATUS "Given no_turbo val ${VAL}")
  if (${NO_TURBO_VAL} EQUAL ${VAL})
    message(STATUS "NO TURBO BOOST already set to ${VAL}")
  else()
    message(STATUS "Switching TURBO BOOST...")
    file(WRITE ${NO_TURBO} ${VAL})
    # Read written value of no_turbo
    file(READ ${NO_TURBO} NO_TURBO_VAL)
    message(STATUS "NO TURBO BOOST set to ${NO_TURBO_VAL}")
  endif()
else()
  message(STATUS "Cannot find NO_TURBO file. Please adjust turbo boost manually")
endif()
