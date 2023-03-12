# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles/LidarDetectionInference_autogen.dir/AutogenUsed.txt"
  "CMakeFiles/LidarDetectionInference_autogen.dir/ParseCache.txt"
  "LidarDetectionInference_autogen"
  )
endif()
