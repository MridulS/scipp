# Download and unpack googlebenchmark at configure time

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

configure_file(${CMAKE_SOURCE_DIR}/CMake/GoogleBenchmark.in
               ${CMAKE_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googlebenchmark-download )

execute_process(COMMAND ${CMAKE_COMMAND} --build .
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googlebenchmark-download )

add_subdirectory(${CMAKE_BINARY_DIR}/googlebenchmark-src
                 ${CMAKE_BINARY_DIR}/googlebenchmark-build)
