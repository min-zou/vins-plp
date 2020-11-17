project( line-descriptor )

cmake_minimum_required(VERSION 2.7)
find_package( OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)

if(COMMAND cmake_policy)
  	cmake_policy(SET CMP0003 NEW)	
endif(COMMAND cmake_policy)
link_directories(${OpenCV_LIBS_DIR})
include_directories(${OpenCV2_INCLUDE_DIRS})

# SET(BUILD_SHARED_LIBS ON)
# SET(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -O3 -mtune=native -march=native")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w   -O3 -march=native")
# set(DCMAKE_TOOLCHAIN_FILE /home/eco/tools/vcpkg/scripts/buildsystems/vcpkg.cmake)
# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

SET(BUILD_SHARED_LIBS ON)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(LIBRARY_OUTPUT_PATH    ${PROJECT_SOURCE_DIR}/lib)

include_directories( include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
list(APPEND LINK_LIBS ${OpenCV_LIBS} )
file(GLOB_RECURSE all_include_files RELATIVE "${PROJECT_SOURCE_DIR}" *.h *.hpp)

link_directories(${PROJECT_SOURCE_DIR}/src/)
file(GLOB_RECURSE all_source_files RELATIVE "${PROJECT_SOURCE_DIR}src/" *.cpp )

add_custom_target( linedesc_includes DEPENDS ${all_include_files} SOURCES ${all_source_files} )

add_library(linedesc ${all_source_files} )
target_link_libraries( linedesc ${LINK_LIBS} ${EIGEN3_LIBS} 
	# ${PROJECT_SOURCE_DIR}/../DBoW2/lib/libDBoW2.so
)

# add_executable( lsd src/main.cc )
# target_link_libraries(lsd linedesc )


