cmake_minimum_required(VERSION 2.8)

project(kalray_project C)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")

include_directories(${PROJECT_SOURCE_DIR}/src/include)

add_subdirectory(src)