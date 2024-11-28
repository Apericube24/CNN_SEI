cmake_minimum_required(VERSION 2.8)

project(3A-SEI-SoC-POO-ProjetFinal)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wconversion -std=gnu99")

include_directories(${PROJECT_SOURCE_DIR}/src/include)

add_subdirectory(src)