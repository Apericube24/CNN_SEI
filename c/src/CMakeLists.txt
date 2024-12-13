set(SOURCES
    main.cpp
    functions.cpp
)

add_executable(main ${SOURCES})

target_include_directories(main PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ~/Downloads/ac_types-master/include
    ~/Downloads/ac_math-master/include
)