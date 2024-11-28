set(SOURCES
    main.c
    functions.c
    parse.c
)

add_executable(main ${SOURCES})

target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)