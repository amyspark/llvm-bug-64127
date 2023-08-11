project(shapes)

cmake_minimum_required(VERSION 3.5)

add_library(io STATIC io.c)

target_include_directories(io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

add_executable(main main.cpp)

target_link_libraries(main PUBLIC io)
