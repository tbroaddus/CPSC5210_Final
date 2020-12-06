cmake_minimum_required(VERSION 3.10)
project(Final)
find_package(MPI REQUIRED)

include_directories($(MPI_INCLUDE_PATH))

add_executable(outer_product outer_product.c)
target_link_libraries(outer_product ${MPI_LIBRARIES})
target_link_libraries(outer_product m)
