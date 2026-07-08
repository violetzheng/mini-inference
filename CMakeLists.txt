cmake_minimum_required(VERSION 3.16)
project(mini_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mini_inference_core
    src/tensor/matrix.cpp
    src/tensor/tensor.cpp
)

target_include_directories(mini_inference_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

enable_testing()

add_executable(test_matrix
    tests/test_matrix.cpp
)

target_link_libraries(test_matrix PRIVATE mini_inference_core)
target_include_directories(test_matrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(test_tensor
    tests/test_tensor.cpp
)

target_link_libraries(test_tensor PRIVATE mini_inference_core)
target_include_directories(test_tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_test(NAME matrix COMMAND test_matrix)
add_test(NAME tensor COMMAND test_tensor)
