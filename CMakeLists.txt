cmake_minimum_required(VERSION 3.16)
project(mini_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mini_inference_core
    src/tensor/matrix.cpp
    src/tensor/tensor.cpp
    src/layers/linear.cpp
    src/layers/rms_norm.cpp
    src/layers/softmax.cpp
    src/layers/rope.cpp
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

add_executable(test_linear
    tests/test_linear.cpp
)

target_link_libraries(test_linear PRIVATE mini_inference_core)
target_include_directories(test_linear PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(test_rms_norm
    tests/test_rms_norm.cpp
)

target_link_libraries(test_rms_norm PRIVATE mini_inference_core)
target_include_directories(test_rms_norm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(test_softmax
    tests/test_softmax.cpp
)

target_link_libraries(test_softmax PRIVATE mini_inference_core)
target_include_directories(test_softmax PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(test_rope
    tests/test_rope.cpp
)

target_link_libraries(test_rope PRIVATE mini_inference_core)
target_include_directories(test_rope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_test(NAME matrix COMMAND test_matrix)
add_test(NAME tensor COMMAND test_tensor)
add_test(NAME linear COMMAND test_linear)
add_test(NAME rms_norm COMMAND test_rms_norm)
add_test(NAME softmax COMMAND test_softmax)
add_test(NAME rope COMMAND test_rope)
