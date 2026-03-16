cmake_minimum_required(VERSION 3.14)
project(ml_train_plugin CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(ml_train STATIC
    src/xtce_parser.cpp
    src/model_io.cpp
)
target_include_directories(ml_train PUBLIC include)

add_executable(test_ml_train tests/test_ml_train.cpp)
target_link_libraries(test_ml_train ml_train)

enable_testing()
add_test(NAME ml_train COMMAND test_ml_train)

add_executable(test_sixdof tests/test_sixdof.cpp)
target_include_directories(test_sixdof PRIVATE include)
add_test(NAME ml_train_sixdof COMMAND test_sixdof)
