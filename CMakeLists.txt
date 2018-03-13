cmake_minimum_required(VERSION 2.8)

project(NN CXX)
set(CMAKE_CXX_FLAGS "-Werror -Wall -Wextra -std=c++17 -pedantic -g")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_COMPILER "g++")

set(SRC
   ${CMAKE_SOURCE_DIR}/src/tools.cc
   ${CMAKE_SOURCE_DIR}/src/main.cc
   ${CMAKE_SOURCE_DIR}/src/neural_net.cc
)

add_executable(nn ${SRC})
add_custom_target(check
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/
                  COMMAND ./testsuite.rb
)

add_custom_target(doc
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                  COMMAND doxygen doc/doc_config
                  COMMAND firefox html/index.html
)
