cmake_minimum_required(VERSION 2.8.3)
project(unitree_legged_sdk)

include_directories(include)

# Use the full path to the lib directory here
link_directories(/home/user/unitree_legged_sdk/lib)

add_compile_options(-std=c++11)

# Specify libraries without 'lib' prefix and '.so' suffix.
set(EXTRA_LIBS pthread unitree_legged_sdk_amd64 lcm)

set(CMAKE_CXX_FLAGS "-O3")

add_subdirectory(pybind11)
pybind11_add_module(robot_interface python_interface.cpp)

# Link libraries
target_link_libraries(robot_interface ${EXTRA_LIBS})

