cmake_minimum_required(VERSION 3.27)
project(WLS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(eigen3 INTERFACE)
target_include_directories(eigen3 INTERFACE ${CMAKE_SOURCE_DIR}/third_party/eigen)


find_package(Boost REQUIRED)

if(Boost_FOUND)
    message(STATUS "Boost found: ${Boost_INCLUDE_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
else()
    message(FATAL_ERROR "Boost not found. Please install Boost or provide its path.")
endif()


add_executable(${PROJECT_NAME} main.cpp utils/readCSV.h utils/WLS.h)

target_link_libraries(${PROJECT_NAME} PRIVATE eigen3)

set(DATA_DIR ${CMAKE_SOURCE_DIR}/data)
add_definitions(-DDATA_DIR="${DATA_DIR}")