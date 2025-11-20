cmake_minimum_required (VERSION 3.5)
project (pc_rwalker_py)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)

set(GC_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../thirdparty/geometry-central")
if(NOT EXISTS "${GC_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "geometry-central submodule not found at ${GC_SOURCE_DIR}")
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(${GC_SOURCE_DIR} ${CMAKE_BINARY_DIR}/geometry-central)

pybind11_add_module(_pc_rwalker pc_rwalker/pc_rwalker.cpp)
set_target_properties(_pc_rwalker PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/pc_rwalker/
)

target_link_libraries (_pc_rwalker PRIVATE
    pybind11::module
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    geometry-central
)

target_include_directories(_pc_rwalker PRIVATE
    ${CMAKE_SOURCE_DIR}/../include
    ${GC_SOURCE_DIR}/deps/nanoflann/include
)
