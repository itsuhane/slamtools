cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
project(slamtools VERSION 0.1.0 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED COMPONENTS LAPACK SuiteSparse EigenSparse SchurSpecializations)

set(SLAMTOOLS_LIB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slamtools/homography.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slamtools/essential.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slamtools/stereo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slamtools/lie_algebra.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slamtools/frame.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slamtools/initializer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slamtools/track.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slamtools/sliding_window.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slamtools/factor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slamtools/preintegrator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slamtools/pnp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slamtools/bundle_adjustor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slamtools/configurator.cpp
)

add_library(slamtools
    ${SLAMTOOLS_LIB_SOURCES}
)

target_include_directories(slamtools
    PRIVATE
        ${CERES_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(slamtools
    PUBLIC
        cxx_std_17
)

target_compile_options(slamtools
    PRIVATE
        -pedantic
        $<$<CONFIG:Debug>:-Wextra>
)

target_link_libraries(slamtools
    PUBLIC
        Eigen3::Eigen
        ${CERES_LIBRARIES}
)
