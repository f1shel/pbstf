cmake_minimum_required(VERSION 3.12)

project(pbstf CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

set(ALEMBIC_DEBUG_WARNINGS_AS_ERRORS OFF CACHE INTERNAL "" FORCE)
set(ALEMBIC_ILMBASE_LINK_STATIC      OFF CACHE INTERNAL "" FORCE)
set(DOCS_PATH                        OFF CACHE INTERNAL "" FORCE)
set(ALEMBIC_BUILD_LIBS               ON  CACHE INTERNAL "" FORCE)
set(ALEMBIC_LIB_USES_BOOST           OFF CACHE INTERNAL "" FORCE)
set(ALEMBIC_LIB_USES_TR1             OFF CACHE INTERNAL "" FORCE)
set(ALEMBIC_SHARED_LIBS              OFF CACHE INTERNAL "" FORCE)
set(USE_ARNOLD                       OFF CACHE INTERNAL "" FORCE)
set(USE_BINARIES                     ON  CACHE INTERNAL "" FORCE)
set(USE_EXAMPLES                     OFF CACHE INTERNAL "" FORCE)
set(USE_HDF5                         OFF CACHE INTERNAL "" FORCE)
set(USE_MAYA                         OFF CACHE INTERNAL "" FORCE)
set(USE_PRMAN                        OFF CACHE INTERNAL "" FORCE)
set(USE_PYALEMBIC                    OFF CACHE INTERNAL "" FORCE)
set(USE_STATIC_BOOST                 OFF CACHE INTERNAL "" FORCE)
set(USE_STATIC_HDF5                  OFF CACHE INTERNAL "" FORCE)
set(USE_TESTS                        OFF CACHE INTERNAL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/alembic)
target_include_directories(Alembic PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/alembic/lib>)

set(TIMER_BUILD_TEST OFF CACHE INTERNAL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/timer)

set(PARALLELUTIL_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/parallel-util)

file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(main ${headers} ${sources})
target_link_libraries(main PUBLIC Eigen3::Eigen Alembic timer parallel-util)

target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external)