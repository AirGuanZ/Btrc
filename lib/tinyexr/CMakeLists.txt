# this cmake file is for compile tests.
# Not usable for your cmake app.
cmake_minimum_required(VERSION 3.5)

set(BUILD_TARGET "tinyexr")
set(SAMPLE_TARGET "test_tinyexr")

project(${BUILD_TARGET} CXX)

# cmake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/sanitizers)
find_package(Sanitizers) # Address sanitizer (-DSANITIZE_ADDRESS=ON)

# Require strict C++11 mode(e.g. `-std=c++11`)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TINYEXR_SOURCES
    ${PROJECT_SOURCE_DIR}/tinyexr.cc
    )

enable_language(C)
add_library(miniz STATIC deps/miniz/miniz.c)
target_include_directories(miniz PUBLIC deps/miniz)
set_target_properties(miniz PROPERTIES FOLDER "ThirdParty")
list(APPEND TINYEXR_EXT_LIBRARIES miniz)

add_library(${BUILD_TARGET} ${TINYEXR_SOURCES} ${TINYEXR_DEP_SOURCES})
add_sanitizers(${BUILD_TARGET})

target_include_directories(${BUILD_TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${BUILD_TARGET} ${TINYEXR_EXT_LIBRARIES} ${CMAKE_DL_LIBS})

# Increase warning level for clang.
IF (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set_source_files_properties(${TINYEXR_SOURCES} PROPERTIES COMPILE_FLAGS "-Weverything -Wno-padded -Wno-c++98-compat-pedantic -Wno-documentation -Wno-unused-member-function")
ENDIF ()
