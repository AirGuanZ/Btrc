project(tinyobjloader)

cmake_minimum_required(VERSION 3.2)

set(tinyobjloader-Source
  ${CMAKE_CURRENT_SOURCE_DIR}/tiny_obj_loader.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tiny_obj_loader.cc
  )
  
add_library(tinyobjloader STATIC ${tinyobjloader-Source})

target_include_directories(tinyobjloader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
