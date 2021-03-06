﻿CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

PROJECT(BTRC-GUI)

FILE(GLOB_RECURSE SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.h")

ADD_EXECUTABLE(BtrcGUI ${SRC})

FOREACH(_SRC IN ITEMS ${SRC})
    GET_FILENAME_COMPONENT(SRC "${_SRC}" PATH)
    STRING(REPLACE "${PROJECT_SOURCE_DIR}/btrc/gui" "" _GRP_PATH "${SRC}")
    STRING(REPLACE "/" "\\" _GRP_PATH "${_GRP_PATH}")
    SOURCE_GROUP("${_GRP_PATH}" FILES "${_SRC}")
ENDFOREACH()

BTRC_SET_CXX_LANG_VERSION(BtrcGUI)

IF(MSVC)
    SET_PROPERTY(
        TARGET BtrcGUI
        PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
ENDIF()

TARGET_INCLUDE_DIRECTORIES(BtrcGUI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
TARGET_LINK_LIBRARIES(BtrcGUI PUBLIC BtrcBuiltin OpenGL::GL glfw glew_s imgui portable_file_dialogs)
