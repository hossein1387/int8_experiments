# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 11)

set (TARGET_NAME "classification_sample")

file (GLOB SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SRC})

link_directories(${LIB_FOLDER})

# Create library file from sources.
add_executable(${TARGET_NAME} ${SRC})

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
COMPILE_PDB_NAME ${TARGET_NAME})

##############################################################################################################################

target_link_libraries(/opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/inference_engine/samples/common/format_reader)
target_link_libraries(/opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/inference_engine/samples/thirdparty/gflags/lib)
target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} IE::ie_cpu_extension format_reader gflags)

include_directories(./include)
include_directories(./)
include_directories(/opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/inference_engine/src/extension)
include_directories(/opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/inference_engine/samples/common/format_reader/)
include_directories(/home/mhossein/inference_engine_samples/thirdparty/gflags/include/)


##############################################################################################################################



if(UNIX)
    target_link_libraries(${TARGET_NAME} ${LIB_DL} pthread)
endif()
