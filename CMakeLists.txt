cmake_minimum_required(VERSION 3.10.0)
project(ai_framework VERSION 0.1.0 LANGUAGES C CXX)


# 输出系统信息
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Not RK3588")
# 检查架构
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "i386")
    set(IS_X86_ARCH TRUE)
else()
    set(IS_X86_ARCH FALSE)
endif()


# find_package(ai_instance_tensorrt QUIET)
# if (ai_instance_tensorrt_FOUND)
#     message(STATUS "Found ai_instance_tensorrt")
#     set(AI_LIB ${ai_instance_tensorrt_LIBS})
#     include_directories(/usr/local/cuda/include)
#     add_definitions(-DTRT=1)
# else ()
#     find_package(ai_instance_onnx QUIET)
#     if (ai_instance_onnx_FOUND)
#         message(STATUS "Found ai_instance_onnx")
#         set(AI_LIB ${ai_instance_onnx_LIBS})
#         add_definitions(-DONNX=1)
#     endif ()
# endif ()
# find_package(ai_instance_rk3588 QUIET)
# if (ai_instance_rk3588_FOUND)
#     message(STATUS "Found ai_instance_rk3588")
#     set(AI_LIB ${ai_instance_rk3588_LIBS})
#     add_definitions(-DRK3588=1)
# endif ()
# find_package(ai_instance_nnrt QUIET)
# if (ai_instance_nnrt_FOUND)
#     message(STATUS "Found ai_instance_nnrt")
#     set(AI_LIB ${ai_instance_nnrt_LIBS})
#     add_definitions(-DNNRT=1)
# endif ()
# message(STATUS "AI_LIB is ${AI_LIB}")

set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

find_package(OpenCV REQUIRED)

include(CTest)
enable_testing()

# 收集 utils 工具源码（供 test 使用），必须在 add_subdirectory(test) 之前定义
file(GLOB_RECURSE UTILS "src/utils/*.cpp")

add_subdirectory(test)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


# set(Eingine_INCLUDE ${CMAKE_SOURCE_DIR}/src/engine)
set(Rtmdet_INCLUDE ${CMAKE_SOURCE_DIR}/src/rtmdet)

#  rknn_api 文件夹路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn/Linux/librknn_api)
#  rknn_api include 路径
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
#  rknn_api lib 路径
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

set(RGA_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rga)
set(RGA_API_INCLUDE_PATH ${RGA_API_PATH}/include)
set(RGA_API_LIB_PATH ${RGA_API_PATH}/libs/Linux/gcc-${LIB_ARCH}/librga.so)

# 用来搜索头文件的目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)



add_subdirectory(src)

