cmake_minimum_required(VERSION 3.10.0)
project(ai_framework VERSION 0.1.0 LANGUAGES C CXX)


# 输出系统信息
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Not RK3588")
# 检查架构
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "i386")
    set(IS_X86_ARCH TRUE)
else()
    set(IS_X86_ARCH FALSE)
endif()


set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)


include(CTest)
enable_testing()

# 收集 utils 工具源码（供 test 使用），必须在 add_subdirectory(test) 之前定义
file(GLOB_RECURSE UTILS "src/utils/*.cpp")

# 供 test 使用的共享目录变量，必须在 add_subdirectory(test) 之前定义
set(IMAGE_PROCESS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/image_process)

set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn/Linux/librknn_api)
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

set(RGA_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rga)
set(RGA_API_INCLUDE_PATH ${RGA_API_PATH}/include)
set(RGA_API_LIB_PATH ${RGA_API_PATH}/libs/Linux/gcc-${LIB_ARCH}/librga.so)

set(TIME_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 添加测试项目
add_subdirectory(test/rtmdet_test)
add_subdirectory(test/threadpool_test)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


# set(Eingine_INCLUDE ${CMAKE_SOURCE_DIR}/src/engine)
set(Rtmdet_INCLUDE ${CMAKE_SOURCE_DIR}/src/rtmdet)

add_subdirectory(src)

