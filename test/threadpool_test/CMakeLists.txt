project(TEST_THREADPOOL)
add_executable(${PROJECT_NAME}
    main.cpp

)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    RK3588=1
    AI_FRAMEWORK_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config"
)





# 添加项目源码根目录（用于 #include "utils/logger.h" 等）
target_include_directories(${PROJECT_NAME} PRIVATE
    ${RKNN_API_INCLUDE_PATH}
    ${RGA_API_INCLUDE_PATH}
    ${IMAGE_PROCESS_DIR}
    ${TIME_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/src
)

# 链接被测库
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${RKNN_API_LIB_PATH}
    ${RGA_API_LIB_PATH}
    engine
    spdlog
    fmt
    systemd
    yaml-cpp
    -Wl,--whole-archive
    detection_preprocess
    detection_postprocess
    -Wl,--no-whole-archive
)
# 注册 CTest 测试
add_test(NAME ThreadPoolTest
    COMMAND TEST_THREADPOOL
)

# 可选：设置测试超时（秒）
set_tests_properties(ThreadPoolTest PROPERTIES
    TIMEOUT 30
)

