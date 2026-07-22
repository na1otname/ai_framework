# ============================================================
# OpenMMLab Test Suite
# ============================================================

# 测试可执行文件
add_executable(test_ai_framework
    main.cpp
)




# 添加项目源码根目录（用于 #include "utils/logger.h" 等）
target_include_directories(test_ai_framework PRIVATE
    ${RKNN_API_INCLUDE_PATH}
    ${RGA_API_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/src
)

# 链接被测库
target_link_libraries(test_ai_framework PRIVATE
    engine
    ${RKNN_API_LIB_PATH}
    ${RGA_API_LIB_PATH}
    spdlog
    fmt
    systemd
)
# 注册 CTest 测试
add_test(NAME OpenMMLabTest
    COMMAND test_openmmlab
)

# 可选：设置测试超时（秒）
set_tests_properties(OpenMMLabTest PROPERTIES
    TIMEOUT 30
)

