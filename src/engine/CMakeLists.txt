add_library(engine SHARED 
    ai_framework.cpp
    ai_framework.h
    ai_instance.cpp
    ai_instance.h
    backend/rk3588.cpp
    backend/rk3588.h
)

target_compile_definitions(engine PRIVATE RK3588=1)

target_include_directories(engine PUBLIC
    ${RKNN_API_INCLUDE_PATH}
    ${RGA_API_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(engine PUBLIC
    ${RKNN_API_LIB_PATH}
    ${RGA_API_LIB_PATH}
    spdlog
    fmt
    systemd
)