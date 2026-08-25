project(ai_framework)

add_executable(${PROJECT_NAME} main.cpp )


target_compile_definitions(ai_framework PRIVATE RK3588=1)

add_subdirectory(engine)
add_subdirectory(image_process)


target_include_directories(${PROJECT_NAME} PUBLIC
    ${IMAGE_PROCESS_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/utils)

target_link_libraries(${PROJECT_NAME} PUBLIC     
    engine
    ${RKNN_API_LIB_PATH}
    ${RGA_API_LIB_PATH}
    -Wl,--whole-archive
    detection_preprocess
    detection_postprocess
    -Wl,--no-whole-archive
    spdlog
    fmt
    systemd
)