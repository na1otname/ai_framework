project(ai_framework)

add_executable(${PROJECT_NAME} main.cpp )

add_subdirectory(engine)
add_subdirectory(image_process)


target_include_directories(${PROJECT_NAME} PUBLIC
    ${IMAGE_PROCESS_DIR}
    ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME} PUBLIC     
    engine
    ${RKNN_API_LIB_PATH}
    ${RGA_API_LIB_PATH}
    spdlog
    fmt
    systemd
    detection_preprocess
    detection_postprocess)