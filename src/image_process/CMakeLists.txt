add_library(yolo_preprocess yolo/yolo_preprocess.cpp)
target_compile_definitions(yolo_preprocess PRIVATE RK3588=1)
target_include_directories(yolo_preprocess
	PUBLIC
		${OpenCV_INCLUDE_DIRS}
		${RGA_API_INCLUDE_PATH}
		${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(yolo_preprocess
	PUBLIC
		${RGA_API_LIB_PATH}
		${OpenCV_LIBS}
)

add_library(yolo_postprocess yolo/yolo_postprocess.cpp)
target_include_directories(yolo_postprocess
	PUBLIC
		${OpenCV_INCLUDE_DIRS}
		${RGA_API_INCLUDE_PATH}
		${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(yolo_postprocess
	PUBLIC
		${RGA_API_LIB_PATH}
		${OpenCV_LIBS}
)