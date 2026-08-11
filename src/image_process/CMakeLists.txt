add_library(detection_preprocess detection/detection_preprocess.cpp)
target_compile_definitions(detection_preprocess PRIVATE RK3588=1)
target_include_directories(detection_preprocess
	PUBLIC
		${OpenCV_INCLUDE_DIRS}
		${RGA_API_INCLUDE_PATH}
		${CMAKE_CURRENT_SOURCE_DIR}
		

)
target_link_libraries(detection_preprocess
	PUBLIC
		${RGA_API_LIB_PATH}
		${OpenCV_LIBS}
)

add_library(detection_postprocess 
	detection/detection_postprocess.cpp
	topdown/topdown_postprocess.h
	topdown/topdown_postprocess.cpp)
target_include_directories(detection_postprocess
	PUBLIC
		${OpenCV_INCLUDE_DIRS}
		${RGA_API_INCLUDE_PATH}
		${CMAKE_CURRENT_SOURCE_DIR}
		${CMAKE_SOURCE_DIR}/src
		engine
)
target_link_libraries(detection_postprocess
	PUBLIC
		${RGA_API_LIB_PATH}
		${OpenCV_LIBS}
		engine
)