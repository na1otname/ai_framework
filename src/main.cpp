#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <memory>

#include <opencv2/opencv.hpp>

#include "engine/ai_instance.h"
#include "detection/detection_postprocess.h"
#include "detection/detection_preprocess.h"
#include "topdown/topdown_process.h"
#include "utils/tools.h"

int main(int, char **)
{
    const char *detect_model_path =
        "/home/orangepi/Code/ai_framework/model/rtmdet_nano_320x320_static_int8.rknn";
    const char *keypoint_model_path = "/home/orangepi/Code/ai_framework/model/rtmpose-m_8xb256_hand_finetune-fp16.rknn";

        const char *image_path = "/home/orangepi/Code/ai_framework/source/test.jpg";

    cv::Mat frame = cv::imread(image_path);

    ai_framework::Engine detect_engine(ai_framework::RKNN_FORMAT, detect_model_path);
    ai_framework::Engine keypoint_engine(ai_framework::RKNN_FORMAT, keypoint_model_path);

    std::shared_ptr<TopdownProcess> topdownprocess_ptr_ =
        std::make_shared<TopdownProcess>(detect_engine.get_config(),
                                         keypoint_engine.get_config(),
                                         0.4f, 0.4f, false);

    // topdownprocess_ptr_->
    return 0;
}
