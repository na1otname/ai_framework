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

    std::unique_ptr<ai_framework::Engine> detect_engine =
        std::make_unique<ai_framework::Engine>(detect_model_path);
    std::unique_ptr<ai_framework::Engine> keypoint_engine =
        std::make_unique<ai_framework::Engine>(keypoint_model_path);

    std::shared_ptr<TopdownProcess> topdownprocess_ptr_ =
        std::make_shared<TopdownProcess>(detect_engine->get_config(),
                                         keypoint_engine->get_config(),
                                         0.4f, 0.4f, false);
    auto input_det_ptr = detect_engine->get_input_tensor_ptr();
    auto output_det_ptr = detect_engine->get_output_tensor_ptr();

    auto input_keypoint_ptr = keypoint_engine->get_input_tensor_ptr();
    auto output_keypoint_prt = keypoint_engine->get_output_tensor_ptr();

    topdownprocess_ptr_->PreProcess({frame}, input_det_ptr);
    detect_engine->DoInference();

    topdownprocess_ptr_->PostProcessDetect(output_det_ptr);

    std::vector<Result> result_ = topdownprocess_ptr_->get_result();
    std::vector<TopdownMeta> meta_info;
    topdownprocess_ptr_->CropImageByDetectBox({frame}, result_, input_keypoint_ptr, meta_info);
    keypoint_engine->DoInference();
    topdownprocess_ptr_->PostProcessRTMPose(output_keypoint_prt, meta_info);

    // 复用 result_：PostProcessRTMPose 已把关键点追加进 result_，重新取回
    result_ = topdownprocess_ptr_->get_result();

    std::vector<std::string> labels = {"hand"};
    // 无显示环境下直接保存标注结果图（避免 cv::imshow 的 GTK 后端初始化失败）
    cv::Mat frame_result = GetImageResult(frame, result_, labels);
    // cv::imshow("test", frame_result);
    // cv::waitKey(0);
    // cv::imwrite("/home/orangepi/Code/ai_framework/source/result.jpg", frame_result);

    // topdownprocess_ptr_->
    return 0;
}
