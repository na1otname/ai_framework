#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <memory>

#include <opencv2/opencv.hpp>

#include "engine/ai_instance.h"
#include "detection_postprocess.h"
#include "detection_preprocess.h"
#include "utils/tools.h"

int main()
{
    // === 1. 加载模型（Rk3588 → TensorData → BindInputAndOutput） ===
    const char *model_path =
        "/home/orangepi/Code/ai_framework/model/rtmdet_nano_320x320_static_int8.rknn";
    printf("[1/6] Loading model: %s\n", model_path);
    auto engine = ai_framework::Engine(ai_framework::RKNN_FORMAT, model_path);

    auto input = engine.get_input_tensor_ptr();
    auto output = engine.get_output_tensor_ptr();

    const char *image_path = "/home/orangepi/Code/ai_framework/source/test.jpg";
    cv::Mat frame = cv::imread(image_path);
    PreProcess preprocessor(320, false);

    preprocessor.Run({frame}, input);
    std::vector<float> conf_threshold = {0.4f};
    PostProcess postprocessor(engine.get_config(), conf_threshold, 0.4f, 0.4f);

    engine.DoInference();
    postprocessor.Run(output);

    std::vector<Result> result = postprocessor.get_result();
    std::vector<std::string> labels = {"hand"};
    // 无显示环境下直接保存标注结果图（避免 cv::imshow 的 GTK 后端初始化失败）
    cv::Mat annotated =
        GetImageResult(frame, preprocessor.get_target_side_length(), result, labels);
    cv::imwrite("/home/orangepi/Code/ai_framework/source/result.jpg", annotated);
    printf("[6/6] Result image saved to source/result.jpg\n");
    // for (const auto res : result)
    // {
    //     cv::rectangle(frame, cv::Point(res.box.x1, res.box.y1), cv::Point(res.box.x2, res.box.y2), cv::Scalar(0, 255, 0), 2);
    //     cv::imwrite("/home/orangepi/Code/ai_framework/source/result.jpg", frame);
    // }

    return 0;
}
