#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <memory>

#include <opencv2/opencv.hpp>

#include "engine/ai_instance.h"
#include "image_process/yolo/yolo_postprocess.h"
#include "image_process/yolo/yolo_preprocess.h"

int main()
{
    // === 1. 加载模型（Rk3588 → TensorData → BindInputAndOutput） ===
    const char *model_path =
        "/home/orangepi/Code/ai_framework/model/rtmdet_nano_320x320_static_int8.rknn";
    printf("[1/6] Loading model: %s\n", model_path);
    auto engine = ai_framework::Engine(ai_framework::RKNN_FORMAT, model_path);

    // 获取 tensor 指针（Engine 内部已完成 Rk3588::Initialize +
    // TensorData 构造 + BindInputAndOutput）
    auto input = engine.get_input_tensor_ptr();
    auto output = engine.get_output_tensor_ptr();

    // 获取模型配置
    const auto &config = engine.get_config();
    auto input_shape =
        config.input_layer_shape.at(config.input_index_to_name.at(0));
    // 模型输入可能是 NHWC(1,H,W,C) 或 NCHW(1,C,H,W)，取 H 和 W
    // NHWC: at(1)=H, at(2)=W, at(3)=C | NCHW: at(1)=C, at(2)=H, at(3)=W
    int model_height, model_width;
    if (input_shape.size() >= 4)
    {
        auto fmt_str = config.input_fmt_str.at(config.input_index_to_name.at(0));
        if (fmt_str == "NHWC")
        {
            model_height = static_cast<int>(input_shape.at(1));
            model_width = static_cast<int>(input_shape.at(2));
        }
        else
        {
            model_height = static_cast<int>(input_shape.at(2));
            model_width = static_cast<int>(input_shape.at(3));
        }
    }
    else
    {
        model_height = model_width = 320;
    }
    int target_side = std::max(model_height, model_width);
    printf("[2/6] Model input: %dx%d (fmt=%s), outputs: %d\n",
           model_width, model_height,
           config.input_fmt_str.at(config.input_index_to_name.at(0)).c_str(),
           config.output_tensors_count);

    // === 3. 读取图片 ===
    const char *img_path =
        "/home/orangepi/Code/ai_framework/source/2024_09_13_08_30_25_001_cbw.jpg";
    printf("[3/6] Reading image: %s\n", img_path);
    cv::Mat img = cv::imread(img_path);
    if (img.empty())
    {
        printf("ERROR: Failed to load image: %s\n", img_path);
        return -1;
    }
    printf("  Image size: %dx%d\n", img.cols, img.rows);

    // === 4. 预处理 ===
    // 与 Python bbox_preprocess 一致：letterbox (resize + pad)，uint8 直传，不做归一化
    printf("[4/6] Preprocessing...\n");
    YoloPreProcess preprocess(target_side, false);
    preprocess.Run({img}, input);

    // === 5. 推理 ===
    printf("[5/6] Running inference...\n");
    engine.DoInference();
    printf("  Inference done.\n");

    // === 6. 后处理 ===
    printf("[6/6] Postprocessing...\n");
    std::vector<float> conf_threshold = {0.5f};
    float sum_conf_threshold = 0.5f;

    PostProcess postprocess(config, conf_threshold, sum_conf_threshold, 0.25f);

    // 构建 output tensors（const_cast 仅用于 PostProcess 读取）
    int output_count = engine.get_output_tensor_count();
    std::vector<void *> out_vec(output_count);
    for (int i = 0; i < output_count; ++i)
    {
        out_vec[i] = const_cast<void *>(output[i]);
    }
    void **out_tensors = out_vec.data();
    postprocess.Run(out_tensors);

    // === 7. 绘制结果 ===
    const auto &results = postprocess.get_result();
    printf("  Detected %zu objects:\n", results.size());

    // 缩放比例：预处理时 YoloPreProcess 将长边等比缩放再 padding
    float ratio = (img.cols >= img.rows)
                      ? (1.0f * img.cols / target_side)
                      : (1.0f * img.rows / target_side);

    const char *label_names[] = {"skeleton"};

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto &res = results[i];

        int x1 = std::max(0, std::min(static_cast<int>(res.box.x1 * ratio), img.cols - 1));
        int y1 = std::max(0, std::min(static_cast<int>(res.box.y1 * ratio), img.rows - 1));
        int x2 = std::max(0, std::min(static_cast<int>(res.box.x2 * ratio), img.cols - 1));
        int y2 = std::max(0, std::min(static_cast<int>(res.box.y2 * ratio), img.rows - 1));

        const char *label = (res.class_id >= 0 && res.class_id < 1)
                                ? label_names[res.class_id]
                                : "unknown";

        // printf("  [%zu] %s: bbox=[%d,%d,%d,%d] conf=%.4f\n",
        //        i, label, x1, y1, x2, y2, res.obj_prob);

        cv::Scalar color(0, 255, 0);
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        char text[128];
        snprintf(text, sizeof(text), "%s %.2f", label, res.obj_prob);
        int baseline = 0;
        cv::Size ts =
            cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        cv::rectangle(img,
                      cv::Point(x1, y1 - ts.height - 10),
                      cv::Point(x1 + ts.width, y1),
                      color, -1);
        cv::putText(img, text, cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }

    const char *output_path = "/home/orangepi/Code/ai_framework/source/output_rtmdet.jpg";
    cv::imwrite(output_path, img);
    // printf("  Result saved to: %s\n", output_path);
    // cv::imshow("test", img);
    // cv::waitKey(0);

    return 0;
}
