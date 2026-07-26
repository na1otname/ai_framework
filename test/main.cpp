#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "engine/ai_instance.h"
#include "image_process/yolo/yolo_postprocess.h"

// ==================== FP16 ↔ FP32 ====================

static inline uint16_t f32_to_f16(float val)
{
    uint32_t x;
    std::memcpy(&x, &val, sizeof(float));
    uint16_t sign = (x >> 16) & 0x8000;
    int exp = static_cast<int>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (((x >> 23) & 0xFF) == 0)
        return sign;
    if (exp >= 31)
        return sign | 0x7C00;
    if (exp <= 0)
        return sign;
    return sign | (exp << 10) | mant;
}

static inline float f16_to_f32(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    if (exp == 0)
        return 0.0f;
    if (exp == 31)
        return sign ? -INFINITY : INFINITY;
    uint32_t f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

// ==================== NPU Native Layout 解包 ====================
// RK3588 NPU zero-copy 输出为 native layout (fmt=2):
// 数据以 16-way channel interleave 方式存储
// 对于每个 (h,w)，16 个连续的 FP16 值对应 16 个 padded channel
// 有效 channel 0..C-1，padding channel C..15 为 0

static void unpack_native_fp16_to_fp32(
    const void *raw, int /*mem_size_bytes*/,
    int C, int H, int W, float *dst)
{
    const uint16_t *src = static_cast<const uint16_t *>(raw);
    const int NATIVE_C = 16;

    for (int h = 0; h < H; ++h)
    {
        for (int w = 0; w < W; ++w)
        {
            int base = (h * W + w) * NATIVE_C;
            for (int c = 0; c < C; ++c)
                dst[c * H * W + h * W + w] = f16_to_f32(src[base + c]);
        }
    }
}

// ==================== main ====================

int main()
{
    // === 1. 加载模型 ===
    const char *model_path =
        "/home/orangepi/Code/ai_framework/model/rtmdet_nano_320x320-fp16.rknn";
    printf("[1/5] Loading model: %s\n", model_path);
    auto engine = ai_framework::Engine(ai_framework::RKNN_FORMAT, model_path);

    // === 2. 读取图片 ===
    const char *img_path =
        "/home/orangepi/Code/inference-rknn/resources/img/test.jpg";
    printf("[2/5] Reading image: %s\n", img_path);
    cv::Mat img = cv::imread(img_path);
    if (img.empty())
    {
        printf("ERROR: Failed to read image\n");
        return -1;
    }
    printf("      original: %dx%d\n", img.cols, img.rows);

    // === 3. Letterbox + 归一化 → FP16 NHWC ===
    printf("[3/5] Letterbox 320x320 → FP16 NHWC...\n");
    const int TARGET_W = 320, TARGET_H = 320;

    float scale = std::min(
        static_cast<float>(TARGET_W) / img.cols,
        static_cast<float>(TARGET_H) / img.rows);
    int scaled_w = static_cast<int>(std::round(img.cols * scale));
    int scaled_h = static_cast<int>(std::round(img.rows * scale));
    int pad_x = (TARGET_W - scaled_w) / 2;
    int pad_y = (TARGET_H - scaled_h) / 2;
    printf("      scale=%.3f  pad=(%d,%d)\n", scale, pad_x, pad_y);

    cv::Mat resized, canvas(TARGET_H, TARGET_W, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(img, resized, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, scaled_w, scaled_h)));

    uint16_t *input_fp16 = static_cast<uint16_t *>(engine.get_input_tensor_ptr()[0]);
    for (int y = 0; y < TARGET_H; ++y)
    {
        const uint8_t *row = canvas.ptr<uint8_t>(y);
        for (int x = 0; x < TARGET_W; ++x)
        {
            // === 归一化方案 (按需切换) ===
            // 方案A: mmdet 标准 BGR 归一化 (适用于未 bake 预处理的模型)
            // float b = (row[x*3+0] - 103.53f) / 57.375f;
            // float g = (row[x*3+1] - 116.28f) / 57.12f;
            // float r = (row[x*3+2] - 123.675f) / 58.395f;

            // 方案B: BGR / 255 → [0,1] (适用于 bake 了 /255 的模型)
            // float b = row[x*3+0] / 255.0f;
            // float g = row[x*3+1] / 255.0f;
            // float r = row[x*3+2] / 255.0f;

            // 方案C: 原始 BGR [0,255] (适用于 mmdeploy 导出、预处理已 bake)
            float b = static_cast<float>(row[x * 3 + 0]);
            float g = static_cast<float>(row[x * 3 + 1]);
            float r = static_cast<float>(row[x * 3 + 2]);

            int base = (y * TARGET_W + x) * 3;
            input_fp16[base + 0] = f32_to_f16(b);
            input_fp16[base + 1] = f32_to_f16(g);
            input_fp16[base + 2] = f32_to_f16(r);
        }
    }

    // === 4. 推理 ===
    printf("[4/5] Running inference...\n");
    engine.DoInference();

    // === 5. 后处理 & 保存 ===
    printf("[5/5] Postprocess & draw...\n");
    const void *const *outputs = engine.get_output_tensor_ptr();
    int out_count = engine.get_output_tensor_count();
    printf("      output tensors: %d\n", out_count);

    // 输出配置: {n_elems, C, H, W, mem_size_bytes}
    struct OutInfo
    {
        int elems, C, H, W, mem_size;
    };
    const OutInfo out_info[6] = {
        {1600, 1, 40, 40, 51200}, // cls_P3
        {400, 1, 20, 20, 12800},  // cls_P4
        {100, 1, 10, 10, 3200},   // cls_P5
        {6400, 4, 40, 40, 51200}, // box_P3
        {1600, 4, 20, 20, 12800}, // box_P4
        {400, 4, 10, 10, 3200},   // box_P5
    };

    std::vector<const float *> output_fp32(6);
    std::vector<float *> fp32_bufs(6);
    for (int i = 0; i < 6; ++i)
    {
        fp32_bufs[i] = new float[out_info[i].elems];
        unpack_native_fp16_to_fp32(
            outputs[i], out_info[i].mem_size,
            out_info[i].C, out_info[i].H, out_info[i].W,
            fp32_bufs[i]);
        output_fp32[i] = fp32_bufs[i];
    }

    std::vector<std::array<int, 4>> cls_dims = {
        {1, 1, 40, 40}, {1, 1, 20, 20}, {1, 1, 10, 10}};
    std::vector<std::array<int, 4>> box_dims = {
        {1, 4, 40, 40}, {1, 4, 20, 20}, {1, 4, 10, 10}};

    RtmdetPostParams post_params;
    post_params.input_width = 320;
    post_params.input_height = 320;
    post_params.strides = {8, 16, 32};
    post_params.score_thr = 0.5f; // 与 Python 一致
    post_params.nms_iou = 0.28f;  // 与 Python 一致

    RtmdetPostProcess postprocess(post_params);
    auto detections = postprocess.Run(output_fp32, cls_dims, box_dims);
    printf("      detections: %zu\n", detections.size());

    for (auto &d : detections)
    {
        float x1 = (d.x1 - pad_x) / scale, y1 = (d.y1 - pad_y) / scale;
        float x2 = (d.x2 - pad_x) / scale, y2 = (d.y2 - pad_y) / scale;
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img.cols - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img.rows - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img.cols - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img.rows - 1)));

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                      cv::Scalar(0, 255, 0), 2);
        char label[32];
        std::snprintf(label, sizeof(label), "%.2f", d.score);
        cv::putText(img, label, cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        printf("  box=[%.1f,%.1f,%.1f,%.1f] score=%.4f\n", x1, y1, x2, y2, d.score);
    }

    const char *save_path = "/home/orangepi/Code/ai_framework/source/result.jpg";
    cv::imwrite(save_path, img);
    printf("\nDone! Saved → %s\n", save_path);

    for (int i = 0; i < 6; ++i)
        delete[] fp32_bufs[i];
    return 0;
}
