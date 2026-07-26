#pragma once

#include <cstdint>
#include <vector>
#include <array>

/**
 * @brief 单个检测结果: [x1, y1, x2, y2, score, class_id]
 */
struct Detection
{
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

/**
 * @brief RTMDet / YOLO 通用后处理参数
 *
 * 默认值对应 RTMDet-nano 在 640×640 输入、stride [8,16,32] 的常见配置。
 */
struct RtmdetPostParams
{
    int input_width = 320;  // 模型输入宽 (rtmdet_nano_320x320)
    int input_height = 320; // 模型输入高
    int num_classes = 1;    // 类别数 (cls_score 的通道数)
    float score_thr = 0.3f; // 分数阈值
    float nms_iou = 0.5f;   // NMS IoU 阈值
    int max_per_img = 100;  // 每张图最多保留的检测框数

    // 三层特征图对应的 stride，顺序与输出一致: [P3, P4, P5]
    std::vector<int> strides = {8, 16, 32};
};

/**
 * @brief RTMDet 后处理器
 *
 输入: 模型原始输出 —— 6 个 tensor (const float*)：
 *       cls_P3, cls_P4, cls_P5,  box_P3, box_P4, box_P5
 * 输出: 检测框列表 Detection (坐标在原图坐标系下)
 *
 * 处理流程:
 *   1. 对每层生成 anchor 中心点 (prior)
 *   2. 拉平 cls_score → sigmoid
 *   3. 用 prior 中心点 ± ltrb 解码出 xyxy
 *   4. 按 score_thr 过滤 → NMS → TopK
 */
class RtmdetPostProcess
{
public:
    RtmdetPostProcess() = default;
    explicit RtmdetPostProcess(const RtmdetPostParams &params);

    /**
     * @brief 执行后处理
     * @param outputs 6 个 float*，顺序: [cls1, cls2, cls3, box1, box2, box3]
     * @param cls_dims 每个 cls tensor 的维度 {N, C, H, W}，共 3 组
     * @param box_dims 每个 box tensor 的维度 {N, C, H, W}，共 3 组 (C=4)
     * @return 检测结果列表
     */
    std::vector<Detection> Run(
        const std::vector<const float *> &outputs,
        const std::vector<std::array<int, 4>> &cls_dims,
        const std::vector<std::array<int, 4>> &box_dims);

    const RtmdetPostParams &params() const { return params_; }

private:
    RtmdetPostParams params_;

    // 内部辅助结构
    struct Box
    {
        float x1, y1, x2, y2;
    };

    /// 为某一层生成 prior 中心点 (映射到原图坐标)
    std::vector<std::pair<float, float>> GeneratePriors(int stride, int h, int w);

    /// sigmoid
    static float Sigmoid(float x);

    /// 计算两个框的 IoU
    static float ComputeIoU(const Box &a, const Box &b);

    /// NMS (贪心)
    std::vector<int> Nms(const std::vector<Box> &boxes,
                         const std::vector<float> &scores,
                         float iou_thr);
};
