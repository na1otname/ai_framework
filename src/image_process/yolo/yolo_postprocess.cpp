#include "yolo_postprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

#include "utils/logger.h"

// ==================== 构造 ====================

RtmdetPostProcess::RtmdetPostProcess(const RtmdetPostParams &params)
    : params_(params) {}

// ==================== 公开接口 Run ====================

std::vector<Detection> RtmdetPostProcess::Run(
    const std::vector<const float *> &outputs,
    const std::vector<std::array<int, 4>> &cls_dims,
    const std::vector<std::array<int, 4>> &box_dims)
{
    const int num_levels = 3;

    // --- 1. 为每一层生成 prior 中心点 (已映射到原图) ---
    std::vector<std::vector<std::pair<float, float>>> mlvl_priors(num_levels);
    for (int i = 0; i < num_levels; ++i)
    {
        int h = cls_dims[i][2]; // H
        int w = cls_dims[i][3]; // W
        mlvl_priors[i] = GeneratePriors(params_.strides[i], h, w);
    }

    // --- 2. 拉平 cls_score 和 bbox_pred，同时做 sigmoid ---
    // 先算总 anchor 数
    int total_anchors = 0;
    for (int i = 0; i < num_levels; ++i)
        total_anchors += cls_dims[i][2] * cls_dims[i][3];

    // 临时存储: 展开后的所有候选框及其分数
    // 为了性能，先做 score 过滤再进 NMS (而不是全存)
    std::vector<Box> candidate_boxes;
    std::vector<float> candidate_scores;
    std::vector<int> candidate_labels;
    candidate_boxes.reserve(total_anchors);
    candidate_scores.reserve(total_anchors);
    candidate_labels.reserve(total_anchors);

    for (int lvl = 0; lvl < num_levels; ++lvl)
    {
        const float *cls_ptr = outputs[lvl];              // [1, C, H, W]
        const float *box_ptr = outputs[num_levels + lvl]; // [1, 4, H, W]

        int C = cls_dims[lvl][1]; // 类别数 (为 1 时是单类)
        int H = cls_dims[lvl][2];
        int W = cls_dims[lvl][3];

        int num_anchors = H * W;
        const auto &priors = mlvl_priors[lvl];

        for (int a = 0; a < num_anchors; ++a)
        {
            // --- 处理 cls_score: [C, H, W] NCHW 排列 ---
            // 对每个类别独立处理：找出最大分数及对应类别
            float max_score = 0.0f;
            int best_cls = 0;
            for (int c = 0; c < C; ++c)
            {
                // NCHW: index = c * H * W + a
                float raw = cls_ptr[c * H * W + a];
                float prob = Sigmoid(raw);
                if (prob > max_score)
                {
                    max_score = prob;
                    best_cls = c;
                }
            }

            if (max_score <= params_.score_thr)
                continue;

            // --- 处理 bbox_pred: [4, H, W] NCHW 排列 ---
            // 通道顺序 (mmdeploy RTMDet): [l, r, t, b]
            float l = box_ptr[0 * H * W + a];
            float r = box_ptr[1 * H * W + a];
            float t = box_ptr[2 * H * W + a];
            float b = box_ptr[3 * H * W + a];

            float cx = priors[a].first;
            float cy = priors[a].second;

            Box box;
            box.x1 = cx - l;
            box.y1 = cy - t;
            box.x2 = cx + r;
            box.y2 = cy + b;

            // 裁剪到图像范围内
            box.x1 = std::max(0.0f, std::min(box.x1, static_cast<float>(params_.input_width)));
            box.y1 = std::max(0.0f, std::min(box.y1, static_cast<float>(params_.input_height)));
            box.x2 = std::max(0.0f, std::min(box.x2, static_cast<float>(params_.input_width)));
            box.y2 = std::max(0.0f, std::min(box.y2, static_cast<float>(params_.input_height)));

            candidate_boxes.push_back(box);
            candidate_scores.push_back(max_score);
            candidate_labels.push_back(best_cls);
        }
    }

    // --- 3. NMS ---
    std::vector<int> keep_idx = Nms(candidate_boxes, candidate_scores, params_.nms_iou);

    // --- 4. TopK ---
    if (static_cast<int>(keep_idx.size()) > params_.max_per_img)
        keep_idx.resize(params_.max_per_img);

    // --- 5. 组装结果 ---
    std::vector<Detection> results;
    results.reserve(keep_idx.size());
    for (int idx : keep_idx)
    {
        Detection det;
        det.x1 = candidate_boxes[idx].x1;
        det.y1 = candidate_boxes[idx].y1;
        det.x2 = candidate_boxes[idx].x2;
        det.y2 = candidate_boxes[idx].y2;
        det.score = candidate_scores[idx];
        det.class_id = candidate_labels[idx];
        results.push_back(det);
    }

    LOG_INFO("RTMDet postprocess: {} raw candidates → {} after NMS → {} final",
             candidate_boxes.size(), keep_idx.size(), results.size());

    return results;
}

// ==================== 生成 Prior 中心点 ====================

std::vector<std::pair<float, float>> RtmdetPostProcess::GeneratePriors(
    int stride, int h, int w)
{
    std::vector<std::pair<float, float>> priors;
    priors.reserve(h * w);

    for (int y = 0; y < h; ++y)
    {
        // 网格左上角坐标 (offset=0, 与 mmdeploy RTMDet 一致)
        float cy = static_cast<float>(y) * static_cast<float>(stride);
        for (int x = 0; x < w; ++x)
        {
            float cx = static_cast<float>(x) * static_cast<float>(stride);
            priors.emplace_back(cx, cy);
        }
    }
    return priors;
}

// ==================== Sigmoid ====================

float RtmdetPostProcess::Sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

// ==================== ComputeIoU ====================

float RtmdetPostProcess::ComputeIoU(const Box &a, const Box &b)
{
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f)
        return 0.0f;
    return inter_area / union_area;
}

// ==================== NMS (贪心, 按 score 降序) ====================

std::vector<int> RtmdetPostProcess::Nms(
    const std::vector<Box> &boxes,
    const std::vector<float> &scores,
    float iou_thr)
{
    const int N = static_cast<int>(boxes.size());
    if (N == 0)
        return {};

    // 按分数降序排列索引
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&scores](int a, int b)
              { return scores[a] > scores[b]; });

    std::vector<int> keep;
    std::vector<bool> suppressed(N, false);

    for (int i = 0; i < N; ++i)
    {
        int idx_i = order[i];
        if (suppressed[idx_i])
            continue;

        keep.push_back(idx_i);

        for (int j = i + 1; j < N; ++j)
        {
            int idx_j = order[j];
            if (suppressed[idx_j])
                continue;

            float iou = ComputeIoU(boxes[idx_i], boxes[idx_j]);
            if (iou > iou_thr)
                suppressed[idx_j] = true;
        }
    }

    return keep;
}
