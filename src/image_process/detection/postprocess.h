#pragma once
#include "ai_instance.h"
#include "utils/types.h"
#include "opencv2/opencv.hpp"
#include "stdint.h"
#include <cmath>
#include <type_traits>

using ai_framework::ModelFormat;

class PostProcess
{
public:
    PostProcess() = delete;
    PostProcess(const ai_framework::Config &config,
                std::vector<float> &conf_threshold, float sum_conf_threshold,
                float iou_threshold = 0.5f);
    void Run(void **&tensors);
    const std::vector<Result> &get_result() const { return result_; }

private:
    void PostProcessDetectSegment(void **&tensors);
    void PostProcessRtmdet(void **&tensors);
    void ProcessSegment(const void *mask_tensor, const void *proto_tensor,
                        Result &result, BboxesIdx bboxes_idx,
                        int output_per_branch);
    void PostProcessPose(void **&tensors);
    uint16_t ProcessDetect(const void *box_tensor, const void *score_tensor,
                           const void *sum_score_tensor, int grid_w, int grid_h,
                           int stride, int index, int output_per_branch);

    uint16_t ProcessRtmdet(const void *box_tensor, const void *score_tensor,
                           const void *sum_score_tensor, int grid_w, int grid_h,
                           int stride, int index, int output_per_branch);

    uint16_t ProcessPose(const void *box_tensor, const void *score_tensor,
                         const void *kpt_tensor, const void *visibility_tensor,
                         int grid_w, int grid_h, int stride, int index,
                         int output_per_branch);

    uint16_t num_of_layers_;
    std::vector<std::string> output_layer_names_;
    std::vector<size_t> output_element_count_;
    std::vector<std::vector<int64_t>> output_layer_shape;
    std::vector<float> zero_points_;
    std::vector<float> scale_;
    std::vector<float> *conf_threshold_;
    float sum_conf_threshold_;
    int model_width_{0};
    int model_height_{0};
    int seg_width_{0};
    int seg_height_{0};
    int dfl_len_{0};
    std::vector<Bbox> bboxes_;
    std::vector<BboxesIdx> bboxes_idx_;
    std::vector<int> class_id_;
    std::vector<float> obj_probs_;
    std::vector<float> kpt;
    std::vector<float> visibilities;
    std::vector<Result> result_;
    ModelType model_type_;
    ModelFormat model_format_;
    float iou_threshold_;
};