
#pragma once
#include "ai_instance.h"
#include "opencv2/opencv.hpp"
#include "im2d.hpp"
#include "utils/types.h"
#include "utils/tools.h"

using ai_framework::ModelFormat;

class TopdownProcess
{
public:
    TopdownProcess() = delete;
    TopdownProcess(const ai_framework::Config &detect_config,
                   const ai_framework::Config &keypoint_config,
                   float detect_sum_conf_threshold,
                   float iou_threshold,
                   bool debug);
    ~TopdownProcess();

    void PreProcess(const std::vector<cv::Mat> &inputs, void *tensors[]);

    void CropImageByDetectBox(const std::vector<cv::Mat> &inputs,
                              const std::vector<Result> &bboxs,
                              void *tensors[],
                              std::vector<TopdownMeta> &metainfo);

    void PostProcessDetect(void **&tensors);

    void PostProcessRTMPose(void **&tensors, std::vector<TopdownMeta> &metainfo);

    const std::vector<Result> &get_result() const { return result_; }

private:
    static int GetInputSideLength(const ai_framework::Config &config);
    static void GetInputSize(const ai_framework::Config &config,
                             int &width,
                             int &height);

    void PostProcessDetect_(void **&tensors);

    void PostProcessRtmdet(void **&tensors);

    uint16_t ProcessRtmdet(const void *box_tensor, const void *score_tensor,
                           const void *sum_score_tensor, int grid_w, int grid_h,
                           int stride, int index, int output_per_branch);

    uint16_t ProcessDetect(const void *box_tensor,
                           const void *score_tensor,
                           const void *sum_score_tensor,
                           int grid_w, int grid_h, int stride,
                           int index, int output_per_branch);

    void MakeSquare(const cv::Mat &src, cv::Mat &dst);
    uint64_t PopulateData(
        const cv::Mat &data,
        float *dst,
        const std::vector<float> &mean = {0, 0, 0},
        const std::vector<float> &std = {255, 255, 255});
    cv::Mat GetAffineTransform(float center_x,
                               float center_y,
                               float scale_width,
                               float scale_height,
                               int output_image_width,
                               int output_image_height,
                               bool inverse = false);

    void InitModelType(const std::string &output_name, ModelType &modeltype_);

private:
    bool debug_ = {false};

    int detect_input_width_;
    int detect_input_height_;

    int keypoint_input_width_;
    int keypoint_input_height_;
    int dfl_len_;

    ModelFormat detect_model_format_;
    ModelFormat keypoint_model_format_;

    ModelType detect_model_type_;
    ModelType keypoint_model_type_;

    float iou_threshold_;
    uint16_t detect_num_of_layers_;
    uint16_t keypoints_num_of_layers_;

    float detect_conf_threshold_{0.4f}; // topdown 单类别检测阈值
    float detect_sum_conf_threshold_;

    std::vector<std::string> detect_output_layer_names_;
    std::vector<size_t> detect_output_element_count_;
    std::vector<std::vector<int64_t>> detect_output_layer_shape;
    std::vector<float> detect_zero_points_;
    std::vector<float> detect_scale_;

    std::vector<std::string> keypoints_output_layer_names_;
    std::vector<size_t> keypoints_output_element_count_;
    std::vector<std::vector<int64_t>> keypoints_output_layer_shape;
    std::vector<float> keypoints_zero_points_;
    std::vector<float> keypoints_scale_;

    std::vector<Result> result_;
    std::vector<Bbox> bboxes_;
    std::vector<BboxesIdx> bboxes_idx_;
    std::vector<int> class_id_;
    std::vector<float> obj_probs_;
};
