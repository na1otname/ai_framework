
#pragma once
#include "opencv2/opencv.hpp"
#include "im2d.hpp"
#include "utils/types.h"

class TopdownProcess
{
public:
    TopdownProcess() = delete;
    TopdownProcess(int detect_target_side_length, int keypoint_target_side_length, bool debug);
    ~TopdownProcess();

    void PreProcess(const std::vector<cv::Mat> &inputs, void *tensors[]);

    void CropImageByDetectBox(const std::vector<cv::Mat> &inputs, const std::vector<Result> &bboxs, void *tensors[]);

private:
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

    bool debug_ = {false};
    int detect_target_side_length_;
    int keypoint_target_side_length_;

private:
};
