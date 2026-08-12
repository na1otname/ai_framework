#pragma once
#include "utils/logger.h"
#include "opencv2/opencv.hpp"

class PreProcess
{
public:
    PreProcess() = delete;
    PreProcess(int target_side_length, bool debug = false);
    void Run(const std::vector<cv::Mat> &input, void *tensors[]);
    const int &get_target_side_length() const { return target_side_length_; }

private:
    void MakeSquare(const cv::Mat &src, cv::Mat &dst);
    uint64_t PopulateData(
        const cv::Mat &data,
        float *dst,
        const std::vector<float> &mean = {0, 0, 0},
        const std::vector<float> &std = {255, 255, 255});
    int target_side_length_;
    bool debug_ = {false};
};