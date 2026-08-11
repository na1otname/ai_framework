#pragma once
#include "chrono"
#include "opencv2/opencv.hpp"
#include "types.h"

void ShowResults(const cv::Mat &original_image, const int target_side_length,
                 const std::vector<Result> &results,
                 std::vector<std::string> &labels, int cv_wait_ms,
                 bool enable_track = false, bool is_save = false,
                 bool hdmi_output = false, bool cv_show = true);
void ShowAndSave(const cv::Mat &image, int cv_wait_ms, bool is_save);
cv::Mat GetImageResult(const cv::Mat &original_image,
                       const int target_side_length,
                       const std::vector<Result> &results,
                       std::vector<std::string> &labels,
                       bool enable_track = false);

void AddWeightedSegment(cv::Mat &image, const cv::Mat &seg_mat, int id);

std::vector<std::string> ReadLabelsFromTextFile(const std::string &filename);
void CoordinateTransformation(float &x, float &y, int width, int height,
                              int target_side_length);

double get_current_time();