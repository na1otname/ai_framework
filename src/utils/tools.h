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

bool ContainsSubString(const std::string &str, const std::string &substring);

inline int32_t __clip(float val, float min, float max);

template <typename T>
T sigmoid(T x);

int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);

float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);

float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale);

void compute_dfl(float *tensor, int dfl_len, float *box);

float fp16_to_f32(uint16_t half);

int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices);

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
                       float xmin1, float ymin1, float xmax1, float ymax1);

int nms(int validCount, std::vector<Bbox> &bboxes, std::vector<int> &order, float threshold);

int nms(const int validCount, const std::vector<Bbox> &bboxes,
        const std::vector<int> classIds, std::vector<int> &order,
        const int filterId, const float threshold);