#pragma once

#include <vector>

#include "opencv2/opencv.hpp"


enum ModelType : uint16_t
{
    DETECTION_V8 = 0,
    DETECTION_V10,
    DETECTION_V11,
    DETECTION_V13,
    POSE_V8,
    SEGMENT_V11,
    DETECTION_V26,
    DETECTION_RTMDE
};
struct Bbox
{
    float x1;
    float y1;
    float x2;
    float y2;
};
struct KeyPoint
{
    float x;
    float y;
    float visibility;
};
struct Result
{
    ModelType model_type;
    Bbox box;
    float obj_prob;
    int class_id;
    cv::Mat seg_mat;
    std::vector<KeyPoint> key_points;
};
typedef struct
{
    int index;
    int sub_index;
    int grid_len;
} BboxesIdx;

