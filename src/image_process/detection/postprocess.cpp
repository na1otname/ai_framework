#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>

#include "iostream"
#include "utils/logger.h"
#include "utils/tools.h"



PostProcess::PostProcess(const ai_framework::Config &config,
                         std::vector<float> &conf_threshold,
                         float sum_conf_threshold,
                         float iou_threshold)
{
    model_format_ = config.model_format;
    iou_threshold_ = iou_threshold;
    num_of_layers_ = config.output_tensors_count;
    for (size_t i = 0; i < num_of_layers_; ++i)
    {
        auto layer = config.output_index_to_name.at(i);
        LOG_DEBUG("Post Index = {}, layer name = {}", i, layer);
        output_layer_names_.push_back(layer);
        output_layer_shape.push_back(config.output_layer_shape.at(layer));
        output_element_count_.push_back(config.output_element_count.at(layer));
        if (config.zero_point.find(layer) != config.zero_point.end())
        {
            zero_points_.push_back(config.zero_point.at(layer));
        }
        if (config.scale.find(layer) != config.scale.end())
        {
            scale_.push_back(config.scale.at(layer));
        }
    }
    conf_threshold_ = &conf_threshold;
    sum_conf_threshold_ = sum_conf_threshold;
    auto input_name = config.input_index_to_name.at(0);
    auto input_shape = config.input_layer_shape.at(input_name);
    // 输入可能是 NHWC (1,H,W,3) 或 NCHW (1,3,H,W)，按实际 fmt 取 H/W
    bool input_nhwc = false;
    auto fmt_it = config.input_fmt_str.find(input_name);
    if (fmt_it != config.input_fmt_str.end() &&
        fmt_it->second.find("NHWC") != std::string::npos)
    {
        input_nhwc = true;
    }
    if (input_nhwc)
    {
        model_height_ = input_shape.at(1);
        model_width_ = input_shape.at(2);
    }
    else
    {
        model_width_ = input_shape.at(3);
        model_height_ = input_shape.at(2);
    }
    auto output_boxes_shape =
        config.output_layer_shape.at(config.output_index_to_name.at(0));
    dfl_len_ = output_boxes_shape.at(1) / 4;
    auto output_boxes_name = config.output_index_to_name.at(0);
    if (ContainsSubString(output_boxes_name, "yolo26_detect"))
    {
        model_type_ = ModelType::DETECTION_V26;
    }
    else if (ContainsSubString(output_boxes_name, "yolov10"))
    {
        model_type_ = ModelType::DETECTION_V10;
    }
    else if (ContainsSubString(output_boxes_name, "yolov8_detect"))
    {
        model_type_ = ModelType::DETECTION_V8;
    }
    else if (ContainsSubString(output_boxes_name, "yolov8_pose") ||
             ContainsSubString(output_boxes_name, "yolo11_pose"))
    {
        model_type_ = ModelType::POSE_V8;
    }
    else if (ContainsSubString(output_boxes_name, "yolo13_detect"))
    {
        model_type_ = ModelType::DETECTION_V13;
    }
    else if (ContainsSubString(output_boxes_name, "yolo11_detect"))
    {
        model_type_ = ModelType::DETECTION_V11;
    }
    else if (ContainsSubString(output_boxes_name, "rtmdet") ||
             ContainsSubString(output_boxes_name, "rtmde") ||
             ContainsSubString(output_boxes_name, "rtm_"))
    {
        model_type_ = ModelType::DETECTION_RTMDE;
    }
    else if (ContainsSubString(output_boxes_name, "yolo11_segment"))
    {
        model_type_ = ModelType::SEGMENT_V11;
        auto tmp = config.output_layer_shape.rbegin();
        seg_width_ = tmp->second.at(3);
        seg_height_ = tmp->second.at(2);
        LOG_DEBUG("yolo11_segment resolution: {}x{}", seg_width_,
                 seg_height_);
    }
    LOG_DEBUG("ModelType: {}", model_type_);
}

void PostProcess::Run(void **&tensors)
{
    result_.clear();
    bboxes_.clear();
    class_id_.clear();
    obj_probs_.clear();
    bboxes_idx_.clear();
    if (model_type_ == ModelType::DETECTION_V8 ||
        model_type_ == ModelType::DETECTION_V26 ||
        model_type_ == ModelType::DETECTION_V10 ||
        model_type_ == ModelType::DETECTION_V11 ||
        model_type_ == ModelType::DETECTION_V13 ||
        model_type_ == ModelType::SEGMENT_V11)
    {
        PostProcessDetectSegment(tensors);
    }
    else if (model_type_ == ModelType::DETECTION_RTMDE)
    {
        PostProcessRtmdet(tensors);
    }
    else if (model_type_ == ModelType::POSE_V8)
    {
        kpt.clear();
        visibilities.clear();
        PostProcessPose(tensors);
    }
}


void PostProcess::PostProcessPose(void **&tensors)
{
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int output_per_branch = num_of_layers_ / 3;
    for (int i = 0; i < 3; ++i)
    {
        int box_idx = i * output_per_branch;
        int score_idx = i * output_per_branch + 1;
        int kpt_idx = i * output_per_branch + 2;
        int visibilities_idx = i * output_per_branch + 3;
        grid_h = output_layer_shape.at(box_idx).at(2);
        grid_w = output_layer_shape.at(box_idx).at(3);
        stride = model_height_ / grid_h;
        validCount +=
            ProcessPose(reinterpret_cast<const void *>(tensors[box_idx]),
                        reinterpret_cast<const void *>(tensors[score_idx]),
                        reinterpret_cast<const void *>(tensors[kpt_idx]),
                        reinterpret_cast<const void *>(tensors[visibilities_idx]),
                        grid_w, grid_h, stride, i, output_per_branch);
    }
    if (validCount <= 0)
    {
        return;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; i++)
    {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(obj_probs_, 0, validCount - 1, indexArray);
    nms(validCount, bboxes_, indexArray, iou_threshold_);
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1)
        {
            continue;
        }
        int n = indexArray[i];
        Result res;
        res.model_type = ModelType::POSE_V8;
        res.box = bboxes_.at(n);
        res.class_id = class_id_.at(n);
        res.obj_prob = obj_probs_.at(n);
        for (int j = 0; j < 34; j += 2)
        {
            auto kpt_x = kpt.at(34 * n + j);
            auto kpt_y = kpt.at(34 * n + j + 1);
            auto visibility = visibilities.at(17 * n + j / 2);
            res.key_points[j / 2].x = kpt_x;
            res.key_points[j / 2].y = kpt_y;
            res.key_points[j / 2].visibility = visibility;
        }
        result_.push_back(res);
    }
}

void PostProcess::PostProcessDetectSegment(void **&tensors)
{
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int output_per_branch = num_of_layers_ / 3;
    for (int i = 0; i < 3; ++i)
    {
        int box_index = i * output_per_branch;
        int score_index = i * output_per_branch + 1;
        int sum_score_index = i * output_per_branch + 2;
        grid_h = output_layer_shape.at(box_index).at(2);
        grid_w = output_layer_shape.at(box_index).at(3);
        stride = model_height_ / grid_h;
        validCount += ProcessDetect(tensors[box_index], tensors[score_index],
                                    tensors[sum_score_index], grid_w, grid_h,
                                    stride, i, output_per_branch);
    }
    //    std::cout << "validCount: " << validCount << std::endl;
    //  LOG_DEBUG("validCount: {}", validCount);
    if (validCount <= 0)
    {
        return;
    }
    std::vector<int> indexArray;
    if (model_type_ == ModelType::DETECTION_V8 ||
        model_type_ == ModelType::DETECTION_V11 ||
        model_type_ == ModelType::DETECTION_V13 ||
        model_type_ == ModelType::SEGMENT_V11)
    {
        for (int i = 0; i < validCount; i++)
        {
            indexArray.push_back(i);
        }
        quick_sort_indice_inverse(obj_probs_, 0, validCount - 1, indexArray);
        std::set<int> class_set(std::begin(class_id_), std::end(class_id_));
        for (auto c : class_set)
        {
            nms(validCount, bboxes_, class_id_, indexArray, c, iou_threshold_);
        }
    }
    for (int i = 0; i < validCount; ++i)
    {
        int n = i;
        Result res;
        if (model_type_ == ModelType::DETECTION_V8 ||
            model_type_ == ModelType::DETECTION_V11 ||
            model_type_ == ModelType::DETECTION_V13 ||
            model_type_ == ModelType::SEGMENT_V11)
        {
            if (indexArray[i] == -1)
            {
                continue;
            }
            n = indexArray[i];
            res.model_type = model_type_;
        }
        else if (model_type_ == ModelType::DETECTION_V10 ||
                 model_type_ == ModelType::DETECTION_V26)
        {
            n = i;
            res.model_type = model_type_;
        }
        res.box = bboxes_.at(n);
        res.class_id = class_id_.at(n);
        // 上面快排的时候，元素有交换
        res.obj_prob = obj_probs_.at(i);
        if (model_type_ == ModelType::SEGMENT_V11)
        {
            int mask_index = bboxes_idx_.at(n).index * output_per_branch + 3;
            int proto_index = num_of_layers_ - 1;
            ProcessSegment(tensors[mask_index], tensors[proto_index], res,
                           bboxes_idx_.at(n), output_per_branch);
        }
        result_.push_back(res);
    }
}

void PostProcess::PostProcessRtmdet(void **&tensors)
{
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    // RTMDet: first half outputs are cls scores, second half are bbox predictions
    int cls_layers = num_of_layers_ / 2;

    for (int i = 0; i < cls_layers; ++i)
    {
        int score_index = i;            // cls tensor
        int box_index = i + cls_layers; // bbox tensor
        grid_h = output_layer_shape.at(box_index).at(2);
        grid_w = output_layer_shape.at(box_index).at(3);
        stride = model_height_ / grid_h;
        validCount += ProcessRtmdet(tensors[box_index], tensors[score_index],
                                    nullptr, grid_w, grid_h, stride, i,
                                    cls_layers);
    }

    if (validCount <= 0)
    {
        return;
    }

    // Sort by confidence
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; i++)
    {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(obj_probs_, 0, validCount - 1, indexArray);
    nms(validCount, bboxes_, indexArray, iou_threshold_);
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1)
        {
            continue;
        }
        int n = indexArray[i];
        Result res;
        res.model_type = ModelType::DETECTION_RTMDE;
        res.box = bboxes_.at(n);
        res.class_id = class_id_.at(n);
        res.obj_prob = obj_probs_.at(n);

        result_.push_back(res);
    }
}

void PostProcess::ProcessSegment(const void *mask_tensor,
                                 const void *proto_tensor, Result &result,
                                 BboxesIdx bboxes_idx,
                                 int output_per_branch)
{
    const float *mask_tensor_float = reinterpret_cast<const float *>(mask_tensor);
    const float *proto_tensor_float =
        reinterpret_cast<const float *>(proto_tensor);
    const int8_t *mask_tensor_int8 =
        reinterpret_cast<const int8_t *>(mask_tensor);
    const int8_t *proto_tensor_int8 =
        reinterpret_cast<const int8_t *>(proto_tensor);
    bool is_qnt = zero_points_.empty() ? false : true;
#define COUNT 32
    float mask[COUNT] = {0};
    cv::Mat proto;
    if (model_format_ != ModelFormat::RKNN_FORMAT)
    {
        for (int i = 0; i < COUNT; i++)
        {
            mask[i] =
                mask_tensor_float[bboxes_idx.sub_index + i * bboxes_idx.grid_len];
        }
        proto = cv::Mat(COUNT, seg_height_ * seg_width_, CV_32FC1,
                        const_cast<float *>(proto_tensor_float));
    }
    else
    {
        auto mask_zp = zero_points_.at(bboxes_idx.index * output_per_branch + 3);
        auto mask_scale = scale_.at(bboxes_idx.index * output_per_branch + 3);
        auto proto_zp = zero_points_.at(num_of_layers_ - 1);
        auto proto_scale = scale_.at(num_of_layers_ - 1);
        // LOG_INFO("mask: {}, {}, proto: {} {}", mask_zp, mask_scale,
        //                    proto_zp, proto_scale);
        for (int i = 0; i < COUNT; ++i)
        {
            auto &element =
                mask_tensor_int8[bboxes_idx.sub_index + i * bboxes_idx.grid_len];
            mask[i] =
                is_qnt ? deqnt_affine_to_f32(element, mask_zp, mask_scale) : 0.0f;
        }
        proto = cv::Mat::zeros(COUNT, seg_height_ * seg_width_, CV_32FC1);
        if (is_qnt)
        {
            float *proto_ptr = proto.ptr<float>();
            for (int i = 0; i < COUNT * seg_height_ * seg_width_; ++i)
            {
                auto &element = proto_tensor_int8[i];
                proto_ptr[i] = deqnt_affine_to_f32(element, proto_zp, proto_scale);
            }
        }
    }
    cv::Mat mask_mat = cv::Mat(1, COUNT, CV_32FC1, mask);
    cv::Mat res_mat = mask_mat * proto;
    res_mat = res_mat.reshape(0, seg_height_);
    auto image_scale = model_height_ / seg_height_;
    auto roi = cv::Rect(result.box.x1 / image_scale, result.box.y1 / image_scale,
                        (result.box.x2 - result.box.x1) / image_scale,
                        (result.box.y2 - result.box.y1) / image_scale);
    auto sub_image = res_mat(roi);
    cv::Mat sub_res_image = sub_image > 0.5;
    result.seg_mat = cv::Mat::zeros(res_mat.size(), CV_8UC1);
    sub_res_image.copyTo(result.seg_mat(roi));
}



uint16_t PostProcess::ProcessRtmdet(const void *box_tensor, const void *score_tensor,
                                    const void *sum_score_tensor, int grid_w, int grid_h,
                                    int stride, int index, int output_per_branch)
{
    // RTMDet: score 层索引 = index，box 层索引 = index + output_per_branch
    // const float *box_tensor_float = reinterpret_cast<const float *>(box_tensor);
    // const float *score_tensor_float = reinterpret_cast<const float *>(score_tensor);
    const int8_t *box_tensor_int8 = reinterpret_cast<const int8_t *>(box_tensor);
    const int8_t *score_tensor_int8 = reinterpret_cast<const int8_t *>(score_tensor);
    const uint16_t *box_tensor_float16 = reinterpret_cast<const uint16_t *>(box_tensor);
    const uint16_t *score_tensor_float16 = reinterpret_cast<const uint16_t *>(score_tensor);

    bool is_qnt = zero_points_.empty() ? false : true;
    const float score_zp = is_qnt ? zero_points_.at(index) : 0.f;
    const float score_scale = is_qnt ? scale_.at(index) : 1.f;
    const float box_zp =
        is_qnt ? zero_points_.at(index + output_per_branch) : 0.f;
    const float box_scale =
        is_qnt ? scale_.at(index + output_per_branch) : 1.f;
    const size_t num_classes = conf_threshold_->size();

    uint16_t valid_count = 0;
    int grid_len = grid_w * grid_h;

    for (int i = 0; i < grid_h; ++i)
    {
        for (int j = 0; j < grid_w; ++j)
        {
            int offset = i * grid_w + j;

            // 逐类读 logits（NCHW 按 k*grid_len 跨步）→ sigmoid → 取最大类
            float max_score = 0.f;
            int max_class_id = -1;
            for (size_t k = 0; k < num_classes; ++k)
            {
                float logit = 0.f;
                if (model_format_ == ModelFormat::ONNX_FORMAT ||
                    model_format_ == ModelFormat::NNRT_FORMAT ||
                    model_format_ == ModelFormat::TRT_FORMAT)
                {
                    // 非量化：logit 来自 score（cls）张量
                    logit = score_tensor_float16[k * grid_len + offset];
                }
                else if (model_format_ == ModelFormat::RKNN_FORMAT)
                {

                    // RKNN 非对称量化：score 层 zp>0 实际以 uint8 存储，
                    // 必须按 (u8 - zp) × scale 反量化，否则高分字节被误读成负数。
                    logit = is_qnt
                                ? deqnt_affine_to_f32(score_tensor_int8[k * grid_len + offset],
                                                      score_zp, score_scale)
                                : fp16_to_f32(score_tensor_float16[k * grid_len + offset]);
                }
                float score = sigmoid(logit);
                if (score > conf_threshold_->at(k) &&
                    score > max_score)
                {
                    max_score = score;
                    max_class_id = (int)k;

                    // LOG_INFO("is_qnt = {}, logit: {}, sigmoid: {}", is_qnt, logit, sigmoid(logit));
                }
            }

            if (max_class_id != -1)
            {
                // 读 box [l, t, r, b]（NCHW 按通道跨步，RTMDet 无 DFL；
                // 通道顺序已核对 mmdet distance_point_bbox_coder）
                float lrtb[4];
                for (int k = 0; k < 4; ++k)
                {
                    if (model_format_ == ModelFormat::ONNX_FORMAT ||
                        model_format_ == ModelFormat::NNRT_FORMAT ||
                        model_format_ == ModelFormat::TRT_FORMAT)
                    {
                        lrtb[k] = box_tensor_float16[k * grid_len + offset];
                    }
                    else if (model_format_ == ModelFormat::RKNN_FORMAT)
                    {

                        // reg 层 zp<0 以 int8 存储；若 zp>0 则按 uint8
                        lrtb[k] = is_qnt
                                      ? deqnt_affine_to_f32(box_tensor_int8[k * grid_len + offset],
                                                            box_zp, box_scale)
                                      : fp16_to_f32(box_tensor_float16[k * grid_len + offset]);
                    }
                }

                // distance2bbox：中心 = j*stride, i*stride（RTMDet with_stride，无 +0.5）
                float cx = (float)j * stride;
                float cy = (float)i * stride;
                Bbox _bbox;
                _bbox.x1 = cx - lrtb[0]; // left
                _bbox.y1 = cy - lrtb[1]; // top
                _bbox.x2 = cx + lrtb[2]; // right
                _bbox.y2 = cy + lrtb[3]; // bottom

                if (_bbox.x2 <= _bbox.x1 || _bbox.y2 <= _bbox.y1)
                {
                    continue;
                }

                bboxes_.push_back(_bbox);
                class_id_.push_back(max_class_id);
                obj_probs_.push_back(max_score);
                valid_count++;
            }
        }
    }

    return valid_count;
}

uint16_t PostProcess::ProcessPose(const void *box_tensor,
                                  const void *score_tensor,
                                  const void *kpt_tensor,
                                  const void *visibility_tensor, int grid_w,
                                  int grid_h, int stride, int index,
                                  int output_per_branch)
{
    const float *box_tensor_float = reinterpret_cast<const float *>(box_tensor);
    const float *score_tensor_float =
        reinterpret_cast<const float *>(score_tensor);
    const float *kpt_tensor_float = reinterpret_cast<const float *>(kpt_tensor);
    const float *visibility_tensor_float =
        reinterpret_cast<const float *>(visibility_tensor);
    const int8_t *box_tensor_int8 = reinterpret_cast<const int8_t *>(box_tensor);
    const int8_t *score_tensor_int8 =
        reinterpret_cast<const int8_t *>(score_tensor);
    const int8_t *kpt_tensor_int8 = reinterpret_cast<const int8_t *>(kpt_tensor);
    const int8_t *visibility_tensor_int8 =
        reinterpret_cast<const int8_t *>(visibility_tensor);
    bool is_qnt = zero_points_.empty() ? false : true;
    int8_t score_thres_i8 =
        is_qnt ? qnt_f32_to_affine(conf_threshold_->at(0),
                                   zero_points_.at(index * output_per_branch + 1),
                                   scale_.at(index * output_per_branch + 1))
               : 0;
    uint16_t valid_count = 0;
    int grid_len = grid_w * grid_h;
    for (int i = 0; i < grid_h; ++i)
    {
        for (int j = 0; j < grid_w; ++j)
        {
            int offset = i * grid_w + j;
            float max_score_float = 0;
            int8_t max_score_int8 =
                is_qnt ? qnt_f32_to_affine(
                             0.0f, zero_points_.at(index * output_per_branch + 1),
                             scale_.at(index * output_per_branch + 1))
                       : 0;
            int max_class_id = -1;
            // 因为pose这个模型只是检测人的类型，所以只有一个种类
            if (model_format_ == ModelFormat::TRT_FORMAT ||
                model_format_ == ModelFormat::ONNX_FORMAT ||
                model_format_ == ModelFormat::NNRT_FORMAT)
            {
                if (score_tensor_float[offset] > conf_threshold_->at(0) &&
                    score_tensor_float[offset] > max_score_float)
                {
                    max_score_float = score_tensor_float[offset];
                    max_class_id = 0;
                }
            }
            else if (model_format_ == ModelFormat::RKNN_FORMAT)
            {
                if (score_tensor_int8[offset] > score_thres_i8 &&
                    score_tensor_int8[offset] > max_score_int8)
                {
                    max_score_int8 = score_tensor_int8[offset];
                    max_class_id = 0;
                }
            }
            if (max_class_id != -1)
            {
                offset = i * grid_w + j;
                float box[4];
                float before_dfl[dfl_len_ * 4];
                for (int k = 0; k < dfl_len_ * 4; ++k)
                {
                    if (model_format_ == ModelFormat::ONNX_FORMAT ||
                        model_format_ == ModelFormat::NNRT_FORMAT ||
                        model_format_ == ModelFormat::TRT_FORMAT)
                    {
                        before_dfl[k] = box_tensor_float[offset];
                    }
                    else if (model_format_ == ModelFormat::RKNN_FORMAT)
                    {
                        before_dfl[k] =
                            is_qnt ? deqnt_affine_to_f32(
                                         box_tensor_int8[offset],
                                         zero_points_.at(index * output_per_branch),
                                         scale_.at(index * output_per_branch))
                                   : 0;
                    }
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len_, box);
                Bbox _bbox;
                _bbox.x1 = (-box[0] + j + 0.5) * stride;
                _bbox.y1 = (-box[1] + i + 0.5) * stride;
                _bbox.x2 = (box[2] + j + 0.5) * stride;
                _bbox.y2 = (box[3] + i + 0.5) * stride;
                bboxes_.push_back(_bbox);
                class_id_.push_back(max_class_id);
                if (model_format_ == ModelFormat::ONNX_FORMAT ||
                    model_format_ == ModelFormat::NNRT_FORMAT ||
                    model_format_ == ModelFormat::TRT_FORMAT)
                {
                    obj_probs_.push_back(max_score_float);
                    offset = i * grid_w + j;
                    for (int k = 0; k < 17; ++k)
                    {
                        auto kpt_x = *(kpt_tensor_float + offset + 2 * k * grid_len);
                        auto kpt_y = *(kpt_tensor_float + offset + (2 * k + 1) * grid_len);
                        auto kpt_visibility =
                            *(visibility_tensor_float + offset + k * grid_len);
                        kpt.push_back(kpt_x);
                        kpt.push_back(kpt_y);
                        visibilities.push_back(kpt_visibility);
                    }
                }
                else if (model_format_ == ModelFormat::RKNN_FORMAT)
                {
                    auto max_score =
                        is_qnt ? deqnt_affine_to_f32(
                                     max_score_int8,
                                     zero_points_.at(index * output_per_branch + 1),
                                     scale_.at(index * output_per_branch + 1))
                               : 0;
                    obj_probs_.push_back(max_score);
                    offset = i * grid_w + j;
                    for (int k = 0; k < 17; ++k)
                    {
                        auto kpt_x = *(kpt_tensor_int8 + offset + 2 * k * grid_len);
                        auto kpt_y = *(kpt_tensor_int8 + offset + (2 * k + 1) * grid_len);
                        auto kpt_visibility =
                            *(visibility_tensor_int8 + offset + k * grid_len);
                        auto x =
                            is_qnt
                                ? deqnt_affine_to_f32(
                                      kpt_x, zero_points_.at(index * output_per_branch + 2),
                                      scale_.at(index * output_per_branch + 2))
                                : 0;
                        auto y =
                            is_qnt
                                ? deqnt_affine_to_f32(
                                      kpt_y, zero_points_.at(index * output_per_branch + 2),
                                      scale_.at(index * output_per_branch + 2))
                                : 0;
                        auto visibility =
                            is_qnt ? deqnt_affine_to_f32(
                                         kpt_visibility,
                                         zero_points_.at(index * output_per_branch + 3),
                                         scale_.at(index * output_per_branch + 3))
                                   : 0;
                        kpt.push_back(x);
                        kpt.push_back(y);
                        visibilities.push_back(visibility);
                    }
                }
                valid_count++;
            }
        }
    }
    return valid_count;
}

uint16_t PostProcess::ProcessDetect(const void *box_tensor,
                                    const void *score_tensor,
                                    const void *sum_score_tensor,
                                    int grid_w, int grid_h, int stride,
                                    int index, int output_per_branch)
{
    const float *box_tensor_float = reinterpret_cast<const float *>(box_tensor);
    const float *score_tensor_float =
        reinterpret_cast<const float *>(score_tensor);
    const float *sum_score_tensor_float =
        reinterpret_cast<const float *>(sum_score_tensor);
    const int8_t *box_tensor_int8 = reinterpret_cast<const int8_t *>(box_tensor);
    const int8_t *score_tensor_int8 =
        reinterpret_cast<const int8_t *>(score_tensor);
    const int8_t *sum_score_tensor_int8 =
        reinterpret_cast<const int8_t *>(sum_score_tensor);
    bool is_qnt = zero_points_.empty() ? false : true;
    int8_t score_sum_thres_i8 =
        is_qnt ? qnt_f32_to_affine(sum_conf_threshold_,
                                   zero_points_.at(index * output_per_branch + 2),
                                   scale_.at(index * output_per_branch + 2))
               : 0;

    uint16_t valid_count = 0;
    int grid_len = grid_w * grid_h;
    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; ++j)
        {
            int offset = i * grid_w + j;
            if (model_format_ == ModelFormat::ONNX_FORMAT ||
                model_format_ == ModelFormat::TRT_FORMAT ||
                model_format_ == ModelFormat::NNRT_FORMAT)
            {
                if (sum_score_tensor_float[offset] < sum_conf_threshold_)
                {
                    continue;
                }
            }
            else if (model_format_ == ModelFormat::RKNN_FORMAT)
            {
                if (sum_score_tensor_int8[offset] < score_sum_thres_i8)
                {
                    continue;
                }
            }
            float max_score_float = 0;
            int8_t max_score_int8 =
                is_qnt ? qnt_f32_to_affine(
                             0.0f, zero_points_.at(index * output_per_branch + 1),
                             scale_.at(index * output_per_branch + 1))
                       : 0;
            int max_class_id = -1;
            for (size_t k = 0; k < conf_threshold_->size(); ++k)
            {
                if (model_format_ == ModelFormat::ONNX_FORMAT ||
                    model_format_ == ModelFormat::TRT_FORMAT ||
                    model_format_ == ModelFormat::NNRT_FORMAT)
                {
                    if (score_tensor_float[offset] > conf_threshold_->at(k) &&
                        score_tensor_float[offset] > max_score_float)
                    {
                        max_score_float = score_tensor_float[offset];
                        max_class_id = k;
                    }
                }
                else if (model_format_ == ModelFormat::RKNN_FORMAT)
                {
                    auto score_thres_i8 =
                        is_qnt ? qnt_f32_to_affine(
                                     conf_threshold_->at(k),
                                     zero_points_.at(index * output_per_branch + 1),
                                     scale_.at(index * output_per_branch + 1))
                               : 0;
                    if (score_tensor_int8[offset] > score_thres_i8 &&
                        score_tensor_int8[offset] > max_score_float)
                    {
                        max_score_int8 = score_tensor_int8[offset];
                        max_class_id = k;
                    }
                }
                offset += grid_len;
            }
            if (max_class_id != -1)
            {
                offset = i * grid_w + j;
                float box[4];
                float before_dfl[dfl_len_ * 4];
                for (int k = 0; k < dfl_len_ * 4; ++k)
                {
                    if (model_format_ == ModelFormat::ONNX_FORMAT ||
                        model_format_ == ModelFormat::TRT_FORMAT ||
                        model_format_ == ModelFormat::NNRT_FORMAT)
                    {
                        before_dfl[k] = box_tensor_float[offset];
                    }
                    else if (model_format_ == ModelFormat::RKNN_FORMAT)
                    {
                        before_dfl[k] =
                            is_qnt ? deqnt_affine_to_f32(
                                         box_tensor_int8[offset],
                                         zero_points_.at(index * output_per_branch),
                                         scale_.at(index * output_per_branch))
                                   : 0;
                    }
                    offset += grid_len;
                }
                Bbox _bbox;
                if (model_type_ == ModelType::DETECTION_V26 ||
                    model_type_ == ModelType::DETECTION_V10)
                {
                    _bbox.x1 = (-before_dfl[0] + j + 0.5) * stride;
                    _bbox.y1 = (-before_dfl[1] + i + 0.5) * stride;
                    _bbox.x2 = (before_dfl[2] + j + 0.5) * stride;
                    _bbox.y2 = (before_dfl[3] + i + 0.5) * stride;
                }
                else
                {
                    compute_dfl(before_dfl, dfl_len_, box);
                    _bbox.x1 = (-box[0] + j + 0.5) * stride;
                    _bbox.y1 = (-box[1] + i + 0.5) * stride;
                    _bbox.x2 = (box[2] + j + 0.5) * stride;
                    _bbox.y2 = (box[3] + i + 0.5) * stride;
                }
                int width_pixel_delta = 1;
                int height_pixel_delta = 1;
                if (model_type_ == ModelType::SEGMENT_V11)
                {
                    height_pixel_delta = model_height_ / seg_height_;
                    width_pixel_delta = model_width_ / seg_width_;
                }
                if (std::abs(_bbox.x1 - _bbox.x2) < width_pixel_delta)
                {
                    LOG_WARN("bbox width is too small: {}",
                             std::abs(_bbox.x1 - _bbox.x2));
                    continue;
                }
                if (std::abs(_bbox.y1 - _bbox.y2) < height_pixel_delta)
                {
                    LOG_WARN("bbox height is too small: {}",
                             std::abs(_bbox.y1 - _bbox.y2));
                    continue;
                }
                bboxes_.push_back(_bbox);
                if (model_format_ == ModelFormat::ONNX_FORMAT ||
                    model_format_ == ModelFormat::TRT_FORMAT ||
                    model_format_ == ModelFormat::NNRT_FORMAT)
                {
                    obj_probs_.push_back(max_score_float);
                }
                else if (model_format_ == ModelFormat::RKNN_FORMAT)
                {
                    auto max_score =
                        is_qnt ? deqnt_affine_to_f32(
                                     max_score_int8,
                                     zero_points_.at(index * output_per_branch + 1),
                                     scale_.at(index * output_per_branch + 1))
                               : 0;
                    obj_probs_.push_back(max_score);
                    //                              LOG_INFO("max_score: {} {},
                    //                              id: {}, box[{} {} {} {}]",
                    //                                                 max_score,
                    //                                                 max_score_int8,
                    //                                                 max_class_id,
                    //                                                 _bbox.x1, _bbox.y1,
                    //                                                 _bbox.x2,
                    //                                                 _bbox.y2);
                }
                class_id_.push_back(max_class_id);
                bboxes_idx_.push_back({index, i * grid_w + j, grid_len});
                valid_count++;
            }
        }
    }
    return valid_count;
}