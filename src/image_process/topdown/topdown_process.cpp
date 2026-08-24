#include "topdown_process.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef RK3588
#include <cstdio>
#endif

TopdownProcess::TopdownProcess(const ai_framework::Config &detect_config,
                               const ai_framework::Config &keypoint_config,
                               float detect_sum_conf_threshold,
                               float iou_threshold,
                               bool debug)
    : debug_(debug)
{
    detect_model_format_ = detect_config.model_format;
    keypoint_model_format_ = keypoint_config.model_format;
    detect_num_of_layers_ = detect_config.output_tensors_count;
    keypoints_num_of_layers_ = keypoint_config.output_tensors_count;
    detect_sum_conf_threshold_ = detect_sum_conf_threshold;
    iou_threshold_ = iou_threshold;
    GetInputSize(detect_config, detect_input_width_, detect_input_height_);
    GetInputSize(keypoint_config, keypoint_input_width_, keypoint_input_height_);

    for (size_t i = 0; i < detect_num_of_layers_; ++i)
    {
        auto layer = detect_config.output_index_to_name.at(i);
        // LOG_INFO("Post Index = {}, layer name = {}", i, layer);
        detect_output_layer_names_.push_back(layer);
        detect_output_layer_shape.push_back(detect_config.output_layer_shape.at(layer));
        detect_output_element_count_.push_back(detect_config.output_element_count.at(layer));
        if (detect_config.zero_point.find(layer) != detect_config.zero_point.end())
        {
            detect_zero_points_.push_back(detect_config.zero_point.at(layer));
        }
        if (detect_config.scale.find(layer) != detect_config.scale.end())
        {
            detect_scale_.push_back(detect_config.scale.at(layer));
        }
    }

    for (size_t i = 0; i < keypoint_config.output_tensors_count; ++i)
    {
        auto layer = keypoint_config.output_index_to_name.at(i);
        // LOG_INFO("Post Index = {}, layer name = {}", i, layer);
        keypoints_output_layer_names_.push_back(layer);
        keypoints_output_layer_shape.push_back(keypoint_config.output_layer_shape.at(layer));
        keypoints_output_element_count_.push_back(keypoint_config.output_element_count.at(layer));
        if (keypoint_config.zero_point.find(layer) != keypoint_config.zero_point.end())
        {
            keypoints_zero_points_.push_back(keypoint_config.zero_point.at(layer));
        }
        if (keypoint_config.scale.find(layer) != keypoint_config.scale.end())
        {
            keypoints_scale_.push_back(keypoint_config.scale.at(layer));
        }
    }

    auto output_boxes_shape =
        detect_config.output_layer_shape.at(detect_config.output_index_to_name.at(0));
    dfl_len_ = output_boxes_shape.at(1) / 4;
    auto output_boxes_name = detect_config.output_index_to_name.at(0);

    auto output_keypoints_name = keypoint_config.output_index_to_name.at(0);

    InitModelType(output_boxes_name, detect_model_type_);
    InitModelType(output_keypoints_name, keypoint_model_type_);
}

TopdownProcess::~TopdownProcess()
{
}

void TopdownProcess::InitModelType(const std::string &output_name, ModelType &modeltype_)
{
    if (ContainsSubString(output_name, "yolo26_detect"))
    {
        modeltype_ = ModelType::DETECTION_V26;
    }
    else if (ContainsSubString(output_name, "yolov10"))
    {
        modeltype_ = ModelType::DETECTION_V10;
    }
    else if (ContainsSubString(output_name, "yolov8_detect"))
    {
        modeltype_ = ModelType::DETECTION_V8;
    }
    else if (ContainsSubString(output_name, "yolov8_pose") ||
             ContainsSubString(output_name, "yolo11_pose"))
    {
        modeltype_ = ModelType::POSE_V8;
    }
    else if (ContainsSubString(output_name, "yolo13_detect"))
    {
        modeltype_ = ModelType::DETECTION_V13;
    }
    else if (ContainsSubString(output_name, "yolo11_detect"))
    {
        modeltype_ = ModelType::DETECTION_V11;
    }
    else if (ContainsSubString(output_name, "rtmdet") ||
             ContainsSubString(output_name, "rtmde") ||
             ContainsSubString(output_name, "rtm_"))
    {
        modeltype_ = ModelType::DETECTION_RTMDE;
    }
    else if (ContainsSubString(output_name, "rtmpose") ||
             ContainsSubString(output_name, "rtmp") ||
             ContainsSubString(output_name, "rtmpose_"))
    {
        modeltype_ = ModelType::POSE_RTMPOSE;
    }

    // LOG_INFO("ModelType: {}", modeltype_);
}

void TopdownProcess::GetInputSize(const ai_framework::Config &config,
                                  int &width,
                                  int &height)
{
    const auto input_name = config.input_index_to_name.at(0);
    const auto &input_shape = config.input_layer_shape.at(input_name);

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
        height = static_cast<int>(input_shape.at(1));
        width = static_cast<int>(input_shape.at(2));
    }
    else
    {
        width = static_cast<int>(input_shape.at(3));
        height = static_cast<int>(input_shape.at(2));
    }
}

int TopdownProcess::GetInputSideLength(const ai_framework::Config &config)
{
    int width = 0;
    int height = 0;
    GetInputSize(config, width, height);
    return std::max(width, height);
}

// detect letterbox
void TopdownProcess::PreProcess(const std::vector<cv::Mat> &input, void *tensors[])
{
    for (size_t i = 0; i < input.size(); ++i)
    {
        auto &original_input = input.at(i);
        int resize_width = detect_input_width_;
        int resize_height = detect_input_height_;
        if (original_input.cols >= original_input.rows)
        {
            float scale = 1.0f * original_input.cols / detect_input_height_;
            resize_height = original_input.rows / scale;
        }
        else
        {
            float scale = 1.0f * original_input.rows / detect_input_width_;
            resize_width = original_input.cols / scale;
        }
        // 记录 letterbox 缩放因子（实际 resize 后宽 / 原图宽，等比缩放），
        // 供后处理把检测框从模型输入坐标系反算回原图坐标。
        // 内容为左上角对齐（无平移偏移），反算只需除以该缩放因子。
        detect_letterbox_scale_ =
            1.0f * resize_width / original_input.cols;
#ifdef RK3588
        // RGA3 版本：imfill 填充背景 114（RGB 灰），improcess 一次性完成
        // resize + BGR→RGB + 写入左上角区域。全程走 RGA3（有 IOMMU，支持
        // 用户态虚拟地址）；makeBorder 仅 RGA2 支持、且 RGA2 无 IOMMU，故不用。
        const size_t out_size =
            (size_t)detect_input_width_ * detect_input_height_ * 3;
        rga_buffer_handle_t src_handle = importbuffer_virtualaddr(
            (void *)original_input.data,
            (int)(original_input.step * original_input.rows));
        rga_buffer_handle_t out_handle =
            importbuffer_virtualaddr(tensors[i], (int)out_size);
        if (!src_handle || !out_handle)
        {
            fprintf(stderr,
                    "[PreProcess] importbuffer_virtualaddr failed (src=%d "
                    "out=%d) tensors[%zu]=%p out_size=%zu src_size=%d\n",
                    (int)src_handle, (int)out_handle, i, tensors[i], out_size,
                    (int)(original_input.step * original_input.rows));
            return;
        }
        rga_buffer_t src_rga = wrapbuffer_handle(
            src_handle, original_input.cols, original_input.rows,
            RK_FORMAT_BGR_888);
        rga_buffer_t out_rga = wrapbuffer_handle(
            out_handle, detect_input_width_, detect_input_height_,
            RK_FORMAT_RGB_888);

        // Step 1: 背景填充 RGB(114,114,114)
        // 注意：imfill 底层走 RGA_COLORFILL 旧接口，在 RGA3 上不支持
        // （RGA_COLORFILL fail: Invalid argument），故背景用 CPU memset
        // 填充（307KB，开销可忽略），缩放仍由 RGA3 improcess 完成。
        memset(tensors[i], 114, out_size);

        // Step 2: 把原图（保持宽高比）缩放到左上角区域，同时 BGR→RGB。
        // improcess 只更新 drect 区域，其余保持 Step 1 的背景色。
        im_opt_t opt = {};
        opt.core = IM_SCHEDULER_RGA3_CORE0; // 强制走 RGA3
        im_rect srect = {0, 0, original_input.cols, original_input.rows};
        im_rect drect = {0, 0, resize_width, resize_height};
        IM_STATUS st_proc = improcess(src_rga, out_rga, rga_buffer_t(), srect,
                                      drect, im_rect(), -1, nullptr, &opt, 0);
        if (st_proc != IM_STATUS_SUCCESS)
        {
            fprintf(stderr, "[PreProcess] improcess failed, status=%d\n",
                    (int)st_proc);
            return;
        }
        releasebuffer_handle(src_handle);
        releasebuffer_handle(out_handle);

        if (debug_)
        {
            cv::Mat debug_mat(detect_input_width_, detect_input_height_,
                              CV_8UC3, tensors[i]);
            cv::imshow("PreProcess Image", debug_mat);
            cv::waitKey(1);
        }
#else
        cv::Mat dst;
        cv::resize(original_input, dst, cv::Size(resize_width, resize_height),
                   cv::INTER_NEAREST);
        cv::Mat res;
        MakeSquare(dst, res);
        PopulateData(res, reinterpret_cast<float *>(tensors[i]));
        if (debug_)
        {
            cv::imshow("PreProcess Image", dst);
            cv::waitKey(1);
        }
#endif
    }
}

void TopdownProcess::CropImageByDetectBox(const std::vector<cv::Mat> &inputs,
                                          const std::vector<Result> &bboxs,
                                          void *tensors[],
                                          std::vector<TopdownMeta> &metainfo)
{
    const int out_w = keypoint_input_width_;
    const int out_h = keypoint_input_height_;
    // 关键点模型输入宽高比（与 mmpose RTMPose-Deploy 示例一致）
    const float aspect_ratio = static_cast<float>(out_w) / static_cast<float>(out_h);
    // 检测框外扩系数
    const float bbox_scale = 1.2f;

    // 防御：检测框数量必须覆盖每个输入图，否则 bboxs[i] 越界读
    if (inputs.empty() || bboxs.size() < inputs.size())
    {
        LOG_WARN("CropImageByDetectBox: invalid bboxs/inputs size (bboxs={}, inputs={})",
                 bboxs.size(), inputs.size());
        return;
    }
    // 防御：调用方可能传入空 metainfo，先按输入图数量扩容，
    // 否则后面 metainfo[i].center 会对空 vector 越界写（破坏堆）
    if (metainfo.size() < inputs.size())
    {
        metainfo.resize(inputs.size());
    }

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        // const cv::Mat &image = inputs[i];
        const Result &bbox = bboxs[i];

        // 把检测框限制在图像范围内，避免越界
        int x1 = std::max(0, static_cast<int>(bbox.box.x1));
        int y1 = std::max(0, static_cast<int>(bbox.box.y1));
        int x2 = std::min(inputs[i].cols, static_cast<int>(bbox.box.x2));
        int y2 = std::min(inputs[i].rows, static_cast<int>(bbox.box.y2));
        x1 = std::min(x1, inputs[i].cols - 1);
        y1 = std::min(y1, inputs[i].rows - 1);
        x2 = std::max(x2, x1 + 1);
        y2 = std::max(y2, y1 + 1);

        // 检测框宽高 + 中心点（原图坐标）
        float box_width = static_cast<float>(x2 - x1);
        float box_height = static_cast<float>(y2 - y1);
        const float center_x = x1 + box_width * 0.5f;
        const float center_y = y1 + box_height * 0.5f;

        // 按关键点模型输入宽高比调整检测框宽高
        if (box_width > aspect_ratio * box_height)
        {
            box_height = box_width / aspect_ratio;
        }
        else if (box_width < aspect_ratio * box_height)
        {
            box_width = box_height * aspect_ratio;
        }

        // 中心点 + scale（外扩 1.2），作为 GetAffineTransform 的输入
        const float scale_image_width = box_width * bbox_scale;
        const float scale_image_height = box_height * bbox_scale;

        const cv::Mat affine = GetAffineTransform(center_x, center_y,
                                                  scale_image_width,
                                                  scale_image_height,
                                                  out_w, out_h);

        // 记录中心点与 scale（与 mmpose 的 center/scale 约定一致），
        // 供后处理用反向仿射把关键点映射回原图
        metainfo[i].center = cv::Point2f(center_x, center_y);
        metainfo[i].scale = cv::Size2f(scale_image_width, scale_image_height);

#ifdef RK3588
        // ---- RK3588：改用 OpenCV warpAffine 完成仿射变换（与 mmpose RTMPose-Deploy 一致）。
        // 之前用 RGA improcess：srect 填的是原图坐标，而 image_contig 是 ROI 局部坐标，
        // 坐标系错位导致 srect 越界、improcess 失败，输出只剩背景 114（纯灰）。
        // RGA 也无任意角度仿射 API，统一走 OpenCV 最稳妥。
        // 注意：RKNN 输入为 uint8 NHWC RGB，warpAffine 得到 BGR，需转 RGB 后直接拷贝，
        // 不能走 PopulateData（那是 float CHW，与 RKNN 输入类型不匹配）。
        cv::Mat bgr_dst;
        cv::warpAffine(inputs[i], bgr_dst, affine, cv::Size(out_w, out_h),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
        cv::Mat rgb_dst;
        cv::cvtColor(bgr_dst, rgb_dst, cv::COLOR_BGR2RGB);
        std::memcpy(tensors[i], rgb_dst.data, (size_t)out_w * out_h * 3);
        if (debug_)
        {
            // cv::imwrite("/home/orangepi/Code/ai_framework/source/result_test.jpg", bgr_dst);
            // 无显示环境下请注释下面两行（GTK 后端初始化可能触发 SIGABRT）
            cv::imshow("CropImageByDetectBox Image", bgr_dst);
            cv::waitKey(1);
        }
#else
        // ---- 非 RK3588：OpenCV warpAffine（与 mmpose RTMPose-Deploy 一致） ----
        cv::Mat dst;
        cv::warpAffine(inputs[i], dst, affine, cv::Size(out_w, out_h),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
        PopulateData(dst, reinterpret_cast<float *>(tensors[i]));
        if (debug_)
        {
            cv::imshow("CropImageByDetectBox Image", dst);
            cv::waitKey(1);
        }
#endif
    }
}

cv::Mat TopdownProcess::GetAffineTransform(float center_x,
                                           float center_y,
                                           float scale_width,
                                           float scale_height,
                                           int output_image_width,
                                           int output_image_height,
                                           bool inverse)
{
    // solve the affine transformation matrix

    // get the three points corresponding to the source picture and the target picture
    cv::Point2f src_point_1;
    src_point_1.x = center_x;
    src_point_1.y = center_y;

    cv::Point2f src_point_2;
    src_point_2.x = center_x;
    src_point_2.y = center_y - scale_width * 0.5;

    cv::Point2f src_point_3;
    src_point_3.x = src_point_2.x - (src_point_1.y - src_point_2.y);
    src_point_3.y = src_point_2.y + (src_point_1.x - src_point_2.x);

    float alphapose_image_center_x = output_image_width / 2;
    float alphapose_image_center_y = output_image_height / 2;

    cv::Point2f dst_point_1;
    dst_point_1.x = alphapose_image_center_x;
    dst_point_1.y = alphapose_image_center_y;

    cv::Point2f dst_point_2;
    dst_point_2.x = alphapose_image_center_x;
    dst_point_2.y = alphapose_image_center_y - output_image_width * 0.5;

    cv::Point2f dst_point_3;
    dst_point_3.x = dst_point_2.x - (dst_point_1.y - dst_point_2.y);
    dst_point_3.y = dst_point_2.y + (dst_point_1.x - dst_point_2.x);

    cv::Point2f srcPoints[3];
    srcPoints[0] = src_point_1;
    srcPoints[1] = src_point_2;
    srcPoints[2] = src_point_3;

    cv::Point2f dstPoints[3];
    dstPoints[0] = dst_point_1;
    dstPoints[1] = dst_point_2;
    dstPoints[2] = dst_point_3;

    // get affine matrix
    cv::Mat affineTransform;
    if (inverse)
    {
        affineTransform = cv::getAffineTransform(dstPoints, srcPoints);
    }
    else
    {
        affineTransform = cv::getAffineTransform(srcPoints, dstPoints);
    }

    return affineTransform;
}

void TopdownProcess::MakeSquare(const cv::Mat &src, cv::Mat &dst)
{
    // 获取图像的宽和高
    int width = src.cols;
    int height = src.rows;
    // 计算需要填充的尺寸
    int border_left = 0;
    int border_right = 0;
    int border_top = 0;
    int border_bottom = 0;

    if (height > width)
    {
        // 左上角对齐：右侧填充
        border_right = height - width;
    }
    else
    {
        // 左上角对齐：底部填充
        border_bottom = width - height;
    }
    // 使用灰色(114,114,114)填充边缘
    cv::copyMakeBorder(src, dst, border_top, border_bottom, border_left,
                       border_right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
}

uint64_t TopdownProcess::PopulateData(
    const cv::Mat &data,
    float *dst,
    const std::vector<float> &mean,
    const std::vector<float> &std)
{
    // 1. 安全检查：校验图像类型通道数及均值/标准差参数有效性
    if (data.empty() || data.channels() != 3 || data.type() != CV_8UC3)
    {
        return 0;
    }
    if (mean.size() < 3 || std.size() < 3)
    {
        return 0;
    }

    const size_t total_pixels = data.total(); // 图像像素总数 (H * W)

    // 2. 计算输出内存中 R、G、B 平面的首地址 (CHW 排列模式)
    float *R_dst = dst;
    float *G_dst = dst + total_pixels;
    float *B_dst = dst + total_pixels * 2;

    // 3. 提取 RGB 的均值和标准差
    const float mean_r = mean[0], mean_g = mean[1], mean_b = mean[2];
    const float std_r = std[0], std_g = std[1], std_b = std[2];

    // 4. 优化遍历：如果内存连续，使用一维指针快速迭代
    if (data.isContinuous())
    {
        const uint8_t *src = data.ptr<uint8_t>();
        for (size_t i = 0; i < total_pixels; ++i)
        {
            // OpenCV 的内存存储顺序为 B, G, R
            uint8_t b = src[i * 3 + 0];
            uint8_t g = src[i * 3 + 1];
            uint8_t r = src[i * 3 + 2];

            // 归一化并按 CHW 分离存入对应的平面内存中
            R_dst[i] = (static_cast<float>(r) - mean_r) / std_r;
            G_dst[i] = (static_cast<float>(g) - mean_g) / std_g;
            B_dst[i] = (static_cast<float>(b) - mean_b) / std_b;
        }
    }
    else
    {
        // 内存不连续时的备选逐行处理
        for (int i = 0; i < data.rows; ++i)
        {
            const uint8_t *row_ptr = data.ptr<uint8_t>(i);
            for (int j = 0; j < data.cols; ++j)
            {
                uint8_t b = row_ptr[j * 3 + 0];
                uint8_t g = row_ptr[j * 3 + 1];
                uint8_t r = row_ptr[j * 3 + 2];

                *R_dst++ = (static_cast<float>(r) - mean_r) / std_r;
                *G_dst++ = (static_cast<float>(g) - mean_g) / std_g;
                *B_dst++ = (static_cast<float>(b) - mean_b) / std_b;
            }
        }
    }

    // 返回填充的总字节大小
    return total_pixels * data.channels() * sizeof(float);
}

void TopdownProcess::PostProcessRTMPose(void **&tensors, std::vector<TopdownMeta> &metainfo)
{
    // RTMPose 输出 simcc_x / simcc_y 两个头，shape 均为 [1, joint_num, extend]
    if (keypoints_output_layer_shape.size() < 2)
    {
        LOG_ERROR("PostProcessRTMPose: missing simcc_x/simcc_y output layers");
        return;
    }

    const int joint_num = static_cast<int>(keypoints_output_layer_shape.at(0).at(1));
    const int extend_width = static_cast<int>(keypoints_output_layer_shape.at(0).at(2));
    const int extend_height = static_cast<int>(keypoints_output_layer_shape.at(1).at(2));

    const float *simcc_x_f32 = reinterpret_cast<const float *>(tensors[0]);
    const float *simcc_y_f32 = reinterpret_cast<const float *>(tensors[1]);
    const uint8_t *simcc_x_int8 = reinterpret_cast<const uint8_t *>(tensors[0]);
    const uint8_t *simcc_y_int8 = reinterpret_cast<const uint8_t *>(tensors[1]);
    const uint16_t *simcc_x_fp16 = reinterpret_cast<const uint16_t *>(tensors[0]);
    const uint16_t *simcc_y_fp16 = reinterpret_cast<const uint16_t *>(tensors[1]);

    const bool is_qnt = !keypoints_zero_points_.empty();

    // 统一把某个 simcc 值解码成 float，避免三元表达式两侧指针类型不一致
    auto read_value = [&](int layer, const float *f32, const uint8_t *i8,
                          const uint16_t *f16, int offset) -> float
    {
        if (keypoint_model_format_ == ModelFormat::ONNX_FORMAT ||
            keypoint_model_format_ == ModelFormat::NNRT_FORMAT ||
            keypoint_model_format_ == ModelFormat::TRT_FORMAT)
        {
            return f32[offset];
        }
        // RKNN_FORMAT
        else if (keypoint_model_format_ == ModelFormat::RKNN_FORMAT)
        {
            return is_qnt ? deqnt_affine_u8_to_f32(i8[offset],
                                                   keypoints_zero_points_.at(layer),
                                                   keypoints_scale_.at(layer))
                          : fp16_to_f32(f16[offset]);
        }
        return 0.0f; // 枚举已全覆盖，仅为消除“非所有路径返回”告警
    };

    // 把关键点追加到每个检测结果上：result_ 与 metainfo 一一对应
    // （CropImageByDetectBox 按 result_ 顺序填充 metainfo）。
    // 注意不能 push_back 新的空 Result（box/class_id 未初始化，会破坏后续绘制）。
    const size_t num_boxes = std::min(metainfo.size(), result_.size());
    const float out_w = static_cast<float>(keypoint_input_width_);
    const float out_h = static_cast<float>(keypoint_input_height_);

    for (size_t b = 0; b < num_boxes; ++b)
    {
        Result &res = result_[b];
        res.key_points.clear();
        res.key_points.reserve(joint_num);
        // 标记为姿态结果，GetImageResult 据此绘制关键点
        res.model_type = ModelType::POSE_RTMPOSE;

        // 反向仿射：把模型输入坐标映射回原图。
        // GetAffineTransform 的正向是均匀缩放 s = scale.width / input_w，
        // 逆向 = (coord - input/2) * scale.width / input_w + center。
        const float inv_scale = metainfo[b].scale.width / out_w;
        const float center_x = metainfo[b].center.x;
        const float center_y = metainfo[b].center.y;

        for (int i = 0; i < joint_num; ++i)
        {
            // x 方向：在 simcc_x 第 i 个关节的 extend_width 个值里找最大值
            int max_x_pos = 0;
            float score_x = read_value(0, simcc_x_f32, simcc_x_int8, simcc_x_fp16,
                                       i * extend_width);
            for (int j = 1; j < extend_width; ++j)
            {
                float v = read_value(0, simcc_x_f32, simcc_x_int8, simcc_x_fp16,
                                     i * extend_width + j);
                if (v > score_x)
                {
                    score_x = v;
                    max_x_pos = j;
                }
            }

            // y 方向：在 simcc_y 第 i 个关节的 extend_height 个值里找最大值
            int max_y_pos = 0;
            float score_y = read_value(1, simcc_y_f32, simcc_y_int8, simcc_y_fp16,
                                       i * extend_height);
            for (int j = 1; j < extend_height; ++j)
            {
                float v = read_value(1, simcc_y_f32, simcc_y_int8, simcc_y_fp16,
                                     i * extend_height + j);
                if (v > score_y)
                {
                    score_y = v;
                    max_y_pos = j;
                }
            }

            // simcc 编码步长为 2，除以 2 得到模型输入坐标系下的像素坐标
            int pose_x = max_x_pos / 2;
            int pose_y = max_y_pos / 2;
            const float score = (score_x + score_y) * 0.5;

            KeyPoint pose_;
            pose_.x = (static_cast<float>(pose_x) - out_w * 0.5f) * inv_scale +
                      center_x;
            pose_.y = (static_cast<float>(pose_y) - out_h * 0.5f) * inv_scale +
                      center_y;
            pose_.visibility = score;

            res.key_points.push_back(pose_);
        }
    }
}

void TopdownProcess::PostProcessDetect(void **&tensors)
{
    result_.clear();
    bboxes_.clear();
    class_id_.clear();
    obj_probs_.clear();
    bboxes_idx_.clear();
    if (detect_model_type_ == ModelType::DETECTION_V8 ||
        detect_model_type_ == ModelType::DETECTION_V26 ||
        detect_model_type_ == ModelType::DETECTION_V10 ||
        detect_model_type_ == ModelType::DETECTION_V11 ||
        detect_model_type_ == ModelType::DETECTION_V13 ||
        detect_model_type_ == ModelType::SEGMENT_V11)
    {
        PostProcessDetect_(tensors);
    }
    else if (detect_model_type_ == ModelType::DETECTION_RTMDE)
    {
        PostProcessRtmdet(tensors);
    }

    // 把检测框从模型输入（letterbox 后）坐标系反算回原图坐标，
    // 保证 get_result() 返回的 Bbox 直接就是原图上的坐标
    // （CropImageByDetectBox / GetImageResult 均按原图坐标使用）。
    for (auto &res : result_)
    {
        RecoverBoxToOriginal(res.box);
    }
}

void TopdownProcess::RecoverBoxToOriginal(Bbox &bbox)
{
    // PreProcess 的 letterbox 为"等比缩放到模型输入 + 左上角对齐"，
    // 原图坐标 = 模型输入坐标 / scale（无平移偏移）。
    const float scale = detect_letterbox_scale_;
    if (scale <= 0.0f)
    {
        return;
    }
    bbox.x1 /= scale;
    bbox.y1 /= scale;
    bbox.x2 /= scale;
    bbox.y2 /= scale;
}

void TopdownProcess::PostProcessDetect_(void **&tensors)
{
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int output_per_branch = detect_num_of_layers_ / 3;
    for (int i = 0; i < 3; ++i)
    {
        int box_index = i * output_per_branch;
        int score_index = i * output_per_branch + 1;
        int sum_score_index = i * output_per_branch + 2;
        grid_h = detect_output_layer_shape.at(box_index).at(2);
        grid_w = detect_output_layer_shape.at(box_index).at(3);
        stride = detect_input_height_ / grid_h;
        validCount += ProcessDetect(tensors[box_index], tensors[score_index],
                                    tensors[sum_score_index], grid_w, grid_h,
                                    stride, i, output_per_branch);
    }
    if (validCount <= 0)
    {
        return;
    }
    std::vector<int> indexArray;
    if (detect_model_type_ == ModelType::DETECTION_V8 ||
        detect_model_type_ == ModelType::DETECTION_V11 ||
        detect_model_type_ == ModelType::DETECTION_V13 ||
        detect_model_type_ == ModelType::SEGMENT_V11)
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
        if (detect_model_type_ == ModelType::DETECTION_V8 ||
            detect_model_type_ == ModelType::DETECTION_V11 ||
            detect_model_type_ == ModelType::DETECTION_V13 ||
            detect_model_type_ == ModelType::SEGMENT_V11)
        {
            if (indexArray[i] == -1)
            {
                continue;
            }
            n = indexArray[i];
            res.model_type = detect_model_type_;
        }
        else if (detect_model_type_ == ModelType::DETECTION_V10 ||
                 detect_model_type_ == ModelType::DETECTION_V26)
        {
            n = i;
            res.model_type = detect_model_type_;
        }
        res.box = bboxes_.at(n);
        res.class_id = class_id_.at(n);
        // 上面快排的时候，元素有交换
        res.obj_prob = obj_probs_.at(i);
        result_.push_back(res);
    }
}

uint16_t TopdownProcess::ProcessDetect(const void *box_tensor,
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
    bool is_qnt = detect_zero_points_.empty() ? false : true;
    int8_t score_sum_thres_i8 =
        is_qnt ? qnt_f32_to_affine(detect_sum_conf_threshold_,
                                   detect_zero_points_.at(index * output_per_branch + 2),
                                   detect_scale_.at(index * output_per_branch + 2))
               : 0;

    uint16_t valid_count = 0;
    int grid_len = grid_w * grid_h;
    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; ++j)
        {
            int offset = i * grid_w + j;
            if (detect_model_format_ == ModelFormat::ONNX_FORMAT ||
                detect_model_format_ == ModelFormat::TRT_FORMAT ||
                detect_model_format_ == ModelFormat::NNRT_FORMAT)
            {
                if (sum_score_tensor_float[offset] < detect_sum_conf_threshold_)
                {
                    continue;
                }
            }
            else if (detect_model_format_ == ModelFormat::RKNN_FORMAT)
            {
                if (sum_score_tensor_int8[offset] < score_sum_thres_i8)
                {
                    continue;
                }
            }
            float max_score_float = 0;
            int8_t max_score_int8 =
                is_qnt ? qnt_f32_to_affine(
                             0.0f, detect_zero_points_.at(index * output_per_branch + 1),
                             detect_scale_.at(index * output_per_branch + 1))
                       : 0;
            int max_class_id = -1;
            // topdown 单类别检测：只判断 class 0，阈值为标量 detect_conf_threshold_
            if (detect_model_format_ == ModelFormat::ONNX_FORMAT ||
                detect_model_format_ == ModelFormat::TRT_FORMAT ||
                detect_model_format_ == ModelFormat::NNRT_FORMAT)
            {
                if (score_tensor_float[offset] > detect_conf_threshold_)
                {
                    max_score_float = score_tensor_float[offset];
                    max_class_id = 0;
                }
            }
            else if (detect_model_format_ == ModelFormat::RKNN_FORMAT)
            {
                auto score_thres_i8 =
                    is_qnt ? qnt_f32_to_affine(
                                 detect_conf_threshold_,
                                 detect_zero_points_.at(index * output_per_branch + 1),
                                 detect_scale_.at(index * output_per_branch + 1))
                           : 0;
                if (score_tensor_int8[offset] > score_thres_i8)
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
                    if (detect_model_format_ == ModelFormat::ONNX_FORMAT ||
                        detect_model_format_ == ModelFormat::TRT_FORMAT ||
                        detect_model_format_ == ModelFormat::NNRT_FORMAT)
                    {
                        before_dfl[k] = box_tensor_float[offset];
                    }
                    else if (detect_model_format_ == ModelFormat::RKNN_FORMAT)
                    {
                        before_dfl[k] =
                            is_qnt ? deqnt_affine_to_f32(
                                         box_tensor_int8[offset],
                                         detect_zero_points_.at(index * output_per_branch),
                                         detect_scale_.at(index * output_per_branch))
                                   : 0;
                    }
                    offset += grid_len;
                }
                Bbox _bbox;
                if (detect_model_type_ == ModelType::DETECTION_V26 ||
                    detect_model_type_ == ModelType::DETECTION_V10)
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
                if (detect_model_format_ == ModelFormat::ONNX_FORMAT ||
                    detect_model_format_ == ModelFormat::TRT_FORMAT ||
                    detect_model_format_ == ModelFormat::NNRT_FORMAT)
                {
                    obj_probs_.push_back(max_score_float);
                }
                else if (detect_model_format_ == ModelFormat::RKNN_FORMAT)
                {
                    auto max_score =
                        is_qnt ? deqnt_affine_to_f32(
                                     max_score_int8,
                                     detect_zero_points_.at(index * output_per_branch + 1),
                                     detect_scale_.at(index * output_per_branch + 1))
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

void TopdownProcess::PostProcessRtmdet(void **&tensors)
{
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    // RTMDet: first half outputs are cls scores, second half are bbox predictions
    int cls_layers = detect_num_of_layers_ / 2;

    for (int i = 0; i < cls_layers; ++i)
    {
        int score_index = i;            // cls tensor
        int box_index = i + cls_layers; // bbox tensor
        grid_h = detect_output_layer_shape.at(box_index).at(2);
        grid_w = detect_output_layer_shape.at(box_index).at(3);
        stride = detect_input_height_ / grid_h;
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

uint16_t TopdownProcess::ProcessRtmdet(const void *box_tensor, const void *score_tensor,
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

    bool is_qnt = detect_zero_points_.empty() ? false : true;
    const float score_zp = is_qnt ? detect_zero_points_.at(index) : 0.f;
    const float score_scale = is_qnt ? detect_scale_.at(index) : 1.f;
    const float box_zp =
        is_qnt ? detect_zero_points_.at(index + output_per_branch) : 0.f;
    const float box_scale =
        is_qnt ? detect_scale_.at(index + output_per_branch) : 1.f;
    const size_t num_classes = 1; // topdown 单类别检测：只取 class 0 的 logits

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
                if (detect_model_format_ == ModelFormat::ONNX_FORMAT ||
                    detect_model_format_ == ModelFormat::NNRT_FORMAT ||
                    detect_model_format_ == ModelFormat::TRT_FORMAT)
                {
                    // 非量化：logit 来自 score（cls）张量
                    logit = score_tensor_float16[k * grid_len + offset];
                }
                else if (detect_model_format_ == ModelFormat::RKNN_FORMAT)
                {

                    // RKNN 非对称量化：score 层 zp>0 实际以 uint8 存储，
                    // 必须按 (u8 - zp) × scale 反量化，否则高分字节被误读成负数。
                    logit = is_qnt
                                ? deqnt_affine_to_f32(score_tensor_int8[k * grid_len + offset],
                                                      score_zp, score_scale)
                                : fp16_to_f32(score_tensor_float16[k * grid_len + offset]);
                }
                float score = sigmoid(logit);
                if (score > detect_conf_threshold_ &&
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
                    if (detect_model_format_ == ModelFormat::ONNX_FORMAT ||
                        detect_model_format_ == ModelFormat::NNRT_FORMAT ||
                        detect_model_format_ == ModelFormat::TRT_FORMAT)
                    {
                        lrtb[k] = box_tensor_float16[k * grid_len + offset];
                    }
                    else if (detect_model_format_ == ModelFormat::RKNN_FORMAT)
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