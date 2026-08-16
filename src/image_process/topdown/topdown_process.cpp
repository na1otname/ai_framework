#include "topdown_process.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef RK3588
#include <cstdio>
#endif

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

// RKNN 非对称量化：zp>0 的层（如 RTMDet 的 score/cls 层）实际以 uint8 存储，
// 必须按 (u8 - zp) × scale 反量化，否则高分字节会被误读成负数 int8。
static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

// IEEE 754 half-precision (float16) → float32 转换
static float fp16_to_f32(uint16_t half)
{
    // 提取符号、指数、尾数
    uint32_t sign = (half & 0x8000u) << 16;
    uint32_t exp = (half >> 10) & 0x1fu;
    uint32_t mant = half & 0x3ffu;

    uint32_t f32;
    if (exp == 0)
    {
        // 零 / 次正规数
        if (mant == 0)
        {
            f32 = sign; // +/-0
        }
        else
        {
            // 次正规数 → 正规化
            while ((mant & 0x400u) == 0)
            {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            exp = 1 + (127 - 15);
            f32 = sign | (exp << 23) | (mant << 13);
        }
    }
    else if (exp == 0x1f)
    {
        // Inf / NaN
        f32 = sign | (0xffu << 23) | (mant << 13);
    }
    else
    {
        // 正规数
        exp = exp + (127 - 15);
        f32 = sign | (exp << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &f32, sizeof(float));
    return result;
}

TopdownProcess::TopdownProcess(const ai_framework::Config &detect_config,
                               const ai_framework::Config &keypoint_config,
                               std::vector<float> &detect_conf_threshold,
                               float detect_sum_conf_threshold,
                               float iou_threshold,
                               bool debug)
    : debug_(debug)
{
    model_format_ = detect_config.model_format;
    iou_threshold_ = iou_threshold;
    detect_num_of_layers_ = detect_config.output_tensors_count;
    keypoints_num_of_layers_ = keypoint_config.output_tensors_count;
    detect_target_side_length_ = GetInputSideLength(detect_config);
    GetInputSize(keypoint_config, keypoint_input_width_, keypoint_input_height_);

    for (size_t i = 0; i < detect_num_of_layers_; ++i)
    {
        auto layer = detect_config.output_index_to_name.at(i);
        LOG_INFO("Post Index = {}, layer name = {}", i, layer);
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
        LOG_INFO("Post Index = {}, layer name = {}", i, layer);
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
}

TopdownProcess::~TopdownProcess()
{
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

void TopdownProcess::Run(void **&tensors)
{
}

// detect letterbox
void TopdownProcess::PreProcess(const std::vector<cv::Mat> &input, void *tensors[])
{
    for (size_t i = 0; i < input.size(); ++i)
    {
        auto &original_input = input.at(i);
        int resize_width = detect_target_side_length_;
        int resize_height = detect_target_side_length_;
        if (original_input.cols >= original_input.rows)
        {
            float scale = 1.0f * original_input.cols / detect_target_side_length_;
            resize_height = original_input.rows / scale;
        }
        else
        {
            float scale = 1.0f * original_input.rows / detect_target_side_length_;
            resize_width = original_input.cols / scale;
        }
#ifdef RK3588
        // RGA3 版本：imfill 填充背景 114（RGB 灰），improcess 一次性完成
        // resize + BGR→RGB + 写入左上角区域。全程走 RGA3（有 IOMMU，支持
        // 用户态虚拟地址）；makeBorder 仅 RGA2 支持、且 RGA2 无 IOMMU，故不用。
        const size_t out_size =
            (size_t)detect_target_side_length_ * detect_target_side_length_ * 3;
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
            out_handle, detect_target_side_length_, detect_target_side_length_,
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
            cv::Mat debug_mat(detect_target_side_length_, detect_target_side_length_,
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

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        const cv::Mat &image = inputs[i];
        const Result &bbox = bboxs[i];

        // 把检测框限制在图像范围内，避免越界
        int x1 = std::max(0, static_cast<int>(bbox.box.x1));
        int y1 = std::max(0, static_cast<int>(bbox.box.y1));
        int x2 = std::min(image.cols, static_cast<int>(bbox.box.x2));
        int y2 = std::min(image.rows, static_cast<int>(bbox.box.y2));
        x1 = std::min(x1, image.cols - 1);
        y1 = std::min(y1, image.rows - 1);
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
        // ---- RK3588：用 RGA 完成仿射变换 ----
        // RGA 无任意角度仿射 API，但 RTMPose 的 GetAffineTransform 在 rot=0 时
        // 退化为"等比缩放 + 平移"（s = out_w / scale_w），源矩形 scale_w×scale_h
        // 映射到整个输出，可用 improcess 的 srect + drect 精确表达。
        // 源矩形可能越界（框靠近图像边缘），先裁剪到图像内，并同步换算 drect。
        cv::Mat image_contig;
        if (image.isContinuous())
        {
            image_contig = image;
        }
        else
        {
            image_contig = image.clone();
        }

        const size_t tensor_bytes = (size_t)out_w * out_h * 3;
        rga_buffer_handle_t src_handle = importbuffer_virtualaddr(
            (void *)image_contig.data,
            (int)(image_contig.step * image_contig.rows));
        rga_buffer_handle_t out_handle =
            importbuffer_virtualaddr(tensors[i], (int)tensor_bytes);
        if (!src_handle || !out_handle)
        {
            fprintf(stderr,
                    "[CropImageByDetectBox] importbuffer_virtualaddr failed "
                    "(src=%d out=%d) tensors[%zu]=%p tensor_bytes=%zu\n",
                    (int)src_handle, (int)out_handle, i, tensors[i],
                    tensor_bytes);
            return;
        }
        rga_buffer_t src_rga = wrapbuffer_handle(
            src_handle, image_contig.cols, image_contig.rows,
            RK_FORMAT_BGR_888);
        rga_buffer_t out_rga = wrapbuffer_handle(
            out_handle, out_w, out_h, RK_FORMAT_RGB_888);

        // Step 1: 背景填充 RGB(114,114,114)。imfill 在 RGA3 不可用，用 CPU memset。
        memset(tensors[i], 114, tensor_bytes);

        // Step 2: 源矩形（仿射的源区域）裁剪到图像内，drect 按统一缩放 s 换算。
        const float s = static_cast<float>(out_w) / scale_image_width;
        const float src_x0 = center_x - scale_image_width * 0.5f;
        const float src_y0 = center_y - scale_image_height * 0.5f;
        const int sx = std::max(0, static_cast<int>(std::lround(src_x0)));
        const int sy = std::max(0, static_cast<int>(std::lround(src_y0)));
        const int sw = std::min(static_cast<int>(std::lround(scale_image_width)),
                                image_contig.cols - sx);
        const int sh = std::min(static_cast<int>(std::lround(scale_image_height)),
                                image_contig.rows - sy);
        const int dx = std::max(0, static_cast<int>(std::lround((sx - src_x0) * s)));
        const int dy = std::max(0, static_cast<int>(std::lround((sy - src_y0) * s)));
        const int dw = std::min(static_cast<int>(std::lround(sw * s)), out_w - dx);
        const int dh = std::min(static_cast<int>(std::lround(sh * s)), out_h - dy);

        im_rect srect = {sx, sy, sw, sh};
        im_rect drect = {dx, dy, dw, dh};

        im_opt_t opt = {};
        opt.core = IM_SCHEDULER_RGA3_CORE0; // 强制走 RGA3（有 IOMMU）
        IM_STATUS st = improcess(src_rga, out_rga, rga_buffer_t(), srect,
                                 drect, im_rect(), -1, nullptr, &opt, 0);
        if (st != IM_STATUS_SUCCESS)
        {
            fprintf(stderr,
                    "[CropImageByDetectBox] improcess failed, status=%d\n",
                    (int)st);
        }
        releasebuffer_handle(src_handle);
        releasebuffer_handle(out_handle);

        if (debug_)
        {
            // tensors[i] 为 RGB 顺序，imshow 按 BGR 解释，仅作调试参考
            cv::Mat debug_mat(out_h, out_w, CV_8UC3, tensors[i]);
            cv::imshow("CropImageByDetectBox Image", debug_mat);
            cv::waitKey(1);
        }
#else
        // ---- 非 RK3588：OpenCV warpAffine（与 mmpose RTMPose-Deploy 一致） ----
        cv::Mat dst;
        cv::warpAffine(image, dst, affine, cv::Size(out_w, out_h),
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

    Result tmp_;
    // 统一把某个 simcc 值解码成 float，避免三元表达式两侧指针类型不一致
    auto read_value = [&](int layer, const float *f32, const uint8_t *i8,
                          const uint16_t *f16, int offset) -> float
    {
        if (model_format_ == ModelFormat::ONNX_FORMAT ||
            model_format_ == ModelFormat::NNRT_FORMAT ||
            model_format_ == ModelFormat::TRT_FORMAT)
        {
            return f32[offset];
        }
        // RKNN_FORMAT
        else if (model_format_ == ModelFormat::RKNN_FORMAT)
        {
            return is_qnt ? deqnt_affine_u8_to_f32(i8[offset],
                                                   keypoints_zero_points_.at(layer),
                                                   keypoints_scale_.at(layer))
                          : fp16_to_f32(f16[offset]);
        }
        return 0.0f; // 枚举已全覆盖，仅为消除“非所有路径返回”告警
    };

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

        // 反向仿射：把模型输入坐标 (pose_x, pose_y) 映射回原图。
        // GetAffineTransform 的正向是均匀缩放 s = scale.width / input_w，
        // 逆向 = (coord - input/2) * scale.width / input_w + center。
        const float out_w = static_cast<float>(keypoint_input_width_);
        const float out_h = static_cast<float>(keypoint_input_height_);
        const float inv_scale = metainfo[i].scale.width / out_w;

        KeyPoint pose_;
        pose_.x = (static_cast<float>(pose_x) - out_w * 0.5f) * inv_scale +
                  metainfo[i].center.x;
        pose_.y = (static_cast<float>(pose_y) - out_h * 0.5f) * inv_scale +
                  metainfo[i].center.y;
        pose_.visibility = score;

        tmp_.key_points.push_back(pose_);

        // TODO: 把 pose_ 追加到结果（如 Result::key_points）并随检测结果返回
    }
    result_.push_back(tmp_);
}