#include "yolo_preprocess.h"

#ifdef RK3588
#include "im2d.hpp"
#include <vector>
#endif

YoloPreProcess::YoloPreProcess(int target_side_length, bool debug)
    : target_side_length_(target_side_length), debug_(debug) {}

void YoloPreProcess::Run(const std::vector<cv::Mat> &input, void *tensors[])
{
    for (size_t i = 0; i < input.size(); ++i)
    {
        auto &original_input = input.at(i);
        int resize_width = target_side_length_;
        int resize_height = target_side_length_;
        if (original_input.cols >= original_input.rows)
        {
            float scale = 1.0f * original_input.cols / target_side_length_;
            resize_height = original_input.rows / scale;
        }
        else
        {
            float scale = 1.0f * original_input.rows / target_side_length_;
            resize_width = original_input.cols / scale;
        }
#ifdef RK3588
        // 使用 RGA 硬件加速预处理：resize + BGR2RGB + padding
        // Step 1: RGA resize（同时做 BGR→RGB 格式转换）
        std::vector<uint8_t> cvt_buf(resize_width * resize_height * 3);
        rga_buffer_t src_rga = wrapbuffer_virtualaddr(
            (void *)original_input.data,
            original_input.cols, original_input.rows,
            RK_FORMAT_BGR_888);
        rga_buffer_t cvt_rga = wrapbuffer_virtualaddr(
            cvt_buf.data(),
            resize_width, resize_height,
            RK_FORMAT_RGB_888);
        imresize(src_rga, cvt_rga);

        // Step 2: 计算边框并填充为正方形，填充色 RGB(114,114,114)
        int border_left = 0, border_right = 0, border_top = 0, border_bottom = 0;
        if (resize_height > resize_width)
        {
            int delta = resize_height - resize_width;
            border_left = delta >> 1;
            border_right = delta - border_left;
        }
        else if (resize_width > resize_height)
        {
            int delta = resize_width - resize_height;
            border_top = delta >> 1;
            border_bottom = delta - border_top;
        }

        rga_buffer_t out_rga = wrapbuffer_virtualaddr(
            tensors[i],
            target_side_length_, target_side_length_,
            RK_FORMAT_RGB_888);
        immakeBorder(cvt_rga, out_rga,
                     border_top, border_bottom, border_left, border_right,
                     IM_BORDER_CONSTANT, 0x727272);

        if (debug_)
        {
            cv::Mat debug_mat(target_side_length_, target_side_length_,
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

void YoloPreProcess::MakeSquare(const cv::Mat &src, cv::Mat &dst)
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
        int delta = height - width;
        border_left += (delta >> 1);
        border_right = border_left;
    }
    else
    {
        int delta = width - height;
        border_top += (delta >> 1);
        border_bottom = border_top;
    }
    // 使用灰色(114,114,114)填充边缘
    cv::copyMakeBorder(src, dst, border_top, border_bottom, border_left,
                       border_right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
}

uint64_t YoloPreProcess::PopulateData(
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