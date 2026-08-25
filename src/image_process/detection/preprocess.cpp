#include "preprocess.h"

#ifdef RK3588
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "im2d.hpp"
#endif

PreProcess::PreProcess(int target_side_length, bool debug)
    : target_side_length_(target_side_length), debug_(debug) {}

void PreProcess::Run(const std::vector<cv::Mat> &input, void *tensors[])
{
    for (size_t i = 0; i < input.size(); ++i)
    {
        auto &original_input = input.at(i);
        if (original_input.empty() || original_input.type() != CV_8UC3)
        {
            LOG_ERROR("[PreProcess] invalid input image: index={} empty={} rows={} cols={} type={}",
                      i, original_input.empty(), original_input.rows,
                      original_input.cols, original_input.type());
            return;
        }
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
        // RGA3 版本：imfill 填充背景 114（RGB 灰），improcess 一次性完成
        // resize + BGR→RGB + 写入左上角区域。全程走 RGA3（有 IOMMU，支持
        // 用户态虚拟地址）；makeBorder 仅 RGA2 支持、且 RGA2 无 IOMMU，故不用。
        const size_t out_size =
            (size_t)target_side_length_ * target_side_length_ * 3;
        const int src_wstride = (original_input.cols + 15) & ~15;
        cv::Mat rga_input(original_input.rows, src_wstride, CV_8UC3);
        original_input.copyTo(
            rga_input(cv::Rect(0, 0, original_input.cols, original_input.rows)));
        rga_buffer_handle_t src_handle = importbuffer_virtualaddr(
            (void *)rga_input.data, (int)(rga_input.step * rga_input.rows));
        rga_buffer_handle_t out_handle =
            importbuffer_virtualaddr(tensors[i], (int)out_size);
        if (!src_handle || !out_handle)
        {
            LOG_ERROR(
                "[PreProcess] importbuffer_virtualaddr failed (src={} "
                "out={}) tensors[{}]={} out_size={} src_size={}\n",
                (int)src_handle, (int)out_handle, i, tensors[i], out_size,
                (int)(original_input.step * original_input.rows));
            return;
        }
        rga_buffer_t src_rga = wrapbuffer_handle(
            src_handle, original_input.cols, original_input.rows,
            RK_FORMAT_BGR_888, src_wstride, original_input.rows);
        rga_buffer_t out_rga = wrapbuffer_handle(
            out_handle, target_side_length_, target_side_length_,
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
        IM_STATUS check = imcheck(src_rga, out_rga, srect, drect);
        if (check != IM_STATUS_SUCCESS && check != IM_STATUS_NOERROR)
        {
            LOG_ERROR("[PreProcess] imcheck failed, status={} src={}x{} dst={}x{}",
                      check, original_input.cols, original_input.rows,
                      resize_width, resize_height);
            releasebuffer_handle(src_handle);
            releasebuffer_handle(out_handle);
            return;
        }
        IM_STATUS st_proc = improcess(src_rga, out_rga, rga_buffer_t(), srect,
                                      drect, im_rect(), -1, nullptr, &opt, 0);
        if (st_proc != IM_STATUS_SUCCESS)
        {
            LOG_ERROR(
                "[PreProcess] improcess failed, status={} src={}x{} stride={} "
                "dst={}x{} resize={}x{}",
                st_proc, original_input.cols, original_input.rows,
                src_wstride,
                target_side_length_, target_side_length_, resize_width,
                resize_height);
            releasebuffer_handle(src_handle);
            releasebuffer_handle(out_handle);
            return;
        }
        releasebuffer_handle(src_handle);
        releasebuffer_handle(out_handle);

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

void PreProcess::MakeSquare(const cv::Mat &src, cv::Mat &dst)
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

uint64_t PreProcess::PopulateData(
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