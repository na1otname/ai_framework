#include "ai_instance.h"
#include "ai_framework.h"
#ifdef RK3588
#include "backend/rk3588.h"
#endif

template <typename T>
int NC1HWC2_to_NCHW(const void *src, void *dst, const int *dims, int channel, int h, int w)
{
    int batch = dims[0];
    int C1 = dims[1];
    int C2 = dims[4];
    int hw_src = dims[2] * dims[3];
    int hw_dst = h * w;

    const T *src_ptr = static_cast<const T *>(src);
    T *dst_ptr = static_cast<T *>(dst);

    for (int i = 0; i < batch; i++)
    {
        const T *src_b = src_ptr + i * C1 * hw_src * C2;
        T *dst_b = dst_ptr + i * channel * hw_dst;
        for (int c = 0; c < channel; ++c)
        {
            int plane = c / C2;
            const T *src_bc = src_b + plane * hw_src * C2;
            int offset = c % C2;
            for (int cur_h = 0; cur_h < h; ++cur_h)
            {
                for (int cur_w = 0; cur_w < w; ++cur_w)
                {
                    int cur_hw = cur_h * w + cur_w;
                    dst_b[c * hw_dst + cur_hw] = src_bc[C2 * cur_hw + offset];
                }
            }
        }
    }

    return 0;
}

int NC1HWC2_to_NCHW(const std::string &type_str, const void *src, void *dst, const int *dims, int channel, int h, int w)
{
    if (type_str == "INT8")
        return NC1HWC2_to_NCHW<int8_t>(src, dst, dims, channel, h, w);
    else if (type_str == "UINT8")
        return NC1HWC2_to_NCHW<uint8_t>(src, dst, dims, channel, h, w);
    else if (type_str == "FP32")
        return NC1HWC2_to_NCHW<float>(src, dst, dims, channel, h, w);
    else if (type_str == "FP16" || type_str == "BF16" || type_str == "INT16" || type_str == "UINT16")
        return NC1HWC2_to_NCHW<uint16_t>(src, dst, dims, channel, h, w); // 2字节类型统一按 16-bit 搬移
    else if (type_str == "INT32" || type_str == "UINT32")
        return NC1HWC2_to_NCHW<uint32_t>(src, dst, dims, channel, h, w);

    return -1; // 未知的类型
}

namespace ai_framework
{
    AiInstancePtr CreateBackend(ModelFormat format)
    {
        switch (format)
        {
#ifdef RK3588
        case RKNN_FORMAT:
            return std::make_shared<Rk3588>();
#endif

#ifdef ONNXRUNTIME
        case ONNX_FORMAT:
            return std::make_shared<OnnxRuntime>();
#endif

#ifdef TRT
        case TRT_FORMAT:
            return std::make_shared<TensorRT>();
#endif
        default:
            return nullptr;
        }
    }

    Engine::Engine(const enum ModelFormat format, const char *model_path)
    {
        instance_ptr_ = CreateBackend(format);

        if (instance_ptr_ == nullptr)
        {
            LOG_ERROR("backend create failed");
            throw std::runtime_error(
                "backend create failed");
        }

        // 初始化模型
        instance_ptr_->Initialize(model_path);

        tensor_data_ptr_ = std::make_shared<ai_framework::TensorData>(instance_ptr_->get_config());

        instance_ptr_->BindInputAndOutput(*tensor_data_ptr_);

        instance_ptr_->PrintLayerInfo();
    }

    void **&Engine::get_input_tensor_ptr()
    {
        return tensor_data_ptr_->get_input_tensor_ptr();
    }

    void **&Engine::get_output_tensor_ptr()
    {
#ifdef RK3588
        if (instance_ptr_->get_config().rknn_zero_copy)
        {
            void **output_tensor_ptr = tensor_data_ptr_->get_output_tensor_ptr();
            for (uint16_t i = 0; i < instance_ptr_->get_config().output_tensors_count; i++)
            {
                const auto &name = instance_ptr_->get_config().output_index_to_name.at(i);
                const auto &shape = instance_ptr_->get_config().output_layer_shape.at(name);
                const auto &native_shape = instance_ptr_->get_config().output_native_layer_shape.at(name);
                if (instance_ptr_->get_config().output_fmt_str.at(name) == "NC1HWC2")
                {
                    void *zero_copy_buf = output_tensor_ptr[i];
                    std::string type_str = instance_ptr_->get_config().output_type_str.at(name);
                    // 1. 计算 NC1HWC2 的总字节大小
                    int batch = native_shape[0];
                    int C1 = native_shape[1];
                    int H_src = native_shape[2];
                    int W_src = native_shape[3];
                    int C2 = native_shape[4];
                    size_t nc1hwc2_size = batch * C1 * H_src * W_src * C2 * sizeof(int8_t);

                    temp_nc1hwc2_buf.resize(nc1hwc2_size);
                    std::memcpy(temp_nc1hwc2_buf.data(), zero_copy_buf, nc1hwc2_size);

                    // 3. 解析维度参数
                    int dims[5] = {batch, C1, H_src, W_src, C2};
                    int channel = shape[1];
                    int h = shape[2];
                    int w = shape[3];
                    NC1HWC2_to_NCHW(type_str, temp_nc1hwc2_buf.data(), zero_copy_buf, dims, channel, h, w);
                }
            }
        }
#endif
        return tensor_data_ptr_->get_output_tensor_ptr();
    }

    const int Engine::get_input_tensor_count()
    {
        return tensor_data_ptr_->get_input_tensor_count();
    }

    const int Engine::get_output_tensor_count()
    {
        return tensor_data_ptr_->get_output_tensor_count();
    }

    const std::map<std::string, std::vector<int64_t>> &Engine::get_input_tensor_shape() const
    {
        return instance_ptr_->get_config().input_layer_shape;
    }

    const std::map<std::string, std::vector<int64_t>> &Engine::get_output_tensor_shape() const
    {
        return instance_ptr_->get_config().output_layer_shape;
    }

    const std::map<std::string, float> &Engine::get_tensor_scale() const
    {
        return instance_ptr_->get_config().scale;
    }

    const std::map<std::string, int> &Engine::get_tensor_zero_point() const
    {
        return instance_ptr_->get_config().zero_point;
    }

    const ModelFormat Engine::get_model_format() const
    {
        return instance_ptr_->get_config().model_format;
    }

    const Config &Engine::get_config() const
    {
        return instance_ptr_->get_config();
    }

    void Engine::DoInference(void)
    {
        instance_ptr_->DoInference();
    }

#ifdef RK3588
    const std::map<std::string, bool> &Engine::get_width_equal_stride() const
    {
        return instance_ptr_->get_config().width_equal_stride;
    }
    const std::map<std::string, uint32_t> &Engine::get_stride() const
    {
        return instance_ptr_->get_config().stride;
    }
#endif

} // namespace ai_framework
