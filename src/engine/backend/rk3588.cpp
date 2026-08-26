#include "rk3588.h"
#include <cstring>
#include <cstdlib>
#include <stdexcept>

const int RK3588_CORE_NUM = 3;

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

// 线程安全的核心轮询分配
int get_core_num()
{
    static int core_num = 0;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    int temp = core_num % RK3588_CORE_NUM;
    core_num++;
    return temp;
}

Rk3588::Rk3588(bool zero_copy, rknn_tensor_type input_type)
    : zero_copy_(zero_copy), input_type_(input_type) {}

Rk3588::Rk3588(rknn_context *ctx_in, bool zero_copy, rknn_tensor_type input_type)
    : dup_ctx_(ctx_in), zero_copy_(zero_copy), input_type_(input_type) {}

Rk3588::~Rk3588()
{
    if (input_)
    {
        // 释放每个输入 tensor 的 CPU 缓冲（非零拷贝模式分配）
        for (size_t i = 0; i < config_.input_tensors_count; ++i)
        {
            if (input_[i].buf)
                free(input_[i].buf);
        }
        free(input_);
        input_ = nullptr;
    }
    if (output_)
    {
        // 释放每个输出 tensor 的 CPU 缓冲（非零拷贝模式分配）
        for (size_t i = 0; i < config_.output_tensors_count; ++i)
        {
            if (output_[i].buf)
                free(output_[i].buf);
        }
        free(output_);
        output_ = nullptr;
    }
    if (input_attr_)
    {
        free(input_attr_);
        input_attr_ = nullptr;
    }
    if (output_attr_)
    {
        free(output_attr_);
        output_attr_ = nullptr;
    }
    if (output_tmp_attr_)
    {
        free(output_tmp_attr_);
        output_tmp_attr_ = nullptr;
    }
    tensor_data_ptr_ = nullptr;
    if (ctx_ != 0)
    {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
}

// 方式 A：从内存 Buffer 初始化
void Rk3588::Initialize(const char *model_data, const uint64_t size)
{
    int ret = 0;
    if (dup_ctx_ != nullptr)
    {
        LOG_DEBUG("Reuse weight from other context via rknn_dup_context");
        // 正确的参数顺序：rknn_dup_context(源Context指针, 目标Context的地址)
        ret = rknn_dup_context(dup_ctx_, &ctx_);
    }
    else
    {
        LOG_DEBUG("Load model from memory buffer and init new context");
        ret = rknn_init(&ctx_, (void *)model_data, size, 0, NULL);
    }

    if (ret != RKNN_SUCC)
    {
        LOG_ERROR("RKNN context init/dup failed! ret={}", ret);
        return;
    }

    config_.model_name_path = "memory_buffer";
    if (!QueryAndConfigureRuntime())
        throw std::runtime_error("RKNN runtime configuration failed");
}

// 方式 B：从文件路径初始化
void Rk3588::Initialize(const char *model_path)
{
    int ret = 0;
    if (dup_ctx_ != nullptr)
    {
        LOG_DEBUG("Reuse weight from other context, skip reading file: {}", model_path);
        ret = rknn_dup_context(dup_ctx_, &ctx_);
    }
    else
    {
        LOG_DEBUG("Load model from file: {} and init new context", model_path);
        FILE *fp = fopen(model_path, "rb");
        if (fp == nullptr)
        {
            LOG_ERROR("fopen {} fail!", model_path);
            return;
        }
        fseek(fp, 0, SEEK_END);
        int model_len = ftell(fp);
        if (model_len <= 0)
        {
            LOG_ERROR("invalid model size for {}", model_path);
            fclose(fp);
            return;
        }
        unsigned char *model = (unsigned char *)malloc(model_len);
        if (model == nullptr)
        {
            LOG_ERROR("malloc model buffer failed for {}", model_path);
            fclose(fp);
            return;
        }
        fseek(fp, 0, SEEK_SET);
        if (model_len != (int)fread(model, 1, model_len, fp))
        {
            LOG_ERROR("fread {} fail!", model_path);
            free(model);
            fclose(fp);
            return;
        }
        fclose(fp);

        ret = rknn_init(&ctx_, model, model_len, 0, NULL);
        free(model);
    }

    if (ret != RKNN_SUCC)
    {
        LOG_ERROR("RKNN context init/dup failed! ret={}", ret);
        return;
    }

    config_.model_name_path = model_path;
    if (!QueryAndConfigureRuntime())
        throw std::runtime_error("RKNN runtime configuration failed");
}

// 核心公共提取函数：处理多NPU绑定、IO属性查询、结构体赋值
bool Rk3588::QueryAndConfigureRuntime()
{
    // 1. 绑定核心 (0 -> CORE_0, 1 -> CORE_1, 2 -> CORE_2)
    switch (get_core_num())
    {
    case 0:
        core_mask_ = RKNN_NPU_CORE_0;
        break;
    case 1:
        core_mask_ = RKNN_NPU_CORE_1;
        break;
    case 2:
        core_mask_ = RKNN_NPU_CORE_2;
        break;
    default:
        core_mask_ = RKNN_NPU_CORE_AUTO;
        break;
    }

    int ret = rknn_set_core_mask(ctx_, core_mask_);
    if (ret < 0)
    {
        LOG_ERROR("rknn_set_core_mask fail! ret={}", ret);
        return false;
    }

    // 2. 查询 SDK 版本
    rknn_sdk_version version;
    if (rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)) == RKNN_SUCC)
    {
        LOG_DEBUG("RKNN API version: {}, Driver version: {}", version.api_version, version.drv_version);
    }

    // 3. 获取输入输出 Tensor 数量
    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        LOG_ERROR("rknn_query IN_OUT_NUM fail! ret={}", ret);
        return false;
    }

    // 4. 分配并查询输入属性
    input_attr_ = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    if (io_num.n_input > 0 && input_attr_ == nullptr)
        return false;
    memset(input_attr_, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        input_attr_[i].index = i;

        ret = zero_copy_ ? rknn_query(ctx_, RKNN_QUERY_NATIVE_INPUT_ATTR, &input_attr_[i], sizeof(rknn_tensor_attr))
                         : rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_[i], sizeof(rknn_tensor_attr));

        input_attr_[i].type = input_type_; // 强制覆盖输入类型，避免模型输入类型不一致导致的推理失败

        if (ret != RKNN_SUCC)
        {
            LOG_ERROR("rknn_query input attr {} fail!", i);
            return false;
        }
    }

    // 5. 分配并查询输出属性
    output_attr_ = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    if (io_num.n_output > 0 && output_attr_ == nullptr)
        return false;
    memset(output_attr_, 0, io_num.n_output * sizeof(rknn_tensor_attr));

    if (zero_copy_)
    {
        output_tmp_attr_ = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
        if (io_num.n_output > 0 && output_tmp_attr_ == nullptr)
            return false;
        memset(output_tmp_attr_, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    }

    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        output_attr_[i].index = i;

        ret = zero_copy_ ? rknn_query(ctx_, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &output_attr_[i], sizeof(rknn_tensor_attr))
                         : rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attr_[i], sizeof(rknn_tensor_attr));
        if (zero_copy_)
        {
            output_tmp_attr_[i].index = i;
            ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_tmp_attr_[i], sizeof(rknn_tensor_attr));
        }

        if (ret != RKNN_SUCC)
        {
            LOG_ERROR("rknn_query output attr {} fail!", i);
            return false;
        }
    }

    // 7. 配置公共基础 config_
    config_.model_format = ai_framework::RKNN_FORMAT;
    config_.input_tensors_count = io_num.n_input;
    config_.output_tensors_count = io_num.n_output;
    config_.rknn_zero_copy = zero_copy_;
    config_.rknn_ctx = ctx_;

    // 8. 填充输入字典映射
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        std::string name(input_attr_[i].name);
        config_.input_index_to_name[i] = name;
        config_.input_element_count[name] = input_attr_[i].n_elems;
        config_.input_single_element_size[name] = input_attr_[i].w_stride;

        std::vector<int64_t> shape;
        for (uint32_t d = 0; d < input_attr_[i].n_dims; d++)
            shape.push_back(input_attr_[i].dims[d]);
        config_.input_layer_shape[name] = shape;

        config_.tensor_size[name] = input_attr_[i].size_with_stride;
        config_.width_equal_stride[name] = (input_attr_[i].w_stride == (uint32_t)input_attr_[i].dims[2]);
        config_.stride[name] = input_attr_[i].w_stride;

        config_.input_fmt_str[name] = get_format_string(input_attr_[i].fmt);
        config_.input_type_str[name] = get_type_string(input_type_);
        config_.input_qnt_type_str[name] = get_qnt_type_string(input_attr_[i].qnt_type);
    }

    // 9. 填充输出字典映射（修正了原先映射到input_fmt_str的错位Bug）
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        std::string name(output_attr_[i].name);
        config_.output_index_to_name[i] = name;
        config_.output_element_count[name] = output_attr_[i].n_elems;
        config_.output_single_element_size[name] = output_attr_[i].w_stride;

        if (zero_copy_)
        {
            std::vector<int64_t> shape;
            for (uint32_t d = 0; d < output_tmp_attr_[i].n_dims; d++)
                shape.push_back(output_tmp_attr_[i].dims[d]);
            config_.output_layer_shape[name] = shape;

            std::vector<int64_t> shape_native;
            for (uint32_t d = 0; d < output_attr_[i].n_dims; d++)
                shape_native.push_back(output_attr_[i].dims[d]);
            config_.output_native_layer_shape[name] = shape_native;
        }
        else
        {
            std::vector<int64_t> shape;
            for (uint32_t d = 0; d < output_attr_[i].n_dims; d++)
                shape.push_back(output_attr_[i].dims[d]);
            config_.output_layer_shape[name] = shape;
        }

        if ((output_attr_[i].type == RKNN_TENSOR_INT8 ||
             output_attr_[i].type == RKNN_TENSOR_UINT8) &&
            (output_attr_[i].qnt_type != RKNN_TENSOR_QNT_NONE))
        {
            config_.scale[name] = output_attr_[i].scale;
            config_.zero_point[name] = output_attr_[i].zp;
        }
        config_.tensor_size[name] = output_attr_[i].size_with_stride;

        // config_.width_equal_stride[name] = (output_attr_[i].w_stride == (uint32_t)output_attr_[i].dims[2]);
        // config_.stride[name] = output_attr_[i].w_stride;

        // LOG_INFO("Output tensor[{}]: name={}, width_equal_stride={}, stride={}",
        //          i, name, config_.width_equal_stride[name], config_.stride[name]);

        config_.output_fmt_str[name] = get_format_string(output_attr_[i].fmt);
        config_.output_type_str[name] = get_type_string(output_attr_[i].type);
        config_.output_qnt_type_str[name] = get_qnt_type_string(output_attr_[i].qnt_type);
    }

    LOG_DEBUG("Rk3588 runtime configuration success on core mask: {}", (int)core_mask_);
    return true;
}

void Rk3588::BindInputAndOutput(ai_framework::TensorData &tensor_data)
{

    if (zero_copy_)
    {
        auto **input_mem = tensor_data.get_input_rknn_tensor_mem_ptr();
        auto **output_mem = tensor_data.get_output_rknn_tensor_mem_ptr();

        for (uint32_t i = 0; i < tensor_data.get_input_tensor_count(); ++i)
        {

            // 分配零拷贝输入内存
            input_mem[i] = rknn_create_mem(ctx_, input_attr_[i].size_with_stride);
            if (input_mem[i] == nullptr)
            {
                LOG_ERROR("rknn_create_mem(input[{}], size={}) failed!", i, input_attr_[i].size_with_stride);
                throw std::runtime_error("RKNN input memory allocation failed");
            }

            int ret = rknn_set_io_mem(ctx_, input_mem[i], &input_attr_[i]);

            if (ret < 0)
            {
                LOG_ERROR("rknn_set_io_mem(input[{}]) fail! ret={}", i, ret);
                rknn_destroy_mem(ctx_, input_mem[i]);
                input_mem[i] = nullptr;
                throw std::runtime_error("RKNN input memory binding failed");
            }

            LOG_DEBUG("zero-copy input mem[{}]: virt={} fd={} size={} fmt={}",
                      i,
                      input_mem[i]->virt_addr,
                      input_mem[i]->fd,
                      input_attr_[i].size_with_stride,
                      input_attr_[i].fmt);

            // 将 DMA buffer 地址暴露给外部，统一通过 get_input_tensor_ptr() 写入数据
            tensor_data.get_input_tensor_ptr()[i] = input_mem[i]->virt_addr;
        }

        for (uint32_t i = 0; i < tensor_data.get_output_tensor_count(); ++i)
        {

            // 分配零拷贝输出内存
            output_mem[i] = rknn_create_mem(ctx_, output_attr_[i].size_with_stride);
            if (output_mem[i] == nullptr)
            {
                LOG_ERROR("rknn_create_mem(output[{}], size={}) failed!", i, output_attr_[i].size_with_stride);
                throw std::runtime_error("RKNN output memory allocation failed");
            }
            int ret = rknn_set_io_mem(ctx_, output_mem[i], &output_attr_[i]);
            if (ret < 0)
            {
                LOG_ERROR("rknn_set_io_mem(output[{}]) fail! ret={}", i, ret);
                rknn_destroy_mem(ctx_, output_mem[i]);
                output_mem[i] = nullptr;
                throw std::runtime_error("RKNN output memory binding failed");
            }
            LOG_DEBUG("zero-copy output mem[{}]: virt={} fd={} size={} fmt={} ",
                      i,
                      output_mem[i]->virt_addr,
                      output_mem[i]->fd,
                      output_attr_[i].size_with_stride,
                      output_attr_[i].fmt);

            // 将 DMA buffer 地址暴露给外部，统一通过 get_output_tensor_ptr() 读取结果
            tensor_data.get_output_tensor_ptr()[i] = output_mem[i]->virt_addr;
        }
    }
    else
    {
        input_ = (rknn_input *)malloc(config_.input_tensors_count * sizeof(rknn_input));
        if (config_.input_tensors_count > 0 && input_ == nullptr)
            throw std::bad_alloc();
        memset(input_, 0, config_.input_tensors_count * sizeof(rknn_input));

        output_ = (rknn_output *)malloc(config_.output_tensors_count * sizeof(rknn_output));
        if (config_.output_tensors_count > 0 && output_ == nullptr)
            throw std::bad_alloc();
        memset(output_, 0, config_.output_tensors_count * sizeof(rknn_output));

        for (size_t i = 0; i < config_.input_tensors_count; ++i)
        {
            input_[i].index = i;
            input_[i].type = input_type_;
            input_[i].fmt = input_attr_[i].fmt;
            input_[i].size = input_attr_[i].size;

            void *buffer = malloc(input_attr_[i].size);
            if (buffer == nullptr)
            {
                LOG_ERROR("malloc CPU buffer failed for input tensor [{}] size={}", i, input_attr_[i].size);
                throw std::bad_alloc();
            }
            memset(buffer, 0, input_attr_[i].size);
            input_[i].buf = buffer;
            // 暴露实际 CPU buffer 地址（而非 &input_[i].buf），预处理直接写入该 buffer，
            // 之后 rknn_inputs_set 使用同一块内存，避免覆盖指针变量造成堆破坏。
            tensor_data.get_input_tensor_ptr()[i] = buffer;
        }

        for (size_t i = 0; i < config_.output_tensors_count; ++i)
        {
            output_[i].index = i;
            output_[i].want_float = 0;  // 保持原始量化类型（int8/uint8/fp16），后处理自行反量化
            output_[i].is_prealloc = 1; // 预分配缓冲，rknn_outputs_get 直接写入，指针稳定

            void *buffer = malloc(output_attr_[i].size);
            if (buffer == nullptr)
            {
                LOG_ERROR("malloc CPU buffer failed for output tensor [{}] size={}", i, output_attr_[i].size);
                throw std::bad_alloc();
            }
            memset(buffer, 0, output_attr_[i].size);
            output_[i].buf = buffer;
            output_[i].size = output_attr_[i].size;
            // 暴露实际输出缓冲地址（而非 &output_[i].buf），后处理直接读取该 buffer。
            tensor_data.get_output_tensor_ptr()[i] = buffer;
        }
    }
    tensor_data_ptr_ = &tensor_data; // 保存指针，供 DoInference() 使用
}

void Rk3588::DoInference()
{
    if (zero_copy_)
    {
        int ret = rknn_run(ctx_, nullptr);
        if (ret < 0)
        {
            LOG_ERROR("rknn_run fail! ret={}", ret);
            return;
        }

        auto output_tensors = tensor_data_ptr_->get_output_tensor_ptr();
        for (size_t i = 0; i < config_.output_tensors_count; ++i)
        {
            const auto &name = config_.output_index_to_name.at(i);
            if (config_.output_fmt_str.at(name) == "NC1HWC2")
            {
                const auto &shape = config_.output_layer_shape.at(name);
                const auto &native_shape = config_.output_native_layer_shape.at(name);
                const std::string type_str = config_.output_type_str.at(name);

                void *zero_copy_buf = output_tensors[i];

                int dims[5] = {
                    static_cast<int>(native_shape[0]),
                    static_cast<int>(native_shape[1]),
                    static_cast<int>(native_shape[2]),
                    static_cast<int>(native_shape[3]),
                    static_cast<int>(native_shape[4])};

                const int channel = static_cast<int>(shape[1]);
                const int h = static_cast<int>(shape[2]);
                const int w = static_cast<int>(shape[3]);

                void *temp_buffer = malloc(output_tmp_attr_[i].size_with_stride);
                if (temp_buffer == nullptr)
                {
                    LOG_ERROR("malloc temporary output buffer failed for tensor [{}]", i);
                    return;
                }

                // LOG_INFO("NCHW size:{}", output_tmp_attr_[i].size_with_stride);
                NC1HWC2_to_NCHW(type_str, zero_copy_buf, temp_buffer, dims, channel, h, w);

                memcpy(output_tensors[i], temp_buffer, output_tmp_attr_[i].size_with_stride);

                free(temp_buffer);
                temp_buffer = nullptr;
            }
        }
    }
    else
    {
        // 1. 上传输入数据（input_ 的 buf 由 BindInputAndOutput 预分配，与预处理写入的地址一致）
        int ret = rknn_inputs_set(ctx_, config_.input_tensors_count, input_);
        if (ret < 0)
        {
            LOG_ERROR("rknn_inputs_set fail! ret={}", ret);
            return;
        }
        // 2. 执行推理
        ret = rknn_run(ctx_, nullptr);
        if (ret < 0)
        {
            LOG_ERROR("rknn_run fail! ret={}", ret);
            return;
        }
        // 3. 获取推理输出结果（is_prealloc=1，写入预分配缓冲，后处理直接读取）
        ret = rknn_outputs_get(ctx_, config_.output_tensors_count, output_, nullptr);
        if (ret < 0)
        {
            LOG_ERROR("rknn_outputs_get failed! ret={}", ret);
            return;
        }
    }
}