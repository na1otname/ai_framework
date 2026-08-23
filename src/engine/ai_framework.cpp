#include "ai_framework.h"

namespace ai_framework
{

    TensorData::TensorData(const Config &config) : input_tensor_count_(config.input_tensors_count),
                                                   output_tensor_count_(config.output_tensors_count)

    {

        // 1.初始化输入张量信息
        input_name_.reserve(input_tensor_count_);
        input_tensor_size_.reserve(input_tensor_count_);

        // 分配 void* 指针数组空间
        if (input_tensor_count_ > 0)
        {
            input_tensor_ptr_ = new void *[input_tensor_count_];
            for (uint16_t i = 0; i < input_tensor_count_; ++i)
            {
                input_tensor_ptr_[i] = nullptr; // 初始化为空指针
                // 获取并保存tensor名称
                std::string name = config.input_index_to_name.at(i);
                input_name_.push_back(name);

                // 获取并保存 tensor大小
                if (config.tensor_size.count(name))
                {
                    input_tensor_size_.push_back(config.tensor_size.at(name));
                }
                else
                {
                    // 如果没有直接提供 tensor_size，通过 数量 * 单个元素大小 计算
                    size_t size = config.input_element_count.at(name) * config.input_single_element_size.at(name);
                    input_tensor_size_.push_back(size);
                }
            }
        }

        // 2. 初始化输出张量相关信息
        output_name_.reserve(output_tensor_count_);
        output_tensor_size_.reserve(output_tensor_count_);

        if (output_tensor_count_ > 0)
        {
            output_tensor_ptr_ = new void *[output_tensor_count_];
            for (uint16_t i = 0; i < output_tensor_count_; ++i)
            {
                output_tensor_ptr_[i] = nullptr;

                std::string name = config.output_index_to_name.at(i);
                output_name_.push_back(name);

                if (config.tensor_size.count(name))
                {
                    output_tensor_size_.push_back(config.tensor_size.at(name));
                }
                else
                {
                    size_t size = config.output_element_count.at(name) * config.output_single_element_size.at(name);
                    output_tensor_size_.push_back(size);
                }
            }
        }

        // 3. TensorRT 平台特定初始化
#ifdef TRT
        trt_unified_memory_ = config.trt_use_unified_memory;
        if (input_tensor_count_ > 0)
        {
            input_tensor_cuda_ptr_ = new void *[input_tensor_count_];
            for (uint16_t i = 0; i < input_tensor_count_; ++i)
            {
                input_tensor_cuda_ptr_[i] = nullptr;
            }
        }
        if (output_tensor_count_ > 0)
        {
            output_tensor_cuda_ptr_ = new void *[output_tensor_count_];
            for (uint16_t i = 0; i < output_tensor_count_; ++i)
            {
                output_tensor_cuda_ptr_[i] = nullptr;
            }
        }
#endif

        // 4. RK3588 平台特定初始化
#ifdef RK3588
        rknn_ctx_ = config.rknn_ctx;
        rknn_zero_copy_ = config.rknn_zero_copy;

        if (input_tensor_count_ > 0)
        {
            // 分配 rknn_tensor_mem* 指针数组
            input_rknn_tensor_mem_ptr_ = new rknn_tensor_mem *[input_tensor_count_];
            for (uint16_t i = 0; i < input_tensor_count_; ++i)
            {
                input_rknn_tensor_mem_ptr_[i] = nullptr;
            }
        }
        if (output_tensor_count_ > 0)
        {
            output_rknn_tensor_mem_ptr_ = new rknn_tensor_mem *[output_tensor_count_];
            for (uint16_t i = 0; i < output_tensor_count_; ++i)
            {
                output_rknn_tensor_mem_ptr_[i] = nullptr;
            }
        }
#endif
    }

    TensorData::~TensorData()
    {
        // 释放主机端（CPU）指针数组
        if (input_tensor_ptr_ != nullptr)
        {
            delete[] input_tensor_ptr_;
            input_tensor_ptr_ = nullptr;
        }
        if (output_tensor_ptr_ != nullptr)
        {
            delete[] output_tensor_ptr_;
            output_tensor_ptr_ = nullptr;
        }

#ifdef TRT
        // 释放 CUDA 端指针数组
        if (input_tensor_cuda_ptr_ != nullptr)
        {
            delete[] input_tensor_cuda_ptr_;
            input_tensor_cuda_ptr_ = nullptr;
        }
        if (output_tensor_cuda_ptr_ != nullptr)
        {
            delete[] output_tensor_cuda_ptr_;
            output_tensor_cuda_ptr_ = nullptr;
        }
#endif

#ifdef RK3588
        // TensorData 的生命周期早于 Engine 内的 Rk3588 backend 结束，
        // 因此这里仍可使用有效的 RKNN context 释放零拷贝内存。
        if (rknn_zero_copy_ && rknn_ctx_ != 0)
        {
            if (input_rknn_tensor_mem_ptr_ != nullptr)
            {
                for (uint16_t i = 0; i < input_tensor_count_; ++i)
                {
                    if (input_rknn_tensor_mem_ptr_[i] != nullptr)
                    {
                        rknn_destroy_mem(rknn_ctx_, input_rknn_tensor_mem_ptr_[i]);
                        input_rknn_tensor_mem_ptr_[i] = nullptr;
                    }
                }
            }
            if (output_rknn_tensor_mem_ptr_ != nullptr)
            {
                for (uint16_t i = 0; i < output_tensor_count_; ++i)
                {
                    if (output_rknn_tensor_mem_ptr_[i] != nullptr)
                    {
                        rknn_destroy_mem(rknn_ctx_, output_rknn_tensor_mem_ptr_[i]);
                        output_rknn_tensor_mem_ptr_[i] = nullptr;
                    }
                }
            }
        }

        // 释放 RKNN 内存指针数组
        if (input_rknn_tensor_mem_ptr_ != nullptr)
        {
            delete[] input_rknn_tensor_mem_ptr_;
            input_rknn_tensor_mem_ptr_ = nullptr;
        }
        if (output_rknn_tensor_mem_ptr_ != nullptr)
        {
            delete[] output_rknn_tensor_mem_ptr_;
            output_rknn_tensor_mem_ptr_ = nullptr;
        }
#endif
    }

    void AiInstance::PrintLayerInfo()
    {
        LOG_DEBUG("=== AiInstance Layer Info ===");

        LOG_DEBUG("model input num: {}, output num: {}", config_.input_tensors_count, config_.output_tensors_count);
        for (uint16_t i = 0; i < config_.input_tensors_count; i++)
        {
            auto &name = config_.input_index_to_name[i];
            auto &shape = config_.input_layer_shape[name];

            // 安全地获取前 4 个维度。如果维度不足 4 维，后面补 0（以匹配原格式 dims=[%d, %d, %d, %d]）
            int64_t d0 = shape.size() > 0 ? shape[0] : 0;
            int64_t d1 = shape.size() > 1 ? shape[1] : 0;
            int64_t d2 = shape.size() > 2 ? shape[2] : 0;
            int64_t d3 = shape.size() > 3 ? shape[3] : 0;

            // 格式化输出所有属性
            // 注意：fmt, type, qnt_type 在 Config 中暂无对应字段，当前使用 "N/A" 占位
            auto input_scale_it = config_.scale.find(name);
            auto input_zp_it = config_.zero_point.find(name);
            std::string input_zp_str = (input_zp_it != config_.zero_point.end()) ? std::to_string(input_zp_it->second) : "N/A";
            std::string input_scale_str = (input_scale_it != config_.scale.end()) ? std::to_string(input_scale_it->second) : "N/A";
            LOG_DEBUG("index={}, name={}, n_dims={}, dims=[{}, {}, {}, {}], n_elems={}, size={}, fmt={}, type={}, qnt_type={}, zp={}, scale={}",
                      i,
                      name.c_str(),
                      shape.size(),
                      d0, d1, d2, d3,
                      config_.input_element_count[name],
                      config_.tensor_size[name],
                      config_.input_fmt_str[name], config_.input_type_str[name], config_.input_qnt_type_str[name], // 如果未来 Config 加了这三个字段，请替换此处
                      input_zp_str,
                      input_scale_str);
        }

        for (uint16_t i = 0; i < config_.output_tensors_count; i++)
        {
            if (config_.rknn_zero_copy)
            {
                auto &name = config_.output_index_to_name[i];
                auto &shape = config_.output_native_layer_shape[name];

                // 安全地获取前 4 个维度。如果维度不足 4 维，后面补 0（以匹配原格式 dims=[%d, %d, %d, %d]）
                int64_t d0 = shape.size() > 0 ? shape[0] : 0;
                int64_t d1 = shape.size() > 1 ? shape[1] : 0;
                int64_t d2 = shape.size() > 2 ? shape[2] : 0;
                int64_t d3 = shape.size() > 3 ? shape[3] : 0;
                int64_t d4 = shape.size() > 4 ? shape[4] : 0; // 如果输出是5维，获取第5维，否则为0

                // 格式化输出所有属性
                // 注意：fmt, type, qnt_type 在 Config 中暂无对应字段，当前使用 "N/A" 占位
                auto output_scale_it = config_.scale.find(name);
                auto output_zp_it = config_.zero_point.find(name);
                std::string output_zp_str = (output_zp_it != config_.zero_point.end()) ? std::to_string(output_zp_it->second) : "N/A";
                std::string output_scale_str = (output_scale_it != config_.scale.end()) ? std::to_string(output_scale_it->second) : "N/A";
                LOG_DEBUG("index={}, name={}, n_dims={}, dims=[{}, {}, {}, {}, {}], n_elems={}, size={}, fmt={}, type={}, qnt_type={}, zp={}, scale={}",
                          i,
                          name.c_str(),
                          shape.size(),
                          d0, d1, d2, d3, d4,
                          config_.output_element_count[name],
                          config_.tensor_size[name],
                          config_.output_fmt_str[name], config_.output_type_str[name], config_.output_qnt_type_str[name],
                          output_zp_str,
                          output_scale_str);
            }
            else
            {
                auto &name = config_.output_index_to_name[i];
                auto &shape = config_.output_layer_shape[name];

                // 安全地获取前 4 个维度。如果维度不足 4 维，后面补 0（以匹配原格式 dims=[%d, %d, %d, %d]）
                int64_t d0 = shape.size() > 0 ? shape[0] : 0;
                int64_t d1 = shape.size() > 1 ? shape[1] : 0;
                int64_t d2 = shape.size() > 2 ? shape[2] : 0;
                int64_t d3 = shape.size() > 3 ? shape[3] : 0;

                // 格式化输出所有属性
                // 注意：fmt, type, qnt_type 在 Config 中暂无对应字段，当前使用 "N/A" 占位
                auto output_scale_it = config_.scale.find(name);
                auto output_zp_it = config_.zero_point.find(name);
                std::string output_zp_str = (output_zp_it != config_.zero_point.end()) ? std::to_string(output_zp_it->second) : "N/A";
                std::string output_scale_str = (output_scale_it != config_.scale.end()) ? std::to_string(output_scale_it->second) : "N/A";
                LOG_DEBUG("index={}, name={}, n_dims={}, dims=[{}, {}, {}, {}], n_elems={}, size={}, fmt={}, type={}, qnt_type={}, zp={}, scale={}",
                          i,
                          name.c_str(),
                          shape.size(),
                          d0, d1, d2, d3,
                          config_.output_element_count[name],
                          config_.tensor_size[name],
                          config_.output_fmt_str[name], config_.output_type_str[name], config_.output_qnt_type_str[name],
                          output_zp_str,
                          output_scale_str);
            }
        }

        LOG_DEBUG("==============================");
    }

}
