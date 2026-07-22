#ifndef _AI_FRAMEWORK_H_
#define _AI_FRAMEWORK_H_

#include <map>
#include <vector>
#include <string>
#ifdef TRT
#include "cuda_runtime_api.h"
#endif
#ifdef RK3588
#include "rknn_api.h"
#endif
#include "ai_instance.h"

namespace ai_framework
{

    class TensorData;
    class AiInstance
    {
    public:
        AiInstance() = default;
        virtual void Initialize(const char *model_data, const uint64_t size) = 0;
        virtual void Initialize(const char *model_path) = 0;
        virtual void BindInputAndOutput(TensorData &tensor_data) = 0;
        virtual void DoInference() = 0;
        size_t get_input_tensor_size();
        size_t get_output_tensor_size();
        uint16_t get_input_tensor_count();
        uint16_t get_output_tensor_count();
        const Config &get_config() const { return config_; }
        void PrintLayerInfo();

    protected:
        Config config_;
    };

    class TensorData
    {
    public:
        TensorData() = delete;
        TensorData(const Config &config);
        ~TensorData();
        void **&get_input_tensor_ptr() { return input_tensor_ptr_; }
        void **&get_output_tensor_ptr() { return output_tensor_ptr_; }
#ifdef TRT
        void **&get_input_tensor_cuda_ptr() { return input_tensor_cuda_ptr_; }
        void **&get_output_tensor_cuda_ptr() { return output_tensor_cuda_ptr_; }
#endif
#ifdef RK3588
        rknn_tensor_mem **&get_input_rknn_tensor_mem_ptr()
        {
            return input_rknn_tensor_mem_ptr_;
        }
        rknn_tensor_mem **&get_output_rknn_tensor_mem_ptr()
        {
            return output_rknn_tensor_mem_ptr_;
        }
#endif
        uint16_t get_input_tensor_count() { return input_tensor_count_; }
        uint16_t get_output_tensor_count() { return output_tensor_count_; }
        std::vector<std::string> &get_input_tensor_name() { return input_name_; }
        std::vector<std::string> &get_output_tensor_name() { return output_name_; }
        const std::vector<uint64_t> &get_input_tensor_size()
        {
            return input_tensor_size_;
        }
        const std::vector<uint64_t> &get_output_tensor_size()
        {
            return output_tensor_size_;
        }

    private:
        void **input_tensor_ptr_{nullptr};
        uint16_t input_tensor_count_{0};
        std::vector<uint64_t> input_tensor_size_;
        std::vector<std::string> input_name_;
        void **output_tensor_ptr_{nullptr};
        uint16_t output_tensor_count_{0};
        std::vector<uint64_t> output_tensor_size_;
        std::vector<std::string> output_name_;
#ifdef TRT
        void **input_tensor_cuda_ptr_{nullptr};
        void **output_tensor_cuda_ptr_{nullptr};
        bool trt_unified_memory_{false};
#endif
#ifdef RK3588
        rknn_tensor_mem **input_rknn_tensor_mem_ptr_{nullptr};
        rknn_tensor_mem **output_rknn_tensor_mem_ptr_{nullptr};
        rknn_context rknn_ctx_;
        bool rknn_zero_copy_;
#endif
    };

} // namespace ai_framework

#endif