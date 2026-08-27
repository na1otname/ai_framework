#ifndef AI_FRAMEWORK_PLATFORM_ROCKCHIP_RK3588_H_
#define AI_FRAMEWORK_PLATFORM_ROCKCHIP_RK3588_H_

#include "ai_framework.h"

#include "rknn_api.h"
#include <mutex>
#include <vector>
#include <map>
#include <string>
// #include "utils/engine_helper.h"

class Rk3588 : public ai_framework::AiInstance
{
public:
    Rk3588(bool zero_copy = true,
           rknn_tensor_type input_type = RKNN_TENSOR_UINT8);
    Rk3588(rknn_context *ctx_in, bool zero_copy = true,
           rknn_tensor_type input_type = RKNN_TENSOR_UINT8);
    ~Rk3588();
    virtual void Initialize(const char *model_data, const uint64_t size) final;
    virtual void Initialize(const char *model_path) final;
    virtual void BindInputAndOutput(ai_framework::TensorData &tensor_data) final;
    virtual void DoInference() final;
    rknn_context *get_context() { return &ctx_; }

private:
    // 抽取出的公共运行时配置函数
    bool QueryAndConfigureRuntime();

    rknn_context ctx_{};
    rknn_context *dup_ctx_{nullptr};
    rknn_core_mask core_mask_;
    rknn_input *input_{nullptr};
    rknn_output *output_{nullptr};
    rknn_tensor_attr *input_attr_{nullptr};
    rknn_tensor_attr *output_attr_{nullptr};
    rknn_tensor_attr *output_tmp_attr_{nullptr};
    std::vector<void *> output_nchw_buffers_;
    bool zero_copy_{true};
    rknn_tensor_type input_type_{RKNN_TENSOR_UINT8};
    ai_framework::TensorData *tensor_data_ptr_{nullptr};
};

#endif // AI_FRAMEWORK_PLATFORM_ROCKCHIP_RK3588_H_