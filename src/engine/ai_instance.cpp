#include "ai_instance.h"
#include "ai_framework.h"
#ifdef RK3588
#include "backend/rk3588.h"
#endif

namespace ai_framework
{
   
    Engine::Engine( const char *model_path)
    {

#ifdef RK3588

       instance_ptr_ =std::make_shared<Rk3588>();
#endif

#ifdef ONNXRUNTIME

        instance_ptr_ =std::make_shared<OnnxRuntime>();
#endif

#ifdef TRT
       instance_ptr_ =std::make_shared<TensorRT>();
#endif

        // instance_ptr_ = CreateBackend(format);

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
