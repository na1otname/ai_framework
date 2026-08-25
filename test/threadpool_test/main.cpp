#include "time/run_once.h"
#include "time/time_duration.h"
#include "utils/tools.h"
#include "utils/logger.h"
#include "utils/videofile.h"
#include "utils/yolo_threadpool.h"
#include "yaml-cpp/yaml.h"
#include <filesystem>

double get_now()
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::duration<double>>(duration)
        .count();
}
int main(int argc, char **argv)
{
    const std::filesystem::path config_dir = AI_FRAMEWORK_CONFIG_DIR;
    YAML::Node model_config = YAML::LoadFile((config_dir / "model.yaml").string());
    int thread_num = model_config["thread_num"].as<int>();
    bool cv_show = model_config["cv_show"].as<bool>();
    auto model_path = (config_dir / model_config["model_path"].as<std::string>()).lexically_normal().string();
    auto video_path = (config_dir / model_config["video_path"].as<std::string>()).lexically_normal().string();
    auto fps = model_config["video_play_fps"].as<float>();
    int delay = 1000.0 / fps;
    LOG_INFO("model_path = {}", model_path);
    YAML::Node postprocess_config =
        YAML::LoadFile((config_dir / "postprocess.yaml").string());
    std::vector<float> confidence_thresholds;
    std::vector<std::string> labels;
    for (const auto &kv : postprocess_config)
    {
        labels.push_back(kv.first.as<std::string>());
        confidence_thresholds.push_back(kv.second["confidence_threshold"].as<float>());
    }
    YoloThreadPool yolo_threadpool(model_path, confidence_thresholds, thread_num);
    std::vector<cv::Mat> input(1);
    VideoFile video_file(video_path);
    input.at(0) = video_file.GetNextFrame();
    if (!input.at(0).empty())
    {
        std::vector<double> time;
        time.push_back(get_now());
        yolo_threadpool.AddInferenceTask(input, time, true);
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
    int image_count = 0;
    int result_count = 0;
    int lost_count = 0;
    std::shared_ptr<YoloThreadPool::YoloInferenceResult> result = nullptr;
    TimeDuration time_duration;

    do
    {
        auto func = [&]
        {
            input.at(0) = video_file.GetNextFrame();
            if (!input.at(0).empty())
            {
                std::vector<double> time;
                time.push_back(get_now());
                yolo_threadpool.AddInferenceTask(input, time, true);
                ++image_count;
            }
            result = yolo_threadpool.GetInferenceResult();
            if (result != nullptr)
            {
                TIME_COST_DEBUG(
                    "ShowResults()",
                    ShowResults(result->original_image.at(0),
                                yolo_threadpool.get_model_input_side_length(),
                                result->results, labels, 1, false, false, false,
                                cv_show));
                ++result_count;
                lost_count = 0;
            }
            else
            {
                lost_count++;
            }
            LOG_DEBUG("image_count: {}, result count: {}, lost count: {}",
                      image_count, result_count, lost_count);
        };
        /* code */
        run_once_with_delay(func, std::chrono::milliseconds(delay));

    } while ((!input.at(0).empty() || yolo_threadpool.get_task_size() || result_count < image_count) && lost_count < 10);

    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(time_duration.DurationSinceLastTime());
    float real_framerate = result_count * 1000.0f / time.count();
    LOG_INFO("Total time: {}ms, and average frame rate is {}fps",
             time.count(), real_framerate);
    cv::destroyAllWindows();
    return 0;
}