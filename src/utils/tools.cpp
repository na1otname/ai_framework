#include "tools.h"

#include "BYTETracker.h"
#include "fstream"
#include "logger.h"

void CoordinateTransformation(float &x, float &y, int width, int height,
                              int target_side_length)
{
    if (width > height)
    {
        int padding = (width - height) >> 1;
        float scale = 1.0f * target_side_length / width;
        y = std::max<float>(y / scale - padding, 0);
        x = std::max<float>(x / scale, 0);
    }
    else
    {
        int padding = (height - width) >> 1;
        float scale = 1.0f * target_side_length / height;
        x = std::max<float>(x / scale - padding, 0);
        y = std::max<float>(y / scale, 0);
    }
}

void drawSkeleton(cv::Mat &img, const std::vector<cv::Point> &points,
                  const std::vector<int> &pairs, const cv::Scalar &color,
                  int thickness)
{
    for (size_t i = 0; i < pairs.size(); i += 2)
    {
        int index1 = pairs[i];
        int index2 = pairs[i + 1];
        if (points[index1].x != -1 && points[index1].y != -1 &&
            points[index2].x != -1 && points[index2].y != -1)
        {
            cv::line(img, points[index1], points[index2], color, thickness);
        }
    }
}

void ProcessPoseImage(cv::Mat &image, Result &result,
                      const int target_side_length)
{
    std::vector<cv::Point> points(17);
    for (int j = 0; j < 17; ++j)
    {
        if (result.key_points[j].visibility <= 0.6)
        {
            points.at(j) = cv::Point(-1, -1);
            continue;
        }
        auto x = result.key_points[j].x;
        auto y = result.key_points[j].y;
        CoordinateTransformation(x, y, image.cols, image.rows, target_side_length);
        points.at(j) = (cv::Point(x, y));
        cv::circle(image, points.at(j), 10, cv::Scalar(0, 0, 255), cv::FILLED,
                   cv::LINE_AA);
    }
    std::vector<int> pairs = {
        0, 1,   // Nose to left eye
        1, 3,   // Left eye to left ear
        0, 2,   // Nose to right eye
        2, 4,   // Right eye to right ear
        0, 5,   // Nose to left shoulder
        5, 7,   // Left shoulder to left elbow
        7, 9,   // Left elbow to left wrist
        0, 6,   // Nose to right shoulder
        6, 8,   // Right shoulder to right elbow
        8, 10,  // Right elbow to right wrist
        5, 6,   // Left shoulder to right shoulder
        11, 12, // Left hip to right hip
        11, 5,  // Left hip to left shoulder
        12, 6,  // Right hip to right shoulder
        11, 13, // Left hip to left knee
        12, 14, // Right hip to right knee
        13, 15, // Left knee to left ankle
        14, 16  // Right knee to right ankle
    };
    drawSkeleton(image, points, pairs, cv::Scalar(255, 0, 0), 2);
}

class KaylordutVideoWriter
{
public:
    KaylordutVideoWriter(std::string filename, int width, int height, float fps)
    {
        writer_ =
            cv::VideoWriter(filename, cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
                            fps, cv::Size(width, height), true);
        if (!writer_.isOpened())
        {
            LOG_ERROR("Cannot create video writer");
            exit(EXIT_FAILURE);
        }
    }
    ~KaylordutVideoWriter()
    {
        if (writer_.isOpened())
        {
            writer_.release();
        }
    }
    void write(const cv::Mat &image) { writer_.write(image); }

private:
    cv::VideoWriter writer_;
};
std::shared_ptr<KaylordutVideoWriter> gVideoWriter;

#define N_CLASS_COLORS (20)
unsigned char class_colors[][3] = {
    {255, 56, 56},   // 'FF3838'
    {255, 157, 151}, // 'FF9D97'
    {255, 112, 31},  // 'FF701F'
    {255, 178, 29},  // 'FFB21D'
    {207, 210, 49},  // 'CFD231'
    {72, 249, 10},   // '48F90A'
    {146, 204, 23},  // '92CC17'
    {61, 219, 134},  // '3DDB86'
    {26, 147, 52},   // '1A9334'
    {0, 212, 187},   // '00D4BB'
    {44, 153, 168},  // '2C99A8'
    {0, 194, 255},   // '00C2FF'
    {52, 69, 147},   // '344593'
    {100, 115, 255}, // '6473FF'
    {0, 24, 236},    // '0018EC'
    {132, 56, 255},  // '8438FF'
    {82, 0, 133},    // '520085'
    {203, 56, 255},  // 'CB38FF'
    {255, 149, 200}, // 'FF95C8'
    {255, 55, 199}   // 'FF37C7'
};

// 函数定义：上色函数
// 参数：
//   image - 待修改的图像
//   mask - 掩码图像
//   color - 要应用的颜色（B, G, R）
void ApplyColorWithMask(cv::Mat &image, const cv::Mat &mask,
                        const cv::Scalar &color)
{
    // 首先检查图像和掩码的尺寸是否一致
    if (image.size() != mask.size())
    {
        throw std::runtime_error("Image and mask sizes do not match");
    }
    // 检查image是否为彩色图像
    if (image.channels() != 3)
    {
        throw std::runtime_error("Image must be a color image");
    }
    // 遍历图像中的每一个像素
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            // 检查掩码中对应的像素值是否不为0
            if (mask.at<uchar>(y, x) != 0)
            {
                // 若掩码对应位置不为零，设置图像该位置的颜色
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(color[0], color[1], color[2]);
            }
        }
    }
}

void AddWeightedSegment(cv::Mat &image, const cv::Mat &seg_mat, int id)
{
    auto width = image.cols;
    auto height = image.rows;
    auto seg_width = seg_mat.cols;
    auto seg_height = seg_mat.rows;
    int x = 0, y = 0, rect_w = seg_width, rect_h = seg_height;
    if (width > height)
    {
        auto padding =
            static_cast<int>((float)(width - height) / width * seg_height / 2);
        LOG_WARN_EXPRESSION(padding < 1, "padding is {}", padding);
        y = std::max(padding - 1, 0);
        rect_h = static_cast<int>((float)height / width * seg_height);
        if (y + rect_h > seg_height)
        {
            LOG_ERROR("y + rect_h > seg_height");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        auto padding =
            static_cast<int>((float)(height - width) / height * seg_width / 2);
        LOG_WARN_EXPRESSION(padding < 1, "padding is {}", padding);
        x = std::max(padding - 1, 0);
        rect_w = static_cast<int>((float)width / height * seg_width);
        if (x + rect_w > seg_width)
        {
            LOG_ERROR("x + rect_w > seg_width");
            exit(EXIT_FAILURE);
        }
    }
    cv::Mat all_size_seg_mat;
    cv::resize(seg_mat(cv::Rect(x, y, rect_w, rect_h)), all_size_seg_mat,
               image.size(), cv::INTER_NEAREST);
    auto count = id % N_CLASS_COLORS;
    cv::Mat mask = cv::Mat::zeros(image.size(), image.type());
    ApplyColorWithMask(mask, all_size_seg_mat,
                       cv::Scalar(class_colors[count][0], class_colors[count][1],
                                  class_colors[count][2]));
    cv::addWeighted(image, 0.8, mask, 0.2, 0, image);
}

cv::Mat GetImageResult(const cv::Mat &original_image,
                       const int target_side_length,
                       const std::vector<Result> &results,
                       std::vector<std::string> &labels, bool enable_track)
{
    auto image = original_image.clone();
    int width = image.cols;
    int height = image.rows;
    std::vector<Object> objects;
    for (auto result : results)
    {
        if (result.model_type == ModelType::SEGMENT_V11)
        {
            AddWeightedSegment(image, result.seg_mat, result.class_id);
        }
        float x1, y1, x2, y2;
        x1 = std::max<float>(result.box.x1, 0);
        y1 = std::max<float>(result.box.y1, 0);
        x2 = std::max<float>(result.box.x2, 0);
        y2 = std::max<float>(result.box.y2, 0);
        CoordinateTransformation(x1, y1, width, height, target_side_length);
        CoordinateTransformation(x2, y2, width, height, target_side_length);
        if (!enable_track)
        {
            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(0, 0, 255), 2);
            char text[256];
            sprintf(text, "%s %.1f%%", labels.at(result.class_id).c_str(),
                    result.obj_prob * 100);
            cv::putText(image, text, cv::Point(x1, y1 + 20), cv::FONT_HERSHEY_COMPLEX,
                        0.65, cv::Scalar(255, 255, 0), 1, cv::LINE_8);
            if (result.model_type == ModelType::POSE_V8)
            {
                ProcessPoseImage(image, result, target_side_length);
            }
        }
        else
        {
            Object object;
            object.rect = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            object.label = result.class_id;
            object.prob = result.obj_prob;
            objects.push_back(object);
        }
    }
    if (enable_track)
    {
        // static std::unique_ptr<BYTETracker> tracker = nullptr;
        // if (tracker == nullptr)
        // {
        //     tracker = std::make_unique<BYTETracker>(25, 25);
        // }
        // std::vector<STrack> output_stracks = tracker->update(objects);
        // for (size_t i = 0; i < output_stracks.size(); ++i)
        // {
        //     std::vector<float> tlwh = output_stracks[i].tlwh;
        //     bool vertical = tlwh[2] / tlwh[3] > 1.6;
        //     if (tlwh[2] * tlwh[3] > 20 && !vertical)
        //     {
        //         Scalar s = tracker->get_color(output_stracks.at(i).track_id);
        //         putText(image, format("%d,", output_stracks.at(i).track_id),
        //                 Point(tlwh[0], tlwh[1] - 5), 0, 0.6, Scalar(0, 0, 255), 2,
        //                 LINE_AA);
        //         rectangle(image, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
        //     }
        // }
    }
    return image;
}
void ShowAndSave(const cv::Mat &image, int cv_wait_ms, bool is_save)
{
    cv::imshow("Result", image);
    waitKey(cv_wait_ms);
    if (is_save)
    {
        if (gVideoWriter == nullptr)
        {
            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);
            stringstream ss;
            ss << put_time(localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
            string filename = "result_" + ss.str() + ".mp4";
            gVideoWriter = make_shared<KaylordutVideoWriter>(filename, image.cols,
                                                             image.rows, 25.0);
        }
        gVideoWriter->write(image);
    }
}

// #include "framebuffer.h"
// std::shared_ptr<FrameBuffer> kFramebuffer;

void ShowResults(const cv::Mat &original_image, const int target_side_length,
                 const std::vector<Result> &results,
                 std::vector<std::string> &labels, int cv_wait_ms,
                 bool enable_track, bool is_save, bool hdmi_output,
                 bool cv_show)
{
    auto image = GetImageResult(original_image, target_side_length, results,
                                labels, enable_track);
    // if (hdmi_output)
    // {
    //     if (kFramebuffer == nullptr)
    //     {
    //         kFramebuffer = std::make_shared<FrameBuffer>("/dev/fb0");
    //     }
    //     kFramebuffer->WriteFrameBuffer(image);
    // }
    if (cv_show)
    {
        ShowAndSave(image, cv_wait_ms, is_save);
    }
}

std::vector<std::string> ReadLabelsFromTextFile(const std::string &filename)
{
    std::ifstream file(filename);
    std::vector<std::string> labels;
    std::string label;
    if (file.is_open())
    {
        while (getline(file, label))
        {
            labels.emplace_back(label);
        }
        file.close();
    }
    else
    {
        LOG_ERROR("Could not open {}", filename);
        exit(EXIT_FAILURE);
    }
    return labels;
}

double get_current_time()
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

bool ContainsSubString(const std::string &str,
                       const std::string &substring)
{
    return str.find(substring) != std::string::npos;
}

inline int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

template <typename T>
T sigmoid(T x)
{
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
}

int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

// RKNN 非对称量化：zp>0 的层（如 RTMDet 的 score/cls 层）实际以 uint8 存储，
// 必须按 (u8 - zp) × scale 反量化，否则高分字节会被误读成负数 int8。
float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

void compute_dfl(float *tensor, int dfl_len, float *box)
{
    for (int b = 0; b < 4; b++)
    {
        float exp_t[dfl_len];
        float exp_sum = 0;
        float acc_sum = 0;
        for (int i = 0; i < dfl_len; i++)
        {
            exp_t[i] = exp(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }

        for (int i = 0; i < dfl_len; i++)
        {
            acc_sum += exp_t[i] / exp_sum * i;
        }
        box[b] = acc_sum;
    }
}

// IEEE 754 half-precision (float16) → float32 转换
float fp16_to_f32(uint16_t half)
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

int quick_sort_indice_inverse(std::vector<float> &input, int left,
                              int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

float CalculateOverlap(float xmin0, float ymin0, float xmax0,
                       float ymax0, float xmin1, float ymin1,
                       float xmax1, float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) +
              (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

int nms(int validCount, std::vector<Bbox> &bboxes,
        std::vector<int> &order, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        if (order[i] == -1)
        {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1)
            {
                continue;
            }
            float xmin0 = bboxes.at(n).x1;
            float ymin0 = bboxes.at(n).y1;
            float xmax0 = bboxes.at(n).x2;
            float ymax0 = bboxes.at(n).y2;
            float xmin1 = bboxes.at(m).x1;
            float ymin1 = bboxes.at(m).y1;
            float xmax1 = bboxes.at(m).x2;
            float ymax1 = bboxes.at(m).y2;
            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1,
                                         xmax1, ymax1);
            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

int nms(const int validCount,
        const std::vector<Bbox> &bboxes,
        const std::vector<int> classIds, std::vector<int> &order,
        const int filterId, const float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        if (order[i] == -1 || classIds[order[i]] != filterId)
        {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[order[j]] != filterId)
            {
                continue;
            }
            float xmin0 = bboxes.at(n).x1;
            float ymin0 = bboxes.at(n).y1;
            float xmax0 = bboxes.at(n).x2;
            float ymax0 = bboxes.at(n).y2;
            float xmin1 = bboxes.at(m).x1;
            float ymin1 = bboxes.at(m).y1;
            float xmax1 = bboxes.at(m).x2;
            float ymax1 = bboxes.at(m).y2;
            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1,
                                         xmax1, ymax1);
            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}
