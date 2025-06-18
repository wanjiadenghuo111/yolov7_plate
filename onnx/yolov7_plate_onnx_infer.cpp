#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

namespace fs = std::filesystem;

// 车牌字符映射表
const std::string plateName = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品";
const std::string plateName2 = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

const float mean_value = 0.588, std_value = 0.193;

// 解码车牌识别结果
std::string decodePlate(const std::vector<int64_t>& preds) {
    int pre = 0;
    std::vector<int64_t> newPreds;
    for (size_t i = 0; i < preds.size(); i++) {
        if (preds[i] != 0 && preds[i] != pre) {
            newPreds.push_back(preds[i]);
        }
        pre = preds[i];
    }
    
    //在 OpenCV 里，cv::putText 函数默认仅支持 ASCII 字符集，所以无法正确绘制汉字。若要绘制汉字，可以借助 FreeType 库，它能支持多种字体，包含中文字体。

    std::string plate;
    for (size_t i = 0; i < newPreds.size(); i++) {
        //只存储 0123456789ABCDEFGHJKLMNPQRSTUVWXYZ 中的字符
        if (newPreds[i] < 42) {
            continue;
        }
        std::cout << "plateName i= " << newPreds[i] << ": "<< plateName2[static_cast<int>(newPreds[i]-42)] << std::endl;
        plate += plateName2[static_cast<int>(newPreds[i]-42)];
    }
    std::cout << "plate is: " << plate << std::endl;

    return plate;
}



// 识别前处理
cv::Mat rec_pre_precessing(const cv::Mat& img, cv::Size size = cv::Size(168, 48)) {
    cv::Mat resized_img;
    // 调整图像大小
    cv::resize(img, resized_img, size);

    cv::Mat float_img;
    // 转换数据类型为 float32 并归一化
    resized_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // 减均值，除标准差
    float_img = (float_img - mean_value) / std_value;

    // 转换维度 (H, W, C) -> (C, H, W)
    std::vector<cv::Mat> channels;
    cv::split(float_img, channels);
    cv::Mat chw;
    cv::merge(channels, chw);
    chw = chw.reshape(1, {3, size.height, size.width});

    // 增加 batch 维度
    chw = chw.reshape(1, {1, 3, size.height, size.width});

    return chw;
}


std::vector<float> rec_pre_processing2(const cv::Mat& img, cv::Size size = cv::Size(168, 48)) {
    // 均值和标准差
    const float mean_value[3] = {0.588f, 0.588f, 0.588f};
    const float std_value[3]  = {0.193f, 0.193f, 0.193f};

    // 1. resize
    cv::Mat resized_img;
    cv::resize(img, resized_img, size);

    // 2. 转 float & 归一化至 0-1
    cv::Mat img_float;
    resized_img.convertTo(img_float, CV_32F, 1.0 / 255.0);

    // 3. 分离通道
    std::vector<cv::Mat> channels(3);
    cv::split(img_float, channels);

    // 4. 按通道归一化
    int h = img_float.rows;
    int w = img_float.cols;
    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - mean_value[c]) / std_value[c];
    }

    // 5. 转为 CHW 并flatten
    std::vector<float> input_data(3 * h * w);
    for (int c = 0; c < 3; ++c) {
        const float* p = (float*)channels[c].data;
        std::copy(p, p + h * w, input_data.begin() + c * h * w);
    }
    // (如需要 batch 维，可以手动 reshape 或直接用 input_data，shape = (1, 3, 48, 168) )
    return input_data;
}



// 车牌识别
std::string get_plate_result(const cv::Mat& img, Ort::Session& session) {
   
    cv::Mat float_img = img.clone();
    cv::cvtColor(img, float_img, cv::COLOR_RGB2BGR);  // 确保保存为BGR格式
    cv::imwrite("get_plate_result.jpg", float_img);

    std::vector<float> input_data  = rec_pre_processing2(float_img);
      
#if 0
    {
            // 文件路径（请根据实际情况修改）
        std::string filePath = "recinput_data.txt";
        
        // 打开文件
        std::ifstream file1(filePath);
        if (!file1.is_open()) {
            std::cerr << "无法打开文件: " << filePath << std::endl;
            return "";
        }
        
        
        // 从文件中读取数据
        std::string line;
        while (std::getline(file1, line)) {
            // 处理每行数据，使用字符串流分割数值
            std::istringstream iss(line);
            //打印line
            //std::cout<< "car line: " << line << std::endl;

            float value;
            while (iss >> value) {
                input_data.push_back(value);
            }
        }
        
        std::cout<< "input_data.size : " << input_data.size() << std::endl;
        // 关闭文件
        file1.close();

    }
         
#endif

    // 准备输入tensor
    std::vector<int64_t> input_shape = {1, 3, 48, 168};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_value = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size()
        );
    
    // 获取输入输出名称，使用智能指针存储
    auto input_name_ptr = session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto output_name_ptr = session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    const char* input_names[] = {input_name_ptr.get()};
    const char* output_names[] = {output_name_ptr.get()};

    // 打印输入输出名称 input_names 和 output_names 
    std::cout<< "session input_names: " << input_names[0] << std::endl;
    std::cout<< "session output_names: " << output_names[0] << std::endl;


    // 运行推理
    Ort::RunOptions run_options;
    auto output_tensors = session.Run(
        run_options, 
        input_names, 
        &input_value, 
        1, 
        output_names, 
        1
    );


    // 获取输出
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const int seq_length = output_shape[1];
    const int vocab_size = output_shape[2];
    
    // 解码输出
    std::vector<int64_t> indexes;
    for (int i = 0; i < seq_length; i++) {
        float* row = output_data + i * vocab_size;
        int max_index = static_cast<int>(std::max_element(row, row + vocab_size) - row);
        indexes.push_back(max_index);
    }
    
    return decodePlate(indexes);
}



// Letter box处理
std::tuple<cv::Mat, float, int, int> my_letter_box(const cv::Mat& img, const cv::Size& size) {
    // 检查图像尺寸是否为0
    if (img.cols == 0 || img.rows == 0) {
        throw std::runtime_error("Input image has zero width or height");
    }

    float r = std::min(static_cast<float>(size.width) / img.cols, 
                      static_cast<float>(size.height) / img.rows);
    int new_w = static_cast<int>(img.cols * r);
    int new_h = static_cast<int>(img.rows * r);
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    
    int top = (size.height - new_h) / 2;
    int bottom = size.height - new_h - top;
    int left = (size.width - new_w) / 2;
    int right = size.width - new_w - left;
    
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    return {padded, r, left, top};
}

// 坐标转换
std::vector<float> xywh2xyxy(const std::vector<float>& box) {
    float x = box[0], y = box[1], w = box[2], h = box[3];
    return {
        x - w / 2, 
        y - h / 2, 
        x + w / 2, 
        y + h / 2
    };
}

// 检测前处理
std::tuple<cv::Mat, float, int, int> detect_pre_processing(const cv::Mat& img, const cv::Size& img_size) {
    // 检查输入图像是否为空
    if (img.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    auto [padded, r, left, top] = my_letter_box(img, img_size);
    
     std::cerr << "my_letter_box: " <<std::strerror(errno) << std::endl;


    // 将 BGR 转换为 RGB
    cv::Mat rgb_img;
    cv::cvtColor(padded, rgb_img, cv::COLOR_BGR2RGB);
    
    // 转换为 float 类型并归一化
    cv::Mat float_img;
    rgb_img.convertTo(float_img, CV_32FC3, 1.0/255.0);
    
    std::cerr << "convertTo: " <<std::strerror(errno) << std::endl;

    // 手动转换维度 (H, W, C) -> (1, C, H, W)
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    cv::Mat chw = cv::Mat::zeros(1, 3 * float_img.rows * float_img.cols, CV_32FC1);
    for (int i = 0; i < 3; ++i) {
        cv::Mat channel_flat = channels[i].reshape(1, 1);
        if (channel_flat.total() != float_img.rows * float_img.cols) {
            throw std::runtime_error("Channel flattening failed");
        }
        channel_flat.copyTo(chw.colRange(i * float_img.rows * float_img.cols, (i + 1) * float_img.rows * float_img.cols));
    }

    std::cerr << "reshape before: " <<std::strerror(errno) << std::endl;

    // 确保数据连续
    if (!chw.isContinuous()) {
        chw = chw.clone();
    }

    // 先重塑为 3 维
    chw = chw.reshape(3, {1, 3 * float_img.rows, float_img.cols});
    // 再调整为 (1, 3, H, W) 形状
    chw = chw.reshape(3, {1, 3, float_img.rows, float_img.cols});

    std::cerr << "reshape after: " <<std::strerror(errno) << std::endl;


    return {chw, r, left, top};
}


// 识别后处理
std::string decodePlate(const std::vector<int>& preds) {
    int pre = 0;
    std::vector<int> newPreds;
    for (int i = 0; i < preds.size(); ++i) {
        if (preds[i] != 0 && preds[i] != pre) {
            newPreds.push_back(preds[i]);
        }
        pre = preds[i];
    }
    std::string plate;
    for (int i : newPreds) {
        plate += plateName[i];
    }
    return plate;
}


// 关键点排列 按照（左上，右上，右下，左下）的顺序排列
cv::Mat order_points(const cv::Mat& pts) {
    cv::Mat rect = cv::Mat::zeros(4, 2, CV_32F);
    
    // 计算每个点的坐标和 (x + y)
    std::vector<float> sums;
    for (int i = 0; i < pts.rows; ++i) {
        sums.push_back(pts.at<float>(i, 0) + pts.at<float>(i, 1));
    }
    
    // 左上角 (x + y 最小)
    auto min_sum_it = std::min_element(sums.begin(), sums.end());
    pts.row(std::distance(sums.begin(), min_sum_it)).copyTo(rect.row(0));
    
    // 右下角 (x + y 最大)
    auto max_sum_it = std::max_element(sums.begin(), sums.end());
    pts.row(std::distance(sums.begin(), max_sum_it)).copyTo(rect.row(2));
    
    // 计算每个点的坐标差 (x - y)
    std::vector<float> diffs;
    for (int i = 0; i < pts.rows; ++i) {
        diffs.push_back(pts.at<float>(i, 0) - pts.at<float>(i, 1));
    }
    
    // 右上角 (x - y 最大)
    auto max_diff_it = std::max_element(diffs.begin(), diffs.end());
    pts.row(std::distance(diffs.begin(), max_diff_it)).copyTo(rect.row(1));
    
    // 左下角 (x - y 最小)
    auto min_diff_it = std::min_element(diffs.begin(), diffs.end());
    pts.row(std::distance(diffs.begin(), min_diff_it)).copyTo(rect.row(3));
    
    return rect;
}

// 透视变换得到矫正后的图像，方便识别
cv::Mat four_point_transform(const cv::Mat& image, const cv::Mat& pts) {
    cv::Mat rect = order_points(pts);
    cv::Point2f tl = rect.at<cv::Point2f>(0);
    cv::Point2f tr = rect.at<cv::Point2f>(1);
    cv::Point2f br = rect.at<cv::Point2f>(2);
    cv::Point2f bl = rect.at<cv::Point2f>(3);

    // 打印排序后的关键点坐标
    std::cout << "Top-left: (" << tl.x << ", " << tl.y << ")" << std::endl;
    std::cout << "Top-right: (" << tr.x << ", " << tr.y << ")" << std::endl;
    std::cout << "Bottom-right: (" << br.x << ", " << br.y << ")" << std::endl;
    std::cout << "Bottom-left: (" << bl.x << ", " << bl.y << ")" << std::endl;

    float widthA = cv::norm(br - bl);
    float widthB = cv::norm(tr - tl);
    float maxWidth = std::max(widthA, widthB);

    float heightA = cv::norm(tr - br);
    float heightB = cv::norm(tl - bl);
    float maxHeight = std::max(heightA, heightB);

    std::cout << "Calculated widthA: " << widthA << std::endl;
    std::cout << "Calculated widthB: " << widthB << std::endl;
    std::cout << "Calculated maxWidth: " << maxWidth << std::endl;
    std::cout << "Calculated heightA: " << heightA << std::endl;
    std::cout << "Calculated heightB: " << heightB << std::endl;
    std::cout << "Calculated maxHeight: " << maxHeight << std::endl;

    // 处理 maxWidth 或 maxHeight 为 0 的情况
    if (maxWidth <= 0 || maxHeight <= 0) {
        std::cerr << "Invalid maxWidth or maxHeight, returning original image." << std::endl;
        return image.clone();
    }

    std::vector<cv::Point2f> dst = {
        {0, 0},
        {maxWidth - 1, 0},
        {maxWidth - 1, maxHeight - 1},
        {0, maxHeight - 1}
    };

    cv::Mat M = cv::getPerspectiveTransform(rect, dst);
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, cv::Size(maxWidth, maxHeight));

    //打印出 warped 更多的值
    std::cout << "Warped matrix dimensions: " << warped.cols << "x" << warped.rows << std::endl;
    std::cout << "Warped matrix type: " << warped.type() << std::endl;
    std::cout << "Warped matrix channels: " << warped.channels() << std::endl;
    std::cout << "Warped matrix depth: " << warped.depth() << std::endl;
    std::cout << "Warped matrix size: " << warped.total() << std::endl;
    std::cout << "Warped matrix step: " << warped.step << std::endl;
    std::cout << "Warped matrix rows: " << warped.rows << std::endl;

     //保存图片
    cv::imwrite("image_warped.jpg", image);

    //保存图片
    cv::imwrite("warped.jpg", warped);

    return warped;
}

// 双层车牌进行分割后识别
cv::Mat get_split_merge(const cv::Mat& img) {
    int h = img.rows;
    int w = img.cols;
    cv::Mat img_upper = img(cv::Rect(0, 0, w, static_cast<int>(5.0 / 12.0 * h)));
    cv::Mat img_lower = img(cv::Rect(0, static_cast<int>(1.0 / 3.0 * h), w, h - static_cast<int>(1.0 / 3.0 * h)));
    cv::Mat resized_upper;
    cv::resize(img_upper, resized_upper, cv::Size(img_lower.cols, img_lower.rows));
    cv::Mat new_img;
    cv::hconcat(resized_upper, img_lower, new_img);
    return new_img;
}

// 识别车牌
struct Result {
    std::vector<float> rect;
    std::vector<std::vector<float>> landmarks;
    std::string plate_no;
    int roi_height;
};

std::vector<Result> rec_plate(const std::vector<std::vector<float>>& outputs, const cv::Mat& img0, Ort::Session& session_rec) {
    std::vector<Result> dict_list;
    for (const auto& output : outputs) {
        Result result_dict;
        result_dict.rect.assign(output.begin(), output.begin() + 4);

        cv::Mat land_marks(4, 2, CV_32F);
        for (int i = 0; i < 4; ++i) {
            land_marks.at<float>(i, 0) = output[5 + 2 * i];
            land_marks.at<float>(i, 1) = output[5 + 2 * i + 1];
        }
        result_dict.landmarks.resize(4, std::vector<float>(2));
        for (int i = 0; i < 4; ++i) {
            result_dict.landmarks[i][0] = land_marks.at<float>(i, 0);
            result_dict.landmarks[i][1] = land_marks.at<float>(i, 1);
        }

        // 打印原始关键点坐标，确认输入是否正确
        std::cout << "Original landmarks:" << std::endl;
        for (int i = 0; i < 4; ++i) {
            std::cout << "(" << land_marks.at<float>(i, 0) << ", " << land_marks.at<float>(i, 1) << ")" << std::endl;
        }

        cv::Mat ordered_pts = order_points(land_marks);
        // 打印排序后的关键点坐标
        std::cout << "Ordered landmarks:" << std::endl;
        for (int i = 0; i < 4; ++i) {
            std::cout << "(" << ordered_pts.at<float>(i, 0) << ", " << ordered_pts.at<float>(i, 1) << ")" << std::endl;
        }

        cv::Mat roi_img = four_point_transform(img0, ordered_pts);
        cv::cvtColor(roi_img, roi_img, cv::COLOR_RGB2BGR);  // 确保保存为BGR格式
        cv::imwrite("roi_img.jpg", roi_img);


        int label = static_cast<int>(output.back());
        // 如果 label 等于 1，认为是双层车牌，进行分割合并操作
        if (label == 1) {
            std::cout << "------------ 如果 label 等于 1,认为是双层车牌，进行分割合并操作: " << label << std::endl;
            roi_img = get_split_merge(roi_img);
        }
        // 打印 roi_img 的高度
        std::cout << "ROI image height before assignment: " << roi_img.rows << std::endl;
        result_dict.roi_height = roi_img.rows;
        result_dict.plate_no = get_plate_result(roi_img, session_rec);


        result_dict.roi_height = roi_img.rows;
        dict_list.push_back(result_dict);
    }
    return dict_list;
}
// xywh 坐标转换为 xyxy 坐标
std::vector<std::vector<float>> xywh2xyxy(const std::vector<std::vector<float>>& boxes) {
    std::vector<std::vector<float>> xywh = boxes;
    for (size_t i = 0; i < boxes.size(); ++i) {
        xywh[i][0] = boxes[i][0] - boxes[i][2] / 2;
        xywh[i][1] = boxes[i][1] - boxes[i][3] / 2;
        xywh[i][2] = boxes[i][0] + boxes[i][2] / 2;
        xywh[i][3] = boxes[i][1] + boxes[i][3] / 2;
    }
    return xywh;
}

// 非极大值抑制
std::vector<int> my_nms(const std::vector<std::vector<float>>& boxes, float iou_thresh) {
    std::vector<int> indices(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        indices[i] = static_cast<int>(i);
    }
    std::sort(indices.begin(), indices.end(), [&boxes](int a, int b) {
        return boxes[a][4] > boxes[b][4];
    });

    std::vector<int> keep;
    while (!indices.empty()) {
        int i = indices[0];
        keep.push_back(i);

        std::vector<int> new_indices;
        for (size_t j = 1; j < indices.size(); ++j) {
            int k = indices[j];
            float x1 = std::max(boxes[i][0], boxes[k][0]);
            float y1 = std::max(boxes[i][1], boxes[k][1]);
            float x2 = std::min(boxes[i][2], boxes[k][2]);
            float y2 = std::min(boxes[i][3], boxes[k][3]);

            float w = std::max(0.0f, x2 - x1);
            float h = std::max(0.0f, y2 - y1);

            float inter_area = w * h;
            float union_area = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) +
                               (boxes[k][2] - boxes[k][0]) * (boxes[k][3] - boxes[k][1]) - inter_area;
            float iou = inter_area / union_area;

            if (iou <= iou_thresh) {
                new_indices.push_back(k);
            }
        }
        indices = new_indices;
    }
    return keep;
}

// 恢复到原图坐标
std::vector<std::vector<float>> restore_box(const std::vector<std::vector<float>>& boxes, float r, float left, float top) {
    std::vector<std::vector<float>> result = boxes;
    for (size_t i = 0; i < boxes.size(); ++i) {
        result[i][0] -= left;
        result[i][2] -= left;
        result[i][5] -= left;
        result[i][7] -= left;
        result[i][9] -= left;
        result[i][11] -= left;

        result[i][1] -= top;
        result[i][3] -= top;
        result[i][6] -= top;
        result[i][8] -= top;
        result[i][10] -= top;
        result[i][12] -= top;

        result[i][0] /= r;
        result[i][2] /= r;
        result[i][5] /= r;
        result[i][7] /= r;
        result[i][9] /= r;
        result[i][11] /= r;

        result[i][1] /= r;
        result[i][3] /= r;
        result[i][6] /= r;
        result[i][8] /= r;
        result[i][10] /= r;
        result[i][12] /= r;
    }
    return result;
}

// 检测后处理
std::vector<std::vector<float>> post_precessing(const std::vector<std::vector<std::vector<float>>>& dets, float r, float left, float top, float conf_thresh = 0.3, float iou_thresh = 0.45) {
    const int num_cls = 2;
    std::vector<std::vector<float>> filtered_dets;
    // 置信度阈值筛选
    for (const auto& det : dets) {
        for (const auto& box : det) {
            if (box[4] > conf_thresh) {
                std::vector<float> new_box = box;
                for (int j = 5; j < 5 + num_cls; ++j) {
                    new_box[j] *= new_box[4];
                }
                filtered_dets.push_back(new_box);
            }
        }
    }

    // 提取边界框
    std::vector<std::vector<float>> boxes;
    for (const auto& det : filtered_dets) {
        boxes.emplace_back(det.begin(), det.begin() + 4);
    }

    // 转换坐标
    std::vector<std::vector<float>> xyxy_boxes = xywh2xyxy(boxes);

    // 提取分数和类别索引
    std::vector<float> scores;
    std::vector<int> indices;
    for (const auto& det : filtered_dets) {
        float max_score = 0;
        int max_index = 0;
        for (int j = 5; j < 5 + num_cls; ++j) {
            if (det[j] > max_score) {
                max_score = det[j];
                max_index = j - 5;
            }
        }
        scores.push_back(max_score);
        indices.push_back(max_index);
    }

    // 提取关键点
    std::vector<std::vector<float>> landmarks;
    const int kpt_b = 5 + num_cls;
    for (const auto& det : filtered_dets) {
        landmarks.push_back({det[kpt_b], det[kpt_b + 1], det[kpt_b + 3], det[kpt_b + 4], det[kpt_b + 6], det[kpt_b + 7], det[kpt_b + 9], det[kpt_b + 10]});
    }

    // 拼接结果
    std::vector<std::vector<float>> output;
    for (size_t i = 0; i < xyxy_boxes.size(); ++i) {
        std::vector<float> row = xyxy_boxes[i];
        row.push_back(scores[i]);
        row.insert(row.end(), landmarks[i].begin(), landmarks[i].end());
        row.push_back(static_cast<float>(indices[i]));
        output.push_back(row);
    }

    // 非极大值抑制
    std::vector<int> keep = my_nms(output, iou_thresh);

    // 筛选保留的结果
    std::vector<std::vector<float>> final_output;
    for (int index : keep) {
        final_output.push_back(output[index]);
    }

    // 恢复到原图坐标
    final_output = restore_box(final_output, r, left, top);

    return final_output;
}
// 检测后处理

class Car { 


public:
    Car() {}
    ~Car() {}



void PrintModelIOInfo(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    
    // 获取输入信息
    size_t num_input_nodes = session.GetInputCount();
    std::cout << "Input information" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        // 获取输入名称
        auto input_name = session.GetInputNameAllocated(i, allocator);
        
        // 获取输入类型信息
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        // 获取数据类型和形状
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::vector<int64_t> shape = tensor_info.GetShape();
        
        // 打印输入信息
        std::cout << "ValueInfo \"" << input_name.get() << "\": type ";
        
        // 数据类型转换
        switch (type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: std::cout << "FLOAT"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: std::cout << "UINT8"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: std::cout << "INT8"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: std::cout << "UINT16"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: std::cout << "INT16"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: std::cout << "INT32"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: std::cout << "INT64"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: std::cout << "DOUBLE"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: std::cout << "UINT32"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: std::cout << "UINT64"; break;
            default: std::cout << "UNKNOWN(" << static_cast<int>(type) << ")"; break;
        }
        
        // 打印形状信息
        std::cout << ", shape [";
        for (size_t j = 0; j < shape.size(); j++) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    // 获取输出信息
    size_t num_output_nodes = session.GetOutputCount();
    std::cout << "\nOutput information" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        // 获取输出名称
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        
        // 获取输出类型信息
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        // 获取数据类型和形状
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::vector<int64_t> shape = tensor_info.GetShape();
        
        // 打印输出信息
        std::cout << "ValueInfo \"" << output_name.get() << "\": type ";
        
        // 数据类型转换
        switch (type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: std::cout << "FLOAT"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: std::cout << "UINT8"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: std::cout << "INT8"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: std::cout << "UINT16"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: std::cout << "INT16"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: std::cout << "INT32"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: std::cout << "INT64"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: std::cout << "DOUBLE"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: std::cout << "UINT32"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: std::cout << "UINT64"; break;
            default: std::cout << "UNKNOWN(" << static_cast<int>(type) << ")"; break;
        }
        
        // 打印形状信息
        std::cout << ", shape [";
        for (size_t j = 0; j < shape.size(); j++) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

// LetterBox 缩放 + 填充
cv::Mat my_letter_box1(const cv::Mat& img, const cv::Size& size, float& r, int& left, int& top) {
    int h = img.rows;
    int w = img.cols;

    // 计算缩放比例
    r = std::min(static_cast<float>(size.height) / h, static_cast<float>(size.width) / w);

    int new_h = static_cast<int>(h * r);
    int new_w = static_cast<int>(w * r);

    top = (size.height - new_h) / 2;
    left = (size.width - new_w) / 2;

    // 缩放图像
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    // 创建目标图像并填充黑边
    cv::Mat letterboxed;
    cv::copyMakeBorder(resized, letterboxed,
                       top, size.height - new_h - top,
                       left, size.width - new_w - left,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));  // BGR

    return letterboxed;
}

// 整体检测前处理
std::vector<float> detect_pre_precessing1(const cv::Mat& img, const cv::Size& img_size, float& r, int& left, int& top) {
    // Step 1: Letterbox (与Python版本一致)
    cv::Mat letterboxed = my_letter_box1(img, img_size, r, left, top);
    cv::imwrite("debug_letterbox.jpg", letterboxed);

    // Step 2: BGR -> RGB (与Python版本一致)
    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    cv::imwrite("debug_rgb.jpg", rgb);

    // Step 3: 转换为CHW格式 (简化版，与Python一致)
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0/255.0);
    
    // 直接转换为连续内存的1x3xHxW张量
    std::vector<float> input_data(float_img.total() * 3);
    memcpy(input_data.data(), float_img.data, float_img.total() * 3 * sizeof(float));
    
    return input_data;
}


#include <opencv2/opencv.hpp>
#include <tuple>

// Letter box function
std::tuple<cv::Mat, float, int, int> my_letter_box2(cv::Mat img, cv::Size size = cv::Size(640, 640)) {
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();
    float r = std::min(size.height * 1.0f / h, size.width * 1.0f / w);
    int new_h = int(h * r);
    int new_w = int(w * r);
    int top = int((size.height - new_h) / 2.0);
    int left = int((size.width - new_w) / 2.0);
    int bottom = size.height - new_h - top;
    int right = size.width - new_w - left;

    // Resize
    cv::Mat img_resize;
    cv::resize(img, img_resize, cv::Size(new_w, new_h));

    // Padding
    cv::Mat img_padded;
    cv::copyMakeBorder(img_resize, img_padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return std::make_tuple(img_padded, r, left, top);
}

// detect_pre_processing function
std::tuple<cv::Mat, float, int, int> detect_pre_processing2(cv::Mat img, cv::Size img_size) {
    cv::Mat letterbox_img;
    float r;
    int left, top;
    std::tie(letterbox_img, r, left, top) = my_letter_box2(img, img_size);

    // BGR to RGB
    cv::Mat img_rgb;
    cv::cvtColor(letterbox_img, img_rgb, cv::COLOR_BGR2RGB);

    // HWCH to CHW and normalization
    img_rgb.convertTo(img_rgb, CV_32F, 1.0 / 255.0); // float32 and /255

    // Change layout: HWC -> CHW (use cv::dnn::blobFromImage)
    std::vector<cv::Mat> rgb_channels(3);
    cv::split(img_rgb, rgb_channels);

    cv::Mat chw_img;
    cv::vconcat(rgb_channels, chw_img); // Concatenate along channel dimension

    // Reshape to batch (1, C, H, W)
    chw_img = chw_img.reshape(1, {1, 3, img_rgb.rows, img_rgb.cols});

    return std::make_tuple(chw_img, r, left, top);
}


private:

std::map <std::string, std::string> image_index;


public: 



bool buildIndex(const std::string& imagePath,bool force_build=false) 
{
    // 参数解析
    std::string detect_model = "../weights/yolov7-lite-s.onnx";
    std::string rec_model = "../weights/plate_rec_color.onnx";
    std::string output_path = "chepairesult";
    int img_size = 640;
    
    // 创建输出目录
    if (!fs::exists(output_path)) {
        fs::create_directory(output_path);
    }
    
    // 初始化ONNX Runtime环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "LicensePlateRecognition");
    Ort::SessionOptions session_options;
    //session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    // 加载模型
    Ort::Session detect_session(env, detect_model.c_str(), session_options);
    Ort::Session rec_session(env, rec_model.c_str(), session_options);
    
      //yolov7-lite-s.onnx 
        /*
        Input information
        --------------------------------------------------------------------------------
        ValueInfo "images": type FLOAT, shape [1, 3, 640, 640],
        Output information
        --------------------------------------------------------------------------------
        ValueInfo "output": type FLOAT, shape [1, 25200, 19],
        */
        //打印Input 和 Output信息
    PrintModelIOInfo(detect_session);
    PrintModelIOInfo(rec_session);

    // 处理所有图片
    std::vector<std::string> image_files = {imagePath};
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (const auto& file : image_files) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // 读取图像
        cv::Mat img = cv::imread(file);
        if (img.empty()) {
            std::cerr << "无法读取图像: " << file << std::endl;
            continue;
        }
        
        std::cout<< "car 读取图像: " << file << std::endl;

        // 检测前处理
        float r;
        int left, top;
        cv::Size img_size_s(img_size, img_size);

        //std::vector<float> input_data = detect_pre_precessing(img, img_size_s, r, left, top);
        std::tuple<cv::Mat, float, int, int>  ddata = detect_pre_processing2(img, img_size_s);
        // 使用 std::get 获取 cv::Mat 并转换为 std::vector<float>
        cv::Mat processed_img = std::get<0>(ddata);
        //std::vector<float> input_data(processed_img.data, processed_img.data + processed_img.total() * processed_img.channels());
        std::vector<float> input_data(processed_img.ptr<float>(), processed_img.ptr<float>() + processed_img.total());


        r = std::get<1>(ddata);
        left = std::get<2>(ddata);
        top = std::get<3>(ddata);

        std::cout<< "car detect_pre_processing over: " << std::endl;
         for (int i = 0; i < 640; i++) {
            //std::cout << input_data[i] << " ";
        }

        #if 0
        {
            // 文件路径（请根据实际情况修改）
        std::string filePath = "input_data.txt";
        
        // 打开文件
        std::ifstream file1(filePath);
        if (!file1.is_open()) {
            std::cerr << "无法打开文件: " << filePath << std::endl;
            return 1;
        }
        
        // 用于存储数据的vector
        std::vector<float> input_data1;
        
        // 从文件中读取数据
        std::string line;
        while (std::getline(file1, line)) {
            // 处理每行数据，使用字符串流分割数值
            std::istringstream iss(line);
            //打印line
            //std::cout<< "car line: " << line << std::endl;

            float value;
            while (iss >> value) {
                input_data1.push_back(value);
            }
        }
        
        std::cout<< "input_data.size : " << input_data.size() << std::endl;
        std::cout<< "input_data1.size : " << input_data1.size() << std::endl;
         
        input_data = input_data1;
        
        // 关闭文件
        file1.close();

        }
        #endif 

        // 构造输入张量
        std::vector<int64_t> input_shape = {1, 3, img_size, img_size};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_value = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size()
        );

        #if 0
        {

        
            // 在运行模型前添加input_value
            std::cout<< "car input_data_ptr: " << std::endl;
            float* input_data_ptr = input_value.GetTensorMutableData<float>();
            for (int i = 0; i < 512; i++) {
                std::cout << input_data_ptr[i] << " ";
            }
            std::cout<<  std::endl;
        }
        #endif

        std::cout<< "car CreateTensor: " << std::endl;

        // 获取输入输出名称，使用智能指针存储
        auto input_name_ptr = detect_session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
        auto output_name_ptr = detect_session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
        const char* input_names[] = {input_name_ptr.get()};
        const char* output_names[] = {output_name_ptr.get()};

        // 打印输入输出名称 input_names 和 output_names 
        std::cout<< "car input_names: " << input_names[0] << std::endl;
        std::cout<< "car output_names: " << output_names[0] << std::endl;

      
        // 运行检测
        Ort::RunOptions run_options;
        auto outputs = detect_session.Run(
            run_options, 
            input_names, 
            &input_value, 
            1, 
            output_names, 
            1
        );
        
        #if 0
        {
            // 在获取输出后添加验证
            std::cout<< "car outputs: " << std::endl;
            float* dets = outputs[0].GetTensorMutableData<float>();
            for (int i = 0; i < 512; i++) {
                std::cout << dets[i] << " ";
            }
            std::cout<<  std::endl;
        }
        #endif

        std::cout<< "car detect_session: " << file << std::endl;

        // 获取检测结果
        float* dets = outputs[0].GetTensorMutableData<float>();
        auto dets_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t num_dets = dets_shape[1] * dets_shape[2];
        std::vector<float> dets_vector(dets, dets + num_dets);

        // 假设 dets_shape 为 [1, 25200, 19]
        const int batch_size = dets_shape[0];
        const int num_boxes = dets_shape[1];
        const int box_attrs = dets_shape[2];
        //打印dets_shape
        std::cout<< "car dets_shape: " << batch_size << " " << num_boxes << " " << box_attrs << std::endl;

        #if 0
        //检查模型输出的形状和数据结构是否与代码中的解析逻辑一致。
        std::cout<< "car detect_session: dets_vector " << dets_vector.size() << std::endl;

        std::cout << "模型输出前50个值: ";
        for (int i = 0; i < 50; i++) {
        std::cout << dets_vector[i] << " ";
        }
        std::cout<<  std::endl;

        //sleep(10);
        #endif
        
        // 将一维向量转换为三维向量
        std::vector<std::vector<std::vector<float>>> dets_3d(batch_size, std::vector<std::vector<float>>(num_boxes, std::vector<float>(box_attrs)));
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < num_boxes; ++i) {
                for (int j = 0; j < box_attrs; ++j) {
                    dets_3d[b][i][j] = dets[b * num_boxes * box_attrs + i * box_attrs + j];
                }
            }
        }


        // 检测后处理
        auto results = post_precessing(dets_3d, r, left, top);
        std::cout<< "car post_processing: find car plate obj results size : " << results.size() << std::endl;
         // 打印结果
        for (const auto& row : results) {
            for (float val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }


        //sleep(100);

        // 车牌识别
        std::vector<Result> result_list = rec_plate(results, img, rec_session);

        // 打印 Result List vector values
        std::cout << "Result List vector values:" << std::endl;
        for (size_t i = 0; i < result_list.size(); ++i) {
            const auto& result = result_list[i];
            std::cout << "Element " << i << ":" << std::endl;
            
            // 打印 Rect
            std::cout << "  Rect: [";
            for (size_t j = 0; j < result.rect.size(); ++j) {
                std::cout << result.rect[j];
                if (j < result.rect.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;

            // 打印 Landmarks
            std::cout << "  Landmarks: [";
            for (size_t j = 0; j < result.landmarks.size(); ++j) {
                std::cout << "[";
                for (size_t k = 0; k < result.landmarks[j].size(); ++k) {
                    std::cout << result.landmarks[j][k];
                    if (k < result.landmarks[j].size() - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]";
                if (j < result.landmarks.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;

            // 打印 Plate No
            std::cout << "  Plate No: " << result.plate_no << std::endl;

            // 打印 ROI Height
            std::cout << "  ROI Height: " << result.roi_height << std::endl;

            // 单独打印车牌号码
            std::cout << result.plate_no << std::endl;
        }


         cv::cvtColor(img, img, cv::COLOR_RGB2BGR);  // 确保保存为BGR格式

            // 绘制结果
        for (const auto& result : result_list) {
            // 绘制边界框
            cv::Point2f pt1(result.rect[0], result.rect[1]);
            cv::Point2f pt2(result.rect[2], result.rect[3]);
            cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0), 2);

            // 绘制关键点
            for (const auto& landmark : result.landmarks) {
                cv::Point2f point(landmark[0], landmark[1]);
                cv::circle(img, point, 5, cv::Scalar(0, 0, 255), -1);
            }

            std::string plate_no = "粤AGC5967";
            // 绘制车牌号码
            cv::putText(img, result.plate_no, cv::Point(result.rect[0], result.rect[1] - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                        
                        
            // 保存结果
            std::string output_file = output_path + "/" + fs::path(file).filename().string();
            cv::imwrite(output_file, img);
            
            //result.plate_no,output_file 存储到 image_index
            image_index[result.plate_no] = output_file;
            std::cout<< "car image_index: " << result.plate_no << ":" << output_file << std::endl;
        }


        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "处理: " << file << " 耗时: " << duration.count() << "ms" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "总共处理 " << image_files.size() << " 张图片, 总耗时: " 
              << total_duration.count() << "ms" << std::endl;
    
    return 0;
}


void showAllCarPlate()
{
    for (const auto& pair : image_index) {
        std::cout << "车牌号码: " << pair.first << ", 图片路径: " << pair.second << std::endl;
    }   
}

std::vector<std::pair<std::string, float>> searchText(const std::string& carnumber, size_t topK = 10) 
{
    std::cout << "searchText carnumber: " << carnumber;
    auto it = image_index.find(carnumber);
    if(it == image_index.end())
    {
        std::cout << "未找到 carnumber :" << carnumber << std::endl;
        return {};
    }

    std::vector<std::pair<std::string, float>> data;
    data.push_back(std::make_pair(it->second, 1.0f));

    std::cout << "找到 carnumber :" << it->second << std::endl;

    return data;
}

};

std::vector<std::string> getImagePathsFromDirectory(const std::string &dirPath)
{
    std::vector<std::string> imagePaths;
    DIR *dir = opendir(dirPath.c_str());
    if (dir == nullptr)
    {
        return imagePaths;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string filename = entry->d_name;
        // 简单判断图片扩展名
        if (filename.find("jpg") != std::string::npos || filename.find("png") != std::string::npos || filename.find("jpeg") != std::string::npos && filename.find("out") == std::string::npos)
        {
            imagePaths.push_back(dirPath + "/" + filename);
        }
    }

    closedir(dir);
    return imagePaths;
}

int main(int argc, char** argv)
{
    Car car;

// 构建索引（扫描图像目录）或者 解码流图片
    std::vector<std::string> images = getImagePathsFromDirectory("../imgs");
    std::cout <<"getImagePathsFromDirectory,size=%d  "<<images.size();

     // 遍历 image_files 中的每个图片地址
    for (const auto &image_file : images)
    {
        car.buildIndex(image_file,true);
    }
    car.showAllCarPlate();

    std::vector<std::pair<std::string, float>> results = car.searchText("AGC5967"); 
}
