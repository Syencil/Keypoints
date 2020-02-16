// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/2/7

#ifndef TENSORRT_UTILS_H
#define TENSORRT_UTILS_H

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

class InputParams{
public:
    const int image_h;
    const int image_w;
    const int image_c;
    const int heatmap_h;
    const int heatmap_w;
    const int heatmap_c;
    InputParams(int ih, int iw, int ic, int hh, int hw, int hc);
};

std::vector<unsigned char> imagePreprocess(const std::string &image_path, const int &image_h, const int &image_w);

cv::Mat renderKeypoint(cv::Mat image, const std::vector<std::vector<float>> &keypoints, int nums_keypoints, float thres);

void saveImage(const cv::Mat &image, const std::string &save_path);

template <typename T>
void write(char*& buffer, const T& val){
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
T read(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

#endif //TENSORRT_UTILS_H
