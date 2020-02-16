// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/2/10
#include "utils.h"

InputParams::InputParams(int ih, int iw, int ic, int hh, int hw, int hc): image_h(ih), image_w(iw), image_c(ic), heatmap_h(hh), heatmap_w(hw), heatmap_c(hc){

}

std::vector<unsigned  char> imagePreprocess(const std::string &image_path, const int &image_h, const int &image_w){
    // image_path ===> BGR/HWC ===> RGB/CHW
    cv::Mat origin_image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat rgb_image = origin_image;
    cv::cvtColor(origin_image, rgb_image, cv::COLOR_BGR2RGB);
    cv::Mat resized_image(image_h, image_w, CV_8UC3);
    cv::resize(rgb_image, resized_image, cv::Size(image_h, image_w));
    std::vector<unsigned char> file_data(resized_image.reshape(1, 1));
    std::vector<unsigned char> CHW;
    int c, h, w, idx;
    for (int i=0;i<file_data.size();++i){
        w = i % image_w;
        h = i / image_w % image_h;
        c = i / image_w / image_h;
        idx = h * image_w * 3 + w * 3 + c;
        CHW.push_back(file_data[idx]);
    }
    return CHW;
}


cv::Mat renderKeypoint(cv::Mat image, const std::vector<std::vector<float>> &keypoints, int nums_keypoints, float thres=0.3){
    int image_h = image.rows;
    int image_w = image.cols;
    int point_x, point_y;
    for (int i=0; i<nums_keypoints; ++i){
        if (keypoints[i][2]>=thres){
            point_x = image_w * keypoints[i][0];
            point_y = image_h * keypoints[i][1];
            cv::circle(image, cv::Point(point_x, point_y), 5, cv::Scalar(255, 204,0), 3);
        }
    }
    return image;
}


void saveImage(const cv::Mat &image, const std::string &save_path){
    cv::imwrite(save_path, image);
}

