// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/2/7

#ifndef TENSORRT_METER_TENSORRT_H
#define TENSORRT_METER_TENSORRT_H

#include <argsParser.h>
#include <common.h>
#include <NvUffParser.h>
#include <buffers.h>
#include <dirent.h>
#include "utils.h"


class Keypoints{
public:
    template <typename T>
    using sample_unique_ptr = std::unique_ptr<T, samplesCommon::InferDeleter>;
private:
    std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine{nullptr};
    samplesCommon::UffSampleParams uff_params;
    nvinfer1::Dims input_dims;
    const int image_h;
    const int image_w;
    const int image_c;
    const int heatmap_h;
    const int heatmap_w;
    const int heatmap_c;

    // 将权重赋给网络
    bool constructNetwork(sample_unique_ptr<nvuffparser::IUffParser> &parser, sample_unique_ptr<nvinfer1::INetworkDefinition> &network);
    // 处理输入，读入图片并存入buffer中
    bool processInput(const samplesCommon::BufferManager &buffer_manager, const std::string &input_tensor_name, const std::string &image_path) const;
    // 输出后处理，得到最终结果
    std::vector<std::vector<float>> processOutput(const samplesCommon::BufferManager &buffer_manager, const std::string &output_tensor_name) const;

public:
    explicit Keypoints(samplesCommon::UffSampleParams uff_params, InputParams input_params);
    bool build();
    bool infer();
    bool tearDown();
};

#endif //TENSORRT_METER_TENSORRT_H
