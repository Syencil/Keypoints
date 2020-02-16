// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/2/7

#include "keypoints_tensorrt.h"

bool Keypoints::constructNetwork(Keypoints::sample_unique_ptr<nvuffparser::IUffParser> &parser,
                             Keypoints::sample_unique_ptr<INetworkDefinition> &network) {
    assert(uff_params.inputTensorNames.size() == 1);
    assert(uff_params.outputTensorNames.size() == 1);
    if (!parser -> registerInput(uff_params.inputTensorNames[0].c_str(), nvinfer1::Dims3(image_c, image_h, image_w), nvuffparser::UffInputOrder::kNCHW)){
        gLogError << "Register Input Failed!" << std::endl;
        return false;
    }
    if (!parser -> registerOutput(uff_params.outputTensorNames[0].c_str())){
        gLogError << "Register Output Failed!" << std::endl;
        return false;
    }
    if (!parser -> parse(uff_params.uffFileName.c_str(), *network, nvinfer1::DataType::kFLOAT)){
        gLogError << "Parse Uff Failed!" << std::endl;
        return false;
    }
    if (uff_params.int8){
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    return true;
}

bool Keypoints::processInput(const samplesCommon::BufferManager &buffer_manager, const std::string &input_tensor_name,
                         const std::string &image_path) const {
    const int input_h = input_dims.d[1];
    const int input_w = input_dims.d[2];
    std::vector<uint8_t> file_data = imagePreprocess(image_path, image_h, image_w);
    if (file_data.size() != input_h * input_w * image_c){
        gLogError << "FileData size is "<<file_data.size()<<" but needed "<<  input_h * input_w * image_c <<std::endl;
        return false;
    }
    auto *host_input_buffer = static_cast<float*>(buffer_manager.getHostBuffer(input_tensor_name));
    for (int i = 0; i < input_h * input_w * image_c; ++i){
        host_input_buffer[i] = static_cast<float>(file_data[i]) / 128.0 - 1;
    }
    return true;
}

std::vector<std::vector<float>>  Keypoints::processOutput(const samplesCommon::BufferManager &buffer_manager, const std::string &output_tensor_name) const {
    auto *origin_output = static_cast<const float*>(buffer_manager.getHostBuffer(output_tensor_name));
    gLogInfo<< "Output: "<< std::endl;
    //  Keypoint index transformation idx_x, idx_y, prob
    std::vector<std::vector<float>> keypoints;
    for (int c = 0; c < heatmap_c; ++c){
        std::vector<float> keypoint;
        int max_idx = -1;
        float max_prob = -1;
//        for (int idx = heatmap_h * heatmap_w * c; idx < heatmap_h * heatmap_w * (c + 1); ++idx){
//            if (origin_output[idx] > max_prob){
//                max_idx = idx;
//                max_prob = origin_output[idx];
//            }
//        }
//        keypoint.push_back(static_cast<float>(max_idx % heatmap_w)  / heatmap_w);
//        keypoint.push_back(static_cast<float>((max_idx / heatmap_w)  % heatmap_h) / heatmap_h);
        // 迷之操作 输入都是kNCHW 输出怎么就是kNHWC了
        for (int idx = c; idx < heatmap_c * heatmap_h * heatmap_w; idx+=heatmap_c){
            if (origin_output[idx] > max_prob){
                max_idx = idx;
                max_prob = origin_output[idx];
            }
        }
        keypoint.push_back(static_cast<float>(max_idx / heatmap_c % heatmap_w)  / heatmap_w);
        keypoint.push_back(static_cast<float>((max_idx / heatmap_c)  / heatmap_w) / heatmap_h);

        keypoint.push_back(max_prob);
        keypoints.push_back(keypoint);
    }
    for (int c = 0; c < heatmap_c; c++){
        gLogInfo << "channel "<< c << " ==> x : "<< keypoints[c][0] << " y : " << keypoints[c][1] << "  prob : " << keypoints[c][2]<< std::endl;
    }
    return keypoints;
}

Keypoints::Keypoints(samplesCommon::UffSampleParams params, InputParams input_params) : uff_params(std::move(params)),  image_h(input_params.image_h), image_w(input_params.image_w), image_c(input_params.image_c), heatmap_h(input_params.heatmap_h), heatmap_w(input_params.heatmap_w), heatmap_c(input_params.heatmap_c){
    gLogInfo << "Keypoints Construction" << std::endl;
}

bool Keypoints::build() {
    auto builder = sample_unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder){
        gLogError << "Create Builder Failed" << std::endl;
        return false;
    }
    auto network = sample_unique_ptr<nvinfer1::INetworkDefinition>(builder -> createNetworkV2(0U));
    if (!network){
        gLogError << "Create Network Failed" << std::endl;
        return false;
    }
    auto parser = sample_unique_ptr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
    if (!parser){
        gLogError << "Create Parser Failed" << std::endl;
        return false;
    }
    if (!constructNetwork(parser, network)){
        gLogError << "Construct Network Failed" << std::endl;
        return false;
    }

    // 配置config
    builder -> setMaxBatchSize(1);
    auto config = sample_unique_ptr<nvinfer1::IBuilderConfig>(builder -> createBuilderConfig());
    if (!config){
        gLogError << "Create Config Failed" << std::endl;
        return false;
    }
    config -> setMaxWorkspaceSize(1_GiB);
    config -> setFlag(BuilderFlag::kGPU_FALLBACK); // 可以使用DLA加速

    if (uff_params.fp16){
        config -> setFlag(BuilderFlag::kFP16);
    }
    if (uff_params.int8){
        config -> setFlag(BuilderFlag::kINT8);
    }
    samplesCommon::enableDLA(builder.get(), config.get(), uff_params.dlaCore, true);
    cuda_engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder -> buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!cuda_engine){
        gLogError << "Create Config Failed" << std::endl;
        return false;
    }

    assert(network -> getNbInputs() == 1);
    assert(network -> getNbOutputs() == 1);
    input_dims = network -> getInput(0) ->getDimensions();
    assert(input_dims.nbDims == 3);

    gLogInfo << "Build Network Success!" << std::endl;
    return true;
}

bool Keypoints::infer() {
    samplesCommon::BufferManager buffer_manager(cuda_engine, uff_params.batchSize);
    auto context = sample_unique_ptr<nvinfer1::IExecutionContext>(cuda_engine -> createExecutionContext());
    if (!context){
        gLogError << "Create Context Failed" << std::endl;
        return false;
    }
    // 获取问价夹下所有Image图片
    std::vector<std::string> images;
    DIR *dir = opendir(uff_params.dataDirs[0].c_str());
    dirent *p = nullptr;
    gLogInfo << "Fetch images in " <<  uff_params.dataDirs[0]<<std::endl;
    float total{0};
    int count{0};
    while((p = readdir(dir)) != nullptr){
        if ('.' != p->d_name[0] && (strstr(p -> d_name, ".jpg") || strstr(p -> d_name, "png"))){
            std::string imagePath = uff_params.dataDirs[0]+"/"+p->d_name;
            gLogInfo<<"--Image : "<<p->d_name<<std::endl;
            if (!processInput(buffer_manager, uff_params.inputTensorNames[0], imagePath)){
                gLogError<<"Process Input Failed!"<<std::endl;
                return false;
            }
            buffer_manager.copyInputToDevice();
            const auto t_start = std::chrono::high_resolution_clock::now();
            for(int i = 0; i< 1000; ++i)
            if (!context -> execute(uff_params.batchSize, buffer_manager.getDeviceBindings().data())){
                gLogError<<"Execute Failed!"<<std::endl;
                return false;
            }


            const auto t_end = std::chrono::high_resolution_clock::now();
            const float elapsed_time = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            total += elapsed_time;
            buffer_manager.copyOutputToHost();

            // 将输出结果渲染
            std::vector<std::vector<float>> keypoints;
            keypoints = processOutput(buffer_manager, uff_params.outputTensorNames[0]);
            cv::Mat ori_img = cv::imread(imagePath, cv::IMREAD_COLOR);
            cv::Mat render_img = renderKeypoint(ori_img, keypoints, heatmap_c, 0.3);
            saveImage(render_img, imagePath.insert(imagePath.length() - 4, "_render"));

            ++count;
        }
    }
    closedir(dir);
    gLogInfo<< "Total run time is " << total <<" ms\n";
    gLogInfo<< "Average over " << count << " files run time is "<<total / count<<" ms\n";
    return true;
}

bool Keypoints::tearDown() {
    nvuffparser::shutdownProtobufLibrary();
    return true;
}