// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2019/12/18
#include "source/keypoints_tensorrt.h"
#include <logger.h>
#include "source/utils.h"
#include "source/my_plugin.h"

const std::string project_name = "TensorRT_Keypoints";
void printHelpInfo()
{
    std::cout << "Usage: ./keypoints [-h or --help] [-d or "
                 "--datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding "
                 "the default. This option can be used multiple times to add "
                 "multiple directories. If no data directories are given, the "
                 "default is to use (data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support "
                 "DLA. Value can range from 0 to n-1, where n is the number of "
                 "DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

samplesCommon::UffSampleParams initial_params(const samplesCommon::Args &args){
    samplesCommon::UffSampleParams params;
    if (args.dataDirs.empty()){
        params.dataDirs.push_back("/work/tensorRT/project/Template/Keypoints/data/images/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.uffFileName = "/work/tensorRT/project/Template/Keypoints/data/uff/keypoints.uff";
    params.inputTensorNames.push_back("Placeholder/inputs_x");
    params.batchSize = 1;
    params.outputTensorNames.push_back("Keypoints/keypoint_1/conv/Sigmoid");
    params.dlaCore = args.useDLACore;
//    params.int8 = args.runInInt8;
    params.int8 = false;
//    params.fp16 = args.runInFp16;
    params.fp16 = false;
    return params;
}

int main(int argc, char **argv){
    REGISTER_TENSORRT_PLUGIN(MyPlugin);

    samplesCommon::Args args;
    if (!samplesCommon::parseArgs(args, argc, argv)){
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    auto sampleTest = Logger::defineTest(project_name, argc, argv);
    Logger::reportTestStart(sampleTest);
    samplesCommon::UffSampleParams params = initial_params(args);
    InputParams input_params(512, 512, 3, 128, 128, 17);
    Keypoints keypoints(params, input_params);
    gLogInfo << "Building and running a GPU inference engine for " << project_name
             << std::endl;
    if (!keypoints.build())
    {
        return Logger::reportFail(sampleTest);
    }
    gLogInfo << "Begine to Infer"
             << std::endl;
    if (!keypoints.infer())
    {
        return Logger::reportFail(sampleTest);
    }
    gLogInfo << "Destroy the engine"
             << std::endl;
    if (!keypoints.tearDown())
    {
        return Logger::reportFail(sampleTest);
    }
    return Logger::reportPass(sampleTest);
}