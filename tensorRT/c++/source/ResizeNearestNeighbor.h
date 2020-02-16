// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/2/11

#ifndef TENSORRT_RESIZENEARESTNEIGHBOR_H
#define TENSORRT_RESIZENEARESTNEIGHBOR_H

#include <cassert>
#include <NvInferPlugin.h>

#include "utils.h"

class UffUpSamplePluginV2 : public nvinfer1::IPluginV2IOExt{
private:
    nvinfer1::Dims mCHW;
    nvinfer1::DataType mDataType;
    float mScale;
    int mOutputHeight;
    int mOutputWidth;

    float mInHostScale{-1.0};
    float mOutHostScale{-1.0};

    std::string mNameSpace;
    const int mThreadNum = sizeof(unsigned long long) * 8 ;
public:
    UffUpSamplePluginV2(const nvinfer1::PluginFieldCollection &fc, float scale=2.0);
    UffUpSamplePluginV2(const void *data, size_t length);
    // IPluginV2
    const char* getPluginType () const override;
    const char *getPluginVersion () const override;
    int getNbOutputs () const override;
    nvinfer1::Dims getOutputDimensions (int index, const nvinfer1::Dims *inputs_dims, int number_input_dims) override;
    int initialize() override;
    void terminate () override;
    size_t getWorkspaceSize (int max_batch_size) const override;
    int enqueue (int batch_size, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;
    size_t getSerializationSize () const override;
    void serialize (void *buffer) const override;
    void destroy () override;
    void setPluginNamespace (const char *plugin_namespace) override;
    const char *getPluginNamespace () const override;

    // IPluginV2Ext
    nvinfer1::DataType getOutputDataType (int index, const nvinfer1::DataType *input_types, int num_inputs) const override;
    bool isOutputBroadcastAcrossBatch (int output_index, const bool *input_is_broadcasted, int num_inputs) const override;
    bool canBroadcastInputAcrossBatch (int input_idx) const override;
    IPluginV2Ext * clone () const override;

    // IPluginV2IOExt
    void configurePlugin (const nvinfer1::PluginTensorDesc *plugin_tensor_desc_input, int num_input, const nvinfer1::PluginTensorDesc *plugin_tensor_desc_output, int num_output) override;
    bool supportsFormatCombination (int pos, const nvinfer1::PluginTensorDesc *inOut, int num_inputs, int num_outputs) const override;

    // Extension
    template <typename Dtype>
    void forwardGpu(const Dtype* input,Dtype * outputint ,int N,int C,int H ,int W, cudaStream_t stream);
};


#endif //TENSORRT_RESIZENEARESTNEIGHBOR_H
