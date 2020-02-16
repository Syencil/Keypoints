// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/2/11

#include "ResizeNearestNeighbor.h"

// <============ 构造函数 ===========>
UffUpSamplePluginV2::UffUpSamplePluginV2(const nvinfer1::PluginFieldCollection &fc, float scale): mScale(scale){

}

UffUpSamplePluginV2::UffUpSamplePluginV2(const void *data, size_t length){
    if (data== nullptr){
        printf("nullptr\n");
    }
    const char *d = static_cast<const char*>(data);
    const char* const start = d;
    mCHW = read<nvinfer1::Dims>(d);
    mDataType = read<nvinfer1::DataType >(d);
    mScale = read<float>(d);
    mOutputHeight = read<int>(d);
    mOutputWidth = read<int>(d);
    if (mDataType == nvinfer1::DataType::kINT8){
        mInHostScale = read<float>(d);
        mOutHostScale = read<float>(d);
    }
    assert(d = start + length);

}


// <============ IPluginV2 ===========>
const char *UffUpSamplePluginV2::getPluginType() const {
    // 保证和IPluginCreator::getPluginName()一致
    return "ResizeNearestNeighbor";
}

const char *UffUpSamplePluginV2::getPluginVersion() const {
    // 保证和IPluginCreator::getPluginVersion()一致
    return "2";
}

int UffUpSamplePluginV2::getNbOutputs() const {
    return 1;
}

nvinfer1::Dims
UffUpSamplePluginV2::getOutputDimensions(int index, const nvinfer1::Dims *inputs_dims, int number_input_dims) {
    assert(number_input_dims==1);
    assert(index == 0);
    assert(inputs_dims[0].nbDims==3);
    mCHW = inputs_dims[0];
    mOutputHeight = inputs_dims[0].d[1] * mScale;
    mOutputWidth = inputs_dims[0].d[2] * mScale;
    return nvinfer1::Dims3(mCHW.d[0], mOutputHeight, mOutputWidth);
}

int UffUpSamplePluginV2::initialize() {
    // 可以用来分配内存
    int input_height = mCHW.d[1];
    int input_widht = mCHW.d[2];
    if (mOutputHeight == int(input_height * mScale) && mOutputWidth == int(input_widht * mScale)){
        return 0;
    } else{
        return 1;
    }
}

void UffUpSamplePluginV2::terminate() {
    // 可以用来释放内存
}

size_t UffUpSamplePluginV2::getWorkspaceSize(int max_batch_size) const {
    // 根据maxBatchSize确定该层所需要的最大内存空间
    return 0;
}

size_t UffUpSamplePluginV2::getSerializationSize() const {
    size_t serialization_size = 0;
    serialization_size += sizeof(nvinfer1::Dims);
    serialization_size += sizeof(nvinfer1::DataType);
    serialization_size += sizeof(float);
    serialization_size += sizeof(int) * 2;
    if (mDataType == nvinfer1::DataType::kINT8){
        serialization_size += sizeof(float) * 2;
    }
    return serialization_size;
}

void UffUpSamplePluginV2::serialize(void *buffer) const {
    char *d = static_cast<char*>(buffer);
    const char* const start = d;
    printf("serialize mScale %f\n", mScale);
    write(d, mCHW);
    write(d, mDataType);
    write(d, mScale);
    write(d, mOutputHeight);
    write(d, mOutputWidth);
    if (mDataType == nvinfer1::DataType::kINT8){
        write(d, mInHostScale);
        write(d, mOutHostScale);
    }
    assert(d == start + getSerializationSize());
}

void UffUpSamplePluginV2::destroy() {
    delete this;
}

void UffUpSamplePluginV2::setPluginNamespace(const char *plugin_namespace) {
    mNameSpace = plugin_namespace;
}

const char *UffUpSamplePluginV2::getPluginNamespace() const {
    return mNameSpace.data();
}


// <============ IPluginV2Ext ===========>
nvinfer1::DataType
UffUpSamplePluginV2::getOutputDataType(int index, const nvinfer1::DataType *input_types, int num_inputs) const {
    assert(index==0);
    assert(input_types!= nullptr);
    assert(num_inputs==1);
    return input_types[index];
}

bool UffUpSamplePluginV2::isOutputBroadcastAcrossBatch(int output_index, const bool *input_is_broadcasted,
                                                       int num_inputs) const {
    return false;
}

bool UffUpSamplePluginV2::canBroadcastInputAcrossBatch(int input_idx) const {
    return false;
}

nvinfer1::IPluginV2Ext *UffUpSamplePluginV2::clone() const {
    auto *plugin = new UffUpSamplePluginV2(*this);
    return plugin;
}


// <============ IPluginV2IOExt ===========>
void UffUpSamplePluginV2::configurePlugin(const nvinfer1::PluginTensorDesc *plugin_tensor_desc_input, int num_input,
                                          const nvinfer1::PluginTensorDesc *plugin_tensor_desc_output, int num_output) {
    assert(num_input==1 && plugin_tensor_desc_input!= nullptr);
    assert(num_output==1 && plugin_tensor_desc_output != nullptr);
    assert(plugin_tensor_desc_input[0].type == plugin_tensor_desc_output[0].type);
    assert(plugin_tensor_desc_input[0].format == nvinfer1::TensorFormat::kLINEAR);
    assert(plugin_tensor_desc_output[0].format == nvinfer1::TensorFormat::kLINEAR);

    mInHostScale = plugin_tensor_desc_input->scale;
    mOutHostScale = plugin_tensor_desc_output->scale;

    mDataType = plugin_tensor_desc_input[0].type;
}

bool UffUpSamplePluginV2::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *plugin_tensor_desc_in_out, int num_inputs,
                                                    int num_outputs) const {
   assert(plugin_tensor_desc_in_out != nullptr);
   assert(num_inputs == num_outputs == 1);
   assert(pos < num_inputs + num_outputs);
   bool condition = true;
   condition &= plugin_tensor_desc_in_out[pos].format == nvinfer1::TensorFormat::kLINEAR;
   condition &= plugin_tensor_desc_in_out[pos].type != nvinfer1::DataType::kINT32;
   condition &= plugin_tensor_desc_in_out[pos].type == plugin_tensor_desc_in_out[0].type;
   return condition;
}





















