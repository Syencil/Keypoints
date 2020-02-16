// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/2/11

#ifndef TENSORRT_MY_PLUGIN_H
#define TENSORRT_MY_PLUGIN_H

#include "ResizeNearestNeighbor.h"
#include <cstring>
#include <bits/unique_ptr.h>

class MyPlugin : public nvinfer1::IPluginCreator {
private:
    std::string mNamespace;
    std::string mPluginName;
    nvinfer1::PluginFieldCollection mFieldCollection{0, nullptr};
    std::vector<std::unique_ptr<UffUpSamplePluginV2>> mPluginUpSample{};
public:
    const char* getPluginName() const override;
    const char* getPluginVersion() const override;
    const nvinfer1::PluginFieldCollection *getFieldNames() override;
    nvinfer1::IPluginV2* createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) override;
    nvinfer1::IPluginV2* deserializePlugin(const char *name, const void *serial_data, size_t serial_length) override;
    void setPluginNamespace (const char *plugin_name_space) override;
    const char* getPluginNamespace() const override;
    ~MyPlugin();
};


#endif //TENSORRT_MY_PLUGIN_H
