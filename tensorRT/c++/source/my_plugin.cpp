// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/2/11

#include "my_plugin.h"

MyPlugin::~MyPlugin() {
    for (auto& item : mPluginUpSample){
        item.reset();
    }
}


const char *MyPlugin::getPluginName() const {
    return "ResizeNearestNeighbor";
}

const char *MyPlugin::getPluginVersion() const {
    return "2";
}

const nvinfer1::PluginFieldCollection* MyPlugin::getFieldNames() {
    // TODO 这里应该是依据参数创建PluginField的
    return &mFieldCollection;
}

nvinfer1::IPluginV2 *MyPlugin::createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) {
    if(!strcmp(name, "ResizeNearestNeighbor")){
        printf("Unkown Plugin Name %s", name);
        return nullptr;
    }
    mPluginUpSample.emplace_back(std::unique_ptr<UffUpSamplePluginV2>(new UffUpSamplePluginV2(*fc)));
    return mPluginUpSample.back().get();

}

nvinfer1::IPluginV2 *MyPlugin::deserializePlugin(const char *name, const void *serial_data, size_t serial_length) {
    auto plugin = new UffUpSamplePluginV2(serial_data, serial_length);
    mPluginName = name;
    return plugin;
}

void MyPlugin::setPluginNamespace(const char *plugin_name_space) {
    mNamespace = plugin_name_space;
}

const char *MyPlugin::getPluginNamespace() const {
    return mNamespace.c_str();
}







