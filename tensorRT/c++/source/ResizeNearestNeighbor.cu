// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/2/12
#include <half.h>
#include <driver_types.h>
#include <cuda.h>
#include <cublas_v2.h>

#include "ResizeNearestNeighbor.h"

static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ int transform_idx(int idx, int C, int H, int W, float scale_factor){
    // 从后往前解idx
    // idx = n*C*H*W + c*H*W + h*W + w
    int w = idx % W;
    idx /= W;
    int h = idx % H;
    idx /= H;
    int c = idx % C;
    idx /= C;
    w /= scale_factor;
    h /= scale_factor;
    int hh = H / scale_factor;
    int ww = W / scale_factor;
    return idx * C * hh * ww + c * hh * ww + h * ww + w;
}


template<typename Dtype>
__global__ void UpSampleKernel(const Dtype *input, Dtype *output, int num_element, float scale_factor, int C, int H, int W){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_element){
        int idx = transform_idx(tid, C, H, W, scale_factor);
        output[tid]=input[idx];
    }
}

template<typename Dtype>
void UffUpSamplePluginV2::forwardGpu(const Dtype *input, Dtype *output, int N, int C, int H, int W, cudaStream_t stream) {
    int num_element = N * C * H * W;
    UpSampleKernel<<<(num_element-1)/mThreadNum+1, mThreadNum, 0, stream>>>(input, output, num_element, mScale, C, H, W);
}

size_t get_size(nvinfer1::DataType dataType){
    switch(dataType){
        case nvinfer1::DataType::kFLOAT :
            return sizeof(float);
        case nvinfer1::DataType::kHALF :
            return sizeof(__half);
        case nvinfer1::DataType::kINT8 :
            return sizeof(int8_t);
        default:
            throw "Unsupported Data Type";
    }
}

int UffUpSamplePluginV2::enqueue(int batch_size, const void *const *inputs, void **outputs, void *workspace,
                                 cudaStream_t stream) {
    const int channel = mCHW.d[0];
    const int input_h = mCHW.d[1];
    const int input_w = mCHW.d[2];
    const int output_h = mOutputHeight;
    const int output_w = mOutputWidth;
    int total_element = batch_size * channel * input_h * input_w;
    if (input_h == output_h && input_w == output_w){
        HANDLE_ERROR(cudaMemcpyAsync(outputs[0], inputs[0], get_size(mDataType) * total_element, cudaMemcpyDeviceToDevice, stream));
        HANDLE_ERROR(cudaStreamSynchronize(stream));
        return 0;
    }
    switch (mDataType){
        case nvinfer1::DataType::kFLOAT :
            forwardGpu<float>((const float *)inputs[0], (float *)outputs[0], batch_size, channel, output_h, output_w, stream);
            break;
        case nvinfer1::DataType::kHALF :
            forwardGpu<__half>((const __half *)inputs[0], (__half *)outputs[0], batch_size, channel, output_h, output_w, stream);
            break;
        case nvinfer1::DataType::kINT8 :
            forwardGpu<int8_t >((const int8_t *)inputs[0], (int8_t *)outputs[0], batch_size, channel, output_h, output_w, stream);
            break;
        default:
            throw "Unsupported Data Type";
    }
    return 0;
}