cmake_minimum_required(VERSION 3.5)
cuda_add_executable(keypoints Keypoints_main.cpp ../../../common_source/common_tensorrt/logger.cpp source/keypoints_tensorrt.cpp source/keypoints_tensorrt.h source/utils.h source/utils.cpp source/ResizeNearestNeighbor.cpp source/ResizeNearestNeighbor.h source/my_plugin.h source/my_plugin.cpp source/ResizeNearestNeighbor.cu)
set_property(TARGET keypoints PROPERTY FOLDER project/keypoints)
target_link_libraries(keypoints libnvinfer.so libnvparsers.so cudart.so  libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so)