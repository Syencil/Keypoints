# Keypoint Detection In Tensorflow and TensorRT C++
## 1.Modified hourglass (Hourglass-104) and ResNet-101<br>

### Introduction
此项目为关键点检测训练以及推理加速代码。训练部分用python3 + tensorflow-1.14完成,推理部分用C++ + tensorRT-6完成。<br>
训练数据集主要为COCO，模型为Hourglass。

### Quick Start
* python3 train_hourglass_coco.py <br>
* python3 core/infer/freeze_graph.py -CUDA 0 -c checkpoints/coco/Hourglass_coco.ckpt -o Hourglass.pb <br>
* python3 demon.py<br>

### Checkpoints
https://drive.google.com/drive/folders/1pjOH1XUQOuMXlfGddQPvVEjXaXXPU7u1?usp=sharing <br>

### Data Format
如果需要使用自己的数据集进行训练，首先需要将数据转换成如下的格式 <br>
(filename1 bxmin,bymin,bxmax,bymax px,py px,py ...) <br>
If multi points have same label<br>
(filename1 bxmin,bymin,bxmax,bymax px,py|px,py px,py ...) <br>
(filename2 bxmin,bymin,bxmax,bymax px,py|px,py px,py|px,py ...) <br>
...<br>


### Inference
在core/infer/infer_utils.py中的一些api可以用来构建一个简单的inference模型。通过Flask包装一下就可以实现简单的线上推理了。操作示例在infer_hourglass.py中，其中bbx需要通过其他模型获取。<br>

### 注意事项
TensorRT部分已经转移到新的仓库下<br>
[https://github.com/Syencil/tensorRT](https://github.com/Syencil/tensorRT)

## 2.TensorRT
## 介绍
此处项目采用CUDA 10 + tensorRT-6完成推理阶段，可实现模型推理加速，支持FP32，FP16
### 开始使用
* 1.pb转uff
	* cd tensorRT/python
	* python3 pb2uff.py
* 2.编译C++文件
	* cd tensorRT/c++
	* cmake .
	* make


## 尚未完成的部分
* ~~1.数据增强 主要是图像旋转增强这一块有问题，会尽快将包括其他的增强方式加入项目~~
* ~~2.TensorRT C++中对upsample plugin的实现，框架现已搭好，会尽快更新~~
* ~~3.通过Hourglass-101构建今年大火的Anchor-free检测器CenterNet：Object as point~~
* ~~4.tensorRT C++数据预处理和python有点不同，并不影响太多，懒得改了。~~
* ~~5.Int 8量化矫正，有空再更新~~

