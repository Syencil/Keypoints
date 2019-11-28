# Keypoint Detection In Tensorflow and TensorRT
## 1.Modified hourglass (Hourglass-71) <br>
### Quick Start
* python3 train_hourglass_coco.py <br>
* python3 core/infer/freeze_graph.py -CUDA 0 -c checkpoints/coco/Hourglass_coco.ckpt -o Hourglass.pb <br>
* python3 demon.py<br>

### Checkpoints
https://drive.google.com/file/d/1qWAqCX9Ql0CDP--ZEshmNb5KSwkkmUhS/view?usp=sharing <br>
However, each ckpt is not fully trained due to limited resources. 
But it's enough as a pre-training model to train your own data.<br>

### Data Format
You should transform your own data format into <br>
(filename1 bxmin,bymin,bxmax,bymax px,py px,py ...) <br>
(filename1 bxmin,bymin,bxmax,bymax px,py px,py ...) <br>
(filename2 bxmin,bymin,bxmax,bymax px,py px,py ...) <br>
...<br>


### Inference
在core/infer/infer_utils.py中的一些api可以用来构建一个简单的inference模型。通过Flask包装一下就可以实现简单的线上推理了。操作示例在infer_hourglass.py中，其中bbx需要通过其他模型获取。


### 注意事项
*此处为记录一下项目中可能遇到的或者遇到过的坑*<br>
* 1.项目中采用ResidualV2的Pre-act结构。输出层不可以加BN，BN会强制将输出分布转换为0-1正态分布，破坏输出<br>
* 2.slim的BN和tf.layer的BN都需要在train的时候手动设置依赖项更新tf.GraphKeys.UPDATE_OPS。
否则BN的moving-mean和moving-var永不更新<br>
* 3.slim的BN默认设置scale为false，即不使用gamma而tf.layer的默认设置为True。我设置的是True<br>
* 4.BN层需要warm-up。当moving-mean开始震荡而不怎么更新了即训练完成，这个过程可能比模型loss收敛的速度要慢。因为BN在inference的时候是用训练集的mean以及var来代替inference中batch的值，
参数需要充分的学习到mean和var的值。这也是在训练时loss不变但val但acc仍然会不断增大但原因之一。<br>
* 5.如果使用placeholder为BN层的is_training参数(不是trainable),BN层中会处于一种使用tf.cond,tf.switch流控制节点(此处可以在tensorRT以及模型图中得到验证)的状态。这样的话每一个BN层都会有两条路径出来，训练太占显存，infer部署的时候还要单独进行剪枝。
有两种解决方案：
	* (1)此处直接设置为True的，训练是没问题的。做val的时候，不调用train_op那么BN的gamma和beta不会更新。并且由于mean和var设置为依赖于train_op更新，所以BN在val时所有参数都没有更新，相当于trainable=False。然而在tf1.x版本中，trainable=False是让BN处于freeze状态，用is_training控制是否infer。和infer不同的时，freeze仍然是使用当前batch的mean和var进行处理。在tf2.x版本中，bn已经改成了当trainable为False的时候是infer状态。所以此版本的Val仍然不是很准确，所以我将bn层的var和mean打印出来，最后以小学习率甚至0学习率进行更新var和mean依次弥补造成的问题。因为当这个"伪val"的loss和acc达到预期之后我们唯一需要等待的就是让模型中的BN层的var和mean震荡起来，warmup<br>
	* (2)使用reuse参数构建两个模型，一个is_training=True，一个is_training=False。训练时调用前者，测试时调用后者。
<br>

## 2.TensorRT
### 开始使用
* 1.pb转uff
	cd tensorRT
	python3 pb2uff.py
* 2.编译C++文件
	框架已构建完毕，如果不使用deconv做上采样而使用tf.image.resize_nearest_neighbor则需要手动实现Plugin，这一部分正在实现中。

## 尚未完成的部分
* 1.数据增强 主要是图像旋转增强这一块有问题，会尽快将包括其他的增强方式加入项目
* 2.TensorRT C++中对upsample plugin的实现，框架现已搭好，会尽快更新
* 3.Evaluation 目前全靠loss和肉眼观察输出结果，只能定性分析。定量分析的话可以通过github上其他项目实现，会尽快将其融入该项目
* 4.通过Hourglass-101构建今年大火的Anchor-free检测器CenterNet：Object as point

