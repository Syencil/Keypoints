# Keypoint Detection In Tensorflow and TensorRT
1.Modified hourglass for key points landmark. Trained in Mpii and COCO.<br>
2.CenterNet (Object as point) trained in Pascal VOC and COCO.<br>
3.TensorRT <br>

# Quick Start
1.python3 train_hourglass_coco.py <br>
2.python3 core/infer/freeze_graph.py -CUDA 0 -c checkpoints/coco/Hourglass_coco.ckpt -o Hourglass.pb <br>
3.python3 demon.py

# Attention
1.输出层之前不可以加BN，BN会强制将输出分布转换为0-1正态分布，破坏输出<br>
2.slim的BN和tf.layer的BN都需要在train的时候手动设置依赖项更新tf.GraphKeys.UPDATE_OPS。
否则BN的moving-mean和moving-var永不更新
3.slim的BN默认设置scale为false，即不使用gamma而tf.layer的默认设置为True。实际上如果BN之后是线性的例如relu则可以不需要scale，
但是如果不是的话则需要。开着比不开好，所以我设置的是True<br>
4.BN层需要warm-up。当moving-mean开始震荡而不怎么更新了即训练完成，这个过程可能比模型loss收敛的速度要慢。
因为BN在inference的时候是用训练集的mean以及var来代替inference中batch的值，
参数需要充分的学习到mean和var的值。这也是在训练时loss不变但val但acc仍然会不断增大但原因之一。

# TensorRT
1.直接使用tensorRT-6进行pb2uff转换会提示缺少Merge，Switch，ResizeImage这三种节点的Warning，需要在启动Engine的时候手动实现这三个节点
并注册。<br>
2.如果使用tensorflow自带的trt进行转换，由于版本问题，tf114只能使用tensorrt4的库，他会将其他不支持的节点保留为原始节点，只转换支持的节点。