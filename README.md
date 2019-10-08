# CenterNet-tensorflow
1.Modified hourglass for key points landmark. Trained in Mpii.<br>
2.CenterNet (Object as point) trained in Pascal VOC and COCO.<br>
3.TensorRT <br>

# Attention
1.输出层之前不可以加BN，BN会强制将输出分布转换为0-1正态分布，破坏输出<br>
2.slim的BN和tf.layer的BN都需要在train的时候手动设置依赖项更新tf.GraphKeys.UPDATE_OPS。
否则BN的moving-mean和moving-var永不更新
3.slim的BN默认设置scale为false，即不使用gamma而tf.layer的默认设置为True。实际上如果BN之后是线性的例如relu则可以不需要scale，
但是如果不是的话则需要。开着比不开好，所以我设置的是True<br>
4.BN层需要warm-up。当moving-mean开始震荡而不怎么更新了即训练完成，这个过程可能比模型loss收敛的速度要慢。
因为BN在inference的时候是用训练集的mean以及var来代替inference中batch的值，
参数需要充分的学习到mean和var的值。这也是在训练时loss不变但val但acc仍然会不断增大但原因之一。
