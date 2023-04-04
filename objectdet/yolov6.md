# YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications[meituan]

https://arxiv.org/abs/2209.02976

https://github.com/meituan/YOLOv6

https://tech.meituan.com/2022/06/23/yolov6-a-fast-and-accurate-target-detection-framework-is-opening-source.html


## 骨干网络设计

采用了repBlock和cspstackRep Block，
repBlock在训练中和推理中分别采用不同的架构，其中在训练阶段采用三分支的结构，分别为1\*1、3\*3、BN skip连接，在推理阶段则将这三种分支融合为同一个分支，而CSPstackrepBlock则在repBlock外侧添加了1\*1卷积的skip操作。

对比了Conv、Repconv和激活Silu、Relu、Lrelu的速度和推理时间对比，在具体使用过程中使用了RepConv+Relu以获得精度和速度均衡，使用Conv+silu以获得更高的模型精度。

## 检测neck设计


设计RepPAN进行更高效的计算，

## 检测头设计

将分类分支和回归分支进行解耦，
优化设计了更高效的解耦检测头，使用了一种Efficient decoupled head，将cls和reg分开进行回归。


## 数据处理(正难样本挖掘)

使用了灰度边框增加模型精度

对于标签分配策略：

### 1. 手工设计

通过IOU中心先验&size的match，借助一定的先验知识进行设计。

### 2. 自动设计

使用ATSS，设置一个自适应阈值完成正负样本匹配，缺点：当网络和数据集确定后，其对应的正负样本比例也确定了，不能随着训练而发生改变。
使用ATSS作为warmup策略。

### 3. 预测感知设计

使用SimOTA，为每个GT选取前dynamic_k个anchor作为正样本，cost=Lcls+labda*Lreg，缺点：不稳定、收敛慢。

### 4. 任务对齐学习

使用TAL的任务度量方式和损失函数引导网络聚焦与高质量样本。
最终使用了TAL的方式进行正负样本分配。


## 损失函数设计

分类损失：CE Loss、Focal loss、Quality Focal loss(QFL)、variFocal Loss(VFL)
VFL对正负样本非堆成加权改善类别不均衡问题，以及融合目标存在置信度和定位精度。

回归损失：IoU loss: GIoU、CIoU、DIoU、SIoU(缩小GT和预测框之间的差距)
probability loss：DFL、DFLv2：将连续的坐标回归问题转化为离散的分类问题。

目标损失：前景和背景的二分类学习，加速网络收敛；输出预测框中目标存在的置信度，减少低质量框的分数，多用于二阶检测模型。

DFL对精度有提升，但是对速度有影响。
引入目标损失会导致精度下降，可能原因是正负样本分配的warmup策略增加了学习难度增加。



## 工程实践
### 1. 使用FP32对int8的量化模型进行蒸馏
使用KDloss进行模型蒸馏，先使用教师网络的软标签进行蒸馏训练，然后使用硬标签对学生网络进行微调，具体策略是使用权重衰减的余弦损失函数。

### 2. 加入灰边

加入灰边可以增加模型对边缘目标的检测精度。




### 4. 模型量化部署

存在两个问题： 1. 使用PTQ会导致掉点厉害 2. 由于RepBlock而无法使用QAT的量化方式。

1. 使用RepOptBlock代替RepBlock，先使用gradient mask训练方式在模型反向传播过程中加入先验，替代结构化参数，第一步进行超参数搜索得到梯度掩码，第二步则在合并结构中加入掩码，然后再进行正常的模型训练。

2. PTQ-进行逐层的量化误差分析，分析每一层的敏感度，查找最敏感层，进行部分量化。

3. 基于channel-wise的模型自蒸馏，即对特征图的channel进行蒸馏。

使用trt进行实际的模型部署，使用“多实例”并发处理和DALI加速预处理。

