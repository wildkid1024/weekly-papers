# 每周论文合集



## 信息检索、搜索

### Möbius Transformation for Fast Inner Product Search on Graph[NIPS'19][Baidu]
- 现有的图上的ANN检索算法依赖于在metric measure(L2) 构建出的类德劳内图，但在no-metric measure(IP) MIPS 上无此性质，效果不佳（// TODO 效果不佳如何证明）
- 提出了 mobius 变换： ，将数据集从 IP space 映射到 L2 sapce，在此基础上建图
- 多个公开数据集上，同等算力下的Recall 精度有明显提升（20% ～ 30%）

### Embedding-based Retrieval in Facebook Search[KDD'20][Facebook]


### Transformer Memory as a Differentiable Search Index[NIPS'22][Google]

### A Neural Corpus Indexer for Document Retrieval[NIPS'22][MSRA]

- 提出了一种端到端的基于Transformer的信息检索架构。
- 传统的信息检索分为两类，一类是基于term的，建立倒排表，根据倒排表进行索引，缺点：无法召回语义相关的文档；另外一类是基于语义的，建立双塔模型，使用ANN进行检索，缺点：在精确匹配上表现较差，二是ANN有欧拉空间的强假设，模型无法合并文档交互？
- 在预处理部分加入了kmens,使文档ID具有分类层次化的特点，然后借助DocT5Query和Document As Query对文档生成query，将生成或真实的<query, docid>对送入到encoder中，使用了前缀可感知的共享权重decode进行解码，最后使用同一query间的输入尽可能接近，不同query间尽可能大进行对比学习，使得算法更加稳定。
- 缺点：在v100机器上，最大的吞吐只有50左右，在实际场景中是不够的。当出现新的文档的时候，docid会发生变化，模型要进行重训练。


## 对抗攻防
### Segment and Complete: Defending Object Detectors against Adversarial Patch Attacks with Robust Patch Detection[CVPR'22][JHU]
- 提出了SAC检测器用于防御目标检测的贴纸攻击，SAC检测器分为两部分，分割部分和补全部分。
- 分割部分使用了将贴纸检测视为图像分割任务，使用UNet进行自对抗训练，找到贴纸大致位置。
- 补全部分使用汉明距离作为预测和GT的度量，补全的是所有在gama重叠范围内的交集。
- 改进了APRICOT数据集，增加了Mask标注。

### Defending Physical Adversarial Attack on Object Detection via Adversarial Patch-Feature Energy[MM'22][South Korea]
- 提出了APE(Adversarial Patch-Feature Energy Masking)模块防御类别为人的目标检测的攻击，APE模块可分为两部分，APE-masking 和 APE-refinement。
- APE-masking部分负责解析出攻击对应的mask，具体做法是通过objectness loss反向传播的L1范式的平方获得FE ，根据干净样本和对抗样本的分布差异确定阈值，大于阈值会被认为是proposal patch，然后通过上采样累加形成最终mask。
- APE-refinement部分根据APE-masking解析出的mask进行加固，具体是根据干净样本的分布均值的比例进行clip（这里似乎假设对抗样本的分布均值比干净的大？），对应位置上的adv patch即为clip平滑后的值。
- 属于图像信号处理的一类工作，有点意思，但套路有点老。

### Defending Person Detection Against Adversarial Patch Attack by using Universal Defensive Frame[TIP'22][KAIST]

### Role of Spatial Context in Adversarial Robustness for Object Detection[CVPR'20][UMBC]


### Physically Adversarial Attacks and Defenses in Computer Vision: A Survey[arXiv'22][Beihang]
- 额外文档，见计算机视觉领域的物理对抗攻防综述


## 系统、编译器设计、优化

### TVM: an automated end-to-end optimizing compiler for deep learning[OSDI'18][UW]
- TVM是一个能够将上层计算图表示编译转化为后端IR的工具。首先是高层次计算优化，将计算图分解为Tensor和计算，然后使用自动优化对低级的Tensor和计算针对特定的硬件进行优化以达到最佳性能。
- 高层次计算图级的优化：算子融合(融合相邻的算子)、常数折叠(提前计算静态值)、静态内存预分配(预分配内存)、数据存储转换(改变数据分布以利用缓存和SIMD特性)。
- 低层次Tensor级优化：利用Halide原则将规划和计算分开，首先引入了领域专用语言Tensor expression表示计算，然后写Schedule进行优化，转化为TVM IR对应着特定硬件表示。具体的Schedule为：带共享内存的循环并行、向量化以利用硬件的SIMD和向量运算、访问执行分离隐藏延迟。
- 自动化优化：使用了Xgboost根据配置进行性能预测，使用真实的测试数据作为训练数据，使用模拟退火的方法进行配置更新；并提供了一个可以交叉编译的分布式远程调用。
- 开创性的工作，不过TVM现在还在开发当中，有些组件还不太稳定，另外还不够用户友好。

### LightSeq: A High Performance Inference Library for Transformers[NAACL'21][ByteDance]
- 主要针对transformer的优化，有3点贡献
- 1. 将粗粒度的节点融合转化为细粒度的节点融合，以避免频繁的kernel启动，例如手写layer norm kernel可以节省内存启动和保存中间结果。
- 2. 层次的自回归搜索，采用了检索和重排的思想。
- 3. 动态的GPU内存重用方案，将前后依赖的结果存在相同的内存。

### LightSeq2: Accelerated Training for Transformer-based Models on GPUs[SC'22][ByteDance]
- 主要针对transformer的优化，有4点贡献
- 将粗粒度节点转化为手写的细粒度节点并进行融合，具体来讲就是将GEMM部分使用cuBLAS来实现，其他的元素级操作(Dropout, ReLU, Reshape)和约减操作(LayerNorm and Softmax)用手写的kernel来代替，主要对Transformer、Embedding、Criterion、Layer-batched cross Attention层进行了分析。
- 有依赖的约减操作的拆分，对LayerNorm层的梯度计算进行拆分，分别并行计算两部分乘法运算以进行加速。对Softmax层实现了不同的模板，并进行参数调节以适应不同大小和不同形状的softmax计算。
- 加速混合精度梯度更新计算。把所有要更新的参数放到一整片内存区，以避免每次更新的时候都要启动kernel去加载和卸载内存，同时可以节省一些内存。
- 悬空张量的内存管理。具体来讲就是将内存分为永久内存和暂时内存，并将训练参数和更新参数要用的最大内存提前分配好，并进行内存复用，可以节省一部分频繁加载卸载的消耗。
- 整体还是偏工程的工作，作为学术的novelty并不那么fancy，不过对于实现还是有些启发的。

## 模型优化
### FastFormers: Highly Efficient Transformer Models for Natural Language Understanding[arxiv'20][MSRA]
- msra文章，但是只是单纯做了模型裁剪、蒸馏和量化，是一篇纯实验结果堆的文章 
- https://github.com/microsoft/fastformers
 
## 行为检测
### MS-TCT: Multi-Scale Temporal ConvTransformer for Action Detection[CVPR'22][Inria]
- 额外文档，见MS-TCT读书笔记(1)

### Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset[CVPR'17][DeepMind]

### An Efficient Spatio-Temporal Pyramid Transformer for Action Detection [ECCV'22]

## 目标检测
### YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors[arxiv'22][IIS]
- 见额外文档，yolov7总结
