## 论文简述

论文题目：Detecting Topology Attacks against Graph Neural Networks

论文地址：*https://arxiv.org/abs/2204.10072*

论文概况：基于领域方差的受害节点检测方法

------

## 算法流程

![image-20230410110820696](F:\Typora\images\image-20230410110820696.png)

其中：γ是候选比率、T是偏随机游走方法的轮次、l是偏随机游走方法的步长，η是衰退率

步骤简述：

1. （可选）训练代理模型*G_p*，文章里用的是一个线性模型：

   <img src="F:\Typora\images\image-20230410111235271.png" alt="image-20230410111235271" style="zoom:80%;" />

2. （可选）计算*d(v)*，选取 top γ% 个nodes当作检测节点。分数*d(v)*表示节点v的预测标签概率最大变化，可以理解为***d(v)*越大，越容易是受害节点**，因为受邻居影响变化大。

   <img src="F:\Typora\images\image-20230410111652295.png" alt="image-20230410111652295" style="zoom:80%;" />

3. 非平滑处理。由于GNN的消息传递和聚合往往倾向于平滑的节点特征，故直接测量领域方差较难，所以计算领域方差分数前先进行“非平滑处理”。策略是一种偏随机游走的方法，**这一步的目的是更新feature**。

   ​	a. 计算e(u, w)，归一化后当作节点u至节点w的转移概率。这里c是预测标签，Z=XW，Z [u] [v] 表示 Z 的 u行v列。

   <img src="F:\Typora\images\image-20230410112250149.png" alt="image-20230410112250149" style="zoom:80%;" />

   ​	b. 并从节点 u 开始进行 T 轮长度为 l 的随机游走。对于每一次行走，feature更新为：

   <img src="F:\Typora\images\image-20230410112710008.png" alt="image-20230410112710008" style="zoom:80%;" />

4. 更新完特征后，采用两种方式计算领域方差。

   ​	a. **基于特征矩阵的方法**。PCA得到Rv的第一个主成分（方差最大的维度），再计算该成分的方差作为最终得分。此方差计算相当于计算Rv的协方差矩阵的最大特征值。cov表示协方差矩阵计算，λi(·)表示第i个最大特征值的计算：

   ![image-20230410115128379](F:\Typora\images\image-20230410115128379.png)

   ​	b. **基于相似矩阵的方法**。基于RBF核定义节点v的相似度矩阵W_sim，其中k是核参数。然后仍计算第一主成分的方差。

   <img src="F:\Typora\images\image-20230410115446150.png" alt="image-20230410115446150" style="zoom:80%;" />

   <img src="F:\Typora\images\image-20230410115504150.png" alt="image-20230410115504150" style="zoom:80%;" />