## 论文简述

论文题目：Robust Heterogeneous Graph Neural Networks against Adversarial Attacks（AAAI）
论文地址：*http://shichuan.org/doc/132.pdf*
论文概况：健壮的异构GNN框架 RoHe 以及一种 Attention Purifier

------

## HGNNs 的脆弱性

![image-20230403162808876](F:\Typora\images\image-20230403162808876.png)

1. **Perturbation enlargement effect**

   如上图（d）所示：传统的GNN，例如GCN，红色扰动点对于 Node P1 来说，只有1/3的扰动影响。
   而对于HGNNs的网络结构而言，通常会采用分层聚合，例如HAN。由于跳过了中间层（例如PAP中的Author层），红色扰动点的影响达到了63/66。

   故，HGNNs存在扰动放大现象，即HGNNs会放大对抗中枢的效应。

2. **Soft attention mechanism**

   如上图（b）所示：不同于同构GNN网络，HGNN会基于metapath，使得扰动边会让 Node P1 链接上许多扰动节点，若后续采用软attention机制，尽管单个扰动节点的attention值并不高，但扰动节点分配的attention**总和**会占不小的比重。基于这一事实，对明显不可靠的邻居分配零注意值的能力对hgnn来说是重要的。

------

## RoHe架构

![](F:\Typora\images\image-20230403165200292.png)

理解架构图的每一步流程：

1. 首先是统一的特征空间，由于图是异构的，即节点属于多个类别，所以不同类别的feature维数会不一致，在传入模型前需要映射到统一的向量空间中

   ![image-20230403183142114](F:\Typora\images\image-20230403183142114.png)

2. 软attention机制，按照传统的attention机制计算注意力值，同时通过点积计算相似度 **e_vu** 来衡量邻居节点的重要性（attention值的计算其实本质上也是点积相似度）

   ![image-20230403183805437](F:\Typora\images\image-20230403183805437.png)

3. 根据异构图给定的 **metapath Φ**，计算转移概率矩阵 **P_Φ**，矩阵中的每个element代表在关系R_i下，节点v到u的概率，并将这个element当作这条 **metapath Φ** 中目标节点v的邻居节点u的**先验置信度**。总体来说就是计算转移概率矩阵，矩阵中每个元素为对应节点邻居u的先验置信度。

   ![image-20230403184938173](F:\Typora\images\image-20230403184938173.png)

4. 根据点积相似度 **e_vu** 和 先验置信度 **P_Φ**，计算**置信分数向量 **confidence score（对应于目标节点的attention向量）。只对 top T 个元素赋予attention值，也就是mask掉低置信分数的节点，把这些节点当作扰动节点。结合 mask 向量和开始的软 attention 向量，得到 attention purifier后的净化attention值。

5. 由于是异构图，所以异构网络后面还有**语义级聚合 Semantic-level Aggregation**，即 hgnn 通常采用语义级attention来**计算每个元路径的重要性**。
