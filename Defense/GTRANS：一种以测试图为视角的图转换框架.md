## 论文简述

论文题目：Empowering Graph Representation Learning with Test-Time Graph Transformation
论文地址：*https://arxiv.org/abs/2210.03561*
论文概况：GTRANS框架，以图数据为中心视角，在测试阶段转换测试图。

------

# 1. 介绍

_As powerful tools for representation learning on graphs, graph neural networks(GNNs) have facilitated various applications from drug discovery to recommender systems. Nevertheless, the effectiveness of GNNs is immensely challenged by issues related to data quality, such as distribution shift, abnormal features and adversarial attacks. Recent efforts have been made on tackling these issues from a modeling perspective which requires additional cost of changing model architectures or re-training model parameters. In this work, we provide a data-centric view to tackle these issues and propose a graph transformation framework named GTRANS which adapts and refines graph data at test time to achieve better performance. We provide theoretical analysis on the design of the framework and discuss why adapting graph data works better than adapting the model. Extensive experiments have demonstrated the effectiveness of GTRANS on three distinct scenarios for eight benchmark datasets where suboptimal data is presented. Remarkably, GTRANS performs the best in most cases with improvements up to 2.8%, 8.2% and 3.8% over the best baselines on three experimental settings._
针对图上的攻击扰动，本文以**数据为中心视角**提出了一种防御策略，一个名为**GTRANS**的图转换框架。**这种策略试图修改模型在测试阶段输入的测试图，来提升模型最终的性能**，这可以视为一种图净化的方法。需要强调的是这种策略并不是改变模型的训练行为，而是修改**测试图**来纠正对抗模式，所以GTRANS具备较好的可移植性。下图大体上概括了这个策略的思想：原始图被添加扰动后显著降低了一系列GNN模型的准确率，而经过转换测试图后，各个GNN模型的准确率得以上升。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1673071822191-001dc347-e601-4521-a84b-b8e7cf807768.png#averageHue=%23f2f1f0&clientId=u30db8e3e-d650-4&from=paste&height=157&id=u2fafb5a3&originHeight=163&originWidth=699&originalType=binary&ratio=1&rotation=0&showTitle=false&size=49307&status=done&style=none&taskId=u3dbc845f-b903-4594-ad3a-ac8c0c53e71&title=&width=674.3636474609375)
_注：至于为什么要修改测试阶段的数据，可以参考这篇，通过辅助任务损失优化特征提取器的测试时间训练(TTT)：_[_https://arxiv.org/pdf/1909.13231v3.pdf_](https://arxiv.org/pdf/1909.13231v3.pdf)
# 2. 方法论
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1673073418944-61c7dd69-fdd2-4d6a-860d-8d39abdfff2b.png#averageHue=%23eeeeee&clientId=u7ed7817f-be05-4&from=paste&height=165&id=uaa5792c1&originHeight=221&originWidth=865&originalType=binary&ratio=1&rotation=0&showTitle=false&size=76070&status=done&style=none&taskId=u24d3f3e6-0597-44f6-89bc-4315af45d17&title=&width=644)
测试图转化的核心可以等价于上述的**TTGT优化问题，也就是如何得到“_g(·)_”**。这里主要面临着两个关键挑战：

1. 如何参数化和优化图变换函数 _g(·) _。
2. 由于测试阶段的标签YTe是未知的，如何制定一个代理loss来指导图变换时的学习过程。

因此，文章提出了GTRANS框架。
## 2.1 图变换函数 _g(·)_
### 2.1.1 Construction
一张图可以被视作_ G _= {**A**, **X**}，**A**∈{0, 1}_N✖N_，**X**∈R_N✖d_ 。所以测试图变换后，可以写为 _G'_  = _g_(**A**, **X**) = (**A'**, **X'**)。更具体的，把**A**和**X**的变换视作注入“扰动Δ”。这里将特性修改建模为一个加性扰动，即 **X' **= **X**+Δ**X**。将结构修改建模为一个异或扰动，即 **A' **= **A**⊕Δ**A**。值得注意，这里的“Δ**A**”也是一个由{0, 1}构成的_N✖N_的矩阵，所以经过异或操作 ΔAij = 1 就代表边翻转。最终图变换函数 _g(·) _就建模为 _g_(**A**, **X**) = (**A**⊕Δ**A**, **X**+Δ**X**)。
注意，为了保证测试图的修改不会严重破坏原图的结构，这里的结构修改需要满足 ||Δ**A**||2 ≤ _B_，也就是限制扰动的程度。
### 2.1.2 Optimization
形式上，_g(·) _的优化是寻求能使目标函数最小化的Δ**A**和Δ**X**：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1673077098609-63534073-0ab8-4719-a42a-9554b7db30e0.png#averageHue=%23efefef&clientId=uedbc75e5-653a-4&from=paste&height=36&id=ue6566c39&originHeight=46&originWidth=701&originalType=binary&ratio=1&rotation=0&showTitle=false&size=9531&status=done&style=none&taskId=ubf7472b4-43de-4bb6-bfaa-e5f28594201&title=&width=554)
共同优化Δ**A**和Δ**X**比较困难，因为它们相互依赖。实际操作中交替优化Δ**A**和Δ**X**。这里Δ**X**是连续的取值，优化迭代比较容易，但是Δ**A**是{0, 1}构成的离散的矩阵，所以存在两个问题：(1)Δ**A**取值是离散的，受二进制约束；(2)对于在大尺度的图上学习时，_N2_项的搜索空间太大。具体的处理方法如下：

   1. 针对第一个问题，首先将Δ**A**的二进制空间放宽到连续的[0, 1]_N✖N_。然后就可以使用投影梯度下降PGD来更新Δ**A**。PGD这个方法可以详细参考[_https://arxiv.org/pdf/1906.04214v3.pdf_](https://arxiv.org/pdf/1906.04214v3.pdf)和[_https://arxiv.org/pdf/2110.14038v3.pdf_](https://arxiv.org/pdf/2110.14038v3.pdf)。
   2. 针对第二个问题，需要减小邻接矩阵的搜索空间。文章中将搜索空间限制在图的现有边缘来解决，因为这通常是稀疏的。
## 2.2 无参数代理loss
GTRANS旨在通过学习来转换测试图，得以提升模型的防御性能。由于测试阶段没有样本的真实标签，就无法通过传统的最小化交叉熵来解决，**此时就需要一个可行的替代loss来指导图转化的过程**。在缺乏标记数据的情况下，文章的解决办法是基于最近的一些工作，**在图上应用自监督学习技术SSL**，从而为TTGT铺平道路([_https://arxiv.org/pdf/2102.10757v5.pdf_](https://arxiv.org/pdf/2102.10757v5.pdf) )。
关于自监督学习SSL：SSL属于无监督学习，它利用无标签数据本身，**构造一个辅助的任务**，针对这个辅助任务，可以从数据本身中得到标签。然后有监督的训练网络。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1673084639469-6ebd8b78-3b00-40b5-a37d-199862c29906.png#averageHue=%23ebebeb&clientId=u1a0fb335-6d77-4&from=paste&height=130&id=ud26c58ff&originHeight=150&originWidth=872&originalType=binary&ratio=1&rotation=0&showTitle=false&size=60098&status=done&style=none&taskId=u006f79a0-a05b-4b35-ba09-d16c0a49987&title=&width=754)
这个定理表明，在分类损失和代理损失的梯度具有正相关时，可以通过一个足够小的学习率执行梯度下降来更新测试图，从而减少测试样本上的分类损失。(5)式中左侧部分，就是通过代理loss的梯度乘上学习率去更新G，并且更新后测试图的分类损失，较于原测试图是减小的。因此，必须找到一个与分类任务共享相关信息的代理任务。
以往的研究表明，**图对比学习任务往往与下游任务高度相关**，这对TTGT是可取的。而对比学习的核心是对比方案，其中来自同一样本的两个增强视图之间的相似性最大化，而来自两个不同样本的视图之间的相似性最小化。这也是文章中代理loss函数的意图，但是略做了改变，由于现有的图对比学习方法需要一个参数化的投影层来将增强表示映射到另一个潜在空间，这不可避免地改变了模型架构。所以文章给出的代理loss是无参数化的，具体式子如下：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1673089962190-59ce5c20-103d-4e65-b653-1928f607130b.png#averageHue=%23f1f1f1&clientId=u1a0fb335-6d77-4&from=paste&height=61&id=ub724e4d8&originHeight=72&originWidth=454&originalType=binary&ratio=1&rotation=0&showTitle=false&size=9658&status=done&style=none&taskId=u03ec4d51-6730-455c-83a2-15357d47d6f&title=&width=383.18182373046875)
基于原始的测试图，文章使用DropEdge([_https://arxiv.org/pdf/1907.10903v4.pdf_](https://arxiv.org/pdf/1907.10903v4.pdf) )作为增广函数_A(·)_，得到增广图_A(G)_。式子中的_Z_、_Z __hat_分别为 _G _和 _A(G) _的节点表征。而_Z wavy_为对应节点的负样本，由节点特征变换生成([_https://arxiv.org/pdf/1809.10341v2.pdf_](https://arxiv.org/pdf/1809.10341v2.pdf) )。故代理loss函数中，第一项鼓励每个节点靠近，第二项推动每个节点远离对应的负样本。
