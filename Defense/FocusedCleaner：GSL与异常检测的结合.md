## 论文简述

论文题目：FocusedCleaner: Sanitizing Poisoned Graphs for Robust GNN-based Node Classification

论文地址：*https://arxiv.org/abs/2210.13815*

论文概况：FocusedCleaner框架，GSL和异常检测算法的结合



## 主要思想

FocusedCleaner框架主要包含了两个部分，一个是双层优化的GSL，一个是异常节点检测器。两部分相互增强，检测模块可以监督GSL，为GSL提高焦点，其实就是让GSL在异常节点簇里**删除**边。

文章提出了两款异常节点检测：ClassDiv Based Anomaly Detector 和 LinkPred Based Anomaly Detector。前者使用了类概率对DGMM检测器训练，后者则是直接用GNN进行训练。无论如何，两个检测器最终的目的，也即输出，均是**异常节点簇**。所以可以把检测器当作一种聚类的算法。基于异常节点簇，可以得到对应的**正常节点簇**。后续对于GSL是基于meta-gradient去净化图，所以将正常节点簇的meta-gradient直接置0，就可以将删除边的操作“聚焦”在那些异常节点间的边上。

值得一提的是，文章中对“图净化”的操作只限制于**删除边**。图净化中可以选择翻转边或删除边，但基于对GASOLINE-DT（翻转）和GASOLINE-Delete（删除）的观察，发现**将净化操作限制于删除边更有效**。其实在图对抗学习中，攻击往往倾向于增添额外的边，防御更倾向于删除异常的边。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1683875769268-e4bc5915-300c-4ec1-b6b1-28407cba6457.png#averageHue=%23efedeb&clientId=u64e4f45b-21e7-4&from=paste&height=106&id=u9c1880ae&originHeight=132&originWidth=504&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=21855&status=done&style=none&taskId=u2ab20e29-7184-4ce6-b8cd-15268b19644&title=&width=403.2)

## 模型实现
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1683875936724-7fb0d839-6954-465b-9fd2-793df8d2ff9a.png#averageHue=%23fcfbfb&clientId=u64e4f45b-21e7-4&from=paste&height=266&id=u67a01e74&originHeight=333&originWidth=1085&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=67102&status=done&style=none&taskId=u19e2b0be-79a4-4b83-9d68-012bf1efe44&title=&width=868)

总体流程：给定Poisoned Graph，先经过检测器，得到victim node cluster及对应的normal node cluster，再基于meta-gradient去更新GSL。

这张总流程图里的第二部分是文章所示例的ClassDiv Based Anomaly Detector，但是这个检测器写的很复杂，一堆乱七八糟的公式，所以还是看LinkPred Based Anomaly Detector比较好。

### LinkPred Based Detector
实际上就是基于一个叫做LinkPred的链路预测模型，去检测潜在的异常边定位受害节点。直观理解就是这个LinkPred链路预测模型为每一条Link分配一个概率，概率较低的链路可视作对抗性噪声。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1683876736061-a01124be-f43a-4d9b-9594-6d4c59c470db.png#averageHue=%23f4f2f0&clientId=u64e4f45b-21e7-4&from=paste&height=86&id=u779109f6&originHeight=107&originWidth=543&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=19129&status=done&style=none&taskId=uf845d1b6-25b9-480f-8851-b5ea3e56960&title=&width=434.4)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1683876753368-24a706bd-f35a-4faa-91fd-74b22ddcdec7.png#averageHue=%23f9f8f7&clientId=u64e4f45b-21e7-4&from=paste&height=85&id=ubdc0ea65&originHeight=106&originWidth=763&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=17588&status=done&style=none&taskId=uc3de5e60-89d3-4ee9-822c-6aedcd8ce1b&title=&width=610.4)

这是LinkPred链路预测模型中的Hidden迭代式和loss函数。其中涉及的几个入参Z、X_PCA会在后续的伪代码中分析。这个检测器可以视为一个需要单独训练的模型，得到的输出是Link的概率，将概率低于一定阈值的链路视为对抗链路，将**异常节点定义为至少与一个对抗链路相连的节点**，除开异常节点簇外的节点均为正常节点。

### Graph Structure Learning
这里的双层结构学习GSL是一个双层优化问题，所谓双层优化就是套了两个argmax/argmin，metattack就是个双层优化的攻击。具体的双层优化问题定义如下，其中**A^R就是经GSL净化后的图**，也就是最终所得到输出。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1683877476503-2a4d0965-acbb-432c-b007-e0885ea5ce2c.png#averageHue=%23faf9f8&clientId=u64e4f45b-21e7-4&from=paste&height=90&id=ubd66cc6b&originHeight=113&originWidth=779&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20209&status=done&style=none&taskId=ube80f88e-601a-4046-b23f-b4f5516bccb&title=&width=623.2)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1683877550396-32ce4966-7c3a-44f0-bcaf-e99b290cb5a7.png#averageHue=%23fbfaf9&clientId=u64e4f45b-21e7-4&from=paste&height=184&id=u066b20fe&originHeight=230&originWidth=750&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=38384&status=done&style=none&taskId=u2e0a3994-abc6-4ef6-bf18-e2f14c15a9a&title=&width=600)

这里Eq(14)是文章中Eq(5a)的修改，其实就是在Eq(14)前加入了检测器。区别就在于式子第一项和第二项下面将V和N取交集，Eq(5b)就是没有加入检测器的，就没有这个N，N是正常节点簇。

这里A^R是外部的优化，也就是总的一个argmax。**后续的meta-gradient就是用这个外部优化问题求梯度得出的**。其中式子的第一项和第二项是内部优化，也就是Eq(5b)，对应文章用的一个线性的GNN，其实也可以换成经典的2-layers的GCN。这里的线性GNN其实也充当了一个代理模型，因为外部优化问题中需要test dataset的标签——Eq(14)第二项的y_hat，所以需要代理GNN生成这部分的伪标签。
最后一个是特征平滑器，这里直接引用的是Pro-GNN文章中提到得那部分。简单回顾一下，Pro-GNN中得GSL有两个优化方向，低秩和稀疏，然后也加上了这个特征平滑器。

## 算法伪代码
下面是整个双层结构学习GSL和检测器得总体伪代码，也就是FocusedCleaner框架的实现流程。理解这个伪代码基本就知道这篇论文的算法实现了。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1683878415861-45112b17-e7a4-4b99-9983-0b7461ff4f75.png#averageHue=%23f5f3f1&clientId=u64e4f45b-21e7-4&from=paste&height=794&id=ubb1e4a1b&originHeight=758&originWidth=635&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=191160&status=done&style=none&taskId=u7eb7a75e-131f-4e3b-a41f-564141dff01&title=&width=665)

```
输入：中毒图、中毒图对应的边集、净化程度（类似攻击的扰动程度）、
      检测器的超参{T, β, τ, η}、
      训练集、验证集、测试集、训练集标签、验证集标签
输出：净化图

Line 2：这里将特征X处理为X^PCA
        首先使用PCA将X的特征维数降为class数量，再结合超参T和softmax，得到最终的X^PCA
Line 3：循环条件 t ≤ B 其实就是限制净化程度，毕竟不能过多的修改图结构

Line 7：这里只看检测器LinkPred Based Anomaly Detector，另一个里面的公式太抽象了
Line 8：双层优化中的内部优化，将W、A^R、X送入GNN中，得到图节点表征Z
				并让这个GNN充当代理模型，得到测试集的伪标签
Line 9：将上面的A^R、Z、X^PCA送入到LinkPred模型中，得到异常节点簇y_ano

Line 11：基于异常节点簇y_ano，区分得到正常节点簇N
Line 12：设置初始值 λ_V = 1 - t / B，这个是文章上面提到的一种权重动态衰减方法。刚开始调高验证集权重，随着迭代轮次慢慢增加，不断的降低验证集权重并增大测试集权重
      	 依据外部优化式子Eq(14)计算梯度，求得meta-gradient
Line 13：将正常节点簇 N 中所链接的那些正常边的meta-gradient置为0
				 意思就是不考虑正常边，只“聚焦”于异常边
Line 14：选取一条链接进行删除
      	 这条链接，连接了至少一个具有最大元梯度的预测受害节点
```
