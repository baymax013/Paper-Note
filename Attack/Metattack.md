## 论文简述

论文题目：Adversarial Attacks on Graph Neural Networks via Meta Learning
论文地址：*https://arxiv.org/abs/1902.08412*
论文概况：经典的全局攻击Metattack

---

Metattack的核心，其实就是训练代理模型（经典2层的GCN）时的一个双层优化问题：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1679036751816-15379242-686f-4136-bbbe-fdaf241ec158.png#averageHue=%23f5f5f5&clientId=ue3b2ca3d-d0e4-4&from=paste&height=58&id=u53fc1984&originHeight=72&originWidth=678&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12948&status=done&style=none&taskId=u58229880-fa14-4947-a85a-a2ebde2ceb6&title=&width=542)
双层优化问题其实就是“先最优化θ参数，再最大化分类误差”。_L_train_ 就是有标签训练的loss，比如常见的交叉熵cross-entropy。而_L_atk_ 则代表最大化分类误差的损失（min和max，只差一个负号)。由于是半监督，Unlabel节点无法用交叉熵这种需要标签的loss，所以这里_L_atk_ 有两种取法：

   1. _L_atk_ = －_L_train_ 。**方法是最大化标签训练的损失**。理由是认为，如果一个模型训练误差就很高，那模型的泛化能力肯定也不好。注意，反过来这么说就不对，如果模型过拟合的话，训练误差很小并不能代表泛化能力好。
   2. _L_atk_ = －_L_self_ 。半监督的常用办法，用有标签的节点训练出一个代理模型，然后用代理模型去给无标签的节点生成伪标签，然后用伪标签去定交叉熵损失。

---

关于优化器，也就是如何处理双层优化问题是Metattack的重点。Metattack采用了**元学习中的元梯度**来解决这个双层优化问题。**核心思想就是将Graph当成一个可学习的超参数，计算G的梯度**。注意，文章是针对图结构进行扰动，所以可以**把G视为A**，X为常量。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1679037997352-4a4770e4-68a7-4f46-804b-07b031c5e23b.png#averageHue=%23f0f0f0&clientId=ue3b2ca3d-d0e4-4&from=paste&height=37&id=u180173e4&originHeight=46&originWidth=641&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=10878&status=done&style=none&taskId=u3539d0fc-a9a8-4857-b453-06a3992618d&title=&width=512.8)
从计算和内存的视角看，上述meta-gradient的计算是昂贵的。所以文章采用了尼科尔和舒尔曼于2018提出的**一种元梯度的启发式方法来近似**，得到最终的meta-gradient计算式子。注意，这里将上面提到的**两种**_**L_atk**_** 取法做了结合**。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1679041230046-72e157c8-3bd7-4f4d-b6c0-9692ed8cf927.png#averageHue=%23efefef&clientId=u6613d377-4871-4&from=paste&height=38&id=ub6b7ee3f&originHeight=48&originWidth=692&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12338&status=done&style=none&taskId=ua1ae67e3-47ac-4419-b902-c30bbbc3b14&title=&width=553.6)
基于此，可以通过meta-gradient对数据执行元更新_M_，来最小化_L_atk_，多轮迭代后得到最终的扰动图：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1679038181416-bd628ff5-b74b-4717-a41a-287be3579c2f.png#averageHue=%23f3f3f3&clientId=ue3b2ca3d-d0e4-4&from=paste&height=38&id=u7c55dbf5&originHeight=48&originWidth=240&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=3863&status=done&style=none&taskId=u2b30bae5-cc96-41c0-986b-e34fe19c8ec&title=&width=192)
其中，一种元更新_M_ 的实例化方法是具有一定步长β的梯度下降：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1679038375657-c1895a17-e445-4e77-9f76-ea3ccbb4931f.png#averageHue=%23e8e8e8&clientId=ue3b2ca3d-d0e4-4&from=paste&height=25&id=u64637ccd&originHeight=31&originWidth=364&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=7145&status=done&style=none&taskId=u0cacfeeb-4633-4769-a453-dd99a658d7e&title=&width=291.2)

---

总结一下Metattack的过程：输入一个原始Graph，经过“Δ Meta Updates”后，输出一个扰动Graph。

- 由于是半监督，代理模型选用2层的GCN来生成伪标签，得到_L_self。_
- 元更新M中计算meta-gradient部分采用元梯度的启发式方法近似，这个启发方法的计算式子中结合了两种_L_atk_ 取法。
- 最后，由于要维持扰动Graph的稀疏性和离散型，在元更新M中实际修改Edge的时候，涉及到一种贪婪算法，即定义一个score function来衡量对_L_atk_ 的影响，每次取最高扰动分数的action去执行。

