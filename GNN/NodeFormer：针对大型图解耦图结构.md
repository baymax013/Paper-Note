## 论文总结：

论文题目：NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification

论文地址：[https://openreview.net/pdf?id=sMezXGG5So](https://openreview.net/pdf?id=sMezXGG5So)

论文概况：适配大型图，解耦图结构信息，将复杂度压在线性



## 主要思想：

​	NodeFormer可以看作是在图上的Transformer，也可以视作是在图上加入attention机制的优化版本，其核心思想是将**图结构解耦**。
文章的出发点在于传统的只基于邻居的消息传递，存在：

   - 对于图中“遥远”的节点会过度挤压（over squashing），在聚合过程中稀释掉这部分信息；
   - GNN有限的感受野使其难以捕捉长距离依赖（long-range dependence）；
   - GNN聚合邻居信息的设计不能很好兼容包含异配关系（heterophily）或连边残缺的图；
   - 以及在极端的没有输入图的情况下，GNN就无法正常工作了。

​	因此，NodeFormer直接舍弃了输入Graph的结构信息，或者可以理解为把输入Graph当作一个全连接图，Graph任意的两两节点都相互连接。这样子下，在消息传递中对于特定节点u，需要考虑图上除节点u之外的所有节点，复杂度为O(N2)。因此，文章最核心的贡献在于提出了一种算子来把特征聚合的复杂度降低到线性O(N)

## 算子实现
​	简单来说，NodeFormer是基于GAT的，因为这个算子的核心思想就是“用**XX近似**去修改attention计算式子”。NodeFormer主要提出了两个修改地方。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1685617795677-a9961696-d6f1-4530-afa1-193059f0b7e7.png#averageHue=%23ececec&clientId=u44784d99-a69c-4&from=paste&height=76&id=uf92b1c43&originHeight=95&originWidth=464&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=21118&status=done&style=none&taskId=ua4b8810f-296f-42db-b171-a5209e07fd6&title=&width=371.2)

### 核方法
​	用核方法对exponential-then-dot这一操作进行近似替换。这样就可以把attention中的**求和项解耦出来**，即下式右侧的结果所包含的两个求和项对所有节点是共享的（即不随节点u变化），因此只需要计算一次来使总复杂度保持在O(N)。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1685617861800-e3c5467e-b6c3-4ed7-9f0b-875c5522bd84.png#averageHue=%23e7e7e7&clientId=u44784d99-a69c-4&from=paste&height=33&id=u8cceb583&originHeight=41&originWidth=393&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=10724&status=done&style=none&taskId=u1c8899bc-76d2-4f68-bb09-2666d42608d&title=&width=314.4)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1685618161628-ea31d7bd-efa1-4fa2-9835-44f8f43382f1.png#averageHue=%23eeeeee&clientId=u44784d99-a69c-4&from=paste&height=78&id=u1ca9368c&originHeight=98&originWidth=775&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=31035&status=done&style=none&taskId=u51ef9692-d444-4232-9f15-7cb7d14647f&title=&width=620)

### Gumbel-Softmax
​	对于Gumbel-Softmax，一个核心思想是对于任意节点，我们其实需要的是找到在每一层中一个“最优”的邻居集合，进行消息传递。所以我们可以把个节点产生的注意力权重视为一个Categorical Distribution，然后从中采样得到邻居集合。但是采样过程不可求导，所以借助Gumbel-Softmax对其进行近似处理。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1685618320330-c15968e5-fcda-4ea0-a391-bc8e9cd59426.png#averageHue=%23f2f2f2&clientId=u44784d99-a69c-4&from=paste&height=83&id=u7cd7c915&originHeight=104&originWidth=784&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=27614&status=done&style=none&taskId=u6859a87a-bcfe-4ca8-a2a2-c2774e2897d&title=&width=627.2)

### Kernerlized Gumbel-Softmax Operator
将上面两部分结合起来，就可以抽象理解为下面加粗的公式。这也是NodeFormer模型实现线性复杂度的核心部件。

​						**Kernerlized Gumbel-Softmax Operator = GAT hidden + Kernerlized + Gumbel-Softmax**

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1685618414737-588d77b6-0f68-4910-b32c-347552b32567.png#averageHue=%23f4f0ee&clientId=u44784d99-a69c-4&from=paste&height=466&id=u7f5c4a90&originHeight=582&originWidth=1096&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=333581&status=done&style=none&taskId=u3b8461f6-6d74-40d5-82e0-3a59e0213e3&title=&width=876.8)

## 利用图结构信息
​	虽然NodeFormer的核心思想是将图结构信息解耦，但是通常来说图结构信息还是包含很多有效信息的。所以文章单独提出了两种简单有效的策略对输入图结构的信息加以利用。这里的“利用”可以理解为“解耦后，再额外补充进图结构包含的信息”
### **Relational Bias**
​	第一种策略是在每层信息传递时，**对观测连边的权重进行加强**，文章为每条边赋予一个共享的可学习的权重，称作relational bias，于是每层的更新公式更改为如下，其中b_(l)是第l层对应的可学习的权重参数，σ是一个可选的非线性映射。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1685618701498-30cbebc0-e3b6-4ed0-82fd-9556d01857d8.png#averageHue=%23f4f4f4&clientId=u44784d99-a69c-4&from=paste&height=63&id=u1acc6d60&originHeight=79&originWidth=417&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=10371&status=done&style=none&taskId=ua6cc6b9b-fc65-429a-ad69-90f28c94389&title=&width=333.6)

### **Edge Regularization Loss**
​	第二种策略是**把观测连边作为监督信号，加入到学习目标函数中**。具体的，我们把模型每层的注意力估计视为一个Categorical Distribution，而观测连边视为样本，于是采用极大似然估计定义一个连边的损失函数。其中d_u表示节点在输入图中的度（degree），Π_uv表示模型中间层的注意力估计。然后把loss_e乘以比例因此λ，加入到总Loss函数中。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1685618825650-1caf1360-23f4-4fa4-94de-3c5999b980aa.png#averageHue=%23f4f4f4&clientId=u44784d99-a69c-4&from=paste&height=83&id=ue91c5d08&originHeight=104&originWidth=523&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=15210&status=done&style=none&taskId=u649adf7c-d3b2-416a-8786-65123611ded&title=&width=418.4)
