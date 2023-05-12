## 论文简述

论文题目：Adversarial Examples on Graph Data: Deep Insights into Attack and Defense
论文地址：*https://arxiv.org/abs/1903.01610*
论文概况：经典的GCN-Jaccard 和 IGAttack攻击方法

------

# 1. 引言

​	图被广泛用用于许多现实关系任务中，诸如社交网络，引文网络，交易网络以及控制流任务。在图数据中最广泛的一个任务是结点分类任务：对于一个图，已知其中一部分结点的标签，目的是对于其他未标注结点做预测。该任务可以应用在许多任务中，例如对于引文网络中论文的主题做预测，对推荐系统中的消费者的类型做预测。深度学习方法因为缺乏鲁棒性而一直被大家所讨论。这些图神经网络模型在被实施了对抗攻击的时候表现出来的脆弱性使得它很难被应用在一些safety-critical场景中。在图神经网络中，结点可以是社交网络中的一个用户，也可以是一个商业网站。一个恶意用户可以修改自己的profile或者是跟目标用户之间建立联系从而误导系统的分类。同样的，通过对特定的商品添加虚假评论也可以欺骗到推荐系统。**不能将已有的非图数据中的对抗攻击方法应用在图神经网络中，原因在于图数据的输入是离散的。**具体而言，图结点的特征通常都是离散的；图数据中的边，尤其是非权重图中的边通常也都是离散的。为了解决这个问题，许多人提出了贪婪算法来攻击图深度学习系统（Nettack）。贪婪算法是不断的迭代，来扰动特征或者边。在这篇文章中，作者指出来，就算是离散数据，也可以通过引入**积分梯度**来计算它的梯度。积分梯度（Integrated gradient）第一次被提出来是用于解决Shapely value，将部分梯度（partial gradient）和输入特征联系到一起。相比较贪婪选择算法，积分梯度可以大大提高结点选择和边选择的效率。跟对抗攻击相比，在图模型上对于对抗攻击的防御还没有怎么被研究。在这篇文章中，作者指出来，**图神经网络模型之所以有脆弱性，是因为这些模型是基于图结构，来不断的聚合结点周围的其他节点的特征。**也就是说，在对某个结点做预测的时候，模型会依赖于周围最邻近结点的特征。作者在对现有的攻击产生的扰动做分析的时候，发现，将一个结点周围的特征不同的其他结点与当前结点建立连边会带来很大的影响。在这篇文章中，作者发现，只要对一个图的邻接矩阵做预处理就可以确定这类边。对于拥有词袋特征（BOW，bag-of-words）的结点而言，Jaccard系数可以有效的衡量结点之间的相似性。通过把不相似的两个结点之间的连边去除，可以有效的防御住有目标的对抗攻击，并且可以保证GCN模型上的准确率不会下降。作者通过使用许多真实世界的数据集，发现提出的攻击和防御是十分有效的。

# 2. 基于积分梯度的攻击：IGAttack
## 2.1 图卷积网络GCN
​	GCN是在半监督结点分类领域被广泛使用的方法。GCN中通过如下公式进行邻居结点的特征聚合，也就是经典的GCN隐层式子：

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675492076603-f4c52e15-1a23-4e28-bd8c-55bcb9cb563d.png#averageHue=%23f2f2f2&clientId=u276f998b-39b4-4&from=paste&height=40&id=uc0a77a46&originHeight=44&originWidth=323&originalType=binary&ratio=1&rotation=0&showTitle=false&size=5731&status=done&style=none&taskId=u48f52150-1f8d-4168-9a62-d9a4db966a2&title=&width=292.3999938964844)

​	整个网络只包含2层，就可以很好的解决半监督的结点分类任务了。模型的总表示如下：

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675491950146-eb46cb85-6a91-4627-9e0a-ab1104c765ee.png#averageHue=%23f2f2f2&clientId=u276f998b-39b4-4&from=paste&height=44&id=ua496b77e&originHeight=49&originWidth=421&originalType=binary&ratio=1&rotation=0&showTitle=false&size=7036&status=done&style=none&taskId=ud48bd409-5170-4502-a51b-3830b9657ff&title=&width=376.8000183105469)

​	其中A是一个预处理过的邻接矩阵（对称矩阵），D是度矩阵（对角阵）

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675491964186-abccc24b-b5bd-4896-b348-de6fd1d0c414.png#averageHue=%23f0f0f0&clientId=u276f998b-39b4-4&from=paste&height=34&id=ucf63381d&originHeight=34&originWidth=164&originalType=binary&ratio=1&rotation=0&showTitle=false&size=2508&status=done&style=none&taskId=udc29b34a-1c72-4ae6-a5c5-dcf9336a75b&title=&width=162.1999969482422)

## 2.2 基于梯度的对抗攻击
​	**对抗学习中，经常基于梯度对深度学习模型进行攻击，攻击者可以使用损失函数的梯度或者是模型的梯度来实施攻击。**两个典型的例子就是FGSM和JSMA攻击。

FGSM攻击是朝着损失函数的梯度上升方向去注入扰动。

JSMA攻击是利用DNN模型的前向偏差，通过了Jacobian矩阵去求得向前传播的偏导数，从而获取哪些像素位置对误分类神经网络有最大的贡献，将这些像素点当作扰动对象。

## 2.3 基于积分梯度的对抗攻击
​	尽管FGSM和JSMA并不是最复杂的攻击方法，它们也没有被很好的应用在图问题中。对于图像数据而言，FGSM和JSMA的成功源于特征在像素空间上的连续性。但这些方法直接应用在图领域不能带来很好的攻击效果。就有一些研究者使用贪婪算法或者是强化学习方法来解决图数据这类离散数据，但是贪婪算法和强化学习通常都是很昂贵并且费时的。

而这篇文章提出的IGAttack，就是将2.2中的“Vanilla Gradients”替换为Google于2017年提出的“Integrated gradient”这一概念。**积分梯度本质上是一种归因分析(Attribution Analysis) 方法**，可以获得特征属性的偏差，换句话说IG衡量了模型对输入的第 i 个分量的敏感程度，而IGAttack就是以此作为摄动优先程度。

> 所谓可视化，简单来说就是对于给定的输入 x 以及模型 F(x)，我们想办法指出 x 的哪些分量对模型的决策有重要影响，或者说对 x 各个分量的重要性做个排序，用专业的话术来说那就是“归因”。

### 2.3.1 积分梯度的定义
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675494429146-4c98cbb1-c0bc-47ae-865b-cc5cc19618a5.png#averageHue=%23f5f5f5&clientId=u719a6a43-2011-4&from=paste&height=61&id=uc9a253f7&originHeight=76&originWidth=575&originalType=binary&ratio=1&rotation=0&showTitle=false&size=10390&status=done&style=none&taskId=u0e7b3e1c-3f1f-4d99-9352-58008e924bd&title=&width=460)

​	给定一个模型F：Rn -> [0, 1]，x’是一个baseline input（基准输入），x是真实的输入。考虑从x’到输入x之间的一条直线路径，积分梯度就是对这条路径上的所有梯度进行累加。例如，对于输入x的第i个特征而言，积分梯度IG可以定义如上。

​	如果是有目标攻击，就是应该要最大化F值，所以当图中的特征或者边是1时，可以选择有最低的负值IG分数的特征或者边，将其修改成0。如果是无目标攻击，就是应该要最小化真实标签的预测分数，所以需要选择有最高IG分数的维度，将其修改成0。

### 2.3.2 基线设置和 Edge Attack
​	但是不像图像一样，我们可以将一个全黑的图像（black image）设置成baseline input。作者使用了一个全0或者是全1的特征/邻接矩阵来表示 1->0 的扰动或者是 0->1 的扰动。当去除一条特定边或者是将一个特定的特征从1改成0后，作者是将邻接矩阵A和特征矩阵X设置成全0，然后逐渐的在当前全0的矩阵上添加边或者特征，然后观察F的整体变化。相反的，当添加一条特定边或者将一个特定的特征从0改成1后，首先将A和X都设置成全1，然后逐渐的移除某条边或者某个特征，然后观察F的整体变化。对于边攻击（edge attack）的IG分数计算如下所示：

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675495706449-27e00952-3fee-4810-8775-71111cbfe028.png#averageHue=%23f6f6f6&clientId=u719a6a43-2011-4&from=paste&height=151&id=ua12096a6&originHeight=189&originWidth=770&originalType=binary&ratio=1&rotation=0&showTitle=false&size=33414&status=done&style=none&taskId=ua2730e7d-4925-45d8-a74d-c60a379c747&title=&width=616)

### 2.3.3 算法伪代码(无目标的IG-JSMA)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675496267903-1fcc3f5a-c3a0-4772-bad2-91720f0986fa.png#averageHue=%23f0f0f0&clientId=u719a6a43-2011-4&from=paste&height=594&id=u0b83b565&originHeight=742&originWidth=675&originalType=binary&ratio=1&rotation=0&showTitle=false&size=134128&status=done&style=none&taskId=u6fb0f8b0-e57a-4693-b015-a6b960a07ce&title=&width=540)

## 2.4 实验结果
### 2.4.1 IGAttack的攻击性
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675500729765-e0de2a84-23c3-4e95-9769-50c3bcf196ea.png#averageHue=%23f7f5f4&clientId=u91ba6999-6ab3-4&from=paste&height=321&id=u19f61525&originHeight=412&originWidth=1110&originalType=binary&ratio=1&rotation=0&showTitle=false&size=48311&status=done&style=none&taskId=u81b45ccd-eedf-4c02-95e4-de9d85b867f&title=&width=864)

​	这里引用了“classification margin”作为衡量攻击性的指标。**classification margin的值越小**，说明真实标签c对应的置信度分数比较低，其他类别的置信度分数比较高，那么就说明**攻击的效果越好**。定义式如下：

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675500614774-b8e3c01b-8339-492a-965a-9b16818d2dde.png#averageHue=%23efefef&clientId=u91ba6999-6ab3-4&from=paste&height=39&id=ubab450cc&originHeight=68&originWidth=459&originalType=binary&ratio=1&rotation=0&showTitle=false&size=9184&status=done&style=none&taskId=ue4393a93-fdba-4816-8aca-7db6d45bcf2&title=&width=261)

### 2.4.2 Node/Edge 的重要性近似

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675500878319-6ba86745-3dc3-42c8-a57b-1456776959e9.png#averageHue=%23fdfbfa&clientId=u91ba6999-6ab3-4&from=paste&height=341&id=ufefbbe9d&originHeight=426&originWidth=1114&originalType=binary&ratio=1&rotation=0&showTitle=false&size=115377&status=done&style=none&taskId=u2e39f7dd-c5ea-4727-8301-55f0bf55c42&title=&width=891.2)

# 3. 防御模型：GCN-Jaccard
## 3.1 Jaccard相似度
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675499878531-0b921d4c-ed43-457f-8ef9-84ad1bf8c96f.png#averageHue=%23f7f7f7&clientId=uc791ee9e-0bfd-4&from=paste&height=79&id=ub1d1ae5c&originHeight=111&originWidth=531&originalType=binary&ratio=1&rotation=0&showTitle=false&size=6867&status=done&style=none&taskId=ube16e5fe-373c-41ce-a922-72a6638add9&title=&width=377.8000183105469)

​	该式子计算了“节点u”和“节点v”之间的Jaccard相似度。M11表示 u 的特征为1，v 的特征也为1。M01，M10类似。

## 3.2 防御假设
​	为了防御GCN上的对抗攻击，作者首先提出一个假设，那就是，GCN模型之所以很容易被攻击，是因为它极强的依赖于图结构和邻居节点特征的聚合。在被攻击图上训练得到的模型就容易受到这个攻击边界的影响。目前公认的一点是，在一个模型上训练得到的对抗样本可以迁移到其他模型中。目前对于GCN模型的对抗攻击是很成功的，原因在于这些被攻击的图可以直接被用于训练一个新模型。基于这种攻击方式，**一个可行的防御方法是让邻接矩阵变得可训练**（trainable）。
> 针对这种想法，作者做了验证。没有采取任何防御措施时节点以0.998的概率被误分类。但如果仅仅在最开始初始化边的权重，不修改其他东西，再对GCN模型训练，就可以让这个节点以0.912的概率正确分类。


​	在观察了攻击的特点后，作者做出了解释：

1. 对边做扰动比直接修改特征要更有效。 在现有的所有攻击方法（FGSM，JSMA和IG-JSMA）中，这一点都是奏效的。只对特征做干扰是很难修改目标节点的预测结果的。而且，很多攻击方法都是倾向于添加边而不是删除边，这样带来的攻击效果要更好。 
2. 第二，有着更多邻居节点的结点更难被攻击。这是在之前的工作中就被提到过的。之前就有学者发现，有着更高degree的结点在干净图和攻击图中都有着更好的分类准确率。
3. 最后，在攻击的时候，倾向于将目标节点和有着不同特征feature或者不同标签label的结点之间建立连接。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675499112520-ac4ca75b-dbb3-41a9-9caa-ef3b7be872a1.png#averageHue=%23f8f7f7&clientId=u719a6a43-2011-4&from=paste&height=344&id=u7442a884&originHeight=430&originWidth=1042&originalType=binary&ratio=1&rotation=0&showTitle=false&size=47017&status=done&style=none&taskId=ub58b12c4-2c69-45ad-94bc-5848cb4c482&title=&width=833.6)
## 3.3 防御策略及优化
​	基于以上对攻击的观察，作者提出一个假设，那就是提出来的防御方法之所以有效，是因为**GCN模型会对那些与当前结点特征相差很大的结点之间的边赋予更小的权重**。实验结果如下，发现确实是，Jaccard相似性分数更高的结点与当前结点的连边的权重更高，越低的权重越低。这里x轴的每个值表示目标节点附近的一条边。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675499281262-9c5f5fdd-5e4a-466c-b5f2-5e30267b01cf.png#averageHue=%23f8f6f5&clientId=u719a6a43-2011-4&from=paste&height=387&id=u0ee8af92&originHeight=587&originWidth=869&originalType=binary&ratio=1&rotation=0&showTitle=false&size=99077&status=done&style=none&taskId=u8ee232eb-3310-4c00-8af8-f3bb73ec58c&title=&width=573.2000122070312)

​	为了使得防御方法更有效，作者表示，**其实可以不需要去学习边的权重**。因为如果要学习边的权重的话，不可避免的就会带来额外的参数。所以可以直接基于如下几点做防御：

   1. 正常结点通常不会与许多相似性不高的结点之间建立连接。
   2. 学习过程本质上为连接两个不同节点的边分配低权重。

​	所以最终GCN-Jaccard的防御策略，就是**图的预处理**，就是“**图净化**”的思想。在训练前对给定的图进行预处理，检查图的邻接矩阵，也就是检查边。所有Jaccard相似度较低的节点（如相似度 = 0）的边被选为候选去除。虽然干净的图也可能有少量这样的边，但我们发现去除这些边对目标节点的预测几乎没有伤害。相反，在某些情况下，这些边的去除可以改善预测。
# 补充：
​	Integrated gradient的概念在2017年就被Google提出来了。积分梯度算作是一种解释深度学习模型输出结果的方法。Integrated gradient将输入的第i个特征的归因定义为：从baseline到输入之间的直线路径的路径积分。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675500970344-66dffcc3-4f68-4ee2-b806-1c092f285ccd.png#averageHue=%23f4f4f4&clientId=u91ba6999-6ab3-4&from=paste&height=71&id=u79dc90a0&originHeight=96&originWidth=808&originalType=url&ratio=1&rotation=0&showTitle=false&size=17008&status=done&style=none&taskId=ub8fe22ef-b723-4ddc-b0d0-dee6cd1cd8a&title=&width=601)

​	这里的baseline一般都是选择全0的矩阵或者向量。这里的baseline之所以选择全0，是因为当我们把一件事情的发生归结于某个原因的时候，没有该原因的情况就是一个baseline基线值。这里的基线值表示的就是无状态。积分梯度之所以被提出来，是因为研究者发现，对于下面的 f(x) 这个函数，当 x=1时，f(1)=1，f'(1) = 0。但是梯度应该是处处都存在的，不应该存在梯度为0的情况，所以要使用其他的梯度计算方法来得到梯度。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1675501059410-458e92fc-9d41-40bc-a5c8-b91a665c455a.png#averageHue=%23f8f8f8&clientId=u91ba6999-6ab3-4&from=paste&height=77&id=ub3c80ed1&originHeight=98&originWidth=617&originalType=url&ratio=1&rotation=0&showTitle=false&size=10286&status=done&style=none&taskId=u8bef1f70-2501-4181-8d60-b4a6bd0a45c&title=&width=483)

​	积分梯度和传统的梯度的差别其实是在于，传统的梯度只选取了当前点计算梯度，但是如果当前点的特征正好处在梯度饱和阶段，计算得到的归因就很小。但是积分梯度相当于是选择了当前输入到基线值之间的无限多个积分点进行加和，可以解决梯度饱和的问题。应用积分梯度的关键是选择一个好的baseline基准值，这样的基准值应该要不包含任何的信号，这样才能把归因结果归结于输入值。
