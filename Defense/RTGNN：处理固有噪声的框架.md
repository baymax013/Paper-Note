## 论文简述

论文题目：Robust Training of Graph Neural Networks via Noise Governance
论文地址：*https://arxiv.org/abs/2211.06614*
论文概况：基于NRGNN的RTGNN，针对固有噪声的处理

------

# 引言和相关工作

本文提出了一个新的RTGNN模型，这可以看作是以往工作NRGNN的优化。这两个模型都是基于**data固有噪声（标签噪声）和半监督**的，两个模型都是针对半监督GNN进行一个**robust training**。这篇文章就是以半监督GNN分类任务的robust training为例。
NRGNN的核心思想是，实际中半监督的sub-node的label是可能是稀缺且有误的，因此通过将有label的节点和无label的节点连接起来，进一步挖掘准确的伪标签，以提供更多的监督。但是直接这么连接，会导致有误的label在GNN消息传递中的错误传播，从而误导GNN对节点表示的学习。
RTGNN就是在NRGNN的基础上，**增加了“区分”这一操作**，将干净的、有误的label分开再进行半监督，同时还引入了自强化和正则化来减少标签噪声的影响。
总的来说，RTGNN用于具有稀缺和标签噪声的GNN的robust training，旨在明确的管理噪声，以实现从干净的label中学习，同时限制噪声label的影响。
# 预设置
目标：半监督GNN节点分类任务的robust training。
G=(V, A, X)，node特征d维，class共C类。GNN开始先计算一个logit向量**o**，用来计算节点p的分类概率。式子如下（softmax）：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677815346987-5a9ff162-4ddd-463e-baea-0bcd20a8b608.png#averageHue=%23f7f7f7&clientId=u7e594919-5c84-4&from=paste&height=74&id=u507c81f7&originHeight=92&originWidth=338&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=6690&status=done&style=none&taskId=u3c75778c-6217-48b3-a8f6-0358357f47c&title=&width=270.4)
标签噪声分两类：uniform和pair。uniform是指噪声是均匀、等可能的；pair是指标签只在非常相似的class中才出错。noise rate设置为小于0.5，即假设大部分打上的label是正确的。
# 方法论
## 总览
首先是于NRGNN一致，**先进行有无label的节点连接**，即通过学习一个edge predictor来推断有无label节点间的潜在联系。添加这些链接，可以导致更有效地消息传递，从而**缓解标签稀疏问题**。
然后就是RTGNN提出的优化，将**增强图送给两个对等的GCN进行显示的噪声治理**。这里用两个对等的GCN，是受mutual learning的启发，也与后续的正则化有关。具体而言，基于small-loss criterion准则区别label是否正确，分为_V_cl _和_V_ns_。分出_V_ns_后，RTGNN还要进一步从中识别其子集_V_sr_，这是对节点的prediction有信心，但对应的label却不一样（也就是可能错误分类的节点，但是模型觉得自己预测的label是正确的）。**对于_V_sr_，采用自强化监督，用模型预测的label去训练**，对于_V_sr_之外的，也就是预测不自信的那部分，用“下加权损失”训练。
对于无label的部分，与自监督技术类似，RTGNN对于预测自信的部分生成**伪标签**进行学习。最后对两个模型引入视图间的和视图内的正则化，以进一步防止模型过拟合噪声。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677825452500-92b52fa0-f7a3-4e31-a640-bb3052e81c28.png#averageHue=%23faf5ef&clientId=u7e594919-5c84-4&from=paste&height=333&id=udb9b1674&originHeight=416&originWidth=1148&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=150623&status=done&style=none&taskId=u6564f1d3-d7d4-4e3e-ac4e-ce64ed966a6&title=&width=918.4)
## 图增强
图增强的核心思想就是NRGNN里的工作，目的是得到一个增广图Augmented Graph，这个图多了一些有label和无label节点的新连接。具体而言，通过一个edge predictor来实现。
**edge predictor是encoder-decoder架构的。其中编码部分用的是GCN提取特征，解码部分利用编码表示计算非负的余弦相似度**。edge predictor基于负采样的重建目标训练，loss函数如下：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677826598496-fa0edb1c-839b-4890-87fd-e86d2b532d88.png#averageHue=%23f3f3f3&clientId=u71c5b0ba-0f5f-4&from=paste&height=47&id=ucfc025c9&originHeight=59&originWidth=509&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=10125&status=done&style=none&taskId=u81ab559e-a845-40ff-b581-55cdc37ba5b&title=&width=407.2)
_N_neg_是每个节点的负样本，_P_n_是负样本的分布，本文采用负抽样来提高计算效率，避免对负节点对的偏差。最后基于decoder得到的两个节点之间的潜在联系，即权值_w_ij_，生成增广图。
注意新添加的链接，首先不是对于全部节点而言的，而是选择top-K个最近的节点，同时还设置了阈值来过滤不可靠的信息。增广图表示如下：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677826971579-8a837014-76a3-43a0-8dfc-12ac0664b0e6.png#averageHue=%23f6f6f6&clientId=u71c5b0ba-0f5f-4&from=paste&height=76&id=u3072a7f9&originHeight=95&originWidth=476&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12153&status=done&style=none&taskId=u6e918119-c9cf-4a09-bd2c-038aff99dad&title=&width=380.8)
## 基于噪声治理的Robust Training
### Labeled Node的区分
这一步的目的是区分出_V_cl _和_V_ns_。
采用small-loss准则区分_V_cl _和_V_ns_。这里基于两个GCN的预测概率，用交叉熵得到的mutual loss作为节点的损失度量，后续所谓的“confidence”也用这个mutual loss来衡量。_V_cl _和_V_ns_的区分具体如下。这里前两个thresholds是clean node的最大上界，noise rate < 0.5，threshold_avg进一步确保小损失节点是clean的，而不管它们的相对级别。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677830260654-4bdb725e-97b6-4fc6-a099-c542b95c43e6.png#averageHue=%23f3f3f3&clientId=u71c5b0ba-0f5f-4&from=paste&height=148&id=ua9d6cfed&originHeight=185&originWidth=468&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=26810&status=done&style=none&taskId=u0dd43de5-ea4c-4abb-80a4-b5cc09b095d&title=&width=374.4)
_V_cl_ 的节点训练就是正常的有监督训练，**loss函数即分类损失**。_V_ns_还要进一步做处理，也就是下面的自强化监督。
### 自强化监督
这一步的目的是选择出_V_ns_中的_V_sr。_
基于两个GCN的预测概率，选定满足以下要求的节点来得到_V_sr_，这里C是label的数量。注意第二行的threshold值是与epoch的轮次 t 有关的，所以_V_sr_在每一轮epoch中都要更新。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677830853126-c7e2c583-821e-4f29-ba92-cb2dda9b7fc7.png#averageHue=%23f2f2f2&clientId=u71c5b0ba-0f5f-4&from=paste&height=99&id=ufd48b1cb&originHeight=124&originWidth=393&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14944&status=done&style=none&taskId=u75e14d88-3a52-4362-b28b-1e1ecddecc9&title=&width=314.4)
用于训练_V_sr_的loss也与有监督的分类损失类似，但在每轮epoch中都要计算一个学习率_μ(i)_，**学习率和loss式子**如下：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677831385741-923245c0-ea02-4b85-b588-2e298cfc2a76.png#averageHue=%23f2f2f2&clientId=u71c5b0ba-0f5f-4&from=paste&height=92&id=u494d16ce&originHeight=115&originWidth=323&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12402&status=done&style=none&taskId=ua8a48fd7-a724-46d1-9ae9-ed5b9fe2ae3&title=&width=258.4)

### Unlabeled Node的训练
与自监督一样，生成伪标签进行训练。所选择生成伪标签的节点满足下式。这里threshold_pse就是所谓的“confidence threshold”，用上面的loss衡量。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677831589396-ae4734e9-1195-48b7-ae35-29b1b39b8c56.png#averageHue=%23f2f2f2&clientId=u71c5b0ba-0f5f-4&from=paste&height=55&id=u3c29259d&originHeight=69&originWidth=540&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=11592&status=done&style=none&taskId=u2276cb10-595c-4905-bd7e-27d2d107415&title=&width=432)
### 一致性正则化
这里的“一致性”是受mutual learning启发，用了两个对等的GCN模仿彼此的预测。通过这种方式，他们互相教授和学习。这里带来了一个“mimicry loss”，采用了KL散度：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677832223478-8d15df31-a858-4976-be95-7816e61f1100.png#averageHue=%23efefef&clientId=u71c5b0ba-0f5f-4&from=paste&height=41&id=ud42b1c7d&originHeight=51&originWidth=418&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=7149&status=done&style=none&taskId=u908dbbab-e03a-4b70-a17c-f064847ef2e&title=&width=334.4)
另外由于遵循局部一致性（也称同质性假设），即链接的节点倾向于属于相同的类。这也带来了一个正则化项：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677832310351-8cff5206-311b-4e3e-8e69-63848b054e0b.png#averageHue=%23f3f3f3&clientId=u71c5b0ba-0f5f-4&from=paste&height=66&id=u372a2ea0&originHeight=83&originWidth=531&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12742&status=done&style=none&taskId=u98864182-e044-447c-92cc-7b4c4afd5aa&title=&width=424.8)
最终**loss函数里的正则化部分**就是由上述两个正则化项组成：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677832372001-1076056e-1561-48b1-868b-256e7a83bed8.png#averageHue=%23f5f5f5&clientId=u71c5b0ba-0f5f-4&from=paste&height=47&id=uc7e8e50a&originHeight=59&originWidth=265&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=4112&status=done&style=none&taskId=ub7ce9adc-c389-4900-9dba-e22dbccb7ca&title=&width=212)
## Loss函数的整理

- **有标签部分**：对于{_V_cl _}、{_V_sr _}以及{_V_ns - V_sr _}_，_也就是具有噪声标签且预测不自信的那部分，这些节点的训练loss可以整合为如下，第三行的γ是下加权因子。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677832946754-adc71aa3-2666-454b-9d92-fbfa5e3bd259.png#averageHue=%23f8f8f8&clientId=u71c5b0ba-0f5f-4&from=paste&height=202&id=u700c1e62&originHeight=252&originWidth=655&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=30992&status=done&style=none&taskId=uc66f67d5-a70d-43f7-bce0-4e88291fb75&title=&width=524)

- **伪标签部分**：

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677832980295-430722ac-479b-4947-86e3-c12a82ed08c5.png#averageHue=%23f5f5f5&clientId=u71c5b0ba-0f5f-4&from=paste&height=73&id=u45ac7792&originHeight=91&originWidth=501&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=10633&status=done&style=none&taskId=u408d428e-4b7a-48e1-912a-20ff5224dfd&title=&width=400.8)

- **总训练loss**：

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1677833080466-591f9e3b-341b-484b-9d89-db18b880adf5.png#averageHue=%23f5f5f5&clientId=u71c5b0ba-0f5f-4&from=paste&height=46&id=u66e11f16&originHeight=58&originWidth=377&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=5678&status=done&style=none&taskId=u58e8e3a7-a0b6-4bee-9d72-3d38ebc2c65&title=&width=301.6)
