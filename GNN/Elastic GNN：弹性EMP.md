## 论文总结：

论文题目：Elastic Graph Neural Networks

论文地址：http://proceedings.mlr.press/v139/liu21k/liu21k.pdf

论文概况：基于L1范式的弹性图信号估计器、弹性消息传递EMP

论文code：https://github.com/lxiaorui/ElasticGNN



## 主要思想：

目前主流的GNN中通过消息传递（MP）聚合特征，而绝大部分这些MP已经被证明是基于L2的图平滑，实现的是一种全局的图平滑。由于基于L2的方法强制全局平滑，并且平滑水平通常在整个图中共享，所以文章的出发点在于**通过L1的图平滑来增强局部平滑**。文章给出的原因是图中往往会有聚类，可以理解为集群/区域，不同区域的平滑程度可能是不同的，也就是不同集群之间的特征差异可能会很大，特征理应是在集群内部是平滑的，因此需要增强GNN的局部平滑自适应能力。

注意，Pro-GNN中也提到了图平滑可以当作增强GNN的robust，因为对抗估计喜欢连接特征差异较大的节点，Pro-GNN把特征平滑当作了优化问题中式子的最后一项。

文章受 _trend filtering 趋势滤波_ 的启发，采用L1范式去实现这一目标，理由是：

- 基于L1的方法对大值的惩罚更少，从而更好地保留不连续或非光滑信号
- 基于L1的方法倾向于促进信号稀疏性以换取不连续

文章先是基于L1范式提出了一种弹性图信号估计器，然后基于此设计了一种利于BP的优化算法，得到了最终新的消息传递方案：弹性消息传递 EMP。将EMP替换GNN中的MP，记作Elastic GNN。
## Elastic Graph Signal Estimator：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1686280126775-2a43f391-7b53-47a2-9278-fe5795bbca14.png#averageHue=%23f6f5f4&clientId=u746ebadd-fb33-4&from=paste&height=128&id=u685eb2a0&originHeight=160&originWidth=698&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=22020&status=done&style=none&taskId=ub5f4c765-8453-4e5e-b768-2732b7f4b24&title=&width=558.4)

上述（8）式是弹性图信号估计器经过一系列优化，得到的最终表达式，本质上同样也是个优化问题。

第一项有两处优化：**归一化Δ、L21范式**。

在图总变化和图趋势滤波的文献中，经常忽略归一化步骤，直接使用图差算子，如GTF。为了获得更好的数值稳定性和处理真实图中不同节点度，文章提出用节点度的平方根对事件矩阵的每一列进行归一化：

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1686280359545-bd847583-0d21-433c-a137-ddfc42a72f6e.png#averageHue=%23f9f8f7&clientId=u746ebadd-fb33-4&from=paste&height=86&id=uaf72f1c9&originHeight=108&originWidth=539&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=10512&status=done&style=none&taskId=u969a1f50-de7b-47c1-83b8-904a5f12bc1&title=&width=431.2)

而从L1改用L21，是由于现实世界图中的节点特征通常是多维的，L1中定义的估计量能够处理多维数据，因为来自不同维度的信号在L1和L2下是可分离的，但这种估计量独立对待每个特征维度，**没有利用特征维度之间的潜在关系**。然而，边缘节点差异的稀疏模式可以在特征维度之间共享。为了更好地利用这种潜在的相关性，文章通过L21范数来耦合多维特征。

第二项中没有什么需要特别注意的，其中 ”tr(A^T, B) = inner product(A, B)“。图拉普拉斯矩阵也照常用的是归一化的图拉普拉斯。

第三项则是一个budget，拉近F和输入X的距离。
## Elastic Message Passing：
文章该部分的推导比较复杂，直接看优化结论。其中γ和β是步长，λ1和λ2是两个超参。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1686280839894-adb37ba6-260d-41cd-88f0-60174de1b374.png#averageHue=%23f1f0ef&clientId=u746ebadd-fb33-4&from=paste&height=285&id=u702c7585&originHeight=356&originWidth=1063&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=69253&status=done&style=none&taskId=u2ab66091-868f-4e6a-9c68-c0299476d4f&title=&width=850.4)

将上面的EMP融入到GNN中，文章将这系列的GNN命名为Elastic GNN：

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1686281034674-3a8e36be-f49c-408c-b046-dc4a21152fe1.png#averageHue=%23f9f7f6&clientId=u746ebadd-fb33-4&from=paste&height=56&id=ub84a5190&originHeight=70&originWidth=428&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=8669&status=done&style=none&taskId=ub991ae1e-9c1d-42da-8561-de7a6a52e42&title=&width=342)
