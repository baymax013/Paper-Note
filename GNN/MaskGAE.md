## 论文总结：

论文题目：What's Behind the Mask: Understanding Masked Graph Modeling for Graph Autoencoders

论文地址：https://arxiv.org/abs/2205.10053

论文概况：将GAE和Mask策略结合

论文code：https://github.com/EdisonLeeeee/MaskGAE



## 主要思想

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1686882210949-b1bf3e20-c16a-4498-809c-e001358350cd.png#averageHue=%23f6f5f3&clientId=ucc26426d-86eb-4&from=paste&height=349&id=ua01991b4&originHeight=436&originWidth=894&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=109569&status=done&style=none&taskId=u713ae261-542b-4f25-bb2b-c16edb1016f&title=&width=715.2)

由Mask掩码策略在CV和NLP领域中的广泛应用带来的启发，文章将这种很有前途的**自监督学习策略**，名为掩码自编码的思想融入到了Graph上，也就是上图右侧的MGM。将MGM当作GAE自监督学习策略的pretext任务，就是文章所提出的模型MaskGAE。

GAE是2016年所提出的一个较为简单的LinkPred模型，GAE是专门用于做链路预测任务的，GAE也是一个自监督的图自动编码模型（GAE全称为Graph AutoEncoder）。GAE基于GCN当作编码器encoder来学习节点表征，接着通过inner product直接计算两个节点间链路的概率。而MaskGAE可以理解为在GAE的基础上，提前针对Graph的结构进行处理。模型采用选定的Mask策略来Mask掉Graph上一部分边，然后将Mask-Graph送入“encoder-decoder”这一经典架构中。

## MGM设计
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1686884740953-973ba2aa-0d53-432e-b344-65acbc8e5814.png#averageHue=%23d2b78e&clientId=ucc26426d-86eb-4&from=paste&height=215&id=uca7f2f4f&originHeight=269&originWidth=1059&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=128424&status=done&style=none&taskId=ua0d4642c-d952-40f8-851a-7152296e31c&title=&width=847.2)

MaskGAE较于GAE来说，最重要的肯定就是如何制定所谓的MGM Task。文章中提到了两个Mask策略：Edge-wise random masking和Path-wise random masking。

- Edge-wise random masking简单而直接的采样特定分布的边子集采样，例如采样边分布为Bernoulli分布。
- Path-wise random masking是文章提出的一种新的结构Mask策略，在采用过程中以路径作为基本处理单元。这里文章执行简单的Random Walk来采样Mask的边集（即路径Path）。这种MGM可以更好地利用结构依赖模式，并为更有意义的MGM任务捕获高阶接近性。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1686885338568-f44699ae-49a7-4b34-9dbe-0e77c6c0ce61.png#averageHue=%23f8f7f6&clientId=ucc26426d-86eb-4&from=paste&height=278&id=u022d6ade&originHeight=348&originWidth=727&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=102061&status=done&style=none&taskId=ua18c74a9-34c5-43ed-9182-efeea92b622&title=&width=581.6)

MaskGAE和一些现有的对比方法都对图应用执行掩蔽。这些对比方法使用边掩蔽作为增强来生成不同的结构视图进行对比，而MaskGAE使用边掩蔽来构建有意义的监督信号。其中Mask策略一个重要作用就是可以**减少成对子图视图之间的冗余**，从而促进自监督学习方案。如上图所示，左侧（a）中没有采用masking，以 a 和 b 为中心视角进行消息传递得到的1-hop和2-hop子图，它们的冗余度就比较高。右侧的（b）中通过mask掉 a 和 b 之间的链接，就可以显著降低1-hop和2-hop子图的冗余度。

## 编码-解码
制定和选定好了相应的MGM作为pretext task，后续就是将Mask-Graph送入到“Encoder-Decoder”架构中。这里的Encoder依旧和GAE及绝大多数论文一致，采用的是一个两层的GCN。
Decoder部分包括了两个decoder：structure decoder和degree decoder。

- 结构解码器其实是GAE中的内容，Graph经过GCN编码后可以通过inner product或MLPs进行解码。
- 度解码器是文章作为一个辅助模型来平衡接近度和结构信息。由于图结构本身的监督信号丰富，所以可以迫使模型在Mask-Graph中近似节点度，以方便训练。文章定义度解码器如下，本质上也是MLPs。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1686886170607-3cf64c4b-45b5-46fd-bb81-011cc8cad85e.png#averageHue=%23faf8f7&clientId=ucc26426d-86eb-4&from=paste&height=56&id=u698fec7e&originHeight=70&originWidth=295&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=5623&status=done&style=none&taskId=u84906ff3-9910-45b0-8775-f2a17e3c621&title=&width=236)

模型的优化目标，即总Loss函数也对应Decoder分为两个部分：

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1686886198278-7f9b57d6-efa6-43bc-8bc5-bad1f19d936c.png#averageHue=%23faf9f8&clientId=ucc26426d-86eb-4&from=paste&height=60&id=u5e1ad034&originHeight=75&originWidth=334&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=5575&status=done&style=none&taskId=u4fab5cbd-945d-4f16-b7ff-3609caa8261&title=&width=267.2)

其中L_GAEs就是GAE中的内容。L_deg对应degree decoder，为回归损失，衡量的是节点度的预测与Mask-Graph中原始节点度的匹配程度。文章根据节点级别计算近似度和原始度之间的均方误差（MSE），其中deg_mask(·)表示Mask-Graph中的节点度。本质上，L_deg可以作为一个正则化器，其定义如下：

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2381046/1686886305615-dd1c7b32-c1be-430d-a094-f64369eec168.png#averageHue=%23f9f8f7&clientId=ucc26426d-86eb-4&from=paste&height=78&id=ua1ed94f8&originHeight=98&originWidth=578&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=15647&status=done&style=none&taskId=u8d0e3d9e-8eb6-4224-a8c7-28759104310&title=&width=462.4)
