## GLoVe(Global Vectors for Word Representation)

GloVe 模型包含一个训练在单词-单词的共同出现次数上的加权的最小二乘模型

## Co-occurrence Matrix

假设单词与单词的 co-occurrence matrix 矩阵用$X$表示，𝑋𝑖𝑗表示单词 𝑗出现在单词$i$的上下文中的次数， $X_{i}=\sum_{k} X_{i k}$表示任何一个单词 k 出现在单词 i 的上下文中的次数，
$$
P_{i j}=P\left(w_{j} \mid w_{i}\right)=\frac{X_{i j}}{X_{i}}
$$
表示单词$j$出现在单词$i$上下文中的概率,所以填充这个矩阵需要遍历一次语料库

## 目标函数

在 skip-gram算法中，我们在输出层使用的是 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 函数计算单词$j$出现在单词$i$上下文的概率
$$
Q_{i j}=\frac{\exp \left(\vec{u}_{j}^{T} \vec{v}_{i}\right)}{\sum_{w=1}^{W} \exp \left(\vec{u}_{w}^{T} \vec{v}_{i}\right)}
$$
如果我们将这个用于全局的数据的话，那么交叉熵损失函数就可以这么算：
$$
I=-\sum_{i \in \operatorname{corpus}} \sum_{j \in \text { context }(i)} \log Q_{i j}
$$
这个公式的本质就是在上一节讲的一句话的skip-gram上上升到对整个文本的处理。如果我们考虑单词上下文大小以及文本大小为 𝑊W.那么交叉熵损失函数可以写成：
$$
J=-\sum_{i=1}^{W} \sum_{j=1}^{W} X_{i j} \log Q_{i j}
$$
上面公式面临的一个问题是，在计算整个文本的时候，计算 𝑄 的 𝑠𝑜𝑓𝑡𝑚𝑎𝑥函数，这个计算量太大了。所以下面想办法优化一下：所以我们根本就不使用交叉熵损失函数，而是使用最小二乘法，那么损失函数就是下面这样：
$$
\hat{\jmath}=\sum_{i=1}^{W} \sum_{j=1}^{W} X_{i}\left(\hat{P}_{i j}-\hat{Q}_{i j}\right)^{2}
$$
其中 $\hat{P}_{i j}=X_{i j}$ and $\hat{Q}_{i j}=\exp \left(\vec{u}_{j}^{T} \vec{v}_{i}\right)$是非正态分布的. 这里的 𝑋𝑖𝑗 等价于 j 出现在 i 的上下文的次数， 而 𝑄̂ 𝑖𝑗是我们通过 skip-gram 预测的次数，所以是最小二乘法。这样的计算量还是很大，习惯上取个对数，公式就变成下面这样了：
$$
\begin{aligned}
\hat{J} &=\sum_{i=1}^{W} \sum_{j=1}^{W} X_{i}\left(\log (\hat{P})_{i j}-\log \left(\hat{Q}_{i j}\right)\right)^{2} \\
&=\sum_{i=1}^{W} \sum_{j=1}^{W} X_{i}\left(\vec{u}_{j}^{T} \vec{v}_{i}-\log X_{i j}\right)^{2}
\end{aligned}
$$
上面的公式中直接使用 𝑋𝑖不一定能够达到最优，因此我们选择 𝑓(𝑋𝑖𝑗)，使用上下文来表示以提高准确率：
$$
\hat{\jmath}=\sum_{i=1}^{W} \sum_{j=1}^{W} f\left(X_{i j}\right)\left(\vec{u}_{j}^{T} \vec{v}_{i}-\log X_{i j}\right)^{2}
$$
