# 10a.树
##### 树的种类很多，以下只讨论这种情况：
- 二叉树
- 树的每一个节点的分裂只涉及到一个特征
- 对于连续型变量，节点分裂规则按这种形式：$x_i<=t$
- 对于离散变量，分裂值让输入空间分成两个组

### 一、回归树
决策树将输入空间$\mathcal{X}$划分成$M$个区域：$\{R_1,...,R_M\}$,且
$$\mathcal{X}=R_1\cup R_2\cup ... \cup R_M$$ and $$R_i \cap R_j=\emptyset, \forall i\neq j$$
那么预测函数就是：
$$f(x)=\sum_{m=1}^{M} c_m\mathcal{1}(x\in R_m)$$
怎样得到$c_1,...,c_m$呢？
对于损失函数$l(\hat{y} ,y)=(\hat{y}-y)^2$来说，$c_m$会是：
$$\hat{c}_m=ave(y_i|x_i \in R_m)$$

如果我们让决策树作足够多的分裂，极限情况下每个区域只包含一个 x ，这样就过拟合了，所以我们需要控制树的复杂度，CART算法利用的是叶子节点的数量作为树的复杂度，也可以使用树的深度

记$|T|$表示树的复杂度，对于任意的复杂度，我们都希望在训练集上决策树可以最小化平方损失函数

做树的分裂时，我们使用贪心算法，即一个节点分裂时只考虑这个节点的性能最优，不做全局计划
###### 节点分裂标准
- $x=(x_1,...,x_d) \in R^d. (d features)$
- 分裂变量 $j \in \{1,...,d\}$
- 分裂值 $s \in R$.

$$R_1(j,s)=\{x|x_j \leqslant s\}$$
$$R_2(j,s)=\{x|x_j > s\}$$

- 对于每一个分裂变量 j 和分裂值 s：
$$\hat{c}_1(j,s)=ave(y_i|x_i \in R_1(j,s))$$
$$\hat{c}_2(j,s)=ave(y_i|x_i \in R_2(j,s))$$

- 最小化以下损失函数得到 j，s：
$$L(j,s)=\sum_{i:x_i \in R_1(j,s)} (y_i-\hat{c}_1(j,s))^2+\sum_{i:x_i \in R_2(j,s)} (y_i-\hat{c}_2(j,s))^2$$

- 怎样选择分裂变量和分裂值呢？
    
    
    对于每一个特征变量中的每一个分裂点都计算上面的损失函数，选择使损失函数最小的那个特征变量和那个分裂点，对于数值型特征变量，将该变量的所有值$x_{j(1)},...,x_{j(n)}$进行排序，我们只需要将临近值之间的值作为分裂值，比如使用两个临近点连线的中点
$$s_j \in \{\frac{1}{2}(x_{j(r)}+x_{j(r+1)})|r=1,...,n-1\}$$
    所以我们只需要尝试n-1个分裂值就可以了
    
- 然后不断的重复这个过程，那树分裂到什么情况下结束呢？

    - 可以限制树的深度
    - 可以限制每个叶子节点至少得含有多少个样本
    - 可以限制节点得至少含有多少个样本才可以继续分裂
    - 可以做剪枝（CART算法利用的使这种方法）
        - 先构造一个非常大的树，比如每个叶子节点含有小于5个样本，然后贪婪地将树修剪回根目录，在验证集上评估性能。去掉了一些分支平均损失就会变大，我们要找到变化程度最小的那个情况，我们可以迭代的尝试所有到根节点的剪枝情况，从而得到一系列的树模型，从中选择在验证集上表现最好的树模型

### 二、分类树
> Soft prediction returns the predicted probability of your data point belonging in one of the classes. Hard label is the class your model predict given the data point. The soft prediction gives you more information about the model's confidence in prediction. The higher the value for the predicted class, the more confident and accurate(in general) the prediction will be.


- 考虑多分类情况：$\mathcal{Y}=\{1,2,...,K\}$
- 记节点 m 代表$R_m$,含有$N_m$个样本
- 记在区域$R_m$中，类别k所占比例为：
$$\hat{p}_{mk}=\frac{1}{N_m}\sum_{\{i:x_i\in R_m\}} \mathcal{1} (y_i=k)$$
- 节点 m 的预测类别为：
$$k(m)=\arg\max_k \hat{p}_{mk}$$
- 节点 m 的预测类别概率分布为：
$$(\hat{p}_{m1},...,\hat{p}_{mK})$$

在节点 m 内，假设预测类别是$k(m)=\arg\max_k \hat{p}_{mk}$, 那么训练函数的错误率是$1-\hat{p}_{mk}$, 通常情况下，分类问题的损失函数是0/1损失，并且它也是容易实现的，但我们可能不用它，如果我们仅仅对树分裂一次，那么使用0/1损失函数很正常，但我们可以重复的不断的分裂节点，并不需要一次性分类成功。

##### 节点的不纯净度度量（Node Impurity Measures）
- Misclassification error:
$$1-\hat{p}_{mk}$$
- Gini index:
$$\sum_{k=1}^{K}\hat{p}_{mk} (1-\hat{p}_{mk})$$
- Entropy or deviance(equivalent to using information gain):
$$-\sum_{k=1}^{K}\hat{p}_{mk}\log\hat{p}_{mk}$$

##### 节点分裂标准
- 记$R_L$和$R_R$为节点 m 即将分裂成的左区域和右区域
- 假设$R_L$含有$N_L$个样本，假设$R_R$含有$N_R$个样本
- 记$Q(R_L)$和$Q(R_R)$分别是各自的不纯净度度量
- 最小化下面的值(weighted average of node impurities)来得到分裂变量和分裂值：
$$N_LQ(R_L)+N_RQ(R_R)$$

对于创建决策树来说，Gini和Entropy似乎更有效，他们偏向于选择更纯净的节点分裂情况，而不是只依赖误分类率，而且一个更纯净的节点分裂情况可能并不会改变误分类率。

### 三、缺失特征值的样本
如果缺失特征值，我们应该怎么做呢？
- 将缺失特征值的样本丢弃
- 具有特征手段的缺失值填补
- 对于类别性特征，将'missing'作为一个新的类别
- 对于决策树，我们可以使用surrogate split
    - 对于每一个内部节点，构造备选特征和分裂值的清单
    - 目标是尽可能的近似原始的分裂情况
    - 备选清单的顺序按照跟原始分裂情况有多近似来排列

### 四、类别型特征
- 假设我们有一个类别型特征，含有 q 个值，我们想要找到将它分成两个组的分裂值点，这种情况有$2^{q-1}-1$种分法，这样执行起来不是很容易
- 对于二分类的分类问题，这里有一个有效的算法：
    - 为每一个类别分配一个数值
    - 将它看成是连续型变量，排序后对它进行分裂值尝试
    - 这个算法来自书CART中

### 五、树有可解释性


# 10b.bootstrap
概率分布的参数，统计量，无偏估计，统计量的分布的参数

bootstrap样本是从原始样本$\mathcal{D}_n=(x_1,...,x_n)$中有放回的重新抽样 n 个实例
bootstrap方法是重复$\mathcal{B}$次bootstrap样本的抽取操作，用这$\mathcal{B}$组bootstrap样本来近似独立同分布的$\mathcal{B}$组原始样本，可以用这种方式近似得到‘均值’这个统计量的方差也就是标准误差，用这种方式得到的标准误差和用独立同分布的数据得到的标准误差非常近似

# 10c.Bagging and Random Forests
> Bagging是Bootstrap aggregating的缩写。 中文意思是自助聚合

### 一、集成方法(Ensemble Methods)
集成方法集合多个模型作为最终模型，有两种：
- 并行集成(Parallel ensenble):每个小模型都独立构建
    - 比如bagging and random forests
    - 集合很多模型来降低误差(high complexity,low bias??)
- 串行集成(Sequential ensembles)
    - 小模型是顺序生成的
    - Try to add new models that do well where previous models lack

### 二、求平均的好处

设$Z_1,Z_2,...,Z_n$独立同分布，$Z$服从$\mathbb{E} Z=\mu$和$VarZ=\sigma$

当我们用$Z_i$估计$\mu$时
- 是无偏的$\mathbb{E} Z_i=\mu$
- 估计的标准误差是$\sigma$

当我们用$Z_i$的平均值估计$\mu$时
- $\mathbb{E} [\frac{1}{n}\sum_{i=1}^{n}Z_i]=\mu$
- $Var [\frac{1}{n}\sum_{i=1}^{n}Z_i]=\frac{\sigma^2}{n}$

很明显，平均值的表现更好，那么，我们可以利用这个降低一般预测函数的标准误差吗？

设有B个独立同分布的训练集，可以得到B个预测函数：$\hat{f}_{1}(x),\hat{f}_{2}(x),...,\hat{f}_{B}(x)$,定义预测函数的平均
$$\hat{f}_{avg}=\frac{1}{B}\sum_{b=1}^{B}\hat{f}_{b}$$
对于固定的$x\in\mathcal{X}$,预测函数的平均是
$$\hat{f}_{avg}(x)=\frac{1}{B}\sum_{b=1}^{B}\hat{f}_{b}(x)$$
其中，$\hat{f}_{avg}$和$\hat{f}_{1}(x),\hat{f}_{2}(x),...,\hat{f}_{B}(x)$都是随机变量。$\hat{f}_{1}(x),\hat{f}_{2}(x),...,\hat{f}_{B}(x)$是独立同分布的(因为训练集是随机的)
- $\hat{f}_{avg}$和$\hat{f}_{b}$有相同的期望，即都是无偏估计
- 但是，$\hat{f}_{avg}$有更小的方差

$$\begin{array}{rcl} Var(\hat{f}_{avg}(x)) & = & \frac{1}{B^2}Var\left(\sum_{b=1}^{B}\hat{f}_{b}(x)\right)\\ \ & = & \frac{1}{B}Var\left(\hat{f}_{1}(x)\right)\end{array}$$

如此看来，似乎$\hat{f}_{avg}=\frac{1}{B}\sum_{b=1}^{B}\hat{f}_{b}$表现更好，但实际上我们并没有B个独立同分布的训练集，那么我们可以利用bootstrap method

### 三、Bagging
由bootstrap method从数据集$\mathcal{D}$中生成B个bootstrap samples$\mathcal{D}^1,\mathcal{D}^2,...,\mathcal{D}^B$，从而得到B个预测函数$\hat{f}_1,\hat{f}_2,...,\hat{f}_B:\mathcal{X}\rightarrow\mathbf{R}$，那么bagged预测函数是这些函数的组合
$$\hat{f}_{bag}(x)=combine\left(\hat{f}_1,\hat{f}_2,...,\hat{f}_B\right)$$
分类问题的combine方式是：consensus class or average probabilities.

对于回归问题bagged预测函数是
$$\hat{f}_{bag}(x)=\frac{1}{B}\sum_{b=1}^{B}\hat{f}_b(x)$$
    
- 表现相似：$\hat{f}_{bag}(x)$的表现跟在独立同分布的B个数据集上得到的bagged预测函数相似
    - $\hat{f}_{bag}(x)$和$\hat{f}_{b}(x)$的期望相同
    - $\hat{f}_{bag}(x)$有更小的方差

#### Random forest
随机森林的核心思想是：利用bagged决策树，但是更改树生成过程中的分裂标准，以此来降低基树之间的联系
- 对树的每一个节点做分裂时，在随机选择的m个特征中选择分裂变量
- 通常$m\approx\sqrt{p}$，其中p是训练集的所有特征
- 也可以利用交叉验证来确定m


# 11a.gradient_boosting
设有一个非线性的输入空间是一维的回归问题，针对这个问题我们可以构建很多模型，比如
- 核函数
- 树
- 神经网络

我们也可以简单的学习线性模型，但在学习模型之前，进行特征工程，即选择一些特征函数(设输入空间是$\mathcal{X}$)：
$$g_1,...,g_M:\mathcal{X}\rightarrow\mathbf{R}$$
那么我们的线性模型就会是：
$$f(x)\sum_{m=1}^{M}v_{m}g_{m}(x)$$
虽然$f(x)$是一个一维的实数，对于回归问题这正是我们想要的，但我们也经常称它为‘分数’
- 可以设置临界值，实现分类
- 可以经过转换得到概率
- 可以经过转换成为概率分布的参数估计
- 可以用来作为排名的依据

所以这是一个应用非常广泛的假设空间，

到目前为止，我们一直在讨论为模型做准备的一个步骤即我们手动的得到一个固定的基函数(或者说特征函数)，而以下我们要讨论的模型框架是，基函数可以自主构建，这个自主构建的过程将会是模型学习过程的一部分，叫做Adaptive basis function modeling
### 一、Adaptive basis function modeling
基假设空间$\mathcal{H}$由这些函数构成：$h:\mathcal{X}\rightarrow\mathbf{R}$，我们将会在这些函数中选择我们的基函数(特征)，所以combined hyposis space是：
$$\mathcal{f}_{M}=\left\{\sum_{m=1}^{M}v_{m}h_{m}(x) \ | \ v_{m}\in\mathbf{R},\ h_{m}\in\mathcal{H},\ m=1,...,M\right\}$$
假设有数据$\mathcal{D}=\left((x_1,y_1),...,(x_n,y_n)\right)$，模型训练是要选择$v_1,...,v_m\in\mathbf{R}$ and $h_1,...,h_m\in\mathcal{H}$去拟合数据$\mathcal{D}$

对于某个损失函数$l(y,\hat{y})$考虑通过empirical risk minimization学习模型：
$$\hat{f}=\arg\min_{f\in\mathcal{F}_{M}}\frac{1}{n} \sum{}_{i=1}^n\ l \left(y_i,f(x_i)\right)$$
则ERM目标函数是：
$$J(v_1,...,v_M,h_1,...,h_M)=\frac{1}{n} \sum{}_{i=1}^n \ l \left(y_i,\sum_{m=1}^{M}v_{m}h_{m}(x)\right)$$
这样的一个目标函数应该怎样优化呢？如果假设基函数可以被$\Theta=\mathbf{R}^b$参数化，那么目标函数是：
$$\mathcal{J}(v_1,...,v_M,\theta_1,...,\theta_M)=\frac{1}{n} \sum{}_{i=1}^n \ l \left(y_i,\sum_{m=1}^{M}v_{m}h(x;\theta_m)\right)$$
可以对$v_m$,$\theta_m$求导吗？可以利用SGD作优化吗？
- 对于某些假设空间和某些损失函数来说，可以的
- 神经网络就是属于这种类型

如果base hypothesis space是树怎么办呢？我们可以将树参数化吗？即使我们可以参数化的得到一些树，树函数对$\theta$也不是连续的，且完全不是可微的，以下我们讨论gradient boosting，它在任何情况下都使用：
- 损失函数训练预测函数$f(x_i)$是可微的
- 可以实现回归问题(e.g. regression tree)

### 二、前向阶段加法模型(Forward Stagewise Additive Modeling)
FSAM是自适应基函数模型的一个迭代优化算法
- 开始设置$f_0\equiv0$
- m-1 步，我们有
$$f_{m-1}=\sum_{i=1}^{m-1}v_{i}h_{i}$$
- 第m步，我们想要得到
    - step direction $h_m\in\mathcal{H} \ (i.e. a basis function)$
    - step size $v_i>0$
- 最后
$$f_m=f_{m-1}+v_{m}h_{m}$$

FSAM for ERM:
- 初始值$f_0\equiv0$
- 第$m=1$到M步
    - 计算
    $$(v_m,h_m)=\arg\min_{v\in\mathbf{R},h\in\mathcal{H}}\frac{1}{n}\sum_{i=1}^{n}l\left(y_i,f_{m-1}(x_i)+v_{m}h_{m}(x_i)\right)$$
    - 则
    $$f_m=f_{m-1}+v_{m}h_{m}$$
- 最终$f_m$就是我们的预测函数

#### $L^2$ Boosting
假设使用square loss，那么在迭代的每一步有目标函数：
$$J(v,h)=\frac{1}{n}\sum\limits_{i=1}^{n}\left(y_{i}-\left[f_{m-1}(x_i)\underbrace{+vh(x_i)}_{new piece}\right]\right)^2$$
<center>If $\mathcal{h}$ is closed under rescaling (i.e. if $h\in\mathcal{H}$, then $vh\in\mathcal{H}$ for all $h\in\mathbf{R}$), then don’t need $v$.</center>
假设不需要$v$，那么目标函数是：
$$J(h)=\frac{1}{n}\sum\limits_{i=1}^{n}\left( [y_{i}-f_{m-1}(x_i)-h(x_i)] \right)^2$$

可以看出：
- 第m步实际上是在用最小二乘法拟合残差
- 如果我们的基函数是回归函数，那么我们就可以解决回归问题

### 三、AdaBoost
AdaBoost is FSAM With Exponential Loss
先省略具体的算法细节
adaboost也是每次迭代时拟合残差吗？毕竟损失函数变了

### 四、Gradient boosting / "Any boost"
我们知道怎样去做带有固定损失函数的FSAM，比如square loss,absolute loss,exponential loss，这些情况最后都是落到了我们知道怎样解决的问题，但对于一般性的FSAM，不知道该怎么去解决，例如logistic loss,cross-entropy loss,以下是解决方案，适用于所有损失函数的解决方案。

我们想要去最小化下面的函数：
$$J(f)=\sum\limits_{i=1}^{n}l(y_i,f(x_i))$$
如果我们对“$f$”求导而不管$f$到底是什么呢？$J(f)$仅仅只依赖于那$n$个训练数据的预测$f$，所以目标函数可以是：
$$J(f)=\sum\limits_{i=1}^{n}l(y_i,f_i)$$
所以，negative gradient step direction at $f$ 是：
$$\begin{array}{rcl} -g & = & -\nabla_{f}J(f)\\ \ & = & -\left(\partial_{f_1}l(y_1,f_1),...,\partial_{f_n}l(y_n,f_n)\right)\end{array}$$
$-g\in\mathbf{R}^n$是我们在$n$个训练数据点上的预测值想要做出的改变
然后以下面的准则寻找基函数：
$$\min_{h\in\mathcal{H}}\sum\limits_{i=1}^{N}(-g_{i}-h(x_i))^2$$
这是在假设空间$\mathcal{H}$上最小二乘回归问题，是想找到最贴近$-g$的$h(x)\in\mathcal{H}$来作为我们的step direction

最后，我们选择step size，有两种方法供选择：
- 第一种$v_m=\arg\min\limits_{v>0}\sum\limits_{i=1}^{n}l(y_i,f_{m-1}(x_i)+v+{m}h_{m}(x_i))$
- 第二种
    - $v=1$作为所有的gradient step
    - 选择一个固定的$v\in(0,1)$,这种叫shrinkage parameter
    - $v=0.1$是一般选择，可以通过超参数优化

### 五、XGBoost
$J(f)$的二阶泰勒展开：
$$J(f+r)=J(f)+[\nabla_{f}J(f)]^{T}r+\frac{1}{2}r^{T}[\nabla_{f}^{2}J(f)]r$$
对于$J(f)=\sum\limits_{i=1}^{n}l(y_i,f_i)$,有
$$J(f+r)=\sum\limits_{i=1}^{n}[l(y_i,f_i)+g_{i}r_{i}+\frac{1}{2}h_{i}r_{i}^{2}]$$
其中，$g_i=\partial_{f_i}l(y_i,f_i)$ and $g_i=\partial_{f_i}^{2}l(y_i,f_i)$
$r$就是前面的step direction，寻找使$J$达到最小化的$r$，在目标函数后加入惩罚项
$$\Omega(r)=\gamma T+\frac{1}{2}\lambda\sum\limits_{i=1}^{T}w_j^2$$
其中$r\in\mathcal{R}$是基假设空间中回归树，$T$是叶子节点的个数，$w_j$是第$j$个叶子节点的预测值

所以第$m$步的目标函数是
$$J(r)=\sum\limits_{i=1}^{n}[g_{i}r(x_i),\frac{1}{2}h_{i}r(x_i)^2]+\Omega(r)$$
在XGBoost中同时也用这个目标函数去决定树的分裂


对目标函数进一步简化
   - 对于一个已知的树，记$q(x_i)$数据$x_i$落在了第几个node，记$w_j$为第$j$个node的预测值
   - XGBoost的每一步都是在寻找一颗树让下面的目标函数达到最小化：

$$\sum\limits_{i=1}^{n}[g_{i}w_{q(x_i)}+\frac{1}{2}h_{i}w_{q(x_i)}^{2}]+\gamma T+\frac{1}{2}\lambda\sum\limits_{i=1}^{T}w_j^2$$
$$=\sum\limits_{leaf nodej=1}^{T}\left[\left(\underbrace{\sum\limits_{i\in{I_j}}g_i}_{G_i} \quad w_j\right)+\frac{1}{2}\left(\underbrace{\sum\limits_{i\in{I_j}}h_i}_{H_j}+\lambda\right)w_j^2\right]+\gamma T$$

其中，$I_j=\{i|q(x_i)=j\}$，也就是那些被分配到同一个node的数据的下标，所以
$$=\sum\limits_{j=1}^{T}\left[G_{j}w_{j}+\frac{1}{2}(H_{j}+\lambda)w_{j}^2\right]+\gamma T$$

对于固定的$q(x)$，(例如固定树呈现的最终的分区)，目标函数是关于叶子节点预测值$w_j$的二次函数，所以容易得到关于$w_j$的最小值点：
$$w_j^*=G_{j}/(H_j+\lambda)$$

将$w_j^*$代回去，可以得到$J$在$w_j$上的最小值
$$-\frac{1}{2}\sum\limits_{j=1}^{T}\frac{G_j^2}{H_{j}+\lambda}+\gamma T$$
我们可以把它看成分区好的树模型的loss，如果时间允许，我们当然可以对所有可能的树做搜索优化，一般情况下是用贪婪算法对树进行一步一步的分裂

假设我们考虑将一个节点的数据分裂成左右两个节点：$L$ and $R$，那么这次分裂的loss是
$$-\frac{1}{2}\left[\frac{G_L^2}{H_{L}+\lambda}+\frac{G_R^2}{H_{R}+\lambda}\right]+2\gamma$$

分裂前的loss，例如树模型只有一个叶节点
$$-\frac{1}{2}\left[\frac{(G_{L}+G_{R})^2}{H_{L}+H_{R}+\lambda}\right]+\gamma$$

可以定义节点分裂前后的loss的差为Gain
$$Gain=\frac{1}{2}\left[\frac{G_L^2}{H_{L}+\lambda}+\frac{G_R^2}{H_{R}+\lambda}-\frac{(G_{L}+G_{R})^2}{H_{L}+H_{R}+\lambda}\right]-\lambda$$

最后，树模型构建的规则是：
递归的选择分裂变量和分裂值使Gain最大化
