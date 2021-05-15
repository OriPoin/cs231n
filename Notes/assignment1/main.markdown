# Assignment1

## Setup ( Conda)

使用本地环境，实验环境为 Gentoo Linux

```bash
neofetch
```

![](sysinfo.png)

### 初始化 Git

使用 git 管理代码。

```bash
git init
git add ./
git commit -sm 'init repo'
```

### 初始化 Miniconda 虚拟环境

Python 使用 Minconda 搭建的虚拟环境。

```bash
conda create -n cs231n python=3.7 #建立使用Python 3.8的虚拟环境
conda activate cs231n # 激活cs231n虚拟环境
pip install numpy scipy ipykernel imageio matplotlib future autopep8
```

```bash
which python
/home/oripoin/.conda/envs/cs231n/bin/python
```

## Goals

使用 KNN 和 SVM/softmax 生成传输路径

## k-Nearest Neighbor classifier

KNN 包括两步：

- 训练阶段，分类器获取数据，记忆
- 测试阶段，分类器

运行 knn.ipynb 的脚本

### 1. 首先初始化 Jupyter Note Book

### 2. 下载数据集

```bash
cd assignment1/cs231n/datasets
./get_datasets.sh
```

### 3. 重载数据，获取大小

```bash
Training data shape:  (50000, 32, 32, 3)
Training labels shape:  (50000,)
Test data shape:  (10000, 32, 32, 3)
Test labels shape:  (10000,)
```

### 4. 每一个类别中的图片示例

![](ShowExamples.png)

### 5. 将训练集和测试集重新排列为一维数组(行向量)

训练集 (5000, 3072)
测试集 (500, 3072)

### 6. 使用两层循环嵌套计算距离

```python
for i in range(num_test):
    for j in range(num_train):
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        diff = np.square(X[i] - self.X_train[j])
        dists[i][j] = np.sum(diff)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
return dists
```

运行代码，通过 htop 简单查看可以发现只使用了一个 CPU 核心

![](LOOP2_CPU.png)

![](LOOP2.jpg)

### 7.图中越亮表示距离越大，一行特别亮代表该测试数据与所有的训练集距离都大，一列特别亮代表该训练数据与所有的测试集都远

### 8.分别使用 K=1 和 K=5 分类

```python
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

index_lbl = np.argsort(dists[i][:])[:k] #截取前K个最近的点
closest_y = self.y_train[index_lbl]

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

dic = {}
for label in closest_y:
    if label in dic.keys():
        dic[label] += 1
    else:
        dic[label] = 1

values = list(dic.values())
keys = list(dic.keys())

y_pred[i] = keys[values.index(max(values))]

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
```

![](KNN_1.png)

### 9.使用一个循环和全部矢量化

```python
        for i in range(num_test):
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            diff = np.square(X[i] - self.X_train)
            dists[i] = np.sum(diff, axis=1)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # (x-y)^2 = x^2-2xy+y^2
        dists = np.expand_dims(np.sum(np.square(X), axis=1), axis=1) - 2 * X.dot(self.X_train.T)
                 + np.expand_dims(np.sum(np.square(self.X_train), axis=1), axis=0)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
```

### 10.比较三种方法的时间

```
Two loop version took 24.084691 seconds
One loop version took 20.828722 seconds
No loop version took 0.145912 seconds
```

### 11.交叉验证

选取以下两组数据，K 最优值为 8，准确率为 0.294000

```python
k_choices = [n for n  in range(1,100,5)]
k_choices = [n for n  in range(1,20)]
```

![](K_1-100.jpg)

![](K_1-20.jpg)

## Training a SVM and Softmax

支持向量机

### 1. 数据预处理

- 将数据集分为训练，验证，测试三个集合
- 图片转为一维向量
- 输入特征归一化

![](SVM_AVG.png)

### 2. 计算梯度

其实可以与损失一起计算


```python
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
        if j == y[i]:
            continue
        margin = scores[j] - correct_class_score + 1  # note delta = 1
        if margin > 0:
            dW[:, y[i]] += -X[i, :]
            dW[:, j] += X[i, :]
dW /= num_train
dW += 2 * reg * W
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
```

### 3.损失函数向量化

### 4.随机梯度下降(SGD)

使用随机梯度下降获得权重，1500次随机操作后，损失情况

![](SVM_1.jpg)

### 5.使用验证集调整超参数，获得 0.39 左右的分类精度(0.385)


![](SVM_2.png)

```
overflow encountered in double_scalars
```

使用 scipy.special.logsumexp 函数处理很小的数值，防止溢出.

调整超参数后上述问题不再复现

### 6.权重可视化

![](SVM_3.png)

每个图像都被其他类型的图像“平均了”。因为这个结果是整体训练集和权重的产物，是全局的最优解。

## Softmax

### 1.数据预处理

### 2.循环计算交叉熵损失

图像种类有10种，每种图像的数量点相同，损失期望为$-log(\frac{1}{10})$

```
loss: 2.355027
sanity check: 2.302585
```

合理

### 3.验证loss和grad

### 4.向量化Softmax并验证

### 5.调整超参数

best validation accuracy achieved during cross-validation: 0.350000

### 6.可视化权重

![](Softmax_0.png)

## Two-Layer Neural Network

## Higher Level Representations: Image Features
