
# Tutorial：手写一个简单线性层 `SimpleLinear`

---

## 1. 目标

这段代码实现了一个最基础的深度学习模块：**线性层（Linear Layer / Fully Connected Layer）**。

代码如下：

```python
class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
        self.weight.requires_grad_(True)
        self.bias = torch.zeros(out_features, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias
```

它实现的数学公式是：

\[
y = xW^T + b
\]

---

# 2. 这段代码整体在干什么

它定义了一个类 `SimpleLinear`，作用是：

- 创建可训练参数 `weight` 和 `bias`
- 接收输入 `x`
- 输出线性变换结果

即：

\[
\text{输入} \rightarrow \text{线性变换} \rightarrow \text{输出}
\]

它本质上就是手写版的 `nn.Linear`。

---

# 3. Python 语法与关键字解读

---

## 3.1 `class`

```python
class SimpleLinear:
```

### 作用
定义一个类。

### 含义
你可以把类理解成一个“模板”或“蓝图”，用来创建对象。

这里 `SimpleLinear` 这个类表示“一种线性层”。

---

## 3.2 `def`

```python
def __init__(...)
def forward(...)
```

### 作用
定义函数/方法。

在类里面定义的函数，通常叫“方法”。

---

## 3.3 `__init__`

```python
def __init__(self, in_features: int, out_features: int):
```

### 作用
构造方法，创建对象时自动执行。

比如：

```python
layer = SimpleLinear(8, 4)
```

就会自动调用：

```python
__init__(self, 8, 4)
```

### 含义
初始化这个层的参数。

---

## 3.4 `self`

### 作用
代表“当前对象本身”。

例如：

```python
self.weight
self.bias
```

表示当前这个层对象自己的属性。

### 注意
`self` 不是 Python 关键字，但它是类方法中约定俗成的写法。

你可以理解为：

- `layer.weight`
- `layer.bias`

就是这个对象里保存的数据。

---

## 3.5 类型标注

```python
in_features: int
out_features: int
x: torch.Tensor
-> torch.Tensor
```

### 作用
说明参数和返回值“期望是什么类型”。

- `in_features: int` 表示希望是整数
- `x: torch.Tensor` 表示希望是张量
- `-> torch.Tensor` 表示返回值是张量

### 注意
这是提示，不是强制。

---

## 3.6 `return`

```python
return x @ self.weight.T + self.bias
```

### 作用
把函数结果返回出去。

---

# 4. `in_features` 和 `out_features` 是什么

这是理解线性层最关键的两个量。

---

## 4.1 `in_features`

表示：

> 每个输入样本有多少个特征

比如一个输入向量是：

```python
[身高, 体重, 年龄]
```

那它有 3 个数，所以：

\[
in\_features = 3
\]

也就是输入维度。

---

## 4.2 `out_features`

表示：

> 这层要输出多少个新特征

如果你想把输入的 3 维信息，映射成 5 个新的特征，那么：

\[
out\_features = 5
\]

也就是输出维度。

---

## 4.3 最直观理解

- `in_features`：输入有几个数
- `out_features`：输出想变成几个数

例如：

```python
SimpleLinear(8, 4)
```

表示：

- 输入是 8 维
- 输出是 4 维

即：

\[
\mathbb{R}^8 \rightarrow \mathbb{R}^4
\]

---

# 5. 参数 `weight` 和 `bias` 在干什么

---

## 5.1 `weight`

```python
self.weight = torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
```

### 含义
创建权重矩阵：

\[
W \in \mathbb{R}^{out\_features \times in\_features}
\]

### 为什么这个 shape 是 `(out_features, in_features)`

因为：

- 每一个输出都要依赖所有输入
- 一行权重负责生成一个输出
- 总共有 `out_features` 个输出

所以矩阵有：

- `out_features` 行
- `in_features` 列

---

## 5.2 `bias`

```python
self.bias = torch.zeros(out_features, requires_grad=True)
```

### 含义
创建偏置向量：

\[
b \in \mathbb{R}^{out\_features}
\]

每个输出维度对应一个偏置。

---

# 6. 初始化为什么这样写

---

## 6.1 权重初始化

```python
torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
```

表示权重先从标准正态分布采样，再按下面比例缩放：

\[
\frac{1}{\sqrt{in\_features}}
\]

### 目的
让初始值不要太大，也不要太小，避免数值不稳定。

如果输入维度很大，直接用普通随机数，输出可能会波动太大。

这是深度学习里常见的初始化思想。

---

## 6.2 偏置初始化为 0

```python
torch.zeros(out_features, requires_grad=True)
```

这是一种常见做法，简单且稳定。

---

# 7. `requires_grad=True` 是什么

---

## 7.1 作用

表示这个张量需要参与梯度计算。

也就是说：

- `weight` 是可学习参数
- `bias` 也是可学习参数

训练时，损失函数会对它们求导：

\[
\frac{\partial L}{\partial W},\quad \frac{\partial L}{\partial b}
\]

然后优化器根据梯度更新它们。

---

## 7.2 数学意义

训练过程中，参数更新一般类似：

\[
W \leftarrow W - \eta \frac{\partial L}{\partial W}
\]

\[
b \leftarrow b - \eta \frac{\partial L}{\partial b}
\]

其中 \(\eta\) 是学习率。

所以 `requires_grad=True` 的意思是：

> 这些值不是固定的，而是模型要学出来的。

---

# 8. `forward` 在干什么

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x @ self.weight.T + self.bias
```

它实现的是：

\[
y = xW^T + b
\]

---

## 8.1 为什么要 `.T`

权重保存的 shape 是：

\[
(out\_features,\ in\_features)
\]

而输入一般是：

\[
(batch,\ in\_features)
\]

矩阵乘法要求：

\[
(batch,\ in\_features) @ (in\_features,\ out\_features)
\]

所以需要把 `weight` 转置：

\[
W^T \in \mathbb{R}^{in\_features \times out\_features}
\]

---

## 8.2 `@` 是什么

`@` 是 Python 的矩阵乘法符号。

```python
x @ self.weight.T
```

表示矩阵乘法，不是普通逐元素相乘。

---

## 8.3 `+ self.bias` 是什么

加上偏置项 \(b\)。

如果输入是一个 batch，PyTorch 会自动广播，把 `bias` 加到每一行上。

---

# 9. 数学表达

---

## 9.1 单个样本

如果输入：

\[
x \in \mathbb{R}^{d_{\text{in}}}
\]

权重：

\[
W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}
\]

偏置：

\[
b \in \mathbb{R}^{d_{\text{out}}}
\]

输出：

\[
y \in \mathbb{R}^{d_{\text{out}}}
\]

公式：

\[
y = Wx + b
\]

在代码写法里等价于：

\[
y = xW^T + b
\]

---

## 9.2 单个输出维度

第 \(i\) 个输出为：

\[
y_i = \sum_{j=1}^{d_{\text{in}}} W_{ij}x_j + b_i
\]

含义：

- 第 \(i\) 个输出会看所有输入
- 对每个输入乘一个权重
- 再求和
- 再加偏置

---

## 9.3 批量输入

如果输入是 batch：

\[
X \in \mathbb{R}^{B \times d_{\text{in}}}
\]

则输出为：

\[
Y = XW^T + b
\]

其中：

\[
Y \in \mathbb{R}^{B \times d_{\text{out}}}
\]

---

# 10. 从深度学习原理看，它到底在做什么

线性层的本质是：

> 对输入特征做加权组合，生成新的特征表示

比如输入有 8 个特征：

\[
x = [x_1, x_2, \dots, x_8]
\]

输出 4 个特征：

\[
y = [y_1, y_2, y_3, y_4]
\]

每个输出都是输入的“加权和”：

\[
y_i = w_i^\top x + b_i
\]

也就是说：

- 模型学习“哪些输入更重要”
- 学习“不同输入如何组合”
- 形成新的表达空间

---

# 11. 从几何角度理解

线性层执行的是一个**仿射变换**：

\[
f(x) = Ax + b
\]

它可以理解为：

- 旋转
- 拉伸
- 压缩
- 投影
- 平移

如果没有 \(b\)，就是纯线性变换；
有 \(b\)，就是仿射变换。

---

# 12. 为什么线性层很重要

它是很多深度学习模型的基础组件：

- MLP
- 分类器最后一层
- Transformer 中的投影层
- 特征映射层

它本身负责：

- 特征组合
- 维度变换
- 投影到新的空间

---

# 13. 为什么它还不够

如果神经网络只有线性层，没有激活函数，那么多个线性层叠起来仍然等价于一个线性层。

例如：

\[
x \rightarrow W_1 \rightarrow W_2
\]

本质上仍然是：

\[
x \rightarrow W
\]

所以深度学习需要在线性层后面加非线性激活函数，如 ReLU：

\[
h = \sigma(xW^T + b)
\]

这样模型才能表达复杂的非线性关系。

---

# 14. 一个具体例子

假设：

```python
layer = SimpleLinear(3, 2)
```

表示：

- 输入 3 维
- 输出 2 维

比如输入：

\[
x = [x_1, x_2, x_3]
\]

权重矩阵：

\[
W =
\begin{bmatrix}
w_{11} & w_{12} & w_{13} \\
w_{21} & w_{22} & w_{23}
\end{bmatrix}
\]

偏置：

\[
b = [b_1, b_2]
\]

输出就是：

\[
y_1 = w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + b_1
\]

\[
y_2 = w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + b_2
\]

---

# 15. 代码逐行总结

---

## 初始化函数

```python
def __init__(self, in_features: int, out_features: int):
```

创建线性层，指定输入维度和输出维度。

---

## 创建权重

```python
self.weight = torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
```

生成形状为 `(out_features, in_features)` 的随机权重，并缩放。

---

## 让权重参与训练

```python
self.weight.requires_grad_(True)
```

表示权重是可学习参数。

---

## 创建偏置

```python
self.bias = torch.zeros(out_features, requires_grad=True)
```

创建长度为 `out_features` 的偏置，初始为 0，也可训练。

---

## 前向传播

```python
return x @ self.weight.T + self.bias
```

执行公式：

\[
y = xW^T + b
\]

---

# 16. 最终总结

这段代码手写实现了一个最基础的线性层，其核心是：

\[
y = xW^T + b
\]

其中：

- `in_features`：输入有几个特征
- `out_features`：输出想生成几个特征
- `weight`：控制每个输入对输出的影响
- `bias`：给输出增加平移项
- `requires_grad=True`：表示这些参数能被训练
- `forward`：完成前向计算

从深度学习角度看，它做的事情是：

> 把原始输入特征按可学习的方式重新组合，映射到新的特征空间。
