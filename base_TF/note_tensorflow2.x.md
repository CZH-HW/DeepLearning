 
tensorflow2.x 版本学习笔记

## 1. 基础介绍
- tensorflow 2.x 版本相较于 1.x 版本具有简易性、更清晰、扩展性三大特征，大大简化 API
- tensorflow + keras
- 提高了 TensorFlow Lite 和 TensorFlow.js 部署模型的能力

>新版本让我们可以忘记 1.x 版本的一系列烦人的概念：计算图 Graph、会话 Session、变量管理 Variable Scope、共享 reyse、define-and-run 等等，可以假设我们对 tensorflow 一无所知，同时 tesorflow 2.x 跟 pytorch 更加接近了，都是动态图，而 tensorflow 1.x 是静态图。

#### tesorflow 2.x 三大优势

1. GPU 加速
   NVIDIA 的 GPU 专门针对深度学习的矩阵运算有并行加速的效果，适合大数据的运算
<br>
   
2. 自动求导
   GradientTape 自动求导工具
<br>

1. 神经网络 API 
   tensorflow 提供了一系列的API接口，我们可以直接调用，不用自己去实现每一层，以及每一层的逻辑
   我们可以调用神经网络的 API 来完成复杂的神经网络的搭建 

---

## 2. 基础操作

#### 2.1 tensor 张量

tensorflow 中的数据载体叫做 tensor 即张量，

tensorflow 的张量在概念上等同于多维数组，我们可以使用它来描述数学中的标量（0 维数组）、向量（1 维数组）、矩阵（2 维数组）等各种量，示例如下：


```python
# 定义一个随机数（标量）
random_float = tf.random.uniform(shape=())

# 定义一个有2个元素的零向量
zero_vector = tf.zeros(shape=(2))

# 定义两个2×2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])
```

张量的重要属性是其形状、类型和值。可以通过张量的`shape`、`dtype`属性和`numpy()`方法获得。例如：

```python
# 查看矩阵A的形状、类型和值
print(A.shape)      # 输出(2, 2)，即矩阵的长和宽均为2
print(A.dtype)      # 输出<dtype: 'float32'>
print(A.numpy())    # 输出[[1. 2.]
                    #      [3. 4.]]
```

小技巧：
>TensorFlow 的大多数 API 函数会根据输入的值自动推断张量中元素的类型（一般默认为 tf.float32 ）。
>不过你也可以通过加入 dtype 参数来自行指定类型，例如 zero_vector = tf.zeros(shape=(2), dtype=tf.int32) 将使得张量中的元素类型均为整数。张量的 numpy() 方法是将张量的值转换为一个 NumPy 数组。

#### 2.2 tensor 属性

```python
with tf.device("cpu"):
    a = tf.constant([1])

with tf.device("gpu"):
    b = tf.range(4)

a.device   # '/job:localhost/replica:0/task:0/device:CPU:0' a变量为CPU版本
b.device   # '/job:localhost/replica:0/task:0/device:GPU:0' b变量为GPU版本

# CPU和GPU版本属性的转换
aa = a.gpu()  # 后续版本此方法会被移除

# 把 tensor 转化为 numpy，numpy 肯定是在 CPU 上的
b.numpy()

```


#### 2.3 类型转换

`tf.convert_to_tensor()`和`tf.cast()`

```python
# 一般用于从 numpy、list 数据转化为 tensor 数据，并指定类型如 int32、float32
tf.convert_to_tensor(x, dtype=tf. )   # x 为待转换的数据

# 执行 tensorflow 中张量数据类型转换
# 第一个参数 x：待转换的数据（张量）
# 第二个参数 dtype：目标数据类型
# 第三个参数 name：可选参数，定义操作的名称
tf.cast(x, dtype=tf. , name=None)   # dtype 可以有 int32、float32、double、bool 等
```

#### 2.4 创建 tensor 

不同方式：
- from numpy, list：
  `tf.convert_to_tensor()`
<br>

- zeros, ones：
  `tf.zeros([,])  # 参数为 list 形式的 shape`

  `tf.zeros_like() # 参数为一个 tensor，等价于 tf.zeros(x.shape)`

  `tf.ones([,])  # 参数为 list 形式的 shape`

  `tf.ones_like() # 参数为一个 tensor，等价于 tf.ones(x.shape)`
<br>

- fill：
  `tf.fill([], data) # 第一个参数为 shape 维度，第二个参数为要填充的值`
<br>

- random 随机初始化：
  `tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) # 用于从服从指定正太分布的数值中取出指定个数的值`
    shape: 输出张量的形状，必选
    mean: 正态分布的均值，默认为 0
    stddev: 正态分布的标准差，默认为 1.0
    dtype: 输出的类型，默认为 tf.float32
    seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
    name: 操作的名称

  `tf.random.truncated_normal(shape, mean=0.0, stddev=1.0, dtype, seed=None, name=None) # 裁剪截断，去掉偏离过大的正太分布，也就是随机出来的数据偏离平均值超过 2 个标准差，这个数据需要重新生成`

  `tf.random.uniform(shape, minval=0, maxval=None, dtype, seed=None, name=None) # 从均匀分布中输出随机值，下限 minval包含在范围内，而上限maxval被排除在外`
<br>

- constant 常量：
  `tf.constant(value, dtype=None, shape=None, name='Const')`

<br>

#### 2.5 tensor operation
- `tf.add()` - 相加
- `tf.matmul()` - 矩阵相乘


自动求导机制

在机器学习中，我们经常需要计算函数的导数。TensorFlow 提供了强大的自动求导机制来计算导数。
以下代码展示了如何使用`tf.GradientTape()`计算函数`y(x) = x^2`在`x = 3`时的导数：

```python

import tensorflow as tf

x = tf.Variable(initial_value=3.)

with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)

y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print([y, y_grad])

```

输出

```python
[<tf.Tensor: id=65, shape=(), dtype=float32, numpy=9.0>, <tf.Tensor: id=69, shape=(), dtype=float32, numpy=6.0>]
```

这里`x`是一个初始化为 3 的变量 （Variable），使用 `tf.Variable()`声明。
与普通张量一样，变量同样具有形状、类型和值三种属性。使用变量需要有一个初始化过程，可以通过在 tf.Variable() 中指定`initial_value`参数来指定初始值。
变量与普通张量的一个重要区别是其默认能够被 tensorFlow 的自动求导机制所求导，因此往往被用于定义机器学习模型的参数。


`tf.GradientTape()`是一个自动求导的记录器，在其中的变量和计算步骤都会被自动记录。
在上面的示例中，变量`x`和计算步骤`y = tf.square(x)`被自动记录，因此可以通过`y_grad = tape.gradient(y, x)`求张量`y`对变量`x`的导数。


同样也可用于矩阵、向量的求导
```python
# 矩阵计算求导
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)

with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))

w_grad, b_grad = tape.gradient(L, [w, b])        # 计算L(w, b)关于w, b的偏导数
print([L.numpy(), w_grad.numpy(), b_grad.numpy()])
```

输出
```python
[62.5, array([[35.], [50.]], dtype=float32), 15.0]
```

`tf.square()`操作代表对输入张量的每一个元素求平方，不改变张量形状。 
`tf.reduce_sum()`操作代表对输入张量的所有元素求和，输出一个形状为空的纯量张量（可以通过`axis`参数来指定求和的维度，不指定则默认对所有元素求和）。
TensorFlow 中有大量的张量操作 API，包括数学运算、张量形状操作（如 tf.reshape()）、切片和连接（如 tf.concat()）等多种类型，可以通过查阅 TensorFlow 的官方 API 文档来进一步了解。





#### 



#### 2.7 不同维度 tensor 数据的典型应用

scalar 标量的典型应用：loss, accuracy
vector 向量的典型应用：bias
matrix 矩阵的典型应用
dimension=3 的 tensor 应用：自燃语言处理
dimension=4 的 tensor 应用：图片
dimension=5 的 tensor 应用：meta-learning



















