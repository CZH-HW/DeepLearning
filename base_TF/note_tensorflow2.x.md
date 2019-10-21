 
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

### 2.1 tensor 张量

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

### 2.2 tensor 属性

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


### 2.3 类型转换

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

### 2.4 创建 tensor 

不同方式：
- from numpy, list：
```python
tf.convert_to_tensor()
```

- zeros, ones：
```python
tf.zeros([,])  # 参数为 list 形式的 shape`

tf.zeros_like() # 参数为一个 tensor，等价于 tf.zeros(x.shape)`

tf.ones([,])  # 参数为 list 形式的 shape`

tf.ones_like() # 参数为一个 tensor，等价于 tf.ones(x.shape)`
```

- fill：
```python
tf.fill([], data) # 第一个参数为 shape 维度，第二个参数为要填充的值
```

- random 随机初始化：
```python
tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) # 用于从服从指定正太分布的数值中取出指定个数的值

    shape: 输出张量的形状，必选
    mean: 正态分布的均值，默认为 0
    stddev: 正态分布的标准差，默认为 1.0
    dtype: 输出的类型，默认为 tf.float32
    seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
    name: 操作的名称
```

```python
tf.random.truncated_normal(shape, mean=0.0, stddev=1.0, dtype, seed=None, name=None) # 裁剪截断，去掉偏离过大的正太分布，也就是随机出来的数据偏离平均值超过 2 个标准差，这个数据需要重新生成`
```

```python
tf.random.uniform(shape, minval=0, maxval=None, dtype, seed=None, name=None) # 从均匀分布中输出随机值，下限 minval包含在范围内，而上限maxval被排除在外
```

- constant 常量：
```
tf.constant(value, dtype=None, shape=None, name='Const')
```

### 2.5 索引和切片

对一个 tensor 的部分数据进行一个读取，也就是 tensor 的索引和切片。

#### 2.5.1 基本索引方式

```python
In [1]:a = tf.random.normal([1,5,5,3]) 
Out[1]: 
<tf.Tensor: id=2388, shape=(1, 5, 5, 3), dtype=float32, numpy=
array([[[[ 1.4477472 ,  1.0481421 ,  1.587761  ],
         [ 0.07396559,  0.26200747, -0.5246982 ],
         [-0.14084913,  1.0599062 ,  3.4249015 ],
         [ 1.0187757 , -0.37970063,  0.25566164],
         [ 0.53958094, -0.1688342 , -0.72403556]],

        [[-1.1711743 , -0.4623126 , -0.19454424],
         [ 0.07401901, -1.3446945 , -1.3799477 ],
         [ 0.09041796,  1.1915997 ,  0.89786416],
         [ 0.30240753,  1.9807535 , -0.95751333],
         [-0.470163  ,  0.52373177, -2.380843  ]],

        [[-1.8001443 ,  0.6973308 ,  0.91308683],
         [-2.0935063 , -0.21714813,  0.01619152],
         [ 0.16683224,  1.1770695 ,  0.29157427],
         [ 1.2871331 ,  0.62891936, -0.34320897],
         [-1.1878803 ,  1.5598491 ,  2.1955924 ]],

        [[ 0.7078166 ,  0.9292463 ,  0.15103252],
         [-1.0771753 ,  0.03676553, -1.0725024 ],
         [-0.07678125, -0.9442275 ,  0.6156368 ],
         [ 0.36243838,  0.15164559, -0.3267987 ],
         [ 1.0649025 ,  1.4520211 ,  2.1470108 ]],

        [[ 0.46937785,  0.56005764, -1.4017457 ],
         [ 0.13682605,  0.24005781,  0.5612903 ],
         [ 0.00468145, -0.05508251,  0.36415946],
         [-1.0916646 , -2.0555508 , -1.066773  ],
         [ 0.00943979, -0.5864858 , -0.06349369]]]], dtype=float32)>

In [2]: a[0][0]                  
Out[2]: 
<tf.Tensor: id=2411, shape=(5, 3), dtype=float32, numpy=
array([[ 1.4477472 ,  1.0481421 ,  1.587761  ],
       [ 0.07396559,  0.26200747, -0.5246982 ],
       [-0.14084913,  1.0599062 ,  3.4249015 ],
       [ 1.0187757 , -0.37970063,  0.25566164],
       [ 0.53958094, -0.1688342 , -0.72403556]], dtype=float32)>

In [3]: a[0][0][0]   
Out[3]: <tf.Tensor: id=2423, shape=(3,), dtype=float32, numpy=array([1.4477472, 1.0481421, 1.587761 ], dtype=float32)>

In [4]: a[0][0][0][2]
Out[4]: <tf.Tensor: id=2439, shape=(), dtype=float32, numpy=1.587761>
```

>注意：这种索引方式比较通用，大家都能接受，但这种索引方式比较单一，需要写多个中括号，看起来程序可读性比较差。只能取一个具体的元素，不支持那种隔断取，倒着取多样取得方式。这样对数据读取存在限制。numpy对这种方式进行了一个很好的拓展。下面展示！

#### 2.5.2 numpy 的索引方式

```python
In [1]:a = tf.random.normal([4,28,28,3])  # 这里可以看作 4 张 28*28 的 3 通道的图片

In [2]: a[1].shape
Out[2]: TensorShape([28, 28, 3])

In [3]: a[1,2].shape      # 第 1 张图片第二行像素点的 3 通道的 RGB 数据
Out[3]: TensorShape([28, 3])

In [4]: a[1,2,3]    # 第 1 张图片第 2 行第 3 列像素点的 3 通道的 RGB 数据
Out[4]: <tf.Tensor: id=2463, shape=(3,), dtype=float32, numpy=array([-0.61048096,  0.02324595, -1.2047269 ], dtype=float32)>

In [5]: a[1,2,3].shape
Out[5]: TensorShape([3])

In [6]: a[1,2,3,2]
Out[6]: <tf.Tensor: id=2471, shape=(), dtype=float32, numpy=-1.2047269>
```

>注意：使用 numpy 方式（使用逗号分隔），程序的可读性强，程序中也可以少很多中括号。

#### 2.5.3 切片（单冒号），start:end

```python
In [1]: b = tf.range(10)
Out[1]: <tf.Tensor: id=2475, shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)>

In [2]: b[2:5]
Out[2]: <tf.Tensor: id=2479, shape=(3,), dtype=int32, numpy=array([2, 3, 4], dtype=int32)>
# 切片为冒号，A:B
# A 为切片开始，B 为切片末尾
# A 和 B 的默认位置为 0 和 -1，A 和 B 不指定时采用默认位置

In [3]: a[0].shape  # 取第 1 张图片
Out[3]: TensorShape([28, 28, 3])

In [4]: a[0,:,:,:].shape    # 和上面等价，取第 1 张图片
Out[4]: TensorShape([28, 28, 3])

In [44]: a[:,:,:,0].shape   # 取出单通道图片
Out[44]: TensorShape([4, 28, 28])
```

>注意：切片希望读取维度的一部分，比如有 4 张图片，希望读取前 2 张图片。切片返回的总是一个标量。比如：这里对于传统的 a[-1] 这样返回的是一个 9；对于切片 a[-1:] 这样返回的是一个向量 [9]。

#### 2.5.4 切片（双冒号），start:end:step

```python
In [1]: a[:,:,:,0].shape
Out[1]: TensorShape([4, 28, 28])

In [2]: a.shape
Out[2]: TensorShape([4, 28, 28, 3])

In [3]: a[0:2,:,:,:].shape  # 取两张图片
Out[3]: TensorShape([2, 28, 28, 3])

In [4]: a[:,0:28:2,0:28:2,:].shape  # 4 张，行和列间隔采样
Out[4]: TensorShape([4, 14, 14, 3])

In [5]: a[:,::2,::2,:].shape    # 同上，行和列间隔采样
Out[5]: TensorShape([4, 14, 14, 3])

In [6]: b = tf.range(10)
Out[6]: <tf.Tensor: id=2475, shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)>

In [49]: b[::-1]   # 倒序
Out[49]: <tf.Tensor: id=2519, shape=(10,), dtype=int32, numpy=array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=int32)>
```

#### 2.5.5 切片（省略号），...

```python
# 当维度很长的时候，如果采用冒号的方式，会很繁琐和不优雅，可以用省略号表示
In [1]: a[...,0].shape
Out[1]: TensorShape([4, 28, 28])
```

#### 2.5.6 常用的方法 Selective Indexing

任意采样
- `tf.gather` 一个维度
```python
tf.gather(
    params,
    indices,
    validate_indices=None,
    axis=None,
    batch_dims=0,
    name=None
)
```
```python
In [1]: a.shape   
Out[1]: TensorShape([4, 35, 8])  # a 中的数据为[班级，学生，课程]

# a 给出数据源，axis 给出班级、学生、课程中的一个维度，indices 给出某一个维度的具体索引
In [53]: tf.gather(a, axis=0, indices=[2,3]).shape                             
Out[53]: TensorShape([2, 35, 8])

In [54]: tf.gather(a, axis=0, indices=[2,1,3,0]).shape  # 班级维度，班级顺序改变规则
Out[54]: TensorShape([4, 35, 8])

In [55]: tf.gather(a, axis=1, indices=[2,3,7,9,16]).shape  # 学生维度，抽样检查，每个班级只采样 5 个学生
Out[55]: TensorShape([4, 5, 8])

In [56]: tf.gather(a, axis=2, indices=[2,3,7]).shape  # 课程数量维度， 
Out[56]: TensorShape([4, 35, 3])
```
>注意：以上可以实现想怎么采样就怎么采样，没必要按着规则来了。

- `tf.gather_nd` 多个维度
```python
tf.gather_nd(
    params,
    indices,
    batch_dims=0,
    name=None
)
```
```python
In [1]: a.shape   
Out[1]: TensorShape([4, 35, 8])  # a 中的数据为[班级，学生，课程]

In [59]: tf.gather_nd(a,[0]).shape
Out[59]: TensorShape([35, 8])

In [61]: tf.gather_nd(a,[[0,0],[1,1]]).shape
Out[61]: TensorShape([2, 8])

In [62]: tf.gather_nd(a,[[0,0],[1,1],[2,2]]).shape                      
Out[62]: TensorShape([3, 8])

In [63]: tf.gather_nd(a,[[0,0,0],[1,1,1],[2,2,2]]).shape
Out[63]: TensorShape([3])

In [64]: tf.gather_nd(a,[[[0,0,0],[1,1,1],[2,2,2]]]).shape
Out[64]: TensorShape([1, 3])
```


- `tf.boolean_mask`
























### 2.6 tensor operation
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


使用常规的科学计算库实现机器学习模型有两个痛点：

1. 经常需要手工求函数关于参数的偏导数。如果是简单的函数或许还好，但一旦函数的形式变得复杂（尤其是深度学习模型），手工求导的过程将变得非常痛苦，甚至不可行。

2. 经常需要手工根据求导的结果更新参数。这里使用了最基础的梯度下降方法，因此参数的更新还较为容易。但如果使用更加复杂的参数更新方法（例如 Adam 或者 Adagrad），这个更新过程的编写同样会非常繁杂。
<br>


tensorflow 下的线性回归

TensorFlow 的 Eager Execution（动态图）模式与 NumPy 的运行方式十分类似，然而提供了更快速的运算（GPU 支持）、自动求导、优化器等一系列对深度学习非常重要的功能。以下展示了如何使用 TensorFlow 计算线性回归。可以注意到，程序的结构和 NumPy 的实现非常类似。这里，TensorFlow 帮助我们做了两件重要的工作：

- 使用`tape.gradient(ys, xs)`自动计算梯度；
- 使用`optimizer.apply_gradients(grads_and_vars)`自动更新模型参数。

```python
# 生成数据
X = tf.constant(X)
y = tf.constant(y)

# 模型参数初始化
a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000   # 迭代次数
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)  # SGD：随机梯度下降算法

for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))  
    # 优化器可以帮助我们根据计算出的求导结果更新模型参数，从而最小化某个特定的损失函数
    # 具体使用方式是调用其 apply_gradients() 方法。

print(a, b)
```

注意到这里，更新模型参数的方法`optimizer.apply_gradients()`需要提供参数`grads_and_vars`，即待更新的变量（如上述代码中的`variables`）及损失函数关于这些变量的偏导数（如上述代码中的`grads`）。

具体而言，这里需要传入一个 Python 列表（List），列表中的每个元素是一个 （变量的偏导数，变量） 对。比如这里是`[(grad_a, a), (grad_b, b)]`。我们通过`grads = tape.gradient(loss, variables)`求出 tape 中记录的 loss 关于 variables = [a, b] 中每个变量的偏导数，也就是 grads = [grad_a, grad_b]，再使用 Python 的 zip() 函数将 grads = [grad_a, grad_b] 和 variables = [a, b] 拼装在一起，就可以组合出所需的参数了。

`zip()`函数是 Python 的内置函数。用自然语言描述这个函数的功能很绕口，但如果举个例子就很容易理解了：如果`a = [1, 3, 5]， b = [2, 4, 6]`，那么`zip(a, b) = [(1, 2), (3, 4), ..., (5, 6)]`。即 “将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表”。在 Python 3 中， zip() 函数返回的是一个 zip 对象，本质上是一个生成器，需要调用 list() 来将生成器转换成列表。

在实际应用中，我们编写的模型往往比这里一行就能写完的线性模型`y_pred = a * X + b`（模型参数为`variables = [a, b]`）要复杂得多。所以，我们往往会编写并实例化一个模型类`model = Model()`，然后使用`y_pred = model(X)`调用模型，使用`model.variables`获取模型参数。


#### 2.8 不同维度 tensor 数据的典型应用

scalar 标量的典型应用：loss, accuracy
vector 向量的典型应用：bias
matrix 矩阵的典型应用
dimension=3 的 tensor 应用：自燃语言处理
dimension=4 的 tensor 应用：图片
dimension=5 的 tensor 应用：meta-learning


## 3 tensorflow 模型建立与训练

动态模型
- 模型的构建：`tf.keras.Model` 和 `tf.keras.layers`

- 模型的损失函数：`tf.keras.losses`

- 模型的优化器：`tf.keras.optimizer`

- 模型的评估：`tf.keras.metrics`


### 3.1 模型（model）与层（layer）

在 TensorFlow 中，推荐使用 Keras（tf.keras）构建模型。Keras 是一个广为流行的高级神经网络 API，简单、快速而不失灵活性，现已得到 TensorFlow 的官方内置和全面支持。

Keras 有两个重要的概念： 模型（Model） 和 层（Layer） 。层将各种计算流程和变量进行了封装（例如基本的全连接层，CNN 的卷积层、池化层等），而模型则将各种层进行组织和连接，并封装成一个整体，描述了如何将输入数据通过各种层以及运算而得到输出。在需要模型调用的时候，使用`y_pred = model(X)`的形式即可。Keras 在`tf.keras.layers`下内置了深度学习中大量常用的的预定义层，同时也允许我们自定义层。

Keras 模型以类的形式呈现，我们可以通过继承`tf.keras.Model`这个 Python 类来定义自己的模型。在继承类中，我们需要重写`__init__()`（构造函数，初始化）和`call(input)`（模型调用）两个方法，同时也可以根据需要增加自定义的方法。

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
        # layer1 = tf.keras.layers.BuiltInLayer(...)
        # layer2 = MyCustomLayer(...)

    def call(self, input):
        # 此处添加模型调用的代码（处理输入并返回输出），例如
        # x = layer1(input)
        # output = layer2(x)
        return output

    # 还可以添加自定义的方法
```

继承 tf.keras.Model 后，我们同时可以使用父类的若干方法和属性，例如在实例化类 model = Model() 后，可以通过 model.variables 这一属性直接获得模型中的所有变量，免去我们一个个显式指定变量的麻烦。
<br>


简单的线性模型`y_pred = a * X + b`，我们可以通过模型类的方式编写如下：

```python
import tensorflow as tf

X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output


# 以下代码结构与前节类似
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)      # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)
```
























