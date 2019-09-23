
[TensorFlow学习笔记]目录
[TOC]

# TensorFlow

## 1. 基础
### 1.1 数据流图

TensorFlow 使用数据流图来规划计算流程，数据流图主要由节点与边（张量）构成

![](https://github.com/CZH-HW/CloudImg/raw/master/NoteDL/tensorflow_数据流图_1.png)

- 每个节点是 Operation 运算操作，即一些数学的操作、激励函数的操作等，作为 Operation 的输入和输出都是 Tensor（张量）
- 节点与节点之间的连接称为边，边是由流动的 Tensor 组成
  有一类特殊的边没有数据流动，这种边是依赖控制（control dependencies）
- 对于数据流图的运行，需要创建 Session 这样一个会话来运行，Session 可以在不同的设备上运行，例如 GPU、CPU 等


session 会话

tensorflow 的内核使用更加高效的 C++ 作为后台，以支撑它的密集计算。tensorflow 把前台(即 python 程序)与后台程序之间的连接称为"会话（Session）"

![](https://github.com/CZH-HW/CloudImg/raw/master/NoteDL/tensorflow_数据流图_4.png)

`Session`作为会话，主要功能是指定操作对象的执行环境，`Session`类构造函数有3个可选参数。
- `target`(可选)：指定连接的执行引擎，多用于分布式场景。
- `graph`(可选)：指定要在`Session`对象中参与计算的图（graph）。
- `config`(可选)：辅助配置`Session`对象所需的参数（限制 CPU 或 GPU 使用数目，设置优化参数以及设置日志选项等）。

`Session.run()`的作用运行数据流图中某一部分或者整个计算图，让其动起来，执行相应的功能

在编译环境中，输入要执行的 Tensorflow 代码，当创建会话执行相应的功能时，首先客户端会创建一个图，接着使用会话的run操作，会将需要执行的操作传入服务端，服务端使用C++语言完成相应的操作， 得出结果后，将结果传回客户端，然后根据代码命令最后输出返回的结果。

总结：
![](https://github.com/CZH-HW/CloudImg/raw/master/NoteDL/tensorflow_数据流图_3.png)


## 2. 
