'''
训练了一个神经网络模型，来对服装图像进行分类，例如运动鞋和衬衫
Fashion MNIST 数据集，其中包含了10个类别中共70,000张灰度图像。图像包含了低分辨率（28 x 28像素）的单个服装物品
使用60,000张图像(训练集)来训练网络和10,000张图像(测试集)来评估网络模型学习图像分类任务的准确程度
'''

# 兼容版本，必须使用新写法
from __future__ import absolute_import, division, print_function, unicode_literals

# 导入TensorFlow和tf.keras和辅助库
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# 加载数据
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 数据归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

'''
# 显示训练集中的前25个图像，并在每个图像下方显示类名。
# 验证数据格式是否正确，我们是否已准备好构建和训练网络。
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

# 构建模型
model = keras.Sequential([
    # 转化为1维数组
    keras.layers.Flatten(input_shape=(28, 28)),  
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 编译模型
# 损失函数 — 这可以衡量模型在培训过程中的准确程度。 我们希望将此函数最小化以"驱使"模型朝正确的方向拟合。
# 优化器 — 这就是模型根据它看到的数据及其损失函数进行更新的方式。
# 评价方式 — 用于监控训练和测试步骤。以下示例使用准确率(accuracy)，即正确分类的图像的分数
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 训练模型
# 1.将训练数据提供给模型 - 在本案例中，他们是train_images和train_labels数组。
# 2.模型学习如何将图像与其标签关联
# 3.我们使用模型对测试集进行预测, 在本案例中为test_images数组。我们验证预测结果是否匹配test_labels数组中保存的标签。
# 通过调用model.fit方法来训练模型 — 模型对训练数据进行"拟合"。
model.fit(train_images, train_labels, epochs=5)

# 评估准确率（测试集）
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# 进行预测
predictions = model.predict(test_images)
# 用图表来查看全部10个类别
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([]) 
    plt.imshow(img, cmap=plt.cm.binary)
  
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
  
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
  
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color( 'blue')


# 绘制前X个测试图像，预测标签和真实标签
# 以蓝色显示正确的预测，红色显示不正确的预测
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()


















