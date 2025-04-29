import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 导入数据集
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# 标准化
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 重塑为图像
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# encoder
encoder_inputs = keras.Input(shape=(28, 28, 1)) 
x = layers.Conv2D(16, 3, activation="relu", padding="same")(encoder_inputs) # 二维卷积层，16个过滤器，3表示每个过滤器的大小是3*3，activation指定了激活函数，引入非线性，帮助网络学习更加复杂的模式， padding 在卷积操作时应该在图像边缘进行填充，以确保输出的图像的尺寸和输入图像相同
#每个过滤器都能检测输入数据中的不同特征。例如，一个过滤器可能专门用来检测垂直边缘，而另一个可能检测水平边缘。增加过滤器的数量可以帮助网络学习图像中更复杂的模式。
# 指定一个卷积层有多少个过滤器时，您就定义了它输出特征图的通道数
# 在卷积神经网络（CNN）中，每个卷积层的过滤器（也称为卷积核）会在输入特征图的每个通道上进行卷积操作。具体步骤如下：
#卷积操作：过滤器在每个输入通道上滑动，并计算过滤器和输入通道上相应区域之间的点积。这样，对于每个通道，过滤器都会产生一个二维激活图（也称为特征图）。
#累加：接着，来自所有输入通道的这些二维特征图会逐像素相加，产生一个单一的二维特征图。如果有偏置项，它会被加到这个累加的特征图的每一个值上。
#激活函数：然后通常会对累加后的特征图应用激活函数（如ReLU），这样每个过滤器就会输出一个激活后的二维特征图。
#在一层卷积中，有多少个过滤器，就会有多少个这样的输出特征图，每个过滤器都能够捕捉输入数据的不同特征。这些输出特征图会堆叠在一起，形成下一层的多通道输入。
#例如，如果一个卷积层的输入有10个通道，这个层有16个过滤器，那么每个过滤器都会对这10个通道各产生一个特征图，然后将这些特征图累加（每个过滤器累加10个特征图），最终得到16个激活后的特征图，因为有16个过滤器。这些特征图在下一层中会作为16个输入通道。
x = layers.MaxPooling2D(2, padding="same")(x)
# 池化层： 降低图像的空间尺寸，同时保留重要的特征信息，可以减少计算量，可以减少过拟合（使得网络难以对特定样本做出精确匹配 ），增强特征检测（比如边缘或纹理)，在池化操作时，通常会使用一个小的窗口（2*2），并在窗口中选取最大值（对于最大池化）来代表整个区域，从而实现降维的目的
x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)
x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
encoder_outputs = layers.MaxPooling2D(2, padding="same")(x)
# 卷积层提取高级特征，池化层减少数据的维度
encoder = keras.Model(encoder_inputs, encoder_outputs, name="encoder")
# 创建了编码器模型
encoder.summary()
# 调用summary方法，可以打印出模型的架构，包括每一层的名称，输出的形状和参数数量

#decoder
decoder_inputs = keras.Input(shape=(4, 4, 8)) # latent space size; tensor
x = layers.Conv2D(8, 3, activation="relu", padding="same")(decoder_inputs)
x = layers.UpSampling2D(2)(x) # 增加数据的空间维度，2意味着每个维度的大小都会增加一倍，这种上采样通常会通过简单的近邻插值来实现
x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
x = layers.UpSampling2D(2)(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
x = layers.UpSampling2D(2)(x)
decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
#使用了sigmoid激活函数，通常用于二值图像或概率图的生成，使得输出值被限制在0和1之间。
decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
decoder.summary()

# 序列模型
autoencoder = keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# training
autoencoder.fit(x_train, x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test))
#在自编码器中，目标是重构输入数据，因此训练数据同时也是目标数据。

# 对测试testing图像进行编码和解码
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

# 展示原始图像和处理后decode图像
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original image
    ax = plt.subplot(2, n, i + 1)
    # 在图形窗口中创建一个子图。plt.subplot的参数定义了网格的布局，
    #这里是2行n列。i + 1定义了当前子图在网格中的位置。因为这是在第一行展示原始图像，所以位置是i + 1。
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstructed image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    # 这行代码在第二行创建一个子图来展示重建的图像。位置通过i + 1 + n确定，这确保了重建的图像正好在原始图像的下方。
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()