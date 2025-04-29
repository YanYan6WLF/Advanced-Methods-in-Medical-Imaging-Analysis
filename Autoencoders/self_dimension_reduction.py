import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train_flat = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test_flat = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# np.prod(x_train.shape[1:])计算了每个图像展平后的大小。

# 架构
#import c
input_dim = 784
encoding_dim = 32

input_layer = keras.Input(shape=(input_dim,))
encoder = keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
#keras.layers.Dense 创建了一个全连接层或称为密集层。这一层具有encoding_dim数量的神经元，并使用ReLU（Rectified Linear Unit）激活函数。这一层将输入数据转换为潜在空间的编码，编码器的任务是从原始输入数据中提取最有意义的特征。
decoder = keras.layers.Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = keras.models.Model(inputs=input_layer, outputs=decoder)

# Compile autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# trianing
history = autoencoder.fit(x_train_flat, x_train_flat,
                          epochs=50,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test_flat, x_test_flat))

# 使用编码器将输入数据编码为低维表示

encoder_model = keras.models.Model(inputs=input_layer, outputs=encoder)
encoded_data = encoder_model.predict(x_test_flat)

#使用前两个主成分绘制二维编码数据
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
encoded_pca = pca.fit_transform(encoded_data)
# PCA（主成分分析）是一种统计方法，用于将数据转换到一个新的坐标系统，使得从第一坐标（第一主成分）开始，每一个坐标上的方差最大化。在这里，它被用来进一步将32维的编码表示降维到2维，以便进行可视化。

plt.scatter(encoded_pca[:, 0], encoded_pca[:, 1], c=y_test)
#c=y_test参数表示点的颜色映射到测试集的标签，这样可以看到不同数字的编码表示在二维空间中的分布
plt.colorbar()
plt.show()