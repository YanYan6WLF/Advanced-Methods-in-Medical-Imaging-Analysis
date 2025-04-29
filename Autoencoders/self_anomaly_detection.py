#1. 导入库

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
#2. 导入数据集

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#3. 标准化

x_train = x_train / 255.0
x_test = x_test / 255.0
#4. 展平

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#5. 定义架构

input_dim = x_train.shape[1]
encoding_dim = 32

input_layer = keras.layers.Input(shape=(input_dim,))
encoder = keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoder = keras.layers.Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = keras.models.Model(inputs=input_layer, outputs=decoder)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#6. 培训

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, 
validation_data=(x_test, x_test))

# Use the trained autoencoder to reconstruct new data points
decoded_imgs = autoencoder. predict(x_test)
#7. 计算原始数据点和重建数据点之间的均方误差 (MSE)

mse = np.mean(np.power(x_test - decoded_imgs, 2), axis=1)
#8. 绘制重建误差分布

plt.hist(mse, bins=50)
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.show()

# Set a threshold for anomaly detection
threshold = np.max(mse)

# Find the indices of the anomalous data points
anomalies = np.where(mse > threshold)[0]

# Plot the anomalous data points
n = min(len(anomalies), 10)
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[anomalies[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[anomalies[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()