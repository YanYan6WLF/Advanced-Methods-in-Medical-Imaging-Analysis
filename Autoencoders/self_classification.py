#1. 导入库

from keras.layers import Input, Dense
from keras.models import Model
#2. 导入数据集

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#3. 标准化

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#4. 扁平化

input_dim = 784
x_train = x_train.reshape(-1, input_dim)
x_test = x_test.reshape(-1, input_dim)
#5. 自动编码器架构

encoding_dim = 32
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)

# Compile autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#6. 培训

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
#7. 提取 MNIST 图像的压缩表示

encoder = Model(input_img, encoded) #建立一个名为encoder的模型
x_train_encoded = encoder.predict(x_train)
x_test_encoded = encoder.predict(x_test)
#8.前馈分类器

clf_input_dim = encoding_dim
clf_output_dim = 10
clf_input = Input(shape=(clf_input_dim,))
clf_output = Dense(clf_output_dim, activation='softmax')(clf_input)
classifier = Model(clf_input, clf_output)

# Compile classifier
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#9. 训练分类器
from keras.utils import to_categorical
y_train_categorical = to_categorical(y_train, num_classes=clf_output_dim)
y_test_categorical = to_categorical(y_test, num_classes=clf_output_dim)
classifier.fit(x_train_encoded, y_train_categorical,
               epochs=50,
               batch_size=256,
               shuffle=True,
               validation_data=(x_test_encoded, y_test_categorical))