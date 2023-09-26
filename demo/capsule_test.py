from demo.Capsule_Keras import *
from keras import utils, layers, models
from keras.datasets import mnist
from keras.models import Model
from keras.layers import *
from tensorflow import keras
import numpy as np
from demo.PrimaryCaps import Length,Mask
from demo import PrimaryCaps

# 准备训练数据
batch_size = 256
num_classes = 10
img_rows, img_cols = 28, 28
# 加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# 换one hot格式
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# 搭建CNN+Capsule分类模型
## 一个常规的 Conv2D 模型
input_image = Input(shape=(None, None, 1))
cnn = Conv2D(64, (3, 3), activation='relu')(input_image)
cnn = AveragePooling2D((2, 2))(cnn)
cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
cnn = Reshape((-1, 128))(cnn)
capsule = Capsule(10, 16, 3, True)(cnn)
fc = Flatten(name='flatten')(capsule)
fc = Dense(256, activation=keras.activations.relu, use_bias=True)(fc)
fc = Dropout(0.5)(fc)

output = Dense(10, activation=keras.activations.softmax, use_bias=True)(fc)

model = Model(inputs=input_image, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=batch_size, epochs=5, verbose=1)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


def CapsNet(input_shape):
    x = Input(shape=input_shape)

    conv1 = Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    primaryCpas = PrimaryCaps.PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    digitCaps = Capsule(num_capsule=10,dim_capsule=16,routings=3,name='digitcpas')(primaryCpas)

    out_caps = Length(name='capsnet')(digitCaps)

    y = Input(shape=(10,))

    masked_by_y = Mask()([digitCaps, y])
    masked = Mask()(digitCaps)

    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * 10))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))