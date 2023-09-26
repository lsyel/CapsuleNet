import CapsNetModel
from keras.datasets import mnist
from keras import utils

if __name__ == '__main__':
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

    model = CapsNetModel.CapsNet(input_shape=(28, 28, 1))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=5, verbose=1)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])