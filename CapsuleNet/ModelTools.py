# coding=utf-8

from keras.models import Sequential, load_model
from keras.layers import InputLayer
from CapsuleKeras import Capsule, Length

custom_ob={'Capsule': Capsule, 'Length': Length}

def model_divide(model, level_number):
    model_first = Sequential()
    model_second = Sequential()

    model_first.add(InputLayer(input_shape=model.layers[0].input_shape[1:]))
    for current_layer in range(0, level_number + 1):
        model_first.add(model.layers[current_layer])

    model_second.add(InputLayer(input_shape=model.layers[level_number + 1].input_shape[1:]))
    for current_layer in range(level_number + 1, len(model.layers)):
        model_second.add(model.layers[current_layer])

    return model_first, model_second


def model_evaluate(model, test_img, test_label):
    score = model.evaluate(test_img, test_label, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def model_train(model, train_img, train_label, model_path, epochs, batch_size):
    model.fit(train_img, train_label, epochs=epochs, batch_size=batch_size)
    model.save(model_path)

    return model


def model_update(origin_model_path, layer_number):
    ## 加载训练好的模型，对其进行更新
    model = load_model(origin_model_path, custom_objects=custom_ob)

    # 查看并设置每层是否可训练
    print("------------------查看模型更新前情况----------------------")
    for layer in model.layers[:layer_number]:
        print(layer.name, ' is trainable? ', layer.trainable)
        layer.trainable = False

    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 检验并查看是否设置成功
    print("------------------查看模型更新后情况----------------------")
    for layer in model.layers:
        print(layer.name, ' is trainable? ', layer.trainable)

    return model


def model_load(path):
    model = load_model(path, custom_objects=custom_ob)
    return model
