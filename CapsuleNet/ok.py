import DataTools as dt
import ModelTools as mt
from pathlib import Path
import CapsNetModel

def check_file(path_name):
    model1_file = Path(path_name)
    if model1_file.exists():
        return True

    return False


if __name__ == '__main__':
    data_path = '../mnist'
    origin_model_path = '../weight/demo1.h5'
    new_model_path = '../weight/demo2.h5'

    # 准备训练数据
    batch_size = 256
    num_classes = 10
    img_rows, img_cols = 28, 28

    first_train, first_label, second_train, second_label = dt.gene_train_data(7, data_path, 12200)
    test_img, test_label = dt.gene_test_data(data_path)

    print('first_train shape:', first_train.shape)
    print('first_label shape:', first_label.shape)
    print(first_train.shape[0], 'train samples')

    print('second_train shape:', second_train.shape)
    print('second_label shape:', second_label.shape)
    print(second_train.shape[0], 'train samples')

    cap_model = CapsNetModel.CapsNet(input_shape=(28, 28, 1))

    if check_file(origin_model_path):
        cap_model = mt.model_load(origin_model_path)
    else:
        cap_model = mt.model_train(cap_model, first_train, first_label, origin_model_path, 5, batch_size)
    mt.model_evaluate(cap_model, test_img, test_label)

    # 第二次模型训练
    print('正确性验证实验')
    second_model = mt.model_update(origin_model_path, 7)
    second_model = mt.model_train(second_model, second_train, second_label, new_model_path, 5, batch_size)
    mt.model_evaluate(second_model, test_img, test_label)
    print('正常训练')
    second_model = mt.model_train(second_model, second_train, second_label, new_model_path, 5, batch_size)
    mt.model_evaluate(second_model, test_img, test_label)

    print('最佳卸载层探究实验')
    for layer_number in range(len(cap_model.layers)-1, 0, -1):
        print('当前更新层：', layer_number)
        second_model = mt.model_update(origin_model_path, layer_number)
        second_model = mt.model_train(second_model, second_train, second_label, new_model_path, 5, batch_size)
        mt.model_evaluate(second_model, test_img, test_label)
