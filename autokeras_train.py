import numpy as np
import autokeras
import autokeras as ak
from keras.datasets import mnist

from keras.utils import plot_model
from keras.models import load_model
from keras.utils import to_categorical

from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import glob
import os
import time
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
from PIL import ImageFile
import random
import cv2

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_img_to_array(dir_name, img_size, type_name, class_names):
    X_Y = []
    for class_name in os.listdir(dir_name + type_name):
        for img_path in glob.glob(dir_name + type_name + '/' + class_name + '/*'):
            img = img_to_array(load_img(img_path, target_size=(img_size, img_size))) / 255.0
            # img = np.array(Image.open(img_path).convert('L').resize((img_size, img_size)))/255.0
            if class_name == class_names[0]:
                X_Y.append([img, 0])
            elif class_name == class_names[1]:
                X_Y.append([img, 1])
            else:
                print("class_name error!")
                print(class_name)
    return X_Y


def array_to_data(dir_name, img_size, type_name, class_names):
    X_Y = load_img_to_array(dir_name, img_size, type_name, class_names)
    random.shuffle(X_Y)
    # X = np.asarray(X)
    # Y = np.asarray(Y)
    # X = X.astype('float32') / 255.0
    # Y = np_utils.to_categorical(Y, 2)
    X = []  # 画像データ
    Y = []  # ラベル情報
    # データセット作成
    for x, y in X_Y:
        X.append(x)
        Y.append(y)
    # numpy配列に変換
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def confirm_data(X, Y, class_names):
    for i in range(0, 4):
        plt.subplot(2, 2, i + 1)
        plt.axis('off')
        plt.title(label=class_names[0] if Y[i] == 0 else class_names[1])
        plt.imshow(X[i])
    plt.show()


with open("../auto_keras/config.json") as f:
    data = json.load(f)

img_size = data["learning"]["img_size"]
dir_name = data["generator"]["dir"]
class_names = data["class_names"]
loss = data["generator"]["loss"]
max_trials = data["generator"]["max_trials"]
X_train, Y_train = array_to_data(dir_name, img_size, 'train', class_names)
X_valid, Y_valid = array_to_data(dir_name, img_size, 'valid', class_names)
X_test, Y_test = array_to_data(dir_name, img_size, 'test', class_names)
# print(X_test)
# print(Y_test)
# print(X_train.shape)
# print(Y_train.shape)

# confirm_data(X_train, Y_train, class_names)

# (X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()
# X_train = X_train.reshape(X_train.shape + (1,))
# X_valid = X_valid.reshape(X_valid.shape + (1,))
# print(X_train.shape)
# print(Y_train.shape)

clf = ak.ImageClassifier(metrics='accuracy',
                         loss=loss,
                         max_trials=max_trials)

es_cb_data = data["early_stopping"]
reduce_lr_data = data["reduce_lr"]
es_cb = EarlyStopping(monitor=es_cb_data["monitor"],
                      min_delta=es_cb_data["min_delta"],
                      patience=es_cb_data["patience"],
                      verbose=es_cb_data["verbose"],
                      mode=es_cb_data["mode"])
reduce_lr = ReduceLROnPlateau(monitor=reduce_lr_data["monitor"],
                              factor=reduce_lr_data["factor"],
                              patience=reduce_lr_data["patience"],
                              min_lr=reduce_lr_data["min_lr"],
                              mode=reduce_lr_data["mode"],
                              verbose=reduce_lr_data["verbose"])
clf.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), callbacks=[es_cb, reduce_lr])
# clf.fit(X_train, Y_train)
y = clf.evaluate(X_test, Y_test)
print(y)

time_name = time.strftime("%Y%m%d-%H%M%S")

model_filename = '../../result/' + time_name + '/models'
my_makedirs(model_filename)
model = clf.export_model()
model.save(model_filename + '/auto_model.h5')

config_filename = '../../result/' + time_name + '/parameters'
my_makedirs(config_filename)
with open(config_filename + "/config.json", 'w') as f:
    json.dump(data, f)
