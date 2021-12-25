import os,random,shutil
import json
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.patheffects as PathEffects
import seaborn as sns
import h5py
# from getROI import getROI
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from GhostNet import GhostModel
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score,f1_score,recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K 
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from GhostNet_v2 import GhostNet
tf.config.experimental_run_functions_eagerly(True)

def _hard_swish(x):
    """Hard swish
    """
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def _relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)

def generate(batch, shape, ptrain, ptest):

    #  Using the data Augmentation in traning data
    datagen1 = ImageDataGenerator(rescale=1. / 255)

    datagen2 = ImageDataGenerator(rescale=1. / 255)


    train_generator = datagen1.flow_from_directory(
        ptrain,
        target_size=(36,36),
        batch_size=batch,
        class_mode='categorical')


    test_generator = datagen2.flow_from_directory(
        ptest,
        target_size=(36,36),
        batch_size=batch,
        class_mode='categorical')

    count1 = 0
    for root, dirs, files in os.walk(ptrain):
        for each in files:  # 训练集个数 number of train dataset 
            count1 += 1



    count2 = 0
    for root, dirs, files in os.walk(ptest):  # 测试集个数 number of test dataset
        for each in files:
            count2 += 1

    return train_generator, test_generator, count1, count2


if __name__ == '__main__':

    with open('config_ghostnet.json', 'r') as f:
        cfg = json.load(f)

    save_dir = cfg['save_dir']
    shape = (int(cfg['height']), int(cfg['width']), 3)
    n_class = int(cfg['class_number'])
    batch = int(cfg['batch'])
    opt = Adam(lr=float(cfg['learning_rate']))

    train_generator, test_generator, count1, count2 = generate(batch, shape[:2], cfg['train_dir'],
                                                                     cfg['test_dir'])

    model = GhostNet(2)
    # model = GhostModel(numclass=n_class, size=cfg['height'], channel=3).GhostNet()
    model.build(input_shape=(None,36,36,3))
    model.summary()
    model.load_weights('mura_ghostnet_test.h5') #, custom_objects={"_hard_swish":_hard_swish,"_relu6":_relu6}
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'],experimental_run_tf_function=False)

    # print(model)
    # print("***")
    # x_test, y_test = test_generator.next() 
    # loss_and_metrics = model.evaluate(x_test, y_test, batch_size=16) 
    # print(loss_and_metrics)
    # print("***")

    # tf.compat.v1.enable_eager_execution(config=None, device_policy=None,  execution_mode=None)
    x_test, y_test = test_generator.next()  #valid
    since=time.time()
    for i in range(1000):
        y_pred = model.predict(x_test)
    time_elapsed=(time.time()-since)
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    y_test=[np.argmax(y) for y in y_test]   #max值
    y_pred=[np.argmax(y) for y in y_pred]     #max值
    precision = precision_score(y_test, y_pred, average='weighted')  #以下三个为评价指标
    recall = recall_score(y_test, y_pred, average='weighted')
    f1score = f1_score(y_test, y_pred, average='weighted')

    print('precision:',precision)
    print('recall:', recall)
    print('f1score', f1score)


