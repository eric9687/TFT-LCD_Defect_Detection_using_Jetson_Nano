from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants

import random,shutil
import json
import pandas as pd

import matplotlib.patheffects as PathEffects
import seaborn as sns

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from MobileNetV3_Small import MobileNetV3_Small

model = tf.keras.models.load_model('mura_mobilenet_master')


###### Convert in to FP32 model

print('Converting to TF-TRT FP32...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32,
                                                               max_workspace_size_bytes=8000000000)

converter = trt.TrtGraphConverterV2(input_saved_model_dir='mura_mobilenet_master',
                                    conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir='mura_mobilenet_master_FP32')
print('Done Converting to TF-TRT FP32')


###### Convert in to FP16 model

print('Converting to TF-TRT FP16...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16,
    max_workspace_size_bytes=8000000000)
converter = trt.TrtGraphConverterV2(
input_saved_model_dir='mura_mobilenet_master', conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir='rmura_mobilenet_master_FP16')
print('Done Converting to TF-TRT FP16')


###### Convert in to INT8 model

print('Converting to TF-TRT INT8...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.INT8, 
    max_workspace_size_bytes=8000000000, 
    use_calibration=True)
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='mura_mobilenet_master', 
    conversion_params=conversion_params)

# def calibration_input_fn():
#     yield (batched_input, )
converter.convert(calibration_input_fn=calibration_input_fn)

converter.save(output_saved_model_dir='resnet50_saved_model_TFTRT_INT8')
print('Done Converting to TF-TRT INT8')

def generate(batch, shape, ptrain, ptest):

    #  Using the data Augmentation in traning data
    datagen1 = ImageDataGenerator(rescale=1. / 255)

    datagen2 = ImageDataGenerator(rescale=1. / 255)


    train_generator = datagen1.flow_from_directory(
        ptrain,
        target_size=shape,
        batch_size=batch,
        class_mode='categorical')


    test_generator = datagen2.flow_from_directory(
        ptest,
        target_size=shape,
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
    
    with open('config_mobilenet.json', 'r') as f:
        cfg = json.load(f)

    save_dir = cfg['save_dir']
    shape = (int(cfg['height']), int(cfg['width']), 3)
    n_class = int(cfg['class_number'])
    batch_size= int(cfg['batch'])

   ## Input the model type
    model = tf.saved_model.load('rmura_mobilenet_master_FP16',tags=[tag_constants.SERVING])
    signature_keys = list(model.signatures.keys())
    print(signature_keys)
    infer = model.signatures['serving_default']
    print(infer.structured_outputs)   

    image_path = './hyun_data/test/Positive/'
    file_lists = os.listdir(image_path)

    since=time.time()

    for file in file_lists:
        tmp_img = image.load_img(image_path+'/'+file, target_size=(36, 36))
        tmp_x = image.img_to_array(tmp_img)
        tmp_x = np.expand_dims(tmp_x, axis=0)
        tmp_x = tf.constant(tmp_x)

        for i in range(1000):
            tmp_labeling = infer(tmp_x)
    
    time_elapsed=(time.time()-since)

    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    exit()


    since=time.time()
 
    for i in range(1000):
        labeling = infer(x)
        # y_pred = labeling['predictions'].numpy()
    time_elapsed=(time.time()-since)

    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # y_test=[np.argmax(y) for y in y_test]   #max值
    # y_pred=[np.argmax(y) for y in y_pred]     #max值
    # precision = precision_score(y_test, y_pred, average='weighted')  #以下三个为评价指标
    # recall = recall_score(y_test, y_pred, average='weighted')
    # f1score = f1_score(y_test, y_pred, average='weighted')

    # print('precision:',precision)
    # print('recall:', recall)
    # print('f1score', f1score)


