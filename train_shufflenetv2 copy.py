import os,random,shutil
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
# import torch
import matplotlib.patheffects as PathEffects
import seaborn as sns
# from getROI import getROI
import tensorflow
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from shufflenetv2 import ShuffleNet
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score,f1_score,recall_score
from keras_applications.imagenet_utils import preprocess_input

def preprocess(x):
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x /= 255.0
    x -=0.5
    x *=2.0
    return x


def generate(batch, shape, ptrain, pval,ptest):
    """Data generation and augmentation
    # Arguments
        batch: Integer, batch size.
        size: Integer, image size.
        ptrain: train dir.
        pval: eval dir.
    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        count1: Integer, number of train set.
        count2: Integer, number of test set.
    """

    #  Using the data Augmentation in traning data
    datagen1 = ImageDataGenerator(rescale=1. / 255)

    datagen2 = ImageDataGenerator(rescale=1. / 255)

    datagen3 = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen1.flow_from_directory(
        ptrain,
        target_size=shape,
        batch_size=batch,
        class_mode='binary')

    validation_generator = datagen2.flow_from_directory(
        pval,
        target_size=shape,
        batch_size=batch,
        class_mode='binary')

    test_generator = datagen3.flow_from_directory(
        pval,
        target_size=shape,
        batch_size=batch,
        class_mode='binary')
    # label_dict =datagen1.flow_from_directory(ptrain, target_size=shape, batch_size=1
    #                               ).class_indices
    # print(label_dict)

    count1 = 0
    for root, dirs, files in os.walk(ptrain):
        for each in files:  # 训练集个数 number of train dataset 
            count1 += 1

    count2 = 0
    for root, dirs, files in os.walk(pval):  # 测试集个数 number of test dataset
        for each in files:
            count2 += 1
    count3 = 0
    for root, dirs, files in os.walk(ptest):  # 测试集个数 number of test dataset
        for each in files:
            count3 += 1

    return train_generator, validation_generator, test_generator, count1, count2, count3

def moveFile_n(fileDirn):
    pathDir = os.listdir(fileDirn)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.2  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        shutil.move(fileDirn + name, tarDirn + name)
    return
def moveFile_p(fileDirp):
    pathDir = os.listdir(fileDirp)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.2  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        shutil.move(fileDirp + name, tarDirp + name)
    return

def train():
    with open('config_sufflenet.json', 'r') as f:
        cfg = json.load(f)

    save_dir = cfg['save_dir']
    shape = (int(cfg['height']), int(cfg['width']), 3)
    print(shape)
    n_class = int(cfg['class_number'])
    batch = int(cfg['batch'])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    model = ShuffleNet(groups=3,pooling='avg',classes=1)
    pre_weights = cfg['weights']

    if pre_weights and os.path.exists(pre_weights):
        model.load_weights(pre_weights, by_name=True)  # 加载模型
    since=time.time()
    opt = Adam(lr=float(cfg['learning_rate']))
    earlystop = EarlyStopping(monitor='val_acc', patience=5, verbose=0, mode='auto')  # 提前终止，降低过拟合
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_dir, '{}_weights.h5'.format(cfg['model'])),
                                 monitor='val_acc', save_best_only=True, save_weights_only=True)  # 保存最佳模型，只保存权重

    # 损失函数

    train_generator, validation_generator, test_generator, count1, count2,count3 = generate(batch, shape[:2], cfg['train_dir'],
                                                                     cfg['eval_dir'],cfg['test_dir'])
    # y_pred = model.predict(validation_generator[0]).ravel()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    hist = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=count1 // batch,
        validation_steps=count2 // batch,
        epochs=cfg['epochs'],
        callbacks=[earlystop, checkpoint])  # 训练时调用的一系列回调函数

    # 连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录
    df = pd.DataFrame.from_dict(hist.history)  # 显示数据
    df.to_csv(os.path.join(save_dir, 'hist_shuffle___.csv'), encoding='utf-8', index=False)
    time_elapsed=time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # print("***")
    # x_test, y_test = test_generator.next() 
    # loss_and_metrics = model.evaluate(x_test, y_test, batch_size=16)
    # print(loss_and_metrics)
    # print("***")

 
    # x_test, y_test = test_generator.next()  #valid
    # y_pred = model.predict(x_test)   #predict
    # y_test=[np.argmax(y) for y in y_test]   #max值
    # y_pred=[np.argmax(y) for y in y_pred]     #max值
    # precision = precision_score(y_test, y_pred, average='weighted')  #以下三个为评价指标
    # recall = recall_score(y_test, y_pred, average='weighted')
    # f1score = f1_score(y_test, y_pred, average='weighted')

    # print('precision:',precision)
    # print('recall:', recall)
    # print('f1score', f1score)

    # #输出ROC曲线
    # fpr, tpr, thresholds_keras = roc_curve(y_test, y_pred)
    # aucc = auc(fpr, tpr)
    # print("AUC : ", aucc)
    # plt.figure()
    # plt.plot(fpr, tpr, 'b', label='Keras AUC = %0.2f' % aucc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()

    #输出acc和loss曲线
    plt.figure()
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

    model.save("mura_shufflenet__.h5") #保存模型
    print("***")


if __name__ == '__main__':
    # root_path = "hyun_negative_data"
    # root_path1 = "hyun_positive_data"
    # getROI().classify(root_path,root_path1)
    #fileDirn = "C:\\Users\\miryu\\Desktop\\dataa\\train\\Negative\\"  # 源图片文件夹路径
    #tarDirn = 'C:\\Users\\miryu\\Desktop\\dataa\\val\\Negative\\'  # 移动到新的文件夹路径
    #moveFile_n(fileDirn)
    #fileDirp = "C:\\Users\\miryu\\Desktop\\dataa\\train\\Positive\\"  # 源图片文件夹路径
    #tarDirp = 'C:\\Users\\miryu\\Desktop\\dataa\\val\\Positive\\'  # 移动到新的文件夹路径
    #moveFile_p(fileDirp)

    train()
