# -*- coding: utf-8 -*-

import os
import glob
import time
import yaml

import numpy as np
import tensorflow as tf  
from nets import model
from datasets import create_tfrecords
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image


def yaml_load():
    with open("parameter.yaml") as stream:
        param = yaml.safe_load(stream)
    return param


def main(testDir):
    param = yaml_load()

    x = tf.compat.v1.placeholder(tf.float32, [1, param['frameNums']*param['mels']])
    y = tf.compat.v1.placeholder(tf.float32,[1,1])
    dataPred = model.build_model(x, y, False, None, param['frameNums'], param['mels'])

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        model_path = os.path.join(param['checkpoint'], 'c2ae.ckpt-6000')
        saver.restore(sess, model_path)
        print('Load Model Params Sucess...')
        
        fileList = glob.glob(os.path.join(testDir, "*_id_00_*"))
        y_true = []
        y_pred = []
        for eachFile in tqdm(fileList):
            fileName = os.path.basename(eachFile)
            if "anomaly" in eachFile:
                y_true.append(1)
            else:
                y_true.append(0)
            data = create_tfrecords.file_to_vector_array(eachFile, 128)
            data = data.flatten() # 后期修改为2D_CNN
            mu = np.mean(data)
            sigma = np.std(data)
            dataNormalize = (data-mu) / sigma

            dataTest = np.expand_dims(dataNormalize, axis=0)
            dataTest = np.float32(dataTest)
            labelTest = -np.ones(shape=[1,], dtype=np.float32)
            labelTest = np.expand_dims(labelTest, axis=1)

            feed_dict = {x:dataTest, y:labelTest}
            dataP = sess.run([dataPred], feed_dict=feed_dict)
            dataP = np.squeeze(dataP[0])
            err = np.mean(abs(dataNormalize - dataP))
            y_pred.append(err)
            
            #plot
            # dataNormalize = np.reshape(dataNormalize, [128,313])
            # dataP = np.reshape(dataP, [128,313])
            
            # plt.figure(0)
            # plt.subplot(311)
            # plt.imshow(dataNormalize)
            # plt.title(fileName)
            # # plt.axis('off')

            # plt.subplot(312)
            # plt.imshow(dataP)
            # # plt.axis('off')

            # plt.subplot(313)
            # plt.imshow(abs(dataP - dataNormalize)) 
            # plt.title(str(err))
            # plt.show()

    
    auc = metrics.roc_auc_score(y_true, y_pred)
    print("AUC:%.2f"%(auc))
    plt.figure(0)
    plt.subplot(211)
    plt.plot(y_true, 'r*')

    plt.subplot(212) 
    plt.plot(y_pred, 'b*') 
    plt.show() 

    # # p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
    
    # # print("pAUC:%.2f"%(p_auc))
    

        

        


if __name__=="__main__":
    tf.compat.v1.disable_eager_execution()
    testDir = r"E:\project\anormal\dcase2020_task2_baseline\dev_data\fan\test"
    main(testDir)
