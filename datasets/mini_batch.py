# -*- coding: utf-8 -*-

import os
import glob
import tensorflow as tf  
import random
import numpy as np 
from datasets.create_tfrecords import file_to_vector_array


class MiniTask(object):
    def __init__(self, data_folders, class_type, split_ratio):
        samedata_file = [os.path.join(data_folders, filename) for filename in os.listdir(data_folders) if class_type in filename]
        diffdata_file = [os.path.join(data_folders, filename) for filename in os.listdir(data_folders) if not class_type in filename]
        same_trainNums = int(len(samedata_file) * split_ratio)
        diff_trainNums = int(len(diffdata_file) * split_ratio)
        np.random.seed(100)
        np.random.shuffle(samedata_file)
        np.random.shuffle(diffdata_file)
        samedata_train = samedata_file[0:same_trainNums]
        samedata_val = samedata_file[same_trainNums:]

        diffdata_train = diffdata_file[0:diff_trainNums]
        diffdata_val = diffdata_file[diff_trainNums:]

        samelabel_train = [-1  for i in range(len(samedata_train))]
        samelabel_val = [-1 for i in range(len(samedata_val))]

        difflabel_train = [1 for i in range(len(diffdata_train))]
        difflabel_val = [1 for i in range(len(diffdata_val))]

        self.samedata_train = list(zip(samedata_train, samelabel_train))
        self.samedata_val = list(zip(samedata_val, samelabel_val))

        self.diffdata_train = list(zip(diffdata_train, difflabel_train))
        self.diffdata_val = list(zip(diffdata_val, difflabel_val))


class dataset():
    def __init__(self, task, split='train', shuffle=True, ratio=0.75, batch_size=32):
        self.task = task
        self.split = split
        self.shuffle = shuffle
        self.samedata = self.task.samedata_train if self.split == 'train' else self.task.samedata_val
        self.diffdata = self.task.diffdata_train if self.split == 'train' else self.task.diffdata_val
        self.sameNums = int(batch_size * ratio)
        self.diffNums = batch_size - self.sameNums
        
    def getitem(self, x):
        filePath = x[0]
        label = x[1]
        data = file_to_vector_array(filePath, n_mels=128)
        data = data.flatten() # 后期修改为2D_CNN
        mu = np.mean(data)
        sigma = np.std(data)
        dataNormalize = (data-mu) / sigma
        dataNormalize = np.expand_dims(dataNormalize,axis=0)
        label = np.expand_dims(np.asarray(label), axis=0)
        # label = np.expand_dims(label, axis=0)
        label = np.float32(label)
        return dataNormalize, label

    def generator(self):
        num_batch = np.math.floor(len(self.samedata) / self.sameNums)
        for i in range(num_batch):
            if(i==0):
                np.random.shuffle(self.samedata)
                np.random.shuffle(self.diffdata)
            same_batch = self.samedata[i * self.sameNums : i * self.sameNums + self.sameNums]
            diff_batch = self.diffdata[i * self.diffNums : i * self.diffNums + self.diffNums]
            batch_list = same_batch + diff_batch
            np.random.shuffle(batch_list)    
            for idx, x in enumerate(batch_list):
                x_single, y_single = self.getitem(x)
                if idx == 0:
                    x_batch = x_single 
                    y_batch = y_single
                else:
                    x_batch = np.concatenate((x_batch, x_single), axis=0)
                    y_batch = np.concatenate((y_batch, y_single), axis=0)
            yield x_batch, y_batch
                

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()

    fileDirs = r"E:\project\anormal\dcase2020_task2_baseline\dev_data\fan\train"
    task = MiniTask(fileDirs, "normal_id_00", 0.8)
    ds = dataset(task, split='train', shuffle=True, ratio=0.75, batch_size=32)
    for i in range(100):
        k = 0
        for x, y in ds.generator():
            k = k+1
            print("{} {}".format(k,x.shape))

    

    

    
