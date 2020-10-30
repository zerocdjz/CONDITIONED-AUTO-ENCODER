# -*- coding: utf-8 -*-

import os
import tensorflow as tf  
import numpy as np
import glob
import librosa
import sys

def load_files_info(fileDirs):
    fileType = "*.wav"
    fileList = glob.glob(os.path.join(fileDirs, fileType))
    fDict = {}
    for filePath in fileList:
        strArray = filePath.split("_") 
        typeID = strArray[-2]
        if not typeID in fDict.keys():
            fDict[typeID] = [filePath]
        else:
            fDict[typeID].append(filePath)  
    return fDict

#feature extractor
def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    y, sr = librosa.load(file_name, sr=None, mono=False)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    # log_mel_spectrogram = 20.0 / power * np.math.log10(mel_spectrogram + sys.float_info.epsilon)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    # 04 calculate total vector size
    # vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # # 05 skip too short clips
    # if vector_array_size < 1:
    #     return np.empty((0, dims))

    # # 06 generate feature vectors by concatenating multiframes
    # vector_array = np.zeros((vector_array_size, dims))
    # for t in range(frames):
    #     vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return log_mel_spectrogram


def convert_to_tfrecords(fDict, saveDirs):
    
    for idx, typeID in enumerate(sorted(fDict.keys())):
        
        fileList = fDict[typeID]
        filename = "Machine" + typeID + "_" + str(len(fileList)) + ".tfrecord"
        output_filename = os.path.join(saveDirs, filename)
        writer = tf.compat.v1.python_io.TFRecordWriter(output_filename)
        for eachFile in fileList:
            data = file_to_vector_array(eachFile, n_mels=128)
            data = data.flatten() # 后期修改为2D_CNN
            mu = np.mean(data)
            sigma = np.std(data)
            dataNormalize = (data-mu) / sigma
            classes = idx
            
            example = tf.train.Example(features=tf.train.Features(feature={
                                        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dataNormalize.tostring()])),
                                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[classes])),
                                        }))
            writer.write(example.SerializeToString())
        writer.close()
        print('convert %d machine to tfrecords...' %(idx))
        


if __name__ == "__main__":
    fileDirs = r"E:\project\anormal\dcase2020_task2_baseline\dev_data\fan\train"
    saveDirs = "./tfrecords"
    fDict = load_files_info(fileDirs)
    convert_to_tfrecords(fDict, saveDirs)
    print("-------")