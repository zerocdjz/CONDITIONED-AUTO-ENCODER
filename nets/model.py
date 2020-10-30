# -*- coding: utf-8 -*-

import os
import tensorflow as tf  
import numpy as np 


def denseBlock(inputs, dims, is_train, layerNum):
    x = tf.compat.v1.layers.dense(inputs, dims, name="dense_" + layerNum)
    x = tf.compat.v1.layers.batch_normalization(x, training=is_train, name="bn_" + layerNum)
    x = tf.nn.relu(x, name='relu_' + layerNum)
    return x 

def FiLM_Layer(feature, y):
    with tf.compat.v1.variable_scope("condition"):
        if len(y.shape) < 2:
            y = tf.expand_dims(y, axis=1)
        x1 = tf.compat.v1.layers.dense(y, 16, name="con_1")
        x1 = tf.compat.v1.nn.sigmoid(x1)

        x2 = tf.compat.v1.layers.dense(y, 16, name="con_2")

        conditionZ = x1 * feature + x2 
    return conditionZ

def build_model(x, y, is_train, reuse, frameNums, mels):
    with tf.compat.v1.variable_scope("c2ae", reuse=reuse) as scope:
        with tf.compat.v1.variable_scope("encode"):
            edenseblock1 = denseBlock(x, 128, is_train, "1")
            edenseblock2 = denseBlock(edenseblock1, 64, is_train, "2")
            edenseblock3 = denseBlock(edenseblock2, 32, is_train, "3")
            featureZ = denseBlock(edenseblock3, 16, is_train, "4")
        # with tf.compat.v1.variable_scope("classify"):
        #     logits = tf.compat.v1.layers.dense(featureZ, 4, name="logits")

        conditionZ = FiLM_Layer(featureZ, y)

        with tf.compat.v1.variable_scope("decode"):
            dblock1 = denseBlock(conditionZ, 128, is_train, "1")
            dblock2 = denseBlock(dblock1, 128, is_train, "2")
            dblock3 = denseBlock(dblock2, 128, is_train, "3")
            dblock4 = denseBlock(dblock3, 128, is_train, "4")
            dout = tf.compat.v1.layers.dense(dblock4, frameNums*mels, name= "out") # frameNums*mel
            # dout = tf.reshape(dout, (frameNums, mels))

    return dout


def calu_loss(x, y, ypred, C, alpha):
    smooth = 1e-6
    #non match
    ynm_index = tf.where(tf.greater(y, 0))
    ynm = tf.gather(ypred, ynm_index)
    xnm = tf.gather(x, ynm_index)
    ynm = tf.squeeze(ynm, axis=1)
    xnm = tf.squeeze(xnm, axis=1)
    #match
    ym_index = tf.where(tf.less(y, 0))
    ym = tf.gather(ypred, ym_index)
    xm = tf.gather(x, ym_index)
    ym = tf.squeeze(ym, axis=1)
    xm = tf.squeeze(xm, axis=1)

    loss_nm = tf.reduce_mean(tf.abs(ynm - C)) + smooth
    # loss_nm = tf.reduce_mean(tf.square(ynm - xnm)) + smooth
    loss_m = tf.reduce_mean(tf.abs(ym - xm)) + smooth
    
    # loss_aux = tf.reduce_mean(tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    loss = alpha * loss_m + (1 - alpha) * loss_nm #+ loss_aux
    
    # correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss
           

if __name__ == "__main__":
    #test
    tf.compat.v1.disable_eager_execution()
    sampleNums = 8 
    frameNums = 5
    mels = 128

    x = tf.compat.v1.placeholder(tf.float32, shape=[sampleNums, frameNums*mels], name='input1')
    y = tf.constant(np.ones(shape=[sampleNums, 1]), tf.float32, name='input2')
    
    ypred = build_model(x, y, True, None, frameNums, mels)
    loss = calu_loss(x, y, ypred, 5, 0.75)
    
    print("-------")
    


