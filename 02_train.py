# -*- coding: utf-8 -*-

import os
import glob
import tensorflow as tf  
import numpy as np
import yaml
from datasets import mini_batch
from nets import model
import time


def yaml_load():
    with open("parameter.yaml") as stream:
        param = yaml.safe_load(stream)
    return param


def main():
    param = yaml_load()
    #02:build model
    x = tf.compat.v1.placeholder(tf.float32, shape=[param['batch_size'], param['frameNums'] * param['mels']], name='input1')
    y = tf.compat.v1.placeholder(tf.float32, shape=[param['batch_size'],], name='input2')

    y_train = model.build_model(x, y, True, None, param['frameNums'], param['mels'])
    loss_train = model.calu_loss(x, y, y_train, 5, 0.75)

    y_val = model.build_model(x, y, False, True, param['frameNums'], param['mels'])
    loss_val = model.calu_loss(x, y, y_val, 5, 0.75)
    #03:train model
    train_step = param['trainNum'] // param['batch_size']
    tf.summary.scalar('train/loss', loss_train)
    tf.summary.scalar('val/loss', loss_val)
    merged = tf.compat.v1.summary.merge_all() 
    summary_writer = tf.compat.v1.summary.FileWriter(param['log'], tf.compat.v1.get_default_graph())

    global_step = tf.compat.v1.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(param['learning_rate'], global_step=global_step, decay_steps=train_step, decay_rate=0.94, staircase=True, name='exponential_decay_learning_rate')
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.compat.v1.control_dependencies(update_ops):
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False).minimize(loss_train, global_step=global_step)
    saver = tf.compat.v1.train.Saver(max_to_keep=10)

    config = tf.compat.v1.ConfigProto(log_device_placement=False,
                                      allow_soft_placement=True)
    with tf.compat.v1.Session(config=config) as sess:
        ckpt = tf.compat.v1.train.get_checkpoint_state(param['checkpoint'])
        if ckpt and tf.compat.v1.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.compat.v1.global_variables_initializer())
       
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        print('Start training...')
        task = mini_batch.MiniTask(param['dataDir'], "normal_id_00", 0.8)
        train_set = mini_batch.dataset(task)
        val_set = mini_batch.dataset(task, split='val')
        for epoch in range(param['epochs']):
            start_time = time.time()
            for x_train_batch, y_train_batch in train_set.generator():
                _ = sess.run(train_op, feed_dict={x:x_train_batch, y:y_train_batch})
            
            if epoch + 1 == 1 or (epoch + 1) % 10 == 0:
                train_loss = 0
                n_batch = 0
                for x_train_batch, y_train_batch in train_set.generator():
                    terr = sess.run(loss_train, feed_dict={x:x_train_batch, y:y_train_batch})
                    train_loss += terr   
                    n_batch += 1
                # summary_writer.add_summary(train_summary, epoch + 1)
                print("Epoch %d of %d took %fs" % (epoch + 1, param['epochs'], time.time() - start_time))
                print("   train loss:%f" % (train_loss / n_batch))

                val_loss = 0
                n_batch = 0
                for x_val_batch, y_val_batch in val_set.generator():
                    verr = sess.run(loss_val, feed_dict={x:x_val_batch, y:y_val_batch})
                    val_loss += verr   
                    n_batch += 1
                print("   val loss:%f" % (val_loss / n_batch))
                model_path = os.path.join(param['checkpoint'], param['model_name'])
                save_path = saver.save(sess, model_path, global_step=global_step)
                print("Model saved in file: ", save_path)

        summary_writer.close()
        coord.request_stop()
        coord.join(threads)
    



if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    main()