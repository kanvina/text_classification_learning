#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import print_function

import os
import sys
import time
import pandas as pd
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from preprocessing import read_vocab, read_category, batch_iter, process_file, build_vocab
import tensorflow.keras as kr
import shutil

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def train():
    print("Configuring TensorBoard and Saver...")


    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    if os.path.exists('tensorboard'):
        shutil.rmtree('tensorboard')

    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss:{1:>6.2}, Train Acc:{2:>7.2%},' \
                      + ' Val Loss:{3:>6.2}, Val Acc:{4:>7.2%}, Time:{5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            feed_dict[model.keep_prob] = config.dropout_keep_prob
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break

def brand_test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss:{0:>6.2}, Test Acc:{1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    # batch_size = 128
    # data_len = len(x_test)
    # num_batch = int((data_len - 1) / batch_size) + 1
    #
    # y_test_cls = np.argmax(y_test, 1)
    # y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    #
    # for i in range(num_batch):  # 逐批次处理
    #     start_id = i * batch_size
    #     end_id = min((i + 1) * batch_size, data_len)
    #     feed_dict = {
    #         model.input_x: x_test[start_id:end_id],
    #         model.keep_prob: 1.0
    #     }
    #     y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)
    #
    # # 评估
    # print("Precision, Recall and F1-Score...")
    # print(metrics.classification_report(y_test_cls, y_pred_cls))
    #
    # # 混淆矩阵
    # print("Confusion Matrix...")
    # cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    # print(cm)
    #
    # time_dif = get_time_dif(start_time)
    # print("Time usage:", time_dif)



def get_train_val_test_data(data_all,ratio_train,ratio_val,ratio_test):
    train_data=[]
    val_data=[]
    test_data=[]

    num_train=int(ratio_train*len(data_all))
    num_val=num_train+int(ratio_val*len(data_all))
    num_test=int((1-ratio_test)*len(data_all))


    for i in range(num_train):
        data = data_all[i]
        train_data.append([data[0], data[1]])

    for i in range(num_train,num_val):
        data=data_all[i]
        val_data.append([data[0],data[1]])

    for i in range(num_test,len(data_all)):
        data=data_all[i]
        test_data.append([data[0],data[1]])



    return train_data,val_data,test_data

if __name__ == '__main__':

    base_dir = 'data/brand'

    data_dir = os.path.join(base_dir, 'name_brand_food.csv')

    train_dir = os.path.join(base_dir, 'train.csv')
    val_dir = os.path.join(base_dir, 'val.csv')
    test_dir = os.path.join(base_dir, 'test.csv')
    data_all=[]
    data_all_list=np.array(pd.read_csv('{0}'.format(data_dir),encoding='GBK')).tolist()
    for data_name in data_all_list:
        name=str(data_name[0])
        category=data_name[1]
        if name.find('(')!=-1:
            name=name.split('(')
            name=str(name[0])
            data_all.append([name,category])
        else:
            data_all.append([name,category])
    np.random.shuffle(data_all)
    train_data,val_data,test_data=get_train_val_test_data(data_all,0.7,0.2,0.1)
    print('训练集数据数量：',len(train_data),'条')
    print('验证集数据数量：', len(val_data), '条')
    print('测试集数据数量：', len(test_data), '条')

    '''
    输出训练集，验证集，测试集至csv格式
    '''
    pd.DataFrame(train_data).to_csv('{0}'.format(train_dir),index=0,encoding='GBK')
    pd.DataFrame(val_data).to_csv('{0}'.format(val_dir), index=0,encoding='GBK')
    # pd.DataFrame(test_data).to_csv('{0}'.format(test_dir), index=0, encoding='GBK')

    vocab_dir = os.path.join(base_dir, 'brand_vocab.txt')
    save_dir = 'checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

    print('Configuring CNN model...')
    config = TCNNConfig()

    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(data_dir, vocab_dir, config.vocab_size)

    categories, cat_to_id = read_category(data_dir)

    words, word_to_id = read_vocab(vocab_dir)

    config.vocab_size = len(words)
    config.num_classes=len(categories)

    model = TextCNN(config)

    train()
    brand_test()



