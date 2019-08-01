# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
import numpy as np
import pandas as pd

from cnn_model import TCNNConfig, TextCNN
from preprocessing import read_category, read_vocab

base_dir = 'data/brand'
vocab_dir = os.path.join(base_dir, 'brand_vocab.txt')
save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category('data/brand/name_brand_food.csv')
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.config.num_classes = len(self.categories)
        self.model = TextCNN(self.config)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = str(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.predictions, feed_dict=feed_dict)
        predictions = self.session.run(self.model.probs, feed_dict=feed_dict)
        max_radio = predictions.max()
        predicted_category = self.categories[y_pred_cls[0]]
        return predicted_category, max_radio


if __name__ == '__main__':
    cnn_model = CnnModel()

    name_list=np.array(pd.read_csv('data/brand/data_test.csv',encoding='GBK'))

    name_cat_predict=[]

    for name in name_list:
        name=name[0]
        if name.find('(') != -1:
            name = name.split('(')
            name = str(name[0])
        elif name.find('（') != -1:
            name = name.split('（')
            name = str(name[0])

        cat_predict,max_radio=cnn_model.predict(name)

        if max_radio>0.7:
            # name_cat_predict.append([name,cat_predict])
            print(name,cat_predict,max_radio)
        # else:
        #     name_cat_predict.append([name,0])
        #     print(name, cat_predict, max_radio)

        name_cat_predict.append([name, cat_predict,max_radio])



    pd.DataFrame(name_cat_predict,columns=['name','category','max_radio']).to_csv('data/brand/predict_result.csv',index=0)
