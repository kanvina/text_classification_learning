# # coding: utf-8
#
# import tensorflow as tf
#
#
# class TCNNConfig(object):
#     """CNN配置参数"""
#
#     embedding_dim = 64  # 词向量维度
#     seq_length = 100 # 序列长度
#     num_classes = 10 # 类别数
#     num_filters = 16  # 卷积核数目
#     kernel_size = 3  # 卷积核尺寸
#     vocab_size = 5000  # 词汇表达小
#     hidden_dim = 128  # 全连接层神经元
#     dropout_keep_prob = 0.5  # dropout保留比例
#     learning_rate = 1e-3  # 学习率
#     batch_size = 64  # 每批训练大小
#     num_epochs = 20  # 总迭代轮次
#     print_per_batch = 50  # 每多少轮输出一次结果
#     save_per_batch = 10  # 每多少轮存入tensorboard
#
#
# class TextCNN(object):
#     """文本分类，CNN模型"""
#
#     def __init__(self, config):
#         self.config = config
#
#         # 三个待输入的数据
#         self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
#         self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
#         self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
#
#         self.cnn()
#
#     def cnn(self):
#         """CNN模型"""
#         # 词向量映射
#         with tf.device('/cpu:0'):
#             embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
#             embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
#
#         with tf.name_scope("cnn"):
#             # CNN layer
#             conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
#             # global max pooling layer
#             gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
#
#         with tf.name_scope("score"):
#             # 全连接层，后面接dropout以及relu激活
#             fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
#             fc = tf.contrib.layers.dropout(fc, self.keep_prob)
#             fc = tf.nn.relu(fc)
#
#             # 分类器
#             self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
#             self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
#
#         with tf.name_scope("optimize"):
#             # 损失函数，交叉熵
#             cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
#             self.loss = tf.reduce_mean(cross_entropy)
#             # 优化器
#             self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
#
#         with tf.name_scope("accuracy"):
#             # 准确率
#             correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
#             self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):

    def __init__(self):
        """CNN配置参数"""

        self.embedding_dim = 128  # 词向量维度
        self.seq_length = 50  # 序列长度
        self.num_classes = 0  # 类别数
        self.num_filters = 28  # 卷积核数目
        self.kernel_size =  [2,3,4,5]# 卷积核尺寸
        self.vocab_size = 5000  # 词汇表达小

        self.hidden_dim = 128  # 全连接层神经元

        self.dropout_keep_prob = 0.5  # dropout保留比例
        self.learning_rate = 1e-3  # 学习率


        self.batch_size = 128 # 每批训练大小
        self.num_epochs = 50 # 总迭代轮次

        self.print_per_batch = 10  # 每多少轮输出一次结果
        self.save_per_batch = 5  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)
        self.l2_reg_lambda = 0.0

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        # """
        # 通过一个隐藏层, 将 one-hot 编码的词 投影 到一个低维空间中.
        # 特征提取器，在指定维度中编码语义特征. 这样, 语义相近的词, 它们的欧氏距离或余弦距离也比较近.
        # self.W可以理解为词向量词典，存储vocab_size个大小为embedding_size的词向量，随机初始化为-1~1之间的值；
        # self.embedded_chars是输入input_x对应的词向量表示；size：[句子数量, sequence_length, embedding_size]
        # self.embedded_chars_expanded是，将词向量表示扩充一个维度（embedded_chars * 1），维度变为[句子数量, sequence_length, embedding_size, 1]，方便进行卷积（tf.nn.conv2d的input参数为四维变量，见后文）
        # 函数tf.expand_dims(input, axis=None, name=None, dim=None)：在input第axis位置增加一个维度（dim用法等同于axis，官方文档已弃用）
        # """
        #
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.config.vocab_size, self.config.embedding_dim], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            # embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        # conv-maxpool-i/filter_shape：卷积核矩阵的大小，包括num_filters个（输出通道数）大小为filter_size*embedding_size的卷积核，输入通道数为1；卷积核尺寸中的embedding_size，相当于对输入文字序列从左到右卷，没有上下卷的过程。
        # conv-maxpool-i/W：卷积核，shape为filter_shape，元素随机生成，正态分布
        # conv-maxpool-i/b：偏移量，num_filters个卷积核，故有这么多个偏移量
        # conv-maxpool-i/conv：conv-maxpool-i/W与self.embedded_chars_expanded的卷积
        # 函数tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)实现卷积计算

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.kernel_size):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer  卷积操作
                filter_shape = [filter_size, self.config.embedding_dim, 1, self.config.num_filters]
                # 随机生成正太分布
                # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # input:输入的词向量，[句子数（图片数）batch, 句子定长（对应图高）,词向量维度（对应图宽）, 1（对应图像通道数）]
                # filter:卷积核，[卷积核的高度，词向量维度（卷积核的宽度），1（图像通道数），卷积核个数（输出通道数）]
                # strides:图像各维步长,一维向量，长度为4，图像通常为[1, x, x, 1]
                # padding:卷积方式，'SAME'为等长卷积, 'VALID'为窄卷积
                # 输出feature map：shape是[batch, height, width, channels]这种形式

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                # value：待池化的四维张量，维度是[batch, height, width, channels]
                # ksize：池化窗口大小，长度（大于）等于4的数组，与value的维度对应，一般为[1,height,width,1]，batch和channels上不池化
                # strides:与卷积步长类似
                # padding：与卷积的padding参数类似
                # 返回值shape仍然是[batch, height, width, channels]这种形式
                pooled_outputs.append(pooled)
            # 池化后的结果append到pooled_outputs中。对每个卷积核重复上述操作，故pooled_outputs的数组长度应该为num_filters。


        # Combine all the pooled features
        # 将pooled_outputs中的值全部取出来然后reshape成[len(input_x),num_filters*len(filters_size)]，然后进行了dropout层防止过拟合，
        # 最后再添加了一层全连接层与softmax层将特征映射成不同类别上的概率
        # 2 3   把池化层输出变成一维向量
        num_filters_total = self.config.num_filters * len(self.config.kernel_size)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # tf.concat(values, concat_dim)连接values中的矩阵，concat_dim指定在哪一维（从0计数）连接。
        # values[i].shape = [D0, D1, ... Dconcat_dim(i), ...Dn]，连接后就是：[D0, D1, ... Rconcat_dim, ...Dn]。
        # 回想pool_outputs的shape，是存在pool_outputs中的若干种卷积核的池化后结果，维度为[len(filter_sizes),
        # batch, height, width, channels=1]，因此连接的第3维为width，即对句子中的某个词，将不同核产生的计算
        # 结果（features）拼接起来。

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.config.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.config.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.probs = tf.nn.softmax(self.logits)

            # self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.probs = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            # 计算张量的尺寸的元素平均值。
            # self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

