import numpy as np
import gc
import os
import random
import tensorflow as tf
from collections import deque
import argparse
import pickle
import heapq
from keras import backend as K, regularizers

SEED = 9
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(1)
# tf.random.set_random_seed(1)
global_step = 0

hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)
hidden_init_critic = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_critic = tf.keras.initializers.Orthogonal(gain=np.sqrt(1), seed=SEED)

hparams = {
    'l2': 0.0001,
    'link_state_dim': 120,
    'readout_units': 120,
    'learning_rate': 0.0002,
    'T': 5,
    'dropout_rate': 0.5,
}


class myModel(tf.keras.Model):

    def __init__(self, hparams, hidden_init_actor, kernel_init_actor):
        super(myModel, self).__init__()
        self.hparams = hparams

        # Define layers here
        self.Message = tf.keras.models.Sequential()  # 聚合
        self.Message.add(tf.keras.layers.Dense(self.hparams['link_state_dim'],
                                               kernel_initializer=hidden_init_actor,
                                               activation=tf.nn.selu, name="FirstLayer"))

        self.Update = tf.keras.layers.GRUCell(self.hparams['link_state_dim'], dtype=tf.float32)  # 更新

        self.Readout = tf.keras.models.Sequential()
        self.Readout.add(tf.keras.layers.Dense(self.hparams['readout_units'],
                                               activation=tf.nn.selu,
                                               kernel_initializer=hidden_init_actor,
                                               kernel_regularizer=regularizers.l2(hparams['l2']),
                                               name="Readout1"))
        # self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(tf.keras.layers.Dense(self.hparams['readout_units'],
                                               activation=tf.nn.selu,
                                               kernel_initializer=hidden_init_actor,
                                               kernel_regularizer=regularizers.l2(hparams['l2']),
                                               name="Readout2"))
        # self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(tf.keras.layers.Dense(1, kernel_initializer=kernel_init_actor, name="Readout3"))

    def build(self, input_shape=None):
        # Create the weights of the layer
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim'] * 2]))
        self.Update.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']]))
        self.Readout.build(input_shape=[None, self.hparams['link_state_dim']])
        self.built = True

    # @tf.function 定义前向传递过程
    def call(self, link_state, states_graph_ids, states_first, states_second, states_num_edges, training=False):
        # Execute T times
        for _ in range(self.hparams['T']):
            # We have the combination of the hidden states of the main edges with the neighbours组合主边的隐藏状态与邻边
            mainEdges = tf.gather(link_state, states_first)  # 提取主节点特征
            neighEdges = tf.gather(link_state, states_second)  # 提取邻居节点特征

            edgesConcat = tf.concat([mainEdges, neighEdges], axis=1)  # 特征拼接

            ### 1.a Message passing for link with all it's neighbours一个用于链接所有邻居的消息传递
            outputs = self.Message(edgesConcat)  # 信息传递

            ### 1.b Sum of output values according to link id index根据链路ID索引的输出值的总和
            edges_inputs = tf.math.unsorted_segment_sum(data=outputs, segment_ids=states_second,
                                                        num_segments=states_num_edges)  # 求和

            ### 2. Update for each link更新每个链路
            # GRUcell needs a 3D tensor as state because there is a matmul: Wrap the link state
            outputs, links_state_list = self.Update(edges_inputs, [link_state])  # 节点更新

            link_state = links_state_list[0]  # 将更新后的link_state重新赋值给原始的link_state变量
            # print(link_state)
        # Perform sum of all hidden states
        edges_combi_outputs = tf.math.segment_sum(link_state, states_graph_ids, name=None)  # 分组求和

        r = self.Readout(edges_combi_outputs, training=training)
        return r


# model = myModel(hparams, hidden_init_actor, kernel_init_actor)
# print(model)


if __name__ == '__main__':

    # Generate random bandwidth values
    bandwidth = np.random.uniform(low=24.0, high=75.0, size=(15, 40))
    latency = np.random.uniform(low=0.0, high=10.0, size=(15, 20))
    pk = np.random.uniform(low=0.0, high=30.0, size=(15, 30))
    situation = np.random.uniform(low=0.0, high=50, size=(15, 30))

    # Assign the bandwidth values to the link_state tensor
    #link_state = tf.constant(bandwidth, dtype=tf.float32)
    #print(link_state)

    # link_state = tf.random.normal((10, 20))
    states_graph_ids = tf.constant([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype=tf.int32)
    states_first = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=tf.int32)
    states_second = tf.constant([1, 2, 3, 0, 2, 4, 0, 1, 9, 6, 11, 13, 12, 10, 9], dtype=tf.int32)
    states_num_edges = 15

    link_state = tf.concat([bandwidth, latency, pk, situation], axis=-1)

    # Instantiate the model
    model = myModel(hparams, hidden_init_actor, kernel_init_actor)

    # Pass input data through the model
    output = model(link_state, states_graph_ids, states_first, states_second, states_num_edges)

    # Print the output
    print(output)
