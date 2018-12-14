import tensorflow as tf
import numpy as np
from game import Board


class PolicyNet(object):
    def __init__(self, train_data_path=None):
        self.train_data_path = train_data_path

    # 自定义卷积层 可以更好的设置参数的初始化
    def conv(self, feature_map, channel_in, channel_out, kernel_size, scope, padding='SAME', dilation_rate=1,
             is_activation=True):
        with tf.variable_scope(scope):
            kernel = tf.get_variable('kernel', shape=[kernel_size, kernel_size, channel_in, channel_out],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('b', shape=[channel_out], initializer=tf.contrib.layers.xavier_initializer())
            out = tf.nn.convolution(feature_map, kernel, padding, dilation_rate=[dilation_rate, dilation_rate],
                                    name='conv') + bias
            if is_activation:
                out = tf.nn.relu(out, name='out')
            return out

    def batch_norm(self, x, training, name):
        with tf.variable_scope(name):
            beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
            axises = np.arange(len(x.shape) - 1)
            batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(training, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def residual_block(self, feature_map, channel_in, channel_out, scope_name, is_training, dilation_rate=(1, 1)):
        with tf.variable_scope(scope_name):
            bottleneck = self.conv(feature_map, channel_in, channel_out, kernel_size=3, scope='c1',
                                   dilation_rate=dilation_rate[0],
                                   is_activation=False)
            bottleneck = self.batch_norm(bottleneck, training=is_training, name='bn1')
            bottleneck = tf.nn.relu(bottleneck)
            bottleneck = self.conv(bottleneck, channel_out, channel_out, kernel_size=3, scope='c2',
                                   dilation_rate=dilation_rate[1], is_activation=False)
            bottleneck = self.batch_norm(bottleneck, training=is_training, name='bn2')
            out = tf.add(bottleneck, feature_map, name='add')
            out = tf.nn.relu(out)
        return out

    def policy_net(self, feature_map, is_training):
        with tf.variable_scope('net'):
            out = self.conv(feature_map, channel_in=21, channel_out=128, kernel_size=3, scope='c1', padding='SAME',
                            is_activation=False)
            out = self.batch_norm(out, training=is_training, name='bn1')
            out = tf.nn.relu(out)

            out = self.residual_block(out, 128, 128, scope_name='res1', is_training=is_training)
            out = self.residual_block(out, 128, 128, scope_name='res2', is_training=is_training)
            out = self.residual_block(out, 128, 128, scope_name='res3', is_training=is_training,
                                      dilation_rate=[2, 4])
            out = self.residual_block(out, 128, 128, scope_name='res4', is_training=is_training,
                                      dilation_rate=[8, 1])
            out = self.residual_block(out, 128, 128, scope_name='res5', is_training=is_training)

            out = self.conv(out, channel_in=128, channel_out=128, kernel_size=3, scope='c2', padding='SAME',
                            is_activation=False)
            out = self.batch_norm(out, training=is_training, name='bn2')
            out = tf.nn.relu(out)
            out = self.conv(out, channel_in=128, channel_out=1, kernel_size=3, scope='c3', padding='SAME',
                            is_activation=False)
            return out

    @staticmethod
    def loss(logits, labels):
        logits = tf.reshape(logits, shape=[-1, 361])
        labels = tf.reshape(labels, shape=[-1, 361])
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    @staticmethod
    def pre_process(x, y):
        x = np.asarray([PolicyNet.process_board(xx, yy, contain_liberty=True) for (xx, yy) in zip(x, y)],
                       dtype=np.float32)
        y = np.asarray([PolicyNet.process_label(yy) for yy in y], dtype=np.float32)
        return x, y

    @staticmethod
    def process_board(board_mtx, y, random=True, contain_liberty=False):
        # 训练时 将数据进行旋转对称 防止过拟合 减少数据相关性
        if random:
            rand = np.random.randint(0, 8)
            if rand <= 3:
                board_mtx = board_mtx.T
                y['current_move'] = (y['current_move'][1], y['current_move'][0])
                y['next_move'] = (y['next_move'][1], y['next_move'][0])
            i = rand % 4
            if i == 1:
                board_mtx = np.rot90(board_mtx)
                y['current_move'] = (18 - y['current_move'][1], y['current_move'][0])
                y['next_move'] = (18 - y['next_move'][1], y['next_move'][0])
                # print(a[2-idx[1]][idx[0]])

            if i == 2:
                board_mtx = np.rot90(board_mtx)
                board_mtx = np.rot90(board_mtx)
                y['current_move'] = (18 - y['current_move'][1], y['current_move'][0])
                y['next_move'] = (18 - y['next_move'][1], y['next_move'][0])
                y['current_move'] = (18 - y['current_move'][1], y['current_move'][0])
                y['next_move'] = (18 - y['next_move'][1], y['next_move'][0])
            if i == 3:
                board_mtx = np.rot90(board_mtx)
                board_mtx = np.rot90(board_mtx)
                board_mtx = np.rot90(board_mtx)
                y['current_move'] = (18 - y['current_move'][1], y['current_move'][0])
                y['next_move'] = (18 - y['next_move'][1], y['next_move'][0])
                y['current_move'] = (18 - y['current_move'][1], y['current_move'][0])
                y['next_move'] = (18 - y['next_move'][1], y['next_move'][0])
                y['current_move'] = (18 - y['current_move'][1], y['current_move'][0])
                y['next_move'] = (18 - y['next_move'][1], y['next_move'][0])
        # 生成传入的19 * 19 * 21 维度的数据
        black_board = np.zeros((19, 19, 1), np.uint8)
        black_board[board_mtx == 1] = 1
        white_board = np.zeros((19, 19, 1), np.uint8)
        white_board[board_mtx == 2] = 1
        # 将棋盘上每个落子点气的数目记录下来 当做输出数据喂进网络
        if contain_liberty:
            black_liberty = np.zeros((19, 19, 8), dtype=np.uint8)
            white_liberty = np.zeros((19, 19, 8), dtype=np.uint8)
            visited = {}
            for i in range(19):
                for j in range(19):
                    if board_mtx[i][j] == 1 and (i, j) not in visited:
                        group = Board.get_group(i, j, board_mtx, visited)
                        num_liberty = Board.check_liberty(group, board_mtx, cnt=True)
                        if num_liberty > 8:
                            num_liberty = 8
                        for pos in group:
                            black_liberty[pos[0]][pos[1]][num_liberty - 1] = 1
                    if board_mtx[i][j] == 2 and (i, j) not in visited:
                        group = Board.get_group(i, j, board_mtx, visited)
                        num_liberty = Board.check_liberty(group, board_mtx, cnt=True)
                        if num_liberty > 8:
                            num_liberty = 8
                        for pos in group:
                            white_liberty[pos[0]][pos[1]][num_liberty - 1] = 1
            black_board = np.concatenate((black_board, black_liberty), axis=2)
            white_board = np.concatenate((white_board, white_liberty), axis=2)
        board = np.concatenate((black_board, white_board), axis=2)
        ones = np.ones((19, 19, 1), dtype=np.uint8)
        last_move = np.zeros((19, 19, 1), dtype=np.uint8)
        if not y['ko_state:']:
            last_move[y['current_move'][0]][y['current_move'][1]] = 1
        else:
            last_move[y['current_move'][0]][y['current_move'][1]] = -1

        is_black_next = np.ones((19, 19, 1), dtype=np.uint8)
        if y['next_to_play'] == 2:
            is_black_next -= 1

        feat = np.concatenate((board, last_move, is_black_next, ones), axis=2)

        return feat

    @staticmethod
    def process_label(y):
        mtx = np.zeros((19, 19, 1), dtype=np.uint8)
        mtx[y['next_move'][0]][y['next_move'][1]] = 1
        return mtx
