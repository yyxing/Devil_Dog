import numpy as np
from policy_net import PolicyNet
from value_net import ValueNet
import tensorflow as tf
import time


class MCTS(object):
    # 初始化整棵树和神经网络
    def __init__(self, policy_model_path, value_model_path, time_limit=20):
        self.time_limit = time_limit
        self.game = None
        self.root = None
        policy_model = PolicyNet()
        value_model = ValueNet()

        policy = tf.Graph()
        with policy.as_default():
            self.policy_board = tf.placeholder(dtype=tf.float32, shape=(None, 19, 19, 21))
            self.p_is_training = tf.placeholder(dtype=tf.bool)
            self.policy_out = policy_model.policy_net(self.policy_board, is_training=self.p_is_training)
            self.policy_loader = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.policy_sess = tf.Session(config=config)
            print('load policy model:', policy_model_path)
            self.policy_loader.restore(self.policy_sess, policy_model_path)

        value = tf.Graph()
        with value.as_default():
            self.value_board = tf.placeholder(dtype=tf.float32, shape=(None, 19, 19, 21))
            self.v_is_training = tf.placeholder(dtype=tf.bool)
            _, self.value_out = value_model.value_net(self.value_board, self.v_is_training)
            print("self.value_out", self.value_out)
            self.value_loader = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.value_sess = tf.Session(config=config)
            print('load value model:', value_model_path)
            self.value_loader.restore(self.value_sess, value_model_path)

    def start(self):
        cnt = 0
        start = time.time()
        while True:
            cnt += 1
            print("simulation No：{0}".format(cnt))
            self.simulation()
            print(time.time() - start)
            for child in self.root.children:
                print('b_win_rate / prior / visit_cnt / move:',
                      child.black_win_rate, child.p, child.N, child.last_move)
            if time.time() - start > self.time_limit:
                self.root.children.sort(key=lambda c: -c.N)
                for child in self.root.children:
                    print(child.N, child.last_move)
                return self.root.children[0].last_move[0]

    # MCTS搜索到叶子节点时 需要扩展 扩展的数据来源于神经网络 将当前局面喂进神经网络
    # 返回当前局面的的价值评估和接下来每一步的概率 从中选取概率最高的8步进行扩展
    def eval_node(self, game, node, is_value=True, width=8):
        board_clone = game.boards[-1].board
        if is_value:
            value_query = ValueNet.process_board(board_clone, {'next_to_play': game.next_to_play,
                                                               'ko_state:': game.ko_state[-1],
                                                               'current_move': game.current_moves[-1]},
                                                 random=False, contain_liberty=True)
            value_query = np.asarray([value_query], np.float32)
            black_win_rate, = self.value_sess.run([self.value_out], feed_dict={self.value_board: value_query,
                                                                               self.v_is_training: False})
            black_win_rate = black_win_rate.reshape((1,))[0]
            node.black_win_rate = black_win_rate
            print("black_win_rate",black_win_rate)
        else:
            policy_query = PolicyNet.process_board(board_clone, {'next_to_play': game.next_to_play,
                                                                 'ko_state:': game.ko_state[-1],
                                                                 'current_move': game.current_moves[-1]},
                                                   random=False, contain_liberty=True)
            policy_query = np.asarray([policy_query], np.float32)
            p, = self.policy_sess.run([self.policy_out], feed_dict={self.policy_board: policy_query,
                                                                    self.p_is_training: False})
            probs = np.reshape(p, (19, 19))
            probs -= np.max(probs)
            probs = np.exp(probs) / np.sum(np.exp(probs))

            ids = np.dstack(np.unravel_index(np.argsort(probs.ravel()), (19, 19)))[0]
            ids = ids[::-1][:width, :]
            moves = [([move[0], move[1]], probs[move[0]][move[1]]) for move in ids]
            print(moves)
            node.next_moves = [move for move in moves if game.is_acceptable(*move[0])]

    # 扩展节点
    def expand(self, game, node):
        if node.next_moves is None:
            self.eval_node(game, node, is_value=False)
        for move in node.next_moves:
            child = Node(move, p=move[1])
            node.children.append(child)

    # 模拟通过多次MCTS搜索 将MCTS树的深度增加
    def simulation(self, expand_limit=1):
        path = [self.root]
        self.root.black_win_rate = -1
        value = -1
        for node in path:
            if not node.is_leaf:
                next_node = self.select(self.game, node)
                path.append(next_node)
                self.game.move(next_node.last_move[0][0], next_node.last_move[0][1])
            else:
                value = node.black_win_rate
                if node.N > expand_limit:
                    self.expand(self.game, node)
                    for child in node.children:
                        self.evaluate(self.game, child)
                    node.is_leaf = False
        print('length of path of this prob:', len(path))
        self.backup(path, value)
        while len(path) != 1:
            self.game.roll_back()
            path.pop()

    # 根据某个值查询当前局面最好的节点
    def select(self, game, node):
        if game.next_to_play == 1:
            node.children.sort(key=lambda c: c.black_win_rate + c.p / (1 + c.N))
        else:
            node.children.sort(key=lambda c: -c.black_win_rate + c.p / (1 + c.N))
        return node.children[-1]

    # 评估函数 将扩展的每一个子节点对应的局面放进神经网络评估胜率
    def evaluate(self, game, node):
        if node.black_win_rate is None:
            game.move(node.last_move[0][0], node.last_move[0][1])
            self.eval_node(game, node, is_value=True)
            game.roll_back()
        return node.black_win_rate

    # 将模拟得到的值沿着路径反向更新
    def backup(self, path, value):
        for node in path:
            node.N += 1
            node.black_win_rate = (float(node.N - 1) * node.black_win_rate + value) / node.N

    # 初始化棋盘
    def set_game(self, game):
        self.game = game
        self.root = Node(move=game.current_moves[-1], p=None)


class Node(object):
    # MCTS的每个节点
    def __init__(self, move, p):
        # 这个节点下的位置
        self.last_move = move
        # 这个节点的概率
        self.p = p
        # 这个节点被访问的次数 和概率做计算作为最后select的依据
        self.N = 0
        # 是否为叶子节点
        self.is_leaf = True
        # 这个节点的子节点
        self.children = []
        # 当前局面黑棋胜率 用1-该胜率则为白棋胜率
        self.black_win_rate = None
        # 下一步棋的位置 用于记录整个树的路径
        self.next_moves = None

