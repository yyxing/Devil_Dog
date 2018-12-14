import numpy as np
import cv2


class Game(object):
    def __init__(self, handicap=0):
        if handicap == 0:
            self.boards = [Board()]
            self.next_to_play = 1
        else:
            board = np.zeros((19, 19), dtype=np.int)
            if handicap == 2:
                board[3][15] = 1
                board[15][3] = 1
            self.boards = [Board(board)]
            self.next_to_play = 2
        # 每一步下完之后的可下位置
        self.grid_size = 35
        self.can_move = [np.ones((19, 19), dtype=np.int)]
        self.ko_state = [False]
        self.current_moves = [(-1, -1)]

    def add_board(self, board):
        self.boards.append(board)
        self.next_to_play = 3 - self.next_to_play

    def move(self, x, y):
        is_suicide, groups_captured, ko_state, forbidden = self.check_ko_or_move(x, y)
        legal = self.is_acceptable(x, y)
        if legal:
            if is_suicide and ko_state:
                self.ko_state.append(True)
            else:
                self.ko_state.append(False)
            self.current_moves.append((x, y))
            board_clone = self.boards[-1].board.copy()
            new_board = Board(board_clone)
            new_board.do_move(x=x, y=y, move_color=self.next_to_play)
            self.add_board(new_board)

            # update next_masks
            self.can_move.append(np.ones((19, 19), dtype=np.int))
            if is_suicide and ko_state:
                self.can_move[-1][forbidden[0]][forbidden[1]] = 0
        else:
            return

    def bind_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xx, yy = int(round(float(x) / self.grid_size)) - 1, int(round(float(y) / self.grid_size)) - 1
            self.move(xx, yy)
            if 'MCTS' in param:
                param['MCTS'] = False

        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.boards) >= 2:
                self.roll_back()

        if event == cv2.EVENT_MBUTTONDOWN:
            xx, yy = int(round(float(x) / self.grid_size)) - 1, int(round(float(y) / self.grid_size)) - 1
            self.move(xx, yy)
            if 'MCTS' in param:
                param['MCTS'] = True

    def check_ko_or_move(self, x, y):
        board_clone = self.boards[-1].board.copy()
        board_clone[x][y] = self.next_to_play
        group = Board.get_group(x, y, board_clone)
        forbidden = None
        is_suicide = not Board.check_liberty(group, board_clone)
        neighbor = Board.get_neighbor(x, y)
        groups_captured = 0
        ko_state = False
        # 如果该棋子为自杀行为 判定他是不是可以提子 或者 打劫
        if is_suicide:
            # 判断这步棋的周围棋子是否为不同颜色
            for cross in neighbor:
                if board_clone[cross[0]][cross[1]] == 3 - self.next_to_play:
                    # 如果为不同颜色 判断棋子死活
                    cross_group = Board.get_group(cross[0], cross[1], board_clone)
                    if not Board.check_liberty(cross_group, board_clone):
                        # 如果死棋 提子组加1
                        groups_captured += 1
                        # 如果是死棋 并且只有一个棋子 说明进行了打劫
                        if len(cross_group) == 1 and len(group) == 1:
                            ko_state = True
                            forbidden = cross_group[0]
        return is_suicide, groups_captured, ko_state, forbidden

    def is_acceptable(self, x=None, y=None):
        board = self.boards[-1]
        legal = self.can_move[-1]
        # 已经下过的棋子不能下
        legal[board.board == 1] = 0
        legal[board.board == 2] = 0
        if x is None:
            return legal
        else:
            is_suicide, groups_captured, ko_state, forbidden = self.check_ko_or_move(x, y)
            if is_suicide and groups_captured == 0:
                legal[x][y] = 0
            return 0 <= x <= 18 and 0 <= y <= 18 and legal[x][y] == 1

    def roll_back(self):
        self.boards.pop()
        self.can_move.pop()
        self.ko_state.pop()
        self.current_moves.pop()
        self.next_to_play = 3 - self.next_to_play

    def get_current_board_img(self, last_move=None):
        return self.boards[-1].show_board(grid_size=self.grid_size, last_move=last_move)

    def roll_back_human(self):
        for i in range(2):
            self.boards.pop()
            self.can_move.pop()
            self.ko_state.pop()
            self.current_moves.pop()
            self.next_to_play = 3 - self.next_to_play


class Board(object):
    def __init__(self, board=None):
        if board is not None:
            self.board = board
        else:
            self.board = np.zeros((19, 19), dtype=np.uint8)

    @staticmethod
    def mtx2str(mtx):
        string = np.array2string(mtx)
        string = string.replace('[', ' ')
        string = string.replace(']', ' ')
        return string

    # param grid_size:格子的大小 line_thickness:线条粗细 last_move:对于上一步进行特殊绘图
    def show_board(self, grid_size=35, line_thickness=2, last_move=None):
        canvas = np.ones((grid_size * 20, grid_size * 20, 3), dtype=np.uint8) * 100
        for x in range(1, 20):
            cv2.line(canvas, (grid_size, grid_size * x), (grid_size * 19, grid_size * x), (0, 0, 0),
                     thickness=line_thickness)
            cv2.line(canvas, (grid_size * x, grid_size), (grid_size * x, grid_size * 19), (0, 0, 0),
                     thickness=line_thickness)
        for i in range(3):
            for j in range(3):
                cv2.circle(canvas, (grid_size * 4 + grid_size * 6 * i, grid_size * 4 + grid_size * 6 * j),
                           radius=int(grid_size / 9), color=(0, 0, 0), thickness=line_thickness)
        for x in range(19):
            for y in range(19):
                if self.board[x][y] == 1:
                    cv2.circle(canvas, ((x + 1) * grid_size, (y + 1) * grid_size), int(grid_size / 2.2),
                               color=(0, 0, 0), thickness=-1)
                if self.board[x][y] == 2:
                    cv2.circle(canvas, ((x + 1) * grid_size, (y + 1) * grid_size), int(grid_size / 2.2),
                               color=(200, 200, 200), thickness=-1)
        if last_move is not None and 0 <= last_move[0] <= 18 and 0 <= last_move[1] <= 18:
            x, y = last_move[0], last_move[1]
            if self.board[x][y] == 1:
                cv2.circle(canvas, ((x + 1) * grid_size, (y + 1) * grid_size), int(grid_size / 3),
                           color=(200, 200, 200), thickness=2)
            if self.board[x][y] == 2:
                cv2.circle(canvas, ((x + 1) * grid_size, (y + 1) * grid_size), int(grid_size / 3), color=(0, 0, 0),
                           thickness=2)
        return canvas

    # 完成下棋的动作 并且进行提子
    def do_move(self, x, y, move_color):
        self.board[x][y] = move_color
        # 提子逻辑 判断这个子周围的子是否为死棋
        for cross in self.get_neighbor(x, y):
            cross_color = self.board[cross[0]][cross[1]]
            if cross_color != 0 and cross_color != move_color:
                group = self.get_group(cross[0], cross[1], self.board)
                if not self.check_liberty(group, self.board):
                    for pos in group:
                        self.board[pos[0]][pos[1]] = 0

    # 获取该棋子周围4个棋子 用于进行BFS 和 判断这个位置是否为死棋
    @staticmethod
    def get_neighbor(x, y):
        neighbor = []
        if x + 1 <= 18:
            neighbor.append((x + 1, y))
        if 0 <= x - 1:
            neighbor.append((x - 1, y))
        if 0 <= y - 1:
            neighbor.append((x, y - 1))
        if y + 1 <= 18:
            neighbor.append((x, y + 1))
        return neighbor

    # 获取以这个棋子为中心的这边棋
    @staticmethod
    def get_group(x, y, board, is_visited=None):
        group = list()
        color = board[x][y]
        group.append((x, y))
        visited = np.zeros((19, 19), np.uint8)
        visited[x][y] = 1
        for pos in group:
            for nei in Board.get_neighbor(*pos):
                if board[nei[0]][nei[1]] == color and visited[nei[0]][nei[1]] == 0:
                    group.append((nei[0], nei[1]))
                    visited[nei[0]][nei[1]] = 1
                    if is_visited is not None:
                        is_visited[pos] = 1
        return group

    # 判断一片棋的死活
    @staticmethod
    def check_liberty(group, board, cnt=False):
        # 如果cnt为true计算一片棋的气的数目
        if not cnt:
            for pos in group:
                for nei in Board.get_neighbor(*pos):
                    if board[nei[0]][nei[1]] == 0:
                        return True
            return False
        else:
            d = {}
            for pos in group:
                for cross in Board.get_neighbor(*pos):
                    if board[cross[0]][cross[1]] == 0:
                        d[cross] = 1
            return len(d)


if __name__ == '__main__':

    game = Game(handicap=0)

    # m = np.random.randint(0, 19, size=(2,), dtype=np.int)
    # m = tuple(m)
    # for num in range(100):
    #     for i in range(10000):
    #         game.mk_move(*m)
    #         if i % 300 == 0:
    #             game = Game()
    #     print(num)

    board_img = game.get_current_board_img()
    cv2.imshow('board_img', board_img)
    cv2.setMouseCallback('board_img', game.bind_click)
    while True:
        board_img = game.get_current_board_img()
        cv2.imshow('board_img', board_img)
        cv2.waitKey(33)
