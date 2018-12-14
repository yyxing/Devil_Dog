from game import Game, Board
from MCTS import MCTS

HANDICAP = 0


class GamePlay(object):
    def __init__(self, policy_net_path, value_net_path):
        self.mcts = MCTS(policy_net_path, value_net_path, time_limit=20)

    def play(self, game):
        self.mcts.set_game(game)
        return self.mcts.start()


def play():
    import cv2
    game_play = GamePlay(
        policy_net_path='./trained_models/policy',
        value_net_path='./trained_models/value')
    game = Game(handicap=HANDICAP)

    while True:
        board_img = game.get_current_board_img()
        cv2.imshow('board_img', board_img)
        param = {'MCTS': False}
        cv2.setMouseCallback('board_img', game.bind_click, param=param)
        cv2.waitKey(33)
        while True:
            before_len = len(game.boards)
            board_img = game.get_current_board_img(last_move=game.current_moves[-1])
            cv2.imshow('board_img', board_img)
            cv2.waitKey(33)
            now_len = len(game.boards)
            if now_len > before_len:
                board_img = game.get_current_board_img(last_move=game.current_moves[-1])
                cv2.imshow('board_img', board_img)
                cv2.waitKey(33)
                latest_board = game.boards[-2]  # board before human move
                next_to_play = game.next_to_play
                board_str = Board.mtx2str(latest_board.board)
                next_to_play = str(next_to_play)
                move, next_to_move, current_board, is_search = game.current_moves[-1], next_to_play, board_str, int(
                    param['MCTS'])
                if int(is_search) == 1:
                    game_play.mcts.time_limit = 20
                else:
                    game_play.mcts.time_limit = 0.5
                while Board.mtx2str(game.boards[-1].board) != current_board:
                    print('roll_back')
                    game.roll_back()
                x, y = move[0], move[1]
                game.move(x, y)

                output = game_play.play(game)
                game.move(output[0], output[1])
                print(output[0], output[1], game.next_to_play)


if __name__ == '__main__':
    play()
