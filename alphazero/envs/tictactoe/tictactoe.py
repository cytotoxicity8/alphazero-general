from typing import List, Tuple, Any

from alphazero.Game import GameState
from alphazero.envs.tictactoe.TicTacToeLogic import Board

import numpy as np
import torch
from torch_geometric.data import Data

NUM_PLAYERS = 2
NUM_CHANNELS = 7
BOARD_SIZE = 3
ACTION_SIZE = BOARD_SIZE ** 2
GRAPH = True
EDGES = []

if GRAPH:
    _dir  = [(1, 0), (0, -1), (-1, 0), (0, 1)]#[(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
    n     = BOARD_SIZE
    for j in range(n):
        for i in range(n):
            lst = [[(i+off[0], j+off[1]), (i+2*off[0], j+2*off[1]), (i-off[0], j-off[1])] for off in _dir]
            #connectedTo = ([pairs[0] for pairs in lst if sum(map(lambda pair : all(0 <= _ < BOARD_SIZE for _ in pair), pairs)) >= 2])
            connectedTo = ([pairs[0] for pairs in lst if all(0 <= _ < BOARD_SIZE for _ in pairs[0])])
            EDGES += [(j*n+i, j2*n+i2)for (i2,j2) in connectedTo]

    EDGES = torch.transpose(torch.tensor(EDGES),0,1).to(int)

OBSERVATION_SIZE = (NUM_CHANNELS, ACTION_SIZE, len(EDGES[0])) if GRAPH else (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)


class Game(GameState):
    def __init__(self, _board=None):
        super().__init__(_board or self._get_board())

    @staticmethod
    def _get_board():
        return Board(BOARD_SIZE)

    def __eq__(self, other: 'Game') -> bool:
        return (
            self._board.pieces == other._board.pieces
            and self._board.n == other._board.n
            and self._player == other._player
            and self.turns == other.turns
        )

    def clone(self) -> 'Game':
        g = Game()
        g._board.pieces = np.copy(self._board.pieces)
        g._player = self._player
        g._turns = self.turns
        return g

    @staticmethod
    def num_players() -> int:
        return NUM_PLAYERS

    @staticmethod
    def action_size() -> int:
        return ACTION_SIZE

    @staticmethod
    def has_draw() -> bool:
        return True

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return OBSERVATION_SIZE

    @staticmethod
    def get_edges() -> torch.Tensor:
        return EDGES

    def _player_range(self):
        return (1, -1)[self.player]

    def valid_moves(self):
        # return a fixed size binary vector
        valids = [0] * self.action_size()

        for x, y in self._board.get_legal_moves():
            valids[self._board.n * x + y] = 1

        return np.array(valids, dtype=np.uint8)

    def play_action(self, action: int) -> None:
        move = (action // self._board.n, action % self._board.n)
        self._board.execute_move(move, self._player_range())
        self._update_turn()

    def win_state(self):
        result = [False] * (NUM_PLAYERS + 1)
        player = self._player_range()

        if self._board.is_win(player):
            result[self.player] = True
        elif self._board.is_win(-player):
            result[self._next_player(self.player)] = True
        elif not self._board.has_legal_moves():
            result[-1] = True

        return np.array(result, dtype=np.uint8)

    def observation(self):
        if not GRAPH:
            return np.expand_dims(np.asarray(self._board.pieces), axis=0).astype(np.float32)
        else:
            nodes   = np.asarray(self._board.pieces).ravel() + 1
            nodes   = np.eye(3)[nodes]
            whoTurn = np.full((BOARD_SIZE**2,1), 2*self._player - 1)

            zeroAway = np.array([[0],[0],[0],[0],[1],[0],[0],[0],[0]])
            oneAway  = np.array([[0],[1],[0],[1],[0],[1],[0],[1],[0]])
            twoAway  = np.array([[1],[0],[1],[0],[0],[0],[1],[0],[1]])

            #middle  = np.zeros((BOARD_SIZE**2,1))
            #middle[len(middle)//2] = 1
            #middle[0], middle[2],middle[6],middle[8] = -1,-1,-1,-1
            nodes   = np.concatenate((nodes,whoTurn, zeroAway,oneAway,twoAway), -1)
            #print(nodes)
            
            ret = Data(torch.from_numpy(nodes), edge_index=EDGES)#.view(-1,1), edge_index=EDGES)
            #print(ret)
            return ret



    def symmetries(self, pi: np.ndarray, winstate) -> List[Tuple[Any, int]]:
        # mirror, rotational
        assert (len(pi) == self._board.n ** 2)

        pi_board = np.reshape(pi, (self._board.n, self._board.n))
        result = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(np.asarray(self._board.pieces), i)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)

                gs = self.clone()
                gs._board.pieces = new_b
                result.append((gs, new_pi.ravel(), winstate))

        return result


def display(game, action=None):
    board = np.array(game._board.pieces)
    n = board.shape[0]

    print("   ", end="")
    for y in range(n):
        print(y, "", end="")
    print("")
    print("  ", end="")
    for _ in range(n):
        print("-", end="-")
    print("--")
    for y in range(n):
        print(y, "|", end="")  # print the row #
        for x in range(n):
            piece = board[y][x]  # get the piece to print
            if piece == -1:
                print("X ", end="")
            elif piece == 1:
                print("O ", end="")
            else:
                if x == n:
                    print("-", end="")
                else:
                    print("- ", end="")
        print("|")

    print("  ", end="")
    for _ in range(n):
        print("-", end="-")
    print("--")
