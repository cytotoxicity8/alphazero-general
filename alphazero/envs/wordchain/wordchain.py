from typing import List, Tuple, Any

from alphazero.Game import GameState
from alphazero.envs.wordchain.WordChainLogic import Board

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import joblib
import pandas as pd

NUM_PLAYERS = 2
NUM_CHANNELS = 7
NUM_LETTERS = 488
ACTION_SIZE = NUM_LETTERS * 2
EDGE_INDEX: torch.Tensor = joblib.load('alphazero/envs/wordchain/data/initial_edge_index.pkl')
EDGE_SIZE = EDGE_INDEX.shape[1]
ATTR_SIZE = 1
NODE_INDEX = torch.eye(NUM_LETTERS)
#NODE_INDEX = np.eye(NUM_LETTERS)
#DATA = pd.read_csv('alphazero/envs/wordchain/data/20241112_전처리3db.csv')
#VALID_DATA = DATA.copy()


class Game(GameState):
    def __init__(self, _board=None):
        super().__init__(_board or self._get_board())
        self._board: Board

    @staticmethod
    def _get_board():
        return Board(NUM_LETTERS)

    def __eq__(self, other: 'Game') -> bool:
        return (
            self._board.matrix == other._board.matrix
            and self._board.head == other._board.head
            and self._player == other._player
            and self.turns == other.turns
        )

    def clone(self) -> 'Game':
        g = Game()
        g._board.matrix = np.copy(self._board.matrix)
        g._board.head = self._board.head
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
        return False

    @staticmethod
    def observation_size() -> Tuple[int, int, int, int, int]:
        return NUM_LETTERS+2, ACTION_SIZE, EDGE_SIZE, ATTR_SIZE, NUM_LETTERS #node dim, action dim, edge dim, attr dim, number of letters

    #@staticmethod
    #def get_edges() -> torch.Tensor:
    #    return EDGES

    #def _player_range(self):
    #    return (1, -1)[self.player]

    def valid_moves(self):
        return np.asarray(self._board.get_legal_moves(), dtype=np.uint8)

    def play_action(self, action: int) -> None:
        self._board.use_word(action)
        self._update_turn()

    def win_state(self):
        result = [False] * NUM_PLAYERS

        if self._board.is_lose():
            result = [True] * NUM_PLAYERS
            result[self.player] = False
        return np.array(result, dtype=np.uint8)

    def observation(self):
        """
        여러 가지 구현 방법이 있음. 일단 가장 간단한 형태인 1번 사용
        1. Graph의 node feature로 head, player 정보를 넣어주고 단순 GNN 
        2. Graph의 embedding을 구하고 head의 node embedding과 player의 정보로 value, policy
        3. NBFNet을 필두로 head에 대한 relative embedding을 구해서 policy 
        """
        #print("Creating x tensor")

        head_indicator = torch.zeros(NUM_LETTERS)
        head_indicator[self._board.head] = 1
        whoTurn = torch.full((NUM_LETTERS, 1), self.player)

        
        x = torch.cat([
            NODE_INDEX,  # (N, N), N=488(NUM_LETTERS)
            head_indicator.view(-1, 1),  # (N, 1)
            whoTurn  # (N, 1)
        ], dim=1)
        

        #head_indicator = np.zeros(NUM_LETTERS)
        #head_indicator[self._board.head] = 1
        #whoTurn = np.full((NUM_LETTERS, 1), self.player)
        
        """
        x = np.concatenate([ 
            NODE_INDEX,
            head_indicator.reshape(-1, 1),
            whoTurn
        ], axis=1)
        """
        
        

        #print("Creating edge tensor")

        edge_weight = self._board.edge_weight #shape: (E,)
        edge_index = EDGE_INDEX #fixed, shape: (2, E)
        edge_attr = edge_weight.view(-1, 1) #shape: (E, 1)

        #edge_weight이 0인 edge들 제거는 나중에 모델 fedding 직전: 이렇게 하는 이유는 Coach 처리방식이 거지 같기 때문

        #print("Creating graph")

        graph = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr = edge_attr)

        #print("Graph created")
        
        return graph

from alphazero.utils import dueum
ch2idx = joblib.load('alphazero/envs/wordchain/data/ch2idx.pkl')
idx2ch = {v: k for k, v in ch2idx.items()}
def display(game: Game, action=None):
    state = game._board
    
    if state.head is None:
        print(f"{idx2ch[action%NUM_LETTERS]}로 첫 글자 제시")
    elif action < NUM_LETTERS:
        print(f"{idx2ch[state.head]}로 시작하고 {idx2ch[action]}으로 끝")
    else:
        print(f"{dueum(idx2ch[state.head])}로 시작하고 {idx2ch[action%NUM_LETTERS]}으로 끝")
