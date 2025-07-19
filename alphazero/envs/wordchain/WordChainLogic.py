import numpy as np
import joblib
import numpy as np
import random
import pandas as pd
from alphazero.utils import dueum
import torch

class Board():    
    dueum_dict : dict = joblib.load('alphazero/envs/wordchain/data/20241113_dueum_dict.pkl')
    edge_dict : dict = joblib.load('alphazero/envs/wordchain/data/edge_dict.pkl')
    initial_matrix : dict = joblib.load('alphazero/envs/wordchain/data/initial_matrix.pkl')
    def __init__(self, num_letters: int, head: int=None):
        """Set up initial configuration."""
        self.head = head
        self.num_letters = num_letters
        self.matrix = Board.initial_matrix.copy()
        self.edge_weight: torch.Tensor = joblib.load('alphazero/envs/wordchain/data/initial_edge_weight.pkl')

    
    def get_legal_moves(self):
        """
        self.head = None (초기 상태): 아무 글자 사용 가능
        아니면, 0~num_letters-1일 땐 두음 적용 없이 사용 가능한 단어, 그 이상일 땐 두음 적용 후 사용 가능한 단어
        """

        if self.head is None: #처음엔 아무 글자 제시 ()
            return np.concatenate((np.ones((self.num_letters), dtype=np.intc), np.zeros((self.num_letters), dtype=np.intc)))  
        
        valid = np.zeros((self.num_letters * 2), dtype=np.intc)
        valid[self.matrix[self.head].nonzero()[0]] = 1
        if Board.dueum_dict.get(self.head) is not None:
            valid[self.matrix[Board.dueum_dict[self.head]].nonzero()[0] + self.num_letters] = 1

        return valid

    def has_legal_moves(self):
        return (sum(self.get_legal_moves()) != 0)

    def is_lose(self) -> bool:
        """
        현재 플레이어가 패배당했는지 판단
        """
        if self.has_legal_moves():
            return False
        else:
            return True

    def use_word(self, action:int):
        """
        self.head가 None이면 초기 상태, 아무 글자 제시 (단어를 쓰진 않음)

        0~num_letters-1 -> 두음 미적용된 채로 그대로 사용,  ex: 름육, 가정 -> 름육을 쓰면 (름, 육) edge 사용
        num_letters~ -> 두음 적용 ex: (름을 받았지만) 늠육 -> 늠육을 쓰면 (늠, 육) edge를 사용
        """
        newMatrix = self.matrix.copy()

        if self.head is not None:
            #print(self.head, action)
            if action < self.num_letters:
                newMatrix[self.head, action % self.num_letters] = self.matrix[self.head, action % self.num_letters] - 1
                self.edge_weight[Board.edge_dict[(self.head, action % self.num_letters)]] -= 1
                assert newMatrix[self.head, action % self.num_letters] >= 0
                #print(self.edge_weight[Board.edge_dict[(self.head, action % self.num_letters)]])
                assert self.edge_weight[Board.edge_dict[(self.head, action % self.num_letters)]] >= 0
            else:
                if Board.dueum_dict.get(self.head) is not None:
                    newMatrix[Board.dueum_dict[self.head], action % self.num_letters] = self.matrix[Board.dueum_dict[self.head], action % self.num_letters] - 1
                    self.edge_weight[Board.edge_dict[(Board.dueum_dict[self.head], action % self.num_letters)]] -= 1
                    assert newMatrix[Board.dueum_dict[self.head], action % self.num_letters] >= 0
                    assert self.edge_weight[Board.edge_dict[(Board.dueum_dict[self.head], action % self.num_letters)]] >= 0
                else:
                    raise ValueError(f'action이 self.num_letter 이상이면 두음법칙을 사용해야 함')

        elif self.head is None: #head가 없는 첫 턴에는 그냥 글자를 제시함
            self.head = action % self.num_letters 
            #random_head_list = np.nonzero(self.matrix[:, action % self.num_letters])[0] #action이 끝말로 나오는 글자 중 아무거나 (self.matrix[random_head, action % self.num_letters]이 0이 아니도록)
            #random_head = np.random.choice(random_head_list)
            #newMatrix[random_head, action % self.num_letters] = self.matrix[random_head, action % self.num_letters] - 1
        
        self.matrix = newMatrix
        self.head = action % self.num_letters #글자 변경



class tmpBoard():


    def __init__(self, n=3, _pieces=None):
        """Set up initial board configuration."""

        self.n = n

        if _pieces is not None:
            self.pieces = _pieces
        else:
            # Create the empty board array.
            self.pieces = [None] * self.n
            for i in range(self.n):
                self.pieces[i] = [0] * self.n

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        """
        moves = []  # stores the legal moves.

        # Get all the empty squares (color==0)
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    newmove = (x, y)
                    moves.append(newmove)
        return moves

    def has_legal_moves(self):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    return True
        return False

    def is_win(self, color):
        """Check whether the given player has collected a triplet in any direction; 
        @param color (1=white,-1=black)
        """
        win = self.n
        # check y-strips
        for y in range(self.n):
            count = 0
            for x in range(self.n):
                if self[x][y] == color:
                    count += 1
                if count == win:
                    return True

        # check x-strips
        for x in range(self.n):
            count = 0
            for y in range(self.n):
                if self[x][y] == color:
                    count += 1
                if count == win:
                    return True

        # check two diagonal strips
        count = 0
        for d in range(self.n):
            if self[d][d] == color:
                count += 1
            if count == win:
                return True

        count = 0
        for d in range(self.n):
            if self[d][self.n - d - 1] == color:
                count += 1
            if count == win:
                return True

        return False

    def execute_move(self, move, color):
        """Perform the given move on the board; 
        color gives the color pf the piece to play (1=white,-1=black)
        """
        (x, y) = move

        # Add the piece to the empty square.
        assert self[x][y] == 0
        self[x][y] = color
