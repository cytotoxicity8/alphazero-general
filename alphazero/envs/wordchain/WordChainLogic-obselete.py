import numpy as np
import joblib
import numpy as np
import random
import pandas as pd
from alphazero.utils import dueum

class Board():    
    dueum_dict : dict = joblib.load('alphazero/envs/wordchain/data/20241113_dueum_dict.pkl')

    def __init__(self, matrix: np.ndarray=None, head: int=None):
        """Set up initial configuration."""
        self.matrix = matrix
        self.head = head
        if self.matrix is None:
            self.matrix = construct_initial_mat()
        #if self.head is None: #초기 상태
        #    self.head = random.randint(0, self.matrix.shape[0]-1) #아무 글자 제시 (끄투처럼)
        self.num_letters = self.matrix.shape[0]

    
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
            if action < self.num_letters:
                newMatrix[self.head, action % self.num_letters] = self.matrix[self.head, action % self.num_letters] - 1
            else:
                if Board.dueum_dict.get(self.head) is not None:
                    newMatrix[Board.dueum_dict[self.head], action % self.num_letters] = self.matrix[Board.dueum_dict[self.head], action % self.num_letters] - 1
                else:
                    raise ValueError(f'action이 self.num_letter 이상이면 두음법칙을 사용해야 함')

        elif self.head is None: #head가 없는 첫 턴에는 그냥 글자를 제시함
            self.head = action % self.num_letters 
            #random_head_list = np.nonzero(self.matrix[:, action % self.num_letters])[0] #action이 끝말로 나오는 글자 중 아무거나 (self.matrix[random_head, action % self.num_letters]이 0이 아니도록)
            #random_head = np.random.choice(random_head_list)
            #newMatrix[random_head, action % self.num_letters] = self.matrix[random_head, action % self.num_letters] - 1
        
        self.matrix = newMatrix
        self.head = action % self.num_letters #글자 변경


def construct_initial_mat():
    """
    두음법칙 고려 없이 순수한 인접행렬 생성
    """
    data = pd.read_csv("alphazero/envs/wordchain/data/20241113_scc.csv")

    head_list = data["앞말"].unique()
    tail_list = data["끝말"].unique()
    ch_list = np.union1d(head_list, tail_list)

    idx2ch = list(ch_list)
    for ch in idx2ch:
        if (dueum(ch) not in idx2ch):
            idx2ch.append(dueum(ch))
            
    ch2idx = dict()
    for idx, ch in enumerate(idx2ch):
        ch2idx[ch] = idx

    adj_mat = np.zeros((len(idx2ch), len(idx2ch)))

    for _, row in data.iterrows():
        head_idx = ch2idx[row['앞말']]
        tail_idx = ch2idx[row['끝말']]
        adj_mat[head_idx][tail_idx] += 1

    #참고
    '''
    for idx in range(len(idx2ch)):
        didx = ch2idx[utils.dueum(idx2ch[idx])]
        if idx != didx:
            dueum_list.append((idx, didx))
            dueum_dict[idx] = didx
    '''

    return adj_mat

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
