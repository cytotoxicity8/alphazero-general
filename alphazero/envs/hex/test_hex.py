"""
To run tests:
pytest-3 connect4
"""
import pyximport; pyximport.install()
from collections import namedtuple
import textwrap
import numpy as np
import cProfile

from .hex import Game

# Tuple of (Board, Player, Game) to simplify testing.
BPGTuple = namedtuple('BPGTuple', 'board player game')


def init_board_from_moves(moves):
    """Returns a BPGTuple based on series of specified moved."""
    game = Game()
    for move in moves:
        game.play_action(move)
    return game

def test_simple_moves():
    g = Game()
    for move in [1,49]:#,2,3,4,5]:
        print(g.observation())
        g.play_action(move)
        print(g._board)

def test_won_BLUE():
    b_moves = list(range(0,7))
    r_moves = [10+b_move for b_move in b_moves]
    all_moves = b_moves + r_moves
    all_moves[::2]  = b_moves
    all_moves[1::2] = r_moves 

    g = init_board_from_moves(all_moves)
    print(g._board)
    assert (g.win_state() == [True, False, False])

def test_won_RED():
    r_moves = list(7*i for i in range(0,7))
    b_moves = [1+r_move for r_move in r_moves]
    all_moves = b_moves + r_moves
    all_moves[::2]  = b_moves
    all_moves[1::2] = r_moves 

    g = init_board_from_moves(all_moves)
    print(g._board)
    assert (g.win_state() == [False, True, False])

def test_symmetries():
    hist = []
    game = Game()
    while not(any(game.win_state())):
        obs  = game.observation()
        move = np.random.choice(np.flatnonzero(game.valid_moves()))
        hist.append(move)
        game.play_action(move)

    for symm, pii in syms:
        print(symm._board)
        #print("----pi-----")
        #print(np.resize(pii, (7,7)))
        #print("----rank-----")
        #print(np.resize(symm._board.rank, (9,9)))
        #print("----connection-----")
        #print(np.resize(symm._board.connectionBoard, (9,9)))

        symm.play_action(np.where(pii ==1)[0][0])
        print(symm.win_state())

def test_patterns():
    b_moves = [7, 16]#[0, 8]
    r_moves = [36]
    all_moves = b_moves + r_moves
    all_moves[::2]  = b_moves
    all_moves[1::2] = r_moves 

    g = init_board_from_moves(all_moves)
    print(g._board)

def testEfficiency():
    n = 10000
    for i in range(0,n):
        game = Game()
        while not(any(game.win_state())):
            move = np.random.choice(np.flatnonzero(game.valid_moves()))
            obs  = game.observation()
            game.play_action(move)
        if i%100 == 0:
            print('.', end='')
    print()




if __name__ == "__main__":
    g = init_board_from_moves([1, 3, 11])
    print(g.observation())
    print(g._board)
    #print(np.resize(g._board.connectionBoard, (11,11)))
    #cProfile.runctx("testEfficiency()", globals(), locals(), "hexBoard_10000_random4.prof")
