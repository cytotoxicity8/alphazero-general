"""
To run tests:
pytest-3 connect4
"""
import pyximport; pyximport.install()
from collections import namedtuple
import textwrap
import numpy as np
import cProfile

from .hex2 import Game, BOARD_SIZE

# Tuple of (Board, Player, Game) to simplify testing.
BPGTuple = namedtuple('BPGTuple', 'board player game')


import torch_geometric as geo_torch
import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(data):
    g = geo_torch.utils.to_networkx(data, node_attrs = ['x'], to_undirected=True)
    lables = nx.get_node_attributes(g, 'x')
    pos = {i:(i%5, i//5) for i in range(5*5+1)}
    nx.draw_networkx(g, labels=lables, pos = pos)
    plt.show()

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


    a = ["d6",
        "g6",
        "f5",
        "h4",
        "g3",
        "i2",
        "i1",
        "h2",
        "h1",
        "g2",
        "g1",
        "f2",
        "f1",
        "e2",
        "e1",
        "c2",
        "d2",
        "c3",
        "d3",
        "c4",
        "d4",
        "c5",
        "d1",
        "c1",
        "b7",
        "d5"]

    b = [(ord(ac[0])-ord("a"), int(ac[1])-1) for ac in a]
    b = [bc[0]+bc[1]*9 for bc in b]

    g = Game()


    #print(g.observation())

    #g = Game()
    #d = (g.observation())

    #print(g.valid_moves())
    #draw_graph(d)
    #print(d.is_undirected())
    #print(d.x)
    #print(d.edge_index)

    """for i in range(100):
        g = Game()
        while not g.win_state().any():
            t  =  np.where(g.valid_moves() == 1)[0]
            mv = np.random.choice(t)
            g.play_action(mv)
            print(g._board)
            g.observation()
        print(i)"""
    
    from alphazero.GenericPlayers import *
    from alphazero.NNetWrapper import NNetWrapper as NNet
    from alphazero.Coach import get_args
    from alphazero.envs.hex.train import args as notCompleteargs, hexxyAgs
    from alphazero.envs.hex.hex2 import Game, display

    args = get_args(notCompleteargs)
    #args.startTemp = 0
    #args.add_root_noise = False
    #args.add_root_temp = False
    args.numMCTSSims = 20
    #args.mctsCanonicalStates = False
    print(args.startTemp)
    nn1 = NNet(Game, args)

    nn1.load_checkpoint(folder=args.checkpoint + '/ahex_9x9_observing_9x9x8_NeuroHex_Virtual_Bridge_and_Ziguarat', 

                        filename='08-iteration-0060.pkl')

    print(nn1.args.bestNet)
    player1 = MCTSPlayer(game_cls=Game, nn=nn1, args=args, verbose=True, print_policy=True)#, draw_mcts=True, draw_depth=1)

    for mv in b:
        player1.update(g, mv)
        g.play_action(mv)

    print(g._board)
    print(player1.play(g))
    
    #print(g.observation())
    #print(g._board)
    #print(np.resize(g._board.connectionBoard, (11,11)))
    #cProfile.runctx("testEfficiency()", globals(), locals(), "hexBoard_10000_random4.prof")



"""
[0.0021098  0.00325031 0.00342755 0.00547886 0.00564127 0.00887101
 0.02091861 0.00226414 0.00307323 0.006929   0.01006659 0.0135122
 0.05171164 0.01163948 0.00241637 0.00548121 0.01784245 0.02916932
 0.04498595 0.08159672 0.00742292 0.00322123 0.0107792  0.
 0.         0.0598687  0.00867646 0.00403627 0.01240392 0.07790414
 0.         0.29749337 0.         0.00383054 0.00344656 0.
 0.         0.         0.         0.         0.00396361 0.00394076
 0.         0.         0.         0.12916456 0.03472131 0.0040482
 0.00320902 0.        ]

[0.00089167 0.00151423 0.00162159 0.00316691 0.00380358 0.00532709
 0.01979924 0.00113379 0.00186562 0.00724956 0.00817175 0.01324032
 0.0665807  0.01115882 0.00100853 0.00588695 0.01985543 0.05432891
 0.08666047 0.19309035 0.00493689 0.00185185 0.01425934 0.
 0.         0.05248431 0.00684829 0.0019385  0.00644819 0.03873497
 0.         0.24479473 0.         0.00302545 0.00114787 0.
 0.         0.         0.         0.         0.00190079 0.00162635
 0.         0.         0.         0.0837749  0.02461061 0.00221376
 0.00135902 0.        ]

[0.00027393 0.00058556 0.00082613 0.00193366 0.00232147 0.00370782
 0.0206877  0.00037245 0.00101464 0.00774926 0.01254152 0.00950818
 0.11150982 0.00718362 0.00035562 0.00431065 0.03172421 0.14889061
 0.10421783 0.13112399 0.00176712 0.00068679 0.01846247 0.
 0.         0.03056548 0.00564356 0.00078366 0.00484371 0.05445246
 0.         0.16715796 0.         0.00183554 0.00059015 0.
 0.         0.         0.         0.         0.00129098 0.00092395
 0.         0.         0.         0.07788321 0.02885596 0.00154884
 0.0006365  0.        ]"""