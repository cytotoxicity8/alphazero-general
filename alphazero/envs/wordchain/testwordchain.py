from torch import multiprocessing as mp 
import pyximport; pyximport.install()


import os
import sys
sys.path.append(os.path.abspath('/root/share/Real/KAIST/word_chain/alphazero-general'))
os.chdir('/root/share/Real/KAIST/word_chain/alphazero-general')

from alphazero.envs.wordchain.WordChainPlayers import HumanWordChainPlayer


from alphazero.Coach import Coach, get_args, ModeOfGameGen
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.wordchain.wordchain import Game
from alphazero.utils import dotdict

import pandas as pd

if __name__ == '__main__':
    game = Game()
    player1 = HumanWordChainPlayer(game)
    player2 = HumanWordChainPlayer(game)
    data = pd.read_csv('alphazero/envs/wordchain/data/20241112_전처리3db.csv')
    valid_data = data.copy()

    while True:
        print('Player 1 turn')
        game.play_action(player1.play(game, valid_data))
        if game.win_state()[0]:
            print('Player 1 won')
            break
        print('Player 2 turn')
        game.play_action(player2.play(game, valid_data))
        if game.win_state()[1]:
            print('Player 2 won')
            break
