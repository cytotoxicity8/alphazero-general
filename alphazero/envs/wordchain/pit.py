import pyximport

pyximport.install()
import os
import sys
sys.path.append(os.path.abspath('/root/share/Real/KAIST/word_chain/alphazero-general'))
os.chdir('/root/share/Real/KAIST/word_chain/alphazero-general')
from alphazero.Arena import Arena
from alphazero.GenericPlayers import *
from alphazero.envs.wordchain.WordChainPlayers import HumanWordChainPlayer
from alphazero.NNetWrapper import NNetWrapper as NNet
from alphazero.Coach import get_args
import pandas as pd
"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
def load_best_net(Game, args, folder, itter):
    nn1 = NNet(Game, args)
    #print(nn1)

    nn1.load_checkpoint(folder='./checkpoint/' + folder, 
                        filename=f'01-iteration-{itter:04d}.pkl')

    #best_net = nn1.args.bestNet
    #nn1.load_checkpoint(folder='./checkpoint/' + folder, 
    #                    filename=f'{best_net:02d}-iteration-{itter:04d}.pkl')

    return best_net, nn1

if __name__ == '__main__':
    from alphazero.envs.wordchain.wordchain import Game, display
    from alphazero.envs.wordchain.train import args as notCompleteargs, argsss
    import random
    
    args = get_args(notCompleteargs)

    args.numMCTSSims = 1000
    args._num_players = 2
    #args.use_head_embed = False
    #args.arenaTemp = 0
    #args.startTemp = 0.5
    #args.add_root_noise = False
    #args.add_root_temp = False


    nn1 = NNet(Game, args)
    #print(nn1)

    nn1.load_checkpoint(folder='./checkpoint/wordchain_custom1_usehead_adjustedelo', 
                        filename=f'00-iteration-0004.pkl')

    #nn2 = NNet(Game, args)
    #nn2.load_checkpoint(folder='./checkpoint/wordchain_custom1_usehead_adjustedelo', 
    #                    filename=f'00-iteration-0005.pkl')

    player1 = MCTSPlayer(game_cls=Game, nn=nn1, args=args, verbose=True)#, verbose=True, print_policy=True)#, draw_mcts=True, draw_depth=1)
    
    #player2 = MCTSPlayer(game_cls=Game, nn=nn2, args=args, verbose=True)#player2 = MCTSPlayer(game_cls=Game, nn=nn2, args=args, verbose=True, print_policy=True)
    #player2 = RawMCTSPlayer(Game, args)
    player2 = HumanWordChainPlayer()
    players = [player1,player2]

    arena = Arena(players, Game, use_batched_mcts=False, args=args, display=display)

    #wins, draws, winrates = arena.play_games(128)#, verbose=True)
    arena.play_game(verbose=True, valid_data=pd.read_csv('alphazero/envs/wordchain/data/20241112_전처리3db.csv'))
    #for i in range(len(wins)):
    #    print(f'player{i+1}:\n\twins: {wins[i]}')
        #print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    #print('draws: ', draws)


