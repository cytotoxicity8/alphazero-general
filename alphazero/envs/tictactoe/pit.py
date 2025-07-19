import pyximport

pyximport.install()

from alphazero.Arena import Arena
from alphazero.GenericPlayers import *
from alphazero.NNetWrapper import NNetWrapper as NNet
from alphazero.Coach import get_args
"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
def load_best_net(Game, args, folder, itter):
    nn1 = NNet(Game, args)
    #print(nn1)

    nn1.load_checkpoint(folder='./checkpoint/' + folder, 
                        filename=f'01-iteration-{itter:04d}.pkl')

    best_net = nn1.args.bestNet
    nn1.load_checkpoint(folder='./checkpoint/' + folder, 
                        filename=f'{best_net:02d}-iteration-{itter:04d}.pkl')

    return best_net, nn1

if __name__ == '__main__':
    from alphazero.envs.tictactoe.tictactoe import Game, display
    from alphazero.envs.tictactoe.train import args as notCompleteargs, argsss
    import random
    
    args = get_args(notCompleteargs)

    #args.numMCTSSims = 2000
    args._num_players = 2
    #args.arenaTemp = 0
    #args.startTemp = 0.5
    #args.add_root_noise = False
    #args.add_root_temp = False


    nn1 = NNet(Game, args)
    #print(nn1)

    nn1.load_checkpoint(folder='./checkpoint/tictacgraph_5_noDropout_norm_deeper_DFS_batch', 
                        filename=f'00-iteration-0002.pkl')

    nn2 = nn1

    player1 = MCTSPlayer(game_cls=Game, nn=nn1, args=args)#, verbose=True, print_policy=True)#, draw_mcts=True, draw_depth=1)
    #player2 = MCTSPlayer(game_cls=Game, nn=nn2, args=args, verbose=True, print_policy=True)
    player2 = RawMCTSPlayer(Game, args)
    players = [player1,player2]

    arena = Arena(players, Game, use_batched_mcts=False, args=args, display=display)

    #wins, draws, winrates = arena.play_games(128)#, verbose=True)
    wins, draws, winrates = arena.play_game(verbose=True)

    for i in range(len(wins)):
        print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    print('draws: ', draws)
