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
if __name__ == '__main__':
    from alphazero.envs.hex.hex import Game, display
    from alphazero.envs.hex.players import HumanHexPlayer, GTPPlayer
    from alphazero.envs.hex.train import args as notCompleteargs
    import random
    args = get_args(notCompleteargs)

    args.numMCTSSims = 2000
    args.arena_batch_size = 64
    args.temp_scaling_fn = lambda x, y, z: 0
    args.add_root_noise = False
    args.add_root_temp = False
    #print(args)
    # all players
    # rp = RandomPlayer(g).play
    # gp = OneStepLookaheadConnect4Player(g).play
    #player2 = HumanHexPlayer()
    #player1 = 
    # nnet players
    nn1 = NNet(Game, args)
    nn1.load_checkpoint('./checkpoint/hex_7x7_observing_7x7x1_Canonical_NoSwap', 'iteration-0019.pkl')
    #nn2 = nn1
    nn2 = NNet(Game, args)
    nn2.load_checkpoint('./checkpoint/hex_7x7_observing_7x7x1_Canonical_NoSwap', 'iteration-0026.pkl')
    #player1 = nn1
    #player2 = nn1.process

    #player1 = NNPlayer(nn=nn1, game_cls=Game,  args=args, verbose=True)
    #player2 = NNPlayer(nn=nn2, game_cls=Game,  args=args, verbose=True)
    player1 = MCTSPlayer(game_cls=Game, nn=nn1, args=args, verbose=True)#, draw_mcts=True, draw_depth=3)
    #args2 = args.copy()
    #args2.numMCTSSims = 10
    player2 = MCTSPlayer(game_cls=Game, nn=nn2, args=args, verbose=True)
    #player2 = RandomPlayer()
    #player2 = RawMCTSPlayer(Game, args)
    #player2 = GTPPlayer()
    players = [player2,player1]
    arena = Arena(players, Game, use_batched_mcts=False, args=args, display=display)

    
    wins, draws, winrates = arena.play_games(1, verbose=True)
    for i in range(len(wins)):
        print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    print('draws: ', draws)
   
    
    #can_process = test_player.supports_process()
    #nnplayer = MCTSPlayer(nn2, Game, args)

    """
    print('PITTING 1 AGAINST 2: ')
    player1.reset()
    game1 = Game()
    game1.play_action(45)
    game1.play_action(46)
    game1.play_action(0)

    game1.play_action(1)

    game2 = game1.clone()
    game2._board.flipInXPlusY()
    game2._player = 1-game2._player

    print("---game1---")
    print(game1._board)
    #print(game1.observation())
    #print("---game2---")
    #print(game2._board)
    #print(game2.observation())

    
    #print(player1.mtcsargs)
    play = (player1.play(game1))
    game1.play_action(play)
    print(game1._board)
    """
    """
    players = [player1] + [player2]
    arena = Arena(players, Game, use_batched_mcts=True, args=args)
    wins, draws, winrates = arena.play_games(256)
    winrate = winrates[0]

    print(f'NEW/BASELINE WINS : {wins[0]} / {sum(wins[1:])} ; DRAWS : {draws}\n')
    print(f'NEW MODEL WINRATE : {round(winrate, 3)}')
    #self.writer.add_scalar('win_rate/baseline', winrate, iteration)
    """