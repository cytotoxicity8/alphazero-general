import pyximport

pyximport.install()

from alphazero.Arena import Arena
from alphazero.GenericPlayers import *
from alphazero.NNetWrapper import NNetWrapper as NNet
from alphazero.Coach import get_args
from datetime import datetime
"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

def load_best_net(Game, args, folder, itter):
    nn1 = NNet(Game, args)
    #print(nn1)

    nn1.load_checkpoint(folder= args.checkpoint + '/' + folder, 
                        filename=f'00-iteration-{itter:04d}.pkl')

    best_net = nn1.args.bestNet
    nn1.load_checkpoint(folder=args.checkpoint + '/' + folder, 
                        filename=f'{best_net:02d}-iteration-{itter:04d}.pkl')

    return best_net, nn1

def playGame(moves, games, players, VERBOSE = False) :
    [_.reset() for _ in players]
    moves = list(moves.copy())
    moves.reverse()

    #print(moves)

    gs = [game() for game in games]

    while not gs[0].win_state().any():
        p = gs[0]._player

        if moves == []:
            mv = players[p].play(gs[p])
        else:
            mv = moves.pop()
            
        for i in range(0, len(players)):
            #print()
            if (gs[i]._player != 0) and i != p and ((players[p].args != None and players[p].args.mctsCanonicalStates) ^ (players[i].args != None and players[i].args.mctsCanonicalStates)):
                    #print("flipping")
                    a, b = (mv % 7, mv// 7)
                    a, b = 6-b, 6 -a
                    players[i].update(gs[i], b*7 + a)
                    gs[i].play_action(b*7 + a)
            else:
                players[i].update(gs[i], mv)
                gs[i].play_action(mv)

        if VERBOSE:
            print(gs[0]._board)

    for i in range(0, len(players)):
        if gs[0].win_state()[i] == 1:
            return i

def genRandomMoves(game, num_moves):
    return np.random.choice(np.arange(game.action_size()-1), size=(num_moves), replace=False)


if __name__ == '__main__':
    from alphazero.envs.hex.hex  import Game as Game1, display as display1
    #from alphazero.envs.hex.hex2 import Game as Game2, display as display2
    Game2 = Game1
    #Game1 = Game2
    from alphazero.envs.hex.players import HumanHexPlayer, GTPPlayer
    from alphazero.envs.hex.train import args as notCompleteargs, hexxyAgs, hexxyAgs2
    import random
    args = get_args(notCompleteargs)

    args.numMCTSSims = 40
    args.arena_batch_size = 64
    args.startTemp = 0
    args.add_root_noise = False
    args.add_root_temp  = False

    args1 = args.copy()
    args2 = args.copy()
    #args2.mctsCanonicalStates = True
    #print(args)
    # all players
    # rp = RandomPlayer(g).play
    # gp = OneStepLookaheadConnect4Player(g).play
    #player2 = HumanHeqxPlayer()
    #player1 = 
    # nnet players
    Itter = 50

    #nn1 = NNet(Game1, args1)
    #nn1.load_checkpoint(folder='./checkpoint/ahex_9x9_observing_9x9x4_NoPop', 
    #                    filename=f'iteration-{Itter:04d}.pkl')

    #print([nn1.args[i] for i in ["fpu_reduction","root_noise_frac","cpuct","lr","value_loss_weight"]], ",")

    #assert False
    #print(nn1.args)
    #assert 1== 0


    #nn2 = nn1
    #player1 = nn1
    #player2 = nn1.process
    
    #print(nn1.args.bestNet)
    #assert 1 == 0
    
    #player1 = NNPlayer(nn=nn1, game_cls=Game,  args=args, verbose=True)
    #player2 = NNPlayer(nn=nn2, game_cls=Game,  args=args, verbose=True)
    #args2 = args.copy()
    #args2.numMCTSSims = 10
    #player2 = HumanHexPlayer(args2)
    #player2 = RandomPlayer()
    #player2 = RawMCTSPlayer(Game1, args1, verbose=False)

    
    for MODE, args.startTemp, args.add_root_temp in [("STRAIGHTCOMPETITION", 0 , False), ("RANDOMSTART",0,False)]:
        print("----------")
        args.numMCTSSims = 200
        print(MODE, args.startTemp)
        VERBOSE = True
        NUM_GAMES = 1
        for Itter in range(35,36,5):
            best_net1, nn1 = load_best_net(Game1, args, "ahex_9x9_observing_9x9x8_NeuroHex_Virtual_Bridge_and_Ziguarat", Itter)
            #best_net2, nn2 = load_best_net(Game2, args, "ahex_9x9_observing_9x9x4", Itter)
            #print(f"For nn1 we have {best_net1} and for nn2 we have {best_net2}")
            player1 = NNPlayer(game_cls=Game1, nn=nn1, args=args)#, verbose=True, print_policy=True)#, draw_mcts=True, draw_depth=1)
            #player2 = NNPlayer(game_cls=Game2, nn=nn2, args=args)#, verbose=True, print_policy=True)

            player2 = GTPPlayer()
            wins  = [0,0,0]
            wins1 = [0,0,0]
            wins2 = [0,0,0]
            if MODE == "RANDOMSTART":
                while sum(wins1) + sum(wins2) < NUM_GAMES:
                    moves = genRandomMoves(Game1, 8)
                    try:
                        won1Way = playGame(moves, [Game1, Game2], [player1,player2],VERBOSE)
                        #print(moves)
                        won2Way = playGame(moves, [Game2, Game1], [player2,player1],VERBOSE)
                    except:
                        continue;
                    if won1Way != won2Way:
                        wins1[won1Way+1] +=1
                print(f"{datetime.now()} - wins {wins1} for {Itter}")
            elif MODE == "STRAIGHTCOMPETITION":
                while sum(wins1) + sum(wins2) < NUM_GAMES:
                    won1Way = playGame([], [Game1, Game2], [player1,player2],VERBOSE)
                    #print(moves)
                    won2Way = playGame([], [Game2, Game1], [player2,player1],VERBOSE)

                    wins1[won1Way+1] += 1
                    wins2[2-won2Way] += 1

                    #print(wins1, wins2)

                print(f"{datetime.now()} - wins going first {wins1} wins going second {wins2} for {Itter}")

    
    
    """
    print("----now MCTS-----")
    for Itter in range(95,101,5):
        VERBOSE = False
        NUM_GAMES = 256
        for MODE, args.startTemp, args.add_root_temp in [("RANDOMSTART", 0 , False)]:#, ("RANDOMSTART",0,False)]:
            best_net1, nn1 = load_best_net(Game1, args, "ahex_9x9_observing_9x9x4", Itter)
            best_net2, nn2 = load_best_net(Game2, args, "ahex_9x9_observing_9x9x4", 50)
            #print(f"For nn1 we have {best_net1} and for nn2 we have {best_net2}")
            player1 = MCTSPlayer(game_cls=Game1, nn=nn1, args=args)#, verbose=True, print_policy=True)#, draw_mcts=True, draw_depth=1)
            player2 = MCTSPlayer(game_cls=Game2, nn=nn2, args=args)#, verbose=True, print_policy=True)

            wins1 = [0,0,0]
            wins2 = [0,0,0]
            if MODE == "RANDOMSTART":
                if Itter == 50:
                    continue
                while sum(wins1) + sum(wins2) < NUM_GAMES:
                    moves = genRandomMoves(Game1, 8)
                    try:
                        won1Way = playGame(moves, [Game1, Game2], [player1,player2],VERBOSE)
                        #print(moves)
                        won2Way = playGame(moves, [Game2, Game1], [player2,player1],VERBOSE)
                    except:
                        continue;
                    if won1Way != won2Way:
                        wins1[won1Way+1] +=1
                print(f"{datetime.now()} - wins {wins1} for {Itter}")
            elif MODE == "STRAIGHTCOMPETITION":
                while sum(wins1) + sum(wins2) < NUM_GAMES:
                    won1Way = playGame([], [Game1, Game2], [player1,player2],VERBOSE)
                    #print(moves)
                    won2Way = playGame([], [Game2, Game1], [player2,player1],VERBOSE)

                    wins1[won1Way+1] += 1
                    wins2[2-won2Way] += 1


                print(f"{datetime.now()} - wins going first {wins1} wins going second {wins2} for {Itter}")
    """
    """
    print("----now bigger MCTS-----")
    args.numMCTSSims = 200
    for Itter in range(10,101,5):
        VERBOSE = False
        NUM_GAMES = 256
        for MODE, args.startTemp, args.add_root_temp in [("RANDOMSTART",0,False)]: #("STRAIGHTCOMPETITION", 1 , True)]:#, ]:
            best_net1, nn1 = load_best_net(Game1, args, "ahex_9x9_observing_9x9x4", Itter)
            best_net2, nn2 = load_best_net(Game2, args, "ahex_9x9_observing_9x9x4", 50)
            #print(f"For nn1 we have {best_net1} and for nn2 we have {best_net2}")
            player1 = MCTSPlayer(game_cls=Game1, nn=nn1, args=args)#, verbose=True, print_policy=True)#, draw_mcts=True, draw_depth=1)
            player2 = MCTSPlayer(game_cls=Game2, nn=nn2, args=args)#, verbose=True, print_policy=True)

            wins1 = [0,0,0]
            wins2 = [0,0,0]
            if MODE == "RANDOMSTART":
                if Itter == 50:
                    continue
                while sum(wins1) + sum(wins2) < NUM_GAMES:
                    moves = genRandomMoves(Game1, 8)
                    try:
                        won1Way = playGame(moves, [Game1, Game2], [player1,player2],VERBOSE)
                        #print(moves)
                        won2Way = playGame(moves, [Game2, Game1], [player2,player1],VERBOSE)
                    except:
                        continue;
                    if won1Way != won2Way:
                        wins1[won1Way+1] +=1
                print(f"{datetime.now()} - wins {wins1} for {Itter}")
            elif MODE == "STRAIGHTCOMPETITION":
                while sum(wins1) + sum(wins2) < NUM_GAMES:
                    won1Way = playGame([], [Game1, Game2], [player1,player2],VERBOSE)
                    #print(moves)
                    won2Way = playGame([], [Game2, Game1], [player2,player1],VERBOSE)

                    wins1[won1Way+1] += 1
                    wins2[2-won2Way] += 1

                    #print(wins1,wins2)

                print(f"{datetime.now()} - wins going first {wins1} wins going second {wins2} for {Itter}")
    
    """
    """
    VERBOSE   = False
    pOrder = [1,2]
    for _ in range(NUM_GAMES):
        pOrder.reverse()
        if VERBOSE:
            if pOrder[0] == 1:
                print("Player 1 Starting")
            else:
                print("Player 2 Starting")
        g1 = Game1()
        g2 = Game2()
        player1.reset()
        player2.reset()

        #print(player1.args.mctsCanonicalStates)
        #print(player2.args.mctsCanonicalStates)

        while not g1.win_state().any():
            p = pOrder[g1.player]

            if p == 1:
                mv = player1.play(g1)
                #print(mv)
                
                player1.update(g1, mv)
                g1.play_action(mv)

                if g2.player == 1 and (args1.mctsCanonicalStates ^ args2.mctsCanonicalStates):
                    a, b = (mv % 7, mv// 7)
                    a, b = g2._board.pairflipInXPlusYCorrection((a,b))
                    mv = b*7 + a
                
                player2.update(g2, mv)
                g2.play_action(mv)

            else:
                mv = player2.play(g2)
                
                player2.update(g2, mv)
                g2.play_action(mv)

                if g1.player == 1 and (args1.mctsCanonicalStates ^ args2.mctsCanonicalStates):
                    a, b = (mv % 7, mv// 7)
                    a, b = g1._board.pairflipInXPlusYCorrection((a,b))
                    mv = (b*7 + a)
                

                player1.update(g1, mv)
                g1.play_action(mv)

            if VERBOSE:
                print(g1._board)

        if g1.win_state()[0] == 1:
            wins[pOrder[0]] +=1
        else:
            wins[pOrder[1]] +=1

        print(wins)
        #"""