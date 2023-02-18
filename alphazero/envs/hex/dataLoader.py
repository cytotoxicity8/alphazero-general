description = """{-
Module      : DataHandler
Description : Manipulating the data from Games and Moves into better form
Maintainer  : River

In games.csv
--------------------------------------------------------------------
"gid" is just unique ID for game,
"boardSize" is size of the board played on,
"moves" is total number of moves played,
"result" is 0 if first player lost 2 if they won,
"rating", "rating" are the ratings of the 2 players in some order

In moves.csv
--------------------------------------------------------------------
"gid" is just unqiue ID for game linking to games.csv,
"nmove" is what number move it is (i.e 1 up to moves),
"jmove" is the move taken
    - resign is let other player win
    - swap is swap position not colour! For example game: https://www.littlegolem.net/jsp/game/game.jsp?gid=771&nmove=2
    - ge is (4,6) i.e reverse(chr(_) for i in "ge") <- OLD CHANGE

-}"""


#-------- Imports ---------------------------------------------------------------
import torch
from alphazero.envs.hex.hex import Game, BOARD_SIZE, CANONICAL_STATE
import numpy as np
from alphazero.utils import get_iter_file
from torch import multiprocessing as mp
from queue import Empty
import os
import pickle

#-------- Classes ---------------------------------------------------------------

#-------- Datatypes -------------------------------------------------------------

#-------- Destructors -----------------------------------------------------------

#-------- Helper Functions ------------------------------------------------------

#-------- Main Functions --------------------------------------------------------

def saveIterationSamples(iteration, output, game_cls):
    num_samples = output.qsize()
    print(f'Saving {num_samples} samples')

    data_tensor = torch.zeros([num_samples, *game_cls.observation_size()])
    policy_tensor = torch.zeros([num_samples, game_cls.action_size()])
    value_tensor = torch.zeros([num_samples, game_cls.num_players() + 1])
    for i in range(num_samples):
        data, policy, value = output.get()
        data_tensor[i] = torch.from_numpy(data)
        policy_tensor[i] = torch.from_numpy(policy)
        value_tensor[i] = torch.from_numpy(value)

    folder = "HumanData/"
    filename = os.path.join(folder, get_iter_file(iteration).replace('.pkl', ''))
    if not os.path.exists(folder): os.makedirs(folder)

    torch.save(data_tensor, filename + '-data.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(policy_tensor, filename + '-policy.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(value_tensor, filename + '-value.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
    del data_tensor
    del policy_tensor
    del value_tensor

def main():
    assert(BOARD_SIZE == 13)
    data = []
    totalMoves = 0
    with open('games.csv') as games:
        with open('moves.csv') as moves:
            # skip past unnecessary headers
            games.readline()
            moves.readline() 

            game = games.readline()
            move = moves.readline().split(',')
            while game:
                gameSplit = game.split(',')
                gID       = int(gameSplit[0])
                boardSize = int(gameSplit[1])
                nMoves    = int(gameSplit[2])
                result    = int(gameSplit[3])
                r1        = float(gameSplit[4])
                r2        = float(gameSplit[5])
                moveList  = []


                while True:
                    #print(move)
                    try:
                        gID2 = int(move[0])
                    except:
                        break;
                    mv   = move[2]
                    if (gID2 != gID):
                        break;
                    
                    if len(mv) == 5:
                        cords = (ord(mv[1]) - ord('a'), ord(mv[2]) - ord('a'))
                        moveList.append(cords)
                    elif (mv[1:-2] == "swap"):
                        cords = (-10,-10)
                        moveList.append(cords)

                    move = moves.readline().split(',')

                if boardSize == 13 and len(moveList) > 2 and (r1 > 1200 or r2 > 1200):
                    totalMoves += len(moveList)
                    data.append((gID, result, r1, r2, moveList))
                game = games.readline()
    #print(totalMoves/len(data))
    #print(data)
    maximum = 2600
    for i in range(1000, 2600, 200):
        databetween = list(filter(lambda x: max(x[2],x[3])>= i and max(x[2],x[3])< i+200, data))
        print("num data points between {} and {} is {}".format(i, i+200, len(databetween)))
    assert False
    #print(max(map(lambda x : x[2], data)))
    #print(max(map(lambda x : x[3], data)))
    print("Loaded")
    game_cls = Game
    n = 0
    outs = [mp.Queue(), mp.Queue(), mp.Queue(), mp.Queue(), mp.Queue(), mp.Queue(), mp.Queue(), mp.Queue()]
    for gID, result, r1, r2, moveList in data:
        n+=1
        #if (n < 6) : continue
        #print(n)
        g = game_cls()
        
        winstate = np.array([result == 2,result == 0,False])
        #print(winstate)
        out = outs[(max(int(r1),int(r2))-1200)//200]
        for move in moveList:
            #print(move)
            #print(g._board)
            if (move == (-10, -10)):
                trueMove = 13*13
            else:
                if CANONICAL_STATE and g._player == 1:
                    move = g._board.pairflipInXPlusYCorrection(move)
                trueMove = move[1] * 13 + move[0]
            assert (trueMove>=0 and trueMove<=13*13)
            # Maybe Add noise?
            pi = np.zeros(13*13+1, dtype=np.float32)
            pi[trueMove] = 1

            #----------------------
            data = g.symmetries(pi, winstate)
            for state, pi, true_winstate in data:
                #print(state.observation(), pi, np.array(true_winstate, dtype=np.float32))
                out.put((
                    state.observation(), pi, np.array(true_winstate, dtype=np.float32)
                ))
            g.play_action(trueMove)
            
        if n%100 == 0:
            print('.', end = '', flush=True)
        if n%5000 == 0:
            print()
        
    for i in range(0,len(outs)):
        saveIterationSamples(i, outs[i], game_cls)
        #print(outs[i])


if __name__ == "__main__":
    print(description)
    main()
