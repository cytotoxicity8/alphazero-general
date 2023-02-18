from alphazero.Game import GameState
from alphazero.GenericPlayers import BasePlayer
from subprocess import Popen, PIPE, DEVNULL

import numpy as np
from alphazero.envs.hex.hex import BOARD_SIZE
from alphazero.envs.hex.hexBoard import BLUE_PLAYER, RED_PLAYER

class HumanHexPlayer(BasePlayer):
    @staticmethod
    def is_human() -> bool:
        return True

    def play(self, state: GameState) -> int:
        print(state._board)

        
        l = input("Where to play? (e.g. A3) : ")
        move = (ord(l[0].upper()) - ord("A"), int(l[1:])-1)
        print(move)
        return move[1] * BOARD_SIZE + move[0]


class GTPPlayer(BasePlayer):
    LOCATION = "~/Hex-Project/benzene-vanilla-cmake/build/src/wolve/wolve"
    commands = ['boardsize {0}'.format(BOARD_SIZE), 
    'param_wolve max_time 0.5']#, 
    #'param_wolve use_time_management 1',
    #'param_wolve use_parallel_solver 1']
    playerCorrispondance = {1 : "black", 0 : "white"}
    def init(self, *args, **kwargs):
        try:
            self.p.terminate()
        except:
            1+1
        self.p = Popen(self.LOCATION, shell=True, stdout=PIPE, stdin=PIPE, stderr=DEVNULL)

        for c in self.commands:
            self.getResponse(c)

    @staticmethod
    def is_human() -> bool:
        return False

    def reset(self) -> None:
        self.playerCorrispondance = {1 : "black", 0 : "white"}
        self.init()


    # Send toSend with newline on end and get back response and second newline returning
    #  response without either newline
    def getResponse(self, toSend : str):
        print("sending {}".format(toSend), end = ' ')
        self.p.stdin.write((toSend+"\n").encode())
        self.p.stdin.flush()
        ret = self.p.stdout.readline()
        self.p.stdout.readline()
        self.p.stdout.flush()
        #self.p.stderr.flush()
        realRet = ret.decode('UTF-8')[2:-1]

        #print("recieving {}".format(realRet))
        return realRet


    def play(self, state:GameState) -> int:
        me   = self.playerCorrispondance.get(state._player)
        them = self.playerCorrispondance.get(1 - state._player)
        print("GTP Player getting move for {}".format("Blue" if me == "white" else "Red"))
        lastMove, p = state._board.lastMove

        lastMoveSTR = chr(lastMove[0] + ord("A")) + str(lastMove[1]+1)

        if lastMove[0] != state._board.ERROR:
            if p == (BLUE_PLAYER, RED_PLAYER)[state._player]:
                print("Detected A swap")
                self.reset()
                self.getResponse("play {0} {1}".format(them, lastMoveSTR))

            else:
                #print("play {0} {1}".format(them, lastMoveSTR))
                self.getResponse("play {0} {1}".format(them, lastMoveSTR))
        resp = self.getResponse("genmove {}".format(me))
        #print(state._board)
    
        #print(resp)
        try:
            move = (ord(resp[0].upper()) - ord("A"), int(resp[1:])-1)
            return move[1] * BOARD_SIZE + move[0]

        except:
            return state.valid_moves()[0]

