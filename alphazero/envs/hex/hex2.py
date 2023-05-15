# cython: language_level=3
# cython: auto_pickle=True
# cython: profile=True


# To compile Cython code
import pyximport; 
pyximport.install()
#-----------------------
#
from typing import List, Tuple, Any

from alphazero.Game import GameState
from alphazero.envs.hex.hexBoard import Board, BLUE_PLAYER, RED_PLAYER, EMPTY

import numpy as np


NUM_PLAYERS             = 2
#MULTI_PLANE_OBSERVATION = False
BOARD_SIZE              = 9
OBSERVATION_BOARD_SIZE  = BOARD_SIZE
MAX_TURNS               = BOARD_SIZE*BOARD_SIZE + 1
CHANNEL_DICT            = {"OnlyBoard" : 1, "BoardAndTurn" :2, "BasicOneHot" : 3, 
        "OneHotTurn" : 4, "OneHotTurnWithNeuroHexLayers" : 8,
        "OneHotTurnWithNeuroHexLayersCut":6,
        "OneHotTurnWithVirtualNeuroHexLayers" : 8}
MODE                    = "OneHotTurn"
NUM_CHANNELS            = CHANNEL_DICT.get(MODE)

# This determines weather we do the flip on the board when it's red's turn to make them
#   appear the same
CANONICAL_STATE = MODE == "OnlyBoard" or MODE == "BasicOneHot"
SWAP_MOVE       = False

class Game(GameState):
    def __init__(self):
        super().__init__(self._get_board())

    @staticmethod
    def _get_board():
        return Board(BOARD_SIZE)

    def __hash__(self) -> int:
        return hash(self._board.board.tobytes() + bytes([self.turns]) + bytes([self._player]))

    def __eq__(self, other: 'Game') -> bool:
        return self._board.board == other._board.board and self._player == other._player and self.turns == other.turns

    def clone(self) -> 'Game':
        game = Game()
        game._board  = self._board.deepcopy()
        game._player = self._player
        game._turns  = self._turns
        game.last_action = self.last_action
        return game

    @staticmethod
    def max_turns() -> int:
        return MAX_TURNS

    @staticmethod
    def has_draw() -> bool:
        return False    

    @staticmethod
    def num_players() -> int:
        return NUM_PLAYERS

    # action == BOARD_SIZE*BOARD_SIZE means swap  
    @staticmethod
    def action_size() -> int:
        return BOARD_SIZE*BOARD_SIZE + 1

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return NUM_CHANNELS, OBSERVATION_BOARD_SIZE, OBSERVATION_BOARD_SIZE

    def valid_moves(self):
        ret = self._board.getPossibleMoves()
        if CANONICAL_STATE and self._player == 1:
            ret = np.flipud(np.fliplr(np.transpose(np.reshape(ret, (BOARD_SIZE, BOARD_SIZE))))).ravel()
        return np.append(ret,(1 if (self._turns == 1 and SWAP_MOVE) else 0))


    def play_action(self, action: int) -> None:
        super().play_action(action)

        me = (BLUE_PLAYER, RED_PLAYER)[self.player]
        realAction = action

        move = (self._board.SWAP, self._board.SWAP) if (action == BOARD_SIZE*BOARD_SIZE) else (action % BOARD_SIZE, action// BOARD_SIZE) 

        if me == RED_PLAYER and CANONICAL_STATE and move != (self._board.SWAP, self._board.SWAP):
            move = self._board.pairflipInXPlusYCorrection(move)

        self._board.play(move, me)
        self._update_turn()

    def win_state(self) -> np.ndarray:
        ret = np.array([False]*(2 + self.has_draw()))
        if (self._board.findCC((-1,0)) == self._board.findCC((BOARD_SIZE,0))):
            # Then Blue Won
            ret[0] = True 

        elif (self._board.findCC((0,-1)) == self._board.findCC((0,BOARD_SIZE))):
            # Then RED Won
            ret[1] = True  
        
        return ret

    def observation(self):
        me = (BLUE_PLAYER, RED_PLAYER)[self.player]
        them = 3 - me

        if MODE == "OnlyBoard":
            # We only get the basic board with default encoding i.e. blue as 1 red as 0

            bOneD = None
            if CANONICAL_STATE and me == RED_PLAYER:
                bOneD = self._board.deepcopy()
                bOneD.flipInXPlusY()
            else:
                bOneD = self._board

            colourFlipper = lambda x : 1*(x==1) + (-1)*(x==2) + x*(x!=1 and x!=2)
            ret = np.vectorize(colourFlipper)(np.reshape(bOneD.board, (BOARD_SIZE+2,BOARD_SIZE+2)))
            if OBSERVATION_BOARD_SIZE == BOARD_SIZE:
                return np.expand_dims(np.delete(np.delete(ret,[0,-1],1),[0,-1],0), axis=0)
            else:
                return np.expand_dims(ret, axis=0)
        
        if MODE == "BoardAndTurn":
            colourFlipper = lambda x : 1*(x==1) + (-1)*(x==2) + x*(x!=1 and x!=2)
            ret = np.vectorize(colourFlipper)(np.reshape(self._board.board, (BOARD_SIZE+2,BOARD_SIZE+2)))
            turn = np.full_like(ret, colourFlipper(me))
            if OBSERVATION_BOARD_SIZE == BOARD_SIZE:
                return np.array([np.delete(np.delete(turn,[0,-1],1),[0,-1],0), np.delete(np.delete(ret,[0,-1],1),[0,-1],0)])
            else:
                return np.array([turn, ret])

        if MODE == "BasicOneHot":
            # Base structure, the default layers are 
            #   0/1/2 Empty/yourPieces/enemyPieces

            bOneD = None
            if CANONICAL_STATE and me == RED_PLAYER:
                bOneD = self._board.deepcopy()
                bOneD.flipInXPlusY(colourCorrect = False)
            else:
                bOneD = self._board
            b = np.reshape(bOneD.board, (BOARD_SIZE+2,BOARD_SIZE+2))

            blank = np.where(b == 0, 1, 0)
            mine  = np.where(b == me, 1, 0)
            yours = np.where(b == them, 1, 0)

            if OBSERVATION_BOARD_SIZE == BOARD_SIZE:
                return np.array([np.delete(np.delete(blank,[0,-1],1),[0,-1],0), 
                                 np.delete(np.delete(mine,[0,-1],1),[0,-1],0),
                                 np.delete(np.delete(yours,[0,-1],1),[0,-1],0)])
            else:
                return np.array([blank,mine,yours])
        
        if MODE == "OneHotTurn":
            # 0/1/2/3 Turn/Empty/BluePieces/RedPieces 
            #  Turn will just be a full layer of who's turn it is (0 for blue 1 for red)
            b      = np.reshape(self._board.board, (BOARD_SIZE+2,BOARD_SIZE+2))
            blank  = np.where(b == 0, 1, 0)
            blues  = np.where(b == BLUE_PLAYER, 1, 0)
            reds   = np.where(b == RED_PLAYER, 1, 0)
            turns  = np.full((BOARD_SIZE+2,BOARD_SIZE+2), self._player)

            if OBSERVATION_BOARD_SIZE == BOARD_SIZE:
                return np.array([np.delete(np.delete(turns,[0,-1],1),[0,-1],0),
                                 np.delete(np.delete(blank,[0,-1],1),[0,-1],0), 
                                 np.delete(np.delete(blues,[0,-1],1),[0,-1],0),
                                 np.delete(np.delete(reds, [0,-1],1),[0,-1],0)])
            else:
                return np.array([turns,blank,blues,reds])

        if MODE == "OneHotTurnWithNeuroHexLayers":
            # 0/1/2/3 Turn/Empty/BluePieces/RedPieces 
            # 4/5 Connected To Blue Left/ Right side
            # 5/6 Connected To Red Top/Bottom
            #  Turn will just be a full layer of who's turn it is (0 for blue 1 for red)
            b      = np.reshape(self._board.board, (BOARD_SIZE+2,BOARD_SIZE+2))
            blank  = np.where(b == 0, 1, 0)
            blues  = np.where(b == BLUE_PLAYER, 1, 0)
            reds   = np.where(b == RED_PLAYER, 1, 0)
            turns  = np.full((BOARD_SIZE+2,BOARD_SIZE+2), self._player)

            c                    = np.reshape(self._board.connectionBoard, (BOARD_SIZE+2,BOARD_SIZE+2))
            
            blueConnectedToLeft  = np.where(c == self._board.cordsToInt((-1,0)), 1, 0)
            blueConnectedToRight = np.where(c == self._board.cordsToInt((BOARD_SIZE, 0)), 1, 0)
            redConnectedToTop    = np.where(c == self._board.cordsToInt((0,-1)), 1, 0)
            redConnectedToBot    = np.where(c == self._board.cordsToInt((0,BOARD_SIZE)), 1, 0)
            
            if OBSERVATION_BOARD_SIZE == BOARD_SIZE:
                return np.array([np.delete(np.delete(turns,[0,-1],1),[0,-1],0),
                    np.delete(np.delete(blank,[0,-1],1),[0,-1],0), 
                    np.delete(np.delete(blues,[0,-1],1),[0,-1],0),
                    np.delete(np.delete(reds, [0,-1],1),[0,-1],0),
                    np.delete(np.delete(blueConnectedToLeft, [0,-1],1),[0,-1],0),
                    np.delete(np.delete(blueConnectedToRight, [0,-1],1),[0,-1],0),
                    np.delete(np.delete(redConnectedToTop, [0,-1],1),[0,-1],0),
                    np.delete(np.delete(redConnectedToBot, [0,-1],1),[0,-1],0)])
            else:
                return np.array([turns,blank,blues,reds, blueConnectedToLeft,
                    blueConnectedToRight, redConnectedToTop, redConnectedToBot])


        if MODE == "OneHotTurnWithNeuroHexLayersCut":
            # 0/1/2/3 Turn/Empty/BluePieces/RedPieces 
            # 4 Connected To Blue Left or Right
            # 5 Connected To Red Top or Bottom
            #  Turn will just be a full layer of who's turn it is (0 for blue 1 for red)
            b      = np.reshape(self._board.board, (BOARD_SIZE+2,BOARD_SIZE+2))
            blank  = np.where(b == 0, 1, 0)
            blues  = np.where(b == BLUE_PLAYER, 1, 0)
            reds   = np.where(b == RED_PLAYER, 1, 0)
            turns  = np.full((BOARD_SIZE+2,BOARD_SIZE+2), self._player)

            c                    = np.reshape(self._board.connectionBoard, (BOARD_SIZE+2,BOARD_SIZE+2))
            
            blueConnectedToLeft  = np.where(c == self._board.cordsToInt((-1,0)), 1, 0)
            blueConnectedToRight = np.where(c == self._board.cordsToInt((BOARD_SIZE, 0)), 1, 0)
            redConnectedToTop    = np.where(c == self._board.cordsToInt((0,-1)), 1, 0)
            redConnectedToBot    = np.where(c == self._board.cordsToInt((0,BOARD_SIZE)), 1, 0)
            
            blueConnected = blueConnectedToLeft + blueConnectedToRight
            redConnected  = redConnectedToTop + redConnectedToBot


            if OBSERVATION_BOARD_SIZE == BOARD_SIZE:
                return np.array([np.delete(np.delete(turns,[0,-1],1),[0,-1],0),
                    np.delete(np.delete(blank,[0,-1],1),[0,-1],0), 
                    np.delete(np.delete(blues,[0,-1],1),[0,-1],0),
                    np.delete(np.delete(reds, [0,-1],1),[0,-1],0),
                    np.delete(np.delete(blueConnected, [0,-1],1),[0,-1],0),
                    np.delete(np.delete(redConnected, [0,-1],1),[0,-1],0)])
            else:
                return np.array([turns,blank,blues,reds, blueConnected, redConnected])


        if MODE == "OneHotTurnWithVirtualNeuroHexLayers":
            # 0/1/2/3 Turn/Empty/BluePieces/RedPieces 
            #  Turn will just be a full layer of who's turn it is (0 for blue 1 for red)
            b      = np.reshape(self._board.board, (BOARD_SIZE+2,BOARD_SIZE+2))
            blank  = np.where(b == 0, 1, 0)
            blues  = np.where(b == BLUE_PLAYER, 1, 0)
            reds   = np.where(b == RED_PLAYER, 1, 0)
            turns  = np.full((BOARD_SIZE+2,BOARD_SIZE+2), self._player)

            c                  = np.reshape(self._board.connectionBoard, (BOARD_SIZE+2,BOARD_SIZE+2))
            # These are sets of the leaders of goups in connection board which are connected
            #  to the appropriate sides of board (with appropriate colour)
            blueConnectedToLeft = {(-1,0)}
            blueConnectedToRight = {(BOARD_SIZE, 0)}
            redConnectedToTop  = {(0,-1)}
            redConnectedToBot  = {(0,BOARD_SIZE)}
            
            sets = [blueConnectedToLeft, blueConnectedToRight]

            for vcs, sets in [(self._board.blueVCs, [blueConnectedToLeft, blueConnectedToRight]),
                              (self._board.redVCs,  [redConnectedToTop, redConnectedToBot])]:
                exploring = 0
                found = True
                while found:
                    found = False
                    for pat in vcs:
                        if pat.requiredMove != None:
                            continue;

                        foundIn = False
                        foundOut = False
                        for cords in pat.toConnect:
                            t = self._board.findCC(cords)
                            foundIn  = foundIn  or t in sets[exploring]
                            foundOut = foundOut or t not in sets[exploring]

                            if foundIn and foundOut:
                                sets[exploring] |= {self._board.findCC(c) for c in pat.toConnect}
                                found = True
                                break;
                    if not found and exploring == 0:
                        found = True
                        exploring = 1

            blueConnectedToLeft  = {self._board.cordsToInt(x) for x in blueConnectedToLeft}
            blueConnectedToRight = {self._board.cordsToInt(x) for x in blueConnectedToRight}

            redConnectedToTop    = {self._board.cordsToInt(x) for x in redConnectedToTop}
            redConnectedToBot    = {self._board.cordsToInt(x) for x in redConnectedToBot}


            blueConnectedToLeftLayer = np.where(np.isin(c, np.array(list(blueConnectedToLeft))), 1, 0)
            blueConnectedToRightLayer = np.where(np.isin(c, np.array(list(blueConnectedToRight))), 1, 0)
            redConnectedToTopLayer = np.where(np.isin(c, np.array(list(redConnectedToTop))), 1, 0)
            redConnectedToBotLayer = np.where(np.isin(c, np.array(list(redConnectedToBot))), 1, 0)

            if OBSERVATION_BOARD_SIZE == BOARD_SIZE:
                return np.array([np.delete(np.delete(turns,[0,-1],1),[0,-1],0),
                    np.delete(np.delete(blank,[0,-1],1),[0,-1],0), 
                    np.delete(np.delete(blues,[0,-1],1),[0,-1],0),
                    np.delete(np.delete(reds, [0,-1],1),[0,-1],0),
                    np.delete(np.delete(blueConnectedToLeftLayer, [0,-1],1),[0,-1],0),
                    np.delete(np.delete(blueConnectedToRightLayer, [0,-1],1),[0,-1],0),
                    np.delete(np.delete(redConnectedToTopLayer, [0,-1],1),[0,-1],0),
                    np.delete(np.delete(redConnectedToBotLayer, [0,-1],1),[0,-1],0)])
            else:
                return np.array([turns,blank,blues,reds, blueConnectedToLeftLayer,
                    blueConnectedToRightLayer, redConnectedToTopLayer, redConnectedToBotLayer])

        raise ValueError('NUM_CHANNELS in hex.pyx is not set to an appropriate value')
            
    def symmetries(self, pi, winstate) -> List[Tuple[Any, int]]:
        # Symmetry 1 : A rotation of pi about x (center of board)
        #   - note board could be even sized
        #      A B C D E 
        #     --------------
        #   1 \\ . . ->. . \\ 1
        #    2 \\ . . . . . \\ 2
        #     3 \\ . . x . . \\ 3
        #      4 \\ . . . . . \\ 4
        #       5 \\ . . <-. . \\ 5
        #           -------------
        #              A B C D E
        # Symmetry 2 : A reflection in x + y = BOARD_SIZE and a colour swap
        #      A B C D E 
        #     --------------
        #   1 \\ . . . . / \\ 1
        #    2 \\ . . . / . \\ 2
        #     3 \\ . . / . . \\ 3
        #      4 \\ . / . . . \\ 4
        #       5 \\ / . . . . \\ 5
        #           -------------
        #              A B C D E  
        ret = []
        pi_board = np.reshape(pi[:-1], (BOARD_SIZE,BOARD_SIZE))
        winningPlayer = 0 if (winstate[0] == 1) else 1
        for flipInXPlusY in [False, True]:
            for rotate in [False, True]:
                # If we are viewing red and blue as the same then don't bother with 
                #  sym 2 as it is already accounted for in the way the board is seen
                if CANONICAL_STATE and flipInXPlusY:
                    continue;
                new_state    = self.clone()
                new_pi       = np.copy(pi_board)
                new_winstate = np.array([0]*(2+self.has_draw()))

                if flipInXPlusY: #Sym 2
                    new_state._board.flipInXPlusY()
                    new_state._player = 1 - self._player
                    new_pi = np.flipud(np.fliplr(np.transpose(new_pi)))

                if rotate: # Sym 1
                    new_state._board.rotate()
                    new_pi = np.fliplr(np.flipud(new_pi))

                if CANONICAL_STATE and new_state._player == winningPlayer:
                    new_winstate[0] = 1
                elif CANONICAL_STATE and new_state._player != winningPlayer:
                    new_winstate[1] = 1
                elif flipInXPlusY:
                    new_winstate[1 - winningPlayer] = 1
                else:
                    new_winstate[winningPlayer] = 1

                ret.append((new_state, np.append(new_pi.ravel(), pi[-1]),new_winstate))
        return ret


def display(board, action=None):
    print(board._board)